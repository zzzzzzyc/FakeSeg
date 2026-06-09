import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from prior_utils import (
    cosine_max_scores,
    normalize_map_tensor,
    purify_authentic_prior_with_similarity,
    region_semantic_feature,
    save_prior_map,
    tensor_to_uint8_map,
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class CASIA2Sample:
    image_path: str
    subset: str
    stem: str
    rel_stem: str
    mask_path: Optional[str] = None

    @property
    def rel_key(self) -> str:
        return os.path.join(self.subset, self.rel_stem)


class CASIA2ImageDataset(Dataset):
    def __init__(self, samples: List[CASIA2Sample], processor):
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        width, height = image.size
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "height": height,
            "width": width,
            "sample_idx": idx,
        }


class CLIPIntermediateFeatureExtractor(torch.nn.Module):
    def __init__(self, clip_tower, intermediate_layer: int):
        super().__init__()
        self.clip_tower = clip_tower
        self.intermediate_layer = intermediate_layer

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, intermediate = self.clip_tower.forward_with_intermediate(images, self.intermediate_layer)
        feature_tokens = intermediate.float()
        batch_size, num_tokens, channels = feature_tokens.shape
        spatial_size = int(num_tokens ** 0.5)
        if spatial_size * spatial_size != num_tokens:
            raise ValueError(f"patch token 数量不是平方数: {num_tokens}")
        feature_map = feature_tokens.transpose(1, 2).reshape(batch_size, channels, spatial_size, spatial_size)
        feature_map = F.normalize(feature_map, dim=1)
        return feature_map


def parse_args():
    parser = argparse.ArgumentParser(description="用 CLIP 单层中间特征结合 PatchCore 组件为 CASIA2 生成 prior maps。")
    parser.add_argument("--root", required=True, help="CASIA2 根目录，包含 Au/Tp/CASIA_2_Groundtruth。")
    parser.add_argument("--output-root", default="", help="prior 输出目录，默认写到 <root>/priors_clip_patchcore_layer{layer}。")
    parser.add_argument("--clip-encoder-path", default="", help="llava/model/multimodal_encoder/clip_encoder.py 路径。")
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", help="CLIP vision tower 名称或本地路径。")
    parser.add_argument("--intermediate-layer", type=int, default=12, help="用于 MD 的单层中间 hidden state 下标。")
    parser.add_argument("--select-layer", type=int, default=-2, help="CLIPVisionTower 初始化参数，保持默认即可。")
    parser.add_argument("--select-feature", choices=["patch", "cls_patch"], default="patch", help="从 CLIP hidden states 取哪种 token。")
    parser.add_argument("--device", default="cuda", help="运行设备，例: cuda 或 cpu。")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch size。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument("--sample-per-image", type=int, default=256, help="每张图每类最多抽多少 patch 进 bank。")
    parser.add_argument("--max-patch-bank", type=int, default=50000, help="每类 patch bank 最大保留数量。")
    parser.add_argument("--max-sem-bank", type=int, default=4096, help="每类 semantic bank 最大保留数量。")
    parser.add_argument("--sampler", choices=["random", "approx_greedy"], default="approx_greedy", help="使用 PatchCore 的 bank 下采样方式。")
    parser.add_argument("--chunk-size", type=int, default=2048, help="semantic 抑制时 query chunk 大小。")
    parser.add_argument("--cache", default="", help="bank 缓存路径，默认写到 output-root/banks.pt。")
    parser.add_argument("--rebuild-cache", action="store_true", help="忽略已有 bank 缓存，强制重建。")
    parser.add_argument("--include-tp-bg", action="store_true", help="把 Tp 中 mask=0 的背景区域也加入 authentic bank。")
    parser.add_argument("--patchcore-src", default="", help="patchcore-inspection-main/src 路径。")
    parser.add_argument("--patch-size", type=int, default=3, help="直接复用 PatchCore 的 patchsize。")
    parser.add_argument("--patch-stride", type=int, default=1, help="直接复用 PatchCore 的 patch stride。")
    parser.add_argument("--num-nn", type=int, default=1, help="PatchCore 最近邻个数，论文 MD 默认建议保持 1。")
    parser.add_argument("--nn-workers", type=int, default=4, help="FAISS worker 数。")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def default_clip_encoder_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "llava", "model", "multimodal_encoder", "clip_encoder.py")
    )


def default_patchcore_src() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "patchcore-inspection-main", "src")
    )


def dynamic_import(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_clip_tower(args, device: torch.device):
    module_path = args.clip_encoder_path or default_clip_encoder_path()
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"找不到 clip_encoder.py: {module_path}")
    clip_module = dynamic_import(module_path, "llava_clip_encoder_local")
    tower_args = SimpleNamespace(
        mm_vision_select_layer=args.select_layer,
        mm_vision_select_feature=args.select_feature,
    )
    clip_tower = clip_module.CLIPVisionTower(args.vision_tower, args=tower_args, delay_load=False)
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    clip_tower.to(device=device, dtype=target_dtype)
    clip_tower.eval()
    return clip_tower


def load_patchcore_modules(args):
    patchcore_src = args.patchcore_src or default_patchcore_src()
    if not os.path.isdir(patchcore_src):
        raise FileNotFoundError(f"找不到 patchcore src 目录: {patchcore_src}")
    if patchcore_src not in sys.path:
        sys.path.insert(0, patchcore_src)
    import patchcore.common as patchcore_common
    import patchcore.patchcore as patchcore_patchcore
    import patchcore.sampler as patchcore_sampler

    return patchcore_common, patchcore_patchcore, patchcore_sampler


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    paths = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def build_mask_index(mask_dir: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in list_images(mask_dir):
        name = os.path.basename(path)
        stem, _ = os.path.splitext(name)
        if not stem.endswith("_gt"):
            continue
        image_stem = stem[:-3]
        if image_stem in index and index[image_stem] != path:
            raise RuntimeError(f"mask 文件名冲突，请先消歧: {image_stem}")
        index[image_stem] = path
    return index


def build_casia2_samples(root: str) -> List[CASIA2Sample]:
    au_dir = os.path.join(root, "Au")
    tp_dir = os.path.join(root, "Tp")
    gt_dir = os.path.join(root, "CASIA_2_Groundtruth")
    mask_index = build_mask_index(gt_dir)

    samples: List[CASIA2Sample] = []
    for image_path in list_images(au_dir):
        rel_path = os.path.relpath(image_path, au_dir)
        rel_stem, _ = os.path.splitext(rel_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        samples.append(CASIA2Sample(image_path=image_path, subset="Au", stem=stem, rel_stem=rel_stem))
    for image_path in list_images(tp_dir):
        rel_path = os.path.relpath(image_path, tp_dir)
        rel_stem, _ = os.path.splitext(rel_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        if stem not in mask_index:
            raise FileNotFoundError(f"没找到 Tp 对应 mask: {stem}_gt.*")
        samples.append(
            CASIA2Sample(
                image_path=image_path,
                subset="Tp",
                stem=stem,
                rel_stem=rel_stem,
                mask_path=mask_index[stem],
            )
        )
    if not samples:
        raise RuntimeError(f"在 {root} 下没有找到有效图像。")
    return samples


def load_binary_mask(mask_path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"读取 mask 失败: {mask_path}")
    mask = cv2.resize(mask, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def fuse_patch_neighbors(
    patches: torch.Tensor,
    patch_shape: Tuple[int, int],
    neighbor_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if neighbor_weights is None:
        neighbor_weights = torch.tensor(
            [[1.0, 2.0, 1.0],
             [2.0, 4.0, 2.0],
             [1.0, 2.0, 1.0]],
            dtype=patches.dtype,
            device=patches.device,
        )
    neighbor_weights = neighbor_weights / neighbor_weights.sum()

    patch_grid = patches.reshape(patch_shape[0], patch_shape[1], *patches.shape[1:])
    padded = F.pad(patch_grid, (0, 0, 0, 0, 0, 0, 1, 1, 1, 1))
    fused = torch.zeros_like(patch_grid)

    for dy in range(3):
        for dx in range(3):
            fused = fused + neighbor_weights[dy, dx] * padded[
                dy:dy + patch_shape[0],
                dx:dx + patch_shape[1],
            ]
    return fused.reshape(-1, *patches.shape[1:])


def patchify_feature_map(feature_map: torch.Tensor, patch_maker) -> Tuple[torch.Tensor, Tuple[int, int]]:
    patches, patch_shape = patch_maker.patchify(feature_map.unsqueeze(0), return_spatial_info=True)
    patches = patches.squeeze(0)
    patches = fuse_patch_neighbors(patches, (patch_shape[0], patch_shape[1]))
    return patches.cpu(), (patch_shape[0], patch_shape[1])


def sample_mask_patches(
    patches: torch.Tensor,
    patch_shape: Tuple[int, int],
    mask: np.ndarray,
    label_value: int,
    max_samples: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    flat_mask = mask.reshape(-1)
    indices = np.flatnonzero(flat_mask == label_value)
    if len(indices) == 0:
        return torch.empty((0,) + tuple(patches.shape[1:]), dtype=patches.dtype)
    if max_samples > 0 and len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    selected = torch.from_numpy(np.asarray(indices)).long()
    return patches[selected]


def sample_bank_with_patchcore(flat_bank: torch.Tensor, max_size: int, sampler_name: str, patchcore_sampler, device: torch.device) -> np.ndarray:
    flat_bank = flat_bank.float().cpu().numpy().astype(np.float32)
    if max_size <= 0 or len(flat_bank) <= max_size:
        return flat_bank

    percentage = max_size / float(len(flat_bank))
    if sampler_name == "random":
        sampler = patchcore_sampler.RandomSampler(percentage)
    elif sampler_name == "approx_greedy":
        sampler = patchcore_sampler.ApproximateGreedyCoresetSampler(percentage, device=device)
    else:
        raise ValueError(f"不支持的 PatchCore 采样器: {sampler_name}")

    sampled = sampler.run(flat_bank)
    return np.asarray(sampled, dtype=np.float32)


def build_scorer(bank_flat: np.ndarray, patchcore_common, args):
    nn_method = patchcore_common.FaissNN(False, args.nn_workers)
    scorer = patchcore_common.NearestNeighbourScorer(
        n_nearest_neighbours=args.num_nn,
        nn_method=nn_method,
    )
    scorer.fit([bank_flat])
    return scorer


def compute_patchcore_prior_map(
    feature_map: torch.Tensor,
    semantic_bank: torch.Tensor,
    scorer,
    patch_maker,
    chunk_size: int,
) -> torch.Tensor:
    patches, patch_shape = patchify_feature_map(feature_map, patch_maker)
    channels = feature_map.shape[0]
    query_semantic = feature_map.reshape(channels, -1).transpose(0, 1).contiguous()
    query_semantic = F.normalize(query_semantic, dim=1)
    semantic_scores = cosine_max_scores(query_semantic, semantic_bank.float(), chunk_size)
    suppressed_patches = patches * (1.0 - semantic_scores.unsqueeze(1).unsqueeze(2).unsqueeze(3))
    patch_scores, _, _ = scorer.predict([suppressed_patches.numpy().astype(np.float32)])
    return torch.from_numpy(np.asarray(patch_scores, dtype=np.float32)).reshape(*patch_shape)


@torch.no_grad()
def build_banks(samples, dataset, feature_extractor, patch_maker, patchcore_sampler, args, device: torch.device) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    auth_patch_bank: List[torch.Tensor] = []
    manip_patch_bank: List[torch.Tensor] = []
    auth_sem_bank: List[torch.Tensor] = []
    manip_sem_bank: List[torch.Tensor] = []
    rng = np.random.default_rng(3407)

    for batch in tqdm(loader, desc="构建 CASIA2 memory bank"):
        pixel_values = batch["pixel_values"].to(device=device, dtype=feature_extractor.clip_tower.dtype)
        feature_maps = feature_extractor(pixel_values).cpu()
        sample_indices = batch["sample_idx"].tolist()

        for feat_map, sample_idx in zip(feature_maps, sample_indices):
            sample = samples[sample_idx]
            feat_height, feat_width = feat_map.shape[-2:]
            patches, patch_shape = patchify_feature_map(feat_map, patch_maker)

            if sample.subset == "Au":
                full_mask = np.ones((patch_shape[0], patch_shape[1]), dtype=np.uint8)
                auth_patch = sample_mask_patches(
                    patches,
                    patch_shape,
                    full_mask,
                    label_value=1,
                    max_samples=args.sample_per_image,
                    rng=rng,
                )
                auth_sem = region_semantic_feature(feat_map, np.ones((feat_height, feat_width), dtype=np.uint8), label_value=1)
                if len(auth_patch) > 0:
                    auth_patch_bank.append(auth_patch)
                if len(auth_sem) > 0:
                    auth_sem_bank.append(auth_sem)
                continue

            mask = load_binary_mask(sample.mask_path, (feat_height, feat_width))
            manip_patch = sample_mask_patches(
                patches,
                patch_shape,
                mask,
                label_value=1,
                max_samples=args.sample_per_image,
                rng=rng,
            )
            manip_sem = region_semantic_feature(feat_map, mask, label_value=1)
            if len(manip_patch) > 0:
                manip_patch_bank.append(manip_patch)
            if len(manip_sem) > 0:
                manip_sem_bank.append(manip_sem)

            if args.include_tp_bg:
                auth_patch = sample_mask_patches(
                    patches,
                    patch_shape,
                    mask,
                    label_value=0,
                    max_samples=args.sample_per_image,
                    rng=rng,
                )
                auth_sem = region_semantic_feature(feat_map, mask, label_value=0)
                if len(auth_patch) > 0:
                    auth_patch_bank.append(auth_patch)
                if len(auth_sem) > 0:
                    auth_sem_bank.append(auth_sem)

    if not auth_patch_bank:
        raise RuntimeError("authentic patch bank 为空，请检查 Au 目录。")
    if not manip_patch_bank:
        raise RuntimeError("manipulated patch bank 为空，请检查 Tp 和 mask。")

    auth_patch_flat = torch.cat(auth_patch_bank, dim=0).reshape(-1, auth_patch_bank[0][0].numel())
    manip_patch_flat = torch.cat(manip_patch_bank, dim=0).reshape(-1, manip_patch_bank[0][0].numel())
    auth_sem_flat = torch.cat(auth_sem_bank, dim=0).reshape(len(auth_sem_bank), -1)
    manip_sem_flat = torch.cat(manip_sem_bank, dim=0).reshape(len(manip_sem_bank), -1)

    banks = {
        "auth_patch_bank": sample_bank_with_patchcore(auth_patch_flat, args.max_patch_bank, args.sampler, patchcore_sampler, device),
        "manip_patch_bank": sample_bank_with_patchcore(manip_patch_flat, args.max_patch_bank, args.sampler, patchcore_sampler, device),
        "auth_sem_bank": sample_bank_with_patchcore(auth_sem_flat, args.max_sem_bank, args.sampler, patchcore_sampler, device),
        "manip_sem_bank": sample_bank_with_patchcore(manip_sem_flat, args.max_sem_bank, args.sampler, patchcore_sampler, device),
    }
    return banks


def load_or_build_banks(samples, dataset, feature_extractor, patch_maker, patchcore_sampler, args, device: torch.device) -> Dict[str, np.ndarray]:
    cache_path = args.cache or os.path.join(args.output_root, "banks.pt")
    if os.path.exists(cache_path) and not args.rebuild_cache:
        return torch.load(cache_path, map_location="cpu")
    banks = build_banks(samples, dataset, feature_extractor, patch_maker, patchcore_sampler, args, device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(banks, cache_path)
    return banks


@torch.no_grad()
def export_priors(samples, dataset, feature_extractor, patch_maker, banks, patchcore_common, args, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    auth_sem_bank = torch.from_numpy(np.asarray(banks["auth_sem_bank"], dtype=np.float32))
    manip_sem_bank = torch.from_numpy(np.asarray(banks["manip_sem_bank"], dtype=np.float32))
    auth_scorer = build_scorer(np.asarray(banks["auth_patch_bank"], dtype=np.float32), patchcore_common, args)
    manip_scorer = build_scorer(np.asarray(banks["manip_patch_bank"], dtype=np.float32), patchcore_common, args)

    for batch in tqdm(loader, desc="导出 prior maps"):
        pixel_values = batch["pixel_values"].to(device=device, dtype=feature_extractor.clip_tower.dtype)
        feature_maps = feature_extractor(pixel_values).cpu()
        sample_indices = batch["sample_idx"].tolist()
        heights = batch["height"].tolist()
        widths = batch["width"].tolist()

        for feat_map, sample_idx, height, width in zip(feature_maps, sample_indices, heights, widths):
            sample = samples[sample_idx]
            if sample.subset != "Tp":
                continue

            manip_prior = compute_patchcore_prior_map(
                feat_map,
                semantic_bank=auth_sem_bank,
                scorer=auth_scorer,
                patch_maker=patch_maker,
                chunk_size=args.chunk_size,
            )
            auth_prior_raw = compute_patchcore_prior_map(
                feat_map,
                semantic_bank=manip_sem_bank,
                scorer=manip_scorer,
                patch_maker=patch_maker,
                chunk_size=args.chunk_size,
            )

            manip_prior = normalize_map_tensor(manip_prior)
            auth_prior = purify_authentic_prior_with_similarity(auth_prior_raw, manip_prior)

            fg_image = tensor_to_uint8_map(manip_prior, height=height, width=width)
            bg_image = tensor_to_uint8_map(auth_prior, height=height, width=width)

            fg_path = os.path.join(args.output_root, "fg", f"{sample.rel_key}.png")
            bg_path = os.path.join(args.output_root, "bg", f"{sample.rel_key}.png")
            save_prior_map(fg_path, fg_image)
            save_prior_map(bg_path, bg_image)


def main():
    args = parse_args()
    if not args.output_root:
        args.output_root = os.path.join(args.root, f"priors_clip_patchcore_layer{args.intermediate_layer}")
    device = resolve_device(args.device)

    patchcore_common, patchcore_patchcore, patchcore_sampler = load_patchcore_modules(args)
    samples = build_casia2_samples(args.root)
    num_au = sum(sample.subset == "Au" for sample in samples)
    num_tp = sum(sample.subset == "Tp" for sample in samples)
    print(f"Au: {num_au}, Tp: {num_tp}, Total: {len(samples)}")

    clip_tower = load_clip_tower(args, device)
    feature_extractor = CLIPIntermediateFeatureExtractor(clip_tower, args.intermediate_layer).to(device)
    feature_extractor.eval()
    patch_maker = patchcore_patchcore.PatchMaker(args.patch_size, stride=args.patch_stride)

    dataset = CASIA2ImageDataset(samples, clip_tower.image_processor)
    banks = load_or_build_banks(samples, dataset, feature_extractor, patch_maker, patchcore_sampler, args, device)
    export_priors(samples, dataset, feature_extractor, patch_maker, banks, patchcore_common, args, device)


if __name__ == "__main__":
    main()
