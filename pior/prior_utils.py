import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 256.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 256.0


@dataclass
class Sample:
    image_path: str
    stem: str
    output_fg_path: str
    output_bg_path: str
    scribble_path: str = ""


class ImageOnlyDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], image_size: int = 512):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"读取图像失败: {sample.image_path}")
        image = image[:, :, ::-1].astype(np.float32)
        height, width = image.shape[:2]
        resized = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        resized = (resized - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(resized).permute(2, 0, 1)
        return {
            "image": tensor,
            "orig_size": (height, width),
            "sample_idx": idx,
        }


def load_backbone(weights_path: str, device: torch.device):
    from model.pvtv2 import pvt_v2_b2

    model = pvt_v2_b2()
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    return model


class PriorFeatureExtractor(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, use_stages: Sequence[int] = (1, 2, 3)):
        super().__init__()
        self.backbone = backbone
        self.use_stages = tuple(use_stages)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)
        selected = [feats[idx] for idx in self.use_stages]
        ref_h, ref_w = selected[0].shape[-2:]
        merged = []
        for feat in selected:
            feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            if feat.shape[-2:] != (ref_h, ref_w):
                feat = F.interpolate(feat, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
            merged.append(feat)
        merged = torch.cat(merged, dim=1)
        merged = F.normalize(merged, dim=1)
        return merged


def read_scribble_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"读取标注失败: {mask_path}")
    return mask


def resize_mask(mask: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def sample_mask_features(
    feature_map: torch.Tensor,
    mask: np.ndarray,
    label_value: int,
    max_samples: int,
    generator: np.random.Generator,
) -> torch.Tensor:
    positions = np.argwhere(mask == label_value)
    if len(positions) == 0:
        return torch.empty((0, feature_map.shape[0]), dtype=feature_map.dtype)
    if max_samples > 0 and len(positions) > max_samples:
        picked = generator.choice(len(positions), size=max_samples, replace=False)
        positions = positions[picked]
    ys = torch.from_numpy(positions[:, 0]).long()
    xs = torch.from_numpy(positions[:, 1]).long()
    sampled = feature_map[:, ys, xs].transpose(0, 1).contiguous().cpu()
    return sampled


def region_semantic_feature(feature_map: torch.Tensor, mask: np.ndarray, label_value: int) -> torch.Tensor:
    positions = np.argwhere(mask == label_value)
    if len(positions) == 0:
        return torch.empty((0, feature_map.shape[0]), dtype=feature_map.dtype)
    ys = torch.from_numpy(positions[:, 0]).long()
    xs = torch.from_numpy(positions[:, 1]).long()
    sampled = feature_map[:, ys, xs].mean(dim=1, keepdim=True).transpose(0, 1).contiguous().cpu()
    sampled = F.normalize(sampled, dim=1)
    return sampled


def tensor_to_uint8_map(score_map: torch.Tensor, height: int, width: int) -> np.ndarray:
    score_map = score_map.reshape(1, 1, *score_map.shape)
    score_map = F.interpolate(score_map, size=(height, width), mode="bilinear", align_corners=False)
    score_map = score_map.squeeze().cpu().numpy()
    score_min = float(score_map.min())
    score_max = float(score_map.max())
    if score_max > score_min:
        score_map = (score_map - score_min) / (score_max - score_min)
    else:
        score_map = np.zeros_like(score_map)
    return np.clip(score_map * 255.0, 0, 255).astype(np.uint8)


def save_prior_map(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def cosine_max_scores(query: torch.Tensor, bank: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
    bank = F.normalize(bank.to(query.device), dim=1)
    out = []
    for start in range(0, query.shape[0], chunk_size):
        q = query[start:start + chunk_size]
        sim = q @ bank.t()
        out.append(sim.max(dim=1).values)
    return torch.cat(out, dim=0)


def min_l2_scores(query: torch.Tensor, bank: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if bank.numel() == 0:
        return torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
    bank = bank.to(query.device)
    out = []
    for start in range(0, query.shape[0], chunk_size):
        q = query[start:start + chunk_size]
        dist = torch.cdist(q, bank)
        out.append(dist.min(dim=1).values)
    return torch.cat(out, dim=0)


def compute_prior_map(
    feature_map: torch.Tensor,
    semantic_bank: torch.Tensor,
    patch_bank: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    channels, height, width = feature_map.shape
    query = feature_map.reshape(channels, -1).transpose(0, 1).contiguous()
    query = F.normalize(query, dim=1)
    semantic_scores = cosine_max_scores(query, semantic_bank, chunk_size)
    suppressed = query * (1.0 - semantic_scores.unsqueeze(1))
    prior_scores = min_l2_scores(suppressed, patch_bank, chunk_size)
    return prior_scores.reshape(height, width)


def purify_authentic_prior(auth_prior: torch.Tensor, manip_prior: torch.Tensor) -> torch.Tensor:
    auth_norm = normalize_map_tensor(auth_prior)
    manip_norm = normalize_map_tensor(manip_prior)
    purified = auth_norm * (1.0 - manip_norm)
    return purified


def purify_authentic_prior_with_similarity(
    auth_prior: torch.Tensor,
    manip_prior: torch.Tensor,
    kernel_size: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    auth_norm = normalize_map_tensor(auth_prior)
    manip_norm = normalize_map_tensor(manip_prior)

    auth_map = auth_norm.unsqueeze(0).unsqueeze(0)
    manip_map = manip_norm.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2

    auth_windows = F.unfold(auth_map, kernel_size=kernel_size, padding=padding).transpose(1, 2).squeeze(0)
    manip_windows = F.unfold(manip_map, kernel_size=kernel_size, padding=padding).transpose(1, 2).squeeze(0)
    local_similarity = F.cosine_similarity(auth_windows, manip_windows, dim=1, eps=eps)
    local_similarity = ((local_similarity + 1.0) * 0.5).clamp(0.0, 1.0).reshape_as(auth_norm)

    purified = torch.clamp(auth_norm - local_similarity * manip_norm, min=0.0)
    return normalize_map_tensor(purified)


def normalize_map_tensor(score_map: torch.Tensor) -> torch.Tensor:
    score_min = score_map.min()
    score_max = score_map.max()
    if float(score_max - score_min) < 1e-8:
        return torch.zeros_like(score_map)
    return (score_map - score_min) / (score_max - score_min)


def sample_feature_bank(
    features: torch.Tensor,
    max_size: int,
    sampler: str,
    device: torch.device,
) -> torch.Tensor:
    if max_size <= 0 or len(features) <= max_size:
        return features
    if sampler == "random":
        indices = torch.randperm(len(features))[:max_size]
        return features[indices]
    if sampler == "approx_greedy":
        return approximate_greedy_coreset(features, max_size=max_size, device=device)
    raise ValueError(f"不支持的采样器: {sampler}")


def approximate_greedy_coreset(
    features: torch.Tensor,
    max_size: int,
    device: torch.device,
    projection_dim: int = 128,
    start_points: int = 10,
) -> torch.Tensor:
    if len(features) <= max_size:
        return features
    feature_device = features.to(device)
    if feature_device.shape[1] > projection_dim:
        mapper = torch.nn.Linear(feature_device.shape[1], projection_dim, bias=False).to(device)
        with torch.no_grad():
            reduced = mapper(feature_device)
    else:
        reduced = feature_device
    reduced = F.normalize(reduced, dim=1)
    start_points = min(start_points, len(reduced))
    start_indices = torch.randperm(len(reduced), device=device)[:start_points]
    anchor_dist = torch.cdist(reduced, reduced[start_indices]).mean(dim=1, keepdim=True)
    selected = []
    with torch.no_grad():
        for _ in range(max_size):
            select_idx = torch.argmax(anchor_dist).item()
            selected.append(select_idx)
            current = torch.cdist(reduced, reduced[select_idx:select_idx + 1])
            anchor_dist = torch.minimum(anchor_dist, current)
    selected = torch.tensor(selected, dtype=torch.long)
    return features[selected]


def build_train_samples(root: str, label_dir: str) -> List[Sample]:
    train_root = os.path.join(root, "train")
    txt_path = os.path.join(root, "train.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    samples = []
    for name in names:
        stem, _ = os.path.splitext(name)
        samples.append(
            Sample(
                image_path=os.path.join(train_root, "Image", name),
                scribble_path=os.path.join(train_root, label_dir, f"{stem}.png"),
                stem=stem,
                output_fg_path=os.path.join(train_root, "trainset_fg", f"{stem}.png"),
                output_bg_path=os.path.join(train_root, "trainset_bg", f"{stem}.png"),
            )
        )
    return samples


def build_test_samples(root: str) -> List[Sample]:
    split_root = os.path.join(root, "test")
    txt_path = os.path.join(root, "test.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    samples = []
    for name in names:
        stem, _ = os.path.splitext(name)
        samples.append(
            Sample(
                image_path=os.path.join(split_root, "Image", name),
                stem=stem,
                output_fg_path=os.path.join(split_root, "fg", f"{stem}.png"),
                output_bg_path=os.path.join(split_root, "bg", f"{stem}.png"),
            )
        )
    return samples
