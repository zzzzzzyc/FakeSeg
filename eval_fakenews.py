import argparse
import cv2
import numpy as np
import os
import sys
from functools import partial

import torch
import torch.nn.functional as F
import tqdm

from model.FAKESEG import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from dataloaders.casia_1_dataset_seg import CASIA1ValDataset
from dataloaders.cocoglide_dataset import CocoGlideValDataset
from dataloaders.labelwild_dataset import LabelInWildValDataset
from dataloaders.imd_dataset import IMDValDataset
from dataloaders.nist2016_dataset import NISTNC2016ValDataset
from dataloaders.columbia_dataset import ColumbiaValDataset
from dataloaders.coverage_dataset import COVERAGEValDataset
from dataloaders.korus_dataset import KorusValDataset
from dataloaders.trainval_dataset import collate_fn_val
from utils import (
    AverageMeter,
    Summary,
    prepare_input,
)
from transformers import logging

logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser(description="SESAME eval-only script without DeepSpeed")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="fakeseg_vis", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument(
        "--local_prior_vis_image",
        default="/home/hpclp/disk/Graphgpt/dataset/IMD2020/1d7uhv/c9ntew0_0.jpg",
        type=str,
    )
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--save_pred_masks", action="store_true", default=False)
    parser.add_argument("--pred_mask_dir", default="", type=str)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)

def _save_pred_mask(pred_binary, image_path, save_dir):
    if not save_dir:
        return

    os.makedirs(save_dir, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{image_name}.png")
    mask = (pred_binary.detach().float().cpu().numpy() > 0).astype(np.uint8) * 255
    cv2.imwrite(save_path, mask)


def _pixel_f1_iou_from_binary(pred, gt, ignore_index=255, eps=1e-6):
    valid = (gt != ignore_index)

    if valid.sum() == 0:
        one = torch.tensor(1.0, device=gt.device)
        return one, one

    pred = pred.bool()
    gt_bin = (gt == 1)
    pred = pred & valid
    gt_bin = gt_bin & valid

    if gt_bin.sum() == 0 and pred.sum() == 0:
        one = torch.tensor(1.0, device=gt.device)
        return one, one

    tp = (pred & gt_bin).sum().float()
    fp = (pred & (~gt_bin) & valid).sum().float()
    fn = ((~pred) & gt_bin & valid).sum().float()

    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, iou


def _normalize_match_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _get_similarity_map_like_repo(sm, shape):
    sm = (sm - sm.min(1, keepdim=True)[0]) / (
        sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0] + 1e-6
    )
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = sm.to(torch.float32)

    target_size = 336
    h, w = shape
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    sm = torch.nn.functional.interpolate(
        sm, (target_size, target_size), mode="bilinear", align_corners=False
    )
    pad_h = (new_h - target_size) // 2
    pad_w = (new_w - target_size) // 2
    padded_sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
    sm = torch.nn.functional.interpolate(
        padded_sm, shape, mode="bilinear", align_corners=False
    )
    sm = sm.permute(0, 2, 3, 1)
    return sm


def _save_similarity_heatmap(similarity_tokens, image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w = img.shape[:2]
    similarity_map = _get_similarity_map_like_repo(
        similarity_tokens.detach().float(), (h, w)
    )
    sim = similarity_map[0, ..., 0].cpu().numpy()
    vis = np.clip(sim, 0, 1)
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    vis = img * 0.3 + vis * 0.7
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)


@torch.inference_mode()
def validate(val_loader, model, global_iters, args):
    f1_meter = AverageMeter("PixelF1", ":6.4f", Summary.SUM)
    iou_meter = AverageMeter("PixelIoU", ":6.4f", Summary.SUM)

    model.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        input_dict = prepare_input(input_dict, args.precision, is_cuda=True)

        output_dict = model(**input_dict)
        pred_masks = output_dict["pred_masks"]
        gt_masks_batch = output_dict["gt_masks"]
        corrected_sim_tokens = output_dict.get("similarity")
        image_paths = input_dict.get("image_paths", [])

        assert len(pred_masks) == len(gt_masks_batch)

        if (
            args.local_rank == 0
            and args.local_prior_vis_image
            and corrected_sim_tokens is not None
            and len(image_paths) > 0
        ):
            target_path = _normalize_match_path(args.local_prior_vis_image)
            for image_idx, image_path in enumerate(image_paths):
                if _normalize_match_path(image_path) != target_path:
                    continue
                save_dir = os.path.join(args.log_dir, "sim_map_vis")
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(save_dir, f"{image_name}_step{global_iters}.png")
                _save_similarity_heatmap(corrected_sim_tokens[image_idx], image_path, save_path)
                break

        batch_f1 = 0.0
        batch_iou = 0.0
        batch_count = 0

        for sample_idx, (pred_mask_i, gt_mask_i) in enumerate(zip(pred_masks, gt_masks_batch)):
            gt_mask_i = gt_mask_i.int()

            if gt_mask_i.dim() == 3:
                if gt_mask_i.shape[0] == 0:
                    gt_binary = torch.zeros(
                        gt_mask_i.shape[-2:],
                        device=gt_mask_i.device,
                        dtype=torch.int32,
                    )
                else:
                    if (gt_mask_i == 255).any():
                        gt_binary = torch.zeros(
                            gt_mask_i.shape[-2:],
                            device=gt_mask_i.device,
                            dtype=torch.int32,
                        )
                        ignore_mask = (gt_mask_i == 255).all(dim=0)
                        positive_mask = (gt_mask_i == 1).any(dim=0)
                        gt_binary[ignore_mask] = 255
                        gt_binary[positive_mask] = 1
                    else:
                        gt_binary = (gt_mask_i > 0).any(dim=0).int()
            else:
                if (gt_mask_i == 255).any():
                    gt_binary = gt_mask_i.clone()
                else:
                    gt_binary = (gt_mask_i > 0).int()

            if pred_mask_i.dim() == 3:
                if pred_mask_i.shape[0] == 0:
                    pred_binary = torch.zeros(
                        pred_mask_i.shape[-2:],
                        device=pred_mask_i.device,
                        dtype=torch.int32,
                    )
                else:
                    pred_binary = (pred_mask_i > 0).any(dim=0).int()
            else:
                pred_binary = (pred_mask_i > 0).int()

            if args.save_pred_masks and sample_idx < len(image_paths):
                _save_pred_mask(
                    pred_binary,
                    image_paths[sample_idx],
                    args.pred_mask_dir,
                )

            f1, iou = _pixel_f1_iou_from_binary(
                pred_binary,
                gt_binary,
                ignore_index=255,
            )

            batch_f1 += f1
            batch_iou += iou
            batch_count += 1

        if batch_count > 0:
            batch_f1 = batch_f1 / batch_count
            batch_iou = batch_iou / batch_count

            f1_meter.update(batch_f1.item(), n=batch_count)
            iou_meter.update(batch_iou.item(), n=batch_count)

    pixel_f1 = f1_meter.avg
    pixel_iou = iou_meter.avg

    if args.local_rank == 0:
        print("pixel_f1: {:.4f}, pixel_iou: {:.4f}".format(pixel_f1, pixel_iou))

    return pixel_f1, pixel_iou


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.save_pred_masks and not args.pred_mask_dir:
        args.pred_mask_dir = os.path.join(args.log_dir, "pred_masks")

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16

    tokenizer, model, vision_tower, _ = load_pretrained_model_SESAME(
        model_path=args.version,
        device_map=None,
        device="cuda",
        torch_dtype=torch_dtype,
        vision_pretrained=args.vision_pretrained,
        vision_tower=args.vision_tower,
    )

    model = model.cuda()
    model.eval()

    val_dataset = IMDValDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        args.image_size,
    )

    assert args.val_batch_size == 1
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=partial(
            collate_fn_val,
            tokenizer=tokenizer,
            use_mm_start_end=args.use_mm_start_end,
        ),
    )

    validate(val_loader, model, 0, args)


if __name__ == "__main__":
    main(sys.argv[1:])
