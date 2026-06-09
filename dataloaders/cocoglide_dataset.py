import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide

IGNORE_LABEL = 255


class CocoGlideValDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = IGNORE_LABEL

    def __init__(
        self,
        base_dir: str,
        tokenizer,
        vision_tower: str,
        image_size: int = 224,
        precision: str = "fp32",
        dataset_root: str = "CocoGlide",
        table_filename: str = "table.csv",
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "If tampered, please segment the manipulated area."
        ),
        sort_files: bool = True,
    ):
        self.base_dir = base_dir
        self.dataset_root = dataset_root
        self.table_filename = table_filename

        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.answer_list = [
            "It's fake. Segmentation provided: [SEG].",
            "It's manipulated. Segmentation provided: [SEG].",
            "It's forged. Segmentation provided: [SEG].",
            "It's tampered. Segmentation provided: [SEG].",
            "It's not real. Segmentation provided: [SEG].",
        ]

        self.question = question
        self.sort_files = sort_files
        self.prior_fg_dir = os.path.join(base_dir, "test", "CocoGlide", "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", "CocoGlide", "test", "bg")

        self.samples = self._collect_samples()
        print(f"[CocoGlide] Tampered-only samples: {len(self.samples)}")
        if len(self.samples) == 0:
            root = os.path.join(base_dir, dataset_root)
            raise RuntimeError(f"No valid CocoGlide samples found under: {root}")

    def __len__(self):
        return len(self.samples)

    def _get_root(self):
        return os.path.join(self.base_dir, self.dataset_root)

    def load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def _collect_samples(self):
        root = self._get_root()
        table_path = os.path.join(root, self.table_filename)
        if not os.path.isfile(table_path):
            return []

        out = []
        with open(table_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                fake_rel = row.get("fake", "").strip()
                mask_rel = row.get("mask", "").strip()
                real_rel = row.get("real", "").strip()
                prompt = row.get("prompt", "").strip()

                if fake_rel == "" or mask_rel == "":
                    continue

                image_path = os.path.join(root, fake_rel)
                mask_path = os.path.join(root, mask_rel)
                real_path = os.path.join(root, real_rel) if real_rel else None

                if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
                    continue

                out.append(
                    dict(
                        image_path=image_path,
                        mask_path=mask_path,
                        real_path=real_path,
                        prompt=prompt,
                        prior_index=idx,
                    )
                )

        if self.sort_files:
            out.sort(key=lambda x: x["prior_index"])
        return out

    def _load_prior_by_index(self, prior_index: int, image_hw):
        h, w = image_hw
        prior_fg = self._load_single_prior(
            os.path.join(self.prior_fg_dir, f"{prior_index}.png"), (h, w)
        )
        prior_bg = self._load_single_prior(
            os.path.join(self.prior_bg_dir, f"{prior_index}.png"), (h, w)
        )
        return prior_fg, prior_bg

    @staticmethod
    def _load_single_prior(prior_path: str, image_hw):
        h, w = image_hw
        if not os.path.isfile(prior_path):
            print(f"[WARN] prior file not found: {prior_path}, using zeros.", flush=True)
            return torch.zeros((1, h, w), dtype=torch.float32)

        prior = cv2.imread(prior_path, cv2.IMREAD_GRAYSCALE)
        if prior is None:
            print(f"[WARN] failed to read prior file: {prior_path}, using zeros.", flush=True)
            return torch.zeros((1, h, w), dtype=torch.float32)

        if prior.shape != (h, w):
            prior = cv2.resize(prior, (w, h), interpolation=cv2.INTER_NEAREST)

        return torch.from_numpy(prior).float().unsqueeze(0) / 255.0

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        prior_index = sample["prior_index"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]

        image_clip = self.clip_image_processor.preprocess(
            image,
            return_tensors="pt",
        )["pixel_values"][0]

        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess_image(
            torch.from_numpy(image_sam).permute(2, 0, 1).float()
        )

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN + "\n" + self.question,
        )
        conv.append_message(conv.roles[1], random.choice(self.answer_list))
        conversations = [conv.get_prompt()]

        sampled_masks = []
        if mask_path is not None and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

            if mask.shape[:2] != ori_size:
                mask = cv2.resize(mask, (ori_size[1], ori_size[0]), interpolation=cv2.INTER_NEAREST)

            mask = (mask > 0).astype(np.float32)
            sampled_masks.append(mask)

        masks = (
            np.stack(sampled_masks, axis=0)
            if len(sampled_masks) > 0
            else np.zeros((0, *ori_size), np.float32)
        )
        masks = torch.from_numpy(masks)

        exists = [masks.shape[0] > 0]
        sam_mask_shape = [resize, masks.shape[-2:]]
        prior_fg, prior_bg = self._load_prior_by_index(prior_index, ori_size)

        return (
            image_path,
            image_sam,
            image_clip,
            conversations,
            masks,
            sam_mask_shape,
            exists,
            prior_fg,
            prior_bg,
        )
