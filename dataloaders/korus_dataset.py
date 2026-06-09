import os
import random

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide
from .prior_utils import load_prior_pair

IGNORE_LABEL = 255


class KorusValDataset(Dataset):
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
        dataset_root: str = "Korus",
        tampered_dirname: str = "tampered-realistic",
        gt_dirname: str = "ground-truth",
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "If tampered, please segment the manipulated area."
        ),
        sort_files: bool = True,
    ):
        self.base_dir = base_dir
        self.dataset_root = dataset_root
        self.tampered_dirname = tampered_dirname
        self.gt_dirname = gt_dirname

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
        self.prior_fg_dir = os.path.join(base_dir, "test", "Korus", "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", "Korus", "test", "bg")

        self.samples = self._collect_samples()
        print(f"[Korus] Tampered-only samples: {len(self.samples)}")
        if len(self.samples) == 0:
            root = os.path.join(base_dir, dataset_root)
            raise RuntimeError(f"No valid Korus samples found under: {root}")

    def __len__(self):
        return len(self.samples)

    def _get_root(self):
        root = os.path.join(self.base_dir, self.dataset_root)
        if os.path.isdir(os.path.join(root, "data-images")):
            return os.path.join(root, "data-images")
        return root

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

    @staticmethod
    def _is_image_file(name: str) -> bool:
        ext = os.path.splitext(name)[1].lower()
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    def _same_stem_any_ext(self, dir_path: str, stem: str):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            path = os.path.join(dir_path, stem + ext)
            if os.path.isfile(path):
                return path
        for name in os.listdir(dir_path):
            if os.path.splitext(name)[0].lower() == stem.lower() and self._is_image_file(name):
                return os.path.join(dir_path, name)
        return None

    def _find_mask_path(self, gt_dir: str, stem: str):
        return self._same_stem_any_ext(gt_dir, stem)

    def _collect_samples(self):
        root = self._get_root()
        if not os.path.isdir(root):
            return []

        out = []
        camera_dirs = [
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
            and name not in ["camera_models", "thumbnails"]
        ]
        if self.sort_files:
            camera_dirs.sort()

        for camera_dir in camera_dirs:
            tampered_dir = os.path.join(camera_dir, self.tampered_dirname)
            gt_dir = os.path.join(camera_dir, self.gt_dirname)
            if not os.path.isdir(tampered_dir) or not os.path.isdir(gt_dir):
                continue

            names = [name for name in os.listdir(tampered_dir) if self._is_image_file(name)]
            if self.sort_files:
                names.sort()

            for name in names:
                image_path = os.path.join(tampered_dir, name)
                stem = os.path.splitext(name)[0]
                mask_path = self._find_mask_path(gt_dir, stem)
                if mask_path is None:
                    continue

                out.append(
                    dict(
                        image_path=image_path,
                        mask_path=mask_path,
                        camera=os.path.basename(camera_dir),
                        id=stem,
                    )
                )
        return out

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

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

            # Korus GT:
            #   black   -> authentic
            #   white   -> tampered
            #   gray    -> collateral damage
            mask = (mask >= 224).astype(np.float32)
            sampled_masks.append(mask)

        masks = (
            np.stack(sampled_masks, axis=0)
            if len(sampled_masks) > 0
            else np.zeros((0, *ori_size), np.float32)
        )
        masks = torch.from_numpy(masks)

        exists = [masks.shape[0] > 0]
        sam_mask_shape = [resize, masks.shape[-2:]]
        prior_fg, prior_bg = load_prior_pair(
            image_path=image_path,
            image_hw=ori_size,
            fg_dir=self.prior_fg_dir,
            bg_dir=self.prior_bg_dir,
        )

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
