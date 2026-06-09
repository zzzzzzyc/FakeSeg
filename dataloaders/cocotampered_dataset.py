import os
import random
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import albumentations as A

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava.constants import DEFAULT_IMAGE_TOKEN

IGNORE_LABEL = 255
from PIL import Image


class COCOListTamperedDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = IGNORE_LABEL

    def __init__(
        self,
        base_dir,
        tokenizer,
        vision_tower,
        image_size=224,
        precision: str = "fp32",
        list_files=None,
    ):
        coco_root = os.path.join(base_dir, "tampCOCO", "tampCOCO")
        self.base_dir = coco_root
        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.answer_list = [
            "It's fake. Segmentation provided: [SEG].",
            "It's manipulated. Segmentation provided: [SEG].",
            "It's forged. Segmentation provided: [SEG].",
            "It's tampered. Segmentation provided: [SEG].",
            "It's not real. Segmentation provided: [SEG]."
        ]

        self.question = (
            "Can you identify if this image is real or tampered image?\n"
            "If tampered, please segment the manipulated area."
        )

        if list_files is None:
            list_files = [
                "bcmc_COCO_list.txt",
                "bcm_COCO_list.txt",
                "cm_COCO_list.txt",
                "sp_COCO_list.txt",
            ]
        self.list_files = list_files

        # image_dir -> (mask_dir, boundary_dir)
        self.dir_map = {
            "bcmc_images": ("bcm_masks", "bcm_boundary"),
            "bcm_images": ("bcm_masks", "bcm_boundary"),
            "cm_images": ("cm_masks", "cm_boundary"),
            "sp_images": ("sp_masks", "sp_boundary"),
        }

        self.samples = self._collect_samples()
        print(f"[COCOList] Tampered-only samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under: {base_dir}")

    def __len__(self):
        return len(self.samples)

    def _infer_dirs_from_img_rel(self, img_rel: str):
        top = img_rel.split("/", 1)[0]
        return self.dir_map.get(top, (None, None))

    def _collect_samples(self):
        out = []
        for lf in self.list_files:
            list_path = os.path.join(self.base_dir, lf)
            if not os.path.isfile(list_path):
                continue

            with open(list_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "," not in line:
                        continue

                    img_rel, mask_rel = line.split(",", 1)
                    img_rel = img_rel.strip()
                    mask_rel = mask_rel.strip()

                    img_path = os.path.join(self.base_dir, img_rel)
                    if not os.path.isfile(img_path):
                        continue

                    is_tampered = False
                    mask_path = None

                    target_mask_path = os.path.join(self.base_dir, mask_rel)
                    if os.path.isfile(target_mask_path):
                        mask_path = target_mask_path
                        is_tampered = True
                    else:
                        mask_dir, _ = self._infer_dirs_from_img_rel(img_rel)
                        if mask_dir is not None:
                            mask_base = os.path.basename(mask_rel)
                            mask_path2 = os.path.join(self.base_dir, mask_dir, mask_base)
                            if os.path.isfile(mask_path2):
                                mask_path = mask_path2
                                is_tampered = True

                    out.append(
                        dict(
                            image_path=img_path,
                            mask_path=mask_path,
                            is_tampered=is_tampered,
                        )
                    )
        return out

    def preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        is_tampered = sample["is_tampered"]
        mask_path = sample["mask_path"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]

        # CLIP
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        # SAM
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess_image(
            torch.from_numpy(image_sam).permute(2, 0, 1).float()
        )

        # conversation
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + self.question)
        conv.append_message(conv.roles[1], random.choice(self.answer_list))
        conversations = [conv.get_prompt()]

        # masks
        if is_tampered and mask_path is not None:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            if mask.ndim == 3: mask = mask[..., 0]
            mask = (mask > 0).astype(np.float32)
            masks = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
            exists = [True]
        else:
            masks = torch.zeros((1, ori_size[0], ori_size[1]), dtype=torch.float32)
            exists = [False]

        sam_mask_shape = [resize, masks.shape[-2:]]

        return (
            image_path,
            image_sam,
            image_clip,
            conversations,
            masks,
            sam_mask_shape,
            exists,
        )

