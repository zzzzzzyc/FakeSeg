import os
import random
import re
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, AutoProcessor
import albumentations as A

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)
IGNORE_LABEL = 255

# REAL_ANSWER_LIST = [
#     "This image is real and does not contain any tampered regions.",
#     "The image is authentic. No tampered areas are present.",
#     "This is a genuine image with no signs of manipulation.",
#     "No tampering is detected in this image. There is no region to mask.",
#     "The image is real. No segmentation mask is required."
# ]

class AutoSpliceDataset(Dataset):
    """
    Expected folder structure under base_dir:
      - Forged_JPEG75/
      - Forged_JPEG90/
      - Forged_JPEG100/
      - Mask/
      - Boundary/   (you generated; filenames identical to Mask filenames)

    Mapping rule:
      Forged:  <id>_<k>.(jpg/png/...)
      Mask:    <id>_mask.(png/...)
      Boundary:<id>_mask.(png/...)   # same basename as mask
    """

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
        sam_img_size: int = 512,
        forged_folders=("Forged_JPEG75", "Forged_JPEG90", "Forged_JPEG100"),
        mask_folder="Mask",
        boundary_folder="boundary",
        include_if_missing_boundary: bool = True,
        answer_list=ANSWER_LIST,
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "Please output segmentation mask."
        ),
        # if you want deterministic ordering
        sort_files: bool = True,
    ):
        # self.base_dir = base_dir
        autospl_root = os.path.join(base_dir, "AutoSplice")
        self.base_dir = autospl_root
        self.tokenizer = tokenizer
        self.precision = precision
        self.sam_img_size = sam_img_size

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.answer_list = answer_list
        self.question = question

        self.forged_folders = list(forged_folders)
        self.mask_folder = mask_folder
        self.boundary_folder = boundary_folder
        self.include_if_missing_boundary = include_if_missing_boundary
        self.sort_files = sort_files

        self.samples = self._collect_samples()
        print(f"[AutoSplice] samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found under: {base_dir}. "
                f"Check folder names and naming rules."
            )

        self.aug = A.Compose(
            [
                A.RandomScale(
                    scale_limit=(-0.5, 0.5),
                    interpolation=1,
                    p=0.5,
                ),
                A.JpegCompression(
                    quality_lower=30,
                    quality_upper=100,
                    p=0.5,
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    # --------- IO helpers ----------
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

    def _create_t_RGB(self, img_RGB: np.ndarray) -> torch.Tensor:
        if self.aug is not None:
            dat = self.aug(image=img_RGB)
            img_RGB = dat["image"]
            if img_RGB.dtype != np.uint8:
                img_RGB = img_RGB.astype(np.uint8)

        h, w = img_RGB.shape[:2]
        grid = 8
        pad_h = ((h + grid - 1) // grid) * grid
        pad_w = ((w + grid - 1) // grid) * grid
        if pad_h != h or pad_w != w:
            temp = np.full((pad_h, pad_w, 3), 127.5, dtype=img_RGB.dtype)
            temp[:h, :w] = img_RGB
            img_RGB = temp

        t_RGB = torch.from_numpy(img_RGB).permute(2, 0, 1).float() / 256.0

        _, H, W = t_RGB.shape
        scale = self.sam_img_size / max(H, W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        t_RGB = torch.nn.functional.interpolate(
            t_RGB.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_bottom = self.sam_img_size - new_h
        pad_right = self.sam_img_size - new_w
        t_RGB_sam = torch.nn.functional.pad(
            t_RGB,
            (0, pad_right, 0, pad_bottom),
            value=127.5 / 256.0,
        )
        return t_RGB_sam

    # --------- filename mapping ----------
    @staticmethod
    def _is_image_file(name: str) -> bool:
        ext = os.path.splitext(name)[1].lower()
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    @staticmethod
    def _extract_id_from_forged_filename(fname: str):
        """
        Expected forged filename like: 39406_0.jpg -> id=39406
        Returns None if doesn't match.
        """
        base = os.path.splitext(os.path.basename(fname))[0]
        m = re.match(r"^(.+?)_\d+$", base)
        if not m:
            return None
        return m.group(1)

    @staticmethod
    def _mask_basename_from_id(id_str: str) -> str:
        return f"{id_str}_mask"

    def _find_mask_path(self, id_str: str):
        mask_dir = os.path.join(self.base_dir, self.mask_folder)
        if not os.path.isdir(mask_dir):
            return None

        # find any extension that matches <id>_mask.*
        stem = self._mask_basename_from_id(id_str)
        # common fast-path
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            p = os.path.join(mask_dir, stem + ext)
            if os.path.isfile(p):
                return p

        # fallback scan
        for n in os.listdir(mask_dir):
            if os.path.splitext(n)[0] == stem and self._is_image_file(n):
                return os.path.join(mask_dir, n)

        return None

    def _find_boundary_path(self, mask_path: str):
        if mask_path is None:
            return None
        bd_dir = os.path.join(self.base_dir, self.boundary_folder)
        if not os.path.isdir(bd_dir):
            return None
        mask_base = os.path.basename(mask_path)  # boundary filename same as mask filename
        p = os.path.join(bd_dir, mask_base)
        return p if os.path.isfile(p) else None

    def _collect_samples(self):
        out = []
        all_folders = self.forged_folders + ["Authentic"]

        for folder in all_folders:
            img_dir = os.path.join(self.base_dir, folder)
            if not os.path.isdir(img_dir):
                continue

            names = os.listdir(img_dir)
            if self.sort_files: names.sort()

            for n in names:
                if not self._is_image_file(n): continue
                img_path = os.path.join(img_dir, n)

                mask_path = None
                boundary_path = None
                is_tampered = False

                # 只有在非 Authentic 文件夹下才去寻找 Mask
                if folder != "Authentic":
                    id_str = self._extract_id_from_forged_filename(n)
                    if id_str:
                        mask_path = self._find_mask_path(id_str)
                        if mask_path:
                            is_tampered = True
                            boundary_path = self._find_boundary_path(mask_path)

                out.append(
                    dict(
                        image_path=img_path,
                        mask_path=mask_path,
                        boundary_path=boundary_path,
                        subset=folder,
                        is_tampered=is_tampered
                    )
                )
        return out

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        boundary_path = sample["boundary_path"]
        is_tampered = sample["is_tampered"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]  # (H,W)

        # noiseprint input (t_RGB)
        t_RGB = self._create_t_RGB(image)

        # CLIP input
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        # SAM input
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]  # resized H,W
        image_sam = self.preprocess_image(
            torch.from_numpy(image_sam).permute(2, 0, 1).float()
        )

        # conversation (same format as your train dataset)
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + self.question)
        conv.append_message(conv.roles[1], random.choice(self.answer_list))
        conversations = [conv.get_prompt()]

        # masks (binary 0/1)
        if is_tampered and mask_path is not None and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            if mask.ndim == 3: mask = mask[..., 0]
            mask = (mask > 0).astype(np.float32)
            masks = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        else:
            masks = torch.zeros((1, ori_size[0], ori_size[1]), dtype=torch.float32)

        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # boundary (binary 0/1)
        sampled_bds = []
        if boundary_path is not None and os.path.exists(boundary_path):
            bd = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
            if bd is not None:
                bd = (bd > 0).astype(np.float32)
                sampled_bds.append(bd)

        boundary = (
            np.stack(sampled_bds, axis=0)
            if len(sampled_bds) > 0
            else np.zeros((0, *ori_size), np.float32)
        )
        boundary = torch.from_numpy(boundary)
        inference = False

        return (
            image_path,
            t_RGB,
            image_sam,
            image_clip,
            conversations,
            masks,
            boundary,
            label,
            resize,
            inference,
        )

