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
import io

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide
from .prior_utils import load_prior_pair
IGNORE_LABEL = 255

class COVERAGEValDataset(Dataset):
    """
    Expected structure:

      base_dir/
        COVERAGE/
          image/
            1.tif
            1t.tif
            2.tif
            2t.tif
            ...
          mask/
            1copy.tif
            1paste.tif
            1forged.tif
            2copy.tif
            2paste.tif
            2forged.tif
            ...
          label/
            *.mat

    We use:
      tempered image: image/*t.tif
      forged mask:    mask/*forged.tif
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
        dataset_root: str = "COVERAGE",
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "If it is tampered, please output the segmentation mask of the manipulated regions."
        ),
        sort_files: bool = True,
        tampered_only: bool = True,
        mask_threshold: int = 0,
    ):
        self.base_dir = base_dir
        self.dataset_root = dataset_root

        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.prior_fg_dir = os.path.join(base_dir, "test", "Coverage", "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", "Coverage", "test", "bg")

        self.answer_list = [
            "It's fake. Segmentation provided: [SEG].",
            "It's manipulated. Segmentation provided: [SEG].",
            "It's forged. Segmentation provided: [SEG].",
            "It's tampered. Segmentation provided: [SEG].",
            "It's not real. Segmentation provided: [SEG]."
        ]

        self.question = question
        self.sort_files = sort_files
        self.tampered_only = tampered_only
        self.mask_threshold = mask_threshold

        self.samples = self._collect_samples()
        print(f"[COVERAGE] Valid samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid COVERAGE samples found under: {self._get_root()}"
            )

    def __len__(self):
        return len(self.samples)

    def _get_root(self):
        return os.path.join(self.base_dir, self.dataset_root)

    # ---------------- IO helpers ----------------
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
    def _is_image_file(name: str):
        return os.path.splitext(name)[1].lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]

    def _collect_samples(self):
        root = self._get_root()
        image_dir = os.path.join(root, "image")
        mask_dir = os.path.join(root, "mask")

        out = []

        if not os.path.isdir(image_dir):
            return out
        if not os.path.isdir(mask_dir):
            return out

        names = os.listdir(image_dir)
        if self.sort_files:
            names.sort()

        for name in names:
            if not self._is_image_file(name):
                continue

            stem, ext = os.path.splitext(name)
            low_stem = stem.lower()

            # tempered images end with 't', e.g. 1t.tif
            if self.tampered_only:
                if not low_stem.endswith("t"):
                    continue

                base_id = stem[:-1]  # "1t" -> "1"
                if base_id == "":
                    continue

                image_path = os.path.join(image_dir, name)
                mask_name = f"{base_id}forged.tif"
                mask_path = os.path.join(mask_dir, mask_name)

                if not os.path.isfile(image_path):
                    continue
                if not os.path.isfile(mask_path):
                    # fallback: case-insensitive search
                    found = None
                    for m in os.listdir(mask_dir):
                        if os.path.splitext(m)[0].lower() == f"{base_id}forged".lower():
                            found = os.path.join(mask_dir, m)
                            break
                    if found is None:
                        continue
                    mask_path = found
                prior_fg_path = os.path.join(self.prior_fg_dir, f"{stem}.png")
                prior_bg_path = os.path.join(self.prior_bg_dir, f"{stem}.png")
                print(
                    f"[COVERAGE][prior-match] image={name} "
                    f"fg={'Y' if os.path.isfile(prior_fg_path) else 'N'} "
                    f"bg={'Y' if os.path.isfile(prior_bg_path) else 'N'}"
                )

                out.append(
                    dict(
                        image_path=image_path,
                        mask_path=mask_path,
                        id=base_id,
                    )
                )
            else:
                # optional: keep both pristine and tampered
                image_path = os.path.join(image_dir, name)

                if low_stem.endswith("t"):
                    base_id = stem[:-1]
                    mask_name = f"{base_id}forged.tif"
                    mask_path = os.path.join(mask_dir, mask_name)
                    if not os.path.isfile(mask_path):
                        found = None
                        for m in os.listdir(mask_dir):
                            if os.path.splitext(m)[0].lower() == f"{base_id}forged".lower():
                                found = os.path.join(mask_dir, m)
                                break
                        mask_path = found
                else:
                    base_id = stem
                    mask_path = None

                out.append(
                    dict(
                        image_path=image_path,
                        mask_path=mask_path,
                        id=base_id,
                    )
                )

        if self.sort_files:
            out.sort(key=lambda x: x["image_path"])

        return out

    # ---------------- mask helpers ----------------
    def _load_mask(self, mask_path: str, ori_size_hw):
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        H, W = ori_size_hw
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # COVERAGE forged masks are typically binary-like:
        # non-zero = forged region
        bin_mask = (mask > self.mask_threshold).astype(np.float32)
        return bin_mask

    # ---------------- main ----------------
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]

        # CLIP input
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        # SAM input
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess_image(
            torch.from_numpy(image_sam).permute(2, 0, 1).float()
        )

        # conversation
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + self.question)
        conv.append_message(conv.roles[1], self.answer_list[0] if len(self.answer_list) else "")
        conversations = [conv.get_prompt()]

        # masks
        sampled_masks = []
        if mask_path is not None and os.path.exists(mask_path):
            mask = self._load_mask(mask_path, ori_size)
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

