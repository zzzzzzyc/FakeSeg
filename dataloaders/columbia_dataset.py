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
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide
from .prior_utils import load_prior_pair
IGNORE_LABEL = 255

class ColumbiaValDataset(Dataset):

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
        dataset_root: str = "Columbia",
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "If it is tampered, please output the segmentation mask of the manipulated regions."
        ),
        sort_files: bool = True,
        tampered_only: bool = True,
    ):
        self.base_dir = base_dir
        self.dataset_root = dataset_root

        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.prior_fg_dir = os.path.join(base_dir, "test", "Columbia", "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", "Columbia", "test", "bg")

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

        self.samples = self._collect_samples()

        print(f"[Columbia] samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError("No Columbia samples found!")

    def __len__(self):
        return len(self.samples)

    def _get_root(self):
        return os.path.join(self.base_dir, self.dataset_root)

    # ---------- IO ----------
    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess_image(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    # ---------- collect ----------
    def _collect_samples(self):
        root = self._get_root()

        splc_dir = os.path.join(root, "4cam_splc")
        auth_dir = os.path.join(root, "4cam_auth")

        out = []

        def find_mask(mask_dir, stem):
            if not os.path.isdir(mask_dir):
                return None

            target = f"{stem}_edgemask"
            for n in os.listdir(mask_dir):
                low = n.lower()
                if not low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    continue
                if "_edgemask_3" in low:
                    continue
                if os.path.splitext(low)[0] == target.lower():
                    return os.path.join(mask_dir, n)

            return None

        # tampered
        if os.path.isdir(splc_dir):
            mask_dir = os.path.join(splc_dir, "edgemask")
            for name in os.listdir(splc_dir):
                if not name.lower().endswith(".tif"):
                    continue

                img_path = os.path.join(splc_dir, name)
                if not os.path.isfile(img_path):
                    continue

                stem = os.path.splitext(name)[0]
                mask_path = find_mask(mask_dir, stem)
                if mask_path is None:
                    continue
                prior_fg_path = os.path.join(self.prior_fg_dir, f"{stem}.png")
                prior_bg_path = os.path.join(self.prior_bg_dir, f"{stem}.png")
                print(
                    f"[Columbia][prior-match] image={name} "
                    f"fg={'Y' if os.path.isfile(prior_fg_path) else 'N'} "
                    f"bg={'Y' if os.path.isfile(prior_bg_path) else 'N'}"
                )

                out.append(
                    dict(
                        image_path=img_path,
                        mask_path=mask_path,
                        id=stem,
                    )
                )

        if not self.tampered_only and os.path.isdir(auth_dir):
            for name in os.listdir(auth_dir):
                if not name.lower().endswith(".tif"):
                    continue

                img_path = os.path.join(auth_dir, name)

                out.append(
                    dict(
                        image_path=img_path,
                        mask_path=None,
                        id=name,
                    )
                )

        if self.sort_files:
            out.sort(key=lambda x: x["image_path"])

        return out

    # ---------- mask ----------
    def _load_mask(self, mask_path, ori_size):
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

        r = mask[:, :, 0]
        g = mask[:, :, 1]

        # green = forged
        bin_mask = (g > r).astype(np.float32)

        H, W = ori_size
        if bin_mask.shape != (H, W):
            bin_mask = cv2.resize(bin_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return bin_mask

    # ---------- main ----------
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]

        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]

        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess_image(
            torch.from_numpy(image_sam).permute(2, 0, 1).float()
        )

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + self.question)
        conv.append_message(conv.roles[1], self.answer_list[0])
        conversations = [conv.get_prompt()]

        # mask
        if mask_path is not None:
            mask = self._load_mask(mask_path, ori_size)
            masks = torch.from_numpy(mask[None, ...]).float()
        else:
            masks = torch.zeros((0, *ori_size), dtype=torch.float32)

        exists = [masks.shape[0] > 0]
        sam_mask_shape = [
            resize,
            masks.shape[-2:],
        ]
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

