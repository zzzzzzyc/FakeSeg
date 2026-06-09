import os
import random
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

class CompRAISEDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = IGNORE_LABEL

    def __init__(
        self,
        base_dir,          # e.g. /path/to/compRAISE
        tokenizer,
        vision_tower,
        image_size=224,
        precision: str = "fp32",
    ):
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # compRAISE/compRAISE/*.jpg
        compraise_root = os.path.join(base_dir, "compRAISE", "compRAISE")
        self.image_dir = compraise_root
        self.image_names = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(".jpg")
        ])

        if len(self.image_names) == 0:
            raise RuntimeError(f"No jpg images found under {self.image_dir}")

        self.samples = self._collect_samples()
        print(f"[CompRAISE] real samples: {len(self.samples)}")
        self.answer_list = ANSWER_LIST

        self.question = (
            "Can you identify if this image is real or tampered image?\n"
            "Please output segmentation mask."
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

    def _collect_samples(self):
        samples = []

        for name in self.image_names:
            image_path = os.path.join(self.image_dir, name)
            if not os.path.isfile(image_path):
                continue

            samples.append(
                dict(
                    image_path=image_path,
                    mask_path=None,
                    boundary_path=None,
                    is_tampered=False,
                )
            )

        return samples

    def preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _create_t_RGB(self, img_RGB, sam_img_size=512):
        if self.aug is not None:
            dat = self.aug(image=img_RGB)
            img_RGB = dat["image"]
            if img_RGB.dtype != np.uint8:
                img_RGB = img_RGB.astype(np.uint8)
            del dat

        h, w = img_RGB.shape[:2]

        grid = 8
        pad_h = ((h + grid - 1) // grid) * grid
        pad_w = ((w + grid - 1) // grid) * grid

        if pad_h != h or pad_w != w:
            temp = np.full((pad_h, pad_w, 3), 127.5, dtype=img_RGB.dtype)
            temp[:h, :w] = img_RGB
            img_RGB = temp

        t_RGB = torch.from_numpy(img_RGB).permute(2, 0, 1).float() / 256.0

        C, H, W = t_RGB.shape
        scale = sam_img_size / max(H, W)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        t_RGB = torch.nn.functional.interpolate(
            t_RGB.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        pad_bottom = sam_img_size - new_h
        pad_right = sam_img_size - new_w

        t_RGB_sam = torch.nn.functional.pad(
            t_RGB,
            (0, pad_right, 0, pad_bottom),
            value=127.5 / 256.0,
        )

        return t_RGB_sam

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image_path"]

        # load image
        image = self.load_image(image_path)
        ori_size = image.shape[:2]

        # noiseprint
        t_RGB = self._create_t_RGB(image)

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

        # # real → no masks / boundary
        # masks = torch.zeros((0, *ori_size), dtype=torch.float32)
        # boundary = torch.zeros((0, *ori_size), dtype=torch.float32)
        # label = torch.ones(ori_size[0], ori_size[1]) * self.ignore_label
        masks = torch.zeros((1, ori_size[0], ori_size[1]), dtype=torch.float32)
        boundary = torch.zeros((1, ori_size[0], ori_size[1]), dtype=torch.float32)
        label = torch.ones(ori_size[0], ori_size[1]) * self.ignore_label

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
            False,
        )

