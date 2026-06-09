import os
import random
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, AutoProcessor
import io

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava.constants import DEFAULT_IMAGE_TOKEN

IGNORE_LABEL = 255

class FantasticRealityDataset(Dataset):
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
        precision="fp32",
        include_real=False,
        include_fake=True,
    ):
        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # dirs
        dataset_dir = os.path.join(base_dir, "FantasticReality_v1", "dataset")

        self.fake_dir = os.path.join(dataset_dir, "ColorFakeImages")
        self.real_dir = os.path.join(dataset_dir, "ColorRealImages")
        self.mask_dir = os.path.join(dataset_dir, "SegmentationFake")
        self.boundary_dir = os.path.join(dataset_dir, "SegmentationFake_boundary")

        self.answer_list = [
            "It's fake. Segmentation provided: [SEG].",
            "It's manipulated. Segmentation provided: [SEG].",
            "It's forged. Segmentation provided: [SEG].",
            "It's tampered. Segmentation provided: [SEG].",
            "It's not real. Segmentation provided: [SEG]."
        ]

        self.real_answer_list = [
            "This is a real image, no segmentation needed.",
            "The image is authentic, no segmentation required.",
            "It is real."
        ]

        self.question = (
            "Can you identify if this image is real or tampered image?\n"
            "If tampered, please segment the manipulated area."
        )

        self.samples = self._collect_samples(
            include_real=include_real,
            include_fake=include_fake,
        )

        print(
            f"[FantasticReality] total samples: {len(self.samples)} | "
            f"fake: {sum(s['is_tampered'] for s in self.samples)} | "
            f"real: {sum(not s['is_tampered'] for s in self.samples)}"
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under: {base_dir}")

    def __len__(self):
        return len(self.samples)

    def _collect_samples(self, include_real=False, include_fake=True):
        samples = []
        IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

        if include_fake:
            fake_names = sorted(
                f for f in os.listdir(self.fake_dir)
                if (not f.startswith(".")) and f.lower().endswith(IMG_EXT)
            )
            for name in fake_names:
                image_path = os.path.join(self.fake_dir, name)
                if not os.path.isfile(image_path):
                    continue

                stem = os.path.splitext(name)[0]
                mask_path = os.path.join(self.mask_dir, stem + ".npz")
                if not os.path.isfile(mask_path):
                    continue

                samples.append(
                    dict(
                        image_path=image_path,
                        mask_path=mask_path,
                        is_tampered=True,
                    )
                )

        if include_real:
            real_names = sorted(
                f for f in os.listdir(self.real_dir)
                if (not f.startswith(".")) and f.lower().endswith(IMG_EXT)
            )
            for name in real_names:
                image_path = os.path.join(self.real_dir, name)
                if not os.path.isfile(image_path):
                    continue

                samples.append(
                    dict(
                        image_path=image_path,
                        mask_path=None,
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
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_npz_mask(self, path):
        data = np.load(path)
        if "mask" in data:
            return data["mask"]
        return data[list(data.keys())[0]]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        is_tampered = sample["is_tampered"]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        # image
        image = self.load_image(image_path)
        ori_size = image.shape[:2]  # (H, W)

        # clip
        image_clip = self.clip_image_processor.preprocess(
            image,
            return_tensors="pt",
        )["pixel_values"][0]
        # image_clip = self.clip_image_processor.preprocess(
        #     t_multi["clip_rgb"],
        #     return_tensors="pt",
        #     do_center_crop=False,
        # )["pixel_values"][0]

        # sam
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]  # (H', W')
        image_sam = self.preprocess_image(
            torch.from_numpy(image_sam).permute(2, 0, 1).float()
        )

        # conversation
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN + "\n" + self.question
            # DEFAULT_IMAGE_TOKEN + "\n" + "<REF>" + "\n" + self.question
        )

        if is_tampered:
            answer = random.choice(self.answer_list)
        else:
            answer = random.choice(self.real_answer_list)

        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]

        # masks
        if is_tampered and mask_path is not None and os.path.exists(mask_path):
            mask = self._load_npz_mask(mask_path)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.float32)

            masks = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
            exists = [True]
        else:
            masks = torch.zeros((0, ori_size[0], ori_size[1]), dtype=torch.float32)
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
