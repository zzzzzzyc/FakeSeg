import os
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


class LabelInWildValDataset(Dataset):
    """
    Expected structure:

      base_dir/
        label_in_wild/
          images/
            *.jpg
          masks/
            *.png

    Pairing rule:
      images/<stem>.jpg  <->  masks/<stem>.png
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
        dataset_root: str = "label_in_wild",
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "If it is tampered, please output the segmentation mask of the manipulated regions."
        ),
        sort_files: bool = True,
        invert_mask: bool = False,
        mask_threshold: int = 0,
    ):
        self.base_dir = base_dir
        self.dataset_root = dataset_root

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
        self.invert_mask = invert_mask
        self.mask_threshold = mask_threshold
        self.prior_fg_dir = os.path.join(base_dir, "test", "ITW", "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", "ITW", "test", "bg")

        self.samples = self._collect_samples()
        print(f"[LabelInWild] Valid samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid label_in_wild samples found under: {self._get_root()}"
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

    # ---------------- collect helpers ----------------
    @staticmethod
    def _is_image_file(name: str):
        return os.path.splitext(name)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    def _collect_samples(self):
        root = self._get_root()
        image_dir = os.path.join(root, "images")
        mask_dir = os.path.join(root, "masks")

        out = []

        if not os.path.isdir(image_dir):
            return out
        if not os.path.isdir(mask_dir):
            return out

        names = [n for n in os.listdir(image_dir) if self._is_image_file(n)]
        if self.sort_files:
            names.sort()

        for name in names:
            stem = os.path.splitext(name)[0]
            image_path = os.path.join(image_dir, name)
            mask_path = os.path.join(mask_dir, stem + ".png")

            if not os.path.isfile(image_path):
                continue
            if not os.path.isfile(mask_path):
                continue

            out.append(
                dict(
                    image_path=image_path,
                    mask_path=mask_path,
                    id=stem,
                )
            )

        return out

    # ---------------- mask helpers ----------------
    def _load_mask(self, mask_path: str, ori_size_hw):
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        H, W = ori_size_hw
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.invert_mask:
            bin_mask = (mask <= self.mask_threshold).astype(np.float32)
        else:
            bin_mask = (mask > self.mask_threshold).astype(np.float32)

        return bin_mask

    # ---------------- main ----------------
    def __getitem__(self, idx: int):
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
        conv.append_message(conv.roles[1], self.answer_list[0] if len(self.answer_list) else "")
        conversations = [conv.get_prompt()]

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

