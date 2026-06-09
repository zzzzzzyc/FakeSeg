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
import io

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from .prior_utils import load_prior_pair

IGNORE_LABEL = 255


class IMDValDataset(Dataset):
    """
    IMD2020 expected structure:
      base_dir/
        IMD2020/
          <subfolder1>/
            c8tf5mq_0.jpg (forged)
            c8tf5mq_0_mask.png
            c8tf5mq_0_boundary.png
            ...
          <subfolder2>/
            ...

    Naming rule in each subfolder:
      Forged:   <id>_<k>.(jpg/png/...)
      Mask:     <id>_<k>_mask.(png/...)
      Boundary: <id>_<k>_boundary.(png/...)
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
            dataset_root="IMD2020",  # <--- NEW: folder under base_dir
            question: str = (
                    "Can you identify if this image is real or tampered image?\n"
                    "If it is tampered, please output the segmentation mask of the manipulated regions."
            ),
            sort_files: bool = True,
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
            "It's not real. Segmentation provided: [SEG]."
        ]

        self.question = question
        self.prior_fg_dir = os.path.join(base_dir, "test", dataset_root, "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", dataset_root, "test", "bg")

        self.sort_files = sort_files

        self.samples = self._collect_samples()
        print(f"[IMD2020Val] Tampered-only samples: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found under: {os.path.join(base_dir, dataset_root)}. "
                f"Check folder names and naming rules."
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

    # --------- filename helpers ----------
    @staticmethod
    def _is_image_file(name: str) -> bool:
        ext = os.path.splitext(name)[1].lower()
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    @staticmethod
    def _split_id_k(fname: str):
        """
        Accept forged filename base like:
          c8tf5mq_0
          abc_12
        Return (id_str, k_str) or (None, None) if not match.
        """
        base = os.path.splitext(os.path.basename(fname))[0]
        m = re.match(r"^(.+?)_(\d+)$", base)
        if not m:
            return None, None
        return m.group(1), m.group(2)

    @staticmethod
    def _same_stem_any_ext(dir_path: str, stem: str):
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            p = os.path.join(dir_path, stem + ext)
            if os.path.isfile(p):
                return p
        # fallback scan
        for n in os.listdir(dir_path):
            if os.path.splitext(n)[0] == stem:
                ext = os.path.splitext(n)[1].lower()
                if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
                    return os.path.join(dir_path, n)
        return None

    def _collect_samples(self):
        root = os.path.join(self.base_dir, self.dataset_root)
        if not os.path.isdir(root):
            return []

        out = []
        # recursive walk
        for dirpath, _, filenames in os.walk(root):
            if not filenames:
                continue

            names = [n for n in filenames if self._is_image_file(n)]
            if self.sort_files:
                names.sort()

            # build a quick set for existence checks (by stem)
            stems = set(os.path.splitext(n)[0] for n in names)

            for n in names:
                stem = os.path.splitext(n)[0]

                # skip mask/boundary files; we only treat plain <id>_<k> as forged candidates
                if stem.endswith("_mask") or stem.endswith("_boundary"):
                    continue

                id_str, k_str = self._split_id_k(n)
                if id_str is None:
                    continue

                mask_stem = f"{id_str}_{k_str}_mask"

                # find mask in same dir
                mask_path = self._same_stem_any_ext(dirpath, mask_stem)
                if mask_path is None:
                    continue  # without mask we can't eval localization
                forged_path = os.path.join(dirpath, n)

                out.append(
                    dict(
                        image_path=forged_path,
                        mask_path=mask_path,
                        subset=os.path.relpath(dirpath, root),  # which subfolder
                        id=f"{id_str}_{k_str}",
                    )
                )
        return out

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]  # (H,W)

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
        conv.append_message(conv.roles[1], self.answer_list[0] if len(self.answer_list) else "")
        conversations = [conv.get_prompt()]

        # masks (binary 0/1)
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
        masks = torch.from_numpy(masks)  # (N,H,W)

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
