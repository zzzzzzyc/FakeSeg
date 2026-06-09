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
from .prior_utils import load_prior_pair

IGNORE_LABEL = 255


class CASIA1ValDataset(Dataset):
    """
    Expected structure under base_dir:

      base_dir/
        CASIA1/
          Tp/
            CM/
              <forged>.(jpg/png/...)
              ...
            Sp/
              <forged>.(jpg/png/...)
              ...
          CASIA_1_groundtruth/
            CM/
              <forged_stem>_gt.(png/jpg/...)
            Sp/
              <forged_stem>_gt.(png/jpg/...)
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
        image_size: int = 224,          # for SAM ResizeLongestSide
        precision: str = "fp32",
        dataset_root: str = "CASIA1",
        tp_dirname: str = "Tp",
        gt_dirname: str = "CASIA_1_groundtruth",
        subsets=("CM", "Sp"),
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "If tampered, please segment the manipulated area."
        ),
        sort_files: bool = True,
        osn_root: str = "ImageForgeriesOSN_Dataset",
        osn_platform: str = None,
    ):
        self.base_dir = base_dir
        self.dataset_root = dataset_root
        self.tp_dirname = tp_dirname
        self.gt_dirname = gt_dirname
        self.subsets = list(subsets)
        self.osn_dir = (
            os.path.join(base_dir, osn_root, f"CASIA_{osn_platform}")
            if osn_platform
            else None
        )

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

        self.real_answer_list = [
            "This is a real image, no segmentation needed.",
            "The image is authentic, no segmentation required.",
            "It is real."
        ]

        self.question = question
        self.sort_files = sort_files
        self.prior_fg_dir = os.path.join(base_dir, "test", "CASIAV1", "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", "CASIAV1", "test", "bg")

        self.samples = self._collect_samples()
        print(f"[CASIA1Val] Tampered-only samples: {len(self.samples)}")
        if len(self.samples) == 0:
            root = os.path.join(base_dir, dataset_root)
            raise RuntimeError(f"No valid samples found under: {root}. Check folder names and naming rules.")

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

    @staticmethod
    def _is_image_file(name: str) -> bool:
        ext = os.path.splitext(name)[1].lower()
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

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

    def _resolve_osn_image(self, original_path: str):
        if not self.osn_dir:
            return original_path

        name = os.path.basename(original_path)
        stem = os.path.splitext(name)[0]
        image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        image_path = os.path.join(self.osn_dir, name)
        if os.path.isfile(image_path):
            return image_path

        if os.path.isdir(self.osn_dir):
            for ext in image_exts:
                candidate = os.path.join(self.osn_dir, stem + ext)
                if os.path.isfile(candidate):
                    return candidate
            for root, _, files in os.walk(self.osn_dir):
                for file_name in files:
                    file_stem, file_ext = os.path.splitext(file_name)
                    if file_name == name or (
                        file_stem == stem and file_ext.lower() in image_exts
                    ):
                        return os.path.join(root, file_name)

        return None

    def _collect_samples(self):
        root = os.path.join(self.base_dir, self.dataset_root)
        tp_root = os.path.join(root, self.tp_dirname)
        gt_root = os.path.join(root, self.gt_dirname)

        if not os.path.isdir(tp_root):
            return []

        out = []
        for subset in self.subsets:
            tp_dir = os.path.join(tp_root, subset)
            gt_dir = os.path.join(gt_root, subset)
            if not os.path.isdir(tp_dir) or not os.path.isdir(gt_dir):
                continue

            names = [n for n in os.listdir(tp_dir) if self._is_image_file(n)]
            if self.sort_files:
                names.sort()

            for n in names:
                forged_path = os.path.join(tp_dir, n)
                stem = os.path.splitext(n)[0]

                # mask: <forged_stem>_gt.(png/jpg/...)
                mask_stem = stem + "_gt"
                mask_path = self._same_stem_any_ext(gt_dir, mask_stem)
                if mask_path is None:
                    continue

                image_path = self._resolve_osn_image(forged_path)
                if image_path is None:
                    continue

                out.append(
                    dict(
                        image_path=image_path,
                        mask_path=mask_path,
                        subset=subset,
                        id=stem,
                    )
                )
        return out

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]  # (H, W)

        # CLIP input
        image_clip = self.clip_image_processor.preprocess(
            image,
            return_tensors="pt",
        )["pixel_values"][0]

        # SAM input
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]  # resized (H, W)
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
        answer = random.choice(self.answer_list)
        conv.append_message(conv.roles[1], answer)
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
        masks = torch.from_numpy(masks)  # (N, H, W)

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
