import os
import random
import re
import csv
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

class NISTNC2016ValDataset(Dataset):
    """
    Expected structure under base_dir:

      base_dir/
        NC2016/
          probe/
            *.jpg
          world/
            *.jpg
          indexes/
            Manipulation-index.csv
            Removal-index.csv
            Splice-index.csv
          reference/
            manipulation/
              <metadata file>
              mask/
                *.png
            removal/
              <metadata file>
              mask/
                *.png
            splice/
              <metadata file>
              mask/
                *.png

    Notes:
    - We use the reference CSV files as the GT source.
    - For validation/inference, we usually load probe image + probe mask.
    - Donor mask is ignored here, because your current pipeline expects one main mask.
    - `invert_mask` is provided because some NIST masks may encode manipulated region as 0.
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
        dataset_root: str = "nist16",
        data_subdir: str = "NC2016_Test0613",
        tasks=("manipulation",),
        question: str = (
            "Can you identify if this image is real or tampered image?\n"
            "If tampered, please segment the manipulated area."
        ),
        sort_files: bool = True,
        target_only: bool = True,
        invert_mask: bool = True,
        mask_threshold: int = 10,
    ):
        self.base_dir = base_dir
        self.dataset_root = dataset_root
        self.data_subdir = data_subdir
        self.tasks = [t.lower() for t in tasks]

        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.question = question
        self.sort_files = sort_files
        self.prior_fg_dir = os.path.join(base_dir, "test", "NC16", "test", "fg")
        self.prior_bg_dir = os.path.join(base_dir, "test", "NC16", "test", "bg")
        self.target_only = target_only
        self.invert_mask = invert_mask
        self.mask_threshold = mask_threshold

        self.answer_list = [
            "It's fake. Segmentation provided: [SEG].",
            "It's manipulated. Segmentation provided: [SEG].",
            "It's forged. Segmentation provided: [SEG].",
            "It's tampered. Segmentation provided: [SEG].",
            "It's not real. Segmentation provided: [SEG]."
        ]

        self.samples = self._collect_samples()
        print(f"[NIST-NC2016] Valid samples: {len(self.samples)}")
        if len(self.samples) == 0:
            root = self._get_data_root()
            raise RuntimeError(
                f"No valid samples found under: {root}. "
                f"Check folder names / reference csv / mask paths."
            )

    def __len__(self):
        return len(self.samples)

    def _get_data_root(self):
        return os.path.join(self.base_dir, self.dataset_root, self.data_subdir)

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

    # ---------------- CSV / path helpers ----------------
    @staticmethod
    def _norm_rel_path(p: str):
        if p is None:
            return None
        p = p.strip()
        if p == "":
            return None
        p = p.replace("\\", "/")
        if p.startswith("/"):
            p = p[1:]
        return p

    @staticmethod
    def _to_bool(v):
        if v is None:
            return False
        v = str(v).strip().lower()
        return v in ("true", "1", "yes", "y", "t")

    def _find_reference_csv(self, task_dir: str):
        if not os.path.isdir(task_dir):
            return None

        candidates = []
        for n in os.listdir(task_dir):
            p = os.path.join(task_dir, n)
            if os.path.isfile(p) and n.lower().endswith(".csv"):
                candidates.append(p)

        if len(candidates) == 0:
            return None

        candidates.sort()
        return candidates[0]

    def _read_pipe_csv(self, csv_path: str):
        rows = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                clean = {}
                for k, v in row.items():
                    kk = k.strip() if isinstance(k, str) else k
                    vv = v.strip() if isinstance(v, str) else v
                    clean[kk] = vv
                rows.append(clean)
        return rows

    def _resolve_rel_path_case_insensitive(self, root, rel_path):
        if rel_path is None:
            return None

        rel_path = rel_path.strip().replace("\\", "/")
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]

        cur = root
        parts = rel_path.split("/")

        for part in parts:
            if not os.path.isdir(cur):
                candidate = os.path.join(cur, part)
                return candidate if os.path.exists(candidate) else None

            entries = os.listdir(cur)
            match = None
            for e in entries:
                if e.lower() == part.lower():
                    match = e
                    break

            if match is None:
                return None

            cur = os.path.join(cur, match)

        return cur if os.path.exists(cur) else None

    def _collect_samples(self):
        root = self._get_data_root()
        ref_root = os.path.join(root, "reference")

        out = []

        for task in self.tasks:
            task_dir = os.path.join(ref_root, task)
            if not os.path.isdir(task_dir):
                continue

            ref_csv = self._find_reference_csv(task_dir)
            if ref_csv is None:
                continue

            rows = self._read_pipe_csv(ref_csv)

            for row in rows:
                is_target = self._to_bool(row.get("IsTarget", "False"))
                if self.target_only and not is_target:
                    continue

                probe_rel = self._norm_rel_path(row.get("ProbeFileName"))
                probe_mask_rel = self._norm_rel_path(row.get("ProbeMaskFileName"))

                if probe_rel is None or probe_mask_rel is None:
                    continue

                image_path = self._resolve_rel_path_case_insensitive(root, probe_rel)
                mask_path = self._resolve_rel_path_case_insensitive(root, probe_mask_rel)

                if image_path is None or not os.path.isfile(image_path):
                    continue
                if mask_path is None or not os.path.isfile(mask_path):
                    continue

                out.append(
                    dict(
                        image_path=image_path,
                        mask_path=mask_path,
                        boundary_path=None,
                        task=task,
                        id=row.get(
                            "ProbeFileID",
                            os.path.splitext(os.path.basename(image_path))[0],
                        ),
                        row=row,
                    )
                )

        if self.sort_files:
            out.sort(key=lambda x: x["image_path"])

        return out

    # ---------------- mask helpers ----------------
    def _load_mask(self, mask_path: str, ori_size_hw):
        """
        Returns binary float32 mask with manipulated region = 1.
        """
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        # ensure same size as image if needed
        H, W = ori_size_hw
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # two conventions:
        # 1) manipulated region is white/nonzero
        # 2) manipulated region is black/zero
        if self.invert_mask:
            bin_mask = (mask <= self.mask_threshold).astype(np.float32)
        else:
            bin_mask = (mask > self.mask_threshold).astype(np.float32)

        return bin_mask

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        image = self.load_image(image_path)
        ori_size = image.shape[:2]  # (H, W)

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

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN + "\n" + self.question
        )

        answer = random.choice(self.answer_list)


        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]

        # mask
        sampled_masks = []
        if mask_path is not None and os.path.exists(mask_path):
            mask = self._load_mask(mask_path, ori_size)
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

