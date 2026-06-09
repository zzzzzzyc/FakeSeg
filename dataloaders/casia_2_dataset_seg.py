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



class CASIADataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = IGNORE_LABEL

    def __init__(
        self,
        base_dir,
        tokenizer,
        vision_tower,
        # samples_per_epoch=10000,
        image_size=224,
        precision: str = "fp32",

    ):
        self.base_dir = base_dir
        # self.samples_per_epoch = samples_per_epoch
        self.tokenizer = tokenizer
        self.precision = precision

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        # dirs
        casia_root = os.path.join(base_dir, "CASIA2")

        self.tp_dir = os.path.join(casia_root, "Tp")
        self.au_dir = os.path.join(casia_root, "Au")
        self.gt_dir = os.path.join(casia_root, "CASIA_2_Groundtruth")
        self.prior_fg_dir = os.path.join(base_dir, "train", "trainset_fg")
        self.prior_bg_dir = os.path.join(base_dir, "train", "trainset_bg")

        IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        self.tp_images = sorted(
            f for f in os.listdir(self.tp_dir)
            if f.lower().endswith(IMG_EXT)
        )

        self.au_images = sorted(
            f for f in os.listdir(self.au_dir)
            if f.lower().endswith(IMG_EXT)
        )

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

        # print(f"Tampered images: {len(self.tp_images)}")
        # print(f"Authentic images: {len(self.au_images)}")
        self.samples = self._collect_samples()
        print(
            f"[CASIA] total samples: {len(self.samples)} | "
            f"tampered: {sum(s['is_tampered'] for s in self.samples)} | "
            f"real: {sum(not s['is_tampered'] for s in self.samples)}"
        )

        self.question = (
            "Can you identify if this image is real or tampered image?\n"
            "If tampered, please segment the manipulated area."
        )

    # def __len__(self):
    #     return self.samples_per_epoch
    def __len__(self):
        return len(self.samples)

    def _collect_samples(self):
        samples = []

        # -------- tampered samples --------
        for name in self.tp_images:
            image_path = os.path.join(self.tp_dir, name)
            if not os.path.isfile(image_path):
                continue

            gt_name = self._get_gt_name(name)
            mask_path = os.path.join(self.gt_dir, gt_name)
            if not os.path.isfile(mask_path):
                continue

            samples.append(
                dict(
                    image_path=image_path,
                    mask_path=mask_path,
                    is_tampered=True,
                )
            )

        # # -------- real (authentic) samples --------
        # for name in self.au_images:
        #     image_path = os.path.join(self.au_dir, name)
        #     if not os.path.isfile(image_path):
        #         continue
        #
        #     samples.append(
        #         dict(
        #             image_path=image_path,
        #             mask_path=None,
        #             is_tampered=False,
        #         )
        #     )

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

    def fix_mask(self, mask, img_h, img_w, max_diff=5):

        mh, mw = mask.shape
        if (mh, mw) == (img_h, img_w):
            return np.ascontiguousarray(mask)

        if (mh, mw) == (img_w, img_h):

            mask = np.rot90(mask, k=1)

            if mask.shape == (img_h, img_w):
                return np.ascontiguousarray(mask)

        if abs(mh - img_h) <= max_diff and abs(mw - img_w) <= max_diff:
            mask = cv2.resize(
                mask,
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST
            )
            return np.ascontiguousarray(mask)

        return None


    def _get_gt_name(self, img_name):
        return os.path.splitext(img_name)[0] + "_gt.png"

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        is_tampered = sample["is_tampered"]

        # load image
        image = self.load_image(image_path)
        ori_size = image.shape[:2]  # (H, W)

        # CLIP image
        image_clip = self.clip_image_processor.preprocess(
            image,
            return_tensors="pt",
        )["pixel_values"][0]

        # SAM image
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]  # resized H, W before pad
        image_sam = self.preprocess_image(
            torch.from_numpy(image_sam).permute(2, 0, 1).float()
        )

        # conversation
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN + "\n" + self.question
        )

        if is_tampered:
            answer = random.choice(self.answer_list)
        else:
            answer = random.choice(self.real_answer_list)

        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]

        # masks
        if is_tampered and mask_path is not None and os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            mask = (mask > 0).astype(np.float32)

            img_h, img_w = ori_size
            # mask = self.fix_mask(mask, img_h, img_w)

            if mask is None:
                print(
                    f"[CASIA][mask-fix-failed] image={image_path} mask={mask_path} "
                    f"image_size={(img_h, img_w)}",
                    flush=True,
                )
                masks = torch.zeros((0, img_h, img_w), dtype=torch.float32)
                exists = [False]
            else:
                mask_sum = float(mask.sum())
                if mask_sum <= 0:
                    print(
                        f"[CASIA][empty-mask] image={image_path} mask={mask_path} "
                        f"image_size={(img_h, img_w)} mask_shape={mask.shape} mask_sum={mask_sum}",
                        flush=True,
                    )
                masks = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
                exists = [True]
        else:
            masks = torch.zeros((0, ori_size[0], ori_size[1]), dtype=torch.float32)
            exists = [False]

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
