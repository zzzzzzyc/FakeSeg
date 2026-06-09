import os

import cv2
import torch


def load_prior_pair(image_path, image_hw, fg_dir=None, bg_dir=None):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    prior_fg = _load_single_prior(_build_prior_path(fg_dir, stem), image_hw)
    prior_bg = _load_single_prior(_build_prior_path(bg_dir, stem), image_hw)
    return prior_fg, prior_bg


def _build_prior_path(prior_dir, stem):
    if prior_dir is None:
        return None
    return os.path.join(prior_dir, stem + ".png")


def _load_single_prior(prior_path, image_hw):
    height, width = image_hw
    if prior_path is None:
        print("[WARN] prior path is None, using zeros.", flush=True)
        return torch.zeros((1, height, width), dtype=torch.float32)
    if not os.path.isfile(prior_path):
        print(f"[WARN] prior file not found: {prior_path}, using zeros.", flush=True)
        return torch.zeros((1, height, width), dtype=torch.float32)

    prior = cv2.imread(prior_path, cv2.IMREAD_GRAYSCALE)
    if prior is None:
        print(f"[WARN] failed to read prior file: {prior_path}, using zeros.", flush=True)
        return torch.zeros((1, height, width), dtype=torch.float32)

    if prior.shape != (height, width):
        prior = cv2.resize(prior, (width, height), interpolation=cv2.INTER_NEAREST)

    prior = torch.from_numpy(prior).float().unsqueeze(0) / 255.0
    return prior
