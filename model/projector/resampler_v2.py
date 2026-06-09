# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List, Union
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .slice_process import slice_image_feature_minicpm
import torchvision.ops.roi_align as roi_align

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    dtype = abs_pos.dtype

    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class AdaptSpatialResampler_v1(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            in_dims=(64, 128, 320, 512),
            out_dim=512,
            grid_size=4,
            num_heads=8,
            roi_output_candidates=((3, 3), (2, 3), (3, 2), (2, 4), (4, 2)),
    ):
        super().__init__()
        self.num_levels = len(in_dims)          # 现在 = 4
        self.grid_size = grid_size
        self.num_queries = grid_size * grid_size
        self.out_dim = out_dim
        self.roi_output_candidates = roi_output_candidates

        self.local_project = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
            for in_dim in in_dims
        ])
        self.level_embed = nn.Embedding(self.num_levels, out_dim)

        # 3) query
        self.query = nn.Parameter(torch.randn(self.num_queries, out_dim) * 0.02)

        # 4) 2D query pos embedding
        pos = get_2d_sincos_pos_embed(out_dim, grid_size)
        self.query_pos = nn.Parameter(
            torch.from_numpy(pos).float(),
            requires_grad=False
        )

        self.ln_q = nn.LayerNorm(out_dim)
        self.ln_k = nn.LayerNorm(out_dim)
        self.ln_v = nn.LayerNorm(out_dim)
        self.ln_post = nn.LayerNorm(out_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.proj = nn.Linear(out_dim, out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cal_best_pooling_size(self, feature_wh_ratio=1.0):
        log_feature_wh_ratio = math.log(feature_wh_ratio)
        best_pooling_size = (3, 3)   # (h, w)
        min_error = float("inf")

        for w, h in self.roi_output_candidates:
            error = abs(log_feature_wh_ratio - math.log(w / h))
            if error < min_error:
                best_pooling_size = (h, w)
                min_error = error
        return best_pooling_size

    def build_rois(self, H, W, device):
        """
        生成 N x N 个 float windows
        输出: [K, 5], K = N*N
        格式: [batch_idx, x1, y1, x2, y2]
        """
        N = self.grid_size
        rois = []

        win_w = W / N
        win_h = H / N

        for i in range(N):
            for j in range(N):
                x1 = j * win_w
                y1 = i * win_h
                x2 = (j + 1) * win_w
                y2 = (i + 1) * win_h
                rois.append([0.0, x1, y1, x2, y2])

        return torch.tensor(rois, dtype=torch.float32, device=device)

    def expand_rois_for_batch(self, rois_template, B):
        K = rois_template.shape[0]
        rois_all = []
        for b in range(B):
            rois_b = rois_template.clone()
            rois_b[:, 0] = b
            rois_all.append(rois_b)
        return torch.cat(rois_all, dim=0)  # [B*K, 5]

    def sample_level_windows(self, feat, level_idx):
        """
        feat: [B, C, H, W]

        return:
            key_roi:   [B, K, L, C]
            value_roi: [B, K, L, C]
        """
        B, _, H, W = feat.shape
        device = feat.device

        # 通道统一
        feat = self.local_project[level_idx](feat)  # [B, out_dim, H, W]

        # level embedding
        lvl_emb = self.level_embed.weight[level_idx].view(1, self.out_dim, 1, 1)
        feat_k = feat + lvl_emb
        feat_v = feat

        wh_ratio = W / max(H, 1)
        out_h, out_w = self.cal_best_pooling_size(wh_ratio)

        rois_template = self.build_rois(H, W, device)
        rois = self.expand_rois_for_batch(rois_template, B)

        key_roi = roi_align(
            feat_k.float(),
            rois.float(),
            output_size=(out_h, out_w),
            spatial_scale=1.0,
            aligned=True,
        ).to(feat_k.dtype)

        value_roi = roi_align(
            feat_v.float(),
            rois.float(),
            output_size=(out_h, out_w),
            spatial_scale=1.0,
            aligned=True,
        ).to(feat_v.dtype)

        K = self.num_queries
        L = out_h * out_w

        # [B*K, C, out_h, out_w] -> [B, K, L, C]
        key_roi = key_roi.flatten(2).transpose(1, 2).reshape(B, K, L, self.out_dim)
        value_roi = value_roi.flatten(2).transpose(1, 2).reshape(B, K, L, self.out_dim)

        return key_roi, value_roi

    def forward(self, feats):
        """
        feats: [c1, c2, c3, c4]
            c1: [B,  64, H1, W1]
            c2: [B, 128, H2, W2]
            c3: [B, 320, H3, W3]
            c4: [B, 512, H4, W4]

        return:
            tokens: [B, N*N, out_dim]
        """
        assert len(feats) == self.num_levels, \
            f"Expected {self.num_levels} levels, got {len(feats)}"

        B = feats[0].shape[0]
        device = feats[0].device
        dtype = feats[0].dtype

        all_keys = []
        all_values = []

        # 每个尺度分别 RoIAlign
        for lvl, feat in enumerate(feats):
            k_l, v_l = self.sample_level_windows(feat, lvl)
            all_keys.append(k_l)
            all_values.append(v_l)

        # 跨尺度拼接
        # key/value: [B, K, total_L, C]
        key_tokens = torch.cat(all_keys, dim=2)
        value_tokens = torch.cat(all_values, dim=2)

        # query: [B, K, C]
        query = self.query.to(device=device, dtype=dtype)
        query_pos = self.query_pos.to(device=device, dtype=dtype)
        query = query.unsqueeze(0).expand(B, -1, -1) + query_pos.unsqueeze(0)

        query = self.ln_q(query)
        key_tokens = self.ln_k(key_tokens)
        value_tokens = self.ln_v(value_tokens)

        # 每个窗口一组 cross-attention
        outs = []
        for k in range(self.num_queries):
            q_k = query[:, k:k+1, :]        # [B, 1, C]
            k_k = key_tokens[:, k, :, :]    # [B, L, C]
            v_k = value_tokens[:, k, :, :]  # [B, L, C]

            out_k, _ = self.attn(q_k, k_k, v_k)
            outs.append(out_k)

        # [B, K, C]
        tokens = torch.cat(outs, dim=1)
        tokens = self.ln_post(tokens)
        tokens = self.proj(tokens)

        return tokens