import torch
import torch.nn.functional as F


class SimiLoss:
    """
    Similarity-map supervision for segmentation.

    This class is designed for pipelines like fakeseg where the similarity map
    is usually a normalized map (0~1) converted from token-image similarity.
    It supports:
    - Gaussian BCE loss
    - Gaussian Dice loss
    """

    def __init__(self, eps: float = 1e-6, ksize: int = 31, sigma: float = 7.0):
        self.eps = eps
        self.ksize = ksize
        self.sigma = sigma

    def _to_bhw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0)  # [H, W] -> [1, H, W]
        if x.dim() == 3:
            return x  # [B, H, W]
        if x.dim() == 4 and x.shape[1] == 1:
            return x[:, 0]  # [B, 1, H, W] -> [B, H, W]
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    def _resize_to_gt(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred/gt in [B, H, W]
        if pred.shape[-2:] == gt.shape[-2:]:
            return pred
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return pred

    def _as_logits(self, pred: torch.Tensor, input_is_logits: bool) -> torch.Tensor:
        if input_is_logits:
            return pred
        # pred is treated as probability-like map in [0,1]
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        return torch.logit(pred)

    def _auto_pos_weight(self, target: torch.Tensor) -> torch.Tensor:
        # target in [B, H, W], soft/binary allowed
        pos = float(target.sum().item())
        total = float(target.numel())
        neg = max(total - pos, 0.0)
        w = neg / max(pos, self.eps)
        return torch.tensor(w, dtype=target.dtype, device=target.device)

    def gaussian_heatmap_from_mask(
        self,
        binary_masks: torch.Tensor,
        ksize: int = None,
        sigma: float = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        binary_masks: [B, H, W], values 0/1
        return: [B, H, W] gaussian-soft map
        """
        k = self.ksize if ksize is None else int(ksize)
        s = self.sigma if sigma is None else float(sigma)

        if k % 2 == 0:
            k += 1

        min_dim = min(binary_masks.shape[-2], binary_masks.shape[-1])
        k = min(k, min_dim)
        if k % 2 == 0:
            k = max(3, k - 1)

        device = binary_masks.device
        dtype = binary_masks.dtype

        coords = torch.arange(k, dtype=dtype, device=device) - (k - 1) / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2 * (s ** 2)))
        kernel = kernel / kernel.sum().clamp_min(self.eps)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        x = binary_masks.float().unsqueeze(1)  # [B,1,H,W]
        pad = min(k // 2, min(x.shape[-2], x.shape[-1]) // 2)
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        heat = F.conv2d(x, kernel).squeeze(1)  # [B,H,W]

        if normalize:
            vmax = heat.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
            heat = heat / vmax.clamp_min(self.eps)
        return heat

    def compute_gaussian_bce_loss(
        self,
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        num_masks: float,
        pos_weight=None,
        input_is_logits: bool = True,
        ksize: int = None,
        sigma: float = None,
    ) -> torch.Tensor:
        pred = self._to_bhw(pred_logits)
        gt = self._to_bhw(gt_mask).float()
        pred = self._resize_to_gt(pred, gt)
        logits = self._as_logits(pred, input_is_logits=input_is_logits)

        gaussian_target = self.gaussian_heatmap_from_mask(
            gt, ksize=ksize, sigma=sigma, normalize=True
        )

        if pos_weight is None:
            pos_weight = self._auto_pos_weight(gaussian_target)
        elif not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(
                float(pos_weight), dtype=logits.dtype, device=logits.device
            )
        else:
            pos_weight = pos_weight.to(dtype=logits.dtype, device=logits.device)

        loss = F.binary_cross_entropy_with_logits(
            logits, gaussian_target, reduction="none", pos_weight=pos_weight
        )
        loss = loss.flatten(1).mean(1).sum() / (float(num_masks) + self.eps)
        return loss

    def compute_gaussian_dice_loss(
        self,
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        num_masks: float,
        pos_weight=None,
        input_is_logits: bool = True,
        ksize: int = None,
        sigma: float = None,
    ) -> torch.Tensor:
        pred = self._to_bhw(pred_logits)
        gt = self._to_bhw(gt_mask).float()
        pred = self._resize_to_gt(pred, gt)
        logits = self._as_logits(pred, input_is_logits=input_is_logits)

        gaussian_target = self.gaussian_heatmap_from_mask(
            gt, ksize=ksize, sigma=sigma, normalize=True
        )

        if pos_weight is None:
            pos_weight = self._auto_pos_weight(gaussian_target)
        elif not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(
                float(pos_weight), dtype=logits.dtype, device=logits.device
            )
        else:
            pos_weight = pos_weight.to(dtype=logits.dtype, device=logits.device)

        probs = logits.sigmoid().flatten(1)
        target = gaussian_target.flatten(1)
        weighted_target = target * pos_weight

        numerator = 2.0 * (probs * weighted_target).sum(-1)
        denominator = probs.sum(-1) + weighted_target.sum(-1)
        loss = 1.0 - (numerator + self.eps) / (denominator + self.eps)
        loss = loss.sum() / (float(num_masks) + self.eps)
        return loss

    def compute_gaussian_losses(
        self,
        pred_map: torch.Tensor,
        gt_mask: torch.Tensor,
        num_masks: float,
        input_is_logits: bool = False,
        pos_weight=None,
        ksize: int = None,
        sigma: float = None,
    ):
        """
        Convenience wrapper.
        For fakeseg similarity maps, keep input_is_logits=False by default.
        """
        bce = self.compute_gaussian_bce_loss(
            pred_map,
            gt_mask,
            num_masks=num_masks,
            pos_weight=pos_weight,
            input_is_logits=input_is_logits,
            ksize=ksize,
            sigma=sigma,
        )
        dice = self.compute_gaussian_dice_loss(
            pred_map,
            gt_mask,
            num_masks=num_masks,
            pos_weight=pos_weight,
            input_is_logits=input_is_logits,
            ksize=ksize,
            sigma=sigma,
        )
        return bce, dice
