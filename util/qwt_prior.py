from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicQWTPrior(nn.Module):
    """Deterministic level-1 Haar QWT prior extractor for RGB images."""

    def __init__(self, patch_size: int, levels: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        if int(levels) != 1:
            raise ValueError("QWT prior v1 only supports levels=1")
        if int(patch_size) <= 0:
            raise ValueError("patch_size must be positive")
        if int(patch_size) % 2 != 0:
            raise ValueError("QWT prior requires an even patch_size")

        self.patch_size = int(patch_size)
        self.levels = int(levels)
        self.eps = float(eps)

    def _prepare_images(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.dtype]:
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [B, 3, H, W]")

        orig_dtype = images.dtype
        work = images.float()
        h_pad = work.shape[-2] % 2
        w_pad = work.shape[-1] % 2
        if h_pad or w_pad:
            work = F.pad(work, (0, w_pad, 0, h_pad), mode="replicate")
        return work, orig_dtype

    def _to_quaternion(self, images: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(
            images.shape[0],
            1,
            images.shape[2],
            images.shape[3],
            device=images.device,
            dtype=images.dtype,
        )
        return torch.cat([zeros, images], dim=1)

    def _haar_decompose(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = q[:, :, 0::2, 0::2]
        b = q[:, :, 0::2, 1::2]
        c = q[:, :, 1::2, 0::2]
        d = q[:, :, 1::2, 1::2]

        ll = 0.5 * (a + b + c + d)
        lh = 0.5 * (a - b + c - d)
        hl = 0.5 * (a + b - c - d)
        hh = 0.5 * (a - b - c + d)
        return ll, lh, hl, hh

    def band_magnitudes(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        work, _ = self._prepare_images(images)
        q = self._to_quaternion(work)
        ll, lh, hl, hh = self._haar_decompose(q)
        mags = tuple(torch.sqrt(torch.sum(band * band, dim=1) + self.eps) for band in (ll, lh, hl, hh))
        return mags  # type: ignore[return-value]

    def extract_patch_prior(self, images: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self.band_magnitudes(images)
        pool = self.patch_size // 2
        if pool <= 0:
            raise ValueError("Invalid pooling kernel derived from patch_size")

        pooled = []
        for band in (ll, lh, hl, hh):
            pooled_band = F.avg_pool2d(band.unsqueeze(1), kernel_size=pool, stride=pool).squeeze(1)
            pooled.append(pooled_band)

        prior = torch.stack(pooled, dim=-1)  # [B, Hp, Wp, 4]
        prior = prior.flatten(1, 2)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return prior

    def subband_reconstruction_loss(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        *,
        detail_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ll_pred, lh_pred, hl_pred, hh_pred = self.band_magnitudes(pred_images)
        ll_tgt, lh_tgt, hl_tgt, hh_tgt = self.band_magnitudes(target_images)

        detail_pred = torch.sqrt(lh_pred.square() + hl_pred.square() + hh_pred.square() + self.eps)
        detail_tgt = torch.sqrt(lh_tgt.square() + hl_tgt.square() + hh_tgt.square() + self.eps)

        loss_ll = F.l1_loss(ll_pred, ll_tgt)
        loss_detail = F.l1_loss(detail_pred, detail_tgt)
        loss_subband = loss_ll + float(detail_weight) * loss_detail

        metrics = {
            "loss_subband": loss_subband.detach(),
            "loss_subband_ll": loss_ll.detach(),
            "loss_subband_detail": loss_detail.detach(),
        }
        return loss_subband, metrics
