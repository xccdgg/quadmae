from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from util.noise import add_noise, fixed_sigma_value, sigma_total_from_pixel, to_bool

__all__ = ["Smooth"]

# Backward-compatible alias for older imports.
_sigma_total_from_pixel = sigma_total_from_pixel


class Smooth(nn.Module):
    ABSTAIN = -1

    def __init__(
        self,
        base_classifier: nn.Module,
        num_classes: int,
        sigma: float,
        *,
        use_quaternion_noise: bool = False,
        levels: int = 1,
        ratio: float = 3.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.base_classifier = base_classifier.eval()
        self.num_classes = num_classes
        self.sigma_pix = fixed_sigma_value(sigma)
        self.use_qwt = to_bool(use_quaternion_noise)
        if self.use_qwt:
            raise ValueError(
                'QWT-corrupted inference cannot be certified with the standard isotropic '
                'Gaussian Cohen radius: matching covariance is not matching distribution. '
                'Use pixel Gaussian noise for certified evaluation. QWT remains available '
                'for training-only corruption until a valid non-Gaussian certificate is implemented.'
            )
        self.ratio = float(ratio)
        self.levels = int(levels)

        if self.use_qwt:
            self.sigma_total = sigma_total_from_pixel(self.sigma_pix, self.ratio)
            self.sigma_low = self.sigma_total / (1.0 + self.ratio)
            self.sigma_high = self.sigma_low * self.ratio
        else:
            self.sigma_total = self.sigma_pix
            self.sigma_low = self.sigma_pix
            self.sigma_high = self.sigma_pix

        if device is None:
            try:
                device = next(base_classifier.parameters()).device
            except StopIteration:
                device = "cpu"
        self.device = torch.device(device)

        if self.use_qwt:
            print(
                f"[Smooth] QWT sigma_pix={self.sigma_pix:.4f} sigma_total={self.sigma_total:.4f} "
                f"sigma_L={self.sigma_low:.4f} sigma_H={self.sigma_high:.4f} ratio={self.ratio}"
            )
        else:
            print(f"[Smooth] Gaussian sigma_pix={self.sigma_pix:.4f}")

        try:
            img_size = self.base_classifier.patch_embed.img_size  # type: ignore[attr-defined]
            self.input_size = img_size[0] if isinstance(img_size, (tuple, list)) else img_size
        except AttributeError:
            self.input_size = 224

    def _ensure_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(0)
        return x.to(self.device)

    def _add_noise(self, imgs: torch.Tensor) -> torch.Tensor:
        return add_noise(
            imgs,
            self.sigma_pix,
            use_quaternion_noise=self.use_qwt,
            levels=self.levels,
            ratio=self.ratio,
            device=self.device,
        )

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return float(proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0])

    @torch.no_grad()
    def predict(self, x: torch.Tensor, n: int, batch_size: int = 512) -> int:
        img = self._ensure_tensor(x)

        counts = np.zeros(self.num_classes, dtype=int)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = end - start
            imgs = img.unsqueeze(0).expand(b, -1, -1, -1)
            imgs = self._add_noise(imgs)
            if imgs.shape[-1] != self.input_size:
                imgs = F.interpolate(
                    imgs,
                    size=self.input_size,
                    mode="bicubic",
                    align_corners=False,
                )
            logits = self.base_classifier(imgs)
            preds = logits.argmax(1).cpu().numpy()
            for p in preds:
                counts[p] += 1
        if (counts == counts.max()).sum() != 1:
            return Smooth.ABSTAIN
        return int(counts.argmax())

    def certify(
        self,
        x: torch.Tensor,
        n0: int,
        n: int,
        alpha: float,
        batch_size: int,
        y: Optional[int] = None,
    ) -> tuple[int, float]:
        img = self._ensure_tensor(x)

        counts_selection = self._sample_counts(img, n0, batch_size)
        top = int(counts_selection.argmax())
        if y is not None and top != y:
            return Smooth.ABSTAIN, 0.0

        counts_estimation = self._sample_counts(img, n, batch_size)
        countA = int(counts_estimation[top])
        p_lower = self._lower_confidence_bound(countA, n, alpha)
        if p_lower < 0.5:
            return Smooth.ABSTAIN, 0.0

        radius = self.sigma_pix * norm.ppf(p_lower)
        return top, radius

    def _sample_counts(self, img: torch.Tensor, m: int, bs: int) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=int)
        with torch.no_grad():
            for start in range(0, m, bs):
                end = min(start + bs, m)
                b = end - start
                imgs = img.unsqueeze(0).expand(b, -1, -1, -1)
                imgs = self._add_noise(imgs)
                if imgs.shape[-1] != self.input_size:
                    imgs = F.interpolate(
                        imgs,
                        size=self.input_size,
                        mode="bicubic",
                        align_corners=False,
                    )
                imgs = imgs.to(self.device)
                preds = self.base_classifier(imgs).argmax(1).cpu().numpy()
                for p in preds:
                    counts[p] += 1
        return counts

    def _sample_predict(self, img: torch.Tensor, m: int, bs: int) -> int:
        counts = self._sample_counts(img, m, bs)
        if (counts == counts.max()).sum() != 1:
            return Smooth.ABSTAIN
        return int(counts.argmax())

    def _sample_count(self, img: torch.Tensor, cls: int, m: int, bs: int) -> int:
        cnt = 0
        with torch.no_grad():
            for start in range(0, m, bs):
                end = min(start + bs, m)
                b = end - start
                imgs = img.unsqueeze(0).expand(b, -1, -1, -1)
                imgs = self._add_noise(imgs)
                if imgs.shape[-1] != self.input_size:
                    imgs = F.interpolate(
                        imgs,
                        size=self.input_size,
                        mode="bicubic",
                        align_corners=False,
                    )
                imgs = imgs.to(self.device)
                preds = self.base_classifier(imgs).argmax(1)
                cnt += int((preds == cls).sum().item())
        return cnt
