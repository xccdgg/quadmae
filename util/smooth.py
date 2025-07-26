"""util/smooth.py — Quaternion‑Wavelet Randomized Smoothing
====================================================================
完整实现，关键特性
------------------
* **像素域 σ_pix → QWT 总噪声 σ_total**：调用 `_sigma_total_from_pixel()`（推导详见论文笔记）
  再拆分成 σ_L, σ_H（σ_H = ratio·σ_L）。
* **仅对四元数虚部 (RGB) 加噪**：每分量方差 σ²/3，实部保持 0。
* **SoftClamp 代替硬 clamp**：`y = limit·tanh(x/limit)`，limit = 3 σ_pix，尾部平滑饱和而主体近似线性。
  训练 / 预测 / 认证 路径保持一致，避免方差被截断。
* **认证半径**：`R = σ_L/(2√3) · Φ⁻¹(p_lower)` —— 低频子带像素域最小标准差。
  （低频 LL: RGB 分量 σ_L/√3，经 1/2 重构。）
* 兼容 ViT/Conv backbone；自动推断输入分辨率。
"""
from __future__ import annotations

import math
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from util.quadatasetgpu import QuaternionWaveletNoise

def _to_bool(v: Any) -> bool:
    """Robust bool conversion for cli strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "t"}
    return bool(v)

__all__ = ["Smooth"]

# ---------------------------------------------------------------------------
# Helper --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _sigma_total_from_pixel(sigma_pix: float, ratio: float) -> float:
    """像素 σ_pix  →  QWT  σ_total  (单层 Haar, RGB 3 虚分量)"""
    return (
        2.0 * (1.0 + ratio)           # 2·(1+r)
        * sigma_pix
        / math.sqrt(1.0 + 3.0 * ratio * ratio)
    )


class SoftClamp(nn.Module):
    """y = limit · tanh(x / limit)。在 (−limit, limit) 近似线性，尾部光滑饱和。"""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.limit * torch.tanh(x / self.limit)


# ---------------------------------------------------------------------------
# Smooth class --------------------------------------------------------------
# ---------------------------------------------------------------------------
class Smooth(nn.Module):
    """Quaternion‑Wavelet randomized smoothing wrapper."""

    ABSTAIN = -1

    # ---------------------------------------------------------------------
    def __init__(
        self,
        base_classifier: nn.Module,
        num_classes: int,
        sigma: float,
        *,
        use_quaternion_noise: bool = True,
        levels: int = 1,
        ratio: float = 3.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.base_classifier = base_classifier.eval()
        self.num_classes = num_classes
        self.sigma_pix = float(sigma)
        self.use_qwt = _to_bool(use_quaternion_noise)
        self.ratio = float(ratio)
        self.levels = int(levels)

        # soft‑clamp limit (≈3 σ 捕获 99.7 % 质量)
        self.soft_limit = 3.0 * self.sigma_pix if self.sigma_pix > 0 else 0.0
        self.soft_clamp = SoftClamp(self.soft_limit) if self.soft_limit > 0 else nn.Identity()

        # 噪声参数 -------------------------------------------------------
        if self.use_qwt:
            # 像素 σ → QWT σ_total → σ_L, σ_H
            self.sigma_total = _sigma_total_from_pixel(self.sigma_pix, self.ratio)
            self.sigma_low = self.sigma_total / (1.0 + self.ratio)
            self.sigma_high = self.sigma_low * self.ratio
            self.sigma_min = min(self.sigma_low, self.sigma_high)
        else:
            # 传统像素高斯
            self.sigma_total = self.sigma_pix
            self.sigma_low = self.sigma_pix
            self.sigma_high = self.sigma_pix
            self.sigma_min = self.sigma_pix

        # device ---------------------------------------------------------
        if device is None:
            try:
                device = next(base_classifier.parameters()).device
            except StopIteration:
                device = "cpu"
        self.device = torch.device(device)

        # 记录参数 --------------------------------------------------------
        if self.use_qwt:
            print(
                f"[Smooth] QWT σ_pix={self.sigma_pix:.4f}  σ_total={self.sigma_total:.4f} "
                f"σ_L={self.sigma_low:.4f}  σ_H={self.sigma_high:.4f}  ratio={self.ratio}  "
                f"soft_limit={self.soft_limit:.4f}"
            )
        else:
            print(
                f"[Smooth] Gaussian σ_pix={self.sigma_pix:.4f}  soft_limit={self.soft_limit:.4f}"
            )

        # 推断输入分辨率 (ViT / Conv)
        try:
            img_size = self.base_classifier.patch_embed.img_size  # type: ignore[attr-defined]
            self.input_size = img_size[0] if isinstance(img_size, (tuple, list)) else img_size
        except AttributeError:
            self.input_size = 224

    # ------------------------------------------------------------------
    # internal helpers --------------------------------------------------
    # ------------------------------------------------------------------
    def _ensure_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(0)
        return x.to(self.device)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.input_size:
            return x
        return F.interpolate(
            x.unsqueeze(0),
            size=self.input_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    # ------------------------------------------------------------------
    # core: add noise + soft‑clamp -------------------------------------
    # ------------------------------------------------------------------
    def _add_noise(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.use_qwt:
            noised = QuaternionWaveletNoise.apply_noise(
                imgs,
                sigma=self.sigma_total,
                filter_name="haar",
                levels=self.levels,
                ratio=self.ratio,
                device=self.device,
            )
            return self.soft_clamp(noised)

        noise = torch.randn_like(imgs) * self.sigma_pix
        return imgs + noise

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Clopper-Pearson lower bound for a Bernoulli proportion."""
        return float(proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0])

    # ------------------------------------------------------------------
    # predict -----------------------------------------------------------
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor, n: int, batch_size: int = 512) -> int:
        """Majority‑vote prediction over *n* noise samples."""
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

    # ------------------------------------------------------------------
    # certify -----------------------------------------------------------
    # ------------------------------------------------------------------
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

        # coarse prediction
        top = self._sample_predict(img, n0, batch_size)
        if top == Smooth.ABSTAIN:
            return Smooth.ABSTAIN, 0.0

        # main sampling for p_lower
        countA = self._sample_count(img, top, n, batch_size)
        p_lower = self._lower_confidence_bound(countA, n, alpha)
        if p_lower < 0.5:
            return Smooth.ABSTAIN, 0.0

        radius = self.sigma_pix * norm.ppf(p_lower)
        return top, radius

    # ------------------------------------------------------------------
    # internal sampling -------------------------------------------------
    # ------------------------------------------------------------------
    def _sample_predict(self, img: torch.Tensor, m: int, bs: int) -> int:
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
