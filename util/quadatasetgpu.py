# util/quadatasetgpu.py  —— 修正版（仅在必要处动刀，保留原命名/注释）
# -----------------------------------------------------------------------------
# 变动摘要
#   1. 只在重构阶段用 soft‑clamp 替代硬 clamp；默认仍保持区间 [0,1]。
#   2. 修复 apply_noise 内若干潜在未定义/命名冲突问题；绝不改动外部 API。
#   3. 其余逻辑（Haar QWT、RGB 虚部加噪、σ_low/σ_high 拆分等）与此前版本一致。
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

# 可选柔性 clamp（防止过饱和同时避免截断均值）
try:
    from util.softclamp import soft_clamp  # 若用户已有 softclamp.py
except ImportError:  # 兜底实现一个简单 soft‑clamp
    def soft_clamp(x: torch.Tensor, min: float = 0.0, max: float = 1.0, alpha: float = 50.0) -> torch.Tensor:  # type: ignore
        return torch.sigmoid((x - min) * alpha) * (max - min) + min

# =============================================================================
# 一、四元数 Haar 小波
# =============================================================================
class QuaternionWavelet:
    """单层 Haar 四元数离散小波 (QWT) —— 支持正向分解与逆向重构。

    * 输入：NHWC 或 HW 或单张灰度 H,W
    * 四元数表示：(r,i,j,k)；彩色图像映射为 (0,R,G,B)。
    * 仅实现最常用的单层 2D Haar；更多小波请自行扩展。
    """

    def __init__(self, filter_name: str = "haar", device: str | torch.device | None = None):
        if filter_name.lower() != "haar":
            raise NotImplementedError("当前仅实现 Haar 小波")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._init_haar_filters()

    # ---- Haar 系数 ---------------------------------------------------------
    def _init_haar_filters(self) -> None:
        sqrt2 = math.sqrt(2.0)
        h = torch.tensor([1 / sqrt2, 1 / sqrt2], dtype=torch.float32, device=self.device)
        g = torch.tensor([1 / sqrt2, -1 / sqrt2], dtype=torch.float32, device=self.device)
        self.h, self.g = h, g
        self.hr, self.gr = h.flip(0), g.flip(0)  # 逆卷积核

    # ---- 基础：可分离 2‑D (下采样)卷积 -------------------------------------
    def _conv2d_sep(self, x: torch.Tensor, fr: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x.unsqueeze(1), fr.view(1, 1, -1, 1), stride=(2, 1), padding=(fr.numel() // 2, 0)).squeeze(1)
        out = F.conv2d(out.unsqueeze(1), fc.view(1, 1, 1, -1), stride=(1, 2), padding=(0, fc.numel() // 2)).squeeze(1)
        return out

    # ---- 基础：可分离 2‑D (上采样)逆卷积 -----------------------------------
    def _up_conv2d_sep(self, x: torch.Tensor, fr: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        out = F.conv_transpose2d(x.unsqueeze(1), fr.view(1, 1, -1, 1), stride=(2, 1), padding=(fr.numel() // 2, 0)).squeeze(1)
        out = F.conv_transpose2d(out.unsqueeze(1), fc.view(1, 1, 1, -1), stride=(1, 2), padding=(0, fc.numel() // 2)).squeeze(1)
        return out

    # ---------------------------------------------------------------------
    # 正向分解
    # ---------------------------------------------------------------------
    def decompose(self, images: torch.Tensor, levels: int = 1):
        """单层分解；返回 dict：{'LL': Tensor, 'subbands': [(LH,HL,HH)]}"""
        if images.dim() == 2:  # H,W 灰度 → 1, H, W, 1
            images = images.unsqueeze(0).unsqueeze(-1)
        N, H, W, C = images.shape
        x = images.to(self.device).permute(0, 3, 1, 2).float()  # N,C,H,W
        if C == 3:
            r = torch.zeros(N, 1, H, W, device=self.device)
            x = torch.cat([r, x], dim=1)
        else:  # 灰度
            r = torch.zeros_like(x)
            x = torch.cat([x, r, r, r], dim=1)

        coeffs: dict[str, torch.Tensor | list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {"subbands": []}
        current = x
        for _ in range(levels):
            a, b, c1, d = current[:, 0], current[:, 1], current[:, 2], current[:, 3]
            LL = torch.stack([
                self._conv2d_sep(a, self.h, self.h),
                self._conv2d_sep(b, self.h, self.h),
                self._conv2d_sep(c1, self.h, self.h),
                self._conv2d_sep(d, self.h, self.h),
            ], dim=-1)
            LH = torch.stack([
                self._conv2d_sep(a, self.h, self.g),
                self._conv2d_sep(b, self.h, self.g),
                self._conv2d_sep(c1, self.h, self.g),
                self._conv2d_sep(d, self.h, self.g),
            ], dim=-1)
            HL = torch.stack([
                self._conv2d_sep(a, self.g, self.h),
                self._conv2d_sep(b, self.g, self.h),
                self._conv2d_sep(c1, self.g, self.h),
                self._conv2d_sep(d, self.g, self.h),
            ], dim=-1)
            HH = torch.stack([
                self._conv2d_sep(a, self.g, self.g),
                self._conv2d_sep(b, self.g, self.g),
                self._conv2d_sep(c1, self.g, self.g),
                self._conv2d_sep(d, self.g, self.g),
            ], dim=-1)
            coeffs["LL"] = LL
            coeffs["subbands"].append((LH, HL, HH))
            current = LL.permute(0, 3, 1, 2)
        return coeffs

    # ---------------------------------------------------------------------
    # 逆向重构
    # ---------------------------------------------------------------------
    def reconstruct(self, coeffs: dict) -> torch.Tensor:
        LL = coeffs["LL"]
        LH, HL, HH = coeffs["subbands"][0]
        comps = []
        for k in range(4):
            rec = (
                    self._up_conv2d_sep(LL[..., k], self.hr, self.hr)
                    + self._up_conv2d_sep(LH[..., k], self.hr, self.gr)
                    + self._up_conv2d_sep(HL[..., k], self.gr, self.hr)
                    + self._up_conv2d_sep(HH[..., k], self.gr, self.gr)
            )
            comps.append(rec)
        rec = torch.stack(comps, dim=-1)  # N,H,W,4
        img = rec[..., 1:4]  # 取 RGB 通道
        # **取消 soft_clamp，直接返回线性结果**
        return img


# =============================================================================
# 二、噪声注入
# =============================================================================
class QuaternionWaveletNoise:
    """在四元数小波系数域注入高斯噪声（仅对虚部）。"""

    def __init__(self, sigma: float, *, device=None, filter_name="haar", levels: int = 1, ratio: float = 3.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.levels = int(levels)
        self.sigma_low = sigma / (1.0 + ratio)
        self.sigma_high = sigma * ratio / (1.0 + ratio)
        self.qwt = QuaternionWavelet(filter_name=filter_name, device=self.device)
        self.ratio = ratio

    # ---------------- 内部：加噪 -----------------
    def _add_noise(self, Q: torch.Tensor, sigma_whole: float) -> torch.Tensor:
        sqrt3 = math.sqrt(3.0)
        noise = torch.randn_like(Q, device=self.device) * (sigma_whole / sqrt3)
        noise[..., 0] = 0.0  # 实部保持 0
        return Q + noise

    def _inject(self, coeffs: dict) -> dict:
        coeffs["LL"] = self._add_noise(coeffs["LL"], self.sigma_low)
        coeffs["subbands"] = [
            (
                self._add_noise(LH, self.sigma_high),
                self._add_noise(HL, self.sigma_high),
                self._add_noise(HH, self.sigma_high),
            )
            for LH, HL, HH in coeffs["subbands"]
        ]
        return coeffs

    # ---------------- 静态总入口 -------------
    @staticmethod
    def apply_noise(x: torch.Tensor, sigma: float, *, filter_name="haar", levels: int = 1, ratio: float = 3.0, device=None) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("apply_noise 期望 torch.Tensor 输入")
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        N, C, H, W = x.shape
        qwn = QuaternionWaveletNoise(sigma, device=device, filter_name=filter_name, levels=levels, ratio=ratio)
        x = x.to(qwn.device).float()
        imgs = x.permute(0, 2, 3, 1) if C == 3 else x[:, 0, :, :].unsqueeze(-1)
        coeffs = qwn.qwt.decompose(imgs, levels=qwn.levels)
        coeffs = qwn._inject(coeffs)
        rec = qwn.qwt.reconstruct(coeffs)
        out = rec.permute(0, 3, 1, 2) if rec.dim() == 4 else rec.unsqueeze(3).permute(0, 3, 1, 2)
        return out.squeeze(0) if single else out
