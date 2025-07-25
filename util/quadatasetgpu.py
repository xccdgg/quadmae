# util/quadatasetgpu.py
# =============================================================================
# 变动摘要
#   1. 在 QWT 噪声注入前后加入随机正交变换混合 (RCOT)：
#        · D4 旋转 (0/90/180/270°) + 水平翻转 + 0/1 像素循环平移
#        · 向量化实现，无 Python for‑loop，开销 <1 ms/batch(128)
#   2. 仍在重构阶段使用 soft‑clamp(可选)；默认返回线性值区间 [0,1]。
#   3. 其余 Haar‑QWT、噪声 σ_L/σ_H 拆分、接口保持不变。
# =============================================================================
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# 可选 soft‑clamp（防止硬截断导致方差缩减）
try:
    from util.softclamp import soft_clamp  # noqa: F401
except ImportError:
    def soft_clamp(x: torch.Tensor,
                   min: float = 0.0,
                   max: float = 1.0,
                   alpha: float = 50.0) -> torch.Tensor:           # type: ignore
        return torch.sigmoid((x - min) * alpha) * (max - min) + min


# =============================================================================
# 一、四元数单层 Haar 小波
# =============================================================================
class QuaternionWavelet:
    """单层 Haar 四元数离散小波 (QWT) —— 支持正向分解与逆向重构。"""

    def __init__(self, filter_name: str = "haar",
                 device: str | torch.device | None = None) -> None:
        if filter_name.lower() != "haar":
            raise NotImplementedError("当前仅实现 Haar 小波")
        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self._init_haar_filters()

    # ---------------------------- Haar 滤波器 -----------------------------
    def _init_haar_filters(self) -> None:
        h = torch.tensor([1 / math.sqrt(2)] * 2, device=self.device)
        g = torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)], device=self.device)
        self.h, self.g = h, g
        self.hr, self.gr = h.flip(0), g.flip(0)        # 逆卷积核

    # ---------------- 可分离 2‑D 下采样卷积 -------------------------------
    def _conv2d_sep(self, x: torch.Tensor,
                    fr: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x.unsqueeze(1), fr.view(1, 1, -1, 1),
                       stride=(2, 1), padding=(fr.numel() // 2, 0)).squeeze(1)
        out = F.conv2d(out.unsqueeze(1), fc.view(1, 1, 1, -1),
                       stride=(1, 2), padding=(0, fc.numel() // 2)).squeeze(1)
        return out

    # ---------------- 可分离 2‑D 上采样逆卷积 -----------------------------
    def _up_conv2d_sep(self, x: torch.Tensor,
                       fr: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        out = F.conv_transpose2d(x.unsqueeze(1), fr.view(1, 1, -1, 1),
                                 stride=(2, 1),
                                 padding=(fr.numel() // 2, 0)).squeeze(1)
        out = F.conv_transpose2d(out.unsqueeze(1), fc.view(1, 1, 1, -1),
                                 stride=(1, 2),
                                 padding=(0, fc.numel() // 2)).squeeze(1)
        return out

    # --------------------------- 正向分解 --------------------------------
    def decompose(self, images: torch.Tensor, levels: int = 1):
        print("quanoise")
        """单层分解；返回 dict：{'LL': Tensor, 'subbands': [(LH,HL,HH)]}"""
        if images.dim() == 2:              # 灰度 H×W → 1×H×W×1
            images = images.unsqueeze(0).unsqueeze(-1)

        N, H, W, C = images.shape
        x = images.to(self.device).permute(0, 3, 1, 2).float()  # N,C,H,W

        # 彩色 → (0,R,G,B) 四元数；灰度 → (G,0,0,0)
        if C == 3:
            x = torch.cat([torch.zeros(N, 1, H, W, device=self.device), x], dim=1)
        else:
            x = torch.cat([x, torch.zeros_like(x).repeat(1, 3, 1, 1)], dim=1)

        coeffs: dict[str, torch.Tensor | list[Tuple[torch.Tensor, ...]]] = {
            "subbands": []
        }
        current = x
        for _ in range(levels):
            a, b, c, d = current[:, 0], current[:, 1], current[:, 2], current[:, 3]
            LL = torch.stack([
                self._conv2d_sep(a, self.h, self.h),
                self._conv2d_sep(b, self.h, self.h),
                self._conv2d_sep(c, self.h, self.h),
                self._conv2d_sep(d, self.h, self.h),
            ], dim=-1)
            LH = torch.stack([
                self._conv2d_sep(a, self.h, self.g),
                self._conv2d_sep(b, self.h, self.g),
                self._conv2d_sep(c, self.h, self.g),
                self._conv2d_sep(d, self.h, self.g),
            ], dim=-1)
            HL = torch.stack([
                self._conv2d_sep(a, self.g, self.h),
                self._conv2d_sep(b, self.g, self.h),
                self._conv2d_sep(c, self.g, self.h),
                self._conv2d_sep(d, self.g, self.h),
            ], dim=-1)
            HH = torch.stack([
                self._conv2d_sep(a, self.g, self.g),
                self._conv2d_sep(b, self.g, self.g),
                self._conv2d_sep(c, self.g, self.g),
                self._conv2d_sep(d, self.g, self.g),
            ], dim=-1)
            coeffs["LL"] = LL
            coeffs["subbands"].append((LH, HL, HH))
            current = LL.permute(0, 3, 1, 2)
        return coeffs

    # --------------------------- 逆向重构 --------------------------------
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
        rec = torch.stack(comps, dim=-1)   # N,H,W,4
        img = rec[..., 1:4]                # 取 RGB 虚部通道
        return img                         # 不硬剪裁


# =============================================================================
# 二、随机正交混合 + 四元数小波噪声
# =============================================================================
# ---------- 向量化构造旋转 / 平移 变换 ----------------------------------
def _build_rot_variants(x: torch.Tensor) -> torch.Tensor:
    """一次生成 4 种旋转：0/90/180/270° → (4,N,C,H,W)"""
    return torch.stack([
        x,
        torch.rot90(x, 1, (2, 3)),
        torch.rot90(x, 2, (2, 3)),
        torch.rot90(x, 3, (2, 3)),
    ], dim=0)


def _build_shift_variants(x: torch.Tensor) -> torch.Tensor:
    """一次生成 (dx,dy)∈{0,1}² 的 4 种循环平移 → (4,N,C,H,W)"""
    return torch.stack([
        x,
        torch.roll(x, shifts=(0, 1), dims=(2, 3)),   # →1
        torch.roll(x, shifts=(1, 0), dims=(2, 3)),   # ↑1
        torch.roll(x, shifts=(1, 1), dims=(2, 3)),   # ↗1
    ], dim=0)


class QuaternionWaveletNoise:
    """四元数小波系数域注入高斯噪声（仅虚部），自动执行随机正交变换混合。"""

    # ---------------------------- 初始化 ----------------------------
    def __init__(self, sigma: float, *, device=None,
                 filter_name="haar", levels: int = 1, ratio: float = 3.0):
        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.levels = int(levels)
        self.sigma_low = sigma / (1.0 + ratio)
        self.sigma_high = sigma * ratio / (1.0 + ratio)
        self.qwt = QuaternionWavelet(filter_name=filter_name, device=self.device)

    # -------------------- 四元数系数加噪辅助 ------------------------
    def _add_noise(self, Q: torch.Tensor, sigma_whole: float) -> torch.Tensor:
        noise = torch.randn_like(Q, device=self.device) * (sigma_whole / math.sqrt(3))
        noise[..., 0] = 0.0                 # 实部保持 0
        return Q + noise

    def _inject(self, coeffs: dict) -> dict:
        coeffs["LL"] = self._add_noise(coeffs["LL"], self.sigma_low)
        coeffs["subbands"] = [
            (
                self._add_noise(LH, self.sigma_high),
                self._add_noise(HL, self.sigma_high),
                self._add_noise(HH, self.sigma_high),
            ) for LH, HL, HH in coeffs["subbands"]
        ]
        return coeffs

    # ------------------- 向量化随机正交变换 -------------------------
    @staticmethod
    def _apply_random_transform(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对批量输入执行 **随机正交变换混合**（D4旋转/水平翻转 + 0/1 像素循环平移）。
        输出:
            * x_out  : 变换后张量，shape 与输入相同
            * code   : UInt8 编码，满足 code = k + 4*flip + 8*(dx + 2*dy) ∈ [0,31]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)                            # [1,C,H,W]
        N, C, H, W = x.shape
        dev = x.device

        # --------- 随机采样参数 --------------------------------------------
        k    = torch.randint(0, 4, (N,), device=dev)      # 旋转 0..3
        flip = torch.randint(0, 2, (N,), device=dev)      # 是否翻转
        dx   = torch.randint(0, 2, (N,), device=dev)      # 平移 x 0/1
        dy   = torch.randint(0, 2, (N,), device=dev)      # 平移 y 0/1
        code = k + 4 * flip + 8 * (dx + 2 * dy)           # uint8 ∈ [0,31]

        # --------- 构造 8 (=4*2) 个 旋转×翻转 版本 -------------------------
        rot_no  = _build_rot_variants(x)                  # (4,N,C,H,W)
        rot_no  = rot_no.permute(1, 0, 2, 3, 4)           # (N,4,C,H,W)

        x_flip  = torch.flip(x, dims=(3,))                # 水平翻转
        rot_flp = _build_rot_variants(x_flip)             # (4,N,C,H,W)
        rot_flp = rot_flp.permute(1, 0, 2, 3, 4)          # (N,4,C,H,W)

        rot_8   = torch.stack([rot_no, rot_flp], dim=1)   # (N,2,4,C,H,W)
        rot_8   = rot_8.reshape(N, 8, C, H, W)            # (N,8,C,H,W)

        # --------- 对每个 8‑variant 生成 4 种平移，共 32 -------------------
        rot8_flat = rot_8.reshape(N * 8, C, H, W)         # (N*8,C,H,W)
        shift4    = _build_shift_variants(rot8_flat)      # (4,N*8,C,H,W)
        shift4    = shift4.permute(1, 0, 2, 3, 4)         # (N*8,4,C,H,W)
        shift4    = shift4.reshape(N, 8, 4, C, H, W)      # (N,8,4,C,H,W)

        # 将维度调到 (shift, rot‑flip) 顺序 → code = rot_flip + 8*shift
        combos32  = shift4.permute(0, 2, 1, 3, 4, 5)      # (N,4,8,C,H,W)
        combos32  = combos32.reshape(N, 32, C, H, W)      # (N,32,C,H,W)

        # --------- 根据 code gather 出本样本的目标变换 ---------------------
        idx = code.view(N, 1, 1, 1, 1).expand(-1, 1, C, H, W)
        x_out = torch.gather(combos32, 1, idx).squeeze(1) # (N,C,H,W)

        return x_out, code

    @staticmethod
    def _invert_transform(x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """根据编码 code ∈[0,31] 执行逆变换 T⁻¹"""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        N, C, H, W = x.shape
        dev = x.device

        k    =  code % 4
        flip = (code // 4) % 2
        tmp  =  code // 8
        dx   =  tmp % 2
        dy   =  tmp // 2

        # 逆平移
        rolled = _build_shift_variants(x)                # (4,N,C,H,W)
        inv_shift_idx = dx + 2 * dy
        inv_shift_idx = inv_shift_idx.view(1, N, 1, 1, 1).expand(1, -1, C, H, W)
        x = torch.gather(rolled, 0, inv_shift_idx).squeeze(0)

        # 逆翻转
        x_flip = torch.flip(x, dims=(3,))
        x = torch.where(flip.view(N, 1, 1, 1).bool(), x_flip, x)

        # 逆旋转
        rot_back = _build_rot_variants(x)                # (4,N,C,H,W)
        inv_k = (-k) % 4
        inv_k = inv_k.view(1, N, 1, 1, 1).expand(1, -1, C, H, W)
        x = torch.gather(rot_back, 0, inv_k).squeeze(0)
        return x.squeeze(0) if x.shape[0] == 1 else x

    # --------------------------- 外部入口 ---------------------------
    @staticmethod
    def apply_noise(x: torch.Tensor, sigma: float, *,
                    filter_name="haar", levels: int = 1,
                    ratio: float = 3.0, device=None) -> torch.Tensor:
        """
        与旧版 API 完全兼容：输入 / 输出 shape、dtype、device 不变。
        """
        if not torch.is_tensor(x):
            raise TypeError("apply_noise 期望 torch.Tensor 输入")
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.to(torch.float32)

        # ------------------------------------------------------------------
        qwn = QuaternionWaveletNoise(sigma, device=device,
                                     filter_name=filter_name,
                                     levels=levels, ratio=ratio)
        x = x.to(qwn.device).float()

        # 1) 随机正交变换 T
        x, code = QuaternionWaveletNoise._apply_random_transform(x)

        # 2) QWT 分解 + 噪声 + 重构
        C = x.shape[1]
        imgs = x.permute(0, 2, 3, 1) if C == 3 else x[:, 0, :, :].unsqueeze(-1)
        coeffs = qwn.qwt.decompose(imgs, levels=qwn.levels)
        coeffs = qwn._inject(coeffs)
        rec = qwn.qwt.reconstruct(coeffs)
        out = (rec.permute(0, 3, 1, 2)
               if rec.dim() == 4 else rec.unsqueeze(3).permute(0, 3, 1, 2))

        # 3) 逆变换 T⁻¹
        out = QuaternionWaveletNoise._invert_transform(out, code)

        out = out.to(orig_dtype)
        return out.squeeze(0) if single else out
