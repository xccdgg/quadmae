# util/datasets.py  ── FULL FILE
# =============================================================================
# 主要改动
# -----------------------------------------------------------------------------
# 1.   AddNoise 把命令行传入的 sigma 解释为 **像素域 σ_pix**，
#      然后用与 util.smooth 相同的公式放大成 σ_total → σ_L / σ_H。
# 2.   支持 SoftClamp（与 Smooth 中的 soft_limit 一致），可选地对加噪结果做
#      光滑饱和，避免硬剪裁带来的方差缩减。
# 3.   其余 DataLoader / Dataset 逻辑与原文件一致。
# =============================================================================
from __future__ import annotations

def _to_bool(v: Any) -> bool:
    """Robust bool conversion for cli strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "t"}
    return bool(v)

import math
from typing import Any, Tuple, Optional

import numpy as np
from PIL import Image                     # noqa: F401  (备用：自定义数据集)
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from util.smooth import _sigma_total_from_pixel
from util.quadatasetgpu import QuaternionWaveletNoise
from util.softclamp     import SoftClamp         # 软饱和单独放在 util/softclamp.py


# -----------------------------------------------------------------------------
# AddNoise ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
class AddNoise:
    r"""把 **像素域 σ_pix** 噪声注入到图像（四元数小波 or 像素高斯）。

    Parameters
    ----------
    sigma : float
        目标像素域 σ_pix。
    use_quaternion_noise : bool, default True
        True → QWT 噪声；False → 传统像素高斯噪声。
    levels : int, default 1
        QWT 分解层数（当前实现只支持 1）。
    ratio : float, default 3.0
        σ_H / σ_L。
    device : str / torch.device, default "cpu"
        生成噪声张量所在设备。
    soft_limit : Optional[float]
        若给定，则对加噪后像素做 SoftClamp；默认自动设成 3 σ_pix。
    """

    def __init__(
        self,
        sigma: float,
        *,
        use_quaternion_noise: bool = True,
        levels: int = 1,
        ratio: float = 3.0,
        device: Any = "cpu",
        soft_limit: Optional[float] = None,
    ) -> None:
        self.sigma_pix:   float = float(sigma)
        self.use_qwt     = _to_bool(use_quaternion_noise)
        self.levels:     int   = int(levels)
        self.ratio:      float = float(ratio)
        self.device                = torch.device(device)
        self.soft_limit: Optional[float] = soft_limit if soft_limit is not None else (
            3.0 * self.sigma_pix if self.sigma_pix > 0 else None
        )
        self.soft_clamp = SoftClamp(self.soft_limit) if self.soft_limit else lambda x: x

        # 像素 σ → σ_total → σ_L / σ_H
        self.sigma_total = _sigma_total_from_pixel(self.sigma_pix, self.ratio)
        self.sigma_low   = self.sigma_total / (1.0 + self.ratio)
        self.sigma_high  = self.sigma_low * self.ratio

        print(
            f"[AddNoise] σ_pix={self.sigma_pix:.4f}  σ_total={self.sigma_total:.4f} "
            f"σ_L={self.sigma_low:.4f}  σ_H={self.sigma_high:.4f}  ratio={self.ratio} "
            f"soft_limit={self.soft_limit}"
        )

    # ------------------------------------------------------------------
    def __call__(self, img: Tensor) -> Tensor:
        """对单张 [C,H,W] 或批量 [B,C,H,W] 图像加噪"""
        if not torch.is_tensor(img):
            # PIL → Tensor
            arr = np.asarray(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)

        if self.sigma_pix <= 0:
            return img

        # ---------------- QWT 噪声 ----------------
        if self.use_qwt:
            single = (img.dim() == 3)
            if single:
                img = img.unsqueeze(0)          # [1,C,H,W]

            noisy = QuaternionWaveletNoise.apply_noise(
                img,
                sigma=self.sigma_total,         # BEFORE /2 per component
                filter_name="haar",
                levels=self.levels,
                ratio=self.ratio,
                device=self.device,
            )
            noisy = self.soft_clamp(noisy)
            out = noisy.squeeze(0) if single else noisy
            return out.to(dtype=torch.float16)

        # ---------------- 像素高斯 ----------------
        noise = torch.randn_like(img) * self.sigma_pix
        noisy = (img + noise).clamp(0.0, 1.0)
        return noisy.to(dtype=torch.float16)


# -----------------------------------------------------------------------------
# NoisyImageDataset  &  builder helpers ---------------------------------------
# -----------------------------------------------------------------------------
class NoisyImageDataset(Dataset):
    """CIFAR-10 + 可选噪声（QWT 或 Pixel 高斯）。"""

    def __init__(
        self,
        data_root: str,
        train: bool,
        input_size: int,
        batch_size: int,
        *,
        use_quaternion_noise: bool = True,
        noise_sigma: float = 0.0,
        levels: int = 1,
        ratio: float = 3.0,
    ) -> None:
        self.batch_size = batch_size

        transform = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
        if noise_sigma and noise_sigma > 0:
            transform.append(
                AddNoise(
                    sigma=noise_sigma,
                    use_quaternion_noise=_to_bool(use_quaternion_noise),
                    levels=levels,
                    ratio=ratio,
                )
            )
        self.transform = transforms.Compose(transform)

        self.dataset = CIFAR10(
            root=data_root,
            train=train,
            transform=self.transform,
            download=True,
        )

    # -------------- Dataset API -----------------
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.dataset[idx]

    # -------------- helper ----------------------
    def get_dataloader(self, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


class DatasetWithInterval(Dataset):
    """Wrap an existing dataset to sample items at a fixed interval."""

    def __init__(self, dataset: Dataset, interval: int) -> None:
        self.dataset = dataset
        self.interval = max(1, int(interval))

    def __getitem__(self, index: int):
        return self.dataset[index * self.interval]

    def __len__(self) -> int:
        return len(self.dataset) // self.interval


# ---------------------------------------------------------------------------
# build_dataset / build_dataset_with_interval  ------------------------------
# ---------------------------------------------------------------------------
def build_dataset(split: str, args):
    """Build dataset for fine-tuning and evaluation."""
    is_train = split.lower() == "train"

    # ImageNet detection based on number of classes
    if getattr(args, "nb_classes", 10) == 1000:
        root = os.path.join(args.data_path, "train" if is_train else "val")
        transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
        return ImageFolder(root=root, transform=transform)

    # Default to CIFAR-10
    return NoisyImageDataset(
        data_root=args.data_path,
        train=is_train,
        input_size=args.input_size,
        batch_size=args.batch_size,
        use_quaternion_noise=_to_bool(args.use_quaternion_noise),
        noise_sigma=args.sigma,
        levels=args.levels,
        ratio=args.ratio,
    )


def build_dataset_with_interval(split: str, args):
    """Build dataset and optionally sample it with a fixed interval."""
    dataset = build_dataset(split, args)
    interval = getattr(args, "sample_interval", 1)
    if interval > 1 and split.lower() != "train":
        dataset = DatasetWithInterval(dataset, interval)
    return dataset
