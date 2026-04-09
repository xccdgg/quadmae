from __future__ import annotations

import math
from typing import Any

import torch

from util.quadatasetgpu import QuaternionWaveletNoise


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "t"}
    return bool(v)


def sigma_total_from_pixel(sigma_pix: float, ratio: float) -> float:
    return 2.0 * (1.0 + ratio) * sigma_pix / math.sqrt(1.0 + 3.0 * ratio * ratio)


def add_noise(
    img: torch.Tensor,
    sigma: float,
    *,
    use_quaternion_noise: bool = False,
    levels: int = 1,
    ratio: float = 3.0,
    device: Any = None,
) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        return img

    if not torch.is_tensor(img):
        raise TypeError("add_noise expects a torch.Tensor input")

    if to_bool(use_quaternion_noise):
        target_device = torch.device(device) if device is not None else img.device
        sigma_total = sigma_total_from_pixel(sigma, float(ratio))
        return QuaternionWaveletNoise.apply_noise(
            img,
            sigma=sigma_total,
            filter_name="haar",
            levels=int(levels),
            ratio=float(ratio),
            device=target_device,
        )

    return img + torch.randn_like(img) * sigma
