from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from util.quadatasetgpu import QuaternionWaveletNoise


@dataclass(frozen=True)
class SigmaSpec:
    kind: str
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "t"}
    return bool(v)


def parse_sigma_spec(value: Any) -> SigmaSpec:
    if isinstance(value, SigmaSpec):
        return value

    if isinstance(value, (int, float)):
        sigma = float(value)
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        return SigmaSpec(kind="fixed", value=sigma)

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()

        if "," in text:
            parts = [part.strip() for part in text.split(",")]
            if len(parts) != 2:
                raise ValueError(f"invalid sigma range: {value!r}")
            low, high = float(parts[0]), float(parts[1])
            if low < 0 or high < 0:
                raise ValueError("sigma range values must be non-negative")
            if high < low:
                raise ValueError("sigma range high must be greater than or equal to low")
            return SigmaSpec(kind="uniform", low=low, high=high)

        sigma = float(text)
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        return SigmaSpec(kind="fixed", value=sigma)

    raise TypeError(f"unsupported sigma value type: {type(value).__name__}")


def format_sigma_spec(value: Any) -> str:
    spec = parse_sigma_spec(value)
    if spec.kind == "fixed":
        return f"{float(spec.value):g}"
    return f"[{float(spec.low):g},{float(spec.high):g}]"


def fixed_sigma_value(value: Any) -> float:
    spec = parse_sigma_spec(value)
    if spec.kind != "fixed":
        raise ValueError(
            "Certification requires a fixed evaluate sigma; "
            f"got training sigma range {format_sigma_spec(spec)}"
        )
    return float(spec.value)


def sigma_spec_max(value: Any) -> float:
    spec = parse_sigma_spec(value)
    if spec.kind == "fixed":
        return float(spec.value)
    return float(spec.high)


def sample_sigma(value: Any, device: Any = None) -> float:
    spec = parse_sigma_spec(value)
    if spec.kind == "fixed":
        return float(spec.value)

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return float(
        torch.empty((), device=target_device)
        .uniform_(float(spec.low), float(spec.high))
        .item()
    )


def sigma_total_from_pixel(sigma_pix: float, ratio: float) -> float:
    return 2.0 * (1.0 + ratio) * sigma_pix / math.sqrt(1.0 + 3.0 * ratio * ratio)


def add_noise(
    img: torch.Tensor,
    sigma: Any,
    *,
    use_quaternion_noise: bool = False,
    levels: int = 1,
    ratio: float = 3.0,
    device: Any = None,
) -> torch.Tensor:
    if not torch.is_tensor(img):
        raise TypeError("add_noise expects a torch.Tensor input")

    target_device = torch.device(device) if device is not None else img.device
    sigma = sample_sigma(sigma, target_device)
    if sigma <= 0:
        return img

    if to_bool(use_quaternion_noise):
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
