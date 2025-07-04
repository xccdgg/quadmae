#!/usr/bin/env python3
"""Visualize RCOT restoration on a single image."""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from models_rcot import rcot_dmae_vit_base_patch16
from util.datasets import AddNoise


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCOT restoration visualization")
    p.add_argument("--img", required=True, help="input image path")
    p.add_argument("--ckpt", required=True, help="RCOT checkpoint")
    p.add_argument("--dmae_ckpt", default=None, help="pretrained DMAE weights")
    p.add_argument("--sigma", type=float, default=0.5, help="pixel noise std")
    p.add_argument(
        "--use_quaternion_noise",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="use quaternion wavelet noise",
    )
    p.add_argument("--levels", type=int, default=1, help="QWT decomposition levels")
    p.add_argument("--ratio", type=float, default=3.0, help="sigma_H / sigma_L")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--no_rcot", action="store_true", help="disable second stage")
    return p.parse_args()


def load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tf = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return tf(img).unsqueeze(0)


def main() -> None:
    args = get_args()

    device = torch.device(args.device)
    model = rcot_dmae_vit_base_patch16(dmae_ckpt=args.dmae_ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.float()
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    img = load_image(args.img, model.base.patch_embed.img_size).to(device)

    if args.sigma > 0:
        noise = AddNoise(
            args.sigma,
            use_quaternion_noise=args.use_quaternion_noise,
            levels=args.levels,
            ratio=args.ratio,
            device=device,
        )
        noisy = noise(img)
    else:
        noisy = img.clone()

    with torch.no_grad():
        restored = model.restore(noisy, use_rcot=not args.no_rcot)

    def to_np(t: torch.Tensor) -> np.ndarray:
        """Convert a tensor image to a NumPy array Matplotlib can plot."""
        return (
            t.squeeze(0)
            .float()
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(to_np(img))
    axes[0].set_title("Clean")
    axes[0].axis("off")
    axes[1].imshow(to_np(noisy))
    axes[1].set_title("Noisy")
    axes[1].axis("off")
    axes[2].imshow(to_np(restored))
    axes[2].set_title("Restored")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
