#!/usr/bin/env python3
# verify_qwt_noise_code.py

"""
验证 util.quadatasetgpu 中 QuaternionWaveletNoise.apply_noise 的像素域单通道方差
—— Monte Carlo 实验与理论值对比。
"""

import numpy as np
import torch
from util.quadatasetgpu import QuaternionWaveletNoise
from util.noise import sigma_total_from_pixel

def main():
    # — 1）参数设定 —
    sigma_pix = 0.5  # 目标像素域噪声标准差
    ratio     = 3.0  # 高频/低频 标准差比

    # — 2）计算 QWT 域总体 σ_total 及子带 σ_L, σ_H —
    sigma_total = sigma_total_from_pixel(sigma_pix, ratio)
    sigma_L     = sigma_total / (1.0 + ratio)
    sigma_H     = sigma_L * ratio

    # — 3）理论单通道像素方差 var_chan = (σ_L² + 3·σ_H²) / 12 —
    var_theo_chan = (sigma_L**2 + 3.0 * sigma_H**2) / 12.0

    print(f"[Verify] sigma_pix={sigma_pix:.4f}, ratio={ratio}")
    print(f"[Verify] sigma_total={sigma_total:.6f}, sigma_L={sigma_L:.6f}, sigma_H={sigma_H:.6f}")
    print(f"[Verify] 理论单通道方差 = {var_theo_chan:.6f}; target={sigma_pix**2:.6f}")

    # — 4）Monte Carlo 实验 — 在 2×2 全零块上叠 N 次噪声，3 通道
    N = 20_000
    device = torch.device("cpu")  # 用 CPU 可以避免显卡资源波动
    zeros = torch.zeros(N, 3, 2, 2, device=device)

    # 调用你的加噪实现
    noised = QuaternionWaveletNoise.apply_noise(
        zeros,
        sigma=sigma_total,
        filter_name="haar",
        levels=1,
        ratio=ratio,
        device=device
    ).cpu().numpy()

    # — 5）计算经验方差 — 对 N,H,W 轴求方差，再对 3 个通道取平均
    chan_vars   = noised.var(axis=(0,2,3))   # 返回长度 3 的 array
    var_emp_chan = chan_vars.mean()

    print(f"[Verify] 实验 per-channel 方差 = {var_emp_chan:.6f}")

    # — 6）误差检查：目标为 sigma_pix**2 —
    rel_err = abs(var_emp_chan - var_theo_chan) / var_theo_chan
    print(f"[Verify] 相对误差 = {rel_err:.4%}")

    assert rel_err < 0.01, f"通道方差误差超过 1%：{rel_err:.2%}"

if __name__ == "__main__":
    main()
