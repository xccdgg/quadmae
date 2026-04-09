#!/usr/bin/env python3
# verify_qwt_noise.py

"""
验证 util.quadatasetgpu 中 QuaternionWaveletNoise.apply_noise 的像素域单通道方差
—— Monte Carlo 实验与理论值对比。
"""

import numpy as np
import torch
from util.quadatasetgpu import QuaternionWaveletNoise

def main():
    # 1) 参数设定：像素域目标标准差 σ_pix 和 高频/低频 比 ratio
    sigma_pix = 0.5
    ratio     = 3.0

    # 2) 计算 QWT 域总噪声 σ_total 及其子带 σ_L, σ_H
    sigma_total = sigma_pix * 4.0 * (1.0 + ratio) / np.sqrt(3.0 * (1.0 + 3.0 * ratio * ratio))
    sigma_L     = sigma_total / (1.0 + ratio)
    sigma_H     = sigma_L * ratio

    # 3) 理论单通道像素噪声方差：
    #    gray-pixel 方差 = (σ_L^2 + 3·σ_H^2) / 4
    #    每通道方差 = gray-pixel 方差 / 3
    var_theo_gray = (sigma_L**2 + 3.0 * sigma_H**2) / 4.0
    var_theo_chan = var_theo_gray / 3.0

    print(f"[Verify] sigma_pix={sigma_pix:.4f}, ratio={ratio:.1f}")
    print(f"[Verify] sigma_total={sigma_total:.6f}, sigma_L={sigma_L:.6f}, sigma_H={sigma_H:.6f}")
    print(f"[Verify] 理论 gray-pixel 方差 = {var_theo_gray:.6f}")
    print(f"[Verify] 理论 per-channel 方差 = {var_theo_chan:.6f}")

    # 4) Monte Carlo 实验：生成 N 个 2×2 全零图块，3 通道
    N = 200_000
    device = torch.device("cpu")  # 强制使用 CPU，避免 NCCL 问题
    zeros = torch.zeros(N, 3, 2, 2, device=device)

    # 5) 调用你的加噪实现
    noised = QuaternionWaveletNoise.apply_noise(
        zeros,
        sigma=sigma_total,
        filter_name="haar",
        levels=1,
        ratio=ratio,
        device=device
    ).cpu().numpy()

    # 6) 计算经验方差
    #    arr.shape = (N, 3, 2, 2)
    #    对 N、H、W 三轴求方差后再对 3 个通道平均
    emp_var_chan = noised.var(axis=(0,2,3)).mean()
    emp_var_gray = emp_var_chan * 3.0

    print(f"[Verify] 实验 gray-pixel 方差 = {emp_var_gray:.6f}")
    print(f"[Verify] 实验 per-channel 方差 = {emp_var_chan:.6f}")

    # 7) 相对误差
    err_gray = abs(emp_var_gray - var_theo_gray) / var_theo_gray
    err_chan = abs(emp_var_chan - var_theo_chan) / var_theo_chan
    print(f"[Verify] gray-pixel 相对误差 = {err_gray:.4%}")
    print(f"[Verify] per-channel 相对误差 = {err_chan:.4%}")

    # 8) 断言：误差必须在 1% 内
    assert err_gray < 0.01, f"灰度像素方差误差过大: {err_gray:.2%}"
    assert err_chan < 0.01, f"通道方差误差过大: {err_chan:.2%}"

if __name__ == "__main__":
    main()
