import math
import sys
import os
from typing import Iterable

import torch
import torchvision

import util.misc as misc
import util.lr_sched as lr_sched

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _unnorm(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo per-channel normalization and clamp into [0, 1] for display."""
    return (img * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def _save_visual_debug(
    samples: torch.Tensor,
    x_hat: torch.Tensor,
    r: torch.Tensor,
    x_refined: torch.Tensor,
    debug_dir: str,
    model,
) -> None:
    """Save input, reconstructions and residual for quick visual inspection.

    * ``samples``      – original clean images in **[0, 1]**.
    * ``x_hat``        – stage-1 reconstruction (normalized).
    * ``r``            – residual *in normalized space*.
    * ``x_refined``    – stage-2 refined reconstruction (normalized).

    The function automatically un-normalizes tensors that live in the DMAE
    pixel space and linearly maps the residual into [0, 1] for visualization.
    """
    os.makedirs(debug_dir, exist_ok=True)

    # 1. original images (already in pixel space)
    torchvision.utils.save_image(samples.clamp(0, 1), f"{debug_dir}/x.png")

    # 2. fetch mean / std buffers from the underlying DMAE
    try:
        mean = model.base.mean  # type: ignore[attr-defined]
        std = model.base.std    # type: ignore[attr-defined]
    except AttributeError:
        mean = getattr(model, "mean", None)
        std = getattr(model, "std", None)
    if mean is None or std is None:
        raise AttributeError("Cannot locate mean/std buffers inside the model – required for visualisation.")

    mean, std = mean.to(x_hat.device), std.to(x_hat.device)

    # 3. stage-1 & stage-2 outputs (convert back to pixel space)
    torchvision.utils.save_image(_unnorm(x_hat, mean, std), f"{debug_dir}/x_hat.png")
    torchvision.utils.save_image(_unnorm(x_refined, mean, std), f"{debug_dir}/x_refined.png")

    # 4. residual – simple min-max scaling to [0, 1]
    r_min, r_max = r.min(), r.max()
    r_vis = (r - r_min) / (r_max - r_min + 1e-8)
    torchvision.utils.save_image(r_vis, f"{debug_dir}/r.png")


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    """RCOT pre-training for **one** epoch with optional visual/debug dumps.

    ``debug`` 操作会在 ``epoch % 20 == 0`` 时触发，避免日志过大。其余训练
    逻辑（学习率调度、梯度累积、TensorBoard 记录等）保持不变。
    """

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Debug config – only active every 20 epochs
    # ------------------------------------------------------------------
    do_debug = (epoch % 20 == 0)
    debug_fp = None
    if do_debug and misc.is_main_process():
        debug_dir = os.path.join(args.output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_fp = open(os.path.join(debug_dir, "debug.log"), "a")
    # ------------------------------------------------------------------

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # learning-rate schedule (per iteration)
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)

        # torch>=1.12 changed autocast namespace
        try:
            autocast_fn = torch.autocast  # type: ignore[attr-defined]
        except AttributeError:
            autocast_fn = torch.cuda.amp.autocast  # pragma: no cover

        with autocast_fn("cuda"):
            x_hat, r, x_refined, loss1, loss2 = model(
                samples, mask_ratio=args.mask_ratio
            )
            loss = loss1 * args.loss1_weight + loss2 * args.loss2_weight

        # visualise & write debug once per epoch when allowed
        if do_debug and data_iter_step == 0 and misc.is_main_process():
            _save_visual_debug(samples, x_hat, r, x_refined, debug_dir, model)

        # debug text (only when enabled)
        if do_debug and debug_fp is not None:
            debug_fp.write(
                f"[Debug] Epoch {epoch}, Batch {data_iter_step}: "
                f"loss1={loss1.item():.4f}, loss2={loss2.item():.4f}\n"
            )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # ---- logging -------------------------------------------------------------------
        metric_logger.update(loss=loss_value)
        lr_current = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr_current)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr_current, epoch_1000x)

    # ---- end-of-epoch housekeeping ----------------------------------------------------
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if debug_fp is not None:
        debug_fp.close()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
