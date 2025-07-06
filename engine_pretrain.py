import math
import sys
import os
from typing import Iterable

import torch
import torchvision

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    debug_dir = os.path.join(args.output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        print(f"[Debug] Batch {data_iter_step}: imgs range [{samples.min():.3f}, {samples.max():.3f}]")

        try:
            autocast = torch.autocast
        except AttributeError:
            autocast = torch.cuda.amp.autocast
        with autocast("cuda"):
            x_hat, r, x_refined, loss1, loss2 = model(samples, mask_ratio=args.mask_ratio)
            loss = loss1 * args.loss1_weight + loss2 * args.loss2_weight
        if data_iter_step == 0:
            torchvision.utils.save_image(samples,     f"{debug_dir}/x.png")
            torchvision.utils.save_image(x_hat,        f"{debug_dir}/x_hat.png")
            torchvision.utils.save_image(r,           f"{debug_dir}/r.png")
            torchvision.utils.save_image(x_refined,    f"{debug_dir}/x_refined.png")
        print(f"[Debug] Batch {data_iter_step}: loss1={loss1.item():.4f}, loss2={loss2.item():.4f}")

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for name, p in model.res_encoder.named_parameters():
                if p.grad is not None:
                    print(f"[Debug] res_encoder.{name} grad_norm={p.grad.norm():.4e}")
            for name, p in model.decoder2.named_parameters():
                if p.grad is not None:
                    print(f"[Debug] decoder2.{name} grad_norm={p.grad.norm():.4e}")
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}