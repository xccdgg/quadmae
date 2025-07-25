import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.5.4"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_rcot

from engine_pretrain import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('RCOT pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='rcot_dmae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--loss1_weight', default=1.0, type=float,
                        help='Weight for stage1 reconstruction loss')
    parser.add_argument('--loss2_weight', default=1.0, type=float,
                        help='Weight for stage2 refinement loss')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--sigma', default=0.5, type=float,
                        help='Std of Gaussian noise')
    parser.add_argument(
        '--use_quaternion_noise',
        type=lambda x: str(x).lower() in ('true', '1', 'yes'),
        default=False,
        help='Use quaternion wavelet noise instead of pixel Gaussian',
    )
    parser.add_argument('--levels', default=1, type=int,
                        help='Levels of QWT decomposition for noise')
    parser.add_argument('--ratio', default=3.0, type=float,
                        help='Sigma_H / Sigma_L ratio for QWT noise')

    parser.add_argument(
        '--freeze_base',
        dest='freeze_base',
        action='store_true',
        help='Freeze base model (encoder & decoder1) during training (default: True)'
    )
    parser.add_argument(
        '--unfreeze_base',
        dest='freeze_base',
        action='store_false',
        help='Unfreeze base model for fine-tuning (train all layers)'
    )
    parser.set_defaults(freeze_base=False)
    parser.add_argument('--dmae_ckpt', default='',
                        help='path to pretrained DMAE checkpoint')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/root/autodl-tmp/data/cifar-10-batches-py', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='autodl-tmp/RCOTDMAE/cifar10/sigma0.25rcotdmae',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='autodl-tmp/RCOTDMAE/cifar10/sigma0.25rcotdmae',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--only_model', action='store_true',
                        help='only load model weights when resuming')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    # in order to add noise, the normalization is done in the dmae model
    train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    dataset_train = datasets.CIFAR10(root=args.data_path,
                                      train=True,
                                      download=True,
                                      transform=train_transform)
    print(dataset_train)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_val = datasets.CIFAR10(
        root=args.data_path,
        train=False,
        download=True,
        transform=val_transform,
    )
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # define the model
    model = models_rcot.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        sigma=args.sigma,
        freeze_base=args.freeze_base,
        dmae_ckpt=args.dmae_ckpt if args.dmae_ckpt else None,
        use_quaternion_noise=args.use_quaternion_noise,
        levels=args.levels,
        ratio=args.ratio,
    )
    model.mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    model.std = torch.tensor([0.2471, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print(
        f"[Config] mask_ratio={args.mask_ratio}, loss1_weight={args.loss1_weight}, "
        f"loss2_weight={args.loss2_weight}, freeze_base={args.freeze_base}"
    )

    if args.freeze_base:
        print("Encoder & Decoder1 are frozen (requires_grad=False).")
        print("BatchNorm layers in encoder/decoder1 set to eval mode.")
        print("Residual Encoder & Decoder2 are trainable (requires_grad=True).")
    else:
        print("Encoder & Decoder1 are not frozen (trainable).")
        print("BatchNorm layers in encoder/decoder1 are in training mode.")
        print("All model components are trainable.")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    new_lr = args.lr * 5
    base_params = []
    new_params = []
    for name, p in model_without_ddp.named_parameters():
        if name.startswith('base.'):
            base_params.append(p)
        else:
            new_params.append(p)
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': new_params, 'lr': new_lr, 'weight_decay': args.weight_decay},
    ], betas=(0.9, 0.95))
    print(f"[Opt] base_lr={args.lr:.2e}, new_module_lr={new_lr:.2e}")
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        load_optimizer=not args.only_model,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_val_loss = float("inf")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        val_stats = evaluate(model, data_loader_val, device, args)
        val_loss = val_stats.get("loss", None)

        if val_loss is not None and val_loss < best_val_loss and args.output_dir:
            best_val_loss = val_loss
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)