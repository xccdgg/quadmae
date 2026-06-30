import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_dataset_with_interval
from util.noise import format_sigma_spec, parse_sigma_spec
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torchvision import datasets, transforms

import models_vit
import models_rcot  # 鏂板锛氬鍏COT妯″瀷

from engine_finetune import *


def _load_pretrained_with_smart_prefix(model: nn.Module, checkpoint_model: dict, args) -> None:
    """
    Load a checkpoint into `model` while adapting key prefixes intelligently:
    - Preserve `base.` prefix for RCOT/TwoStageDMAE models (model has `base.*`).
    - Add `base.` prefix for pure DMAE checkpoints when loading into RCOT models.
    - Strip `base.` when loading RCOT checkpoints into plain ViT models.
    - Interpolate position embedding using a temporary dict without changing prefixes for load.
    Also removes mismatched classification head weights and reports unexpected missing keys.
    """
    # Unwrap state_dict-like containers and ensure float tensors
    if isinstance(checkpoint_model, dict) and hasattr(checkpoint_model, "state_dict"):
        checkpoint_model = checkpoint_model.state_dict()

    ckpt_items = {}
    for k, v in checkpoint_model.items():
        if isinstance(v, torch.Tensor):
            v = v.float()
        # strip potential DistributedDataParallel prefix
        if k.startswith("module."):
            k = k[7:]
        ckpt_items[k] = v

    state_dict = model.state_dict()

    # Map keys to expected names in current model
    mapped = {}
    for k, v in ckpt_items.items():
        if k in state_dict:
            mapped[k] = v
        elif ("base." + k) in state_dict:
            mapped["base." + k] = v
        elif k.startswith("base.") and k[5:] in state_dict:
            mapped[k[5:]] = v
        # else: drop unknown keys

    # Interpolate pos_embed in a temporary dict (expects key name 'pos_embed')
    backbone = model.base if hasattr(model, "base") else model
    expected_pos_key = "base.pos_embed" if "base.pos_embed" in state_dict else ("pos_embed" if "pos_embed" in state_dict else None)
    if expected_pos_key and (expected_pos_key in mapped):
        tmp = {"pos_embed": mapped[expected_pos_key]}
        interpolate_pos_embed(backbone, tmp)
        mapped[expected_pos_key] = tmp["pos_embed"]

    # Remove mismatched classification head weights (shape differs due to num_classes)
    head_keys = [k for k in ("head.weight", "head.bias") if k in state_dict]
    for k in head_keys:
        if k in mapped and mapped[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del mapped[k]

    # Load
    msg = model.load_state_dict(mapped, strict=False)
    print(msg)

    # Ensure all params are float32
    for _, param in model.named_parameters():
        if isinstance(param, nn.Parameter) and param.dtype != torch.float32:
            param.data = param.data.float()

    # Report unexpected missing keys (allow classifier and optional fc_norm)
    allowed_missing = set(head_keys)
    if getattr(args, "use_qwt_prior_adapter", False):
        allowed_missing.update({
            'prior_gate',
            'prior_ln.weight',
            'prior_ln.bias',
            'prior_proj.weight',
            'prior_proj.bias',
        })
    # Allow entire RCOT classifier module to be absent from checkpoint (pretrain often has no classifier)
    if hasattr(model, 'classifier'):
        allowed_missing.update({k for k in state_dict.keys() if k.startswith('classifier.')})
    if getattr(args, "global_pool", False):
        allowed_missing.update({"fc_norm.weight", "fc_norm.bias"})
    unexpected_missing = set(msg.missing_keys) - allowed_missing
    if unexpected_missing:
        print("Unexpected missing keys (not allowed):")
        for key in sorted(unexpected_missing):
            print(f"  {key}")
    else:
        print("All missing keys are expected.")



def get_args_parser():
    parser = argparse.ArgumentParser('RCOT finetuning on CIFAR-10', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', default=False,
                        help='whether to train distributed')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # certified accuracy parameters
    parser.add_argument('--sigma', default=0.25, type=parse_sigma_spec,
                        help='Std of noise, or a training range like "[0,0.75]"')
    parser.add_argument('--use_quaternion_noise', default=True,
                        help='Use quaternion wavelet noise instead of pixel Gaussian')
    parser.add_argument('--levels', default=1, type=int,
                        help='Levels of QWT decomposition for noise')
    parser.add_argument('--ratio', default=3.0, type=float,
                        help='Sigma_H / Sigma_L ratio for QWT noise')
    parser.add_argument('--use_qwt_prior_adapter', action='store_true',
                        help='Enable deterministic QWT prior adapter')
    parser.add_argument('--qwt_prior_levels', default=1, type=int,
                        help='Levels for deterministic QWT prior extractor (v1 only supports 1)')
    parser.add_argument('--subband_loss_weight', default=0.1, type=float,
                        help='Reserved for interface parity; only used during pretraining')
    parser.add_argument('--subband_loss_detail_weight', default=0.5, type=float,
                        help='Reserved for interface parity; only used during pretraining')
    parser.add_argument('--sample_interval', default=50, type=int,
                        help="the interval of sampling during test")

    # consistency regularization parameters
    parser.add_argument('--con_reg', action='store_true', default=False,
                        help='enable consistency regularization')
    parser.add_argument('--num_noise_sample', default=2, type=int,
                        help='Number of Gaussian samples per input')
    parser.add_argument('--reg_lbd', default=2.0, type=float,
                        help='Weight of K-L divergence')
    parser.add_argument('--reg_eta', default=0.5, type=float,
                        help='Weight of entropy')
    parser.add_argument('--use_head', action='store_true', help='Enable classification head for RCOT-DMAE model')

    return parser


class DatasetWithInterval(torch.utils.data.Dataset):
    '''
    sampling data with interval from a given dataset
    '''

    def __init__(self, dataset, interval):
        self.dataset = dataset
        self.interval = interval

    def __getitem__(self, index):
        return self.dataset[index * self.interval]

    def __len__(self):
        return len(self.dataset) // self.interval


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

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    val_transform = transforms.ToTensor()

    dataset_train = datasets.CIFAR10(root=args.data_path,
                                     train=True,
                                     download=True,
                                     transform=train_transform)
    dataset_val = datasets.CIFAR10(root=args.data_path,
                                   train=False,
                                   download=True,
                                   transform=val_transform)

    dataset_certify = DatasetWithInterval(dataset_val, 10)

    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_certify = torch.utils.data.DistributedSampler(
                dataset_certify, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_certify = torch.utils.data.SequentialSampler(dataset_certify)
    else:
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_certify = torch.utils.data.SequentialSampler(dataset_certify)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
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
        drop_last=False
    )

    data_loader_certify = torch.utils.data.DataLoader(
        dataset_certify, sampler=sampler_certify,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 鏀寔閫氳繃鍙傛暟閫夋嫨RCOT-DMAE鑱斿悎缁撴瀯
    if args.model == 'rcot_dmae_vit_base_patch16':
        model = models_rcot.rcot_dmae_vit_base_patch16(use_head=args.use_head, num_classes=args.nb_classes)
    else:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            use_qwt_prior_adapter=args.use_qwt_prior_adapter,
            qwt_prior_levels=args.qwt_prior_levels,
        )

    model.mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    model.std = torch.tensor([0.2471, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.resume)
        checkpoint_model = checkpoint.get('model', checkpoint)
        _load_pretrained_with_smart_prefix(model, checkpoint_model, args)




    elif args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint.get('model', checkpoint)
        _load_pretrained_with_smart_prefix(model, checkpoint_model, args)

        head_module = getattr(model, 'head', None)
        if head_module is not None and hasattr(head_module, 'weight'):
            trunc_normal_(head_module.weight, std=2e-5)
            if head_module.bias is not None:
                nn.init.constant_(head_module.bias, 0.0)

    model = model.float().to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    print("randomized smoothing/training noise sigma: {}".format(format_sigma_spec(args.sigma)))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    # 鈥斺€斺€?鏂规 A锛歠inetune 鏃跺湪鍔犺浇瀹屾贩鍚堢簿搴?checkpoint 鍚庯紝閲嶆柊寮哄埗 cast 鏁翠釜妯″瀷涓?float32 鈥斺€斺€?
    if args.distributed:
        model.module.float()
    else:
        model.float()

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        test_stats = evaluate_radius_0(data_loader_val, model, device, args.sigma, stride=100)
        print(
            f"Accuracy on radius 0 of the network on the {len(dataset_val)} test images: {test_stats['acc1_r0']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_r0_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.con_reg:
            train_stats = train_one_epoch_con_reg(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )
        else:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if (epoch + 1) % 1 == 0:
            test_stats_r0 = evaluate_radius_0(data_loader_certify, model, device, args.sigma, stride=25,use_quaternion_noise=args.use_quaternion_noise,levels=args.levels,ratio=args.ratio,)
            print(
                f"Accuracy on radius 0 of the network on the {len(dataset_val)} test images: {test_stats_r0['acc1_r0']:.1f}%")
            if args.output_dir and misc.is_main_process() and test_stats_r0['acc1_r0'] > max_r0_accuracy:
                misc.save_named_model(
                    args=args, filename='best-r0.pth', epoch=epoch, model=model,
                    model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
            max_r0_accuracy = max(max_r0_accuracy, test_stats_r0['acc1_r0'])
            print(f'Max accuracy on radius 0: {max_r0_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.add_scalar('perf/acc1_r0', test_stats_r0['acc1_r0'], epoch)

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'sigma': format_sigma_spec(args.sigma)}

        if args.output_dir and misc.is_main_process():
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            misc.cleanup_epoch_checkpoints(args.output_dir, keep_epoch=epoch)
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
