import torch
import numpy as np
import argparse
import os
from pathlib import Path
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import csv

import util.misc as misc
from util.smooth import Smooth
from util.noise import fixed_sigma_value

import models_vit
import models_rcot

from engine_finetune import certify_evaluate_dist
from torchvision import transforms, datasets


def get_args_parser():
    parser = argparse.ArgumentParser('RCOT certification on CIFAR-10', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--eval', action='store_true', default=True,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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
    
    # certified parameters
    parser.add_argument('--sigma', default=0.25,
                        help='fixed standard deviation for randomized smoothing')
    parser.add_argument('--sample_interval', default=1, type=int,
                        help="the interval of sampling during test")
    parser.add_argument('--num', default=1000, type=int,
                        help="the samples for evaluate radius")
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
    parser.add_argument('--use_qwt_prior_adapter', action='store_true',
                        help='Enable deterministic QWT prior adapter')
    parser.add_argument('--qwt_prior_levels', default=1, type=int,
                        help='Levels for deterministic QWT prior extractor (v1 only supports 1)')
    parser.add_argument('--subband_loss_weight', default=0.1, type=float,
                        help='Reserved for interface parity; only used during pretraining')
    parser.add_argument('--subband_loss_detail_weight', default=0.5, type=float,
                        help='Reserved for interface parity; only used during pretraining')

    parser.add_argument('--use_rcot', action='store_true',
                        help='Apply RCOT restoration before certification')
    parser.add_argument('--rcot_ckpt', default='', type=str,
                        help='path to RCOT checkpoint')
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
    try:
        args.sigma = fixed_sigma_value(args.sigma)
    except ValueError as exc:
        raise ValueError(
            "CIFAR-10 certification requires a fixed evaluate sigma. "
            "Use --sigma 0.25 or --sigma 0.5; do not use --sigma \"[0,0.75]\" here."
        ) from exc

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    dataset_val = datasets.CIFAR10(root=args.data_path,
                                    train=False,
                                    download=True,
                                    transform=transforms.ToTensor())
    dataset_val = DatasetWithInterval(dataset_val, args.sample_interval)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print('len(sampler_val)', len(sampler_val))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    rcot_model = False
    cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    cifar_std = torch.tensor([0.2471, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    if args.model in models_vit.__dict__:
        model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            use_qwt_prior_adapter=args.use_qwt_prior_adapter,
            qwt_prior_levels=args.qwt_prior_levels,
        )
        model.mean = cifar_mean.clone()
        model.std = cifar_std.clone()
    elif hasattr(models_rcot, args.model):
        rcot_model = True
        ctor = getattr(models_rcot, args.model)
        model = ctor(use_head=True, num_classes=args.nb_classes)
        if hasattr(model, 'mean'):
            model.mean = cifar_mean.clone()
            model.std = cifar_std.clone()
        if hasattr(model, 'base'):
            model.base.mean = cifar_mean.clone()
            model.base.std = cifar_std.clone()
    else:
        raise KeyError(f"Unknown model architecture: {args.model}")
    if rcot_model and hasattr(model, 'use_head'):
        head_module = getattr(model, 'head', None)
        if head_module is None or not hasattr(head_module, 'in_features'):
            raise RuntimeError('Unable to determine RCOT classification head input dimension.')
        if getattr(head_module, 'out_features', None) != args.nb_classes:
            model.head = torch.nn.Linear(head_module.in_features, args.nb_classes)
        model.use_head = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None)

    restorer = None
    if args.use_rcot:
        restorer = models_rcot.rcot_dmae_vit_base_patch16(use_head=args.use_head, num_classes=args.nb_classes)
        if args.rcot_ckpt:
            ckpt = torch.load(args.rcot_ckpt, map_location="cpu")
            restorer.load_state_dict(ckpt.get("model", ckpt), strict=False)
        restorer.to(device)
        restorer.eval()

    class _LogitsOnlyModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x):
            out = self.module(x)
            return out[-1] if isinstance(out, tuple) else out


    # switch to evaluation mode
    model.eval()
    threshold=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
    classifier_for_smooth = model
    if rcot_model:
        classifier_for_smooth = _LogitsOnlyModule(model)

    if args.sigma:
        smoothed_classifier = Smooth(
            classifier_for_smooth,
            num_classes,
            args.sigma,
            use_quaternion_noise=args.use_quaternion_noise,
            levels=args.levels,
            ratio=args.ratio,
        )
        test_stats = certify_evaluate_dist(
            data_loader_val,
            smoothed_classifier,
            device,
            threshold,
            args.num,
            restorer=restorer,
            use_rcot=args.use_rcot,
        )
        print('* Load model from {}'.format(args.resume))
        print('* Interval of sampling: {}(number of datapoints: {})'.format(args.sample_interval, len(dataset_val)))
        print('* Randomized smoothing with sigma {}'.format(args.sigma))
        print('* Certiﬁed test accuracy:')
        for thres in threshold:
            print('* Acc@r={radius:.2f} {acc:.3f}'.format(
                radius=thres, 
                acc=test_stats['Acc@r={radius:.2f}'.format(radius=thres)]))
    else: # test on sigma = (0.25, 0.5, 1.0)
        for sigma in [0.25, 0.5, 1.0]:
            smoothed_classifier = Smooth(
                classifier_for_smooth,
                num_classes,
                sigma,
                use_quaternion_noise=args.use_quaternion_noise,
                levels=args.levels,
                ratio=args.ratio,
            )
            test_stats = certify_evaluate_dist(
                data_loader_val,
                smoothed_classifier,
                device,
                threshold,
                args.num,
                restorer=restorer,
                use_rcot=args.use_rcot,
            )
            print('* Load model from {}'.format(args.resume))
            print('* Interval of sampling: {}(number of datapoints: {})'.format(args.sample_interval, len(dataset_val)))
            print('* Randomized smoothing with sigma {}'.format(sigma))
            print('* Certiﬁed test accuracy:')
            for thres in threshold:
                print('* Acc@r={radius:.2f} {acc:.3f}'.format(
                    radius=thres, 
                    acc=test_stats['Acc@r={radius:.2f}'.format(radius=thres)]))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "certified_accuracy.csv")
        print(f"[DEBUG] Writing certified results to {csv_path}")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["radius", "certified_acc_percent"])
            for r in threshold:
                key = f"Acc@r={r:.2f}"
                val = test_stats[key] * 100
                writer.writerow([r, val])
        print("✅ Certified accuracies saved to", csv_path)

    exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    num_classes = 10
    main(args)
