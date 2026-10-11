#!/usr/bin/env python3
"""Estimate RS certification runtime using an actual forward-pass throughput benchmark.

This script does NOT evaluate certified accuracy and does NOT run official Dual RS.
It is for budgeting a single GPU (e.g., RTX 4090 24 GB) before scheduling
100,000,000+ model forward passes.

Usage (repo root, with project dependencies installed):
    python scripts/profile_single_gpu_certification.py --eval-images 1000 --n 1000 --batch-sizes 8 16 32
    python scripts/profile_single_gpu_certification.py --eval-images 10000 --n 10000 --batch-sizes 8 16 32

For each batch size, reports a LOWER BOUND for classifier forward passes:
input preprocessing, data movement, Gaussian sampling, class counting, IO,
confidence intervals, and Dual RS sigma-estimator passes are NOT included.
"""
import argparse
import time
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-images", type=int, default=1000)
    parser.add_argument("--n", type=int, default=1000,
                        help="Monte Carlo estimation draws PER image (not batch count)")
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--autocast", choices=["bf16", "fp16", "none"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to measure representative throughput.")
    if any(b <= 0 for b in args.batch_sizes) or args.eval_images <= 0 or args.n <= 0:
        raise ValueError("batch sizes, images and samples must be positive")

    import models_vit  # imported here so --help works without timm installed
    device = torch.device("cuda")
    model = models_vit.vit_base_patch16(
        num_classes=args.num_classes, global_pool=True).eval().to(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    params = sum(p.numel() for p in model.parameters())
    total_draws = args.eval_images * (args.n + args.n0)
    full_test_draws = 10000 * (10000 + args.n0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, memory={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB")
    print(f"ViT-B params={params/1e6:.1f}M, input={args.input_size}x{args.input_size}, dtype={args.autocast}")
    print(f"Requested samples: {args.eval_images:,} images x ({args.n:,}+{args.n0:,}) = {total_draws:,} draws")
    print("WARNING: quoted time is a classifier-forward-only LOWER BOUND; Dual RS needs additional estimator inference.")
    enabled = args.autocast != "none"
    dtype = torch.bfloat16 if args.autocast == "bf16" else torch.float16

    def forward(x):
        with torch.amp.autocast("cuda", enabled=enabled, dtype=dtype):
            out = model(x)
        return out

    for bs in args.batch_sizes:
        try:
            x = torch.rand(bs, 3, args.input_size, args.input_size, device=device)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(args.warmup):
                    forward(x)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.repeats):
                    forward(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            throughput = args.repeats * bs / elapsed
            elapsed_hours = total_draws / throughput / 3600
            full_hours = full_test_draws / throughput / 3600
            memory = torch.cuda.max_memory_allocated() / 2**30
            print(f"bs={bs:<3} {throughput:>8.1f} img/s | allocated_peak={memory:.2f}GiB | "
                  f"requested >= {elapsed_hours:.1f}h | full CIFAR-10 >= {full_hours:.1f}h")
            del x
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"bs={bs:<3} OOM; try a smaller batch size")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
