import argparse
from pathlib import Path
import random


def get_args_parser():
    parser = argparse.ArgumentParser("Generate a fixed ImageNet class subset")
    parser.add_argument("--data_path", required=True, type=str, help="ImageNet root containing train/")
    parser.add_argument("--output", required=True, type=str, help="Output txt file, one wnid per line")
    parser.add_argument("--num_classes", default=200, type=int, help="Number of classes to sample")
    parser.add_argument("--seed", default=20260323, type=int, help="Sampling seed")
    return parser


def main(args):
    train_root = Path(args.data_path) / "train"
    classes = sorted(entry.name for entry in train_root.iterdir() if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No ImageNet class directories found under {train_root}")
    if args.num_classes > len(classes):
        raise ValueError(f"Requested {args.num_classes} classes, but only found {len(classes)}")

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(classes, args.num_classes))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"Wrote {len(selected)} classes to {output_path}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
