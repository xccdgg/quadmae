from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import PIL
from PIL import Image  # noqa: F401
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from util.noise import (
    add_noise,
    format_sigma_spec,
    parse_sigma_spec,
    sample_sigma,
    sigma_spec_max,
    sigma_total_from_pixel,
    to_bool as _to_bool,
)


class AddNoise:
    def __init__(
        self,
        sigma: Any,
        *,
        use_quaternion_noise: bool = True,
        levels: int = 1,
        ratio: float = 3.0,
        device: Any = "cpu",
    ) -> None:
        self.sigma_spec = parse_sigma_spec(sigma)
        self.sigma_label = format_sigma_spec(self.sigma_spec)
        self.sigma_pix = sigma_spec_max(self.sigma_spec)
        self.use_qwt = _to_bool(use_quaternion_noise)
        self.levels = int(levels)
        self.ratio = float(ratio)
        self.device = torch.device(device)
        self.last_sigma_pix: float | None = None

        self.sigma_total = sigma_total_from_pixel(self.sigma_pix, self.ratio)
        self.sigma_low = self.sigma_total / (1.0 + self.ratio)
        self.sigma_high = self.sigma_low * self.ratio

        print(
            f"[AddNoise] sigma_pix={self.sigma_label} max_sigma_total={self.sigma_total:.4f} "
            f"max_sigma_L={self.sigma_low:.4f} max_sigma_H={self.sigma_high:.4f} ratio={self.ratio}"
        )

    def __call__(self, img: Tensor) -> Tensor:
        if not torch.is_tensor(img):
            arr = np.asarray(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)

        sigma_pix = sample_sigma(self.sigma_spec, img.device)
        self.last_sigma_pix = sigma_pix
        if sigma_pix <= 0:
            return img

        return add_noise(
            img,
            sigma_pix,
            use_quaternion_noise=self.use_qwt,
            levels=self.levels,
            ratio=self.ratio,
            device=self.device,
        )


class NoisyImageDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        train: bool,
        input_size: int,
        batch_size: int,
        *,
        use_quaternion_noise: bool = True,
        noise_sigma: float = 0.0,
        levels: int = 1,
        ratio: float = 3.0,
    ) -> None:
        self.batch_size = batch_size

        transform = [transforms.ToTensor()]
        if noise_sigma and sigma_spec_max(noise_sigma) > 0:
            transform.append(
                AddNoise(
                    sigma=noise_sigma,
                    use_quaternion_noise=_to_bool(use_quaternion_noise),
                    levels=levels,
                    ratio=ratio,
                )
            )
        transform.append(transforms.Resize((input_size, input_size)))
        self.transform = transforms.Compose(transform)

        self.dataset = CIFAR10(
            root=data_root,
            train=train,
            transform=self.transform,
            download=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        return self.dataset[idx]

    def get_dataloader(self, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


class DatasetWithInterval(Dataset):
    def __init__(self, dataset: Dataset, interval: int) -> None:
        self.dataset = dataset
        self.interval = max(1, int(interval))

    def __getitem__(self, index: int):
        return self.dataset[index * self.interval]

    def __len__(self) -> int:
        return len(self.dataset) // self.interval


def load_class_subset_file(path: str | None) -> list[str] | None:
    if not path:
        return None

    with open(path, "r", encoding="utf-8") as handle:
        classes = [line.strip() for line in handle if line.strip()]

    if not classes:
        raise ValueError(f"class_subset_file is empty: {path}")
    if len(classes) != len(set(classes)):
        raise ValueError(f"class_subset_file contains duplicate wnids: {path}")
    return classes


def _is_image_file(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def _looks_like_flat_imagenet_val(root: str) -> bool:
    if not os.path.isdir(root):
        return False

    has_subdir = False
    has_image = False
    for entry in os.scandir(root):
        if entry.is_dir():
            has_subdir = True
            break
        if entry.is_file() and _is_image_file(entry.name):
            has_image = True
    return has_image and not has_subdir


def _has_direct_images(root: str) -> bool:
    if not os.path.isdir(root):
        return False
    return any(
        entry.is_file() and _is_image_file(entry.name)
        for entry in os.scandir(root)
    )


def _resolve_existing_path(explicit_path: str | None, *candidates: str) -> str:
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Path does not exist: {explicit_path}")

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError("No valid path found in candidates")


def _load_imagenet_idx_to_wnid(meta_path: str) -> dict[int, str]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "scipy is required to parse ImageNet meta.mat when val/ is a flat directory"
        ) from exc

    def _scalar(value):
        while isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        return value

    synsets = loadmat(meta_path, squeeze_me=True)["synsets"]
    idx_to_wnid: dict[int, str] = {}

    for entry in synsets:
        if hasattr(entry, "tolist"):
            entry = entry.tolist()
        if not isinstance(entry, (list, tuple)) or len(entry) < 5:
            continue

        ilsvrc_id = int(_scalar(entry[0]))
        wnid = str(_scalar(entry[1]))
        num_children = int(_scalar(entry[4]))

        if num_children == 0:
            idx_to_wnid[ilsvrc_id] = wnid

    if not idx_to_wnid:
        raise RuntimeError(f"Failed to parse ImageNet wnid mapping from {meta_path}")

    return idx_to_wnid


def _find_imagenet_train_classes(
    train_root: str,
    class_subset: list[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(train_root) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class subdirectories found under {train_root}")
    if class_subset is not None:
        missing = [cls_name for cls_name in class_subset if cls_name not in classes]
        if missing:
            raise KeyError(f"Subset classes missing under train root {train_root}: {missing[:8]}")
        classes = list(class_subset)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


class FilteredImageFolder(ImageFolder):
    def __init__(self, root: str, transform=None, class_subset: list[str] | None = None):
        super().__init__(root=root, transform=transform)

        if class_subset is None:
            return

        subset_classes = list(class_subset)
        subset_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(subset_classes)}
        missing = [cls_name for cls_name in subset_classes if cls_name not in self.class_to_idx]
        if missing:
            raise KeyError(f"Subset classes missing under {root}: {missing[:8]}")

        filtered_samples = []
        filtered_targets = []
        for path, old_target in self.samples:
            cls_name = self.classes[old_target]
            if cls_name not in subset_class_to_idx:
                continue
            new_target = subset_class_to_idx[cls_name]
            filtered_samples.append((path, new_target))
            filtered_targets.append(new_target)

        if not filtered_samples:
            raise RuntimeError(f"No samples remain after filtering {root} with the provided class subset")

        self.classes = subset_classes
        self.class_to_idx = subset_class_to_idx
        self.samples = filtered_samples
        self.targets = filtered_targets
        self.imgs = self.samples


class FlatImageNetValDataset(Dataset):
    def __init__(
        self,
        root: str,
        train_root: str,
        transform=None,
        *,
        ground_truth_path: str,
        meta_path: str,
        class_subset: list[str] | None = None,
    ):
        self.root = root
        self.transform = transform
        self.loader = Image.open

        self.classes, self.class_to_idx = _find_imagenet_train_classes(train_root, class_subset)
        full_train_classes, _ = _find_imagenet_train_classes(train_root, None)
        idx_to_wnid = _load_imagenet_idx_to_wnid(meta_path)

        images = sorted(
            os.path.join(root, name)
            for name in os.listdir(root)
            if os.path.isfile(os.path.join(root, name)) and _is_image_file(name)
        )
        if not images:
            raise FileNotFoundError(f"No validation images found under {root}")

        with open(ground_truth_path, "r", encoding="utf-8") as handle:
            raw_lines = [line.strip() for line in handle if line.strip()]
        if not raw_lines:
            raise RuntimeError(f"No validation labels found in {ground_truth_path}")

        first_parts = raw_lines[0].split()
        if len(first_parts) == 1:
            val_label_ids = [int(line) for line in raw_lines]
            if len(images) != len(val_label_ids):
                raise RuntimeError(
                    f"Validation image count ({len(images)}) does not match ground-truth count ({len(val_label_ids)})"
                )
            image_label_pairs = list(zip(images, val_label_ids))
        else:
            image_by_name = {os.path.basename(path): path for path in images}
            image_label_pairs = []
            for line in raw_lines:
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid validation label line in {ground_truth_path}: {line!r}")
                image_name = parts[0]
                label_id = int(parts[-1])
                image_path = image_by_name.get(image_name)
                if image_path is None:
                    candidate = os.path.join(root, image_name)
                    if not os.path.isfile(candidate):
                        raise FileNotFoundError(
                            f"Validation image {image_name!r} from {ground_truth_path} is missing under {root}"
                        )
                    image_path = candidate
                image_label_pairs.append((image_path, label_id))

        # Some ImageNet val label files use 0-based class indices aligned with the
        # alphabetically sorted train/ wnid folders instead of official 1-based
        # ILSVRC ids from devkit/meta.mat. Detect that case conservatively.
        uses_zero_based_class_index = any(label_id == 0 for _, label_id in image_label_pairs)

        self.samples = []
        self.targets = []
        for image_path, label_id in image_label_pairs:
            if uses_zero_based_class_index:
                if 0 <= label_id < len(full_train_classes):
                    wnid = full_train_classes[label_id]
                else:
                    raise KeyError(
                        f"0-based ImageNet class index {label_id} is out of range for train root {train_root}"
                    )
            else:
                wnid = idx_to_wnid.get(label_id)
                if wnid is None and (label_id + 1) in idx_to_wnid:
                    # Some non-official label files store 0-based devkit ids.
                    wnid = idx_to_wnid[label_id + 1]
                if wnid is None:
                    raise KeyError(f"ILSVRC label id {label_id} is missing from {meta_path}")
            if wnid not in self.class_to_idx:
                if class_subset is not None:
                    continue
                raise KeyError(f"WNID {wnid} from {meta_path} is missing under train root {train_root}")
            target = self.class_to_idx[wnid]
            self.samples.append((image_path, target))
            self.targets.append(target)

        if not self.samples:
            raise RuntimeError(f"No validation samples remain after filtering {root} with the provided class subset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with self.loader(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _build_imagenet_transform(split: str, args):
    is_train = split.lower() == "train"

    if is_train:
        primary_tfl, secondary_tfl, _ = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa if args.aa != "None" else None,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            separate=True,
        )
        return transforms.Compose(
            [
                primary_tfl,
                secondary_tfl,
                transforms.ToTensor(),
            ]
        )

    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)

    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
        ]
    )


def build_dataset(split: str, args):
    is_train = split.lower() == "train"
    class_subset = load_class_subset_file(getattr(args, "class_subset_file", ""))
    if class_subset is not None and hasattr(args, "nb_classes"):
        if int(args.nb_classes) != len(class_subset):
            raise ValueError(
                f"nb_classes ({args.nb_classes}) must match class_subset_file size ({len(class_subset)})"
            )

    if getattr(args, "nb_classes", 10) == 1000 or class_subset is not None:
        root = os.path.join(args.data_path, "train" if is_train else "val")
        transform = _build_imagenet_transform(split, args)
        prefer_flat_val = (
            not is_train
            and (
                _looks_like_flat_imagenet_val(root)
                or (
                    _has_direct_images(root)
                    and (
                        bool(getattr(args, "imagenet_val_ground_truth", ""))
                        or bool(getattr(args, "imagenet_meta", ""))
                    )
                )
            )
        )
        if prefer_flat_val:
            data_root = Path(args.data_path)
            ground_truth_path = _resolve_existing_path(
                getattr(args, "imagenet_val_ground_truth", None),
                str(data_root / "ILSVRC2012_validation_ground_truth.txt"),
                str(data_root / "devkit" / "data" / "ILSVRC2012_validation_ground_truth.txt"),
                str(data_root / "ILSVRC2012_devkit_t12" / "data" / "ILSVRC2012_validation_ground_truth.txt"),
            )
            meta_path = _resolve_existing_path(
                getattr(args, "imagenet_meta", None),
                str(data_root / "meta.mat"),
                str(data_root / "devkit" / "data" / "meta.mat"),
                str(data_root / "ILSVRC2012_devkit_t12" / "data" / "meta.mat"),
            )
            return FlatImageNetValDataset(
                root=root,
                train_root=os.path.join(args.data_path, "train"),
                transform=transform,
                ground_truth_path=ground_truth_path,
                meta_path=meta_path,
                class_subset=class_subset,
            )
        return FilteredImageFolder(root=root, transform=transform, class_subset=class_subset)

    return NoisyImageDataset(
        data_root=args.data_path,
        train=is_train,
        input_size=args.input_size,
        batch_size=args.batch_size,
        use_quaternion_noise=_to_bool(args.use_quaternion_noise),
        noise_sigma=args.sigma,
        levels=args.levels,
        ratio=args.ratio,
    )


def build_dataset_with_interval(split: str, args):
    dataset = build_dataset(split, args)
    interval = getattr(args, "sample_interval", 1)
    if interval > 1 and split.lower() != "train":
        dataset = DatasetWithInterval(dataset, interval)
    return dataset
