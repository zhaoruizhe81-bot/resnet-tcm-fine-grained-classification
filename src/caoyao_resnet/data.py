from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_split_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if (root / "train").is_dir():
        return root
    if (root / "split_dataset" / "train").is_dir():
        return root / "split_dataset"
    raise FileNotFoundError(
        f"没有找到可用的数据集切分目录。请确认 {root} 或 {root / 'split_dataset'} 下存在 train/val/test。"
    )


def build_transforms(image_size: int) -> dict[str, transforms.Compose]:
    resize_size = max(256, int(image_size * 1.15))
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
        "eval": transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    }


def create_imagefolder(split_root: Path, split_name: str, image_size: int) -> datasets.ImageFolder:
    transforms_map = build_transforms(image_size)
    transform_key = "train" if split_name == "train" else "eval"
    dataset = datasets.ImageFolder(split_root / split_name, transform=transforms_map[transform_key])
    dataset.classes = [str(name) for name in dataset.classes]
    return dataset


def create_dataloaders(
    *,
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> tuple[dict[str, DataLoader], list[str], dict[str, int]]:
    split_root = resolve_split_root(data_root)
    train_dataset = create_imagefolder(split_root, "train", image_size)
    val_dataset = create_imagefolder(split_root, "val", image_size)

    datasets_map: dict[str, datasets.ImageFolder] = {
        "train": train_dataset,
        "val": val_dataset,
    }
    if (split_root / "test").is_dir():
        datasets_map["test"] = create_imagefolder(split_root, "test", image_size)

    pin_memory_enabled = pin_memory and device.type == "cuda"
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory_enabled,
        "persistent_workers": num_workers > 0,
    }
    dataloaders: dict[str, DataLoader] = {}
    for split_name, dataset in datasets_map.items():
        dataloaders[split_name] = DataLoader(
            dataset,
            shuffle=split_name == "train",
            **loader_kwargs,
        )

    dataset_sizes = {split_name: len(dataset) for split_name, dataset in datasets_map.items()}
    return dataloaders, train_dataset.classes, dataset_sizes
