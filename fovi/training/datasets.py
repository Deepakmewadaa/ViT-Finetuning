from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


class SyntheticImageDataset(Dataset):
    """Synthetic dataset for testing and benchmarking without requiring full ImageNet downloads."""

    def __init__(
        self,
        num_samples: int = 128,
        image_size: int = 256,
        num_classes: int = 100,
        channels: int = 3,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.channels = channels

        # Pre-generate deterministic images
        torch.manual_seed(42)
        self.data = torch.randn(num_samples, channels, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], int(self.labels[idx].item())


def build_cifar_dataset(
    root: str = "./data",
    is_cifar100: bool = True,
    train: bool = True,
    download: bool = True,
    transform: Optional[Callable] = None,
) -> Dataset:
    """Build standard torchvision CIFAR-10 / CIFAR-100 dataset."""
    try:
        from torchvision.datasets import CIFAR10, CIFAR100
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required to load CIFAR datasets.")

    if transform is None:
        if train:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    cls = CIFAR100 if is_cifar100 else CIFAR10
    return cls(root=root, train=train, download=download, transform=transform)


def build_image_folder_dataset(
    root_dir: str,
    train: bool = True,
    image_size: int = 256,
) -> Dataset:
    """Build dataset from standard ImageNet-100 or ImageNet-1K folder structure (train/val subdirectories)."""
    try:
        from torchvision.datasets import ImageFolder
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required for ImageFolder datasets.")

    split_dir = os.path.join(root_dir, "train" if train else "val")
    if not os.path.exists(split_dir):
        split_dir = root_dir

    if train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    return ImageFolder(root=split_dir, transform=transform)
