"""
Dataset module matching FOVI paper specifications:
- Preserves full image field-of-view (resizing to 224x224 or 256x256 rather than aggressive cropping,
  since foveation and saccadic fixations provide the multi-scale focus).
- Stratified 80% train (40k), 10% val (5k), 10% test (5k) splits.
- Standard ImageNet normalization.
"""

import os
import json
import random
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DEFAULT_DATA_DIR = r"C:\Users\deepa\Downloads\train\train"
SPLIT_CACHE_PATH = r"d:\Research\dataset_splits.json"

# Standard ImageNet normalization for ViTs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_or_load_splits(
    data_dir: str = DEFAULT_DATA_DIR,
    cache_path: str = SPLIT_CACHE_PATH,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, List[Dict[str, str]]]:
    """
    Creates or loads a stratified train/val/test split.
    Guarantees equal class distribution and zero data leakage.
    """
    if os.path.exists(cache_path):
        print(f"[Dataset] Loading existing split metadata from {cache_path}")
        with open(cache_path, "r") as f:
            splits = json.load(f)
        return splits

    print(f"[Dataset] Scanning directory: {data_dir} ...")
    data_path = Path(data_dir)
    assert data_path.is_dir(), f"Data directory '{data_dir}' does not exist!"

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    assert len(class_dirs) > 0, f"No class directories found in '{data_dir}'"
    print(f"[Dataset] Found {len(class_dirs)} classes.")

    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

    rng = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []

    valid_extensions = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        images = [
            str(p) for p in class_dir.iterdir()
            if p.is_file() and p.suffix in valid_extensions
        ]
        images.sort()
        rng.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            train_samples.append({"path": img, "label": class_idx, "class_name": class_name})
        for img in val_imgs:
            val_samples.append({"path": img, "label": class_idx, "class_name": class_name})
        for img in test_imgs:
            test_samples.append({"path": img, "label": class_idx, "class_name": class_name})

    splits = {
        "num_classes": len(class_dirs),
        "class_to_idx": class_to_idx,
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"[Dataset] Created stratified splits -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    return splits


class ImageListDataset(Dataset):
    """PyTorch Dataset that loads images on the fly from a sample manifest."""
    def __init__(self, samples: List[Dict[str, str]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        image_path = item["path"]
        label = item["label"]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, label


def get_transforms(img_size: int = 224):
    """
    Paper-aligned transforms:
    - Resize to full FOV (224x224) + Random Horizontal Flip
    - Avoids artificial aggressive cropping as the foveated saccades handle local focus.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = 64,
    num_workers: int = 2,
    img_size: int = 224,
    cache_path: str = SPLIT_CACHE_PATH
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    splits = create_or_load_splits(data_dir=data_dir, cache_path=cache_path)
    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = ImageListDataset(splits["train"], transform=train_transform)
    val_dataset = ImageListDataset(splits["val"], transform=eval_transform)
    test_dataset = ImageListDataset(splits["test"], transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, splits["num_classes"]
