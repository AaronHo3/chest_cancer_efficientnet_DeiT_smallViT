"""
PyTorch Dataset and DataLoaders for chest CT 4-class classification.
Maps folder names (long and short) to class indices; uses ImageNet normalization.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ImageNet normalization for pretrained models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Class index order: adenocarcinoma, large_cell_carcinoma, squamous_cell_carcinoma, normal
CLASS_NAMES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "squamous_cell_carcinoma",
    "normal",
]


def folder_name_to_class_index(folder_name: str) -> Optional[int]:
    """
    Map dataset folder name to class index (0-3).
    Handles long names (e.g. adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib) and short (e.g. adenocarcinoma).
    Returns None if folder does not match any known class.
    """
    name = folder_name.lower().replace(" ", ".")
    # Order matters: check adenocarcinoma first, then squamous, then large.cell, then normal
    if "adenocarcinoma" in name:
        return 0
    if "squamous" in name:
        return 2
    if "large" in name and "cell" in name:
        return 1
    if "normal" in name:
        return 3
    return None


def get_transforms(image_size: int = 224, train: bool = False):
    """Build transforms with optional augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CTScanDataset(Dataset):
    """Dataset of chest CT images from Data/Train, Data/Valid, or Data/Test."""

    def __init__(self, data_root: str, split: str, transform=None):
        """
        data_root: path to dataset root (contains Data/Train, Data/Valid, Data/Test).
        split: 'Train', 'Valid', or 'Test'.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.samples = []  # list of (path, class_index)

        split_dir = self.data_root / "Data" / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for folder_name in sorted(split_dir.iterdir()):
            if not folder_name.is_dir():
                continue
            class_idx = folder_name_to_class_index(folder_name.name)
            if class_idx is None:
                continue
            for f in folder_name.iterdir():
                if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    self.samples.append((str(f), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Return train, valid, and test DataLoaders.
    Train uses augmentation; Valid and Test use same deterministic resize + center crop.
    """
    train_ds = CTScanDataset(
        data_root, "Train",
        transform=get_transforms(image_size, train=True),
    )
    valid_ds = CTScanDataset(
        data_root, "Valid",
        transform=get_transforms(image_size, train=False),
    )
    test_ds = CTScanDataset(
        data_root, "Test",
        transform=get_transforms(image_size, train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader, test_loader
