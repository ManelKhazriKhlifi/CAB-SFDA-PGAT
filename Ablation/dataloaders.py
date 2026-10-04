"""
dataloaders.py
Dataset and DataLoader construction, including the imbalance sampler (Table 13).
"""
import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import CFG

IMG_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms():
    src_train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tgt_weak_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tgt_strong_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return src_train_tf, tgt_weak_tf, tgt_strong_tf


class TwoViewTarget(Dataset):
    def __init__(self, base_ds, weak_tf, strong_tf):
        self.base = base_ds
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.weak_tf(img), self.strong_tf(img), -1


def make_imbalanced_sampler(dataset, target_ratio, num_classes, seed=42):
    """Sub-sample dataset to achieve max/min class ratio ≈ target_ratio."""
    labels = np.array([y for _, y in dataset.samples])
    class_counts = np.bincount(labels, minlength=num_classes)
    min_count = class_counts.min()
    max_allowed = int(min_count * target_ratio)

    weights = np.zeros(len(labels), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        if class_counts[c] > max_allowed:
            keep = rng.choice(idx_c, max_allowed, replace=False)
            weights[keep] = 1.0
        else:
            weights[idx_c] = 1.0
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=int(weights.sum()),
        replacement=False,
    )


def build_loaders(imbalance_ratio=None):
    src_train_tf, tgt_weak_tf, tgt_strong_tf = build_transforms()

    source_root = f"{CFG.DATA_ROOT}/data/source_aid_split"
    target_root = f"{CFG.DATA_ROOT}/data/target_clrs_split"

    src_train = torchvision.datasets.ImageFolder(f"{source_root}/train",
                                                 transform=src_train_tf)
    src_test = torchvision.datasets.ImageFolder(f"{source_root}/test",
                                                transform=src_train_tf)
    tgt_train_raw = torchvision.datasets.ImageFolder(f"{target_root}/train",
                                                     transform=None)
    tgt_test = torchvision.datasets.ImageFolder(
        f"{target_root}/test",
        transform=src_train_tf,
    )

    tgt_train = TwoViewTarget(tgt_train_raw, tgt_weak_tf, tgt_strong_tf)

    src_train_loader = DataLoader(src_train, batch_size=CFG.src_batch_size,
                                  shuffle=True, num_workers=CFG.num_workers,
                                  pin_memory=True)
    src_test_loader = DataLoader(src_test, batch_size=CFG.src_batch_size,
                                 shuffle=False, num_workers=CFG.num_workers,
                                 pin_memory=True)

    if imbalance_ratio is not None:
        sampler = make_imbalanced_sampler(tgt_train_raw, imbalance_ratio,
                                          CFG.num_classes)
        tgt_unl_loader = DataLoader(tgt_train, batch_size=CFG.sf_batch_size,
                                    sampler=sampler,
                                    num_workers=CFG.num_workers, pin_memory=True)
    else:
        tgt_unl_loader = DataLoader(tgt_train, batch_size=CFG.sf_batch_size,
                                    shuffle=True,
                                    num_workers=CFG.num_workers, pin_memory=True)

    tgt_test_loader = DataLoader(tgt_test, batch_size=CFG.sf_batch_size,
                                 shuffle=False, num_workers=CFG.num_workers,
                                 pin_memory=True)

    return (src_train_loader, src_test_loader, tgt_unl_loader,
            tgt_test_loader, tgt_train_raw)
