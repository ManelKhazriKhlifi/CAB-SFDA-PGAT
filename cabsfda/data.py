"""Splits, transforms, datasets and loaders.

Splits are produced once by prepare_data.py and stored as CSV files
(data/splits/<domain>/{train,test}.csv). Every method, seed and experiment
reads the same files, so the split is fixed and can be released with the code.

Target labels: the adaptation dataset returns the label in a separate field
('y_hidden'). Adaptation code may only pass it to logging functions
(pseudo-label quality); it is never used by a loss, a threshold, or any
model/checkpoint/hyper-parameter selection.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import REPO_ROOT, resolve_path
from .utils import make_generator, seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ splits
def split_csv_path(cfg, domain: str, split: str, role: str | None = None) -> Path:
    """<splits_dir>/<domain>/<split>.csv, or <role>_<split>.csv when that file exists.

    prepare_data.py and import_splits.py write one split per domain, used whether the domain
    is the source or the target. Role-specific files (source_train.csv, target_test.csv, ...)
    are used instead when they exist."""
    d = resolve_path(cfg.splits_dir) / domain
    if role is not None and (d / f"{role}_{split}.csv").exists():
        return d / f"{role}_{split}.csv"
    return d / f"{split}.csv"


def read_split(cfg, domain: str, split: str, role: str | None = None):
    """Returns (paths, labels) for split in {'train', 'test'}; role in {None, 'source', 'target'}."""
    path = split_csv_path(cfg, domain, split, role)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python import_splits.py` "
                                f"(pre-split folders) or `python prepare_data.py` (raw datasets) first.")
    paths, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            p = Path(row["path"])
            paths.append(str(p if p.is_absolute() else REPO_ROOT / p))
            labels.append(int(row["label"]))
    return paths, np.asarray(labels, dtype=np.int64)


def imbalanced_subset(labels: np.ndarray, num_classes: int, ratio: float, base: int,
                      seed: int, order: str = "canonical"):
    """Indices of a sub-sample whose class sizes decay geometrically from `base`
    (largest class) to `base / ratio` (smallest class).

    order='canonical': class 0 is the largest, class K-1 the smallest.
    order='random'   : class ranking is a seeded permutation.
    Returns (indices, per-class counts, class ranking)."""
    rng = np.random.default_rng(seed)
    ranking = list(range(num_classes)) if order == "canonical" else list(rng.permutation(num_classes))
    idx_out, counts = [], np.zeros(num_classes, dtype=int)
    for rank, c in enumerate(ranking):
        pool = np.flatnonzero(labels == c)
        want = int(round(base * ratio ** (-rank / max(1, num_classes - 1))))
        if want > len(pool):
            raise ValueError(f"class {c} has {len(pool)} adaptation images, {want} requested; "
                             f"lower imb_base")
        take = rng.choice(pool, size=want, replace=False)
        idx_out.append(take)
        counts[c] = want
    return np.sort(np.concatenate(idx_out)), counts, ranking


# ------------------------------------------------------------------ transforms
def build_transforms(img_size: int, strong_flip: bool = True):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    resize = transforms.Resize((img_size, img_size))
    test = transforms.Compose([resize, transforms.ToTensor(), norm])
    weak = transforms.Compose([resize, transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    strong = transforms.Compose([
        resize,
        *([transforms.RandomHorizontalFlip()] if strong_flip else []),
        transforms.RandomCrop(img_size, padding=8),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(), norm])
    return {"test": test, "weak": weak, "strong": strong, "source_train": weak}


def _load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


# ------------------------------------------------------------------ datasets
class LabeledImages(Dataset):
    """(image, label, index) - for source training and evaluation."""

    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), np.asarray(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(_load_rgb(self.paths[i])), int(self.labels[i]), i


class UnlabeledImages(Dataset):
    """(image, index) - target adaptation images for banks/clustering (no labels)."""

    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(_load_rgb(self.paths[i])), i


class TargetViews(Dataset):
    """Unlabeled target images with a weak view and `n_strong` strong views.

    'y_hidden' is the ground-truth label, returned only so that pseudo-label
    quality can be logged. It must not reach any loss."""

    def __init__(self, paths, labels, weak, strong, n_strong=1):
        self.paths, self.labels = list(paths), np.asarray(labels)
        self.weak, self.strong, self.n_strong = weak, strong, n_strong

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = _load_rgb(self.paths[i])
        out = {"weak": self.weak(img), "idx": i, "y_hidden": int(self.labels[i])}
        for k in range(self.n_strong):
            out["strong" if k == 0 else f"strong{k}"] = self.strong(img)
        return out


def make_loader(dataset, batch_size, shuffle, seed, num_workers, train=False):
    # A final batch of size 1 breaks BatchNorm in train mode; drop it only in that case.
    drop_last = train and (len(dataset) % batch_size == 1)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=make_generator(seed),
                      persistent_workers=train and num_workers > 0)


class TaskData:
    """All loaders for one (task, seed) adaptation run."""

    def __init__(self, cfg, info, seed, n_strong=1):
        tf = build_transforms(cfg.img_size, cfg.strong_flip)
        self.info = info
        tgt_paths, tgt_labels = read_split(cfg, info["target"], "train", role="target")
        test_paths, test_labels = read_split(cfg, info["target"], "test", role="target")
        self.subset_counts = None
        if cfg.imb_subset:
            # same sub-sample for every method and seed (fixed imb_subset_seed)
            idx, counts, ranking = imbalanced_subset(tgt_labels, info["num_classes"], cfg.imb_ratio,
                                                     cfg.imb_base, seed=cfg.imb_subset_seed,
                                                     order=cfg.imb_class_order)
            tgt_paths = [tgt_paths[i] for i in idx]
            tgt_labels = tgt_labels[idx]
            self.subset_counts, self.subset_ranking = counts.tolist(), [int(r) for r in ranking]
        self.adapt_paths, self.adapt_labels = tgt_paths, tgt_labels
        self.test_paths, self.test_labels = test_paths, test_labels
        self.n_adapt, self.n_test = len(tgt_paths), len(test_paths)
        nw, bs = cfg.num_workers, cfg.batch_size
        # shuffled two-view loader for training
        self.train = make_loader(TargetViews(tgt_paths, tgt_labels, tf["weak"], tf["strong"], n_strong),
                                 bs, True, seed, nw, train=True)
        # deterministic loader over the adaptation images (memory banks, clustering); no labels
        self.adapt_eval = make_loader(UnlabeledImages(tgt_paths, tf["test"]), bs, False, seed, nw)
        # held-out target test split (evaluation only)
        self.test = make_loader(LabeledImages(test_paths, test_labels, tf["test"]), bs, False, seed, nw)
        self.weak_tf = tf["weak"]
