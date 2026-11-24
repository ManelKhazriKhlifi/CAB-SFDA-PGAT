#!/usr/bin/env python
"""Tiny synthetic stand-in for the eight datasets (smoke tests only).

--layout raw (default): the folder layout expected by configs/datasets.yaml under --out (used as
  DATA_ROOT): one folder per class, named with the first alias of each class.
--layout splitfolders: pre-split folders, <out>/{source,target}_<name>_split/{train,test}/<Class>/.
  Some test images of target_whu_split are also put in its train folder, and one AID image is
  stored twice under two names, so that the audit of import_splits.py has something to detect.

Images have class-dependent stripe patterns and a domain-dependent colour/contrast shift; class
sizes differ per domain so that imbalance handling is exercised."""
import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from cabsfda.config import load_datasets

SPLIT_FOLDERS = {   # canonical class -> folder name used in the pre-split layout
    "farmland": "Farmland", "forest": "Forest", "industrial": "Industrial", "meadow": "Meadow",
    "parking": "Parking", "residential": "Residential", "river": "River", "beach": "Beach",
    "circular_farmland": "CircularFarmland", "cloud": "Cloud", "desert": "Desert", "mountain": "Mountain",
    "rectangular_farmland": "Rectangular farmland", "snowberg": "Snowberg"}
SPLIT_NAME = {"AID": "aid", "CLRS": "clrs", "MLRSN": "mlrsn", "RSSCN7": "rsscn7", "NWPU": "nwpu",
              "NaSC": "nasc", "WHU": "whu", "EuroSAT": "eurosat"}


def make_images(rng, n, size, ci, n_cls, tint, di):
    out = []
    for _ in range(n):
        yy, xx = np.mgrid[0:size, 0:size]
        angle = np.pi * ci / n_cls
        freq = 0.25 + 0.05 * ci
        base = 0.5 + 0.5 * np.sin(freq * (xx * np.cos(angle) + yy * np.sin(angle)) + rng.uniform(0, 6))
        img = np.stack([base * tint[k] for k in range(3)], -1)
        img += rng.normal(0, 0.15 + 0.05 * (di % 3), img.shape)
        out.append(Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["raw", "splitfolders"], default="raw")
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--size", type=int, default=40)
    a = ap.parse_args()
    os.environ["DATA_ROOT"] = str(Path(a.out).resolve())
    ds = load_datasets()
    rng = np.random.default_rng(0)
    for di, (dname, spec) in enumerate(ds["domains"].items()):
        tint = rng.uniform(0.6, 1.4, 3)
        classes = list(spec["classes"])
        for ci, (cls, aliases) in enumerate(spec["classes"].items()):
            n = a.n_per_class + (ci * 3 if di % 2 else 0)          # mild imbalance in half of the domains
            imgs = make_images(rng, n, a.size, ci, len(classes), tint, di)
            if a.layout == "raw":
                parts = aliases[0].split("+")          # "A+B": the class gathers several folders
                for k, part in enumerate(parts):
                    folder = Path(spec["root"]) / part
                    folder.mkdir(parents=True, exist_ok=True)
                    for i, im in enumerate(imgs):
                        if i % len(parts) == k:
                            im.save(folder / f"{cls}_{i:04d}.jpg")
                if dname == "AID" and ci == 0:         # one image stored twice (duplicate removal)
                    shutil.copy(folder / f"{cls}_0000.jpg", folder / f"copy_{cls}_0000.jpg")
                continue
            folder_name = "Industry" if (cls == "industrial" and dname in ("WHU", "EuroSAT")) else SPLIT_FOLDERS[cls]
            names = [f"{cls}_{i:02d}.jpg" for i in range(n)]
            order = list(range(n))
            random.Random(f"{dname}/{cls}").shuffle(order)
            n_tr = int(0.7 * n + 0.5)
            split = {"train": order[:n_tr], "test": order[n_tr:]}
            for role in ("source", "target"):
                for sp in ("train", "test"):
                    idx = split[sp]
                    if dname == "WHU" and role == "target":
                        idx = order[:-1] if sp == "train" else order[: n_tr]   # test images also in train
                    d = Path(a.out) / f"{role}_{SPLIT_NAME[dname]}_split" / sp / folder_name
                    d.mkdir(parents=True, exist_ok=True)
                    for i in idx:
                        imgs[i].save(d / names[i])
            if dname == "AID" and ci == 0:                             # one image stored twice
                src = Path(a.out) / "source_aid_split" / "train" / folder_name / names[split["train"][0]]
                shutil.copy(src, src.with_name("copy_of_" + src.name))
        print(f"{dname}: {a.out}")


if __name__ == "__main__":
    main()
