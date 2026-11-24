#!/usr/bin/env python
"""Build the fixed, stratified train/test splits of every domain from the raw datasets.

  export DATA_ROOT=/path/to/raw/datasets      # or edit configs/datasets.yaml
  python prepare_data.py                       # all domains
  python prepare_data.py --resize-cache 128    # also store 128x128 copies (faster loading)
  python prepare_data.py --reference splits    # reproduce the released splits of the paper
  python prepare_data.py --domains WHU,EuroSAT --force

Procedure (identical to import_splits.py): every image is identified by the SHA-1 of its bytes;
identical files are kept once and images filed under two classes are dropped. The unique images
of each class are shuffled with a fixed seed and split 70/30 (stratified), and the same split
is used whether the domain is the source or the target.

With --reference DIR (default layout: splits/<domain>/{train,test}.csv, as released with the
paper), the images are assigned to train/test as in the released files, matched by SHA-1 and,
for files whose bytes differ (e.g. re-encoded copies), by class and file name. Unmatched images
are reported and left out.

Writes data/splits/<domain>/{train,test}.csv (path, label, class_name, orig_path, sha1) and
data/splits/summary.{csv,json} with per-class counts and imbalance ratios.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from cabsfda.config import REPO_ROOT, load_config, load_datasets

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_class_folders(domain, spec, root: Path):
    """canonical class -> list of folders. An alias "A+B" gathers several folders."""
    if not root.is_dir():
        sys.exit(f"[{domain}] root folder not found: {root}\n  set DATA_ROOT or edit configs/datasets.yaml")
    subdirs = {norm(p.name): p for p in root.iterdir() if p.is_dir()}
    out = {}
    for cls, aliases in spec["classes"].items():
        hit = None
        for alias in aliases:
            parts = [norm(x) for x in alias.split("+")]
            if all(x in subdirs for x in parts):
                hit = [subdirs[x] for x in parts]
                break
        if hit is None:
            sys.exit(f"[{domain}] class '{cls}': none of {aliases} found in {root}\n"
                     f"  available folders: {sorted(p.name for p in subdirs.values())}")
        out[cls] = hit
    used = [p.name for ps in out.values() for p in ps]
    if len(set(used)) != len(used):
        sys.exit(f"[{domain}] a folder is mapped to two classes: {used}")
    return out


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_reference(ref_dir: Path, domain: str):
    """{sha1: split}, {(class, file name): split} from the released split files."""
    by_hash, by_name = {}, {}
    for split in ("train", "test"):
        f = ref_dir / domain / f"{split}.csv"
        if not f.exists():
            sys.exit(f"[{domain}] reference split not found: {f}")
        with open(f) as fh:
            for row in csv.DictReader(fh):
                by_hash[row["sha1"]] = split
                by_name[(row["class_name"], row.get("file", ""))] = split
    return by_hash, by_name


def cache_png(p: Path, size: int, cache_dir: str, d: str, cls: str, h: str) -> Path:
    path = REPO_ROOT / cache_dir / str(size) / d / cls / f"{h[:16]}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(p) as im:
            im.convert("RGB").resize((size, size), Image.BILINEAR).save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", default=None, help="comma list (default: all)")
    ap.add_argument("--out", default=None, help="default: splits_dir of the config")
    ap.add_argument("--reference", default=None, help="folder with released splits (e.g. splits)")
    ap.add_argument("--resize-cache", type=int, default=0, help="store resized PNG copies of this size")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    a = ap.parse_args()

    ds = load_datasets()
    frac, split_seed = ds["split"]["train_fraction"], ds["split"]["seed"]
    domains = a.domains.split(",") if a.domains else list(ds["domains"])
    out_root = REPO_ROOT / (a.out or load_config().splits_dir)
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    ref_dir = (REPO_ROOT / a.reference) if a.reference else None

    for d in domains:
        spec = ds["domains"][d]
        classes = ds["benchmarks"][spec["benchmark"]]["classes"]
        if list(spec["classes"]) != classes:
            sys.exit(f"[{d}] classes {list(spec['classes'])} must equal the benchmark list {classes}")
        ddir = out_root / d
        if (ddir / "train.csv").exists() and not a.force:
            print(f"[{d}] split exists ({ddir}); use --force to rebuild")
            continue
        folders = resolve_class_folders(d, spec, Path(spec["root"]))
        files = [(cls, p) for cls in classes for f in folders[cls]
                 for p in sorted(f.rglob("*")) if p.suffix.lower() in EXTS and p.is_file()]
        with ThreadPoolExecutor(a.workers) as ex:
            hashes = list(ex.map(lambda t: sha1(t[1]), files, chunksize=64))
        first, labels_of = {}, {}                  # sha1 -> (class, path): first copy in path order
        for (cls, p), h in zip(files, hashes):
            labels_of.setdefault(h, set()).add(cls)
            if h not in first or str(p) < str(first[h][1]):
                first[h] = (cls, p)
        conflicts = {h for h, c in labels_of.items() if len(c) > 1}
        n_dup = len(files) - len(first)
        if ref_dir is not None:
            by_hash, by_name = read_reference(ref_dir, d)
        rows = {"train": [], "test": []}
        per_class, unmatched = [], 0
        for label, cls in enumerate(classes):
            hs = sorted(h for h, (c, _) in first.items() if c == cls and h not in conflicts)
            if spec.get("max_per_class"):
                hs = sorted(random.Random(f"{split_seed}/{d}/{cls}/max").sample(hs, min(len(hs), int(spec["max_per_class"]))))
            if ref_dir is None:
                rng = random.Random(f"{split_seed}/{d}/{cls}")
                rng.shuffle(hs)
                n_tr = int(frac * len(hs) + 0.5)
                assign = {h: ("train" if i < n_tr else "test") for i, h in enumerate(hs)}
            else:
                assign = {}
                for h in hs:
                    s = by_hash.get(h) or by_name.get((cls, first[h][1].name))
                    if s is None:
                        unmatched += 1
                    else:
                        assign[h] = s
            for h in sorted(assign, key=lambda h: str(first[h][1])):
                p = first[h][1]
                path = cache_png(p, a.resize_cache, a.cache_dir, d, cls, h) if a.resize_cache else p
                rows[assign[h]].append({"path": rel(path), "label": label, "class_name": cls,
                                        "orig_path": rel(p), "sha1": h})
            n_tr = sum(1 for s in assign.values() if s == "train")
            per_class.append({"class": cls, "folder": "+".join(f.name for f in folders[cls]),
                              "n": len(assign), "n_train": n_tr, "n_test": len(assign) - n_tr})
        assert not ({r["sha1"] for r in rows["train"]} & {r["sha1"] for r in rows["test"]})
        ddir.mkdir(parents=True, exist_ok=True)
        digests = {}
        for split in ("train", "test"):
            with open(ddir / f"{split}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["path", "label", "class_name", "orig_path", "sha1"])
                w.writeheader()
                w.writerows(rows[split])
            digests[split] = hashlib.sha1((ddir / f"{split}.csv").read_bytes()).hexdigest()[:12]
        ns = [c["n"] for c in per_class]
        summary[d] = {"benchmark": spec["benchmark"], "root": spec["root"], "per_class": per_class,
                      "n": sum(ns), "n_train": len(rows["train"]), "n_test": len(rows["test"]),
                      "imbalance_ratio": round(max(ns) / max(1, min(ns)), 3), "train_fraction": frac,
                      "split_seed": split_seed, "csv_sha1": digests, "duplicates_removed": n_dup,
                      "label_conflicts_removed": len(conflicts),
                      "reference": str(a.reference) if a.reference else None}
        print(f"[{d}] {sum(ns)} images, train {len(rows['train'])} / test {len(rows['test'])}, "
              f"imbalance ratio {max(ns) / max(1, min(ns)):.2f} ({n_dup} duplicate files, "
              f"{len(conflicts)} images in two classes removed)")
        if ref_dir is not None:
            n_ref = len(by_hash)
            print(f"    reference: {sum(ns)} of {n_ref} released images matched, {unmatched} local images not in it")
        for c in per_class:
            print(f"    {c['class']:22s} <- {c['folder']:36s} {c['n']:6d} ({c['n_train']}/{c['n_test']})")

    out_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1))
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "class", "folder", "n", "n_train", "n_test", "domain_imbalance_ratio"])
        for d, s in summary.items():
            for c in s["per_class"]:
                w.writerow([d, c["class"], c["folder"], c["n"], c["n_train"], c["n_test"], s["imbalance_ratio"]])
    print(f"summary -> {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
