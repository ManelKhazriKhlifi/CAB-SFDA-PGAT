#!/usr/bin/env python
"""Export the splits used by the experiments in a portable form (for release with the paper).

  python export_splits.py                     # data/splits -> splits/

For every domain, writes splits/<domain>/{train,test}.csv with columns sha1, label, class_name,
file (file name of the original image), plus splits/summary.csv. No local paths are written.
Anyone with the raw datasets can rebuild exactly these splits with
`python prepare_data.py --reference splits`.
"""
import argparse
import csv
import shutil
from pathlib import Path

from cabsfda.config import REPO_ROOT, load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="default: splits_dir of the config (data/splits)")
    ap.add_argument("--out", default="splits")
    a = ap.parse_args()
    src = REPO_ROOT / (a.src or load_config().splits_dir)
    out = REPO_ROOT / a.out
    n = 0
    for ddir in sorted(p for p in src.iterdir() if (p / "train.csv").exists()):
        (out / ddir.name).mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            with open(ddir / f"{split}.csv") as fi, open(out / ddir.name / f"{split}.csv", "w", newline="") as fo:
                w = csv.DictWriter(fo, fieldnames=["sha1", "label", "class_name", "file"])
                w.writeheader()
                for row in csv.DictReader(fi):
                    if "sha1" not in row:
                        raise SystemExit(f"{ddir}/{split}.csv has no sha1 column; rebuild it with prepare_data.py "
                                         f"or import_splits.py")
                    w.writerow({"sha1": row["sha1"], "label": row["label"], "class_name": row["class_name"],
                                "file": Path(row["orig_path"]).name})
                    n += 1
    if (src / "summary.csv").exists():
        shutil.copy(src / "summary.csv", out / "summary.csv")
    print(f"{n} images -> {out}/")


if __name__ == "__main__":
    main()
