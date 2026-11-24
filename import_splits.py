#!/usr/bin/env python
"""Build the split files from pre-split folders and audit them.

Expected layout (searched up to two levels below --src):
    <dir>/source_<name>_split/{train,test}/<class>/<images>
    <dir>/target_<name>_split/{train,test}/<class>/<images>
with <name> in aid, clrs, mlrsn, rsscn7, nwpu, nasc, whu, eurosat.

  python import_splits.py --src /path/to/folders                       # audit + disjoint split
  python import_splits.py --src /path/to/folders --resize-cache 128    # also cache 128x128 PNGs (faster)
  python import_splits.py --src /path/to/folders --audit-only

Every image is identified by the SHA-1 of its bytes, so copies of one image under different
folders or names count once. All unique images of a domain (union of its four folders) are
split into a stratified 70/30 train/test split (seed 2024, as in prepare_data.py); train and
test are disjoint by content, and the same split is used whether the domain is the source or
the target. Images that occur in two different classes are dropped. Writes
<out>/<domain>/{train,test}.csv (path, label, class_name, orig_path, sha1).

The audit (<out>/audit.txt, audit.json) reports, per domain: images per folder and class,
train/test overlap inside each folder, whether the source and target copies agree, duplicates,
and images filed under two classes.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from cabsfda.config import REPO_ROOT, load_config, load_datasets

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
FOLDER_NAME = {"AID": "aid", "CLRS": "clrs", "MLRSN": "mlrsn", "RSSCN7": "rsscn7", "NWPU": "nwpu",
               "NaSC": "nasc", "WHU": "whu", "EuroSAT": "eurosat"}
ROLES, SPLITS = ("source", "target"), ("train", "test")


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_split_root(src: Path) -> Path:
    """Directory that contains source_aid_split etc. (src itself, or up to two levels below)."""
    for depth in range(3):
        hits = sorted(src.glob("/".join(["*"] * depth + ["source_*_split"])) if depth else src.glob("source_*_split"))
        if hits:
            roots = sorted({h.parent for h in hits})
            if len(roots) > 1:
                sys.exit(f"several folders contain source_*_split: {roots}; pass one of them with --src")
            return roots[0]
    sys.exit(f"no source_<name>_split folder found in {src} (searched two levels deep)")


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def match_classes(domain, spec, classes, folder: Path):
    """canonical class -> class folder inside one split folder."""
    sub = {norm(p.name): p for p in folder.iterdir() if p.is_dir()}
    out = {}
    for cls in classes:
        cands = [*spec["classes"][cls], cls]
        hit = next((sub[norm(a)] for a in cands if norm(a) in sub), None)
        if hit is None:
            sys.exit(f"[{domain}] class '{cls}' (aliases {cands}) not found in {folder}\n"
                     f"  folders there: {sorted(p.name for p in sub.values())}\n"
                     f"  add the folder name to the aliases in configs/datasets.yaml")
        out[cls] = hit
    extra = sorted(set(p.name for p in sub.values()) - {p.name for p in out.values()})
    if extra:
        sys.exit(f"[{domain}] class folders not mapped to any class in {folder}: {extra}")
    return out


def rel(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def scan(root: Path, ds, domains, workers):
    """{domain: {(role, split): [(class, path, sha1)]}}"""
    found, jobs = {}, []
    for d in domains:
        spec = ds["domains"][d]
        classes = ds["benchmarks"][spec["benchmark"]]["classes"]
        found[d] = {}
        for role in ROLES:
            for split in SPLITS:
                folder = root / f"{role}_{FOLDER_NAME[d]}_split" / split
                if not folder.is_dir():
                    sys.exit(f"[{d}] missing folder {folder}")
                for cls, cdir in match_classes(d, spec, classes, folder).items():
                    files = sorted(p for p in cdir.rglob("*") if p.suffix.lower() in EXTS and p.is_file())
                    if not files:
                        sys.exit(f"[{d}] no images in {cdir}")
                    for p in files:
                        jobs.append((d, role, split, cls, p))
    print(f"hashing {len(jobs)} files with {workers} threads ...", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        digests = list(ex.map(lambda j: sha1(j[4]), jobs, chunksize=64))
    for (d, role, split, cls, p), h in zip(jobs, digests):
        found[d].setdefault((role, split), []).append((cls, p, h))
    return found


def audit_domain(d, items, classes):
    rep = {"folders": {}, "overlap": {}, "copies_agree": {}, "label_conflicts": 0, "per_class_unique": {}}
    sets = {}
    for (role, split), lst in items.items():
        counts = {c: 0 for c in classes}
        for cls, _, _ in lst:
            counts[cls] += 1
        hashes = [h for _, _, h in lst]
        sets[(role, split)] = set(hashes)
        rep["folders"][f"{role}_{split}"] = {"n": len(lst), "unique": len(set(hashes)),
                                             "duplicates_inside": len(hashes) - len(set(hashes)),
                                             "per_class": counts}
    for role in ROLES:
        tr, te = sets[(role, "train")], sets[(role, "test")]
        rep["overlap"][role] = {"train_and_test": len(tr & te), "test": len(te),
                                "pct_of_test": round(100 * len(tr & te) / max(1, len(te)), 1)}
    for split in SPLITS:
        s, t = sets[("source", split)], sets[("target", split)]
        rep["copies_agree"][split] = {"identical": s == t, "only_in_source": len(s - t),
                                      "only_in_target": len(t - s)}
    labels = defaultdict(set)
    for lst in items.values():
        for cls, _, h in lst:
            labels[h].add(cls)
    conflicts = {h for h, c in labels.items() if len(c) > 1}
    rep["label_conflicts"] = len(conflicts)
    for c in classes:
        rep["per_class_unique"][c] = sum(1 for h, cs in labels.items() if cs == {c})
    rep["unique_images"] = sum(rep["per_class_unique"].values())
    return rep, labels, conflicts


def cache_image(src: Path, h: str, d: str, cls: str, size: int, cache_root: Path):
    out = cache_root / str(size) / d / cls / f"{h}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.convert("RGB").resize((size, size), Image.BILINEAR).save(out.with_suffix(".tmp.png"))
        out.with_suffix(".tmp.png").replace(out)
    return out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "class_name", "orig_path", "sha1"])
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder containing (or two levels above) source_*_split")
    ap.add_argument("--out", default=None, help="default: splits_dir of the config (data/splits)")
    ap.add_argument("--domains", default=None, help="comma list (default: all eight)")
    ap.add_argument("--resize-cache", type=int, default=0, help="store resized PNG copies (e.g. 128)")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    a = ap.parse_args()

    ds = load_datasets()
    frac, split_seed = ds["split"]["train_fraction"], ds["split"]["seed"]
    domains = a.domains.split(",") if a.domains else list(ds["domains"])
    root = find_split_root(Path(a.src).expanduser().resolve())
    out_root = REPO_ROOT / (a.out or load_config().splits_dir)
    print(f"split folders: {root} -> {out_root}")
    if not a.audit_only and not a.force and any((out_root / d).exists() for d in domains):
        sys.exit(f"{out_root} already has split files; use --force to rebuild (or another --out)")

    found = scan(root, ds, domains, a.workers)
    audit, lines, summary = {}, [], {}
    cache_root = REPO_ROOT / a.cache_dir
    for d in domains:
        spec = ds["domains"][d]
        classes = ds["benchmarks"][spec["benchmark"]]["classes"]
        rep, labels, conflicts = audit_domain(d, found[d], classes)
        audit[d] = rep
        f = rep["folders"]
        lines.append(f"[{d}] unique images {rep['unique_images']}; label conflicts {rep['label_conflicts']}")
        for k in ("source_train", "source_test", "target_train", "target_test"):
            lines.append(f"    {k:13s} {f[k]['n']:6d} files ({f[k]['duplicates_inside']} duplicate copies)")
        for role in ROLES:
            o = rep["overlap"][role]
            flag = "   <-- TEST IMAGES ALSO IN TRAIN" if o["train_and_test"] else ""
            lines.append(f"    {role}: train/test overlap {o['train_and_test']}/{o['test']} test images "
                         f"({o['pct_of_test']}%){flag}")
        for split in SPLITS:
            c = rep["copies_agree"][split]
            lines.append(f"    source vs target copy, {split}: " + ("identical" if c["identical"] else
                         f"differ ({c['only_in_source']} only in source, {c['only_in_target']} only in target)"))
        if a.audit_only:
            continue

        first = {}                              # sha1 -> (class, path): first copy in sorted path order
        for lst in found[d].values():
            for cls, p, h in lst:
                if h in conflicts:
                    continue
                if h not in first or str(p) < str(first[h][1]):
                    first[h] = (cls, p)
        rows = {"train": [], "test": []}
        per_class = []
        for label, cls in enumerate(classes):
            hs = sorted(h for h, (c, _) in first.items() if c == cls)
            rng = random.Random(f"{split_seed}/{d}/{cls}")
            rng.shuffle(hs)
            n_tr = int(frac * len(hs) + 0.5)
            for split, chunk in (("train", hs[:n_tr]), ("test", hs[n_tr:])):
                for h in sorted(chunk, key=lambda h: str(first[h][1])):
                    p = first[h][1]
                    path = cache_image(p, h, d, cls, a.resize_cache, cache_root) if a.resize_cache else p
                    rows[split].append({"path": rel(path), "label": label, "class_name": cls,
                                        "orig_path": rel(p), "sha1": h})
            per_class.append({"class": cls, "folder": cls, "n": len(hs), "n_train": n_tr,
                              "n_test": len(hs) - n_tr})
        digests = {s: write_csv(out_root / d / f"{s}.csv", rows[s]) for s in SPLITS}
        n_train, n_test = len(rows["train"]), len(rows["test"])
        assert not ({r["sha1"] for r in rows["train"]} & {r["sha1"] for r in rows["test"]})
        ns = [c["n"] for c in per_class]
        summary[d] = {"benchmark": spec["benchmark"], "root": str(root), "per_class": per_class,
                      "n": sum(ns), "n_train": n_train, "n_test": n_test,
                      "imbalance_ratio": round(max(ns) / max(1, min(ns)), 3), "train_fraction": frac,
                      "split_seed": split_seed, "csv_sha1": digests}
        lines.append(f"    -> train {n_train} / test {n_test} (disjoint by content)")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "audit.json").write_text(json.dumps({"split_root": str(root), "domains": audit}, indent=1))
    (out_root / "audit.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not a.audit_only:
        (out_root / "summary.json").write_text(json.dumps(summary, indent=1))
        with open(out_root / "summary.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["domain", "class", "folder", "n", "n_train", "n_test", "domain_imbalance_ratio"])
            for d, s in summary.items():
                for c in s["per_class"]:
                    w.writerow([d, c["class"], c["folder"], c["n"], c["n_train"], c["n_test"], s["imbalance_ratio"]])
    print(f"audit -> {out_root / 'audit.txt'}")


if __name__ == "__main__":
    main()
