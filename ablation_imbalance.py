import os
import json
import random
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Subset

from config import CFG
from dataloaders import (build_loaders, TwoViewTarget,
                         tgt_weak_tf, tgt_strong_tf)
from train_source import get_or_train_source
from adapt import run_adaptation
from evaluate import run_final_evaluation


def make_imbalanced_subset(base_ds, ratio, seed=0, base=200):
    rng = random.Random(seed)
    targets = np.array([s[1] for s in base_ds.samples])
    classes = np.unique(targets)
    counts = {c: max(10, int(base / (ratio ** (i / max(1, len(classes) - 1)))))
              for i, c in enumerate(classes)}
    idx = []
    for c in classes:
        cls_idx = np.where(targets == c)[0].tolist()
        rng.shuffle(cls_idx)
        idx.extend(cls_idx[: min(counts[c], len(cls_idx))])
    return Subset(base_ds, idx)


def run():
    # The study is defined on the MLRSN target. Use a task whose target IS
    # MLRSN (A2M / C2M / R2M) instead of overriding the target, so the
    # source model and the output folder always match the task.
    if CFG.target_name != "MLRSN":
        raise SystemExit(f"ablation_imbalance.py needs an MLRSN-target task "
                         f"(A2M, C2M or R2M); got TASK={CFG.task}.")
    loaders = build_loaders()
    src_ckpt = get_or_train_source(loaders)

    tgt_train_raw = loaders["tgt_train_raw"]
    root = os.path.join(CFG.out_dir, "ablation_imbalance")

    results = {}
    for r in [1.0, 1.5, 2.0, 3.0, 5.0]:
        print(f"\n=== MLRSN imbalance ratio {r} ===")
        sub      = make_imbalanced_subset(tgt_train_raw, r, seed=CFG.seed)
        sub_two  = TwoViewTarget(sub, tgt_weak_tf, tgt_strong_tf)
        loader_r = DataLoader(sub_two,
                              batch_size=CFG.sf_batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              pin_memory=CFG.pin_memory)

        local = dict(loaders)
        local["tgt_unl"] = loader_r

        ratio_dir = os.path.join(root, f"ratio_{r}")
        out = run_adaptation(src_ckpt, local, verbose=True, out_dir=ratio_dir)
        report = run_final_evaluation(out["final_ckpt"], local,
                                      out_dir=ratio_dir, method=f"imbalance_{r}")
        # Minority = tail half of classes (fewest samples after subsampling);
        # fixed across ratios so values are comparable.
        k = len(report["per_class_f1"])
        minority = list(range((k + 1) // 2, k))
        results[str(r)] = {
            "overall_acc":      float(report["overall_acc"] * 100),
            "macro_f1":         float(report["macro_f1"] * 100),
            "minority_f1":      float(np.mean([report["per_class_f1"][c] for c in minority]) * 100),
            "minority_classes": minority,
        }
        print(f"ratio {r}: OA {results[str(r)]['overall_acc']:.2f}  "
              f"Minority F1 {results[str(r)]['minority_f1']:.2f}")

    path = os.path.join(root, "ablation_imbalance.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", path)


if __name__ == "__main__":
    run()