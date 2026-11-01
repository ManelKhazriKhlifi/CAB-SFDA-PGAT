import os
import json
import numpy as np
import torch

from config import CFG, DEVICE
from model import ResNetBackbone


def evaluate_with_cm(model, loader, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            for t, p in zip(y.view(-1), pred.view(-1)):
                cm[int(t.item()), int(p.item())] += 1

    true_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    per_class_recall    = [(cm[c, c] / true_counts[c]) if true_counts[c] else 0.0
                           for c in range(num_classes)]
    per_class_precision = [(cm[c, c] / pred_counts[c]) if pred_counts[c] else 0.0
                           for c in range(num_classes)]
    per_class_f1        = [2 * p * r / (p + r) if (p + r) else 0.0
                           for p, r in zip(per_class_precision, per_class_recall)]

    overall_acc = cm.trace() / max(1, cm.sum())
    return {
        "cm": cm.tolist(),
        "overall_acc": float(overall_acc),
        "balanced_acc": float(np.mean(per_class_recall)),
        "macro_f1": float(np.mean(per_class_f1)),
        "per_class_recall":    [float(x) for x in per_class_recall],
        "per_class_precision": [float(x) for x in per_class_precision],
        "per_class_f1":        [float(x) for x in per_class_f1],
    }


def run_final_evaluation(final_ckpt_path, loaders, arch=None,
                         out_dir=None, method="cab_sfda"):
    """Evaluate once and write final_evaluation.json into `out_dir`.

    Every method / ablation variant must pass its OWN out_dir so results
    never overwrite each other.
    """
    arch = arch or CFG.backbone
    out_dir = out_dir or os.path.join(CFG.out_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    model = ResNetBackbone(num_classes=CFG.num_classes,
                           feat_dim=CFG.feat_dim,
                           arch=arch).to(DEVICE)
    model.load_state_dict(torch.load(final_ckpt_path, map_location=DEVICE))

    report = evaluate_with_cm(model, loaders["tgt_test"], CFG.num_classes)
    print(f"\n=== FINAL ({CFG.task} | seed={CFG.seed}) ===")
    print(f"Overall accuracy:  {report['overall_acc']*100:.2f}%")
    print(f"Balanced accuracy: {report['balanced_acc']*100:.2f}%")
    print(f"Macro F1:          {report['macro_f1']*100:.2f}%")

    out = {
        "method": method,
        "checkpoint_path": final_ckpt_path,
        "task": CFG.task,
        "source": CFG.source_name,
        "target": CFG.target_name,
        "seed": CFG.seed,
        "protocol": {"src_epochs": CFG.src_epochs,
                     "sf_epochs":  CFG.sf_epochs,
                     "checkpoint": "final_epoch"},
        "metrics": report,
    }
    path = os.path.join(out_dir, "final_evaluation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved final evaluation:", path)
    return report