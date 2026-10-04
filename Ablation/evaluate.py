"""
evaluate.py
Evaluation utilities: accuracy, per-class metrics, confusion matrix.
"""
import numpy as np
import torch

from config import CFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(1, total)


def evaluate_with_cm(model, loader, num_classes=CFG.num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for t, p in zip(y.view(-1), pred.view(-1)):
                cm[int(t.item()), int(p.item())] += 1

    true_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)

    rec = [cm[c, c] / true_counts[c] if true_counts[c] > 0 else 0.0
           for c in range(num_classes)]
    prec = [cm[c, c] / pred_counts[c] if pred_counts[c] > 0 else 0.0
            for c in range(num_classes)]
    f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0.0
          for p, r in zip(prec, rec)]

    return {
        "cm": cm.tolist(),
        "overall_acc": float(cm.trace() / max(1, cm.sum())),
        "per_class_recall": [float(x) for x in rec],
        "per_class_precision": [float(x) for x in prec],
        "per_class_f1": [float(x) for x in f1],
        "macro_recall": float(np.mean(rec)),
        "balanced_acc": float(np.mean(rec)),
        "macro_precision": float(np.mean(prec)),
        "macro_f1": float(np.mean(f1)),
    }
