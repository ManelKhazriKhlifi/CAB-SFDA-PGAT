"""Evaluation on a labelled split: OA, balanced accuracy, macro-F1, per-class P/R/F1, confusion."""
from __future__ import annotations

import numpy as np
import torch

from .utils import get_device


@torch.no_grad()
def predict(model, loader, return_feats=False):
    """Runs the model in eval mode. Returns dict of numpy arrays (logits, labels[, feats])."""
    device = get_device()
    model.eval()
    logits, labels, feats = [], [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        z, f = model(x.to(device, non_blocking=True), return_feat=True)
        logits.append(z.float().cpu())
        labels.append(torch.as_tensor(y))
        if return_feats:
            feats.append(f.float().cpu())
    out = {"logits": torch.cat(logits).numpy(), "labels": torch.cat(labels).numpy()}
    if return_feats:
        out["feats"] = torch.cat(feats).numpy()
    return out


def classification_metrics(pred: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = np.diag(cm).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)  # 0 if never predicted
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    return {
        "oa": 100.0 * tp.sum() / max(1, cm.sum()),
        "balanced_acc": 100.0 * recall[present].mean(),     # = macro recall
        "macro_f1": 100.0 * f1[present].mean(),
        "macro_precision": 100.0 * precision[present].mean(),
        "per_class": {"precision": (100 * precision).tolist(), "recall": (100 * recall).tolist(),
                      "f1": (100 * f1).tolist(), "support": support.tolist(),
                      "predicted": predicted.tolist()},
        "confusion": cm.tolist(),
        "n": int(cm.sum()),
    }


def evaluate(model, loader, num_classes, return_raw=False, return_feats=False):
    raw = predict(model, loader, return_feats=return_feats)
    m = classification_metrics(raw["logits"].argmax(1), raw["labels"], num_classes)
    return (m, raw) if return_raw else m


def unsupervised_scores(logits: np.ndarray, tau: float = 0.05, chunk: int = 2048) -> dict:
    """Label-free model-selection scores on the unlabeled adaptation images.

    snd: Soft Neighborhood Density (Saito et al., ICCV 2021) - entropy of the softmax over the
         cosine similarities (temperature tau = 0.05) between each sample's softmax output and those
         of all other samples, averaged; higher = denser, better-clustered predictions.
    pred_class_entropy: entropy of the predicted-class histogram / log K (1 = all classes used
         equally, 0 = a single class: collapse).
    mean_entropy: mean prediction entropy (confidence)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = torch.as_tensor(logits, dtype=torch.float32, device=device).softmax(1)
    K = p.shape[1]
    pn = torch.nn.functional.normalize(p, dim=1)
    ent = []
    for i in range(0, len(pn), chunk):
        sim = pn[i:i + chunk] @ pn.T / tau
        rows = torch.arange(i, min(i + chunk, len(pn)), device=device)
        sim[torch.arange(len(rows), device=device), rows] = float("-inf")     # exclude self
        q = sim.softmax(1)
        ent.append(-(q * torch.log(q.clamp_min(1e-12))).sum(1))
    snd = float(torch.cat(ent).mean()) if len(pn) > 1 else float("nan")
    hist = torch.bincount(p.argmax(1), minlength=K).float()
    f = hist / hist.sum()
    pce = float(-(f[f > 0] * f[f > 0].log()).sum() / np.log(K))
    me = float(-(p * p.clamp_min(1e-12).log()).sum(1).mean())
    return {"snd": snd, "pred_class_entropy": pce, "mean_entropy": me}
