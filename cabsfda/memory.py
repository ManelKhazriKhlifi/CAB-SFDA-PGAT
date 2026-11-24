"""Class-aware momentum prototype memory (paper Sec. 3.3).

One prototype per class, updated as an EMA of the batch mean of the (raw,
un-normalised) 128-d weak-view features of the samples accepted by PGAT. The
cumulative accepted count n_c drives the inverse-square-root class weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ClassMemory:
    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9,
                 mode: str = "ema", device="cpu"):
        if mode not in ("ema", "batch", "static", "none"):
            raise ValueError(f"memory_mode {mode}")
        self.K, self.m, self.mode = num_classes, momentum, mode
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)
        self.counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.frozen = False          # True after static initialisation

    @property
    def enabled(self):
        return self.mode != "none"

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        """feats are detached, pre-step weak-view features of the selected samples."""
        if not self.enabled:
            return
        for c in labels.unique().tolist():
            sel = labels == c
            n = int(sel.sum())
            self.counts[c] += n
            if self.frozen:
                continue                       # static mode: prototypes fixed, counts still tracked
            fmean = feats[sel].mean(0)
            if not self.initialized[c] or self.mode == "batch":
                self.prototypes[c] = fmean     # first accepted batch initialises the prototype
                self.initialized[c] = True
            else:
                self.prototypes[c] = self.m * self.prototypes[c] + (1 - self.m) * fmean

    @torch.no_grad()
    def init_static(self, feats: torch.Tensor, labels: torch.Tensor):
        """'static' ablation: prototypes computed once from the source model's confident target predictions."""
        for c in range(self.K):
            sel = labels == c
            if sel.any():
                self.prototypes[c] = feats[sel].mean(0)
                self.initialized[c] = True
        self.frozen = True

    def class_weights(self, scheme: str = "inv_sqrt") -> torch.Tensor:
        """w_c proportional to n_c^{-1/2} (inv_sqrt), n_c^{-1} (inv_freq) or 1 (uniform), mean 1.

        Classes with no accepted sample yet (n_c = 0) receive the largest weight
        among classes that have samples (all weights equal while no class has
        any), so that a class accepted for the first time does not get an
        extreme weight."""
        if scheme == "uniform" or not self.enabled:
            return torch.ones(self.K, device=self.counts.device)
        n = self.counts.float()
        seen = n > 0
        if not seen.any():
            return torch.ones(self.K, device=n.device)
        power = 0.5 if scheme == "inv_sqrt" else 1.0
        if scheme not in ("inv_sqrt", "inv_freq"):
            raise ValueError(f"weighting {scheme}")
        w = torch.zeros_like(n)
        w[seen] = n[seen].pow(-power)
        w[~seen] = w[seen].max()
        return w * self.K / w.sum()

    def proto_loss(self, feats: torch.Tensor, labels: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
        """Accepted samples whose prototype exists. cosine (default): mean (1 - cos(f_i, mu_{y_i}));
        mse: (1 / (d |I+|)) sum ||f_i - mu_{y_i}||^2 (depends on the feature scale)."""
        if feats.numel() == 0 or not self.enabled:
            return feats.new_zeros(())
        valid = self.initialized[labels]
        if not valid.any():
            return feats.new_zeros(())
        if kind == "cosine":
            return (1 - F.cosine_similarity(feats[valid], self.prototypes[labels[valid]], dim=1)).mean()
        if kind == "mse":
            return F.mse_loss(feats[valid], self.prototypes[labels[valid]])
        raise ValueError(f"proto_loss {kind}")

    def state(self):
        return {"counts": self.counts.tolist(), "initialized": self.initialized.tolist()}
