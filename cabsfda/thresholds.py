"""Pseudo-label acceptance policies.

PGAT (ours, paper Sec. 3.4) and the controls used in the threshold ablation:
  pgat            per-class threshold from confidence + prototype agreement (ours)
  pgat_global     one threshold for all classes, same reliability formula pooled over the batch
  pgat_conf_only  per-class, R_c = q_c (no prototype term)
  fixed           global constant threshold
  flexmatch       class-wise flexible thresholds of FlexMatch (Zhang et al., NeurIPS 2021)
  freematch       self-adaptive thresholds of FreeMatch (Wang et al., ICLR 2023)

Ordering: by default a policy's state used for batch t comes from batches < t
(the paper's causal lag). `update_before_mask` is True for FreeMatch (its
original ordering) and for the 'no causal lag' PGAT ablation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Policy:
    update_before_mask = False

    def __init__(self, K, device):
        self.K, self.device = K, device
        self.tau = torch.full((K,), 0.0, device=device)

    def mask(self, conf, pred, idx=None):
        return conf >= self.tau[pred]

    def update(self, probs, conf, pred, feats, memory, idx=None):
        pass

    def snapshot(self):
        return {"tau": self.tau.tolist()}


class FixedThreshold(Policy):
    def __init__(self, K, device, value):
        super().__init__(K, device)
        self.tau.fill_(float(value))


class PGAT(Policy):
    def __init__(self, K, device, cfg, mode="per_class"):
        super().__init__(K, device)
        self.mode = mode
        self.base, self.gamma = cfg.tau_base, cfg.gamma
        self.tmin, self.tmax = cfg.tau_min, cfg.tau_max
        self.lam = 1.0 if mode == "conf_only" else cfg.lambda_r
        self.m, self.n_min, self.proto_min = cfg.tau_momentum, cfg.n_min, cfg.proto_min_count
        self.tau.fill_(self.base)
        self.q = torch.zeros(K, device=device)
        self.r = torch.zeros(K, device=device)
        self.R = torch.zeros(K, device=device)
        self.active = torch.zeros(K, dtype=torch.bool, device=device)

    def _raw(self, R):
        return (self.base - self.gamma * (R - 0.5)).clamp(self.tmin, self.tmax)

    def _proto_ready(self, memory):
        if memory is None or not memory.enabled:
            return torch.zeros(self.K, dtype=torch.bool, device=self.device)
        return memory.initialized & (memory.counts >= self.proto_min)

    @torch.no_grad()
    def update(self, probs, conf, pred, feats, memory, idx=None):
        ready = self._proto_ready(memory)
        if self.mode == "global":
            if conf.numel() < self.n_min:
                return
            q = conf.mean().clamp(0, 1)
            sel = ready[pred]
            if sel.any() and self.lam < 1:
                cos = (F.normalize(feats[sel], dim=1) * F.normalize(memory.prototypes[pred[sel]], dim=1)).sum(1)
                r = ((1 + cos.mean()) / 2).clamp(0, 1)
            else:
                r = q
            R = (self.lam * q + (1 - self.lam) * r).clamp(0, 1)
            self.tau.mul_(self.m).add_((1 - self.m) * self._raw(R))
            self.q.fill_(float(q)), self.r.fill_(float(r)), self.R.fill_(float(R))
            return
        for c in range(self.K):
            sel = pred == c
            if int(sel.sum()) < self.n_min:
                continue
            q = conf[sel].mean().clamp(0, 1)
            if ready[c] and self.lam < 1:
                cos = (F.normalize(feats[sel], dim=1) * F.normalize(memory.prototypes[c][None], dim=1)).sum(1)
                r = ((1 + cos.mean()) / 2).clamp(0, 1)
            else:
                r = q                                  # warm-up / no memory: confidence only
            R = (self.lam * q + (1 - self.lam) * r).clamp(0, 1)
            self.tau[c] = self.m * self.tau[c] + (1 - self.m) * self._raw(R)
            self.q[c], self.r[c], self.R[c] = q, r, R
            self.active[c] = bool(ready[c])

    def snapshot(self):
        return {"tau": self.tau.tolist(), "q": self.q.tolist(), "r": self.r.tolist(),
                "R": self.R.tolist(), "proto_active": self.active.tolist()}


class FlexMatch(Policy):
    """tau_c = p_cutoff * M(beta_c), M(x) = x / (2 - x), beta_c = sigma_c / max(...).
    sigma_c = number of adaptation samples whose latest confident (>= p_cutoff) prediction is c."""

    def __init__(self, K, device, n_samples, p_cutoff, warmup=True):
        super().__init__(K, device)
        self.p, self.warmup = p_cutoff, warmup
        self.selected = torch.full((n_samples,), -1, dtype=torch.long, device=device)
        self.beta = torch.zeros(K, device=device)
        self._refresh()

    def _refresh(self):
        counts = torch.bincount(self.selected + 1, minlength=self.K + 1).float()   # index 0 = unused
        # as in the official code, beta is only recomputed while no class holds every sample
        if counts[1:].max() < self.selected.numel():
            denom = counts.max() if self.warmup else counts[1:].max()
            self.beta = counts[1:] / denom.clamp_min(1)
        self.tau = self.p * self.beta / (2 - self.beta)

    @torch.no_grad()
    def update(self, probs, conf, pred, feats, memory, idx=None):
        sel = conf >= self.p
        self.selected[idx[sel]] = pred[sel]
        self._refresh()

    def snapshot(self):
        return {"tau": self.tau.tolist(), "beta": self.beta.tolist()}


class FreeMatch(Policy):
    """Self-adaptive thresholding: tau_c = tau_global * p_model_c / max(p_model)."""
    update_before_mask = True

    def __init__(self, K, device, momentum):
        super().__init__(K, device)
        self.m = momentum
        self.time_p = torch.tensor(1.0 / K, device=device)
        self.p_model = torch.full((K,), 1.0 / K, device=device)
        self._refresh()

    def _refresh(self):
        self.tau = self.time_p * self.p_model / self.p_model.max()

    @torch.no_grad()
    def update(self, probs, conf, pred, feats, memory, idx=None):
        self.time_p = self.m * self.time_p + (1 - self.m) * conf.mean()
        self.p_model = self.m * self.p_model + (1 - self.m) * probs.mean(0)
        self._refresh()

    def snapshot(self):
        return {"tau": self.tau.tolist(), "tau_global": float(self.time_p)}


def build_policy(cfg, K, device, n_samples):
    t = cfg.threshold
    if t == "pgat":
        p = PGAT(K, device, cfg, "per_class")
    elif t == "pgat_global":
        p = PGAT(K, device, cfg, "global")
    elif t == "pgat_conf_only":
        p = PGAT(K, device, cfg, "conf_only")
    elif t == "fixed":
        p = FixedThreshold(K, device, cfg.fixed_tau)
    elif t == "flexmatch":
        p = FlexMatch(K, device, n_samples, cfg.tau_base, cfg.flexmatch_warmup)
    elif t == "freematch":
        p = FreeMatch(K, device, cfg.freematch_momentum)
    else:
        raise ValueError(f"unknown threshold policy '{t}'")
    if isinstance(p, PGAT) and not cfg.threshold_lag:
        p.update_before_mask = True
    return p
