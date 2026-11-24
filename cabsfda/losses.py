"""Losses shared by CAB-SFDA and the baselines."""
from __future__ import annotations

import torch.nn.functional as F


def entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(1)


def im_loss(logits, mode="standard"):
    """Information maximisation (SHOT): mean entropy - entropy of the mean prediction."""
    p = logits.softmax(1)
    ent = entropy(p).mean()
    if mode == "entropy_only":
        return ent
    return ent - entropy(p.mean(0, keepdim=True)).mean()


def kl_t(logits_a, logits_b, T):
    """KL(q_a || q_b) with q = softmax(z / T); gradients flow into both arguments
    (used by the symmetric consistency variant)."""
    log_qa = F.log_softmax(logits_a / T, 1)
    return (log_qa.exp() * (log_qa - F.log_softmax(logits_b / T, 1))).sum(1).mean()


def consistency_loss(logits_w, logits_s, T, mode="symmetric"):
    if mode == "symmetric":
        return 0.5 * (kl_t(logits_w, logits_s, T) + kl_t(logits_s, logits_w, T))
    if mode == "oneway":           # weak view as fixed target (stop-gradient), strong view as student
        return kl_t(logits_w.detach(), logits_s, T)
    if mode == "none":
        return logits_w.new_zeros(())
    raise ValueError(f"cons_mode {mode}")


def weighted_pl_loss(logits_s, targets, weights):
    """(1/|I+|) sum_i w_{y_i} CE(z_i^s, y_i) over accepted samples."""
    if targets.numel() == 0:
        return logits_s.new_zeros(())
    return (F.cross_entropy(logits_s, targets, reduction="none") * weights[targets]).mean()
