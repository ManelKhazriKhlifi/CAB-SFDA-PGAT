"""
losses.py
All loss functions used by CAB-SFDA, with ablation-aware variants.
"""
import torch
import torch.nn.functional as F


def entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(dim=1)


def info_max_loss(logits):
    """Prior-aware information maximization (OURS)."""
    p = torch.softmax(logits, dim=1)
    return entropy(p).mean() - entropy(p.mean(dim=0, keepdim=True)).mean()


def standard_im_loss(logits):
    """SHOT-style entropy minimization (ablation D)."""
    p = torch.softmax(logits, dim=1)
    return -entropy(p).mean()


def kl_divergence_with_temperature(logits_a, logits_b, T=0.5):
    pa = torch.log_softmax(logits_a / T, dim=1)
    qa = torch.softmax(logits_a / T, dim=1)
    return (qa * (pa - torch.log_softmax(logits_b / T, dim=1))).sum(dim=1).mean()


def consistency_loss(logits_w, logits_s, T=0.5, mode="symmetric"):
    kl_ws = kl_divergence_with_temperature(logits_w, logits_s, T=T)
    if mode == "one_way":
        return kl_ws
    kl_sw = kl_divergence_with_temperature(logits_s, logits_w, T=T)
    return 0.5 * (kl_ws + kl_sw)
