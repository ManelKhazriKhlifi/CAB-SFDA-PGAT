
import torch


def entropy(p, eps=1e-8):
    return -(p * (p + eps).log()).sum(dim=1)


def info_max_loss(logits):
    p = torch.softmax(logits, dim=1)
    return entropy(p).mean() - entropy(p.mean(dim=0, keepdim=True)).mean()


def entropy_min_loss(logits):
    p = torch.softmax(logits, dim=1)
    return entropy(p).mean()


def kl_with_temperature(logits_a, logits_b, T=0.5):
    pa = torch.log_softmax(logits_a / T, dim=1)
    qa = torch.softmax(logits_a / T, dim=1)
    return (qa * (pa - torch.log_softmax(logits_b / T, dim=1))).sum(dim=1).mean()