"""
threshold.py
Prototype-Guided Class-Adaptive Threshold (PGAT) with ablation modes.
"""
import torch
import torch.nn.functional as F


class PrototypeGuidedClassThreshold:
    def __init__(self, num_classes, init=0.90, tau_min=0.70, tau_max=0.95,
                 momentum=0.90, min_samples=5, reliability_weight=0.50,
                 gamma=0.20, proto_min_count=20, mode="pgat_full", device=None):
        self.num_classes = num_classes
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.momentum = momentum
        self.min_samples = min_samples
        self.reliability_weight = reliability_weight
        self.gamma = gamma
        self.base_tau = init
        self.proto_min_count = proto_min_count
        self.mode = mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.thresholds = torch.full((num_classes,), float(init),
                                     dtype=torch.float32, device=self.device)
        self.conf_reliability = torch.zeros(num_classes, device=self.device)
        self.proto_reliability = torch.zeros(num_classes, device=self.device)
        self.joint_reliability = torch.zeros(num_classes, device=self.device)
        self.prototype_active = torch.zeros(num_classes, dtype=torch.bool,
                                            device=self.device)

    @torch.no_grad()
    def update(self, conf, labels, feats, memory):
        mode = self.mode

        if mode in ("fixed_090", "fixed_080"):
            return

        if mode == "class_agnostic":
            q = conf.mean().clamp(0.0, 1.0)
            if memory.initialized.any():
                f_n = F.normalize(feats, p=2, dim=1)
                p_n = F.normalize(memory.prototypes, p=2, dim=1)
                cos = (f_n.unsqueeze(1) * p_n.unsqueeze(0)).sum(-1)
                cos_pred = cos.gather(1, labels.unsqueeze(1)).squeeze(1)
                r = ((1.0 + cos_pred.mean()) / 2.0).clamp(0.0, 1.0)
            else:
                r = q
            R = (self.reliability_weight * q + (1 - self.reliability_weight) * r).clamp(0, 1)
            raw = (self.base_tau - self.gamma * (R - 0.5)).clamp(self.tau_min, self.tau_max)
            self.thresholds[:] = self.momentum * self.thresholds + (1 - self.momentum) * raw
            self.joint_reliability[:] = R
            self.conf_reliability[:] = q
            self.proto_reliability[:] = r
            return

        for c in range(self.num_classes):
            idx = (labels == c)
            if idx.sum().item() < self.min_samples:
                continue

            q_c = conf[idx].mean().clamp(0.0, 1.0)
            self.conf_reliability[c] = q_c

            if mode == "per_class_no_warmup":
                proto_ready = bool(memory.initialized[c].item())
            else:
                proto_ready = (
                    bool(memory.initialized[c].item())
                    and float(memory.counts[c].item()) >= float(self.proto_min_count)
                )
            self.prototype_active[c] = proto_ready

            if proto_ready:
                fc = F.normalize(feats[idx], p=2, dim=1)
                pc = F.normalize(memory.prototypes[c].unsqueeze(0), p=2, dim=1)
                cos = (fc * pc).sum(dim=1)
                r_c = ((1.0 + cos.mean()) / 2.0).clamp(0.0, 1.0)
            else:
                r_c = q_c

            self.proto_reliability[c] = r_c

            lam = self.reliability_weight
            R_c = (lam * q_c + (1 - lam) * r_c).clamp(0.0, 1.0)
            self.joint_reliability[c] = R_c

            raw = (self.base_tau - self.gamma * (R_c - 0.5)).clamp(self.tau_min, self.tau_max)
            self.thresholds[c] = self.momentum * self.thresholds[c] + (1 - self.momentum) * raw

    @torch.no_grad()
    def get_mask(self, conf, labels):
        return conf.ge(self.thresholds[labels])
