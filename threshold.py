import torch
import torch.nn.functional as F

from config import DEVICE


class PrototypeGuidedClassThreshold:
    def __init__(self, num_classes, init=0.90, tau_min=0.70, tau_max=0.95,
                 momentum=0.90, min_samples=5, reliability_weight=0.50,
                 gamma=0.20, proto_min_count=20):
        self.num_classes        = num_classes
        self.tau_min            = tau_min
        self.tau_max            = tau_max
        self.momentum           = momentum
        self.min_samples        = min_samples
        self.reliability_weight = reliability_weight
        self.gamma              = gamma
        self.base_tau           = init
        self.proto_min_count    = proto_min_count

        self.thresholds        = torch.full((num_classes,), float(init),
                                            dtype=torch.float32, device=DEVICE)
        self.conf_reliability  = torch.zeros(num_classes, device=DEVICE)
        self.proto_reliability = torch.zeros(num_classes, device=DEVICE)
        self.joint_reliability = torch.zeros(num_classes, device=DEVICE)
        self.prototype_active  = torch.zeros(num_classes, dtype=torch.bool, device=DEVICE)

    @torch.no_grad()
    def update(self, conf, labels, feats, memory):
        for c in range(self.num_classes):
            idx = (labels == c)
            if idx.sum().item() < self.min_samples:
                continue

            q_c = conf[idx].mean().clamp(0.0, 1.0)
            self.conf_reliability[c] = q_c

            proto_ready = (bool(memory.initialized[c].item())
                           and float(memory.counts[c].item()) >= float(self.proto_min_count))
            self.prototype_active[c] = proto_ready

            if proto_ready:
                feats_c = F.normalize(feats[idx], p=2, dim=1)
                proto_c = F.normalize(memory.prototypes[c].unsqueeze(0), p=2, dim=1)
                cos = (feats_c * proto_c).sum(dim=1)
                r_c = ((1.0 + cos.mean()) / 2.0).clamp(0.0, 1.0)
            else:
                r_c = q_c

            self.proto_reliability[c] = r_c
            lam = self.reliability_weight
            R_c = (lam * q_c + (1.0 - lam) * r_c).clamp(0.0, 1.0)
            self.joint_reliability[c] = R_c

            raw_tau = (self.base_tau - self.gamma * (R_c - 0.5)).clamp(self.tau_min, self.tau_max)
            self.thresholds[c] = (self.momentum * self.thresholds[c]
                                  + (1.0 - self.momentum) * raw_tau)

    @torch.no_grad()
    def update_global(self, conf, labels, feats, memory):
        """Class-agnostic ablation: ONE threshold shared by all classes,
        driven by the same reliability formula pooled over the whole batch."""
        if conf.numel() < self.min_samples:
            return
        q = conf.mean().clamp(0.0, 1.0)
        ready = memory.initialized[labels] & (memory.counts[labels] >= float(self.proto_min_count))
        if ready.any():
            f_n = F.normalize(feats[ready], p=2, dim=1)
            p_n = F.normalize(memory.prototypes[labels[ready]], p=2, dim=1)
            r = ((1.0 + (f_n * p_n).sum(dim=1).mean()) / 2.0).clamp(0.0, 1.0)
        else:
            r = q
        lam = self.reliability_weight
        R = (lam * q + (1.0 - lam) * r).clamp(0.0, 1.0)
        raw_tau = (self.base_tau - self.gamma * (R - 0.5)).clamp(self.tau_min, self.tau_max)
        self.thresholds[:] = self.momentum * self.thresholds + (1.0 - self.momentum) * raw_tau
        self.conf_reliability[:] = q
        self.proto_reliability[:] = r
        self.joint_reliability[:] = R

    @torch.no_grad()
    def get_mask(self, conf, labels):
        return conf.ge(self.thresholds[labels])