import torch
import torch.nn.functional as F

from config import DEVICE


class ClassMemory:
    def __init__(self, num_classes, feat_dim, momentum=0.9, eps=1e-6):
        self.num_classes = num_classes
        self.feat_dim    = feat_dim
        self.m           = momentum
        self.eps         = eps
        self.prototypes  = torch.zeros(num_classes, feat_dim, device=DEVICE)
        self.counts      = torch.zeros(num_classes, device=DEVICE)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=DEVICE)

    @torch.no_grad()
    def update(self, feats, labels):
        for c in range(self.num_classes):
            idx = (labels == c).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            fmean = feats[idx].mean(dim=0)
            if not self.initialized[c]:
                self.prototypes[c] = fmean.detach()
                self.initialized[c] = True
            else:
                self.prototypes[c] = (self.m * self.prototypes[c]
                                      + (1.0 - self.m) * fmean.detach())
            self.counts[c] += idx.numel()

    def get_weights(self):
        inv = 1.0 / torch.sqrt(self.counts + self.eps)
        return (inv / inv.sum().clamp_min(self.eps) * self.num_classes).detach()

    def proto_loss(self, feats, labels):
        if feats.size(0) == 0:
            return torch.tensor(0.0, device=feats.device)
        valid = self.initialized[labels]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=feats.device)
        return F.mse_loss(feats[valid], self.prototypes[labels[valid]])