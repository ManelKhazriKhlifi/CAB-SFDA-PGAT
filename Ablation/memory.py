"""
memory.py
Class-aware momentum prototype memory with ablation-aware modes.
"""
import torch
import torch.nn.functional as F


class ClassMemory:
    def __init__(self, num_classes, feat_dim, momentum=0.9, eps=1e-6,
                 mode="ema_class_aware", device=None):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.eps = eps
        self.mode = mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prototypes = torch.zeros(num_classes, feat_dim, device=self.device)
        self.counts = torch.zeros(num_classes, device=self.device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def update(self, feats, labels):
        """Ablation-aware prototype update."""
        if self.mode == "none":
            return

        if self.mode == "ema":
            fmean = feats.mean(dim=0)
            for c in range(self.num_classes):
                if not self.initialized[c]:
                    self.prototypes[c] = fmean.detach()
                    self.initialized[c] = True
                else:
                    self.prototypes[c] = self.m * self.prototypes[c] + (1 - self.m) * fmean.detach()
            return

        for c in range(self.num_classes):
            idx = (labels == c).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            fmean = feats[idx].mean(dim=0)

            if self.mode == "static":
                if not self.initialized[c]:
                    self.prototypes[c] = fmean.detach()
                    self.initialized[c] = True
            elif self.mode == "batch":
                self.prototypes[c] = fmean.detach()
                self.initialized[c] = True
            elif self.mode == "ema_class_aware":
                if not self.initialized[c]:
                    self.prototypes[c] = fmean.detach()
                    self.initialized[c] = True
                else:
                    self.prototypes[c] = self.m * self.prototypes[c] + (1 - self.m) * fmean.detach()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self.counts[c] += idx.numel()

    def get_weights(self, mode="inverse_sqrt"):
        """Ablation-aware class weights."""
        if mode == "unweighted":
            return torch.ones(self.num_classes, device=self.device)
        if mode == "inverse_freq":
            inv = 1.0 / (self.counts + self.eps)
        elif mode == "inverse_sqrt":
            inv = 1.0 / torch.sqrt(self.counts + self.eps)
        else:
            raise ValueError(f"Unknown class balancing: {mode}")
        inv = inv / inv.sum().clamp_min(self.eps) * self.num_classes
        return inv.detach()

    def proto_loss(self, feats, labels):
        if feats.size(0) == 0:
            return torch.tensor(0.0, device=feats.device)
        valid = self.initialized[labels]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=feats.device)
        return F.mse_loss(feats[valid], self.prototypes[labels[valid]])
