"""Backbone G (ResNet) + projection head + linear classifier H."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

_ARCHS = {
    "resnet18": (models.resnet18, "IMAGENET1K_V1", 512),
    "resnet34": (models.resnet34, "IMAGENET1K_V1", 512),
    "resnet50": (models.resnet50, "IMAGENET1K_V2", 2048),
    "resnet101": (models.resnet101, "IMAGENET1K_V2", 2048),
}


class Net(nn.Module):
    """features (ResNet up to global pooling) -> feature_head (Linear-BN-ReLU, 128-d) -> classifier."""

    def __init__(self, num_classes: int, arch: str = "resnet50", feat_dim: int = 128,
                 pretrained: bool = True):
        super().__init__()
        if arch not in _ARCHS:
            raise ValueError(f"arch must be one of {list(_ARCHS)}")
        ctor, weights, out_dim = _ARCHS[arch]
        bb = ctor(weights=weights if pretrained else None)
        self.features = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                      bb.layer1, bb.layer2, bb.layer3, bb.layer4, bb.avgpool)
        self.feature_head = nn.Sequential(nn.Linear(out_dim, feat_dim), nn.BatchNorm1d(feat_dim),
                                          nn.ReLU(inplace=True))
        self.classifier = nn.Linear(feat_dim, num_classes)
        nn.init.kaiming_normal_(self.feature_head[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.feature_head[0].bias)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.classifier.bias)

    def embed(self, x):
        return self.feature_head(self.features(x).flatten(1))

    def forward(self, x, return_feat: bool = False):
        f = self.embed(x)
        logits = self.classifier(f)
        return (logits, f) if return_feat else logits


def build_model(cfg, num_classes: int) -> Net:
    return Net(num_classes, arch=cfg.arch, feat_dim=cfg.feat_dim, pretrained=cfg.pretrained)


def adaptation_parameters(model: Net, freeze_classifier: bool):
    for p in model.classifier.parameters():
        p.requires_grad_(not freeze_classifier)
    return [p for p in model.parameters() if p.requires_grad]
