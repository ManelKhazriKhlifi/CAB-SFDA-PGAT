import torch
import torch.nn as nn
from torchvision import models

from config import CFG, DEVICE


class ResNetBackbone(nn.Module):
    _ARCHS = {
        "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1, 512),
        "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1, 512),
        "resnet50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
    }

    def __init__(self, num_classes=7, feat_dim=128, in_channels=3,
                 use_pretrained=True, arch="resnet50"):
        super().__init__()
        ctor, weights, proj_dim = self._ARCHS[arch]
        bb = ctor(weights=weights if use_pretrained else None)

        self.features = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4, bb.avgpool,
        )
        if in_channels != 3:
            self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7,
                                         stride=2, padding=3, bias=False)
        self.proj_dim = proj_dim
        self.feature_head = nn.Sequential(
            nn.Linear(proj_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

        nn.init.kaiming_normal_(self.feature_head[0].weight,
                                mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.feature_head[0].bias, 0)
        nn.init.kaiming_normal_(self.classifier.weight,
                                mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, return_feat=False):
        f = self.features(x).flatten(1)
        f = self.feature_head(f)
        logits = self.classifier(f)
        return (logits, f) if return_feat else logits


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(1, total)