"""
model.py
ResNet backbone with a projection head for CAB-SFDA.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


def build_backbone(name):
    if name == "resnet18":
        return tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V2)
    if name == "resnet50":
        return tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    if name == "resnet101":
        return tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V2)
    raise ValueError(f"Unknown backbone: {name}")


class ResNetBackbone(nn.Module):
    def __init__(self, num_classes=7, feat_dim=128, in_channels=3,
                 use_pretrained=True, backbone_name="resnet50"):
        super().__init__()
        self.backbone = build_backbone(backbone_name) if use_pretrained \
                        else build_backbone(backbone_name)

        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.backbone.avgpool,
        )

        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                            stride=2, padding=3, bias=False)

        self.proj_dim = self.backbone.fc.in_features
        self.feature_head = nn.Sequential(
            nn.Linear(self.proj_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

        nn.init.kaiming_normal_(self.feature_head[0].weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.constant_(self.feature_head[0].bias, 0)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out',
                                nonlinearity='relu')
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, return_feat=False):
        f_map = self.features(x)
        f_map = f_map.view(f_map.size(0), -1)
        f = self.feature_head(f_map)
        logits = self.classifier(f)
        if return_feat:
            return logits, f
        return logits
