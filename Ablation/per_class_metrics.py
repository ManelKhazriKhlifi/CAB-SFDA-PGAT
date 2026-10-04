"""
per_class_metrics.py
Load the best adapted checkpoint and print per-class metrics.
"""
import os, json
import numpy as np
import torch

from config import CFG
from model import ResNetBackbone
from dataloaders import build_loaders
from evaluate import evaluate_with_cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    _, _, _, tgt_test_loader, _ = build_loaders()
    ckpt = os.path.join(CFG.out_dir, "source_aid_resnet50.pth")   # source
    model = ResNetBackbone(num_classes=CFG.num_classes,
                           feat_dim=CFG.feat_dim,
                           backbone_name=CFG.backbone).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    res = evaluate_with_cm(model, tgt_test_loader)
    class_names = tgt_test_loader.dataset.classes

    print(f"{'Class':<14}{'Precision':>11}{'Recall':>11}{'F1':>11}")
    for i, name in enumerate(class_names):
        print(f"{name:<14}{100*res['per_class_precision'][i]:>10.2f}%"
              f"{100*res['per_class_recall'][i]:>10.2f}%"
              f"{100*res['per_class_f1'][i]:>10.2f}%")
    print(f"{'MA':<14}{100*res['macro_precision']:>10.2f}%"
          f"{100*res['macro_recall']:>10.2f}%"
          f"{100*res['macro_f1']:>10.2f}%")

    with open(os.path.join(CFG.out_dir, "per_class_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
