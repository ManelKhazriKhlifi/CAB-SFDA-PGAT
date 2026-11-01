"""Shared helpers used by all baseline implementations."""
import os
import json
import torch

from config import CFG, DEVICE
from model import ResNetBackbone


def load_student(src_ckpt_path, arch=None, freeze_classifier=True):
    """Load a fresh ResNetBackbone from a source checkpoint."""
    arch = arch or CFG.backbone
    m = ResNetBackbone(num_classes=CFG.num_classes,
                       feat_dim=CFG.feat_dim,
                       arch=arch).to(DEVICE)
    m.load_state_dict(torch.load(src_ckpt_path, map_location=DEVICE))
    for p in m.classifier.parameters():
        p.requires_grad_(not freeze_classifier)
    return m


def make_optimizer(model, lr):
    """Adam over all currently-trainable parameters."""
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)


def save_results(model, name, accept_history, extra=None):
    """Save final-epoch checkpoint + JSON log. Return dict for caller."""
    method_dir = os.path.join(CFG.out_dir, name)
    os.makedirs(method_dir, exist_ok=True)

    ckpt = os.path.join(method_dir, "student_final.pth")
    torch.save(model.state_dict(), ckpt)

    log = {
        "method": name,
        "task": CFG.task,
        "seed": CFG.seed,
        "protocol": {
            "src_epochs": CFG.src_epochs,
            "sf_epochs":  CFG.sf_epochs,
            "checkpoint": "final_epoch",
        },
        "accept_history": accept_history,
    }
    if extra:
        log.update(extra)

    with open(os.path.join(method_dir, "adaptation_results.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"[{name}] Saved final-epoch model: {ckpt}")
    return {"final_ckpt": ckpt, "model": model, "accept_history": accept_history}