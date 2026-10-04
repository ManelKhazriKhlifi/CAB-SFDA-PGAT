"""
ablation_backbone.py
ResNet-18 vs ResNet-50 vs ResNet-101.
"""
import os, json
from config import CFG
from adapt import run_adaptation


class A: pass


def run(tag, backbone):
    CFG.backbone = backbone
    a = A()
    a.disable_pl = False
    a.disable_im = False
    a.disable_proto = False
    a.disable_cons = False
    a.class_balancing = "inverse_sqrt"
    a.proto_mode = "ema_class_aware"
    a.threshold_mode = "pgat_full"
    a.loss_refinement = "symmetric_prior_im"
    a.classifier_state = "frozen_bn_frozen"
    a.imbalance_ratio = None
    best, _ = run_adaptation(a, tag=tag)
    print(f"[{tag}] best OA = {best:.2f}%")
    return best


if __name__ == "__main__":
    results = {}
    for bb in ["resnet18", "resnet50", "resnet101"]:
        results[bb] = run(f"backbone_{bb}", bb)
    with open(os.path.join(CFG.out_dir, "ablation_backbone.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(results)
