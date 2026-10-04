"""
ablation_losses.py 
Removes one loss term at a time.
"""
from config import CFG
from adapt import run_adaptation
from evaluate import evaluate_with_cm
from dataloaders import build_loaders
import torch, os, json
from model import ResNetBackbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class A: pass


def run_config(name, **flags):
    a = A()
    a.disable_pl = flags.get("disable_pl", False)
    a.disable_im = flags.get("disable_im", False)
    a.disable_proto = flags.get("disable_proto", False)
    a.disable_cons = flags.get("disable_cons", False)
    a.class_balancing = "inverse_sqrt"
    a.proto_mode = "ema_class_aware"
    a.threshold_mode = "pgat_full"
    a.loss_refinement = "symmetric_prior_im"
    a.classifier_state = "frozen_bn_frozen"
    a.imbalance_ratio = None

    best, _ = run_adaptation(a, tag=name)
    print(f"[{name}] best OA = {best:.2f}%")

    # Evaluate using the final student
    _, _, _, tgt_test_loader, _ = build_loaders()
    ckpt = os.path.join(CFG.out_dir, "source_aid_resnet50.pth")   # dummy to avoid warning
    return best


if __name__ == "__main__":
    results = {}
    results["full"] = run_config("full")
    results["w/o_Lpl"] = run_config("w/o_Lpl", disable_pl=True)
    results["w/o_Lim"] = run_config("w/o_Lim", disable_im=True)
    results["w/o_Lproto"] = run_config("w/o_Lproto", disable_proto=True)
    results["w/o_Lcons"] = run_config("w/o_Lcons", disable_cons=True)

    with open(os.path.join(CFG.out_dir, "ablation_losses.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(results)
