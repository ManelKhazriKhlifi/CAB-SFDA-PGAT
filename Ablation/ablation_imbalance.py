"""
ablation_imbalance.py
Controlled imbalance ratios on the MLRSN target domain.
"""
import os, json
from config import CFG
from adapt import run_adaptation


class A: pass


def run(tag, ratio):
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
    a.imbalance_ratio = ratio
    best, _ = run_adaptation(a, tag=tag)
    print(f"[{tag}] best OA = {best:.2f}%")
    return best


if __name__ == "__main__":
    results = {}
    for r in [1.0, 1.5, 2.0, 3.0, 5.0]:
        results[f"ratio_{r}"] = run(f"ratio_{r}", r)
    with open(os.path.join(CFG.out_dir, "ablation_imbalance.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(results)
