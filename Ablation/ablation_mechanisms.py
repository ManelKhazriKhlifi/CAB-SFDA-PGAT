"""
ablation_mechanisms.py 
Runs one ablation per mechanism block.
"""
import os, json
from config import CFG
from adapt import run_adaptation


class A: pass


def run(tag, **overrides):
    a = A()
    # defaults
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
    for k, v in overrides.items():
        setattr(a, k, v)
    best, _ = run_adaptation(a, tag=tag)
    print(f"[{tag}] best OA = {best:.2f}%")
    return best


if __name__ == "__main__":
    results = {}

    # (A) Class-balancing
    results["A_unweighted"] = run("A_unweighted", class_balancing="unweighted")
    results["A_inverse_freq"] = run("A_inverse_freq", class_balancing="inverse_freq")
    results["A_inverse_sqrt"] = run("A_inverse_sqrt", class_balancing="inverse_sqrt")

    # (B) Prototype memory
    results["B_no_proto"] = run("B_no_proto", proto_mode="none")
    results["B_static"] = run("B_static", proto_mode="static")
    results["B_batch"] = run("B_batch", proto_mode="batch")
    results["B_ema"] = run("B_ema", proto_mode="ema")
    results["B_ema_class_aware"] = run("B_ema_class_aware", proto_mode="ema_class_aware")

    # (C) Threshold
    results["C_fixed_090"] = run("C_fixed_090", threshold_mode="fixed_090")
    results["C_fixed_080"] = run("C_fixed_080", threshold_mode="fixed_080")
    results["C_class_agnostic"] = run("C_class_agnostic", threshold_mode="class_agnostic")
    results["C_no_lag"] = run("C_no_lag", threshold_mode="per_class_no_lag")
    results["C_no_warmup"] = run("C_no_warmup", threshold_mode="per_class_no_warmup")
    results["C_pgat_full"] = run("C_pgat_full", threshold_mode="pgat_full")

    # (D) Loss refinement
    results["D_one_way_kl"] = run("D_one_way_kl", loss_refinement="one_way_kl")
    results["D_standard_im"] = run("D_standard_im", loss_refinement="standard_im")
    results["D_symmetric_prior_im"] = run("D_symmetric_prior_im",
                                          loss_refinement="symmetric_prior_im")

    # (E) Classifier state
    results["E_trainable"] = run("E_trainable", classifier_state="trainable")
    results["E_frozen_bn_train"] = run("E_frozen_bn_train",
                                        classifier_state="frozen_bn_train")
    results["E_frozen_bn_frozen"] = run("E_frozen_bn_frozen",
                                        classifier_state="frozen_bn_frozen")

    with open(os.path.join(CFG.out_dir, "ablation_mechanisms.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(results)
