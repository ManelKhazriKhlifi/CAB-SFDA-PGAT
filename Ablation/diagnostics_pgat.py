"""
diagnostics_pgat.py
Run after a normal adaptation.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

from config import CFG
from adapt import run_adaptation


class A: pass


def get_diag(epoch_results):
    final = epoch_results[-1]
    return {
        "OA": final["target_test_acc"],
        "tau_bar": float(np.mean(final["class_thresholds"])),
        "q_bar": float(np.mean(final["confidence_reliability"])),
        "r_bar": float(np.mean(final["prototype_reliability"])),
        "R_bar": float(np.mean(final["joint_reliability"])),
        "sigma_counts": int(np.sum(final["prototype_counts"])),
        "acc_rate": final["pseudo_label_acceptance_rate"],
        "n_active": int(np.sum(final["prototype_active"])),
    }


def plot_class_weight_evolution(epoch_results, out_dir, tag):
    weights = []
    for rec in epoch_results:
        counts = np.array(rec["prototype_counts"], dtype=np.float64)
        inv = 1.0 / np.sqrt(counts + 1e-6)
        inv = inv / inv.sum() * len(counts)
        weights.append(inv)
    weights = np.array(weights)

    plt.figure(figsize=(10, 6))
    for c in range(weights.shape[1]):
        plt.plot(np.arange(1, len(weights) + 1), weights[:, c],
                 marker="o", label=f"Class {c}")
    plt.axhline(1.0, ls="--", c="k", alpha=0.5)
    plt.xlabel("Epoch"); plt.ylabel("Class weight $w_c$")
    plt.title(f"Evolution of class weights — {tag}")
    plt.grid(alpha=0.3); plt.legend(ncol=2, fontsize=9)
    out = os.path.join(out_dir, f"class_weight_evolution_{tag}.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print("Saved:", out)


if __name__ == "__main__":
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

    _, epoch_results = run_adaptation(a, tag="pgat_full")

    diag = get_diag(epoch_results)
    print(json.dumps(diag, indent=2))
    with open(os.path.join(CFG.out_dir, "pgat_diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    plot_class_weight_evolution(epoch_results, CFG.out_dir, "pgat_full")
