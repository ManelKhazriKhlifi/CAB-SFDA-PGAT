

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_HERE) == "Ablation":
    sys.path.insert(0, _HERE)
else:
    sys.path.insert(0, os.path.join(_HERE, "Ablation"))

from config import CFG
from dataloaders import build_loaders
from train_source import get_or_train_source
from adapt import run_adaptation
from evaluate import run_final_evaluation
from per_class_metrics import save_confusion_and_metrics
from diagnostics_pgat import plot_threshold_trajectories, run_tsne


def main():
    print(f"Task={CFG.task}  {CFG.source_name}→{CFG.target_name}  "
          f"classes={CFG.num_classes}  seed={CFG.seed}")
    out_dir = os.path.join(CFG.out_dir, "cab_sfda")
    print(f"Protocol: src={CFG.src_epochs}, sf={CFG.sf_epochs}, "
          f"final-epoch ckpt, no early stop, freeze_bn={CFG.freeze_bn}")
    print(f"Output:  {out_dir}")

    print(">>> [1/6] Building data loaders ...")
    loaders = build_loaders()

    print(">>> [2/6] Training / reusing source model ...")
    src_ckpt = get_or_train_source(loaders)

    print(">>> [3/6] Running source-free adaptation ...")
    out = run_adaptation(src_ckpt, loaders, verbose=True, out_dir=out_dir)

    print(">>> [4/6] Evaluating final model on target test ...")
    report = run_final_evaluation(out["final_ckpt"], loaders,
                                  out_dir=out_dir, method="cab_sfda")

    print(">>> [5/6] Saving confusion matrix + per-class metrics ...")
    save_confusion_and_metrics(report, loaders["class_names"], out_dir=out_dir)

    print(">>> [6/6] Saving threshold trajectories + t-SNE plots ...")
    plot_threshold_trajectories(
        out["threshold_history"],
        save_path=os.path.join(out_dir, "class_threshold_trajectories.pdf"),
    )
    run_tsne(src_ckpt, out["final_ckpt"], loaders, max_samples=2000,
             out_dir=out_dir)

    print(f"\nAll outputs written to {out_dir}")


if __name__ == "__main__":
    main()
