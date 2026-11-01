import os
import json
import torch

from config import CFG, DEVICE
from model import ResNetBackbone, accuracy
from train_source import train_source
from adapt import run_adaptation
from evaluate import run_final_evaluation


def run(loaders):
    root = os.path.join(CFG.out_dir, "ablation_backbone")
    results = {}
    for arch in ["resnet18", "resnet50", "resnet101"]:
        print(f"\n=== Backbone: {arch} ===")
        arch_dir = os.path.join(root, arch)
        os.makedirs(arch_dir, exist_ok=True)

        ckpt = os.path.join(arch_dir, f"source_{arch}.pth")
        model = ResNetBackbone(num_classes=CFG.num_classes,
                               feat_dim=CFG.feat_dim,
                               arch=arch).to(DEVICE)
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            # own log_path: must not overwrite the main source_training_results.json
            model = train_source(model, loaders["src_train"], loaders["src_test"],
                                 log_path=os.path.join(arch_dir,
                                                       "source_training_results.json"))
            torch.save(model.state_dict(), ckpt)
        src_only = accuracy(model, loaders["tgt_test"])

        out = run_adaptation(ckpt, loaders, arch=arch, verbose=True,
                             out_dir=arch_dir)
        report = run_final_evaluation(out["final_ckpt"], loaders, arch=arch,
                                      out_dir=arch_dir, method=f"cab_sfda_{arch}")

        results[arch] = {
            "source_only_acc": float(src_only),
            "adapted_acc":     float(report["overall_acc"] * 100),
        }

    path = os.path.join(root, "ablation_backbone.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", path)


if __name__ == "__main__":
    from dataloaders import build_loaders
    run(build_loaders())