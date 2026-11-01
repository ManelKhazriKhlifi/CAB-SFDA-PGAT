import os
import json

from config import CFG
from dataloaders import build_loaders
from train_source import get_or_train_source
from adapt import run_adaptation
from evaluate import run_final_evaluation


def run(loaders):
    src_ckpt = get_or_train_source(loaders)

    configs = {
        "full":         dict(im=CFG.lambda_im, cons=CFG.lambda_cons,
                             pl=CFG.lambda_pl, proto=CFG.lambda_proto),
        "w/o L_pl":     dict(im=CFG.lambda_im, cons=CFG.lambda_cons,
                             pl=0.0,          proto=CFG.lambda_proto),
        "w/o L_im":     dict(im=0.0,          cons=CFG.lambda_cons,
                             pl=CFG.lambda_pl, proto=CFG.lambda_proto),
        "w/o L_proto":  dict(im=CFG.lambda_im, cons=CFG.lambda_cons,
                             pl=CFG.lambda_pl, proto=0.0),
        "w/o L_cons":   dict(im=CFG.lambda_im, cons=0.0,
                             pl=CFG.lambda_pl, proto=CFG.lambda_proto),
    }

    root = os.path.join(CFG.out_dir, "ablation_loss")
    results = {}
    for name, kw in configs.items():
        print(f"\n=== {name} ===")
        # "w/o L_pl" → "wo_L_pl" (a "/" would create a sub-folder)
        var_dir = os.path.join(root, name.replace("w/o ", "wo_").replace(" ", "_"))
        orig = (CFG.lambda_im, CFG.lambda_cons, CFG.lambda_pl, CFG.lambda_proto)
        CFG.lambda_im, CFG.lambda_cons, CFG.lambda_pl, CFG.lambda_proto = (
            kw["im"], kw["cons"], kw["pl"], kw["proto"]
        )
        try:
            out = run_adaptation(src_ckpt, loaders, verbose=True, out_dir=var_dir)
            report = run_final_evaluation(out["final_ckpt"], loaders,
                                          out_dir=var_dir, method=name)
            results[name] = {"final_acc": float(report["overall_acc"] * 100)}
        finally:
            (CFG.lambda_im, CFG.lambda_cons, CFG.lambda_pl, CFG.lambda_proto) = orig

    path = os.path.join(root, "ablation_loss.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", path)


if __name__ == "__main__":
    run(build_loaders())