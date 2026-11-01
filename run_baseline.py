import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import CFG                       # noqa: E402
from dataloaders import build_loaders         # noqa: E402
from train_source import get_or_train_source  # noqa: E402
from evaluate import run_final_evaluation     # noqa: E402

from baselines import shot, nrc, cowa, mbns, gsfda, aad, dac, smna  # noqa: E402


METHODS = {
    "shot":  shot.run,
    "nrc":   nrc.run,
    "cowa":  cowa.run,
    "mbns":  mbns.run,
    "gsfda": gsfda.run,
    "aad":   aad.run,
    "dac":   dac.run,
    "smna":  smna.run,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS.keys()))
    args = ap.parse_args()

    print(f"[baseline] method={args.method}  task={CFG.task}  seed={CFG.seed}")

    loaders  = build_loaders()
    src_ckpt = get_or_train_source(loaders)

    out = METHODS[args.method](src_ckpt, loaders, verbose=True)

    # Evaluate on the target test set (single, final-epoch)
    report = run_final_evaluation(out["final_ckpt"], loaders,
                                  out_dir=os.path.join(CFG.out_dir, args.method),
                                  method=args.method)

    print(f"\n[{args.method}] overall_acc = {report['overall_acc']*100:.2f}%")
    print(f"[{args.method}] output dir = {CFG.out_dir}/{args.method}/")


if __name__ == "__main__":
    main()