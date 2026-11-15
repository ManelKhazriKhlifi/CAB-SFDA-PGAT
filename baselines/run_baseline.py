import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

METHOD_NAMES = ["shot", "nrc", "cowa", "mbns", "gsfda", "aad", "dac", "smna"]
_ap = argparse.ArgumentParser()
_ap.add_argument("--method", required=True, choices=METHOD_NAMES)
ARGS = _ap.parse_args()

os.environ["METHOD"] = ARGS.method

from config import CFG                       
from dataloaders import build_loaders         
from train_source import get_or_train_source  
from evaluate import run_final_evaluation     
from run_logging import write_run_results     
from per_class_metrics import save_confusion_matrix, save_per_class_metrics  
import shot, nrc, cowa, mbns, gsfda, aad, dac, smna  


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
    args = ARGS
    print(f"[baseline] method={args.method}  task={CFG.task}  seed={CFG.seed}")
    print(f"[baseline] output dir = {CFG.out_dir}   source dir = {CFG.src_dir}")

    loaders  = build_loaders()
    src_ckpt = get_or_train_source(loaders)

    out = METHODS[args.method](src_ckpt, loaders, verbose=True)

    report = run_final_evaluation(out["final_ckpt"], loaders)
    save_confusion_matrix(report, loaders["class_names"], args.method)
    save_per_class_metrics(report, loaders["class_names"])

    summary_path = os.path.join(CFG.out_dir, f"seed_{CFG.seed}_all_results.json")
    write_run_results(CFG.out_dir, args.method, summary_path=summary_path)

    print(f"\n[{args.method}] overall_acc = {report['overall_acc']*100:.2f}%")
    print(f"[{args.method}] results: {CFG.out_dir}/results.json  |  "
          f"seed summary: {summary_path}")
    print(f"[{args.method}] output dir = {CFG.out_dir}/")


if __name__ == "__main__":
    main()