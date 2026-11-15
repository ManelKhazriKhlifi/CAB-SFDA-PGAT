
import csv
import hashlib
import json
import os
import time

import torch

from config import CFG


CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "0"))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class RunLogger:
    

    def __init__(self, name, init_ckpt_path=None, hparams=None, run_dir=None):
        self.name = name
        self.dir = run_dir or CFG.out_dir
        os.makedirs(self.dir, exist_ok=True)

        self._csv = open(os.path.join(self.dir, "train_log.csv"), "w", newline="")
        self._txt = open(os.path.join(self.dir, "train.log"), "w")
        self._writer = None
        self._sums, self._n = {}, 0
        self.epochs = []
        self._t0 = time.time()

    def step(self, epoch, **terms):
        for k, v in terms.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v)
        self._n += 1

    def end_epoch(self, epoch, accept_rate=None, model=None, **metrics):
        means = {k: s / max(self._n, 1) for k, s in self._sums.items()}
        rec = {"epoch": epoch, "elapsed_s": round(time.time() - self._t0, 1), **means}
        if accept_rate is not None:
            rec["accept_rate"] = accept_rate
        rec.update({k: float(v) for k, v in metrics.items()})

        if model is not None and CKPT_EVERY and epoch % CKPT_EVERY == 0:
            fn = f"ckpt_ep{epoch:03d}.pth"
            path = os.path.join(self.dir, fn)
            torch.save(model.state_dict(), path)
            rec["checkpoint"] = {"file": fn, "sha256": sha256(path)}

        self.epochs.append(rec)
        self._sums, self._n = {}, 0

        row = {k: v for k, v in rec.items() if k != "checkpoint"}
        if "checkpoint" in rec:
            row["checkpoint"] = rec["checkpoint"]["file"]
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

        shown = {k: v for k, v in rec.items()
                 if k not in ("epoch", "elapsed_s", "checkpoint")}
        terms = "  ".join(f"{k}={v:.4f}" for k, v in shown.items())
        self._txt.write(f"[{self.name}] {CFG.task} s{CFG.seed} ep {epoch}  "
                        f"{terms}  t={rec['elapsed_s']}s\n")
        self._txt.flush()
        self._csv.flush()

    def close(self):
        self._csv.close()
        self._txt.close()
        with open(os.path.join(self.dir, "loss_curves.json"), "w") as f:
            json.dump(self.epochs, f, indent=2)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_run_results(run_dir, method, summary_path=None):
    """Write <run_dir>/results.json (final metrics + per-epoch adaptation history) and
    add/replace this run's entry in the summary file covering all runs."""
    ev = _read_json(os.path.join(run_dir, "final_evaluation.json")) or {}
    metrics = ev.get("metrics", {})

    results = {
        "method": method,
        "task": CFG.task,
        "seed": CFG.seed,
        "source": ev.get("source", CFG.source_name),
        "target": ev.get("target", CFG.target_name),
        "final": {k: metrics.get(k) for k in ("overall_acc", "balanced_acc", "macro_f1",
                                               "per_class_recall", "per_class_precision",
                                               "per_class_f1", "cm")},
        "adaptation_epochs": _read_json(os.path.join(run_dir, "loss_curves.json")) or [],
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if summary_path:
        summary = _read_json(summary_path) or {}
        key = f"{results['task']}/{method}/seed_{results['seed']}"
        summary[key] = {
            "method": method, "task": results["task"], "seed": results["seed"],
            "overall_acc": metrics.get("overall_acc"),
            "balanced_acc": metrics.get("balanced_acc"),
            "macro_f1": metrics.get("macro_f1"),
            "epochs": results["adaptation_epochs"],
        }
        tmp = summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(sorted(summary.items())), f, indent=2)
        os.replace(tmp, summary_path)
    return results
