"""One adaptation run: data -> source model -> adaptation -> final-epoch evaluation -> result.json."""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import cab_sfda
from .baselines import BASELINES
from .config import task_info
from .data import TaskData
from .metrics import evaluate, predict, unsupervised_scores
from .source import load_source_model, source_dir, source_only_eval, train_source
from .utils import append_jsonl, environment_info, load_json, log, save_json, seed_everything

METHODS = {"cab_sfda": cab_sfda, **BASELINES}


def n_strong(method):
    return 1 if method == "cab_sfda" else METHODS[method].N_STRONG


def run_one(cfg, task: str, seed: int, method: str, run_dir: Path, meta: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    if result_path.exists():
        log(f"skip (done): {run_dir}")
        return load_json(result_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "log.jsonl").unlink(missing_ok=True)
    save_json({"task": task, "seed": seed, "method": method, "meta": meta or {}, "config": dict(cfg)},
              run_dir / "config.json")
    info = task_info(task, cfg)
    if method not in METHODS:
        raise KeyError(f"unknown method {method}; choose from {list(METHODS)}")

    t0 = time.time()
    try:
        train_source(cfg, info["source"], seed)             # cached and locked
        seed_everything(seed, cfg.deterministic)
        data = TaskData(cfg, info, seed, n_strong=n_strong(method))
        src_metrics = source_only_eval(cfg, info, seed, data.test, save_feats=cfg.save_features)
        model = load_source_model(cfg, info, seed)
        seed_everything(seed, cfg.deterministic)            # identical RNG state for every method
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_adapt = time.time()
        out = METHODS[method].adapt(model, data, cfg, info, lambda rec: append_jsonl(rec, run_dir / "log.jsonl"))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compute = {
            # whole adaptation (incl. per-epoch clustering / memory banks) divided by the epochs
            "adapt_seconds": time.time() - t_adapt,
            "seconds_per_epoch": (time.time() - t_adapt) / max(1, cfg.epochs),
            "peak_reserved_mib": (torch.cuda.max_memory_reserved() / 2 ** 20) if torch.cuda.is_available() else None,
            "peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
            # timings are only comparable between runs that had a GPU to themselves
            "jobs_per_gpu": int(os.environ.get("CABSFDA_JOBS_PER_GPU", "0")) or None,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        final, raw = evaluate(model, data.test, info["num_classes"], return_raw=True,
                              return_feats=cfg.save_features)
        # label-free scores on the unlabeled adaptation images (model selection, collapse check)
        unsup = unsupervised_scores(predict(model, data.adapt_eval)["logits"])
    except Exception:
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    if cfg.save_predictions:
        np.savez_compressed(run_dir / "predictions.npz", logits=raw["logits"].astype(np.float16),
                            labels=raw["labels"])
    if cfg.save_features:
        np.savez_compressed(run_dir / "features.npz", feats=raw["feats"].astype(np.float16),
                            labels=raw["labels"])
    result = {
        **{k: info[k] for k in ("task", "source", "target", "benchmark", "classes", "num_classes")},
        "seed": seed, "method": method, "arch": cfg.arch, "meta": meta or {},
        "n_adapt": data.n_adapt, "n_test": data.n_test,
        "adapt_class_counts": np.bincount(data.adapt_labels, minlength=info["num_classes"]).tolist(),
        "test_class_counts": np.bincount(data.test_labels, minlength=info["num_classes"]).tolist(),
        "imbalanced_subset": ({"ratio": cfg.imb_ratio, "base": cfg.imb_base, "counts": data.subset_counts,
                               "ranking": data.subset_ranking} if data.subset_counts is not None else None),
        "source_model": str(source_dir(cfg, info["source"], seed)),
        "source_only": src_metrics,
        "final": final,                       # model after the LAST adaptation epoch
        "unsupervised": unsup,                # SND etc. on the adaptation images (no labels)
        "diagnostics": out.get("diagnostics", {}),
        "history": out.get("history", []),
        "wall_time_s": time.time() - t0,
        "compute": compute,
        "env": environment_info(),
        "config": dict(cfg),
    }
    save_json(result, result_path)
    (run_dir / "error.txt").unlink(missing_ok=True)
    log(f"done {method} {task} seed{seed}: source-only {src_metrics['oa']:.2f} -> adapted {final['oa']:.2f} "
        f"({time.time() - t0:.0f}s)")
    return result
