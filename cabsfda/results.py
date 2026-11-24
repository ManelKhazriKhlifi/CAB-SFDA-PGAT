"""Loading run results and computing seed-level aggregates (shared by aggregate.py, make_figures.py,
check_results.py).

Benchmark averages are always computed the same way for every method: average over the tasks
of the benchmark within each seed, then mean and sample std (ddof=1) over seeds. A benchmark
average is only reported when every task is present for every seed used.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from .config import resolve_path
from .utils import load_json


def load_results(results_dir) -> tuple[pd.DataFrame, dict]:
    """Returns (dataframe with one row per run, {run_dir: full result dict})."""
    root = resolve_path(results_dir) / "runs"
    rows, full = [], {}
    for f in sorted(root.glob("*/*/*/*/*/seed*/result.json")):
        r = load_json(f)
        rel = f.parent.relative_to(root).parts
        group, method, variant, arch, task = rel[:5]
        d = str(f.parent)
        full[d] = r
        pc = r["final"]["per_class"]
        rows.append({"group": group, "method": method, "variant": variant, "arch": arch, "task": task,
                     "seed": int(r["seed"]), "benchmark": r["benchmark"], "oa": r["final"]["oa"],
                     "bacc": r["final"]["balanced_acc"], "mf1": r["final"]["macro_f1"],
                     "src_oa": r["source_only"]["oa"], "run_dir": d,
                     "f1": pc["f1"], "recall": pc["recall"], "precision": pc["precision"],
                     "src_f1": r["source_only"]["per_class"]["f1"]})
    df = pd.DataFrame(rows)
    return df, full


def select(df, group=None, method=None, variant=None, arch=None, task=None):
    m = pd.Series(True, index=df.index)
    for col, val in (("group", group), ("method", method), ("variant", variant), ("arch", arch), ("task", task)):
        if val is not None:
            m &= df[col].isin(val) if isinstance(val, (list, tuple, set)) else df[col] == val
    return df[m]


def stat(values):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return {"mean": np.nan, "std": np.nan, "n": 0, "values": []}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v),
            "values": v.tolist()}


def task_stat(sub, task, col="oa"):
    s = sub[sub.task == task].sort_values("seed")
    out = stat(s[col].tolist())
    out["seeds"] = s.seed.tolist()
    out["dirs"] = s.run_dir.tolist()
    return out


def per_seed_bench(sub, tasks, col="oa", seeds=None):
    """{seed: average over `tasks`} for seeds where all tasks exist."""
    out = {}
    for seed, g in sub.groupby("seed"):
        if seeds is not None and seed not in seeds:
            continue
        g = g.drop_duplicates("task").set_index("task")
        if all(t in g.index for t in tasks):
            out[int(seed)] = float(np.mean([g.loc[t, col] for t in tasks]))
    return out


def bench_stat(sub, tasks, col="oa"):
    ps = per_seed_bench(sub, tasks, col)
    out = stat(list(ps.values()))
    out["seeds"] = list(ps)
    out["dirs"] = sub[sub.task.isin(tasks) & sub.seed.isin(list(ps))].run_dir.tolist()
    n_seeds_any = sub.seed.nunique()
    out["complete"] = len(ps) == n_seeds_any and len(ps) > 0
    return out


def class_value(sub, task, cls_idx, col="f1"):
    s = sub[sub.task == task].sort_values("seed")
    out = stat([v[cls_idx] for v in s[col]])
    out["dirs"] = s.run_dir.tolist()
    return out


def fmt(st, digits=2, std=True):
    if st is None or st.get("n", 0) == 0 or np.isnan(st["mean"]):
        return "--"
    digits = st.get("digits", digits)
    if std and st.get("n", 0) > 1:          # no "± 0.00" for a single run
        return f"{st['mean']:.{digits}f} ± {st['std']:.{digits}f}"
    return f"{st['mean']:.{digits}f}"
