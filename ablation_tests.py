"""Statistical tests of the mechanism-wise ablation (Table 13): each variant of a group against the
default CAB-SFDA, with the same design as the baseline comparison of Table 17.

For every variant of the selected group(s):
  * paired t-test over the seeds of the per-seed benchmark means (cross-scene and cross-sensor),
    paired by seed (same source model per seed);
  * two-sided Wilcoxon signed-rank test over the 12 cross-scene task means (mean over seeds).
Every family of tests (one per column) is Holm-adjusted over the variants of the selected group(s).

Data source (one of):
  * the per-run result files of a results directory (full precision; default: results_dir of the
    configuration, override with --results-dir or CABSFDA_SET="results_dir=...");
  * paper_results/run_inventory.csv (--inventory; accuracies rounded to two decimals, which can
    create ties in the signed ranks).

Usage:
  python ablation_tests.py                       # group C (threshold mechanism), result files
  python ablation_tests.py --groups A B C D E F  # all groups (Holm over all their variants)
  python ablation_tests.py --inventory paper_results/run_inventory.csv
Outputs a table on stdout and, with --out, a CSV file.

Caveats: with three seeds, the t-tests have little power; the 12 cross-scene tasks share four
datasets, source models, and test sets, so the task-level tests are exploratory; a non-significant
difference does not show that two variants are equivalent.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cabsfda.experiments import MECHANISM_ABLATION, benchmark_tasks

DEFAULT_KEY = ("main", "cab_sfda", "full", "resnet50")


def load_frame(a) -> pd.DataFrame:
    """DataFrame with columns group, method, variant, arch, task, seed, oa."""
    if a.inventory:
        df = pd.read_csv(a.inventory).rename(columns={"adapted_oa": "oa"})
    else:
        from cabsfda.config import load_config
        from cabsfda.results import load_results
        cfg = load_config()
        rdir = a.results_dir or cfg.results_dir
        df, _ = load_results(rdir)
        if df.empty:
            raise SystemExit(f"no results found in {rdir}")
    df["seed"] = df["seed"].astype(int)
    return df[["group", "method", "variant", "arch", "task", "seed", "oa"]]


def runs(df, group, method, variant, arch):
    s = df[(df.group == group) & (df.method == method) & (df.variant == variant) & (df.arch == arch)]
    return s.drop_duplicates(["task", "seed"])


def per_seed_mean(sub, tasks):
    out = {}
    for seed, g in sub.groupby("seed"):
        g = g.set_index("task")
        if all(t in g.index for t in tasks):
            out[int(seed)] = float(np.mean([g.loc[t, "oa"] for t in tasks]))
    return out


def task_means(sub, tasks):
    m = sub.groupby("task").oa.mean()
    return [float(m[t]) if t in m.index else np.nan for t in tasks]


def holm(pvals):
    """Holm step-down adjustment; NaN entries are left out and stay NaN."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    adj, running = [np.nan] * len(pvals), 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - r) * pvals[i]))
        adj[i] = running
    return adj


def paired_t(a: dict, b: dict):
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return np.nan, np.nan, len(common)
    x, y = np.array([a[s] for s in common]), np.array([b[s] for s in common])
    d = x - y
    if np.ptp(d) <= 1e-6:                      # identical difference in every seed: t undefined
        p = 1.0 if abs(d.mean()) < 1e-6 else np.nan
    else:
        p = float(stats.ttest_rel(x, y).pvalue)
    return float(d.mean()), p, len(common)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["C"], help="ablation groups to test (A-F)")
    ap.add_argument("--inventory", help="paper_results/run_inventory.csv instead of the result files")
    ap.add_argument("--results-dir", help="results directory (default: results_dir of the configuration)")
    ap.add_argument("--out", help="write the table to this CSV file")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = load_frame(a)
    B = benchmark_tasks()
    scene, sensor = B["cross_scene"], B["cross_sensor"]
    ref = runs(df, *DEFAULT_KEY)
    if ref.empty:
        raise SystemExit("no results of the default CAB-SFDA (main/cab_sfda/full/resnet50)")
    ref_scene, ref_sensor = per_seed_mean(ref, scene), per_seed_mean(ref, sensor)
    ref_tasks = task_means(ref, scene)

    variants = [(v, lab) for v, (_, grp, lab) in MECHANISM_ABLATION.items() if grp in a.groups]
    rows = []
    for v, lab in variants:
        sub = runs(df, "ablation_mechanisms", "cab_sfda", v, "resnet50")
        if sub.empty:
            print(f"warning: no results for {v}")
            continue
        d_sc, p_sc, n_sc = paired_t(per_seed_mean(sub, scene), ref_scene)
        d_se, p_se, n_se = paired_t(per_seed_mean(sub, sensor), ref_sensor)
        xs, ys = task_means(sub, scene), ref_tasks
        ok = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
        if len(ok) >= 6 and any(abs(x - y) > 1e-9 for x, y in ok):
            w = stats.wilcoxon([x for x, _ in ok], [y for _, y in ok])
            p_w, n_w = float(w.pvalue), len(ok)
            wins = sum(x > y for x, y in ok)
        else:
            p_w, n_w, wins = np.nan, len(ok), np.nan
        rows.append({"variant": v, "label": lab,
                     "d_scene": d_sc, "p_scene": p_sc, "n_seeds_scene": n_sc,
                     "d_sensor": d_se, "p_sensor": p_se, "n_seeds_sensor": n_se,
                     "tasks_better": wins, "n_tasks": n_w, "p_wilcoxon": p_w})
    if not rows:
        raise SystemExit("no ablation results found for the selected groups")
    for col in ("p_scene", "p_sensor", "p_wilcoxon"):
        for r, p in zip(rows, holm([r[col] for r in rows])):
            r[col + "_holm"] = p

    def f(p):
        return "  n/a " if np.isnan(p) else ("<0.001" if p < 0.001 else f"{p:6.3f}")
    print("Variant vs. default CAB-SFDA (difference = variant - default, in pp; p: raw / Holm over "
          f"{len(rows)} variants)\n")
    print(f"{'variant':38s} {'d scene':>8s} {'p t-test':>15s} {'d sensor':>9s} {'p t-test':>15s} "
          f"{'tasks>':>7s} {'p Wilcoxon':>15s}")
    for r in rows:
        print(f"{r['label'][:38]:38s} {r['d_scene']:+8.2f} {f(r['p_scene'])}/{f(r['p_scene_holm'])} "
              f"{r['d_sensor']:+9.2f} {f(r['p_sensor'])}/{f(r['p_sensor_holm'])} "
              f"{'' if np.isnan(r['tasks_better']) else int(r['tasks_better']):>4}/{r['n_tasks']:<2} "
              f"{f(r['p_wilcoxon'])}/{f(r['p_wilcoxon_holm'])}")
    print("\nd scene / d sensor: difference of the benchmark means, averaged over the seeds; "
          "tasks>: cross-scene tasks on which the variant is more accurate.")
    if a.inventory:
        print("Data: run inventory (two-decimal accuracies); use the result files for full precision.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"written: {a.out}")


if __name__ == "__main__":
    main()
