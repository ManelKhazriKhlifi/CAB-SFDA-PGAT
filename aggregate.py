#!/usr/bin/env python
"""Build every table of the paper from the run logs.

  python aggregate.py                 # -> tables/*.tex, tables/*.csv, tables/summary.md

Every number is computed from results/runs/**/result.json; tables/provenance.csv lists the
run folders behind each cell and tables/missing.txt lists cells without data ('--').
Benchmark averages: mean over tasks within each seed, then mean +- sample std over seeds,
for every method alike.
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

from cabsfda.baselines import BASELINES, DISPLAY_NAMES
from cabsfda.config import REPO_ROOT, load_config, load_datasets
from cabsfda.experiments import (ALL_BACKBONES, CROSS_SCENE_MLRSN_TARGETS, IMB_RATIOS, MECHANISM_ABLATION, MECHANISM_OURS, SENSITIVITY, THRESHOLD_RANGE,
                                 benchmark_tasks, _fmt)
from cabsfda.results import (bench_stat, class_value, fmt, load_results, per_seed_bench, select, stat,
                             task_stat)

BASE = list(BASELINES)
_TEX = {"±": "$\\pm$", "Σ": "$\\Sigma$", "τ": "$\\tau$", "q̄": "$\\bar{q}$", "r̄": "$\\bar{r}$",
        "R̄": "$\\bar{R}$", "−": "$-$", "_": "\\_", "%": "\\%", "&": "\\&"}


def tex(text: str) -> str:
    for k, v in _TEX.items():
        text = text.replace(k, v)
    return text


class Table:
    def __init__(self, name, caption, header, higher_is_better=True, mark_best=False, mark_skip=()):
        self.name, self.caption, self.header = name, caption, header
        self.rows, self.mark_best, self.hib, self.mark_skip = [], mark_best, higher_is_better, set(mark_skip)

    def add(self, label, cells, rule_before=False):
        self.rows.append((label, cells, rule_before))

    def _marks(self, cells):
        vals = [c["mean"] if isinstance(c, dict) and c.get("n", 0) and i not in self.mark_skip else np.nan
                for i, c in enumerate(cells)]
        order = [i for i in np.argsort(vals)[::-1 if self.hib else 1] if not np.isnan(vals[i])]
        return (order[0] if order else None), (order[1] if len(order) > 1 else None)

    @staticmethod
    def _text(c):
        return fmt(c) if isinstance(c, dict) else str(c)

    def write(self, out: Path, prov, missing):
        with open(out / f"{self.name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", *self.header])
            for label, cells, _ in self.rows:
                w.writerow([label, *[self._text(c) for c in cells]])
        lines = ["% " + self.caption, "\\begin{tabular}{l" + "c" * len(self.header) + "}", "\\toprule",
               " & ".join(["", *[tex(h) for h in self.header]]) + " \\\\", "\\midrule"]
        md = [f"### {self.name}", "", self.caption, "", "| | " + " | ".join(self.header) + " |",
              "|---" * (len(self.header) + 1) + "|"]
        for label, cells, rule in self.rows:
            b1, b2 = self._marks(cells) if self.mark_best else (None, None)
            tcells = []
            for i, c in enumerate(cells):
                t = tex(self._text(c))
                if i == b1:
                    t = f"\\textbf{{{t}}}"
                elif i == b2:
                    t = f"\\underline{{{t}}}"
                tcells.append(t)
                if isinstance(c, dict):
                    prov.append([self.name, label, self.header[i], ";".join(c.get("dirs", []))])
                    if c.get("n", 0) == 0:
                        missing.append(f"{self.name}: {label} / {self.header[i]}")
            if rule:
                lines.append("\\midrule")
            lines.append(" & ".join([tex(label), *tcells]) + " \\\\")
            md.append(f"| {label} | " + " | ".join(self._text(c) for c in cells) + " |")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (out / f"{self.name}.tex").write_text("\n".join(lines) + "\n")
        return "\n".join(md) + "\n"


def minority_index(full, df, task):
    """Smallest class of `task`'s target adaptation split (by count)."""
    sub = select(df, group="main", task=task)
    if sub.empty:
        return None, None
    r = full[sub.iloc[0].run_dir]
    k = int(np.argmin(r["adapt_class_counts"]))
    return k, r["classes"][k]


MAIN_ARCH = load_config().arch


def fmt_p(p):
    if p is None or np.isnan(p):
        return "p n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def method_sub(df, method, group=None):
    return select(df, group=group or ("main" if method == "cab_sfda" else "baselines"), method=method,
                  arch=MAIN_ARCH)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    cfg = load_config()
    df, full = load_results(a.results or cfg.results_dir)
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("no results found")
        return
    B = benchmark_tasks()
    ds = load_datasets()
    prov, missing, md = [], [], ["# Tables generated from run logs", ""]
    tables = []
    cab = method_sub(df, "cab_sfda")
    methods = BASE + ["cab_sfda"]
    names = [DISPLAY_NAMES[m] for m in methods]

    # ------------------------------------------------------------ dataset summary
    summ_path = REPO_ROOT / cfg.splits_dir / "summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        t = Table("dataset_summary", "Images per domain (fixed stratified split) and imbalance ratio (max/min class).",
                  ["Benchmark", "Images", "Train (adapt.)", "Test", "Imbalance ratio"])
        for d, s in summ.items():
            t.add(d, [s["benchmark"], s["n"], s["n_train"], s["n_test"], f"{s['imbalance_ratio']:.2f}"])
        tables.append(t)
        t = Table("dataset_class_counts", "Images per class (train/test).",
                  sorted({c["class"] for s in summ.values() for c in s["per_class"]}))
        for d, s in summ.items():
            per = {c["class"]: f"{c['n']} ({c['n_train']}/{c['n_test']})" for c in s["per_class"]}
            t.add(d, [per.get(h, "") for h in t.header])
        tables.append(t)

    # ------------------------------------------------------------ main results
    def main_table(name, caption, task_rows, avg_rows):
        t = Table(name, caption, ["Source only", *names], mark_best=True, mark_skip=(0,))
        for task in task_rows:
            src = stat(cab[cab.task == task].sort_values("seed").src_oa.tolist())
            src["dirs"] = cab[cab.task == task].run_dir.tolist()
            t.add(task, [src, *[task_stat(method_sub(df, m), task) for m in methods]])
        for label, tasks in avg_rows:
            src = bench_stat(cab, tasks, "src_oa")
            t.add(label, [src, *[bench_stat(method_sub(df, m), tasks) for m in methods]], rule_before=True)
        return t

    tables.append(main_table("main_cross_scene", "Cross-scene OA (%), mean ± std over seeds; Avg. = per-seed mean over the 12 tasks.",
                             B["cross_scene"], [("Avg.", B["cross_scene"])]))
    tables.append(main_table("main_cross_sensor", "Cross-sensor OA (%), mean ± std over seeds; averages computed per seed.",
                             B["nwpu_nasc"] + B["whu_eurosat"],
                             [("Avg. NWPU-NaSC", B["nwpu_nasc"]), ("Avg. WHU-EuroSAT", B["whu_eurosat"]),
                              ("Avg. cross-sensor", B["cross_sensor"])]))

    # ------------------------------------------------------------ reliability (run level)
    all_tasks = [*B["cross_scene"], *B["cross_sensor"]]

    def reliability(sub):
        """Seed-to-seed variability and negative transfer of one method/variant over all tasks."""
        stds = [task_stat(sub, t)["std"] for t in all_tasks if (sub.task == t).sum() > 1]
        tmean = sub.groupby("task")[["oa", "src_oa"]].mean()
        degraded = 0
        for row in sub.itertuples():
            pce = full[row.run_dir].get("unsupervised", {}).get("pred_class_entropy")
            degraded += int((pce is not None and pce < 0.5) or row.oa < row.src_oa - 10)
        return {"median_std": float(np.median(stds)) if stds else np.nan,
                "scene_avg_std": bench_stat(sub, B["cross_scene"])["std"],
                "tasks_below_source": int((tmean.oa < tmean.src_oa).sum()),
                "runs_below_source": int((sub.oa < sub.src_oa).sum()), "degraded_runs": degraded,
                "tasks": int(sub.task.nunique()), "runs": int(len(sub))}

    t = Table("reliability", "Reliability over all tasks and seeds. Median std: median over tasks of the "
              "seed-to-seed std of OA; Avg. std: std of the per-seed cross-scene average; Tasks/Runs < source: "
              "mean OA of a task / OA of a run below that of the source model; Degraded: runs that collapsed "
              "(predicted-class entropy < 0.5 on the adaptation images) or ended > 10 pp below their source model.",
              ["Median std (pp)", "Avg. std cross-scene (pp)", "Tasks < source", "Runs < source", "Degraded runs",
               "Runs"])
    for m in methods:
        sub = method_sub(df, m)
        if sub.empty:
            continue
        r = reliability(sub)
        t.add(DISPLAY_NAMES[m], [f"{r['median_std']:.2f}", f"{r['scene_avg_std']:.2f}", r["tasks_below_source"],
                                 r["runs_below_source"], r["degraded_runs"], r["runs"]],
              rule_before=(m == "cab_sfda"))
    tables.append(t)
    with open(out / "run_level.csv", "w", newline="") as f:    # same statistics for every group/variant
        w = csv.writer(f)
        keys = ["median_std", "scene_avg_std", "tasks_below_source", "runs_below_source", "degraded_runs",
                "tasks", "runs"]
        w.writerow(["group", "method", "variant", "arch", *keys])
        for (g, m, v, arch), sub in df.groupby(["group", "method", "variant", "arch"]):
            r = reliability(sub)
            w.writerow([g, m, v, arch, *[f"{r[k]:.2f}" if isinstance(r[k], float) else r[k] for k in keys]])

    # ------------------------------------------------------------ per-class (CAB-SFDA)
    for metric in ("recall", "precision", "f1"):
        for bench in ("cross_scene", "nwpu_nasc", "whu_eurosat"):
            classes = ds["benchmarks"][bench]["classes"]
            t = Table(f"perclass_{metric}_{bench}", f"CAB-SFDA per-class {metric} (%) on the target test split; "
                      f"MA = macro average (computed per seed).", [*classes, "MA"])
            for task in B[bench]:
                cells = [class_value(cab, task, k, metric) for k in range(len(classes))]
                col = {"recall": "bacc", "f1": "mf1"}.get(metric)
                if col:
                    ma = task_stat(cab, task, col)
                else:
                    s = cab[cab.task == task].sort_values("seed")
                    ma = stat([float(np.mean(v)) for v in s[metric]])
                    ma["dirs"] = s.run_dir.tolist()
                t.add(task, [*cells, ma])
            tables.append(t)

    # ------------------------------------------------------------ PGAT diagnostics
    t = Table("pgat_diagnostics", "PGAT diagnostics in the final adaptation epoch (all quantities over the same samples). "
              "Cap = number of samples seen over all epochs (upper bound of Σ counts).",
              ["Accept. rate (%)", "Σ counts", "Cap", "mean τ", "q̄", "r̄", "R̄", "Active protos",
               "PL acc. accepted (%)", "corr(n_acc, n_true)"])
    for task in B["cross_scene"] + B["cross_sensor"]:
        s = cab[cab.task == task].sort_values("seed")
        if s.empty:
            t.add(task, ["--"] * len(t.header))
            continue
        dg = [full[d]["diagnostics"] for d in s.run_dir]
        last = [full[d]["history"][-1] for d in s.run_dir]

        def st(vals, digits=2):
            x = stat([v for v in vals if v is not None])
            x["dirs"] = s.run_dir.tolist()
            x["digits"] = digits
            return x
        t.add(task, [st([g["accept_rate_final_epoch"] for g in dg]), st([g["sum_counts"] for g in dg], 0),
                     st([g["sum_counts_cap"] for g in dg], 0), st([g["tau_mean_final"] for g in dg], 4),
                     st([g["q_mean_final_epoch"] for g in dg], 4), st([g["r_mean_final_epoch"] for g in dg], 4),
                     st([g["R_mean_final_epoch"] for g in dg], 4), st([g["prototypes_active_final"] for g in dg], 1),
                     st([g["pl_accuracy_accepted_final_epoch"] for g in dg]),
                     st([h.get("corr_accepted_vs_true_counts") for h in last], 3)])
    tables.append(t)

    # ------------------------------------------------------------ loss ablation
    k_min, cls_min = minority_index(full, df, "A2M")
    min_lbl = f"A2M F1 {cls_min}" if cls_min else "A2M minority F1"

    def ablation_cells(sub, src=False):
        oa, f1col = ("src_oa", "src_f1") if src else ("oa", "f1")
        a2m = class_value(sub, "A2M", k_min, f1col) if k_min is not None else {"n": 0}
        return [bench_stat(sub, B["cross_sensor"], oa), bench_stat(sub, B["cross_scene"], oa), a2m,
                task_stat(sub, "G2N", oa)]
    abl_header = ["Cross-sensor mean", "Cross-scene mean", min_lbl, "G2N OA"]
    t = Table("ablation_losses", f"Loss ablation (CAB-SFDA). {min_lbl}: F1 of the smallest class of the A2M "
              f"target ({cls_min}).", abl_header)
    t.add("Source only", ablation_cells(cab, src=True))
    for v, label in [("no_pl", "w/o L_pl"), ("no_im", "w/o L_im"), ("no_proto", "w/o L_proto"),
                     ("no_cons", "w/o L_cons")]:
        t.add(label, ablation_cells(select(df, group="ablation_losses", variant=v)))
    t.add("Full CAB-SFDA", ablation_cells(cab), rule_before=True)
    tables.append(t)

    # ------------------------------------------------------------ mechanism ablation
    def acc_rate(sub):
        vals = {}
        for seed, g in sub[sub.task.isin(B["cross_scene"])].groupby("seed"):
            if g.task.nunique() == len(B["cross_scene"]):
                vals[seed] = np.mean([full[d]["diagnostics"].get("accept_rate_final_epoch", np.nan)
                                      for d in g.run_dir])
        x = stat(list(vals.values()))
        x["dirs"] = sub.run_dir.tolist()
        return x
    t = Table("ablation_mechanisms", "Mechanism ablation (CAB-SFDA). Acceptance rate: cross-scene mean, final epoch.",
              [*abl_header, "Accept. rate (%)"])
    for sec in "ABCDEF":
        t.add(MECHANISM_OURS[sec], [*ablation_cells(cab), acc_rate(cab)], rule_before=sec != "A")
        for v, (_, s_, label) in MECHANISM_ABLATION.items():
            if s_ == sec:
                sub = select(df, group="ablation_mechanisms", variant=v)
                t.add(f"  {label}", [*ablation_cells(sub), acc_rate(sub)])
    tables.append(t)

    # ------------------------------------------------------------ backbone
    t = Table("backbone", "Backbone study (CAB-SFDA).", abl_header)
    for arch in ALL_BACKBONES:
        sub = cab if arch == MAIN_ARCH else select(df, group="backbone", variant=arch)
        t.add(arch, ablation_cells(sub))
    tables.append(t)

    # ------------------------------------------------------------ imbalance
    imb = select(df, group="imbalance")
    if not imb.empty:
        imb_tasks = [x for x in CROSS_SCENE_MLRSN_TARGETS if x in set(imb.task)]
        meths = [m for m in ["cab_sfda", *BASE] if m in set(imb.method)]
        t = Table("imbalance", "Controlled imbalance of the unlabeled MLRSN adaptation subset (sources: "
                  + ", ".join(imb_tasks) + "). OA (%) averaged over sources per seed; evaluation on the full MLRSN test split.",
                  [f"ratio {r:g}" for r in IMB_RATIOS], mark_best=False)
        tmin = Table("imbalance_minority_f1", "F1 (%) of the smallest class of the adaptation subset, averaged over sources per seed.",
                     [f"ratio {r:g}" for r in IMB_RATIOS])
        best = {}
        for m in meths:
            row, rowf = [], []
            for r in IMB_RATIOS:
                sub = select(imb, method=m, variant=f"ratio={_fmt(r)}")
                row.append(bench_stat(sub, imb_tasks))
                # minority class = smallest class of the subset (last of the ranking)
                vals = {}
                for seed, g in sub.groupby("seed"):
                    if g.task.nunique() == len(imb_tasks):
                        f1s = []
                        for d in g.run_dir:
                            rr = full[d]
                            kk = int(rr["imbalanced_subset"]["ranking"][-1])   # smallest class (same at ratio 1)
                            f1s.append(rr["final"]["per_class"]["f1"][kk])
                        vals[seed] = np.mean(f1s)
                x = stat(list(vals.values()))
                x["dirs"] = sub.run_dir.tolist()
                rowf.append(x)
            t.add(DISPLAY_NAMES[m], row)
            tmin.add(DISPLAY_NAMES[m], rowf)
            best[m] = row
        if "cab_sfda" in best and len(best) > 1:
            gaps = []
            for i, r in enumerate(IMB_RATIOS):
                cands = [(best[m][i]["mean"], m) for m in best if m != "cab_sfda" and best[m][i]["n"]]
                if cands and best["cab_sfda"][i]["n"]:
                    bm = max(cands)
                    gaps.append(f"{best['cab_sfda'][i]['mean'] - bm[0]:+.2f} vs {DISPLAY_NAMES[bm[1]]}")
                else:
                    gaps.append("--")
            t.add("CAB-SFDA − best baseline", gaps, rule_before=True)
        tables += [t, tmin]

    # ------------------------------------------------------------ sensitivity (CSV + table)
    sens = select(df, group="sensitivity")
    if not sens.empty:
        t = Table("sensitivity", "Hyper-parameter sensitivity (CAB-SFDA): benchmark means (per-seed averaging). "
                  "Post-hoc analysis with target labels; not used to choose the configuration.",
                  ["Cross-scene mean", "Cross-sensor mean"])
        defaults = load_config()
        for k, vals in SENSITIVITY.items():
            for v in sorted({*vals, defaults[k]}):
                sub = cab if v == defaults[k] else select(sens, variant=f"{k}={_fmt(v)}")
                t.add(f"{k} = {v:g}" + (" (default)" if v == defaults[k] else ""),
                      [bench_stat(sub, B["cross_scene"]), bench_stat(sub, B["cross_sensor"])])
        tables.append(t)
    trange = select(df, group="threshold_range")
    if not trange.empty:
        t = Table("threshold_range", f"Threshold range study with gamma = {THRESHOLD_RANGE['gamma']} (tau_max = 0.95).",
                  ["Cross-scene mean", "Cross-sensor mean"])
        for v in THRESHOLD_RANGE["tau_min"]:
            sub = select(trange, variant=f"tau_min={_fmt(v)}")
            t.add(f"[{v:g}, 0.95]", [bench_stat(sub, B["cross_scene"]), bench_stat(sub, B["cross_sensor"])])
        tables.append(t)

    # ------------------------------------------------------------ significance tests
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")   # identical values -> nan p
    sig_rows = []
    cols = [*B["cross_scene"], *B["cross_sensor"], "Avg. cross-scene", "Avg. cross-sensor"]

    def seed_values(sub, col_name):
        if col_name.startswith("Avg."):
            tasks = B["cross_scene"] if "scene" in col_name else B["cross_sensor"]
            return per_seed_bench(sub, tasks)
        s = sub[sub.task == col_name]
        return dict(zip(s.seed.astype(int), s.oa))
    raw_p = {}
    for m in BASE:
        msub = method_sub(df, m)
        for c in cols:
            a_, b_ = seed_values(cab, c), seed_values(msub, c)
            common = sorted(set(a_) & set(b_))
            if len(common) < 2:
                continue
            x, y = np.array([a_[s] for s in common]), np.array([b_[s] for s in common])
            d = x - y
            # identical difference in every seed: the t statistic is undefined -> no p-value
            if np.ptp(d) > 1e-6:                       # OA in %, tolerance for float round-off
                p_pair = stats.ttest_rel(x, y).pvalue
            else:
                p_pair = 1.0 if abs(d.mean()) < 1e-6 else np.nan
            p_welch = stats.ttest_ind(x, y, equal_var=False).pvalue
            raw_p[(m, c)] = p_pair
            sig_rows.append({"baseline": m, "column": c, "n_seeds": len(common), "diff_mean": d.mean(),
                             "p_paired": p_pair, "p_welch": p_welch})
    # Holm correction over the baselines, per column
    for c in cols:
        ps = [(raw_p[(m, c)], m) for m in BASE if (m, c) in raw_p and not np.isnan(raw_p[(m, c)])]
        ps.sort()
        k = len(ps)
        running = 0.0
        for i, (p, m) in enumerate(ps):
            running = max(running, min(1.0, (k - i) * p))
            for r in sig_rows:
                if r["baseline"] == m and r["column"] == c:
                    r["p_paired_holm"] = running
    # Wilcoxon over the 12 cross-scene task means
    for m in BASE:
        msub = method_sub(df, m)
        xs = [task_stat(cab, tk)["mean"] for tk in B["cross_scene"]]
        ys = [task_stat(msub, tk)["mean"] for tk in B["cross_scene"]]
        ok = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
        if len(ok) >= 6:
            w = stats.wilcoxon([x for x, _ in ok], [y for _, y in ok])
            sig_rows.append({"baseline": m, "column": "Wilcoxon over cross-scene tasks", "n_seeds": len(ok),
                             "diff_mean": float(np.mean([x - y for x, y in ok])), "p_paired": w.pvalue,
                             "p_welch": np.nan, "p_paired_holm": np.nan})
    if sig_rows:
        with open(out / "significance_full.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["baseline", "column", "n_seeds", "diff_mean", "p_paired",
                                              "p_paired_holm", "p_welch"])
            w.writeheader()
            w.writerows(sig_rows)
        show = ["Avg. cross-scene", "Avg. cross-sensor", *B["cross_sensor"], "Wilcoxon over cross-scene tasks"]
        t = Table("significance", "CAB-SFDA vs. each baseline: OA difference (CAB-SFDA minus baseline, percentage "
                  "points) and two-sided paired t-test over seeds (same source model per seed), Holm-adjusted over the "
                  "baselines; last column: Wilcoxon signed-rank test over the 12 cross-scene task means. n/a: the "
                  "difference is identical in every seed (t-test undefined). All p-values (raw, Holm, Welch) in significance_full.csv.", show)
        for m in BASE:
            cells = []
            for c in show:
                r = next((r for r in sig_rows if r["baseline"] == m and r["column"] == c), None)
                if r is None:
                    cells.append("--")
                else:
                    p = r["p_paired"] if c.startswith("Wilcoxon") else r.get("p_paired_holm", r["p_paired"])
                    cells.append(f"{r['diff_mean']:+.2f} ({fmt_p(p)})")
            t.add(DISPLAY_NAMES[m], cells)
        tables.append(t)

    # ------------------------------------------------------------ complexity (paper "Model efficiency")
    tim = select(df, group="timing", arch=MAIN_ARCH)
    src_name = "timing"
    if tim.empty:                     # fall back to the main/baseline runs (timings only valid with 1 job per GPU)
        tim = df[((df.group == "main") & (df.method == "cab_sfda")) | (df.group == "baselines")]
        tim = tim[(tim.arch == MAIN_ARCH) & (tim.task == "W2E")]
        src_name = "main/baselines"
    if not tim.empty:
        def cstat(sub, key):
            vals = [full[d].get("compute", {}).get(key) for d in sub.run_dir]
            st_ = stat([v for v in vals if v is not None])
            st_["dirs"] = sub.run_dir.tolist()
            return st_
        jobs = sorted({str(full[d].get("compute", {}).get("jobs_per_gpu")) for d in tim.run_dir})
        devs = sorted({str(full[d].get("compute", {}).get("device")) for d in tim.run_dir})
        for task in sorted(set(tim.task)):
            t = Table(f"complexity_{task}", f"Computational cost on {task} (runs of group '{src_name}'; device: "
                      f"{', '.join(devs)}; jobs per GPU: {', '.join(jobs)}). GPU memory = peak reserved memory "
                      f"during adaptation; time = adaptation wall time / epochs (includes clustering and memory-bank "
                      f"passes). OA from the same runs.", ["OA (%)", "GPU memory (MiB)", "Time per epoch (s)"])
            for m in methods:
                sub = tim[(tim.method == m) & (tim.task == task)]
                if sub.empty:
                    t.add(DISPLAY_NAMES[m], ["--"] * 3)
                    continue
                oa = task_stat(sub, task)
                mem, sec = cstat(sub, "peak_reserved_mib"), cstat(sub, "seconds_per_epoch")
                for st_ in (mem, sec):
                    st_["digits"] = 0 if st_ is mem else 1
                t.add(DISPLAY_NAMES[m], [oa, mem, sec])
            tables.append(t)

    for t in tables:
        md.append(t.write(out, prov, missing))
    (out / "summary.md").write_text("\n".join(md))
    with open(out / "provenance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "row", "column", "run_dirs"])
        w.writerows(prov)
    (out / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    print(f"{len(tables)} tables -> {out}/ ({len(missing)} empty cells, see missing.txt)")


if __name__ == "__main__":
    main()
