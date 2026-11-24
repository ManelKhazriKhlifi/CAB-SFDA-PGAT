#!/usr/bin/env python
"""Figures from the run logs -> figures/*.pdf

  python make_figures.py                 # everything available
  python make_figures.py --no-tsne       # skip t-SNE (slow)

Averages are computed per seed over tasks, then over seeds (same as the tables), and every
curve is drawn from the same values that appear in tables/sensitivity.csv.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cabsfda.config import REPO_ROOT, load_config, load_datasets  # noqa: E402
from cabsfda.experiments import SENSITIVITY, THRESHOLD_RANGE, _fmt, benchmark_tasks  # noqa: E402
from cabsfda.results import bench_stat, load_results, select, task_stat  # noqa: E402
from cabsfda.source import source_dir  # noqa: E402

MAIN_ARCH = load_config().arch
LABELS = {"lambda_pl": r"$\lambda_{pl}$", "lambda_cons": r"$\lambda_{cons}$", "gamma": r"$\gamma$",
          "lambda_proto": r"$\lambda_{proto}$", "ema_momentum": r"EMA momentum $\alpha$",
          "lr": r"$\eta_t$", "temp_cons": r"$T$"}


def _curves(xs, subs, B, xlabel, path, vline=None, logx=False):
    """Two panels (cross-scene, cross-sensor): one line per task, black = benchmark average
    (per-seed averaging; missing when a task of the benchmark has no run for that value)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (bname, tasks) in zip(axes, [("Cross-scene", B["cross_scene"]), ("Cross-sensor", B["cross_sensor"])]):
        for t in tasks:
            ys = [task_stat(s, t)["mean"] for s in subs]
            if not np.all(np.isnan(ys)):
                ax.plot(xs, ys, marker="o", lw=1, alpha=0.6, label=t)
        avg = [bench_stat(s, tasks) for s in subs]
        ax.errorbar(xs, [a["mean"] for a in avg], yerr=[a["std"] for a in avg], color="k", lw=2.5,
                    marker="s", capsize=3, label="Average (per seed)")
        if vline is not None:
            ax.axvline(vline, color="grey", ls="--", lw=1)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("OA (%)")
        ax.set_title(bname)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sensitivity(df, out, defaults, B):
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    sens = select(df, group="sensitivity")
    for k, vals in SENSITIVITY.items():
        xs = sorted([*vals, defaults[k]])
        subs = [cab if v == defaults[k] else select(sens, variant=f"{k}={_fmt(v)}") for v in xs]
        if all(s.empty for v, s in zip(xs, subs) if v != defaults[k]):
            continue                      # no sensitivity runs for this hyper-parameter
        _curves(xs, subs, B, LABELS.get(k, k), out / f"sensitivity_{k}.pdf", vline=defaults[k], logx=(k in ("lr",)))
    tr = select(df, group="threshold_range", arch=MAIN_ARCH)
    if not tr.empty:
        xs = THRESHOLD_RANGE["tau_min"]
        _curves(xs, [select(tr, variant=f"tau_min={_fmt(v)}") for v in xs], B,
                r"$\tau_{min}$ ($\tau_{max}$ = %g, $\gamma$ = %g)" % (defaults["tau_max"], THRESHOLD_RANGE["gamma"]),
                out / "threshold_range.pdf")


def pgat_dynamics(df, full, out, ds):
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    for task in sorted(set(cab.task)):
        s = cab[(cab.task == task)].sort_values("seed")
        r = full[s.iloc[0].run_dir]
        hist = r["history"]
        classes = r["classes"]
        ep = [h["epoch"] for h in hist]
        tau = np.array([h["policy"]["tau"] for h in hist])
        prec = np.array([h["per_class"]["pl_precision"] for h in hist], dtype=float)
        acc_share = np.array([np.array(h["per_class"]["n_accepted"]) / max(1, h["n_accepted"]) for h in hist])
        true_share = np.array(r["adapt_class_counts"]) / sum(r["adapt_class_counts"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for c, name in enumerate(classes):
            axes[0].plot(ep, tau[:, c], label=name)
            axes[1].plot(ep, prec[:, c], label=name)
            axes[2].plot(ep, acc_share[:, c] - true_share[c], label=name)
        axes[0].set_title(f"{task}: per-class threshold (end of epoch)")
        axes[1].set_title("pseudo-label precision of accepted samples (%)")
        axes[2].set_title("accepted share - true share")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"pgat_dynamics_{task}_seed{r['seed']}.pdf")
        plt.close(fig)


def confusion(df, full, out):
    """Row-normalised confusion matrices (mean over seeds) for CAB-SFDA (confusion_<task>.pdf) and every
    baseline (confusion_<method>_<task>.pdf; the paper's 'Baseline' subsection shows SHOT)."""
    subs = [("cab_sfda", select(df, group="main", method="cab_sfda", arch=MAIN_ARCH))]
    base = select(df, group="baselines", arch=MAIN_ARCH)
    subs += [(m, base[base.method == m]) for m in sorted(set(base.method))]
    for method, sub in subs:
        for task in sorted(set(sub.task)):
            s = sub[sub.task == task]
            cms = []
            for d in s.run_dir:
                cm = np.array(full[d]["final"]["confusion"], dtype=float)
                cms.append(cm / cm.sum(1, keepdims=True).clip(min=1))
            cm = 100 * np.mean(cms, 0)
            classes = full[s.iloc[0].run_dir]["classes"]
            fig, ax = plt.subplots(figsize=(1 + 0.7 * len(classes), 1 + 0.6 * len(classes)))
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 60 else "black")
            ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(classes)), classes, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            name = "CAB-SFDA" if method == "cab_sfda" else method
            ax.set_title(f"{name} {task} (row-normalised %, mean of {len(cms)} seeds)", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / (f"confusion_{task}.pdf" if method == "cab_sfda" else f"confusion_{method}_{task}.pdf"))
            plt.close(fig)


def radar(df, out, B):
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    tasks = [t for t in B["cross_scene"] if t in set(cab.task)]
    if len(tasks) < 3:
        return
    src = [task_stat(cab, t, "src_oa")["mean"] for t in tasks]
    ada = [task_stat(cab, t, "oa")["mean"] for t in tasks]
    ang = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    for vals, lab in ((src, "Source only"), (ada, "CAB-SFDA")):
        ax.plot(ang + ang[:1], vals + vals[:1], marker="o", label=lab)
        for a_, v in zip(ang, vals):
            ax.text(a_, v, f"{v:.1f}", fontsize=6)
    ax.set_xticks(ang, tasks)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "source_vs_adapted_radar.pdf")
    plt.close(fig)


def tsne(df, full, out, cfg):
    from sklearn.manifold import TSNE
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    for d in cab[cab.seed == 0].run_dir:
        r = full[d]
        try:
            fa = np.load(f"{d}/features.npz")
            c = cfg.copy()
            c.arch = r["arch"]
            fb = np.load(source_dir(c, r["source"], 0) / f"features_on_{r['target']}.npz")
        except FileNotFoundError:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, f, title in ((axes[0], fb, "source model"), (axes[1], fa, "CAB-SFDA")):
            n = len(f["labels"])
            sel = np.random.default_rng(0).choice(n, size=min(n, 3000), replace=False)
            perp = float(min(30, max(2, (len(sel) - 1) // 3)))          # must be < number of points
            emb = TSNE(init="pca", random_state=0, perplexity=perp).fit_transform(f["feats"][sel].astype(np.float32))
            for c_, name in enumerate(r["classes"]):
                m = f["labels"][sel] == c_
                ax.scatter(emb[m, 0], emb[m, 1], s=3, label=name)
            ax.set_title(f"{r['task']}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[1].legend(markerscale=4, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"tsne_{r['task']}.pdf")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--no-tsne", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    df, full = load_results(cfg.results_dir)
    if df.empty:
        print("no results found")
        return
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    B = benchmark_tasks()
    sensitivity(df, out, cfg, B)
    pgat_dynamics(df, full, out, load_datasets())
    confusion(df, full, out)
    radar(df, out, B)
    if not a.no_tsne:
        tsne(df, full, out, cfg)
    print(f"figures -> {out}/")


if __name__ == "__main__":
    main()
