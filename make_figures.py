#!/usr/bin/env python
"""Figures from the run results -> figures/*.pdf

  python make_figures.py                       # everything available
  python make_figures.py --no-tsne             # skip t-SNE (slow the first time)
  python make_figures.py --only cm radar       # only some figure types (cm, tsne, sens, radar, pgat)
  python make_figures.py --paper figures_paper # also write the files with the names and folders of the
                                               # paper project (figure/CM, figure/tsne, figure/hyper)

Every figure is drawn at the size at which it is printed in the paper (see PRINT_WIDTH below), so
that the font sizes set here (7-9 pt) are the printed sizes. Include each figure in LaTeX with the
width given in PRINT_WIDTH (text width of the cas-sc template: 6.48 in = 468 pt); a different width
scales all text by the same factor.

Averages are computed per seed over tasks, then over seeds (same as the tables), and every curve
is drawn from the same values that appear in tables/sensitivity.csv. The t-SNE embeddings are
cached in <out>/tsne_cache/, so that re-drawing the figures with another style is fast.
"""
import argparse
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cabsfda.config import REPO_ROOT, load_config  # noqa: E402
from cabsfda.experiments import SENSITIVITY, THRESHOLD_RANGE, _fmt, benchmark_tasks  # noqa: E402
from cabsfda.results import bench_stat, load_results, select, task_stat  # noqa: E402

# ----------------------------------------------------------------------------- style
TEXT_WIDTH_IN = 6.48                    # \textwidth of the paper (468 pt)
PRINT_WIDTH = {                         # figure width in inches = LaTeX width in the paper
    "cm": 0.48 * TEXT_WIDTH_IN,         # \includegraphics[width=0.48\textwidth], two per row
    "tsne": 0.95 * TEXT_WIDTH_IN,       # width=0.95\textwidth
    "curves": 0.95 * TEXT_WIDTH_IN,     # width=0.95\textwidth (sensitivity, threshold range)
    "radar": 0.60 * TEXT_WIDTH_IN,      # width=0.6\textwidth
    "pgat": 0.95 * TEXT_WIDTH_IN,
}
FS = {"base": 8, "small": 7, "title": 9, "cell": 7.5, "cell_small": 6.5}

plt.rcParams.update({
    "font.size": FS["base"], "axes.titlesize": FS["title"], "axes.labelsize": FS["base"],
    "xtick.labelsize": FS["small"], "ytick.labelsize": FS["small"], "legend.fontsize": FS["small"],
    "legend.handlelength": 1.6, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.5, "grid.linewidth": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts (editable, accepted by publishers)
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300,
})
MAIN_ARCH = load_config().arch
LABELS = {"lambda_pl": r"$\lambda_{pl}$", "lambda_cons": r"$\lambda_{cons}$", "gamma": r"$\gamma$",
          "lambda_proto": r"$\lambda_{proto}$", "ema_momentum": r"EMA momentum $\alpha$ (log scale of $1-\alpha$)",
          "lr": r"Learning rate $\eta_t$", "temp_cons": r"Temperature $T$"}
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "*", "p"]


def pretty(name: str) -> str:
    """Class name for display: 'rectangular_farmland' -> 'rectangular farmland'."""
    return name.replace("_", " ")


def tick_label(name: str, width: int = 11) -> str:
    """Class name for a confusion-matrix tick: long names are broken over two lines."""
    s = pretty(name)
    if len(s) > width and " " in s:
        a, b = s.split(" ", 1)
        return f"{a}\n{b}"
    return s


def save(fig, path, extra=()):
    fig.savefig(path)
    for p in extra:
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, p)
    plt.close(fig)


# ----------------------------------------------------------------------------- plotting primitives
def plot_confusion(cm, classes, title, path, extra=()):
    """Row-normalised confusion matrix (%), printed at PRINT_WIDTH['cm']."""
    k = len(classes)
    w = PRINT_WIDTH["cm"]
    fig, ax = plt.subplots(figsize=(w, w * 0.98))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    fs = FS["cell"] if k <= 7 else FS["cell_small"]
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=fs,
                    color="white" if cm[i, j] > 55 else "black")
    ax.set_xticks(range(k), [pretty(c) for c in classes], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(k), [tick_label(c) for c in classes])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(length=2, pad=2)
    if title:
        ax.set_title(title, pad=4)
    save(fig, path, extra)


def plot_curves(xs, panels, xlabel, path, vline=None, scale="linear", extra=()):
    """panels: [(title, {task: [y per x]}, (avg_mean, avg_std))]. One line per task and the
    benchmark average in black; the legend of each panel is drawn in its own area below the
    panel, so that it never hides a curve.
    scale: "linear", "log", or "log1m" (x placed at 1 - x on a reversed log axis and labelled with
    x; used for the EMA momentum, whose values 0.995 and 0.999 would overlap on a linear axis)."""
    n = len(panels)
    ticks = list(xs)
    if scale == "log1m":
        xs = [1 - x for x in xs]
        vline = None if vline is None else 1 - vline
    fig = plt.figure(figsize=(PRINT_WIDTH["curves"], 3.2), layout="constrained")
    gs = fig.add_gridspec(2, n, height_ratios=[1, 0.38])
    cmap = plt.get_cmap("tab20")
    for p, (title, lines, (am, asd)) in enumerate(panels):
        ax = fig.add_subplot(gs[0, p])
        for i, (t, ys) in enumerate(lines.items()):
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], color=cmap(i % 20), lw=0.9, ms=3,
                    alpha=0.85, label=t)
        ax.errorbar(xs, am, yerr=asd, color="k", lw=1.8, marker="s", ms=4, capsize=2.5,
                    label="Average", zorder=5)
        if vline is not None:
            ax.axvline(vline, color="grey", ls="--", lw=0.8, zorder=0)
        if scale in ("log", "log1m"):
            ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:g}" for x in ticks])
        if scale == "log1m":
            ax.invert_xaxis()
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("OA (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        lax = fig.add_subplot(gs[1, p])
        lax.axis("off")
        h, lab = ax.get_legend_handles_labels()
        lax.legend(h, lab, loc="upper center", ncol=min(5, len(h)), frameon=False, handletextpad=0.4,
                   borderaxespad=0)
    save(fig, path, extra)


def plot_radar(tasks, src, ada, path, extra=()):
    """Source-model vs. adapted OA per task. The values are written under each task name
    (source -> adapted), so that no label is drawn on the curves."""
    w = PRINT_WIDTH["radar"]
    fig = plt.figure(figsize=(w, w))
    ax = fig.add_subplot(111, polar=True)
    ang = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False)
    lo = 10 * np.floor((min(min(src), min(ada)) - 8) / 10)
    ax.set_ylim(lo, 100)
    for vals, lab, col in ((src, "Source model", "tab:orange"), (ada, "CAB-SFDA", "tab:blue")):
        v = np.r_[vals, vals[:1]]
        a = np.r_[ang, ang[:1]]
        ax.plot(a, v, marker="o", ms=3.5, lw=1.2, color=col, label=lab)
        ax.fill(a, v, color=col, alpha=0.08)
    ax.set_xticks(ang, [f"{t}\n{s_:.1f} \u2192 {a_:.1f}" for t, s_, a_ in zip(tasks, src, ada)])
    ax.tick_params(axis="x", pad=9, labelsize=FS["small"])
    ax.tick_params(axis="y", labelsize=FS["cell_small"], labelcolor="0.4")
    ax.set_rlabel_position(180 / len(tasks))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, path, extra)


def plot_tsne(embs, labels, classes, titles, path, extra=()):
    """embs/labels: one per panel (source model, adapted); one shared legend below the panels."""
    fig, axes = plt.subplots(1, len(embs), figsize=(PRINT_WIDTH["tsne"], 0.47 * PRINT_WIDTH["tsne"]))
    cmap = plt.get_cmap("tab10")
    for ax, emb, lab, title in zip(np.atleast_1d(axes), embs, labels, titles):
        size = 2.5 if len(lab) > 1500 else (6 if len(lab) > 400 else 12)    # larger points for small targets
        for c, name in enumerate(classes):
            m = lab == c
            ax.scatter(emb[m, 0], emb[m, 1], s=size, color=cmap(c % 10), alpha=0.8, linewidths=0,
                       label=pretty(name), rasterized=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(classes), 5 if len(classes) > 7 else len(classes)),
               markerscale=3, frameon=False, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.08 if len(classes) <= 7 else 0.12, 1, 1))
    save(fig, path, extra)


def plot_pgat(ep, tau, prec, share_gap, classes, task, path):
    fig, axes = plt.subplots(1, 3, figsize=(PRINT_WIDTH["pgat"], 2.2))
    cmap = plt.get_cmap("tab10")
    for c, name in enumerate(classes):
        kw = dict(color=cmap(c % 10), lw=1, label=pretty(name))
        axes[0].plot(ep, tau[:, c], **kw)
        axes[1].plot(ep, prec[:, c], **kw)
        axes[2].plot(ep, share_gap[:, c], **kw)
    for ax, t in zip(axes, [f"{task}: class threshold", "Accepted PL precision (%)", "Accepted - true share"]):
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(classes), 5), frameon=False, bbox_to_anchor=(0.5, -0.02))
    rows = int(np.ceil(len(classes) / 5))
    fig.tight_layout(rect=(0, 0.06 + 0.06 * rows, 1, 1))
    save(fig, path)


# ----------------------------------------------------------------------------- figures from results
def sensitivity(df, out, defaults, B, paper):
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    sens = select(df, group="sensitivity")

    def panels(subs):
        res = []
        for bname, tasks in (("Cross-scene", B["cross_scene"]), ("Cross-sensor", B["cross_sensor"])):
            lines = {}
            for t in tasks:
                ys = [task_stat(s, t)["mean"] for s in subs]
                if not np.all(np.isnan(ys)):
                    lines[t] = ys
            avg = [bench_stat(s, tasks) for s in subs]
            res.append((bname, lines, ([a["mean"] for a in avg], [a["std"] for a in avg])))
        return res

    for k, vals in SENSITIVITY.items():
        xs = sorted({*vals, defaults[k]})
        subs = [cab if v == defaults[k] else select(sens, variant=f"{k}={_fmt(v)}") for v in xs]
        if all(s.empty for v, s in zip(xs, subs) if v != defaults[k]):
            continue                      # no sensitivity runs for this hyper-parameter
        name = f"sensitivity_{k}.pdf"
        plot_curves(xs, panels(subs), LABELS.get(k, k), out / name, vline=defaults[k],
                    scale={"lr": "log", "ema_momentum": "log1m"}.get(k, "linear"),
                    extra=[paper / "hyper" / name] if paper else ())
    tr = select(df, group="threshold_range", arch=MAIN_ARCH)
    if not tr.empty:
        xs = THRESHOLD_RANGE["tau_min"]
        plot_curves(xs, panels([select(tr, variant=f"tau_min={_fmt(v)}") for v in xs]),
                    r"$\tau_{min}$ ($\tau_{max}$ = %g, $\gamma$ = %g)" % (defaults["tau_max"], THRESHOLD_RANGE["gamma"]),
                    out / "threshold_range.pdf", extra=[paper / "hyper" / "threshold_range.pdf"] if paper else ())


def pgat_dynamics(df, full, out):
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    for task in sorted(set(cab.task)):
        s = cab[(cab.task == task)].sort_values("seed")
        r = full[s.iloc[0].run_dir]
        hist = r["history"]
        ep = [h["epoch"] for h in hist]
        tau = np.array([h["policy"]["tau"] for h in hist])
        prec = np.array([h["per_class"]["pl_precision"] for h in hist], dtype=float)
        acc_share = np.array([np.array(h["per_class"]["n_accepted"]) / max(1, h["n_accepted"]) for h in hist])
        true_share = np.array(r["adapt_class_counts"]) / sum(r["adapt_class_counts"])
        plot_pgat(ep, tau, prec, acc_share - true_share[None, :], r["classes"], task,
                  out / f"pgat_dynamics_{task}_seed{r['seed']}.pdf")


def confusion(df, full, out, paper, titles=True):
    """Row-normalised confusion matrices (mean over seeds) for CAB-SFDA (confusion_<task>.pdf) and every
    baseline (confusion_<method>_<task>.pdf; the paper shows SHOT)."""
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
            name = "CAB-SFDA" if method == "cab_sfda" else method.upper()
            fname = f"confusion_{task}.pdf" if method == "cab_sfda" else f"confusion_{method}_{task}.pdf"
            extra = []
            if paper and method in ("cab_sfda", "shot"):
                extra = [paper / "CM" / f"{'cab' if method == 'cab_sfda' else 'shot'}_{task}.pdf"]
            plot_confusion(cm, classes, f"{name}: {task}" if titles else None, out / fname, extra)


def radar(df, out, B, paper):
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    tasks = [t for t in B["cross_scene"] if t in set(cab.task)]
    if len(tasks) < 3:
        return
    src = [task_stat(cab, t, "src_oa")["mean"] for t in tasks]
    ada = [task_stat(cab, t, "oa")["mean"] for t in tasks]
    plot_radar(tasks, src, ada, out / "source_vs_adapted_radar.pdf",
               extra=[paper / "source_vs_adapted_radar.pdf"] if paper else ())


def tsne(df, full, out, cfg, paper):
    from sklearn.manifold import TSNE

    from cabsfda.source import source_dir
    cache = out / "tsne_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cab = select(df, group="main", method="cab_sfda", arch=MAIN_ARCH)
    for d in cab[cab.seed == 0].run_dir:
        r = full[d]
        cfile = cache / f"{r['task']}.npz"
        if cfile.exists():
            z = np.load(cfile)
            embs, labs = [z["emb_src"], z["emb_ada"]], [z["lab_src"], z["lab_ada"]]
        else:
            try:
                fa = np.load(f"{d}/features.npz")
                c = cfg.copy()
                c.arch = r["arch"]
                fb = np.load(source_dir(c, r["source"], 0) / f"features_on_{r['target']}.npz")
            except FileNotFoundError:
                continue
            embs, labs = [], []
            for f in (fb, fa):
                n = len(f["labels"])
                sel = np.random.default_rng(0).choice(n, size=min(n, 3000), replace=False)
                perp = float(min(30, max(2, (len(sel) - 1) // 3)))          # must be < number of points
                embs.append(TSNE(init="pca", random_state=0, perplexity=perp)
                            .fit_transform(f["feats"][sel].astype(np.float32)))
                labs.append(f["labels"][sel])
            np.savez(cfile, emb_src=embs[0], emb_ada=embs[1], lab_src=labs[0], lab_ada=labs[1])
        plot_tsne(embs, labs, r["classes"], [f"{r['task']}: source model", f"{r['task']}: CAB-SFDA"],
                  out / f"tsne_{r['task']}.pdf", extra=[paper / "tsne" / f"tsne_{r['task']}.pdf"] if paper else ())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--only", nargs="+", choices=["cm", "tsne", "sens", "radar", "pgat"],
                    help="draw only these figure types")
    ap.add_argument("--paper", help="also copy the figures of the paper into this folder, with the names of "
                                    "the paper project (CM/cab_<task>.pdf, CM/shot_<task>.pdf, tsne/, hyper/)")
    ap.add_argument("--no-cm-titles", action="store_true", help="omit the titles of the confusion matrices")
    a = ap.parse_args()
    cfg = load_config()
    df, full = load_results(cfg.results_dir)
    if df.empty:
        print("no results found")
        return
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    paper = (REPO_ROOT / a.paper) if a.paper else None
    want = set(a.only or ["cm", "tsne", "sens", "radar", "pgat"])
    if a.no_tsne:
        want.discard("tsne")
    B = benchmark_tasks()
    if "sens" in want:
        sensitivity(df, out, cfg, B, paper)
    if "pgat" in want:
        pgat_dynamics(df, full, out)
    if "cm" in want:
        confusion(df, full, out, paper, titles=not a.no_cm_titles)
    if "radar" in want:
        radar(df, out, B, paper)
    if "tsne" in want:
        tsne(df, full, out, cfg, paper)
    print(f"figures -> {out}/" + (f" (paper copies -> {paper}/)" if paper else ""))


if __name__ == "__main__":
    main()
