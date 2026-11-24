"""Every experiment of the paper as an explicit list of runs.

A run is (group, method, variant, task, seed, arch). The variant name maps to config
overrides defined here, so a run folder name always identifies exactly what was run.
Run folders: <results_dir>/runs/<group>/<method>/<variant>/<arch>/<task>/seed<seed>/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .baselines import BASELINES
from .config import all_tasks, load_config, load_datasets, resolve_path, task_info

SEEDS = [0, 1, 2]
CROSS_SCENE_MLRSN_TARGETS = ["A2M", "C2M", "R2M"]

# ---------------------------------------------------------------- ablation variants (CAB-SFDA)
LOSS_ABLATION = {                      # paper Table "loss-wise ablation"; 'full' = main runs
    "no_pl": {"lambda_pl": 0.0},
    "no_im": {"lambda_im": 0.0},
    "no_proto": {"lambda_proto": 0.0},      # memory still updated: class weights and r_c kept
    "no_cons": {"lambda_cons": 0.0},
}

MECHANISM_ABLATION = {                 # (overrides, table section, label)
    "A_uniform": ({"weighting": "uniform"}, "A", "Unweighted cross-entropy"),
    "A_inv_freq": ({"weighting": "inv_freq"}, "A", "Inverse-frequency weights (1/n_c)"),
    "B_no_memory": ({"memory_mode": "none"}, "B",
                    "No memory (no L_proto, uniform weights, PGAT on confidence only)"),
    "B_static": ({"memory_mode": "static"}, "B", "Static prototypes (source model, never updated)"),
    "B_batch": ({"memory_mode": "batch"}, "B", "Batch prototypes (no EMA)"),
    "B_ungated": ({"memory_update": "all"}, "B", "Class-wise EMA updated with all predictions (no PGAT gating)"),
    "C_fixed_090": ({"threshold": "fixed", "fixed_tau": 0.90}, "C", "Fixed global threshold 0.90"),
    "C_fixed_085": ({"threshold": "fixed", "fixed_tau": 0.85}, "C", "Fixed global threshold 0.85"),
    "C_fixed_080": ({"threshold": "fixed", "fixed_tau": 0.80}, "C", "Fixed global threshold 0.80"),
    "C_global": ({"threshold": "pgat_global"}, "C", "Class-agnostic PGAT (single threshold)"),
    "C_conf_only": ({"threshold": "pgat_conf_only"}, "C", "Per-class PGAT, confidence only (lambda_r = 1)"),
    "C_flexmatch": ({"threshold": "flexmatch"}, "C", "FlexMatch class-wise thresholds"),
    "C_freematch": ({"threshold": "freematch"}, "C", "FreeMatch self-adaptive thresholds"),
    "C_no_lag": ({"threshold_lag": False}, "C", "PGAT without causal lag"),
    "D_symmetric": ({"cons_mode": "symmetric", "teacher": "none"}, "D",
                    "Symmetric KL between the student's own views, no teacher"),
    "D_entropy_only": ({"im_mode": "entropy_only"}, "D", "Entropy minimisation only (no diversity term)"),
    "E_trainable_clf": ({"freeze_classifier": False}, "E", "Trainable classifier"),
    "E_bn_train": ({"bn_mode": "train"}, "E", "BN statistics updated on target batches"),
    "F_no_teacher": ({"teacher": "none"}, "F", "No EMA teacher (the student labels its own data)"),
    "F_proto_mse": ({"proto_loss": "mse"}, "F", "Prototype loss as MSE on raw features"),
}
MECHANISM_OURS = {"A": "Inverse-square-root weights (ours)", "B": "Class-wise EMA gated by PGAT (ours)",
                  "C": "PGAT, per-class, causal lag (ours)",
                  "D": "One-way distillation from the EMA teacher + standard IM (ours)",
                  "E": "Frozen classifier + frozen BN statistics (ours)",
                  "F": "EMA teacher + cosine prototype loss (ours)"}

SENSITIVITY = {                        # key: values (default value comes from the main runs)
    "lambda_pl": [0.1, 0.5, 1.0],
    "lambda_cons": [0.1, 0.5, 1.0],
    "lambda_proto": [0.05, 0.3, 0.5],
    "gamma": [0.05, 0.2, 0.3, 0.5],
    "lr": [3e-3, 5e-3, 2e-2],
    "ema_momentum": [0.9, 0.95, 0.995, 0.999],
    "temp_cons": [0.5, 2.0],
}
# Label-free hyper-parameter selection (select_hparams.py): a small grid around the defaults on
# PILOT_TASKS (seed 0), ranked by SND on the unlabeled adaptation images. Target labels are not read.
PILOT_GRID = {
    "default": {},
    "cons1.0": {"lambda_cons": 1.0},
    "cons0.1": {"lambda_cons": 0.1},
    "proto0.5": {"lambda_proto": 0.5},
    "ema0.999": {"ema_momentum": 0.999},
    "ema0.95": {"ema_momentum": 0.95},
    "pl1.0": {"lambda_pl": 1.0},
    "lr5e-3": {"lr": 5e-3},
}
PILOT_TASKS = ["A2C", "M2R", "G2N", "E2W"]
# Optional: range study. With gamma = 0.1 tau_min never binds (raw threshold in [0.85, 0.95]),
# so the range is only studied with gamma = 0.5, where the raw threshold spans [0.65, 1.15].
THRESHOLD_RANGE = {"tau_min": [0.5, 0.6, 0.7, 0.8], "gamma": 0.5}

IMB_RATIOS = [1.0, 1.5, 2.0, 3.0, 5.0]
ALL_BACKBONES = ["resnet18", "resnet50", "resnet101"]   # the configured `arch` is the main one


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def variant_overrides(group: str, variant: str) -> dict:
    if group in ("main", "baselines", "backbone", "timing"):
        return {}
    if group == "baselines_bn_train":
        return {"bn_mode": "train"}
    if group == "ablation_losses":
        return dict(LOSS_ABLATION[variant])
    if group == "ablation_mechanisms":
        return dict(MECHANISM_ABLATION[variant][0])
    if group == "sensitivity":
        k, v = variant.split("=")
        return {k: SENSITIVITY_TYPES.get(k, float)(v)}
    if group == "threshold_range":
        return {"tau_min": float(variant.split("=")[1]), "gamma": THRESHOLD_RANGE["gamma"]}
    if group == "imbalance":
        return {"imb_subset": True, "imb_ratio": float(variant.split("=")[1])}
    if group == "pilot":
        return dict(PILOT_GRID[variant])
    raise KeyError(f"unknown group {group}")


SENSITIVITY_TYPES = {}       # every sensitivity key is a float


@dataclass
class Run:
    group: str
    method: str
    variant: str
    task: str
    seed: int
    arch: str = ""
    overrides: dict = field(default_factory=dict)

    def run_dir(self, results_dir) -> Path:
        return (resolve_path(results_dir) / "runs" / self.group / self.method / self.variant / self.arch
                / self.task / f"seed{self.seed}")

    def args(self):
        return ["--group", self.group, "--method", self.method, "--variant", self.variant,
                "--task", self.task, "--seed", str(self.seed), "--arch", self.arch]


GROUPS = ["main", "baselines", "ablation_losses", "ablation_mechanisms", "backbone", "imbalance",
          "sensitivity", "threshold_range", "baselines_bn_train", "timing", "pilot"]
# Complexity table (GPU memory, seconds per adaptation epoch): every method on these tasks, seed 0.
# Launch it with --jobs-per-gpu 1 so that no other job shares the GPU during the measurement.
TIMING_TASKS = ["W2E"]
DEFAULT_GROUPS = ["main", "baselines", "ablation_losses", "ablation_mechanisms", "backbone",
                  "imbalance", "sensitivity"]


def build_runs(group: str, seeds=None, tasks=None, imbalance_methods=None) -> list[Run]:
    seeds = list(seeds or SEEDS)
    tasks = list(tasks or all_tasks())
    main_arch = load_config().arch
    runs = []
    if group == "main":
        runs = [Run("main", "cab_sfda", "full", t, s) for t in tasks for s in seeds]
    elif group in ("baselines", "baselines_bn_train"):
        runs = [Run(group, m, "default", t, s) for m in BASELINES for t in tasks for s in seeds]
    elif group == "ablation_losses":
        runs = [Run(group, "cab_sfda", v, t, s) for v in LOSS_ABLATION for t in tasks for s in seeds]
    elif group == "ablation_mechanisms":
        runs = [Run(group, "cab_sfda", v, t, s) for v in MECHANISM_ABLATION for t in tasks for s in seeds]
    elif group == "backbone":
        runs = [Run(group, "cab_sfda", a, t, s, arch=a) for a in ALL_BACKBONES if a != main_arch
                for t in tasks for s in seeds]
    elif group == "sensitivity":
        runs = [Run(group, "cab_sfda", f"{k}={_fmt(v)}", t, s)
                for k, vals in SENSITIVITY.items() for v in vals for t in tasks for s in seeds]
    elif group == "threshold_range":
        runs = [Run(group, "cab_sfda", f"tau_min={_fmt(v)}", t, s)
                for v in THRESHOLD_RANGE["tau_min"] for t in tasks for s in seeds]
    elif group == "pilot":
        p_tasks = [t for t in PILOT_TASKS if t in tasks] or tasks
        runs = [Run(group, "cab_sfda", v, t, seeds[0]) for v in PILOT_GRID for t in p_tasks]
    elif group == "timing":
        t_tasks = [t for t in TIMING_TASKS if t in tasks] or tasks[:1]
        runs = [Run(group, m, "default", t, seeds[0]) for m in ["cab_sfda", *BASELINES] for t in t_tasks]
    elif group == "imbalance":
        methods = imbalance_methods or ["cab_sfda", *BASELINES]
        imb_tasks = [t for t in CROSS_SCENE_MLRSN_TARGETS if t in tasks]
        runs = [Run(group, m, f"ratio={_fmt(r)}", t, s)
                for m in methods for r in IMB_RATIOS for t in imb_tasks for s in seeds]
    else:
        raise KeyError(f"unknown group {group}; known: {GROUPS}")
    for r in runs:
        r.overrides = variant_overrides(r.group, r.variant)
        r.arch = r.arch or main_arch
    return runs


def source_jobs(runs: list[Run]):
    """Unique (source domain, arch, seed) needed by a list of runs."""
    need = sorted({(task_info(r.task)["source"], r.arch, r.seed) for r in runs})
    return need


def benchmark_tasks():
    ds = load_datasets()
    out = {b: [t for t, (_, _, bb) in ds["tasks"].items() if bb == b] for b in ds["benchmarks"]}
    out["cross_sensor"] = out["nwpu_nasc"] + out["whu_eurosat"]
    return out
