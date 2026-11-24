#!/usr/bin/env python
"""Run a single experiment (or train a single source model).

Examples
  python run.py --task A2C --seed 0                                  # CAB-SFDA, main configuration
  python run.py --task A2C --seed 0 --method shot --group baselines  # a baseline
  python run.py --group ablation_mechanisms --variant C_fixed_085 --task E2W --seed 1
  python run.py --source AID --seed 0 --arch resnet50                # source model only
  python run.py --task A2C --seed 0 --set lr=1e-4 --variant my_test --group scratch

Finished runs are skipped (their result.json exists). Use launch.py to run whole groups on
several GPUs.
"""
import argparse
import sys

from cabsfda.config import load_config, parse_overrides
from cabsfda.experiments import Run, variant_overrides
from cabsfda.runner import run_one
from cabsfda.source import train_source


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task")
    ap.add_argument("--source", help="train the source model of this domain and exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="cab_sfda")
    ap.add_argument("--group", default="main")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra config overrides")
    ap.add_argument("--defaults", default=None, help="alternative defaults.yaml")
    a = ap.parse_args(argv)

    extra = parse_overrides(a.set)
    if a.source:
        cfg = load_config({**({"arch": a.arch} if a.arch else {}), **extra}, a.defaults)
        train_source(cfg, a.source, a.seed)
        return 0
    if not a.task:
        ap.error("--task or --source is required")

    variant = a.variant or {"main": "full", "baselines": "default", "baselines_bn_train": "default"}.get(a.group)
    if variant is None:
        ap.error("--variant is required for this group")
    try:
        overrides = variant_overrides(a.group, variant)
    except KeyError:
        if not extra:
            raise
        overrides = {}                     # free-form group: overrides come from --set only
    arch = a.arch or (variant if a.group == "backbone" else None)
    overrides = {**overrides, **({"arch": arch} if arch else {}), **extra}
    cfg = load_config(overrides, a.defaults)
    if a.group == "main" and a.seed == 0:
        cfg.save_features = True           # t-SNE material for the main runs
    run = Run(a.group, a.method, variant, a.task, a.seed, cfg.arch, overrides)
    run_one(cfg, a.task, a.seed, a.method, run.run_dir(cfg.results_dir),
            meta={"group": a.group, "variant": variant, "overrides": overrides})
    return 0


if __name__ == "__main__":
    sys.exit(main())
