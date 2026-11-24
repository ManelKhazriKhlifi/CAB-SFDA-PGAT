#!/usr/bin/env python
"""Label-free hyper-parameter selection for CAB-SFDA from the `pilot` runs.

  python launch.py --groups pilot --gpus 0,1,2,3,4,5 --jobs-per-gpu 2
  python select_hparams.py            # prints the ranking, writes configs/selected.yaml

Each pilot configuration (cabsfda/experiments.py: PILOT_GRID) is run on PILOT_TASKS with seed 0.
Configurations are ranked by their mean rank of SND (Soft Neighborhood Density, Saito et al.,
ICCV 2021) over the pilot tasks; SND is computed on the unlabeled adaptation images. A
configuration is excluded if any pilot run collapses (predicted-class entropy < --min-pce on the
adaptation images). Target labels are never read: the test accuracy is not loaded unless
--show-oa is given, and even then it is not used for the choice.

The chosen overrides go to configs/selected.yaml, which every later run reads on top of
configs/defaults.yaml (delete the file, or CABSFDA_NO_SELECTED=1, to go back to the defaults).
"""
import argparse

import numpy as np
import yaml

from cabsfda.config import REPO_ROOT, load_config
from cabsfda.experiments import PILOT_GRID, PILOT_TASKS
from cabsfda.results import load_results, select


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-pce", type=float, default=0.8, help="collapse threshold on predicted-class entropy")
    ap.add_argument("--show-oa", action="store_true", help="also print test OA (NOT used for the choice)")
    ap.add_argument("--dry-run", action="store_true", help="do not write configs/selected.yaml")
    a = ap.parse_args()
    cfg = load_config()
    df, full = load_results(cfg.results_dir)
    pil = select(df, group="pilot") if not df.empty else df
    if pil.empty:
        raise SystemExit("no pilot runs: python launch.py --groups pilot ...")
    tasks = [t for t in PILOT_TASKS if t in set(pil.task)]
    snd, pce = {}, {}
    for v in PILOT_GRID:
        for t in tasks:
            sub = pil[(pil.variant == v) & (pil.task == t)]
            if sub.empty:
                continue
            u = full[sub.iloc[0].run_dir].get("unsupervised", {})
            snd[v, t], pce[v, t] = u.get("snd", np.nan), u.get("pred_class_entropy", np.nan)
    complete = [v for v in PILOT_GRID if all((v, t) in snd for t in tasks)]
    missing = [v for v in PILOT_GRID if v not in complete]
    ranks = {v: [] for v in complete}
    for t in tasks:
        order = sorted(complete, key=lambda v: -snd[v, t])          # higher SND = better
        for r, v in enumerate(order, 1):
            ranks[v].append(r)
    rows = []
    for v in complete:
        collapsed = min(pce[v, t] for t in tasks) < a.min_pce
        rows.append((collapsed, float(np.mean(ranks[v])), v))
    rows.sort()
    head = f"{'config':10s} {'mean rank':>9s}  " + "  ".join(f"{t:>12s}" for t in tasks) + "  collapse"
    print("SND (higher is better) and predicted-class entropy per pilot task; no target labels used\n")
    print(head)
    for collapsed, mr, v in rows:
        cells = "  ".join(f"{snd[v, t]:6.3f}/{pce[v, t]:4.2f}" for t in tasks)
        print(f"{v:10s} {mr:9.2f}  {cells}  {'YES' if collapsed else 'no'}")
    if missing:
        print(f"\nincomplete (not ranked): {', '.join(missing)}")
    if a.show_oa:
        print("\nTest OA (for reference only, NOT used for the choice):")
        for _, _, v in rows:
            print(f"{v:10s} " + "  ".join(f"{float(pil[(pil.variant == v) & (pil.task == t)].oa.iloc[0]):12.2f}"
                                          for t in tasks))
    ok = [r for r in rows if not r[0]]
    if not ok:
        raise SystemExit("\nevery configuration collapsed on some pilot task - do not launch the full run")
    best = ok[0][2]
    print(f"\nselected: {best}  overrides {PILOT_GRID[best] or '(defaults)'}")
    if not a.dry_run:
        out = REPO_ROOT / "configs" / "selected.yaml"
        out.write_text(yaml.safe_dump({
            "selected": best, "overrides": PILOT_GRID[best], "criterion": "mean rank of SND over pilot tasks",
            "pilot_tasks": tasks,
            "snd": {v: {t: float(snd[v, t]) for t in tasks} for v in complete}}, sort_keys=False))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
