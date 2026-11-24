#!/usr/bin/env python
"""Audit every result file for internal consistency before numbers go into the paper.

  python check_results.py            # exit code 1 if any check fails

Checks: metrics recomputed from the confusion matrix; per-class recalls are attainable with
the test-set class sizes; the CAB-SFDA accepted-sample counts equal the sum of per-epoch
acceptances and stay below the number of samples seen; the run folder matches its stored
config; every method sees the same source model and source-only accuracy for a given
(task, seed, backbone); every group has the same seeds for all tasks.
"""
import sys
from collections import defaultdict

import numpy as np

from cabsfda.config import load_config
from cabsfda.experiments import variant_overrides
from cabsfda.results import load_results


def main():
    cfg = load_config()
    df, full = load_results(cfg.results_dir)
    if df.empty:
        print("no results found")
        return 0
    fails, n_checks = [], 0

    def check(ok, msg):
        nonlocal n_checks
        n_checks += 1
        if not ok:
            fails.append(msg)

    src_seen = defaultdict(set)
    for row in df.itertuples():
        r, d = full[row.run_dir], row.run_dir
        fin = r["final"]
        cm = np.array(fin["confusion"])
        check(cm.sum() == r["n_test"] == sum(r["test_class_counts"]), f"{d}: test size mismatch")
        check(abs(100 * np.trace(cm) / cm.sum() - fin["oa"]) < 1e-6, f"{d}: OA != trace/total")
        rec = fin["per_class"]["recall"]
        sup = fin["per_class"]["support"]
        present = [i for i, s in enumerate(sup) if s > 0]
        check(abs(np.mean([rec[i] for i in present]) - fin["balanced_acc"]) < 1e-6, f"{d}: macro recall mismatch")
        for i in present:
            k = rec[i] * sup[i] / 100
            check(abs(k - round(k)) < 1e-6, f"{d}: class {i} recall {rec[i]} not attainable with {sup[i]} images")
        conf = r["config"]
        check(conf["arch"] == row.arch, f"{d}: folder arch {row.arch} != config {conf['arch']}")
        try:
            for k, v in variant_overrides(row.group, row.variant).items():
                check(conf[k] == v, f"{d}: override {k}={v} not applied (config has {conf[k]})")
        except KeyError:
            pass
        src_seen[(row.task, row.seed, row.arch)].add((r["source_model"], round(r["source_only"]["oa"], 6)))
        if r["method"] == "cab_sfda":
            g = r["diagnostics"]
            check(g["sum_counts"] <= g["sum_counts_cap"], f"{d}: accepted count {g['sum_counts']} > cap {g['sum_counts_cap']}")
            acc = sum(h["n_accepted"] for h in r["history"])
            if conf["memory_update"] == "gated" and conf["memory_mode"] != "none":
                check(acc == g["sum_counts"], f"{d}: memory counts {g['sum_counts']} != accepted samples {acc}")
            check(0 <= g["accept_rate_final_epoch"] <= 100, f"{d}: acceptance rate out of range")
            check(len(r["history"]) == conf["epochs"], f"{d}: {len(r['history'])} epochs logged, {conf['epochs']} configured")
    for key, vals in src_seen.items():
        check(len(vals) == 1, f"{key}: runs disagree on source model / source-only OA: {vals}")
    for (g, m, v, a), sub in df.groupby(["group", "method", "variant", "arch"]):
        seed_sets = {t: frozenset(s.seed) for t, s in sub.groupby("task")}
        check(len(set(seed_sets.values())) == 1, f"{g}/{m}/{v}/{a}: tasks have different seeds {dict(seed_sets)}")

    # collapse / degradation warnings (not failures: they describe results, not bookkeeping errors)
    warns = []
    for row in df.itertuples():
        r = full[row.run_dir]
        pce = r.get("unsupervised", {}).get("pred_class_entropy")
        if pce is not None and pce < 0.5:
            warns.append(f"{row.run_dir}: COLLAPSE (predicted-class entropy {pce:.2f} on the adaptation images)")
        elif row.oa < row.src_oa - 10:
            warns.append(f"{row.run_dir}: adapted OA {row.oa:.1f} is {row.src_oa - row.oa:.1f} pp below source-only")
    print(f"{len(df)} runs, {n_checks} checks, {len(fails)} failed; {len(warns)} collapse/degradation warnings")
    for f in fails[:200]:
        print("  FAIL", f)
    for g, sub in df.groupby(["group", "method", "variant"]):
        n = sum(1 for w in warns if any(w.startswith(d) for d in sub.run_dir))
        if n:
            print(f"  WARN {'/'.join(g)}: {n} of {len(sub)} runs collapsed or fell >10 pp below source-only")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
