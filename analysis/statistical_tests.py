import json
import os
import numpy as np
from scipy import stats

TASKS     = ["A2C", "A2M", "A2R", "C2A", "C2M", "C2R",
             "M2A", "M2C", "M2R", "R2A", "R2C", "R2M",
             "N2G", "G2N", "E2W", "W2E"]
SEEDS     = [0, 1, 2]
# display name -> folder name used by run_baseline.py --method
BASELINES = {"SHOT": "shot", "NRC": "nrc", "CoWA": "cowa", "MBNS": "mbns",
             "G-SFDA": "gsfda", "AaD": "aad", "DaC": "dac", "SMNA": "smna"}
OURS      = "cab_sfda"
RUN_DIR   = "./runs"


def load_oa(task, seed, method=OURS):
    path = f"{RUN_DIR}/{task}/seed_{seed}/{method}/final_evaluation.json"
    with open(path) as f:
        d = json.load(f)
    # guard against files written by a different method
    if d.get("method", method) != method:
        raise ValueError(f"{path} was written by {d['method']!r}, not {method!r}")
    return d["metrics"]["overall_acc"] * 100


def paired_ttest(baseline, task):
    ours   = np.array([load_oa(task, s)           for s in SEEDS])
    theirs = np.array([load_oa(task, s, BASELINES[baseline]) for s in SEEDS])
    t, p = stats.ttest_rel(ours, theirs)
    return float(t), float(p)


def main():
    os.makedirs("analysis", exist_ok=True)
    results = {}
    for task in TASKS:
        results[task] = {}
        for base in BASELINES:
            try:
                t, p = paired_ttest(base, task)
                results[task][base] = {"t": t, "p": p}
            except (FileNotFoundError, ValueError) as e:
                results[task][base] = {"t": None, "p": None, "note": str(e)}

    header = f"{'Baseline':<10}" + "".join(f"{t:>12}" for t in TASKS)
    print(header); print("-" * len(header))
    for base in BASELINES:
        row = f"{base:<10}"
        for task in TASKS:
            p = results[task][base]["p"]
            cell = "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")
            row += f"{cell:>12}"
        print(row)

    with open("analysis/pvalues.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("analysis/pvalues.csv", "w") as f:
        f.write("task,baseline,t_stat,p_value\n")
        for task in TASKS:
            for base in BASELINES:
                r = results[task][base]
                if r["p"] is not None:
                    f.write(f"{task},{base},{r['t']:.6f},{r['p']:.6e}\n")

    print("Saved analysis/pvalues.json and .csv")


if __name__ == "__main__":
    main()
