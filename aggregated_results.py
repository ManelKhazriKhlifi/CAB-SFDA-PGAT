import argparse
import json
import os
import numpy as np


GROUPS = {
    "cross_scene": [
        "A2C", "A2M", "A2R", "C2A", "C2M", "C2R",
        "M2A", "M2C", "M2R", "R2A", "R2C", "R2M",
    ],
    "nwpu_nasc":   ["N2G", "G2N"],
    "eurosat_whu": ["E2W", "W2E"],
}
TASKS = [t for ts in GROUPS.values() for t in ts]
SEEDS = [0, 1, 2]
RUN_DIR = os.environ.get("RESULTS_DIR", "./Results_PGAT")


def load_seed_metrics(task, seed, method="cab_sfda"):
    path = os.path.join(RUN_DIR, task, f"seed_{seed}", method, "final_evaluation.json")
    with open(path) as f:
        return json.load(f)["metrics"]


def summarize(task, seeds=SEEDS, method="cab_sfda"):
    oa, bacc, f1 = [], [], []
    for s in seeds:
        m = load_seed_metrics(task, s, method)
        oa.append(m["overall_acc"] * 100)
        bacc.append(m["balanced_acc"] * 100)
        f1.append(m["macro_f1"] * 100)
    return {
        "task": task,
        "seeds": seeds,
        "oa":       {"mean": float(np.mean(oa)),
                     "std":  float(np.std(oa, ddof=1)),
                     "per_seed": oa},
        "bacc":     {"mean": float(np.mean(bacc)),
                     "std":  float(np.std(bacc, ddof=1)),
                     "per_seed": bacc},
        "macro_f1": {"mean": float(np.mean(f1)),
                     "std":  float(np.std(f1, ddof=1)),
                     "per_seed": f1},
    }


def group_average(aggregates):
    """Per seed: average over tasks; then mean ± std of those averages across seeds."""
    avg = {"task": "AVG", "seeds": next(iter(aggregates.values()))["seeds"]}
    for k in ("oa", "bacc", "macro_f1"):
        per_seed = np.mean([a[k]["per_seed"] for a in aggregates.values()], axis=0)
        avg[k] = {"mean": float(np.mean(per_seed)),
                  "std":  float(np.std(per_seed, ddof=1)),
                  "per_seed": per_seed.tolist()}
    return avg


def save(aggregates, out):
    with open(out, "w") as f:
        json.dump(aggregates, f, indent=2)
    print(f"\nSaved: {out}")

    csv_path = os.path.splitext(out)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("task,oa_mean,oa_std,bacc_mean,bacc_std,f1_mean,f1_std\n")
        for t, a in aggregates.items():
            f.write(f"{t},"
                    f"{a['oa']['mean']:.4f},{a['oa']['std']:.4f},"
                    f"{a['bacc']['mean']:.4f},{a['bacc']['std']:.4f},"
                    f"{a['macro_f1']['mean']:.4f},{a['macro_f1']['std']:.4f}\n")
    print(f"Saved: {csv_path}")


def print_table(aggregates):
    header = (f"\n{'Task':<6}"
              f"{'OA (mean ± std)':>24}"
              f"{'BAcc (mean ± std)':>26}"
              f"{'MacroF1 (mean ± std)':>26}")
    print(header)
    print("-" * 82)
    for task, a in aggregates.items():
        print(f"{task:<6}"
              f"{a['oa']['mean']:>12.2f} ± {a['oa']['std']:.2f}"
              f"{a['bacc']['mean']:>14.2f} ± {a['bacc']['std']:.2f}"
              f"{a['macro_f1']['mean']:>14.2f} ± {a['macro_f1']['std']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, choices=TASKS + [None])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default="aggregated_results.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--method", default="",
                        help="optional subfolder under seed_N/ (e.g. cab_sfda | shot | nrc ...); "
                             "empty = files directly in seed_N/")
    args = parser.parse_args()

    if args.all:
        # one output file per group: <out>_cross_scene.json/.csv, <out>_nwpu_nasc..., <out>_eurosat_whu...
        runs = {g: (ts, f"{os.path.splitext(args.out)[0]}_{g}.json") for g, ts in GROUPS.items()}
    elif args.task:
        runs = {args.task: ([args.task], args.out)}
    else:
        parser.error("Provide --task TASK or --all")

    for name, (tasks, out) in runs.items():
        aggregates = {}
        for t in tasks:
            try:
                aggregates[t] = summarize(t, seeds=args.seeds, method=args.method)
            except FileNotFoundError as e:
                print(f"[skip] {t}: {e}")

        if not aggregates:
            print(f"No results found for {name}.")
            continue

        if len(aggregates) > 1:
            aggregates["AVG"] = group_average(aggregates)

        print(f"\n=== {name} ===")
        print_table(aggregates)
        save(aggregates, out)


if __name__ == "__main__":
    main()