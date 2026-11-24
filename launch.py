#!/usr/bin/env python
"""Run whole experiment groups on one or more GPUs, resumably.

  python launch.py --groups main,baselines --gpus 0,1,2,3 --jobs-per-gpu 2
  python launch.py --groups all --gpus 0,1                 # every default group
  python launch.py --groups main --gpus 0 --dry-run        # list what would run

Stage 1 trains every missing source model, stage 2 runs the adaptation jobs. Runs whose
result.json already exists are skipped, so the command can be re-launched after an
interruption. Each job's output goes to <results_dir>/logs/...; failed jobs are listed at the
end and in <results_dir>/launch_status.json (re-launching retries them).
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from cabsfda.config import load_config, parse_overrides, resolve_path
from cabsfda.experiments import DEFAULT_GROUPS, GROUPS, build_runs, source_jobs
from cabsfda.source import source_dir
from cabsfda.utils import save_json

HERE = Path(__file__).resolve().parent


def run_pool(jobs, slots, results_dir, label, jobs_per_gpu=1):
    """jobs: list of (name, argv). slots: list of CUDA device strings (or None for CPU)."""
    if not jobs:
        return []
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    failed, lock, done = [], threading.Lock(), [0]
    logs = resolve_path(results_dir) / "logs"

    def worker(dev):
        while True:
            try:
                name, argv = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ)
            env["CABSFDA_JOBS_PER_GPU"] = str(jobs_per_gpu)      # recorded with the timings
            if dev is not None:
                env["CUDA_VISIBLE_DEVICES"] = dev
            else:                               # CPU: share the cores between parallel jobs
                env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // len(slots))))
            log_file = logs / (name.replace("/", "__") + ".log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(log_file, "w") as fh:
                rc = subprocess.call([sys.executable, str(HERE / "run.py"), *argv], cwd=HERE, env=env,
                                     stdout=fh, stderr=subprocess.STDOUT)
            with lock:
                done[0] += 1
                status = "ok" if rc == 0 else f"FAILED (rc={rc}, see {log_file})"
                print(f"[{label} {done[0]}/{len(jobs)}] {name} on {dev or 'cpu'} "
                      f"{time.time() - t0:.0f}s {status}", flush=True)
                if rc != 0:
                    failed.append({"job": name, "log": str(log_file)})

    threads = [threading.Thread(target=worker, args=(d,), daemon=True) for d in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default="main", help=f"comma list from {GROUPS} or 'all'")
    ap.add_argument("--gpus", default="0", help="comma list of GPU ids, or 'cpu'")
    ap.add_argument("--jobs-per-gpu", type=int, default=1)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tasks", default=None, help="comma list (default: all 16 tasks)")
    ap.add_argument("--imbalance-methods", default=None, help="comma list (default: CAB-SFDA + all baselines)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="overrides passed to every job (e.g. num_workers=4)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    groups = DEFAULT_GROUPS if a.groups == "all" else a.groups.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    tasks = a.tasks.split(",") if a.tasks else None
    imb_methods = a.imbalance_methods.split(",") if a.imbalance_methods else None
    extra = parse_overrides(a.set)
    if "arch" in extra or "results_dir" in extra:
        ap.error("set arch/results_dir in configs/defaults.yaml or via CABSFDA_SET, not with --set")
    cfg = load_config(extra)
    set_args = ["--set", *a.set] if a.set else []

    runs = [r for g in groups for r in build_runs(g, seeds, tasks, imb_methods)]
    todo = [r for r in runs if not (r.run_dir(cfg.results_dir) / "result.json").exists()]
    src = [(d, arch, s) for d, arch, s in source_jobs(todo)
           if not (source_dir(_with_arch(cfg, arch), d, s) / "model.pt").exists()]
    print(f"{len(runs)} runs in {groups}: {len(runs) - len(todo)} done, {len(todo)} to run; "
          f"{len(src)} source models to train", flush=True)
    if a.dry_run:
        for d, arch, s in src:
            print(f"  source {d} {arch} seed{s}")
        for r in todo[:50]:
            print("  " + " ".join(r.args()))
        if len(todo) > 50:
            print(f"  ... and {len(todo) - 50} more")
        return 0

    slots = ([None] if a.gpus == "cpu" else a.gpus.split(",")) * a.jobs_per_gpu
    t0 = time.time()
    failed = run_pool([(f"source/{d}/{arch}/seed{s}", ["--source", d, "--seed", str(s), "--arch", arch, *set_args])
                       for d, arch, s in src], slots, cfg.results_dir, "source", a.jobs_per_gpu)
    if failed:
        print("source training failed; fix and re-launch:", *failed, sep="\n  ")
        return 1
    jobs = [("/".join([r.group, r.method, r.variant, r.arch, r.task, f"seed{r.seed}"]), r.args() + set_args)
            for r in todo]
    failed = run_pool(jobs, slots, cfg.results_dir, "run", a.jobs_per_gpu)
    save_json({"groups": groups, "seeds": seeds, "n_runs": len(runs), "n_launched": len(jobs),
               "failed": failed, "hours": (time.time() - t0) / 3600},
              resolve_path(cfg.results_dir) / "launch_status.json")
    print(f"finished in {(time.time() - t0) / 3600:.2f} h; {len(failed)} failed")
    for f in failed:
        print("  FAILED", f["job"], "->", f["log"])
    return 1 if failed else 0


def _with_arch(cfg, arch):
    c = cfg.copy()
    c.arch = arch
    return c


if __name__ == "__main__":
    sys.exit(main())
