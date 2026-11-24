#!/usr/bin/env bash
# Full reproduction of every experiment in the paper.
#
#   conda activate cabsfda
#   DATA_ROOT=/path/to/datasets GPUS=0,1,2,3 JOBS_PER_GPU=2 bash run_all.sh     # raw datasets
#   SPLIT_SRC=/path/to/split/folders GPUS=0,1 bash run_all.sh                   # pre-split folders
#
# The splits of the paper are rebuilt from the released files in splits/ when that folder exists
# (prepare_data.py --reference splits); otherwise they are built with the same procedure.
# configs/selected.yaml (label-free pilot of the paper) is used as is; delete it to redo the pilot.
# Optional: EXTRA=1 also runs the appendix groups (baselines with BN statistics updated on the
# target, threshold-range study).
# Safe to re-run after an interruption: splits are not rebuilt and finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")"

GPUS=${GPUS:-0}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
SEEDS=${SEEDS:-0,1,2}
WORKERS=${WORKERS:-6}                  # DataLoader workers per job
L="python launch.py --gpus $GPUS --jobs-per-gpu $JOBS_PER_GPU --seeds $SEEDS --set num_workers=$WORKERS"

echo "== 1/7 data splits"
if [[ -f data/splits/summary.json ]]; then
  echo "data/splits exists (delete it to rebuild)"
elif [[ -n "${SPLIT_SRC:-}" ]]; then
  python import_splits.py --src "$SPLIT_SRC" --resize-cache 128
elif [[ -d splits ]]; then
  python prepare_data.py --reference splits --resize-cache 128
else
  python prepare_data.py --resize-cache 128
fi

echo "== 2/7 label-free pilot (SND) -> configs/selected.yaml"
if [[ -f configs/selected.yaml ]]; then
  echo "configs/selected.yaml exists (delete it to redo the selection)"
else
  CABSFDA_NO_SELECTED=1 $L --seeds 0 --groups pilot
  CABSFDA_NO_SELECTED=1 python select_hparams.py
fi

echo "== 3/7 main results + baselines";  $L --groups main,baselines
echo "   collapse check";                  python check_results.py --groups main baselines
echo "== 4/7 ablations + backbones";     $L --groups ablation_losses,ablation_mechanisms,backbone
echo "== 5/7 imbalance + sensitivity";   $L --groups imbalance,sensitivity
echo "== 6/7 timing (one job per GPU)"
python launch.py --gpus "$GPUS" --jobs-per-gpu 1 --seeds 0 --set num_workers=$WORKERS --groups timing
if [[ "${EXTRA:-0}" == "1" ]]; then
  echo "== extra: appendix groups";       $L --groups baselines_bn_train,threshold_range
fi
echo "== 7/7 audit, tables and figures"; python check_results.py; python aggregate.py && python make_figures.py
echo "done: tables/ figures/ (provenance in tables/provenance.csv)"
