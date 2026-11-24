#!/usr/bin/env bash
# End-to-end test of the whole pipeline on a tiny synthetic dataset. Every experiment group,
# every method and every ablation variant is run for a few tasks with a ResNet-18 on 32x32
# images for 2 epochs, then the audit, tables and figures are produced. Numbers are meaningless;
# the test only checks that everything runs. Writes only under smoke/ (delete it afterwards).
#
#   bash smoke_test.sh            # full check (~30 min on 2 CPU cores, a few minutes on a GPU: DEV=0)
#   QUICK=1 bash smoke_test.sh    # 2 tasks, main + baselines + ablations (~5 min on CPU)
set -euo pipefail
cd "$(dirname "$0")"

export DATA_ROOT=$PWD/smoke/raw
export CABSFDA_SET="results_dir=smoke/results splits_dir=smoke/splits arch=resnet18 pretrained=false \
img_size=32 src_epochs=2 epochs=2 batch_size=16 src_batch_size=16 num_workers=0 amp=false \
proto_min_count=5 imb_base=8"
DEV=${DEV:-cpu}            # DEV=0 to use GPU 0
JOBS=${JOBS:-2}
L="python launch.py --gpus $DEV --jobs-per-gpu $JOBS"

rm -rf smoke
python make_synthetic_data.py --out "$DATA_ROOT" --n-per-class 16 --size 32
python prepare_data.py --force
# released-split round trip: export the splits, rebuild them from the reference files, compare
python export_splits.py --src smoke/splits --out smoke/splits_export
python prepare_data.py --reference smoke/splits_export --out smoke/splits_ref --force > /dev/null
python - <<'PY'
import csv, glob, os
for f in sorted(glob.glob("smoke/splits/*/*.csv")):
    g = f.replace("smoke/splits/", "smoke/splits_ref/")
    a = {r["sha1"] for r in csv.DictReader(open(f))}
    b = {r["sha1"] for r in csv.DictReader(open(g))}
    assert a == b, f"reference split differs: {f}"
print("released-split round trip OK")
PY
python tests/test_core.py

if [[ "${QUICK:-0}" == "1" ]]; then
  $L --groups main,baselines --seeds 0,1 --tasks E2W,W2E
  $L --groups ablation_mechanisms,ablation_losses --seeds 0 --tasks E2W
else
  $L --groups main,baselines --seeds 0,1                                   # all 16 tasks
  $L --groups ablation_losses,ablation_mechanisms --seeds 0,1 --tasks A2M,E2W,W2E,N2G,G2N
  $L --groups backbone --seeds 0 --tasks E2W
  $L --groups imbalance --seeds 0 --tasks A2M,C2M,R2M --imbalance-methods cab_sfda,smna,shot
  $L --groups baselines_bn_train --seeds 0 --tasks E2W
  $L --groups sensitivity,threshold_range --seeds 0 --tasks E2W,W2E
fi

CABSFDA_NO_SELECTED=1 $L --groups pilot --seeds 0 --tasks E2W
CABSFDA_NO_SELECTED=1 python select_hparams.py --dry-run
python check_results.py
python aggregate.py --out smoke/tables
python make_figures.py --out smoke/figures
echo "SMOKE TEST PASSED - outputs in smoke/tables and smoke/figures"
