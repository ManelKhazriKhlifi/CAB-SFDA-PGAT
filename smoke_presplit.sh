#!/usr/bin/env bash
# Smoke test of the pre-split-folder workflow on synthetic data
# (<dir>/{source,target}_<name>_split/{train,test}/<Class>/): audit, disjoint re-split by content
# and a short run of every method. Numbers are meaningless. Writes only under smoke_presplit/.
#
#   bash smoke_presplit.sh        # a few minutes on CPU; DEV=0 to use GPU 0
set -euo pipefail
cd "$(dirname "$0")"
DEV=${DEV:-cpu}
JOBS=${JOBS:-2}
L="python launch.py --gpus $DEV --jobs-per-gpu $JOBS"
FAKE=smoke_presplit/folders

rm -rf smoke_presplit
python make_synthetic_data.py --out $FAKE --layout splitfolders --n-per-class 16 --size 32
python import_splits.py --src $FAKE --out smoke_presplit/splits --audit-only
grep -q "TEST IMAGES ALSO IN TRAIN" smoke_presplit/splits/audit.txt || { echo "overlap not detected"; exit 1; }
python import_splits.py --src $FAKE --out smoke_presplit/splits --resize-cache 32 --cache-dir smoke_presplit/cache --force

export CABSFDA_SET="results_dir=smoke_presplit/results splits_dir=smoke_presplit/splits arch=resnet18 \
pretrained=false img_size=32 src_epochs=2 epochs=2 batch_size=16 src_batch_size=16 num_workers=0 amp=false \
proto_min_count=5 imb_base=8"
$L --groups main,baselines --seeds 0,1 --tasks E2W,W2E
python check_results.py
python aggregate.py --out smoke_presplit/tables
echo "PRE-SPLIT SMOKE TEST PASSED - outputs in smoke_presplit/"
