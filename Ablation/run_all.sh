#!/usr/bin/env bash
# run_all.sh — Run all ablations in sequence.
set -e

echo "==> 1. Source pretraining"
python train_source.py

echo "==> 2. Loss-wise ablation"
python ablation_losses.py

echo "==> 3. Mechanism-wise ablation"
python ablation_mechanisms.py

echo "==> 4. Backbone ablation"
python ablation_backbone.py

echo "==> 5. Imbalance sensitivity"
python ablation_imbalance.py

echo "==> 6. PGAT diagnostics + class weight evolution"
python diagnostics_pgat.py

echo "==> 7. Per-class F1 table"
python per_class_metrics.py

echo "All ablations done."
