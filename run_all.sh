#!/usr/bin/env bash
# run_all.sh — Run CAB-SFDA for all tasks × three seeds, then aggregate.
#
# Usage:
#     bash run_all.sh                    # all tasks, all seeds
#     bash run_all.sh A2C                # one task, all seeds
#     bash run_all.sh A2C C2A            # two tasks, all seeds

set -e

export DATA_ROOT="${DATA_ROOT:-D:/Downloads/data}"

ALL_TASKS=(A2C A2M A2R C2A C2M C2R M2A M2C M2R R2A R2C R2M N2G G2N E2W W2E)
SEEDS=(0 1 2)

if [ "$#" -eq 0 ]; then
    TASKS=("${ALL_TASKS[@]}")
else
    TASKS=("$@")
fi

for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "=============================================="
        echo "TASK=$TASK  SEED=$SEED"
        echo "=============================================="
        TASK="$TASK" SEED="$SEED" python CAB_SFDA_PGAT.py
    done
done

echo "ALL RUNS FINISHED."