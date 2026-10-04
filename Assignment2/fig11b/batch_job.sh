#!/usr/bin/env bash
# UCloud "Batch mode" script: runs (a shard of) the full grid, then collects and plots.
# To split the grid over K UCloud jobs, make K copies of this file with
# SHARD=0..K-1 and NUM_SHARDS=K and submit one job per copy.
set -eo pipefail

SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

FIG11B_DIR="${FIG11B_DIR:-$(dirname "$(find /work -maxdepth 5 -path '*/fig11b/env.sh' 2>/dev/null | head -n1)")}"
source "$FIG11B_DIR/env.sh"
nvidia-smi || true

python "$SCRIPT_DIR/run_grid.py" \
    --sizes $SIZES --n-min "$N_MIN" --n-max "$N_MAX" --n-step "$N_STEP" --seed "$TRAIN_SEED" \
    --jobs-per-gpu "$JOBS_PER_GPU" --shard "$SHARD" --num-shards "$NUM_SHARDS" || true

python "$SCRIPT_DIR/collect_results.py"
python "$SCRIPT_DIR/plot_fig11b.py"
