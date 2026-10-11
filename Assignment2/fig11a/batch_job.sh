#!/usr/bin/env bash
# UCloud batch-mode script: one instance, one NVIDIA B200.
# Mount the project folder (the directory that contains fig11a/) under /work.
#
# The grid is 60 widths x 3 sample sizes = 180 runs of 500k SGD steps.
# Split it by uploading one copy of this script per instance, each with its
# own SHARD, or generate those copies with make_batch_scripts.sh.
# A killed job resumes: checkpoints and finished JSON results live on /work.
#
# SHARD is 0-based. NUM_SHARDS is the number of instances you launch.
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

set -eo pipefail

if [ -z "${FIG11A_DIR:-}" ]; then
    _env_sh="$(find /work -maxdepth 5 -path '*/fig11a/env.sh' -print -quit 2>/dev/null || true)"
    FIG11A_DIR="$(dirname "$_env_sh")"
fi
if [ -z "$FIG11A_DIR" ] || [ ! -f "$FIG11A_DIR/env.sh" ]; then
    echo "fig11a/env.sh not found under /work. Mount the project folder that contains fig11a/." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$FIG11A_DIR/config.sh"

file_mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }

wait_for_dir() {
    # $1 lock dir, $2 path that means success, $3 command to run while holding the lock.
    # Several instances share /work and may enter this at the same time; mkdir is the claim.
    local lock="$1" ready="$2" cmd="$3" i now mt
    mkdir -p "$(dirname "$lock")"
    for i in $(seq 1 720); do
        if [ -e "$ready" ]; then
            return 0
        fi
        if [ -d "$lock" ]; then
            now="$(date +%s)"
            mt="$(file_mtime "$lock")"
            # Steal only if the holder has been gone for 3 hours (setup downloads a
            # CUDA wheel; a live install must not be torn down).
            if [ $((now - mt)) -gt 10800 ]; then
                rm -rf "$lock"
            fi
        fi
        if mkdir "$lock" 2>/dev/null; then
            if bash -c "$cmd"; then
                rm -rf "$lock"
                return 0
            fi
            rm -rf "$lock"
            echo "command failed while creating $ready" >&2
            return 1
        fi
        sleep 30
    done
    echo "timed out waiting for $ready" >&2
    return 1
}

echo "CNN Figure 11a  shard ${SHARD}/${NUM_SHARDS}  host=$(hostname)  $(date)"
nvidia-smi -L || nvidia-smi || true

wait_for_dir "$PROJECT_ROOT/.setup-cnn.lockdir" "$ENV_DIR/.ready" \
    "bash '$FIG11A_DIR/00_setup_env.sh'"
# shellcheck disable=SC1091
source "$FIG11A_DIR/env.sh"

wait_for_dir "$DATA_DIR/.prepare.lockdir" "$DATA_DIR/subsets.npz" \
    "bash '$FIG11A_DIR/01_prepare_data.sh'"

python "$SCRIPT_DIR/run_grid.py" \
    --n-values $N_VALUES \
    --seed "$TRAIN_SEED" \
    --max-steps "$MAX_STEPS" \
    --jobs-per-gpu "$JOBS_PER_GPU" \
    --shard "$SHARD" --num-shards "$NUM_SHARDS" || true

python "$SCRIPT_DIR/collect_results.py" || true
python "$SCRIPT_DIR/plot_fig11a.py" || true
