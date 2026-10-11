#!/usr/bin/env bash
# Train one CNN (width C) on the n=N noisy CIFAR-10 subset, then evaluate it.
#   usage: bash train_one.sh C N [SEED]
# Resumable from checkpoints-cnn/<run>/checkpoint.pt. Skips a run whose JSON exists.
# Smoke test (timing): MAX_STEPS=200 bash train_one.sh 8 12500
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

C="$1"
N="$2"
SEED="${3:-$TRAIN_SEED}"
[ -n "$C" ] && [ -n "$N" ] || { echo "usage: $0 C N [SEED]" >&2; exit 1; }

RUN="$(python -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from runname import run_name; print(run_name($C, $N, $SEED, $MAX_STEPS))")"
SAVE="$CKPT_DIR/$RUN"
OUT="$RESULT_DIR/$RUN.json"
mkdir -p "$SAVE" "$RESULT_DIR"

if [ -f "$OUT" ]; then
    echo "[$RUN] already evaluated: $OUT"
    exit 0
fi
[ -f "$DATA_DIR/subsets.npz" ] || { echo "missing $DATA_DIR/subsets.npz - run 01_prepare_data.sh" >&2; exit 1; }

file_mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }

LOCK="$SAVE/running.lock"
STALE_SECONDS="${STALE_SECONDS:-3600}"
if [ -d "$LOCK" ]; then
    hb="$SAVE/heartbeat"
    now="$(date +%s)"
    if [ -f "$hb" ]; then
        age=$((now - $(file_mtime "$hb")))
    else
        age=$((now - $(file_mtime "$LOCK")))
    fi
    if [ "$age" -lt "$STALE_SECONDS" ]; then
        echo "[$RUN] another job is training this run; skipping"
        exit 0
    fi
    echo "[$RUN] removing stale lock"
    rm -rf "$LOCK"
fi
mkdir "$LOCK"
touch "$SAVE/heartbeat"
cleanup() { rm -rf "$LOCK"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "[$RUN] $(date -Is 2>/dev/null || date) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-all}"
python "$SCRIPT_DIR/train_one.py" \
    --width "$C" --n "$N" --seed "$SEED" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr0 "$LR0" --lr-period "$LR_PERIOD" \
    --workers "$NUM_WORKERS" \
    --save-every "$SAVE_EVERY" --eval-every "$EVAL_EVERY" --log-every "$LOG_EVERY" \
    --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR" --result-dir "$RESULT_DIR"
