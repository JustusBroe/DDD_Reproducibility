#!/usr/bin/env bash
# UCloud batch script: runs every line of a plan file (default ucloud/plan.txt) as a
# queue, PER_GPU runs at a time on each GPU, then makes all figures.
# Each run checkpoints every 25 epochs: if the job hits its time limit, submit it
# again and every run resumes from last.pt (finished runs exit immediately).
#
#   bash ucloud/run_epochwise.sh                         # paper plan, ResNets 1000 epochs
#   bash ucloud/run_epochwise.sh ADAM_EPOCHS=4000        # extend to the paper's full 4K epochs
#   bash ucloud/run_epochwise.sh PER_GPU=6 PLAN=my_plan.txt
# KEY=VALUE arguments can also go in UCloud's "Extra options" field.
# Runs ucloud/setup.sh first if the environment does not exist yet.
set -eo pipefail
for kv in "$@"; do export "$kv"; done

# UCloud may run a copy of this script, so fall back to searching the mounted folders.
EPOCH_DD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$EPOCH_DD_DIR/train.py" ]; then
    EPOCH_DD_DIR="$(dirname "$(find /work -maxdepth 6 -path '*/epoch_dd/train.py' 2>/dev/null | head -n1)")"
fi
[ -f "$EPOCH_DD_DIR/train.py" ] || { echo "epoch_dd/ not found under /work - add it under Storage" >&2; exit 1; }
cd "$EPOCH_DD_DIR"
echo "epoch_dd = $EPOCH_DD_DIR"
[ -x .venv/bin/python ] || bash ucloud/setup.sh
source .venv/bin/activate

PLAN="${PLAN:-ucloud/plan.txt}"
export ADAM_EPOCHS="${ADAM_EPOCHS:-1000}"   # paper: 4000
export SGD_EPOCHS="${SGD_EPOCHS:-1279}"     # 1279 x 391 steps = 500K steps, as in the paper
PER_GPU="${PER_GPU:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                # e.g. "--amp" (report it as a deviation if used)

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 1 ] || { echo "no GPU visible" >&2; exit 1; }
nvidia-smi
mkdir -p logs
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$PLAN")
SLOTS=$((NGPU * PER_GPU))
echo "${#LINES[@]} runs, $NGPU GPU(s), $SLOTS at a time, ADAM_EPOCHS=$ADAM_EPOCHS SGD_EPOCHS=$SGD_EPOCHS"

i=0
running=0
failed=0
for line in "${LINES[@]}"; do
    if [ "$running" -ge "$SLOTS" ]; then
        wait -n || failed=$((failed + 1))
        running=$((running - 1))
    fi
    args=$(eval echo "$line")
    gpu=$((i % NGPU))
    log="logs/$(echo "$args" | tr -d '-' | tr -s ' ' '_').out"
    echo "[$(date +%H:%M)] GPU $gpu: python train.py $args"
    CUDA_VISIBLE_DEVICES=$gpu python train.py $args $EXTRA_ARGS >> "$log" 2>&1 &
    running=$((running + 1))
    i=$((i + 1))
done
while [ "$running" -gt 0 ]; do
    wait -n || failed=$((failed + 1))
    running=$((running - 1))
done
echo "[$(date +%H:%M)] all runs finished, $failed failed (see logs/)"

python plot_epochwise.py --mode all
exit "$failed"
