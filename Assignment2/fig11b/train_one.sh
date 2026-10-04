#!/usr/bin/env bash
# Train one Transformer (d_model=D) on the n=N subset, then evaluate it.
#   usage: bash train_one.sh D N [SEED]
# Resumable: re-running continues from checkpoint_last.pt. Skips runs whose result exists.
# Smoke test (timing): MAX_UPDATE=2000 bash train_one.sh 64 4000
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

D="$1"
N="$2"
SEED="${3:-$TRAIN_SEED}"
[ -n "$D" ] && [ -n "$N" ] || { echo "usage: $0 D N [SEED]" >&2; exit 1; }

RUN="d${D}_n${N}_s${SEED}"
[ "$MAX_UPDATE" != "80000" ] && RUN="${RUN}_u${MAX_UPDATE}"
DATA="$BIN_DIR/n$N"
SAVE="$CKPT_DIR/$RUN"
OUT="$RESULT_DIR/$RUN.json"

if [ -f "$OUT" ]; then
    echo "[$RUN] already evaluated: $OUT"
    exit 0
fi
[ -f "$DATA/train.de-en.en.bin" ] || { echo "missing $DATA - run 01_prepare_data.sh" >&2; exit 1; }
mkdir -p "$SAVE" "$RESULT_DIR"

# Vaswani et al. (2017) Sec. 5.3: lr = d^-0.5 * min(step^-0.5, step * warmup^-1.5).
# fairseq's inverse_sqrt scheduler reproduces this exactly with peak lr = d^-0.5 * warmup^-0.5.
LR=$(python -c "print($D ** -0.5 * $WARMUP ** -0.5)")
echo "[$RUN] d_model=$D d_ff=$((4 * D)) n=$N seed=$SEED peak_lr=$LR"

# Resuming a finished checkpoint would make fairseq-train do one extra update.
if python "$SCRIPT_DIR/evaluate.py" --check-only \
    --checkpoint "$SAVE/checkpoint_last.pt" --data "$DATA" --expected-updates "$MAX_UPDATE"; then
    echo "[$RUN] training already complete, evaluating"
else
    # --save-interval 1000000 disables end-of-epoch saves (an epoch is only ~25 updates
    # at n=4000); checkpoints are written every SAVE_INTERVAL_UPDATES and at the end.
    fairseq-train "$DATA" \
        --task translation --source-lang de --target-lang en \
        --arch transformer_iwslt_de_en \
        --encoder-layers 6 --decoder-layers 6 \
        --encoder-attention-heads 8 --decoder-attention-heads 8 \
        --encoder-embed-dim "$D" --decoder-embed-dim "$D" \
        --encoder-ffn-embed-dim $((4 * D)) --decoder-ffn-embed-dim $((4 * D)) \
        --share-decoder-input-output-embed \
        --dropout 0.0 --attention-dropout 0.0 --activation-dropout 0.0 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 --weight-decay 0.0 \
        --clip-norm 0.0 \
        --lr "$LR" --lr-scheduler inverse_sqrt --warmup-updates "$WARMUP" --warmup-init-lr 0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
        --max-tokens "$MAX_TOKENS" --max-update "$MAX_UPDATE" \
        --seed "$SEED" \
        --disable-validation \
        --save-dir "$SAVE" --save-interval 1000000 \
        --save-interval-updates "$SAVE_INTERVAL_UPDATES" --keep-interval-updates 1 \
        --no-epoch-checkpoints \
        --log-format simple --log-interval 1000 \
        $EXTRA_TRAIN_ARGS
fi

python "$SCRIPT_DIR/evaluate.py" \
    --checkpoint "$SAVE/checkpoint_last.pt" --data "$DATA" \
    --expected-updates "$MAX_UPDATE" \
    --size "$D" --n "$N" --seed "$SEED" --out "$OUT"

rm -f "$SAVE"/checkpoint_*_*.pt
