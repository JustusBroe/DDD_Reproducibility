#!/usr/bin/env bash
# Download CIFAR-10 (once) and write cifar-data/subsets.npz.
# CPU only, a few minutes the first time. Safe to re-run.
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

mkdir -p "$DATA_DIR"
python "$SCRIPT_DIR/prepare_data.py" \
    --data-dir "$DATA_DIR" \
    --noise "$NOISE" \
    --subset-seed "$SUBSET_SEED" \
    --noise-seed "$NOISE_SEED"
