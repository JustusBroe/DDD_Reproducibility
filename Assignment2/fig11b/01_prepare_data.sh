#!/usr/bin/env bash
# One-time data preparation (CPU only, ~15 min):
#   1. fairseq's prepare-iwslt14.sh: download, Moses tokenisation, lowercasing,
#      train/valid split (every 23rd line -> valid), test = dev2010+dev2012+tst2010+tst2011+tst2012,
#      joint BPE (10k merges) learned on the FULL train split.
#   2. Binarise the full data once -> fixes dict.de.txt / dict.en.txt and the test set.
#   3. Draw nested random subsets of n sentence pairs and binarise each with the fixed dicts.
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
if [ ! -f "$TEXT_DIR/test.en" ]; then
    bash "$FAIRSEQ_DIR/examples/translation/prepare-iwslt14.sh"
fi
wc -l "$TEXT_DIR"/{train,valid,test}.{de,en}

if [ ! -f "$FULL_BIN/test.de-en.en.bin" ]; then
    fairseq-preprocess --source-lang de --target-lang en \
        --trainpref "$TEXT_DIR/train" --validpref "$TEXT_DIR/valid" --testpref "$TEXT_DIR/test" \
        --destdir "$FULL_BIN" --workers 8
fi
wc -l "$FULL_BIN"/dict.{de,en}.txt

python "$SCRIPT_DIR/make_subsets.py" \
    --text-dir "$TEXT_DIR" --full-bin "$FULL_BIN" --bin-dir "$BIN_DIR" \
    --n-min "$N_MIN" --n-max "$N_MAX" --n-step "$N_STEP" --seed "$SUBSET_SEED"
