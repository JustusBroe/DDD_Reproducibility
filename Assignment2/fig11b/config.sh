# Shared paths and hyperparameters. Sourced by every other script.
# PROJECT_ROOT is the folder that contains fig11b/ (on UCloud e.g. /work/ATDL_A2).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"

CONDA_DIR="$PROJECT_ROOT/miniforge3"
ENV_DIR="$PROJECT_ROOT/envs/fairseq"
FAIRSEQ_DIR="$PROJECT_ROOT/fairseq-repo"

DATA_DIR="$PROJECT_ROOT/data"
TEXT_DIR="$DATA_DIR/iwslt14.tokenized.de-en"
BIN_DIR="$PROJECT_ROOT/data-bin"
FULL_BIN="$BIN_DIR/full"
CKPT_DIR="$PROJECT_ROOT/checkpoints"
LOG_DIR="$PROJECT_ROOT/logs"
RESULT_DIR="$PROJECT_ROOT/results"

# PyTorch build. 2.4.1/cu121 covers V100/A100/H100/L40S.
# Blackwell GPUs (B200, RTX 50xx) need e.g. TORCH_VERSION=2.7.1 TORCH_CUDA=cu128.
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"

# Grid (identical to the authors' CSVs)
SIZES="${SIZES:-64 80}"
N_MIN="${N_MIN:-4000}"
N_MAX="${N_MAX:-120000}"
N_STEP="${N_STEP:-2000}"
SUBSET_SEED="${SUBSET_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"

# Training (paper B.1/B.3, Vaswani et al. 2017 Sec. 5.3)
MAX_UPDATE="${MAX_UPDATE:-80000}"
WARMUP="${WARMUP:-4000}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-5000}"
