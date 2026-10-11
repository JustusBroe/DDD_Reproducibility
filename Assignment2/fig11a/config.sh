# Shared paths and hyperparameters. Sourced by every other script.
# PROJECT_ROOT is the folder that contains fig11a/ (on UCloud e.g. /work/ATDL_A2).
# Artefacts sit next to the fig11b ones, under their own names, so the two
# experiments can share one mounted project folder.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"

CONDA_DIR="$PROJECT_ROOT/miniforge3"
ENV_DIR="$PROJECT_ROOT/envs/cnn"

DATA_DIR="$PROJECT_ROOT/cifar-data"
CKPT_DIR="$PROJECT_ROOT/checkpoints-cnn"
LOG_DIR="$PROJECT_ROOT/logs-cnn"
RESULT_DIR="$PROJECT_ROOT/results-cnn"

# PyTorch build. 2.7.1/cu128 is needed for Blackwell GPUs (UCloud B200)
# and also covers A100/H100. torchvision 0.22.1 matches that torch release.
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"

# Figure 11a (Nakkiran et al., arXiv:1912.02292), Appendix B.2 / B.4.
# Widths and sample sizes are fixed in runname.py: c = 1..40 step 1, then
# 42..80 step 2; n = 12500, 25000, 50000. Override with WIDTHS / N_VALUES.
N_VALUES="${N_VALUES:-12500 25000 50000}"
SUBSET_SEED="${SUBSET_SEED:-0}"
NOISE_SEED="${NOISE_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-0}"
NOISE="${NOISE:-0.1}"

# SGD, no momentum, no weight decay. gamma(t) = 0.1 / sqrt(1 + floor(t/512)).
# The main text trains these CNNs for 500K gradient steps (not epochs), so that
# runs with different train-set sizes take the same number of updates.
# Appendix B.4 once says "500K epochs" for Figure 11(a); that would multiply
# the budget by hundreds and break the cross-size comparison the schedule is
# written for. MAX_STEPS is the gradient-step budget. Smoke test:
#   MAX_STEPS=200 bash train_one.sh 8 12500
MAX_STEPS="${MAX_STEPS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR0="${LR0:-0.1}"
LR_PERIOD="${LR_PERIOD:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-200}"

# How many trainings share the one GPU on this UCloud instance.
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

export PROJECT_ROOT CONDA_DIR ENV_DIR DATA_DIR CKPT_DIR LOG_DIR RESULT_DIR
export TORCH_VERSION TORCHVISION_VERSION TORCH_CUDA
export N_VALUES SUBSET_SEED NOISE_SEED TRAIN_SEED NOISE
export MAX_STEPS BATCH_SIZE LR0 LR_PERIOD NUM_WORKERS SAVE_EVERY EVAL_EVERY LOG_EVERY
export JOBS_PER_GPU
