#!/usr/bin/env bash
# One-time setup, run inside an interactive UCloud job (GPU machine) with the
# project folder mounted. Everything is installed inside PROJECT_ROOT so it
# persists across jobs:
#   miniforge3/        conda
#   envs/fairseq/      Python 3.10 env with PyTorch + fairseq (editable)
#   fairseq-repo/      git clone of facebookresearch/fairseq
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

echo "PROJECT_ROOT = $PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"

# fairseq pins hydra-core 1.0 / omegaconf 2.0: these break on Python >= 3.11,
# and omegaconf 2.0's metadata is rejected by pip >= 24.1.
if [ ! -d "$ENV_DIR" ]; then
    conda create -y -p "$ENV_DIR" -c conda-forge \
        python=3.10 "pip<24.1" gcc_linux-64=12 gxx_linux-64=12 perl wget git
fi
conda activate "$ENV_DIR"

pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
pip install "numpy<2" cython wheel subword-nmt sacremoses pandas matplotlib

if [ ! -d "$FAIRSEQ_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_DIR"
fi
cd "$FAIRSEQ_DIR"
echo "fairseq commit: $(git rev-parse HEAD)" | tee "$PROJECT_ROOT/fairseq_commit.txt"
# setup.py imports torch, hence --no-build-isolation. CUDA_HOME is unset so the
# optional CUDA kernels (not needed here) are not compiled.
env -u CUDA_HOME pip install --no-build-isolation -e .

python - <<'EOF'
import torch, fairseq, numpy
print("torch", torch.__version__, "| fairseq", fairseq.__version__, "| numpy", numpy.__version__)
print("CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
EOF
fairseq-train --help > /dev/null && echo "fairseq-train OK"
