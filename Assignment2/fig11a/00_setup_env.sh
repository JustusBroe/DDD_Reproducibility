#!/usr/bin/env bash
# One-time setup, run inside a UCloud job (interactive or the first batch job)
# with the project folder mounted. Installed under PROJECT_ROOT so it persists:
#   miniforge3/     conda (shared with fig11b if that was set up already)
#   envs/cnn/       Python 3.11 + PyTorch cu128 + torchvision
# Re-running is a no-op once envs/cnn/.ready exists.
set -eo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

echo "PROJECT_ROOT = $PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT"

if [ -f "$ENV_DIR/.ready" ]; then
    echo "env already installed at $ENV_DIR"
else
    if [ -d "$ENV_DIR" ]; then
        echo "removing incomplete env at $ENV_DIR"
        rm -rf "$ENV_DIR"
    fi
    if [ ! -x "$CONDA_DIR/bin/conda" ]; then
        curl -fsSL -o /tmp/miniforge.sh \
            https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
    fi
    eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
    conda create -y -p "$ENV_DIR" -c conda-forge python=3.11 pip
    conda activate "$ENV_DIR"
    pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
    pip install numpy pandas matplotlib
    touch "$ENV_DIR/.ready"
fi

eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
conda activate "$ENV_DIR"
python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
EOF
