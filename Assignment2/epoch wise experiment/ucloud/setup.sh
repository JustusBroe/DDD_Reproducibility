#!/usr/bin/env bash
# One-time setup on UCloud (run_epochwise.sh calls it automatically if needed).
# Installs into epoch_dd/.venv so it persists across jobs, downloads CIFAR-10/100
# and the paper's reference curves, and checks that PyTorch can use the GPU.
#   bash /work/<folder>/epoch_dd/ucloud/setup.sh
set -eo pipefail
EPOCH_DD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$EPOCH_DD_DIR/train.py" ]; then
    EPOCH_DD_DIR="$(dirname "$(find /work -maxdepth 6 -path '*/epoch_dd/train.py' 2>/dev/null | head -n1)")"
fi
cd "$EPOCH_DD_DIR"
VENV="$EPOCH_DD_DIR/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    if ! python3 -m venv "$VENV" 2>/dev/null; then
        # Minimal images often lack python3-venv/pip: use a private Miniforge Python instead.
        rm -rf "$VENV"
        MF="$EPOCH_DD_DIR/.miniforge"
        if [ ! -x "$MF/bin/python" ]; then
            URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
            (curl -fsSL -o /tmp/miniforge.sh "$URL" || wget -qO /tmp/miniforge.sh "$URL")
            bash /tmp/miniforge.sh -b -p "$MF"
        fi
        "$MF/bin/python" -m venv "$VENV"
    fi
fi
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
# B200 (Blackwell) GPUs need a CUDA >= 12.8 build; the default Linux wheels of recent PyTorch are.
python -c "import torch, torchvision" 2>/dev/null || pip install torch torchvision
pip install -q numpy pandas matplotlib

python -c "
import torchvision as tv
for cls in (tv.datasets.CIFAR10, tv.datasets.CIFAR100):
    cls('data', train=True, download=True); cls('data', train=False, download=True)"
python fetch_reference.py

python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, capability {torch.cuda.get_device_capability(i)}")
x = torch.randn(1024, 1024, device="cuda")
print("GPU matmul OK:", float((x @ x).sum()) == float((x @ x).sum()))
EOF
echo "setup done"
