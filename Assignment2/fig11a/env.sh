# Activates the persistent conda env. Usage: source fig11a/env.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

if [ -x "$CONDA_DIR/bin/conda" ] && [ -f "$ENV_DIR/.ready" ]; then
    eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
    conda activate "$ENV_DIR"
else
    echo "[env.sh] conda env not found at $ENV_DIR - run fig11a/00_setup_env.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
