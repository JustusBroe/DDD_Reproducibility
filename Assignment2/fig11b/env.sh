# Activates the persistent conda env. Usage: source fig11b/env.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

if [ -x "$CONDA_DIR/bin/conda" ] && [ -d "$ENV_DIR" ]; then
    eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
    conda activate "$ENV_DIR"
else
    echo "[env.sh] conda env not found at $ENV_DIR - run fig11b/00_setup_env.sh first." >&2
fi

# fairseq checkpoints are pickled; torch>=2.6 would otherwise refuse to load them.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
