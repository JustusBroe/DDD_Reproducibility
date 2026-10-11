#!/usr/bin/env bash
# UCloud "Initialization" parameter for interactive jobs (Terminal / Coder / JupyterLab).
# Makes every terminal opened in the job activate the CNN env automatically.
# Requires the project folder (containing fig11a/) to be mounted under /work.

if [ -n "${FIG11A_DIR:-}" ]; then
    ENV_SH="$FIG11A_DIR/env.sh"
else
    ENV_SH="$(find /work -maxdepth 5 -path '*/fig11a/env.sh' -print -quit 2>/dev/null || true)"
fi

if [ -z "$ENV_SH" ] || [ ! -f "$ENV_SH" ]; then
    echo "[ucloud_init] fig11a/env.sh not found under /work/*/ - did you mount the project folder?" >&2
    exit 0
fi

echo "source '$ENV_SH'" >> ~/.bashrc
echo "[ucloud_init] added 'source $ENV_SH' to ~/.bashrc"
