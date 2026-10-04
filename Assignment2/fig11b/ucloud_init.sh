#!/usr/bin/env bash
# UCloud "Initialization" parameter for interactive jobs (Terminal / Coder / JupyterLab).
# Makes every terminal opened in the job activate the fairseq env automatically.
# Requires the project folder (containing fig11b/) to be mounted under /work.

ENV_SH="${FIG11B_DIR:+$FIG11B_DIR/env.sh}"
ENV_SH="${ENV_SH:-$(find /work -maxdepth 5 -path '*/fig11b/env.sh' 2>/dev/null | head -n1)}"

if [ -z "$ENV_SH" ] || [ ! -f "$ENV_SH" ]; then
    echo "[ucloud_init] fig11b/env.sh not found under /work/*/ - did you mount the project folder?" >&2
    exit 0
fi

echo "source '$ENV_SH'" >> ~/.bashrc
echo "[ucloud_init] added 'source $ENV_SH' to ~/.bashrc"
