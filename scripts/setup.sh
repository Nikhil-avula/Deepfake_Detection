#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PATH="${DEEPGUARD_VENV:-/workspace/deepguard-venv}"
export PIP_CACHE_DIR=/workspace/deepguard-runtime/pip
export MPLCONFIGDIR=/workspace/deepguard-runtime/matplotlib
export XDG_CACHE_HOME=/workspace/deepguard-runtime/cache
export TORCH_HOME=/workspace/deepguard-runtime/torch
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
if [ ! -x "$VENV_PATH/bin/python" ]; then python3 -m venv "$VENV_PATH"; fi
"$VENV_PATH/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
"$VENV_PATH/bin/python" -m pip install -r requirements.txt
"$VENV_PATH/bin/python" -m pip install -e .
"$VENV_PATH/bin/python" -m pip check
