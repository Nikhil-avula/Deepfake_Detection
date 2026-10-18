#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VENV_PATH="${DEEPGUARD_VENV:-/workspace/deepguard-venv}"
export MPLCONFIGDIR=/workspace/deepguard-runtime/matplotlib
export XDG_CACHE_HOME=/workspace/deepguard-runtime/cache
export TORCH_HOME=/workspace/deepguard-runtime/torch
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
"$VENV_PATH/bin/python" -m pip check
"$VENV_PATH/bin/python" -m pytest -q
