#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_HOME=/workspace/deepguard-runtime/torch
export MPLCONFIGDIR=/workspace/deepguard-runtime/matplotlib
VENV_PATH="${DEEPGUARD_VENV:-/workspace/deepguard-venv}"
"$VENV_PATH/bin/python" -m deepguard.cli train
"$VENV_PATH/bin/python" -m deepguard.cli robustness
