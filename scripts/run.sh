#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_HOME=/workspace/deepguard-runtime/torch
export MPLCONFIGDIR=/workspace/deepguard-runtime/matplotlib
VENV_PATH="${DEEPGUARD_VENV:-/workspace/deepguard-venv}"
exec "$VENV_PATH/bin/streamlit" run app/main.py --server.headless true --server.address 0.0.0.0 --server.port 8501 --server.maxUploadSize 100 --browser.gatherUsageStats false
