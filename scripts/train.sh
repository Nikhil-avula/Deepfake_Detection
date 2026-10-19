#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_HOME=/workspace/deepguard-runtime/torch
export MPLCONFIGDIR=/workspace/deepguard-runtime/matplotlib
VENV_PATH="${DEEPGUARD_VENV:-/workspace/deepguard-venv}"
if [[ $# -gt 0 && "$1" != "--balanced" ]]; then
  previous=()
  if [[ -f data/manifests/splits.csv ]]; then previous=(--previous-splits data/manifests/splits.csv); fi
  "$VENV_PATH/bin/python" -m deepguard.cli prepare-dataset "$1" "${previous[@]}"
  shift
fi
options=()
for option in "$@"; do
  if [[ "$option" == "--balanced" ]]; then options+=(--balanced); else echo "Unknown training option: $option" >&2; exit 2; fi
done
"$VENV_PATH/bin/python" -m deepguard.cli train "${options[@]}"
"$VENV_PATH/bin/python" -m deepguard.cli robustness
