#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${1:-cu124}"
if [[ "$MODE" != "cu124" && "$MODE" != "cpu" ]]; then
  echo 'Usage: bash scripts/setup_environment.sh [cu124|cpu]' >&2
  exit 2
fi
# All environment/package/cache/temp writes remain under the project directory.
mkdir -p "$ROOT/.cache/pip" "$ROOT/.cache/tmp"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export TMPDIR="$ROOT/.cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
python -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
if [[ "$MODE" == "cu124" ]]; then
  "$ROOT/.venv/bin/python" -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
  if [[ -f "$ROOT/requirements-cu124.lock" ]]; then
    "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-cu124.lock"
  fi
else
  "$ROOT/.venv/bin/python" -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
fi
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT[metrics,simulation,plots]"
"$ROOT/.venv/bin/python" -m scripts.doctor --config "$ROOT/configs/lrs2.yaml" --depth 0 --examples 1
