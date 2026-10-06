#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
DATASET="${1:-lrs2}"
if [[ $# -gt 0 ]]; then shift; fi
case "$DATASET" in
  lrs2|librispeech) ;;
  *) echo 'Usage: bash scripts/launch_2gpu.sh [lrs2|librispeech] [training options]' >&2; exit 2 ;;
esac
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo 'Project .venv is missing; run scripts/setup_environment.sh first.' >&2
  exit 1
fi
cd "$ROOT"
mkdir -p "$ROOT/.cache/tmp" "$ROOT/.cache/torchrun"
export TMPDIR="$ROOT/.cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$ROOT/.venv/bin/python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  --log-dir "$ROOT/.cache/torchrun" \
  -m scripts.train \
  --config "$ROOT/configs/${DATASET}_capacity_stride10.yaml" \
  --output "$ROOT/runs/${DATASET}_stride10_2gpu" \
  --device cuda --execution-backend chunkwise "$@"
