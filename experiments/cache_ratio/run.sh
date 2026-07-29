#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

"$PYTHON" throughput_eval/reproduce_block_cache.py \
  --suite block_cache_single_a100 \
  --context-batch-groups 120000x8 \
  --cache-ratios 0.025 \
  --baseline-cache-ratio 0.05 \
  --rounds 1 \
  --gen-len 100 \
  --retrieval-budget 0.018 \
  --estimation-budget 0.232 \
  --seed 2025 \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}" \
  --require-idle-gpu \
  --output-dir experiments/cache_ratio/rerun
