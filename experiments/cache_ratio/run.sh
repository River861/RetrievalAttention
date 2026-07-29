#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python throughput_eval/reproduce_block_cache.py \
  --suite block_cache_single_a100 \
  --context-batch-groups 120000x1,120000x8,240000x1,480000x1 \
  --cache-ratios 0.005,0.05,0.1 \
  --baseline-cache-ratio 0.05 \
  --rounds 2 \
  --gen-len 100 \
  --retrieval-budget 0.018 \
  --estimation-budget 0.232 \
  --seed 2025 \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}" \
  --require-idle-gpu \
  --output-dir experiments/cache_ratio/rerun
