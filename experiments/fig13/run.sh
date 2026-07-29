#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python throughput_eval/reproduce_fig13.py \
  --suite figure13 \
  --methods Full_Flash_Attn,RetroInfer \
  --context-lens 30000,60000,120000 \
  --rounds 1 \
  --gen-len 100 \
  --cache-ratio 0.05 \
  --retrieval-budget 0.018 \
  --estimation-budget 0.232 \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0}" \
  --require-idle-gpu \
  --output-dir experiments/fig13/rerun
