# RetroInfer 120Kx8 block-cache baseline protocol

## Scope

This baseline records the unmodified RetroInfer offload block-cache behavior for the streaming-cache objective at:

- `context_len=120000`
- `batch_size=8`
- `cache_ratio=0.05`
- `baseline_cache_ratio=0.05`
- `rounds=2`
- `gen_len=100`
- `retrieval_budget=0.018`
- `estimation_budget=0.232`
- `seed=2025`
- `CUDA_VISIBLE_DEVICES=0`

No kernel/backend/autotune/source code was edited for this run. The repository source revision recorded by the harness is `7096eab9190da389bbc75c1992140d1432d9d8ec`; the worktree already contained research-stage artifacts and `research/PIPELINE_STATE.json` changes before this baseline reproduction.

## Post-reset applicability to rotating-slot campaign

Verified during the `baseline_post_reset_closeout` mission: this artifact remains the canonical unmodified/default RetroInfer comparator for the CPU full-KV plus `<=m` rotating GPU block-cache-slot campaign, so no rerun was needed. The reset changes the next optimize-stage mechanism, not the baseline command, shape, API, or default-off behavior.

Binding facts checked: current `HEAD` still matches `7096eab9190da389bbc75c1992140d1432d9d8ec`; `research/PIPELINE_STATE.json` is at `current_stage=baseline`; both parsed rows and raw logs are `120000x8`, `cache_ratio=0.05`, `gen_len=100`, `seed=2025`, `use_cuda_graph=true`, `returncode=0`, and `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`; memory samples peak at the recorded `41960.0 MiB`; block-cache residency remains `5952.0 MiB / 5.8125 GiB`; watched `RETROINFER_*` optimization and slot-rotation environment variables were unset for the binding check. No slot-rotation implementation, opt013 rerun, Figure 13 run, or new benchmark harness was performed in this closeout.

## Command

Run from `/home/v-xuchuanluo/RetroInfer` with the installed project virtualenv:

One-line exact form:

```bash
.venv/bin/python throughput_eval/reproduce_block_cache.py --suite block_cache_baseline_single_a100 --context-batch-groups 120000x8 --cache-ratios 0.05 --baseline-cache-ratio 0.05 --rounds 2 --gen-len 100 --retrieval-budget 0.018 --estimation-budget 0.232 --seed 2025 --cuda-visible-devices 0 --require-idle-gpu --stop-on-error --output-dir research/baseline_block_cache_single_a100_120k_b8_cr0p05
```

Readable wrapped form:

```bash
.venv/bin/python throughput_eval/reproduce_block_cache.py \
  --suite block_cache_baseline_single_a100 \
  --context-batch-groups 120000x8 \
  --cache-ratios 0.05 \
  --baseline-cache-ratio 0.05 \
  --rounds 2 \
  --gen-len 100 \
  --retrieval-budget 0.018 \
  --estimation-budget 0.232 \
  --seed 2025 \
  --cuda-visible-devices 0 \
  --require-idle-gpu \
  --stop-on-error \
  --output-dir research/baseline_block_cache_single_a100_120k_b8_cr0p05
```

The harness exited with return code `0`.

## Environment

- Python: `3.11.15`, executable `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`
- PyTorch runtime: `torch==2.5.1+cu124`, `torch.version.cuda == 12.4`
- Key packages: `vllm==0.6.5`, `transformers==4.49.0`, `flash-attn==2.7.3`, `flashinfer-python==0.2.4+cu124torch2.5`, `triton==3.1.0`, `pybind11==2.12.0`, `retroinfer_kernels` importable, `weighted_flash_decoding==0.1`
- GPU: NVIDIA A100 80GB PCIe, compute capability 8.0, 81920 MiB, driver 580.173.02
- Harness idle check: `--require-idle-gpu`; `nvidia-smi` showed no running GPU processes at environment collection and the harness rechecked before each run.
- Runtime is ready for this baseline. Source rebuild remains separately blocked for CUDA-extension rebuilds because no CUDA 12.4-compatible `nvcc`/`CUDA_HOME` was proven (`torch` CUDA 12.4 runtime, `/usr/bin/nvcc` CUDA 12.0, `/usr/local/cuda-13.3/bin/nvcc` CUDA 13.3).

## Measurement policy

- Each round launches a fresh `throughput_eval/test.py` process via the existing harness; no new benchmark harness was created.
- `--use_cuda_graph` is enabled by default in `reproduce_block_cache.py` and is passed to `test.py`.
- No extra synthetic warmup or manual autotune pass was added. The measured behavior is the project-native sequence: allocate GPU buffers and CPU pinned memory, prefill, capture CUDA graphs for RetroInfer when enabled, then decode.
- Synchronization follows the unmodified code:
  - `model_hub/LLM.py` calls `torch.cuda.synchronize()` immediately before and after prefill timing.
  - RetroInfer cache code uses its existing CUDA streams/events and synchronizations, including copy-event waits and graph capture.
  - `throughput_eval/test.py` collects CUDA memory stats after generation, and that helper synchronizes each CUDA device before reading allocator counters.
  - Decode timing is not altered by this protocol; it uses the repository's native `decode_latency_s` measurement.
- Process peak GPU memory is sampled by the harness every `0.5` seconds with `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory`, scoped to the launched runner PID and descendants. The primary process-memory metric is `peak_process_gpu_memory_mib`.
- Block-cache memory is reported by `kv_cache.block_cache_metadata()` and parsed into `block_cache_total_bytes`, `block_cache_total_mib`, and `block_cache_total_gib`.

## Correctness oracle

Each `test.py` run prints `RETROINFER_OUTPUT_JSON`. A run is correctness-passing only if:

1. The child `test.py` return code is `0`.
2. The harness status is `passed` and `failure_class == "success"`.
3. `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth == true`.

Both rounds passed. The NIAH ground truth was `9879991`, and all 8 decoded outputs in each round contained it.

## Raw artifact paths

- Artifact directory: `research/baseline_block_cache_single_a100_120k_b8_cr0p05`
- Environment: `research/baseline_block_cache_single_a100_120k_b8_cr0p05/environment.json`
- Config/commands: `research/baseline_block_cache_single_a100_120k_b8_cr0p05/config.json`, `research/baseline_block_cache_single_a100_120k_b8_cr0p05/commands.jsonl`
- Per-run parsed results: `research/baseline_block_cache_single_a100_120k_b8_cr0p05/results.jsonl`, `research/baseline_block_cache_single_a100_120k_b8_cr0p05/results.csv`
- Summary tables: `research/baseline_block_cache_single_a100_120k_b8_cr0p05/summary.csv`, `research/baseline_block_cache_single_a100_120k_b8_cr0p05/per_round_table.csv`, `research/baseline_block_cache_single_a100_120k_b8_cr0p05/deltas.csv`
- Raw logs:
  - `research/baseline_block_cache_single_a100_120k_b8_cr0p05/raw_logs/block_cache_baseline_single_a100_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`
  - `research/baseline_block_cache_single_a100_120k_b8_cr0p05/raw_logs/block_cache_baseline_single_a100_retroinfer_ctx120000_bsz8_cr0p05_r2.txt`
- GPU memory samples:
  - `research/baseline_block_cache_single_a100_120k_b8_cr0p05/memory_samples/block_cache_baseline_single_a100_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`
  - `research/baseline_block_cache_single_a100_120k_b8_cr0p05/memory_samples/block_cache_baseline_single_a100_retroinfer_ctx120000_bsz8_cr0p05_r2.jsonl`

## Baseline result summary

| metric | value |
|---|---:|
| total runs | 2 |
| passed runs | 2 |
| correctness | pass |
| block-cache memory | 5.8125 GiB / 5952.0 MiB |
| block-cache share of A100 80GB | 7.265625% |
| mean peak process GPU memory | 41960.0 MiB |
| max peak process GPU memory | 41960.0 MiB |
| mean decode throughput | 198.8364143870935 tokens/s |
| mean e2e latency | 244.12288427352905 s |
| mean e2e generated-token throughput | 3.277045618 tokens/s |
