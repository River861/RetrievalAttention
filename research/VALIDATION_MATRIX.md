# opt013 uninitialized late block-cache allocation validation matrix

验证对象：`RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late` + `RETROINFER_LATE_BLOCK_CACHE_INIT=uninitialized` + `RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream` + `RETROINFER_ASYNC_CLUSTER_ID_COPY=1` + `RETROINFER_LAYER_CACHE_CAPACITY_SCALE=1=0.75,2-3=0.75,25=0.75,28-29=0.75` + `RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`。`RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY`、`RETROINFER_STAGE_INDEX_METADATA_LAYERS`、`RETROINFER_STREAM_ONLY_LAYERS`、`RETROINFER_LAYER_CACHE_RESIDENCY`、`RETROINFER_CACHE_TELEMETRY` 均保持 unset/off。本文档只绑定 `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/` 已有两轮 A100 120000x8/cache_ratio=0.05 结果；未重跑健康 baseline、未运行 Figure 13、未 rebuild CUDA/C++、未编辑 `research/PIPELINE_STATE.json`。

## 结论

opt013 **retained**，但只作为 bounded memory/e2e tradeoff：两轮 raw logs 均 `returncode=0`、`RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`、8/8 outputs 命中 groundtruth `9879991`，因此下面内存/性能数字可引用。opt013 保持 opt012 的 resident block-cache size 与 peak process GPU memory，e2e generated throughput 从 opt012 的 3.264410 tok/s 提升到 3.268888 tok/s；decode 从 opt012 的 201.1317 tok/s 回退到 198.0366 tok/s。因此不声明 decode speedup、广义加速、或 profiler-proven overlap。

| 指标 | Canonical baseline | opt010 forced-late low6 | opt011 split-late GPU metadata | opt012 pinned side-stream | opt013 uninitialized late block cache | Delta vs baseline | Delta vs opt010 | Delta vs opt011 | Delta vs opt012 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Block cache bytes | 6241124352 | 5948571648 | 5948571648 | 5948571648 | 5948571648 | -292552704 (-4.688%) | +0 (+0.000%) | +0 (+0.000%) | +0 (+0.000%) |
| Block cache MiB / GiB | 5952.0 / 5.812500000 | 5673.0 / 5.540039062 | 5673.0 / 5.540039062 | 5673.0 / 5.540039062 | 5673.0 / 5.540039062 | -279.0 MiB (-4.688%) | +0.0 MiB (+0.000%) | +0.0 MiB (+0.000%) | +0.0 MiB (+0.000%) |
| Peak process GPU memory MiB (mean/max) | 41960.0 / 41960.0 | 29884.0 / 29884.0 | 35610.0 / 35610.0 | 29884.0 / 29884.0 | 29884.0 / 29884.0 | -12076.0 (-28.780%) | +0.0 (+0.000%) | -5726.0 (-16.080%) | +0.0 (+0.000%) |
| Decode throughput tok/s (mean) | 198.836414 | 198.585321 | 198.562825 | 201.131671 | 198.036572 | -0.799843 (-0.402%) | -0.548749 (-0.276%) | -0.526254 (-0.265%) | -3.095099 (-1.539%) |
| E2E latency s (mean) | 244.122884 | 245.266826 | 244.172720 | 245.067283 | 244.734262 | +0.611378 (+0.250%) | -0.532563 (-0.217%) | +0.561542 (+0.230%) | -0.333021 (-0.136%) |
| E2E generated throughput tok/s (`8*100/e2e_latency_s`, per-round mean) | 3.277045618 | 3.261779085 | 3.276369282 | 3.264409640 | 3.268887524 | -0.008158094 (-0.249%) | +0.007108440 (+0.218%) | -0.007481758 (-0.228%) | +0.004477885 (+0.137%) |

性能解释边界：resident block-cache byte reduction comes from the retained low6 fractional-capacity configuration shared by opt010/opt011/opt012/opt013, not from `torch.empty`. The 29884 MiB peak-process GPU memory win versus baseline is a prefill/allocation-lifecycle result of late allocation plus pinned prefill metadata and low6 capacity; opt013 does not further reduce resident block-cache bytes or peak memory versus opt012. The only opt013-specific positive timing evidence is the two-round e2e improvement versus opt012; decode regressed.

## Source behavior validated by opt013

`cache_hub/retroinfer_cache.py:_env_late_block_cache_init_policy()` adds default-off `RETROINFER_LATE_BLOCK_CACHE_INIT`:

| Env value family | Effective policy | Behavior |
| --- | --- | --- |
| unset / empty | `zero`, env `unset` | Preserves default API behavior: late block-cache K/V tensors are allocated with `torch.zeros`; telemetry reason `policy_unset_zero_fill`. |
| `default`, `zero`, `zeros`, `zero_fill`, `zero_filled` | `zero` | Explicit zero-fill behavior; no uninitialized tensors. |
| `empty`, `uninitialized`, `uninitialised`, `no_zero`, `no_zero_fill` | `uninitialized` | Opt-in `torch.empty` only for `cache_keys/cache_values` allocated inside forced-late `prepare_cache()` after prefill. It is not applied to preallocated block cache, index metadata, execution buffers, or steady-zone tensors. |
| invalid values | error | Raises `ValueError` naming `RETROINFER_LATE_BLOCK_CACHE_INIT` and accepted zero/uninitialized families. |

If block cache is preallocated before prefill while the env asks for `uninitialized`, source metadata sets `block_cache_late_init_reason=not_applied_to_preallocated_block_cache`; opt013's measured path is forced-late, so `block_cache_preallocated_before_prefill=false`, `block_cache_allocated_after_prefill=true`, and `block_cache_late_init_effective=true`.

uninitialized-safety invariant: `WaveBufferCPU` starts all clusters as cache misses and marks cache hits only after cache admission; Python waits for the update path and then `gather_copy_and_scatter()` writes admitted K/V blocks into `cache_keys/cache_values` before later hit reads can source from those tensors. opt013 raw evidence shows this invariant was exercised with `block_cache_late_init_mode=uninitialized_empty`, `block_cache_late_uninitialized_tensor_count=64`, `block_cache_late_uninitialized_bytes=5948571648`, 2/2 correctness, and zero sampler errors.

## Allocation lifecycle evidence

| Phase | Evidence |
| --- | --- |
| Prefill setup | `RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late`, decision `allocate_after_prefill`; `block_cache_preallocated_before_prefill=false`. Index metadata remains pinned CPU during prefill because `RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream`: `index_metadata_prefill_gpu_bytes=0`, `index_metadata_prefill_cpu_bytes=7847116800`, `index_metadata_prefill_host_pinned_bytes=7847116800`. |
| `prepare_cache()` after prefill | `block_cache_allocation_prepare_cache_calls=1`; late K/V block-cache tensors allocated after prefill with `torch.empty`, mode `uninitialized_empty`; 64 tensors / 5,948,571,648 bytes recorded as uninitialized. |
| Metadata migration before decode | Index metadata migrates once to GPU before decode: `copy_mode=pinned_h2d_non_blocking_side_stream`, `copy_bytes=7847116800`, `copy_count=128`, `stream_count=1`, `sync_count=1`, `sync_point=after_allocate_computation_buffer_before_decode`; mean elapsed 22.973 ms and prepare window 590.833 ms. This is mechanism/timing metadata, not profiler overlap proof. |
| Decode/cache update | Decode uses normal RetroInfer cache admission; `gather_copy_and_scatter` writes admitted K/V blocks before cache-hit reads. Both rounds completed with `all_outputs_contain_groundtruth=true`. |

## 2-round real-hardware evidence

| Round | Status | Groundtruth oracle | Late init | Allocation lifecycle | Index metadata lifecycle | Block cache bytes | Peak GPU MiB | Sampler errors | Decode tok/s | E2E latency s | E2E tok/s | Raw log |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r1 | passed, returncode 0 | True, 8/8 | policy=uninitialized; mode=uninitialized_empty; tensors=64; bytes=5948571648 | late / allocate_after_prefill; preallocated=false; allocated_after_prefill=true; prepare_calls=1 | prefill_gpu=0; pinned_cpu=7847116800; current_gpu=7847116800; late_copy=7847116800; side_stream sync after computation-buffer allocation | 5948571648 | 29884.0 | 0 | 195.253579 | 243.923672 | 3.279714478 | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05/raw_logs/opt013_uninitialized_late_block_cache_retroinfer_ctx120000_bsz8_cr0p05_r1.txt` |
| r2 | passed, returncode 0 | True, 8/8 | policy=uninitialized; mode=uninitialized_empty; tensors=64; bytes=5948571648 | late / allocate_after_prefill; preallocated=false; allocated_after_prefill=true; prepare_calls=1 | prefill_gpu=0; pinned_cpu=7847116800; current_gpu=7847116800; late_copy=7847116800; side_stream sync after computation-buffer allocation | 5948571648 | 29884.0 | 0 | 200.819565 | 245.544852 | 3.258060571 | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05/raw_logs/opt013_uninitialized_late_block_cache_retroinfer_ctx120000_bsz8_cr0p05_r2.txt` |

## Validation coverage

| 合同项 | 结果 | 证据 |
| --- | --- | --- |
| Environment/toolchain | Pass for execution: project `.venv` Python 3.11.15, torch 2.5.1+cu124, torch CUDA 12.4, A100 80GB, `triton==3.1.0`, installed `retroinfer_kernels` symbols and `weighted_flash_decoding==0.1` import. Source rebuild remains blocked/non-goal because no CUDA 12.4-compatible `nvcc`/`CUDA_HOME` is proven (`/usr/bin/nvcc` CUDA 12.0; `/usr/local/cuda-13.3/bin/nvcc` CUDA 13.3). | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/ENV_AUDIT.md`, `ENV_AUDIT.raw.txt`, current focused `.venv` audit, `research/GROUND_TRUTH.md` |
| Shape/config | Pass: A100 80GB, `context_len=120000`, `batch_size=8`, `gen_len=100`, `cache_ratio=0.05`, retrieval/estimation budgets `0.018/0.232`, seed 2025, CUDA graph enabled, idle-GPU process memory sampling. | opt013 `config.json`, `commands.jsonl`, `results.jsonl` |
| Correctness | Pass before metrics: both raw logs contain `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`; each round has 8/8 outputs containing groundtruth `9879991`; both returncode 0 and harness status passed. | opt013 `raw_logs/`, `results.jsonl`, `OUTCOME.json` |
| `RETROINFER_LATE_BLOCK_CACHE_INIT` fallback/default/error behavior | Pass by source: unset/empty/default/zero families preserve zero-fill; uninitialized aliases are explicit opt-in for late `prepare_cache()` K/V tensors only; invalid values raise `ValueError`; preallocated block-cache path does not apply uninitialized mode. | `cache_hub/retroinfer_cache.py:_env_late_block_cache_init_policy`, `_new_late_block_cache_tensor`, `prepare_cache`, opt013 `CHANGES.md` |
| Late block-cache allocation lifecycle | Pass: forced late allocation avoided prefill block-cache allocation and allocated after prefill exactly once. | opt013 `results.jsonl`, `summary.csv`, raw logs line `Block cache allocation: policy=late` |
| Uninitialized safety invariant | Pass at source+oracle level: raw rows record `block_cache_late_init_safety=WaveBufferCPU starts all clusters as cache misses and marks cache hits only after gather_copy_and_scatter writes admitted K/V blocks`; both rounds correct. | `cache_hub/retroinfer_cache.py`, opt013 raw logs and `results.jsonl` |
| Pinned late index-metadata migration inherited from opt012 | Pass: pinned CPU prefill metadata, one-time side-stream H2D migration before decode, current GPU metadata restored to 7,847,116,800 bytes. No profiler overlap claim. | opt013 `results.jsonl`, `summary.csv`; opt012 validation source behavior |
| Low6 fractional capacity | Pass: active layers 32/32; layers 1/2/3/25/28/29 are 558/744 pages with scale 0.75; other layers 744 pages; total block-cache bytes 5,948,571,648. | opt013 `summary.csv`, `results.jsonl` |
| Buffer and async cluster-id metadata | Pass: `buffer_nprobe_multiplier=3.0`, `buffer_pages=804`, `buffer_min_pages=64`, `execution_stride=6599`; async cluster-id copy enabled with separate-stream D2H non-blocking metadata and sync before wave-buffer access. This is metadata, not profiler overlap evidence. | opt013 `results.jsonl`, `summary.csv` |
| Dtype/numerical behavior | Pass by reuse rationale: block-cache dtype bytes are 2 (bf16). opt013 changes Python allocation initialization only and does not change/rebuild CUDA kernels or numerical formulas; admitted K/V values are copied into cache before cache-hit reads. Prior retained validation on the same `.venv`/cu124 stack checked bf16 `batch_gemm_softmax` vs torch reference at `atol=1e-3` with max abs diff `0.000244140625` and exact `gather_copy_vectors`. | `research/validation_low6_async_buffer_mult_3p0_120k_b8_cr0p05/aux_checks.log`, opt013 `CHANGES.md`, `SOURCE_DIFF.patch` |
| Gradient/inference-only scope | Pass scoped to inference only: harness calls `llm.generate(..., do_sample=False)` and exercises no backward, optimizer, training, or gradient path. No backprop safety claim is made. | `throughput_eval/test.py`, opt013 `config.json` |
| Determinism/stability | Pass at oracle/metadata level: same seed, 2/2 correctness, stable allocation metadata, stable late-init metadata, stable block-cache bytes, stable peak process GPU memory, sampler error total 0. Timing varies normally; no stronger deterministic timing claim. | opt013 `results.jsonl`, `summary.csv`, `per_round_table.csv` |
| Execution status vs idea status | Pass: validation execution succeeded (`returncode=0`, failure_class success/none in artifacts); source rebuild remains an environment blocker with `environment_rebuild_toolchain_cuda_version_mismatch`, not a failed kernel idea; idea status is retained bounded tradeoff, not broad speedup. | opt013 `OUTCOME.json`, `research/GROUND_TRUTH.md`, this matrix |

## Raw artifacts

- Baseline: `research/BASELINE_RESULT.json` and `research/baseline_block_cache_single_a100_120k_b8_cr0p05/`
- Opt010 comparison point: `attempts/opt_attempt_010_late_block_cache_allocation/OUTCOME.json` and `attempts/opt_attempt_010_late_block_cache_allocation/late_allocation_low6_async_buffer3_120k_b8_cr0p05/`
- Opt011 comparison point: `attempts/opt_attempt_011_split_late_index_prefill_residency/OUTCOME.json` and `attempts/opt_attempt_011_split_late_index_prefill_residency/raw/harness_120k_b8_cr0p05/`
- Opt012 comparison point: `attempts/opt_attempt_012_pinned_late_metadata_migration/OUTCOME.json` and `attempts/opt_attempt_012_pinned_late_metadata_migration/raw/harness_120k_b8_cr0p05/`
- Opt013 retained candidate: `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/OUTCOME.json`
- Opt013 raw run directory: `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05/`
- Opt013 structured results: `results.jsonl`, `summary.csv`, `per_round_table.csv`
- Opt013 source/evidence: `CHANGES.md`, `SOURCE_DIFF.patch`, `ENV_AUDIT.md`, `ENV_AUDIT.raw.txt`, `raw/harness_120k_b8_cr0p05/environment.json`
