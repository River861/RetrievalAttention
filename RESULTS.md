# RetroInfer opt013 uninitialized late block-cache allocation report

本 report-stage 结论绑定到 validated `opt013`：`uninitialized_late_block_cache_allocation`。有效范围只覆盖 single A100 80GB、`context_len=120000`、`batch_size=8`、`gen_len=100`、`cache_ratio=0.05`、2 rounds、seed 2025 的同口径验证。opt013 继承 opt012 的 forced-late block-cache allocation、low6 fractional capacity、pinned side-stream late index-metadata migration、async cluster-id copy 和 `RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`，新增默认关闭的 late block-cache K/V `torch.empty` allocation policy。

结论边界：opt013 是 bounded same-memory/e2e tradeoff。它保持 opt012 的 resident block-cache size 与 peak process GPU memory，e2e generated throughput 从 opt012 的 `3.264409640 tok/s` 提升到 `3.268887524 tok/s`；decode throughput 从 opt012 的 `201.131671 tok/s` 回退到 `198.036572 tok/s`。因此不声明 decode speedup、广义 speedup、或 profiler-proven transfer/compute overlap。resident block-cache byte reduction versus baseline 来自继承的 low6 fractional-capacity 配置，不来自 `torch.empty`。

## 核心指标

| 指标 | Canonical baseline | opt010 forced-late low6 | opt011 split-late GPU metadata | opt012 pinned side-stream | Current opt013 | Delta vs baseline | Delta vs opt012 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correctness | 2/2 pass | 2/2 pass | 2/2 pass | 2/2 pass | 2/2 pass | unchanged | unchanged |
| Block cache bytes | 6241124352 | 5948571648 | 5948571648 | 5948571648 | 5948571648 | -292552704 (-4.688%) | +0 (+0.000%) |
| Block cache MiB / GiB | 5952.0 / 5.812500000 | 5673.0 / 5.540039062 | 5673.0 / 5.540039062 | 5673.0 / 5.540039062 | 5673.0 / 5.540039062 | -279.0 MiB (-4.688%) | +0.0 MiB (+0.000%) |
| Late uninitialized block-cache bytes | 0 | 0 | 0 | 0 | 5948571648 | opt013-only telemetry | +5948571648 |
| Peak process GPU memory MiB mean/max | 41960.0 / 41960.0 | 29884.0 / 29884.0 | 35610.0 / 35610.0 | 29884.0 / 29884.0 | 29884.0 / 29884.0 | -12076.0 (-28.780%) | +0.0 (+0.000%) |
| Decode throughput tok/s mean | 198.836414 | 198.585321 | 198.562825 | 201.131671 | 198.036572 | -0.799843 (-0.402%) | -3.095099 (-1.539%) |
| E2E latency s mean | 244.122884 | 245.266826 | 244.172720 | 245.067283 | 244.734262 | +0.611378 (+0.250%) | -0.333021 (-0.136%) |
| E2E generated throughput tok/s (`8*100/e2e_latency_s`) | 3.277046 | 3.261779 | 3.276369 | 3.264410 | 3.268888 | -0.008158 (-0.249%) | +0.004478 (+0.137%) |

opt013 retained metrics：2/2 correctness、`5673.0 MiB` block cache、`29884.0 MiB` peak process GPU memory、`198.036572 tok/s` decode、`244.734262 s` mean e2e latency、`3.268887524 tok/s` e2e generated throughput、64 late K/V cache tensors / `5,948,571,648` bytes allocated through the uninitialized late-init path。

## 运行环境

| 项目 | 值 |
| --- | --- |
| Repository HEAD | `7096eab9190da389bbc75c1992140d1432d9d8ec` |
| Python | `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`, Python `3.11.15` |
| Platform | `Linux-6.17.0-1020-azure-x86_64-with-glibc2.39` |
| GPU | `NVIDIA A100 80GB PCIe`, compute capability `8.0`, validation environment records total memory `81920 MiB`, driver `580.173.02` |
| Runtime packages | torch `2.5.1+cu124`, torch CUDA `12.4`, vLLM `0.6.5`, transformers `4.49.0`, flash-attn `2.7.3`, flashinfer `0.2.4+cu124torch2.5`, triton `3.1.0`, pybind11 `2.12.0`, numpy `1.26.4`, `retroinfer_kernels` import OK, `weighted_flash_decoding` `0.1` |
| Installed kernel symbols | `ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, `batch_gemm_softmax` imported successfully |
| Source rebuild status | `blocked_for_source_rebuild`: execution used installed cu124 runtime/extensions. CUDA/C++ source rebuild remains blocked/non-goal because no CUDA 12.4-compatible `nvcc`/`CUDA_HOME` is proven (`/usr/bin/nvcc` is CUDA 12.0 and `/usr/local/cuda-13.3/bin/nvcc` is CUDA 13.3). This is an environment/toolchain blocker, not an opt013 idea failure. |

## Commands and raw artifacts

Healthy baseline、Figure 13、以及无关扩展矩阵均未重跑。Comparisons reuse `research/BASELINE_RESULT.json` plus opt010/opt011/opt012 preserved artifacts.

Validated opt013 command:

```bash
env -u RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY \
  -u RETROINFER_STAGE_INDEX_METADATA_LAYERS \
  -u RETROINFER_STREAM_ONLY_LAYERS \
  -u RETROINFER_LAYER_CACHE_RESIDENCY \
  -u RETROINFER_CACHE_TELEMETRY \
  RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late \
  RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream \
  RETROINFER_LATE_BLOCK_CACHE_INIT=uninitialized \
  RETROINFER_ASYNC_CLUSTER_ID_COPY=1 \
  RETROINFER_LAYER_CACHE_CAPACITY_SCALE='1=0.75,2-3=0.75,25=0.75,28-29=0.75' \
  RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0 \
  .venv/bin/python throughput_eval/reproduce_block_cache.py \
    --suite opt013_uninitialized_late_block_cache \
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
    --output-dir attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05
```

Report-stage frontier refresh:

```bash
PYTHONPATH=/home/v-xuchuanluo/Argus .venv/bin/python -m argus_skill.verticals.kernel_engineering.frontier_watch record --project-root /home/v-xuchuanluo/RetroInfer --stage report --input attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/frontier_report_input.json
python -m argus_skill.verticals.kernel_engineering.frontier_watch check --project-root /home/v-xuchuanluo/RetroInfer --stage report
```

Primary evidence paths:

| 用途 | 路径 |
| --- | --- |
| Authoritative validation result | `research/VALIDATION_RESULT.json` |
| Validated coverage matrix | `research/VALIDATION_MATRIX.md` |
| Baseline result and raw directory | `research/BASELINE_RESULT.json`, `research/baseline_block_cache_single_a100_120k_b8_cr0p05/` |
| Opt010 comparison preserved | `attempts/opt_attempt_010_late_block_cache_allocation/OUTCOME.json`, `attempts/opt_attempt_010_late_block_cache_allocation/late_allocation_low6_async_buffer3_120k_b8_cr0p05/` |
| Opt011 comparison preserved | `attempts/opt_attempt_011_split_late_index_prefill_residency/OUTCOME.json`, `attempts/opt_attempt_011_split_late_index_prefill_residency/raw/harness_120k_b8_cr0p05/` |
| Opt012 comparison preserved | `attempts/opt_attempt_012_pinned_late_metadata_migration/OUTCOME.json`, `attempts/opt_attempt_012_pinned_late_metadata_migration/raw/harness_120k_b8_cr0p05/` |
| Opt013 outcome | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/OUTCOME.json` |
| Opt013 run directory | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05/` |
| Opt013 structured outputs | `results.jsonl`, `results.csv`, `summary.csv`, `per_round_table.csv`, `config.json`, `environment.json`, `commands.jsonl` |
| Opt013 raw logs | `raw_logs/opt013_uninitialized_late_block_cache_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`, `raw_logs/opt013_uninitialized_late_block_cache_retroinfer_ctx120000_bsz8_cr0p05_r2.txt` |
| Opt013 memory samples | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05/memory_samples/` |
| Environment/code evidence | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/ENV_AUDIT.md`, `ENV_AUDIT.raw.txt`, `SOURCE_DIFF.patch`, `CHANGES.md` |
| Frontier report binding | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/frontier_report_input.json`, `frontier_report_record.raw.txt`, `frontier_report_check.raw.txt`, `research/frontier/report.json` record `e139a3500c1ef270`, ledger `research/FRONTIER_WATCH.jsonl` |

## Correctness and stability

Both opt013 rounds returned `returncode=0`, harness status `passed`, and `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`; each round had 8/8 generated outputs containing groundtruth `9879991`. Sampler errors were 0, observed process-count peak was 1, and allocation/index-metadata/late-init metadata was stable across both rounds.

| Round | Status | Groundtruth oracle | Late init | Allocation / migration | Block cache MiB | Peak GPU MiB | Decode tok/s | E2E latency s | E2E tok/s | Raw log |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| r1 | passed, returncode 0 | true, 8/8 | `policy=uninitialized`, `mode=uninitialized_empty`, tensors 64, bytes 5948571648 | `late` / `allocate_after_prefill`; pinned side-stream metadata copy 7847116800 bytes; sync `after_allocate_computation_buffer_before_decode` | 5673.0 | 29884.0 | 195.253579 | 243.923672 | 3.279714 | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05/raw_logs/opt013_uninitialized_late_block_cache_retroinfer_ctx120000_bsz8_cr0p05_r1.txt` |
| r2 | passed, returncode 0 | true, 8/8 | `policy=uninitialized`, `mode=uninitialized_empty`, tensors 64, bytes 5948571648 | `late` / `allocate_after_prefill`; pinned side-stream metadata copy 7847116800 bytes; sync `after_allocate_computation_buffer_before_decode` | 5673.0 | 29884.0 | 200.819565 | 245.544852 | 3.258061 | `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05/raw_logs/opt013_uninitialized_late_block_cache_retroinfer_ctx120000_bsz8_cr0p05_r2.txt` |

Timing uncertainty is bounded to two real-hardware rounds, not a generalized distribution. opt013 decode stdev is `3.935747 tok/s`; memory sampling is process-level `nvidia-smi` at `0.5s`, so peak process memory is observed at sampler granularity. Correctness oracle, block-cache bytes, late allocation decision, late init mode, index metadata migration policy, buffer pages, execution stride, and peak process GPU memory were stable across the two rounds.

## Dispatch/API and implementation boundaries

| Surface | Reported behavior |
| --- | --- |
| `RETROINFER_LATE_BLOCK_CACHE_INIT` | Default-off runtime control. Unset/empty/default/zero/zeros/zero_fill/zero_filled preserve original zero-filled behavior. `empty`/`uninitialized`/`uninitialised`/`no_zero`/`no_zero_fill` are explicit opt-in and use `torch.empty` only for forced-late `prepare_cache()` `cache_keys/cache_values` tensors. Invalid values raise `ValueError` naming the env and accepted families. |
| Uninitialized safety invariant | `WaveBufferCPU` initializes clusters as misses and marks cache hits only after cache admission; Python waits for the update path and then `gather_copy_and_scatter()` writes admitted K/V blocks into `cache_keys/cache_values` before later hit reads can source from those tensors. The claim is limited to the measured late-allocation path. |
| Preallocated/default block cache | If block cache is preallocated before prefill while `RETROINFER_LATE_BLOCK_CACHE_INIT=uninitialized` is set, uninitialized mode is not applied and source metadata records `not_applied_to_preallocated_block_cache`. |
| `RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY` | Opt013 uses forced `late` allocation. The measured path records `block_cache_preallocated_before_prefill=false`, `block_cache_allocated_after_prefill=true`, and `block_cache_allocation_prepare_cache_calls=1`. |
| Pinned late metadata migration | Inherited from opt012: prefill metadata stays pinned CPU (`prefill_gpu_bytes=0`, `prefill_host_pinned_bytes=7847116800`); `prepare_cache()` performs one `pinned_h2d_non_blocking_side_stream` H2D migration of `7847116800` bytes and synchronizes before decode. Host-side mean elapsed is `22.973207 ms`; prepare-window mean is `590.832519 ms`. This is mechanism/timing metadata, not profiler overlap proof. |
| Low6 fractional capacity | `RETROINFER_LAYER_CACHE_CAPACITY_SCALE=1=0.75,2-3=0.75,25=0.75,28-29=0.75`; layers 1/2/3/25/28/29 are 558 pages, other layers are 744 pages, total pages `22692`, total vectors `181536`, total bytes `5948571648`. |
| Async cluster-id copy | `RETROINFER_ASYNC_CLUSTER_ID_COPY=1`; metadata reports `separate_stream_d2h_non_blocking`, destination `pinned_cpu_cluster_ids`, stream count 1, sync point `before_wave_buffer_batch_access`, overlap window `estimation_zone_gather_and_weighted_decoding`. This is code-path metadata, not profiler overlap evidence. |
| Buffer multiplier | `RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`; metadata reports `buffer_pages=804`, `buffer_min_pages=64`, `buffer_probe_pages=804`, `buffer_retrieval_clusters=134`, `execution_stride=6599`. |
| Disabled non-target mechanisms | `RETROINFER_CACHE_TELEMETRY`, `RETROINFER_LAYER_CACHE_RESIDENCY`, `RETROINFER_STREAM_ONLY_LAYERS`, and `RETROINFER_STAGE_INDEX_METADATA_LAYERS` stayed unset/off; opt013 is not an opt009 staging/stream-only rerun and not a new benchmark harness. |
| Code/API scope | Source changes are limited to Python allocation/harness metadata surfaces (`cache_hub/retroinfer_cache.py`, `throughput_eval/reproduce_block_cache.py` in `SOURCE_DIFF.patch`). No CUDA/C++ kernels were changed or rebuilt, and no public model API contract is claimed changed. |

## Frontier evidence and claim boundaries

Fresh report-stage frontier record `e139a3500c1ef270` binds opt013 in `research/frontier/report.json` and append-only `research/FRONTIER_WATCH.jsonl`. Official PyTorch documentation supports the core allocation boundary: `torch.empty` returns uninitialized memory, `torch.zeros` returns zero-filled tensors, and uninitialized memory is valid only when code guarantees initialization before use. Official CUDA/PyTorch guidance supports pinned host memory, non-blocking copies, side streams, and explicit synchronization as mechanisms that can enable async transfer, but profiler/timeline evidence is required before claiming actual transfer/compute overlap. Adjacent KV-cache offload, paged, and tiered-residency systems support framing this as a workload- and hardware-dependent memory-throughput tradeoff; they are context, not proof that opt013 generalizes beyond the measured RetroInfer point.

## Regressions, uncertainty, and limitations

- Opt013 does not reduce resident block-cache bytes or peak process GPU memory versus opt012: both remain `5673.0 MiB` and `29884.0 MiB`.
- The only opt013-specific positive timing evidence is e2e generated throughput versus opt012: `+0.004478 tok/s` (`+0.137%`). Decode regressed versus opt012 by `-3.095099 tok/s` (`-1.539%`), versus baseline by `-0.799843 tok/s` (`-0.402%`), versus opt010 by `-0.548749 tok/s` (`-0.276%`), and versus opt011 by `-0.526254 tok/s` (`-0.265%`).
- Opt013 e2e generated throughput is still lower than canonical baseline by `0.008158 tok/s` (`-0.249%`) and lower than opt011 by `0.007482 tok/s` (`-0.228%`); it is not an opt011 or baseline e2e replacement.
- No Nsight/PyTorch profiler trace or CUDA timeline was collected. This report does not claim transfer/compute overlap, allocation-kernel speedup, or decode speedup.
- The uninitialized allocation safety claim is limited to opt-in forced-late K/V block-cache tensors under the recorded WaveBuffer miss-before-admission-write invariant. It is not generalized to preallocated cache, index metadata, execution buffers, training, gradients, or other tensor values.
- bf16 installed-kernel numerical checks are reused from prior retained validation on the same `.venv`/cu124 stack because opt013 did not change or rebuild CUDA/C++ kernels.
- Source rebuild status remains separate from idea status: execution succeeded using installed cu124 extensions, while source rebuild remains blocked by the CUDA 12.4 compiler/CUDA_HOME mismatch.
- This report did not rerun a healthy baseline, did not run Figure 13, did not overwrite comparison artifacts, and does not rely on editing `research/PIPELINE_STATE.json`.
- Independent review remains required before stage closure.
