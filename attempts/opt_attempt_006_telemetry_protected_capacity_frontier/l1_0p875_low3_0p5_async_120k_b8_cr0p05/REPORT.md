# RetroInfer block-cache capacity single-A100 report

- Generated at: `2026-07-29T13:55:08+00:00`
- Suite: `opt_attempt_006_l1_0p875_low3_0p5_async`
- Model/task: `gradientai/Llama-3-8B-Instruct-Gradient-1048k` / `NIAH`
- Context lengths: `[120000]`
- Batch sizes: `[8]`
- Decode tokens per request: `100`
- Rounds per point: `1`
- Seed: `2025`
- Retrieval/estimation budget: `0.018` / `0.232`
- Cache ratios: `[0.05]`
- Baseline/default ratio for deltas: `0.05`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- GPU observed: `0, NVIDIA A100 80GB PCIe, 81920 MiB, 580.173.02`
- Package availability: `flash_attn=found, flashinfer=found, minference=missing, retroinfer_kernels=found, torch=found, transformers=found, vllm=found`

## 1. Paper/code mapping for block cache

| paper_concept | paper_evidence | code_mapping |
| --- | --- | --- |
| Tripartite attention zones | Paper Section 1/4.2 partitions tokens into steady, retrieval, and estimation zones; steady and retrieval compute precise attention, estimation approximates by centroids. | `retroinfer_cache.static_pattern_start/end` and `steady_zone_keys/values` implement the steady zone; `nprobe`/`nprobe_new` are retrieval-zone cluster counts; `es_cluster_num` is the estimation-zone count. |
| Wave buffer | Paper Section 4.1/4.3 says the wave buffer contains a GPU block cache, a steady-zone buffer, and an execution buffer, with a CPU-resident buffer manager. | `cache_hub/retroinfer_cache.py` creates per-layer `WaveBufferCPU`, `cache_keys/cache_values`, `steady_zone_*`, and `execution_buffer_*`; `wave_buffer_cpu.cpp` owns the CPU `BufferManager`. |
| GPU KV block cache capacity | Paper Section 5.1 sets GPU cache size to 5% of all KV vectors and physical block size to 2KB. | `config.compute_retroinfer_block_cache_capacity()` turns `cache_ratio` into cached clusters/pages; `cache_ratio=0.05` is the paper/default 5% setting, while `cache_ratio=0.0` preserves the code fallback `3 * retrieval_clusters`. Python `page_size=8` bf16 vectors gives 2KB per K or V page. |
| Logical clusters vs physical blocks | Paper Section 4.3 describes cluster-level access, fixed-size KV blocks, and a cluster mapping table. | `WaveBufferCPU` stores `ClusterDescriptor` entries with `inBlockCache`, GPU block IDs, CPU start index, block count, and LRU pointer; `BufferManager.capacity` is the per-group block-cache page budget. |
| Synchronous access and asynchronous update | Paper Section 4.3 states block-cache access is on the critical path, but replacement/update is asynchronous. | `WaveBufferCPU::para_batch_access()` synchronously determines hit/miss blocks, then queues `para_batch_updata()`; Python waits with `wave_buffer.sync()` before `gather_copy_and_scatter()` updates cache tensors. |

Design choice: this experiment varies `--cache_ratio`, because that value now feeds the same `compute_retroinfer_block_cache_capacity()` helper used by the runtime constructor before it allocates `cache_keys/cache_values` and instantiates `WaveBufferCPU`. The 0.05 point matches the paper's 5% GPU cache setting; `cache_ratio=0.0` remains the repository's backward-compatible fallback when included.

## 2. Execution and artifact integrity

- Planned runs: `1`
- Successful measured runs: `1`
- Raw stdout/stderr: `raw_logs/*.txt`.
- Process-level GPU memory samples: `memory_samples/*.jsonl`.
- Structured run records: `results.jsonl` and `results.csv`.
- Aggregates and default-relative deltas: `summary.csv`, `per_round_table.csv`, and `deltas.csv`.

## 3. Per-round throughput and memory

| context_len | batch_size | cache_ratio | cache_role | round | block_cache_total_gib | block_cache_percent_of_a100_80gb | block_cache_percent_of_peak_process_memory | non_block_cache_peak_process_memory_mib | decode_throughput_tokens_s | peak_process_gpu_memory_mib | failure_class | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120000 | 8 | 0.0500 | paper_default_5pct | 1 | 5.5176 | 6.8970 | 13.5622 | 36010.0000 | 194.6929 | 41660.0000 | success | passed |

## 4. Block-cache memory accounting and aggregate performance

| context_len | batch_size | cache_ratio | cache_role | block_cache_total_gib | block_cache_percent_of_a100_80gb | mean_block_cache_percent_of_peak_process_memory | mean_non_block_cache_peak_process_memory_mib | round_decode_throughput_tokens_s | mean_decode_throughput_tokens_s | variance_decode_throughput_tokens_s | mean_peak_process_gpu_memory_mib | mean_torch_cuda_peak_allocated_mib | mean_torch_cuda_peak_reserved_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120000 | 8 | 0.0500 | paper_default_5pct | 5.5176 | 6.8970 | 13.5622 | 36010.0000 | r1=194.692875 | 194.6929 |  | 41660.0000 | 38255.5688 | 41130.0000 |

## 5. Deltas relative to cache_ratio 0.05

| context_len | batch_size | cache_ratio | block_cache_total_gib_delta | peak_process_gpu_memory_delta_mib | peak_process_gpu_memory_delta_pct | block_cache_percent_of_peak_process_memory_delta | non_block_cache_peak_process_memory_delta_mib | decode_throughput_delta_tokens_s | decode_throughput_delta_pct | decode_throughput_ratio_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120000 | 8 | 0.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## 6. Context-specific tradeoff summary

| context_len | batch_size | best_throughput_cache_ratio | best_throughput_delta_pct | best_throughput_memory_delta_pct | lowest_memory_cache_ratio | lowest_memory_delta_pct | lowest_memory_throughput_delta_pct | best_memory_saving_within_5pct_throughput | best_tradeoff_score_cache_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120000 | 8 | 0.0500 | 0.0000 | 0.0000 | 0.0500 | 0.0000 | 0.0000 |  | 0.0500 |

## 7. Curve artifacts

Curves are generated directly from the measured rows in `summary.csv`.

| artifact |
| --- |
| curves/curve_points.csv |
| curves/block_cache_block_cache_peak_share.svg |
| curves/block_cache_block_cache_peak_share.pdf |
| curves/block_cache_block_cache_peak_share.png |
| curves/block_cache_peak_process_gpu_memory.svg |
| curves/block_cache_peak_process_gpu_memory.pdf |
| curves/block_cache_peak_process_gpu_memory.png |
| curves/block_cache_decode_throughput.svg |
| curves/block_cache_decode_throughput.pdf |
| curves/block_cache_decode_throughput.png |
| curves/curve_manifest.json |

## 8. Trend, tradeoff, and comparability note

Best mean decode throughput in the measured matrix occurred at context 120000, batch 8, cache_ratio 0.05; the lowest measured process peak memory occurred at context 120000, batch 8, cache_ratio 0.05. Deltas in the table are computed only within identical context/batch/gen/seed groups against cache_ratio 0.05.

The block cache changes GPU memory directly through the per-layer `cache_keys/cache_values` tensors and also changes data-transfer behavior through the LRU-managed hit/miss path. Throughput therefore need not vary monotonically with estimated cache bytes: very small caches can reduce memory but increase CPU-to-GPU miss traffic, while larger caches consume more GPU memory and can help only if their extra capacity raises the hit ratio enough to offset allocation and copy overhead.

Comparability note: the paper reports the default 5% GPU cache on an A100 80GB server with a specific CPU/NUMA/PCIe setup. This artifact uses the visible single A100 80GB GPU, the local CPU/NUMA topology, the repository NIAH prompts, and a bounded pressure matrix. Absolute numbers are host-specific; the valid conclusion is the within-host trend across cache capacities under fixed non-cache variables.

## 9. Artifact manifest

- `config.json`: exact matrix configuration, command plan, and paper/code mapping.
- `commands.jsonl`: exact command for every planned run.
- `environment.json`: hardware, git, package, CUDA, and NUMA observations.
- `raw_logs/*.txt`: unedited stdout/stderr, including structured `RETROINFER_RESULT_JSON` output.
- `memory_samples/*.jsonl`: sampled process-level GPU memory events.
- `results.jsonl` / `results.csv`: per-run parsed records.
- `per_round_table.csv`: report-ready per-round table.
- `summary.csv`: grouped means, stdev, variance, and block-cache capacity metadata.
- `deltas.csv`: memory and throughput deltas against cache_ratio 0.05 within identical non-cache settings.
- `curves/curve_points.csv` and `curves/curve_manifest.json`: reconstructable cache-ratio curve inputs and generated curve manifest.
- `curves/block_cache_*.{pdf,svg,png}`: generated memory-share, peak-memory, and throughput curves.
