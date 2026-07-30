# RetroInfer CPU full-KV plus rotating GPU slot scope

## Scope boundary

This scope refresh supersedes the prior static per-layer capacity, stream-only layer, and page-geometry candidate enumeration. The next optimize-stage implementation must investigate the exact requested architecture: CPU memory owns the complete block-cache K/V body for all transformer layers, while GPU memory owns only a bounded rotating working set of block-cache slots. This file is scope evidence only; no implementation, baseline run, Figure 13 run, or new benchmark harness is authorized in this stage.

## Exact target architecture

- **CPU ownership**: allocate a complete per-layer CPU block-cache K/V owner for all `L` layers. The CPU owner is the durable source of truth across decode tokens and layer transitions. Prefer pinned host memory if the allocation succeeds within host RAM and memlock limits; if pinned allocation fails or is resource-prohibitive, classify it as an execution/environment blocker for this idea, not as a failed kernel mechanism.
- **GPU residency**: allocate at most `m` physical GPU block-cache K/V slots. Do not allocate one long-lived `cache_keys/cache_values` tensor pair per layer. Compatibility aliases may point layer APIs at slots, but telemetry must count actual GPU slot tensors, not aliases.
- **Ring semantics**: each slot carries `{slot_id, resident_layer, generation, state}` where state is one of `free`, `h2d_pending`, `resident`, `compute_in_use`, or `d2h_pending`. A layer may read a slot only when `resident_layer == layer_idx`, `generation` matches the expected generation, and its H2D event has completed.
- **Layer pipeline**: while layer `i` executes, asynchronously prefetch future layer `i + m - 1` into the slot that will be reused by the ring, and asynchronously unload/synchronize the updated layer `i - 1` after its last use. H2D and D2H must use separate CUDA streams and events from the compute stream. The compute stream waits only at slot consumption or when a slot is about to be overwritten.
- **D2H ownership rule**: D2H of a layer slot may start only after `weighted_flash_decoding`, `WaveBufferCPU::sync()`, and `gather_copy_and_scatter` admission/update for that layer have completed on the slot. Once D2H is recorded, CPU becomes authoritative for that layer generation and the slot is not reusable until the D2H event completes.
- **H2D ownership rule**: H2D must copy the full CPU-owned K/V body for the target layer into a free/reusable GPU slot. After the H2D event completes, GPU is authoritative for that layer generation until the layer's post-admission D2H completes.
- **No dirty reads or in-flight overwrite**: every slot overwrite must prove no compute graph, gather/scatter update, H2D, or D2H is still using the previous generation. All pending transfers must be zero at decode end and before metadata reporting.

## Current lifecycle binding

- `cache_hub/retroinfer_cache.py` currently computes `cache_size` from `compute_retroinfer_block_cache_capacity()`, constructs one `WaveBufferCPU` per layer with that layer's `cache_sizes[ldx]`, then allocates per-layer GPU `cache_keys/cache_values` when preallocated or later in `prepare_cache()`.
- Existing CPU pinned `list_keys/list_values` store organized clustered K/V for all layers and are used as the miss source in `gather_copy_and_concat`. They are not a substitute for the requested CPU full block-cache K/V owner unless the implementation explicitly proves the block-cache K/V body and LRU state are represented losslessly and D2H-updated after admission.
- `WaveBufferCPU` initializes cluster descriptors as `inBlockCache=false`, performs hit/miss lookup in `batch_access`, updates LRU/admission in `batch_update`, and exposes cache block ids that are logical per-layer block-cache indices. Under slot rotation these logical block ids must remain per-layer logical ids; the physical slot address is a separate layer-to-slot mapping.
- `gather_copy_and_concat` currently reads steady-zone K/V, CPU list K/V misses, and GPU block-cache hits into the execution buffer on the current CUDA stream. Under slot rotation its `key_data3/value_data3` arguments must be the resident slot for `layer_idx`.
- `gather_copy_and_scatter` currently writes admitted pages from the execution buffer back into GPU block-cache tensors on the current stream or the async admission stream. Under slot rotation it writes into the resident slot; D2H copies the whole updated slot after this stream work is complete.
- `sparse_attention_with_cudagraph()` captures per-layer attention/update graphs that reference `self.cache_keys[layer_idx]` and `self.cache_values[layer_idx]`. Slot rotation must preserve fixed device addresses for graph replay by allocating all GPU slots before capture and ensuring each layer graph captures its deterministic slot address, e.g. `layer_idx % m`. If that fixed-address invariant cannot be preserved, the implementation must disable the slot-rotation path for CUDA graph or report a graph-address blocker.

## CUDA graph compatibility constraints

Official CUDA graph documentation separates definition, instantiation, and replay; graph replay assumes the captured operation structure and referenced resources remain valid. NVIDIA also documents cross-stream event capture rules and prohibits synchronization on captured streams/events. The implementation must therefore:

1. Allocate CPU owners, GPU slots, H2D/D2H streams, and events before graph capture.
2. Keep slot tensor addresses stable for the lifetime of captured graphs.
3. Avoid allocating replacement slot tensors between graph replays.
4. Keep H2D/D2H orchestration outside captured attention/update graphs unless graph nodes use fixed slot addresses and explicit supported graph update APIs.
5. Preserve existing `use_cuda_graph=True` behavior by default when the new feature is disabled.

## Public API contract

Existing user-facing APIs and benchmark commands must continue to work unchanged:

- `config.generate_config(model_name, context_len, "RetroInfer", retrieval_budget, estimation_budget, cache_ratio, use_cuda_graph, gpu_only)`
- `llm.generate(attention_type="RetroInfer", ..., attn_config=attn_config, prefill_method=...)`
- Existing config keys remain compatible: `static_pattern_start/end`, `core`, `n_centroids`, `n_segment`, `pages_per_cluster`, `retrieval_budget`, `estimation_budget`, `cache_ratio`, `buffer_cluster_num`, `use_cuda_graph`, `gpu_only`.
- `RETROINFER_RESULT_JSON`, `RETROINFER_OUTPUT_JSON`, and `block_cache_metadata()` remain parseable by `throughput_eval/reproduce_block_cache.py`.

The new path must be opt-in and default off. Acceptable knobs are additive, for example `RETROINFER_BLOCK_CACHE_SLOT_ROTATION=1` and `RETROINFER_BLOCK_CACHE_GPU_SLOTS=<m>`, with defaults preserving current per-layer GPU cache behavior.

## Allowed edit surface

Primary allowed surfaces:

- `cache_hub/retroinfer_cache.py`: allocation, `prepare_cache()`, CPU pinned block-cache owner, GPU slot tensors, H2D/D2H streams/events, slot generation state, sparse attention, CUDA graph capture/replay, and `block_cache_metadata()`.
- `library/retroinfer/retroinfer_kernels/src/wave_buffer_cpu.cpp`: only if logical block-cache ownership, LRU/admission state, or exposed telemetry must be adjusted for slot rotation. Keep Python call semantics compatible.
- `library/retroinfer/retroinfer_kernels/src/gather_copy.cu`: only if gather/scatter must accept slot-backed tensors differently while preserving current wrapper semantics or adding compatible wrappers.
- `throughput_eval/reproduce_block_cache.py` and `throughput_eval/test.py`: metric-field extensions only; do not create a new harness.
- `config/config.py` and model adapters only for additive default-off configuration plumbing.

Do not modify `research/PIPELINE_STATE.json`, overwrite final `experiments/cache_ratio` artifacts, run Figure 13, recreate vendor kernels/libraries, or replace the benchmark harness.

## Correctness oracle and bounded experiment contract

Correctness comes before timing. The first mechanism probe must be bounded and must pass the existing NIAH oracle:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python throughput_eval/test.py \
  --context_len 120000 --batch_size 8 --gen_len 100 \
  --task_name NIAH --attn_type RetroInfer ...
```

Required oracle fields:

- process return code `0`;
- `RETROINFER_RESULT_JSON` exists with `failure_class == success`;
- `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth == true`;
- no stale-slot, pending-transfer, CUDA graph address, or buffer-overflow error.

Only after the mechanism probe is correct may the minimal real experiment run at `context_len=120000`, `batch_size=8`, `cache_ratio=0.05`, `gen_len=100`, `seed=2025`, comparing current canonical baseline and opt013. Report block-cache GPU residency, peak process GPU memory, decode throughput, e2e generated throughput, correctness, host memory, transfer bytes/counts, and wait overhead. No speed claim is valid without real A100 measurement.

## Required telemetry

`block_cache_metadata()` and harness fields must expose enough evidence to audit that the real K/V body rotated:

- feature enabled/env, requested `m`, actual GPU slot count, and `actual_gpu_slot_count <= m`;
- CPU full block-cache bytes, pinned bytes, pageable bytes, allocation success/failure, and host memlock/host RAM facts when available;
- nominal logical layer count `L`, nominal per-layer pages/bytes, CPU owner bytes per layer, and actual GPU slot bytes;
- per-layer slot mapping and per-slot `resident_layer`, `generation`, and state transitions;
- H2D bytes/counts per layer and total, D2H bytes/counts per layer and total;
- H2D/D2H stream count, event wait count, wait enqueue time, explicit synchronization time, and final pending H2D/D2H count;
- slot overwrite prevention counters, generation mismatch counters, and dirty-read prevention counters;
- CUDA graph mode: disabled/off, fixed-slot-address graph, or graph blocker reason;
- path-executed boolean proving the CPU-owner plus GPU-slot path ran, not metadata staging, direct stream-only gather, late allocation, or static capacity scaling.

## Initial `m` rationale

For the canonical 120K x batch 8, cache ratio 5% shape, the current full GPU block-cache geometry is:

- `L=32`, `batch_size=8`, `kv_head=8`, `cache_pages_per_layer=744`, `page_size=8`, `head_dim=128`, `dtype_bytes=2`;
- K+V bytes per layer = `2 * 8 * 8 * 744 * 8 * 128 * 2 = 195,035,136` bytes = `186 MiB`;
- CPU full logical block-cache K/V for all layers = `5.8125 GiB`;
- audited host memory: `MemTotal=226,765,912 kB`, `MemAvailable=220,147,364 kB`, `ulimit -l=28,345,736 kB`, so the 5.8125 GiB CPU owner is below the current memlock limit before accounting for other pinned buffers;
- canonical current GPU cache residency at 5% is 5.8125 GiB, and opt013 reports 5,673 MiB block-cache residency with 29,884 MiB peak process GPU memory.

Use **`m=8`** as the initial implementation setting. It allocates `8 * 186 MiB = 1.453125 GiB` of physical GPU block-cache slots, saving roughly 4.36 GiB of GPU block-cache residency versus full 32-layer residency while giving a seven-layer prefetch lead. At an assumed A100 PCIe practical H2D bandwidth of roughly 20-24 GiB/s, a 186 MiB full-layer H2D copy costs about 7.6-9.3 ms; opt013's 198 tok/s decode throughput at batch 8 implies about 40 ms per decode step or about 1.25 ms/layer, so seven intervening layers are the smallest lead that plausibly hides one full-layer H2D copy. This is a planning estimate, not a speed claim.

Use **`m=12`** as the only optional comparator if telemetry shows H2D wait dominates with `m=8`. It raises physical slot residency to `2.1796875 GiB` and gives an eleven-layer lead. Do not run a broad `m` matrix in this mission.

## Explicit non-approximations

The following do not satisfy this scope:

- staging only index metadata while leaving per-layer GPU block-cache K/V allocated;
- stream-only direct gather from `list_keys/list_values` without rotating the block-cache K/V body;
- late allocation/uninitialized allocation of all per-layer GPU cache tensors;
- static per-layer capacity scaling, stream-only layer selection, or page-geometry changes;
- unified-memory-only designs that skip explicit H2D/D2H streams/events/generation ownership;
- telemetry that reports logical layer aliases as reduced GPU residency without proving actual physical slot count.

If pinned host memory, CUDA graph fixed-address behavior, OOM, or PCIe transfer time becomes a clear blocker, preserve the code/logs and stop with a negative result or revised plan. Do not fall back to old static candidates for this inserted task.
