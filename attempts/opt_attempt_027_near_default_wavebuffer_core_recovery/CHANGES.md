# opt_attempt_027_near_default_wavebuffer_core_recovery

## Code changes

- No source edits were made for opt027.
- This attempt reused the existing default-off `RETROINFER_WAVE_BUFFER_CORE_OVERRIDE` runtime path and the opt023 lower-memory stack.
- `SOURCE_DIFF.patch` records the relevant dirty source surfaces present during measurement; `SOURCE_PROVENANCE.json` records that opt027 itself did not change source.

## Screen

- Fresh environment audit used the project `.venv` with torch CUDA 12.4 and installed `retroinfer_kernels`; source rebuild remains blocked by the audited CUDA 12.4 `nvcc`/`CUDA_HOME` mismatch and was not attempted.
- Harness shape for all screens: `context_len=120000`, `batch_size=8`, `gen_len=100`, `cache_ratio=0.05`, `seed=2025`.
- Stack: late block-cache allocation, pinned side-stream metadata migration, uninitialized late block cache, async cluster-id copy, `RETROINFER_LAYER_CACHE_CAPACITY_SCALE=1=0.75,2-3=0.75,25=0.75,28-29=0.75,31=0.75`, `RETROINFER_BUFFER_NPROBE_MULTIPLIER=2.75`, scratch init uninitialized, async wave/cache-admission/stream-only/staging unset.

## Result

All three one-round screens passed correctness, kept `use_cuda_graph=true`, had zero buffer-overflow warnings, and preserved the lower-memory stack (`block_cache_total_mib=5626.5`, `peak_process_gpu_memory_mib=29820.0`). None recovered both opt013 throughput gates, so no two-round confirmation was allowed.

| Candidate | Decode tok/s | E2E generated tok/s | Gate reason |
| --- | ---: | ---: | --- |
| core=21 | 192.933180 | 3.276620 | `decode_regressed_vs_opt013` |
| core=20 | 194.650527 | 3.261882 | `e2e_regressed_vs_opt013` |
| core=18 | 195.404773 | 3.257243 | `e2e_regressed_vs_opt013` |

Core=21 was the best e2e one-round result but regressed decode versus opt013; core=18 was the best decode result among opt027 screens but still regressed decode and e2e versus opt013.
