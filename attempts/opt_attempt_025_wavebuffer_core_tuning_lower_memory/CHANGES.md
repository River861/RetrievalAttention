# opt_attempt_025_wavebuffer_core_tuning_lower_memory

## Code changes

- Added default-off `RETROINFER_WAVE_BUFFER_CORE_OVERRIDE` handling in the offload `retroinfer_cache` path.
- The public API is unchanged: model loaders still pass `retroinfer_config["core"]`; when the env var is unset or empty, the existing config-generated core value is used.
- When set to a positive integer not exceeding `os.cpu_count()`, the override becomes the effective core count used for `retroinfer_kernels.ThreadPool` and each `WaveBufferCPU`.
- Exported the configured/effective core values and override status through block-cache metadata and the existing `reproduce_block_cache.py` result/summary surfaces.

## Validation and screen

- Project runtime audit used `.venv` with torch CUDA 12.4 and installed `retroinfer_kernels`; CUDA/C++ source rebuild remains blocked by the audited CUDA 12.4 `nvcc`/`CUDA_HOME` mismatch and was not attempted.
- Compile/default check: `.venv/bin/python -m py_compile cache_hub/retroinfer_cache.py throughput_eval/reproduce_block_cache.py throughput_eval/test.py`; unset override resolved to the existing NUMA-derived core count 22, and override `16` resolved to effective core 16.
- Predeclared core overrides: 16, 12, 8.
- Harness shape for all screens: `context_len=120000`, `batch_size=8`, `gen_len=100`, `cache_ratio=0.05`, `seed=2025`, opt023 lower-memory stack.

## Result

All three one-round screens passed correctness and preserved the lower-memory stack (`block_cache_total_mib=5626.5`, `peak_process_gpu_memory_mib=29820.0`), but no core override recovered both decode and e2e throughput versus opt013.

| Candidate | Decode tok/s | E2E generated tok/s | Gate reason |
| --- | ---: | ---: | --- |
| core=16 | 197.030129 | 3.273543 | `decode_regressed_vs_opt013` |
| core=12 | 196.379681 | 3.257124 | `e2e_regressed_vs_opt013` |
| core=8 | 199.745636 | 3.258779 | `e2e_regressed_vs_opt013` |

No two-round confirmation was run because every predeclared one-round candidate failed at least one required throughput gate.
