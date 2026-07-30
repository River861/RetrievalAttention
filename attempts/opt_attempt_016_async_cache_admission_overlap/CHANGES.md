# opt016 async cache-admission overlap

## Code changes

- `cache_hub/retroinfer_cache.py`
  - Added default-off `RETROINFER_ASYNC_CACHE_ADMISSION`.
  - When enabled, block-cache admission (`gather_copy_and_scatter`, or the existing captured update CUDAGraph under `--use_cuda_graph`) is launched on a per-device side stream after `wave_buffer.sync()` confirms update indices are ready.
  - Added explicit current-stream `wait_event` ordering before the next execution-buffer reuse/cache read on that device, covering the shared execution buffer hazard and the same-layer later-read safety requirement.
  - Added final metadata-time draining so attempt telemetry reports no stale pending admission.
  - Exported mechanism telemetry: mode, stream count, launch/sync counts, pending layers, sync point, overlap window, and CUDAGraph-update usage.
- `throughput_eval/reproduce_block_cache.py`
  - Added `RETROINFER_ASYNC_CACHE_ADMISSION` to captured harness env metadata and exported the new async-admission telemetry fields into raw and summary CSVs.

## Evidence and result

- Fresh environment/capability audit: `ENV_AUDIT.md`, `ENV_AUDIT.raw.txt`.
- Fresh optimize frontier record/check: `frontier_optimize_input.json`, `frontier_record.raw.txt`, `frontier_check.raw.txt`.
- Python compile check passed for `cache_hub/retroinfer_cache.py`, `throughput_eval/reproduce_block_cache.py`, and `throughput_eval/test.py`.
- A100 120000x8/gen_len=100/cache_ratio=0.05 opt013-capacity isolation passed correctness with async admission active (`3168` launches/syncs) but decode fell to `195.417338 tok/s` versus opt013 `198.036572 tok/s`; block cache and peak GPU memory were unchanged (`5673.0 MiB`, `29884.0 MiB`).
- The mission-specified lower-memory schedule `1=0.75,2-3=0.75,25=0.625,28-29=0.625` passed correctness and reduced memory (`5602.5 MiB`, `29812.0 MiB`) but failed the retention gate: decode `192.108815 tok/s` and e2e latency `245.498946 s` were worse than opt013.
- No candidate was retained; no 2-round retained-candidate rerun was warranted.
