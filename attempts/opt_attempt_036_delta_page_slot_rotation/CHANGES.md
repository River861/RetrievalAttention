# opt036 delta/page slot rotation

Implemented an additive, default-off page-delta mode on top of the existing opt035 slot rotation:

- `RETROINFER_BLOCK_CACHE_SLOT_ROTATION=1` keeps the existing full-layer rotating-slot behavior unchanged unless the new `RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DELTA=1` flag is also set.
- Delta mode preserves the pinned CPU full logical block-cache K/V owner and `<=m` fixed GPU slots with CUDA graph-compatible slot addresses.
- Before `gather_copy_and_concat` or its captured CUDA graph replay, delta mode materializes only WaveBuffer hit pages from the CPU owner into the resident GPU slot.
- After `gather_copy_and_scatter`/admission, delta mode flushes only `update_cache_indices` dirty pages from the slot back to the CPU owner.
- Added telemetry for copy mode, full-layer copy counts/bytes, page H2D/D2H counts/bytes/listed pages by layer, hit-page materialization, dirty-page flushes, pending transfers, and page-index/ownership violations.
- Extended `throughput_eval/reproduce_block_cache.py` to preserve the new telemetry fields in raw, summary, and report artifacts.

Bounded mechanism probe:

- Command artifact: `raw/mechanism_probe/commands.jsonl`
- Results: `raw/mechanism_probe/results.jsonl`
- Raw log: `raw/mechanism_probe/raw_logs/opt036_delta_page_slot_rotation_probe_retroinfer_ctx30000_bsz1_cr0p05_r1.txt`
- Shape: `context_len=30000`, `batch_size=1`, `cache_ratio=0.05`, `gen_len=20`, CUDA graph enabled.
- Outcome: NIAH passed, `path_executed=true`, `copy_mode=page_delta`, `actual_gpu_slot_count=8<=m`, full-layer H2D/D2H counts and bytes were zero, page H2D/D2H counters were nonzero, final pending transfers were zero, and ownership/page-index violations were zero.

Canonical 120K measurement:

- Command artifact: `raw/harness_120k_b8_cr0p05/commands.jsonl`
- Results: `raw/harness_120k_b8_cr0p05/results.jsonl`
- Raw log: `raw/harness_120k_b8_cr0p05/raw_logs/opt036_delta_page_slot_rotation_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`
- Shape: `context_len=120000`, `batch_size=8`, `cache_ratio=0.05`, `gen_len=100`, seed `2025`, CUDA graph enabled.
- Outcome: NIAH passed for all 8 batch items and telemetry proved true page-delta behavior: `copy_mode=page_delta`, `path_executed=true`, CPU owner bytes `6241124352`, `actual_gpu_slot_count=8<=m`, full-layer H2D/D2H counts and bytes `0`, page H2D `3136` transfers / `225964638208` bytes, page D2H `3168` transfers / `17335660544` bytes, final pending transfers `0`, ownership/page-index/generation violations `0`.

Performance verdict:

- Opt036 preserved opt035 block-cache GPU residency (`1488.0 MiB`) and peak process GPU memory (`28010.0 MiB`), saving `13950.0 MiB` peak process GPU memory versus the canonical baseline and `1874.0 MiB` versus opt013.
- Opt036 reduced total opt035 slot-transfer bytes from `1238668148736` to `243300298752` bytes (`-80.36%`), but decode throughput fell to `1.761` tokens/s versus baseline `198.836`, opt013 `198.037`, and opt035 `25.039`.
- The idea is performance-refuted at the canonical shape: Python-level page-slice copies reduce bytes but create much worse decode/e2e latency.
