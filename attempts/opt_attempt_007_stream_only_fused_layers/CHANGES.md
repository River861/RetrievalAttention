# opt007 stream-only fused layers changes

- Added default-off `RETROINFER_STREAM_ONLY_LAYERS` parsing to the RetroInfer offload cache.
- Selected layers now receive zero GPU block-cache capacity, keep cluster cumsums for organized list KV, and use installed `gather_copy_cluster_and_concat_fuse` to assemble the execution buffer directly from steady-zone GPU KV plus pinned CPU/list KV.
- Stream-only layers skip `WaveBufferCPU.batch_access()`, LRU sync/admission telemetry, and `gather_copy_and_scatter` GPU block-cache admission during decode; unselected layers keep the existing block-cache path.
- Exposed stream-only metadata and env capture through `block_cache_metadata()` and `throughput_eval/reproduce_block_cache.py`.
- Exhausted the bounded three-candidate opt007 try budget under the 120000x8/gen_len=100/cache_ratio=0.05, async-copy enabled, telemetry-disabled protocol:
  - `stream_low3_async`: `RETROINFER_STREAM_ONLY_LAYERS=2-3,28`, default capacity; correctness passed, block cache 5.267578 GiB, peak GPU 41416 MiB, decode 178.740 tok/s.
  - `stream_l28_low6_async`: `RETROINFER_STREAM_ONLY_LAYERS=28` plus validated low6 capacity on the other low6 layers; correctness passed, block cache 5.403809 GiB, peak GPU 41532 MiB, decode 188.942 tok/s.
  - `stream_l2_low6_async`: `RETROINFER_STREAM_ONLY_LAYERS=2` plus validated low6 capacity on the other low6 layers; correctness passed, block cache 5.403809 GiB, peak GPU 41532 MiB, decode 192.509 tok/s.
- No candidate is retained: all three improve memory versus the canonical baseline, but all regress decode throughput versus both the 198.836 tok/s baseline and the 201.180 tok/s validated low6+async reference. `OUTCOME.json` now requests replan rather than validation/report.
