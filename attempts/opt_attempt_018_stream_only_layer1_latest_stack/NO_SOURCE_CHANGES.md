# opt018 no source changes

This attempt introduced no source-code changes. It reused the existing `RETROINFER_STREAM_ONLY_LAYERS` fused stream-only gather support in `cache_hub/retroinfer_cache.py` and the existing opt013 late-allocation stack. The candidate behavior was selected entirely by environment variables:

- `RETROINFER_STREAM_ONLY_LAYERS=1`
- `RETROINFER_LAYER_CACHE_CAPACITY_SCALE=2-3=0.75,25=0.75,28-29=0.75`
- opt013 late allocation/uninitialized/pinned-metadata/async-cluster-id/low5 remaining-capacity settings

No CUDA/C++ rebuild, new harness, or Figure 13 path was used.
