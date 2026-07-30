# opt035 CPU full-KV rotating GPU slots

Implemented a default-off `RETROINFER_BLOCK_CACHE_SLOT_ROTATION=1` path with `RETROINFER_BLOCK_CACHE_GPU_SLOTS=8`. The enabled path allocates pinned CPU full logical block-cache K/V for all layers, only 8 physical GPU K/V slots, fixed layer-to-slot aliases for CUDA graph capture, separate H2D/D2H streams/events, generation ownership checks, and telemetry for slot state, transfer counts/bytes, pending transfers, and violation counters.

The bounded 30K probe and canonical 120K x batch 8 x cache_ratio 0.05 x gen_len 100 run both passed the NIAH groundtruth oracle. Canonical telemetry proved `path_executed=true`, CPU full K/V bytes `6241124352`, actual GPU slot count `8 <= m`, GPU block-cache residency `1488.0 MiB`, H2D `3183` copies / `620796837888` bytes, D2H `3168` copies / `617871310848` bytes, zero pending transfers at metadata, and zero ownership violations.

The result is a negative performance tradeoff rather than a failed implementation: peak process GPU memory dropped to `28010 MiB` versus baseline `41960 MiB` and opt013 `29884 MiB`, but decode throughput fell to `25.04 tokens/s` versus roughly `198 tokens/s`. The full-layer PCIe rotation traffic is the clear bottleneck at this shape, so m=12 was not run.
