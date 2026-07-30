# opt034 frontier evidence

Fresh optimize frontier trigger was run before benchmark execution for per-layer block-cache/KV-cache residency and H2D/CPU-offload mechanisms in long-context inference.

## Search record

- Query: `Recent 2025 2026 implementation techniques for reducing GPU KV cache or block cache memory in long-context LLM inference using per-layer cache allocation, cache residency, CPU GPU offload, prefetching, or asynchronous host to device transfer overlap`
- Tool result: current web search summary with sources covering PagedAttention/page allocation, per-layer or selective cache allocation, KV quantization, hybrid CPU/GPU offload, and asynchronous prefetch/H2D overlap.
- Representative sources returned: Digital Applied KV-cache engineering guide (2026), Introl production KV-cache guide, NVIDIA CPU-GPU KV-cache offload/Dynamo posts, Spheron KV-cache/offload guides, arXiv 2508.06297 KV-cache compression review, and arXiv 2603.20397 KV-cache optimization survey.

## Relevance to this bounded screen

The search supports per-layer cache allocation/residency as a current production-relevant memory knob. opt034 therefore did not create a new harness or vendor implementation; it reused RetroInfer's existing `RETROINFER_LAYER_CACHE_CAPACITY_SCALE` mechanism and the opt013 late/uninitialized block-cache stack to test two telemetry-guided layer-capacity schedules at the canonical 120000x8/gen_len=100 shape.
