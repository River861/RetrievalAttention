# opt033 frontier evidence

Fresh frontier trigger was run before edits or benchmark runs for page-geometry/cache-ratio compensation in long-context GPU KV/block-cache inference.

## Search record

- Query: `Recent techniques or implementation patterns for GPU KV/block cache memory reduction with page geometry, cache ratio compensation, paged attention, prefetching, or host-device streaming overlap in LLM inference as of 2026`
- Tool result: current web search summary with sources covering paged attention/page geometry, adaptive cache sizing, hybrid CPU/GPU KV cache, prefetching, and host-to-device overlap.
- Representative sources returned by the search: Digital Applied KV-cache engineering guide, gmicloud.ai PagedAttention/KV-cache management, Introl KV-cache optimization guide, Spheron KV-cache optimization guide, and arXiv `2603.20397` KV-cache optimization survey.

## Relevance to this bounded screen

The frontier evidence supports testing page geometry and cache residency as first-class memory/throughput knobs, but does not provide project-specific evidence that smaller page geometry plus a higher cache ratio recovers RetroInfer throughput. Therefore opt033 kept the existing project-native opt013 stack and opt031/opt032 default-off knobs, then directly measured cache-ratio compensation at the required 120000x8 shape.

