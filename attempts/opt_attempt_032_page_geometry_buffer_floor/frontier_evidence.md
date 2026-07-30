# opt032 frontier evidence

Fresh trigger: page-geometry / buffer-sizing mechanism pivot, searched 2026-07-30 before edits.

Evidence used for scope selection:

- Current KV-cache serving work continues to treat block/page geometry as a memory-throughput tradeoff: smaller blocks reduce residency or fragmentation but can increase metadata/page-walk and transfer overhead.
- Host offload, prefetch, and async H2D overlap remain the main streaming directions, but RetroInfer already has pinned host offload, side-stream migration, and `WaveBufferCPU`; the bounded next step was therefore to decouple existing WaveBuffer capacity from page=1 block-cache residency instead of adding a new offload framework.
- Representative sources surfaced by the search: NVIDIA CPU-GPU KV-cache offload and memory sharing (`https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/`), recent KV-cache engineering/page-attention guides (`https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide`, `https://appscale.blog/en/blog/kv-cache-engineering-llm-inference-paged-attention-prefill-decode-disaggregation-2026`), and a 2026 KV-cache optimization survey (`https://arxiv.org/abs/2603.20397`).

Decision: implement default-off `RETROINFER_BUFFER_PAGES_PER_CLUSTER_FLOOR` in the existing Python runtime path, benchmark only installed kernels through `throughput_eval/reproduce_block_cache.py`, and avoid CUDA/C++ rebuild or new harness.
