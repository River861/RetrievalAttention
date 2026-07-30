# opt031 frontier evidence

Fresh trigger: mechanism pivot to page-geometry / streaming block-cache residency, searched 2026-07-30 before edits.

Key evidence used for scope selection:

- Current KV-cache memory work still centers on paged KV/block management; page/block grain size is a first-order memory/metadata/throughput tradeoff.
- Host offload and async H2D/prefetch are the dominant streaming extensions, but RetroInfer already has installed pinned-host/offload, side-stream copy, and `WaveBufferCPU` paths, so the bounded low-risk pivot was to expose the existing native page geometry rather than add a new offload framework.
- Representative sources surfaced by the search: NVIDIA CPU-GPU memory sharing for KV cache offload (`https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/`), LMCache offload/orchestration (`https://arxiv.org/html/2510.09665v2`), and recent paged-attention/KV-cache engineering surveys (`https://appscale.blog/en/blog/kv-cache-engineering-llm-inference-paged-attention-prefill-decode-disaggregation-2026`, `https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide`).

Decision: implement a default-off `RETROINFER_PAGES_PER_CLUSTER_OVERRIDE` in the existing Python config/runtime path and benchmark only installed kernels through `throughput_eval/reproduce_block_cache.py`; no CUDA/C++ rebuild or new harness.
