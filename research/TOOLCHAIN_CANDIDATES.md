# Toolchain candidates for CPU full-KV rotating GPU slots

Environment-stage refresh: `research/ENVIRONMENT_AUDIT.json` generated at `2026-07-30T07:19:53.205442+00:00` by the RetroInfer project runtime (`/home/v-xuchuanluo/RetroInfer/.venv/bin/python`) with Argus on `PYTHONPATH`. Argus registry state is `refreshed_at=2026-07-18`, `entry_count=90`, `available_count=15` in this project/A100 environment.

## Selected reuse stack

| Surface | Current evidence | Decision |
| --- | --- | --- |
| RetroInfer `.venv` | Python 3.11.15; `torch==2.5.1+cu124`; CUDA available on NVIDIA A100 80GB PCIe; imports pass for `vllm`, `transformers`, `flash_attn`, `flashinfer`, `weighted_flash_decoding`, `triton`, `pybind11`, and `retroinfer_kernels`. | Reuse as the only project runtime for environment, correctness, profiling, and harness work. |
| Installed `retroinfer_kernels` | `ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, and `batch_gemm_softmax` import successfully. `gather_copy_*` consumes tensor arguments on the current CUDA stream. | Reuse for the first slot-rotation preflight; do not rebuild extensions unless a later ABI/logical-ownership need is proven. |
| PyTorch CUDA primitives | PyTorch CUDA 12.4 runtime is active; existing code already uses pinned tensors, `torch.cuda.Stream`, CUDA events, and `non_blocking=True` copies. | Reuse for pinned CPU full-KV owners, H2D/D2H streams/events, generation waits, and telemetry. |
| Nsight Systems | `nsys` at `/usr/local/bin/nsys`, version `2026.1.3.425-261338342291v0`. | Selected profiler for later copy/compute overlap timelines. |
| Existing `throughput_eval` harness | `throughput_eval/reproduce_block_cache.py` already records block-cache bytes, peak process GPU memory, decode/e2e throughput, logs, and summaries. | Reuse with metric-field extensions only after environment stage closes. |

## Current specialist/frontier candidates

| Need | Registry/frontier candidates | Local status | Decision for this objective |
| --- | --- | --- | --- |
| CPU/GPU KV offload and serving cache managers | Registry: `vllm`, `flashinfer`; frontier references: vLLM KV offloading connector/usage guide, LMCache, TensorRT-LLM, SGLang, NVIDIA Dynamo/KVBM-style KV managers. | `vllm==0.6.5` and `flashinfer==0.2.4+cu124torch2.5` import locally; LMCache/TensorRT-LLM/SGLang/Dynamo are not project dependencies. | Use as design comparison only. Do not replace RetroInfer, its API, or the harness with an external serving stack. |
| Pinned-memory async H2D/D2H copies | PyTorch pinned-memory/non-blocking copy guidance and CUDA asynchronous concurrent execution docs. | Project runtime supports PyTorch CUDA; host memory is `MemTotal=226765912 kB`, `MemAvailable=220201200 kB`, `RLIMIT_MEMLOCK=28345736 kB`. | Reuse PyTorch pinned tensors plus explicit H2D/D2H streams/events. Treat pinned-allocation failure as an execution/environment blocker. |
| CUDA graph fixed-address behavior | PyTorch/NVIDIA CUDA graph memory guidance: captured resources must stay alive and fixed-address for replay. | Existing `sparse_attention_with_cudagraph()` captures per-layer gather/attention/update graphs that reference `self.cache_keys[layer_idx]` and `self.cache_values[layer_idx]`. | Slot rotation must allocate all GPU slots before capture and map layer graphs to deterministic fixed slot addresses, or disable/report graph incompatibility for the opt-in path. |
| Profiling overlap and diagnostics | Registry: `nsight_systems`, `compute_sanitizer`; CUDA timeline profiling docs. | `nsys` and `compute-sanitizer` installed; `ncu` absent. | Use Nsight Systems for H2D/D2H/compute overlap evidence; use Compute Sanitizer only if CUDA-extension changes become possible. |

## Rejected or deferred

| Candidate | Reason |
| --- | --- |
| TensorRT-LLM, SGLang, vLLM engine swap, LMCache integration | Engine/cache-stack replacements would violate the RetroInfer API and existing-harness contract for this bounded mission. |
| TileLang/Triton/CUTLASS/CuTe rewrite | The target mechanism is explicit block-cache K/V residency and transfer ownership, not a new attention/kernel backend. |
| Source rebuild of `library/retroinfer` | Blocked: `CUDA_HOME` is unset, `torch.utils.cpp_extension.CUDA_HOME` resolves to `/usr`, `/usr/bin/nvcc` is CUDA 12.0, `/usr/local/cuda-13.3/bin/nvcc` is CUDA 13.3, and no CUDA 12.4-compatible `nvcc` is proven. |
| Nsight Compute (`ncu`) | Not installed and not required for the current overlap/timeline need covered by Nsight Systems. |

Frontier references checked for this refresh: vLLM KV Offloading Usage Guide (`https://docs.vllm.ai/en/latest/features/kv_offloading_usage/`), LMCache repository (`https://github.com/LMCache/LMCache`), PyTorch pinned-memory/non-blocking tutorial (`https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html`), PyTorch CUDA graph memory notes (`https://pytorch.org/docs/stable/notes/cuda.html#memory-issues`), and CUDA asynchronous execution docs (`https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution`).
