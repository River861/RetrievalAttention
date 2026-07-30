# Infrastructure reuse plan for CPU full-KV rotating GPU slots

This environment-stage decision is for the exact inserted architecture: CPU memory owns the complete logical block-cache K/V for all layers, while GPU memory owns at most `m` reusable physical block-cache slots. No implementation, correctness probe, baseline, benchmark, or experiment was run in this stage.

## Reuse decisions

| Surface | Decision | Evidence |
| --- | --- | --- |
| Project `.venv` | Reuse for runtime/profile/correctness stages. | Fresh audit uses `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`; Python 3.11.15; `torch==2.5.1+cu124`; CUDA available on NVIDIA A100 80GB PCIe; imports pass for the project stack. |
| Installed RetroInfer kernels | Reuse installed extensions for the next Python-only slot-rotation preflight. | `retroinfer_kernels` exposes `ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, and `batch_gemm_softmax`. The gather/scatter kernels consume tensor arguments on the current CUDA stream, so fixed-address GPU slot tensors can be supplied without changing the installed ABI. |
| PyTorch CUDA primitives | Reuse for CPU-owner allocation and transfer orchestration. | PyTorch CUDA streams/events, pinned host tensors, and `non_blocking=True` copies are available in the project runtime; existing `cache_hub/retroinfer_cache.py` already uses those primitives for async metadata/list/copy paths. |
| Profilers and diagnostics | Reuse Nsight Systems; keep Compute Sanitizer as a later diagnostic. | `nsys` is installed at `/usr/local/bin/nsys`; `compute-sanitizer` is installed at `/usr/bin/compute-sanitizer`; `ncu` is absent and not required for H2D/D2H overlap timelines. |
| Existing harness | Reuse `throughput_eval` after environment stage closes. | `throughput_eval/reproduce_block_cache.py` already parses block-cache metadata, peak process GPU memory, decode/e2e throughput, logs, and summaries; extensions should be metric-field additions only. |

## Source rebuild status

Source rebuild is **blocked**, separate from runtime capability and separate from the slot-rotation idea status.

| Check | Result |
| --- | --- |
| `CUDA_HOME` | unset |
| `torch.utils.cpp_extension.CUDA_HOME` | `/usr` |
| `/usr/bin/nvcc` | CUDA 12.0 (`Build cuda_12.0.r12.0/compiler.32267302_0`) |
| `/usr/local/cuda-12.4/bin/nvcc` | absent |
| `/usr/local/cuda-12/bin/nvcc` | absent |
| `/usr/local/cuda-13.3/bin/nvcc` | CUDA 13.3 (`Build cuda_13.3.r13.3/compiler.38244171_0`) |
| Rebuild verdict | `blocked_for_source_rebuild`, `failure_class=environment_rebuild_toolchain_cuda_version_mismatch`, `idea_status=not_evaluated` |

Before any extension rebuild, prove a CUDA 12.4-compatible `CUDA_HOME`/`nvcc` and rebuild `library/retroinfer` inside `/home/v-xuchuanluo/RetroInfer/.venv`. Do not infer rebuild readiness from `torch==2.5.1+cu124`.

## Python-only implementation path

The next bounded implementation preflight may proceed without a source rebuild if it keeps the C++/CUDA extension ABI unchanged:

1. Allocate CPU full block-cache K/V owners in Python, preferring pinned tensors and classifying pinned allocation failure as an environment/resource blocker.
2. Allocate at most `m` fixed-address GPU slot tensors before CUDA graph capture.
3. Maintain Python-owned slot state `{slot_id, resident_layer, generation, state}` and explicit H2D/D2H streams/events.
4. Pass the resident slot tensors to the existing `gather_copy_and_concat` and `gather_copy_and_scatter` call sites.
5. Preserve `WaveBufferCPU` as the per-layer logical hit/admission/LRU manager; physical GPU slot selection must remain separate from logical block ids.
6. Extend `block_cache_metadata()` and the existing harness fields for the required telemetry.

If the implementation discovers that `WaveBufferCPU` logical ownership or `gather_copy_*` ABI must change, the path becomes blocked until CUDA 12.4 rebuild tooling is proven; do not silently fall back to metadata staging, stream-only gather, late allocation, or static capacity scaling.

## Host-memory and pinned-memory gate

For the canonical 120K, batch 8, cache ratio 5%, gen_len 100 shape, the scoped full CPU block-cache K/V owner is 6,241,124,352 bytes (5.8125 GiB). Live host facts are `MemTotal=226765912 kB`, `MemAvailable=220201200 kB`, and `RLIMIT_MEMLOCK=28345736 kB`, so the initial pinned-owner probe is capacity-plausible but not proven: existing pinned list/metadata buffers also consume memlock, and allocation failure remains an environment/resource blocker.

## Current gate

Environment artifacts are refreshed and select the reuse stack above. Downstream baseline, correctness, profiling, benchmark, and slot-rotation implementation work remain locked until the environment-stage review accepts these artifacts.
