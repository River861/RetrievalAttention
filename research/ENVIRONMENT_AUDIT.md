# Kernel Environment Audit

- Generated: `2026-07-30T07:19:53.205442+00:00`
- Project: `/home/v-xuchuanluo/RetroInfer`
- Target Python: `Python 3.11.15` (`/home/v-xuchuanluo/RetroInfer/.venv/bin/python`)
- Ready: **YES**
- Requested capabilities: `torch, cuda_cpp, profiling`

## GPU

| index | name | compute capability | memory MiB | driver |
|---:|---|---|---:|---|
| 0 | NVIDIA A100 80GB PCIe | 8.0 | 81920 | 580.173.02 |

## Capability gate

| capability | ready | missing |
|---|---|---|
| `torch` | yes |  |
| `triton` | yes |  |
| `tilelang` | no | tilelang |
| `cuda_cpp` | yes |  |
| `cutlass_cute` | no | CUTLASS/CuTe source, package, or CUTLASS_PATH |
| `profiling` | yes |  |
| `sanitizer` | yes |  |

## Blocking findings

- None.

## Warnings

- Unrelated installed distribution is inconsistent: No broken requirements found.

## Dependency closure

- `pip check`: **clean**
- No broken requirements found.

## Specialized ecosystem detected

Catalog entries: `90`; detected in this environment/project: `15`.

| id | version | detected by |
|---|---|---|
| `cuda_toolkit` |  | executables=nvcc,ptxas,cuobjdump,nvdisasm |
| `pytorch_custom_ops` | 2.5.1 | imports=torch |
| `torch_inductor` | 2.5.1 | imports=torch |
| `triton` | 3.1.0 | imports=triton |
| `triton_gluon` | 3.1.0 | imports=triton.language.extra.libdevice |
| `tilelang` |  | executables=nvcc |
| `flash_attention` | 2.7.3 | imports=flash_attn |
| `flashinfer` | 0.2.4+cu124torch2.5 | imports=flashinfer |
| `xformers` | 0.0.28.post3 | imports=xformers |
| `vllm` | 0.6.5 | imports=vllm |
| `nsight_systems` |  | executables=nsys |
| `compute_sanitizer` |  | executables=compute-sanitizer |
| `google_benchmark` |  | source=benchmark |
| `cmake_ninja` |  | executables=ninja |
| `pybind11` | 2.12.0 | imports=pybind11 |

## Environment-first rule

Do not start kernel implementation until the selected capability is green. Use repository extras, lockfiles, containers, benchmark runners, and mature vendor/framework primitives before writing replacement infrastructure. A missing compiler, package, profiler, or architecture flag is an environment failure—not evidence that the kernel mechanism is wrong.

## RetroInfer slot-rotation environment decision

- Audit command: `PYTHONPATH=/home/v-xuchuanluo/Argus PATH=/home/v-xuchuanluo/RetroInfer/.venv/bin:$PATH /home/v-xuchuanluo/RetroInfer/.venv/bin/python -m argus_skill.verticals.kernel_engineering.environment_audit collect --project-root . --target-python /home/v-xuchuanluo/RetroInfer/.venv/bin/python --report research/ENVIRONMENT_AUDIT.json --markdown research/ENVIRONMENT_AUDIT.md --require torch --require cuda_cpp --require profiling`
- Runtime is proved in the project `.venv`: `torch==2.5.1+cu124`, CUDA available on NVIDIA A100 80GB PCIe, and imports pass for `retroinfer_kernels` plus the project CUDA attention stack.
- Installed `retroinfer_kernels` is selected for the next Python-only preflight: `ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, and `batch_gemm_softmax` import successfully.
- Source rebuild is blocked, not idea-failed: `CUDA_HOME` is unset, `torch.utils.cpp_extension.CUDA_HOME` resolves to `/usr`, `/usr/bin/nvcc` is CUDA 12.0, `/usr/local/cuda-12.4/bin/nvcc` and `/usr/local/cuda-12/bin/nvcc` are absent, and `/usr/local/cuda-13.3/bin/nvcc` is CUDA 13.3.
- No source rebuild, correctness probe, baseline reproduction, benchmark, or optimization was run in the environment stage.
