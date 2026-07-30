# Kernel Environment Audit

- Generated: `2026-07-30T04:27:51.623818+00:00`
- Project: `/home/v-xuchuanluo/RetroInfer`
- Target Python: `Python 3.11.15` (`.venv/bin/python`)
- Ready: **YES**
- Requested capabilities: `torch, triton, profiling`

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
| `cuda_cpp` | no | ninja or cmake |
| `cutlass_cute` | no | cuda_cpp, CUTLASS/CuTe source, package, or CUTLASS_PATH |
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

Catalog entries: `90`; detected in this environment/project: `14`.

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
| `pybind11` | 2.12.0 | imports=pybind11 |

## Environment-first rule

Do not start kernel implementation until the selected capability is green. Use repository extras, lockfiles, containers, benchmark runners, and mature vendor/framework primitives before writing replacement infrastructure. A missing compiler, package, profiler, or architecture flag is an environment failure—not evidence that the kernel mechanism is wrong.
