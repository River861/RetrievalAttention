# Project-native setup and capability audit

## Native setup from repository files

`README.md` defines the intended runtime:

- CUDA 12.4, with `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` as the fallback container image.
- Python 3.10.16 in a conda environment.
- Conda packages: `mkl`, `libstdcxx-ng`.
- Python dependencies from `requirements.txt`: `torch==2.5.1`, `vllm==0.6.5`, `transformers==4.49.0`, `pybind11==2.12.0`, plus benchmark/evaluation packages.
- Extra packages from README: `flash-attn==2.7.3 --no-build-isolation`, `flashinfer-python==0.2.4` from the cu124/torch2.5 index, and `git+https://github.com/Starmys/flash-attention.git@weighted`.
- Native kernel install: `cd library/`, clone NVIDIA CUTLASS into `library/cutlass`, then `cd retroinfer && pip install .`.

`library/retroinfer/setup.py` builds three extensions:

| Extension | Sources | Notes |
| --- | --- | --- |
| `retroinfer_kernels.WaveBuffer` | `retroinfer_kernels/src/wave_buffer_cpu.cpp` | C++/OpenMP CPU-side wave-buffer and LRU block-cache manager. |
| `retroinfer_kernels.Copy` | `retroinfer_kernels/src/gather_copy.cu` | CUDA gather/copy/scatter path for execution buffer and GPU block-cache admission. |
| `retroinfer_kernels.gemm_softmax` | `retroinfer_kernels/src/batch_gemm_softmax.cu` | CUDA/CUTLASS centroid scoring kernel. |

The setup script selects `CUDA_HOME` from the environment, then `torch.utils.cpp_extension.CUDA_HOME`, then `/usr/local/cuda-12`; it expects C++17 for CUDA extensions and links `cuda`/`cudart`.

## Fresh scope-stage audit for CPU full-KV slot rotation

Executed from `/home/v-xuchuanluo/RetroInfer` with the project `.venv`, not the default system Python:

```bash
./.venv/bin/python - <<'PY'
import importlib, json, os, shutil, subprocess, sys
facts = {"executable": sys.executable, "python": sys.version.split()[0], "CUDA_HOME_env": os.environ.get("CUDA_HOME")}
mods = ["torch","vllm","transformers","flash_attn","flashinfer","weighted_flash_decoding","triton","pybind11","retroinfer_kernels"]
for name in mods:
    mod = importlib.import_module(name)
    facts[name] = getattr(mod, "__version__", "import-ok")
import torch, torch.utils.cpp_extension as ce
facts["torch_cuda"] = torch.version.cuda
facts["torch_cuda_available"] = torch.cuda.is_available()
facts["torch_cpp_extension_CUDA_HOME"] = ce.CUDA_HOME
facts["gpu0"] = torch.cuda.get_device_name(0)
print(json.dumps(facts, indent=2, sort_keys=True))
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Observed runtime/package facts:

| Item | Observation |
| --- | --- |
| Project runtime | `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`, Python 3.11.15 |
| `torch` | `2.5.1+cu124`; `torch.version.cuda == 12.4`; CUDA available |
| `vllm` | `0.6.5` |
| `transformers` | `4.49.0` |
| `flash_attn` | `2.7.3` |
| `flashinfer` | `0.2.4+cu124torch2.5` |
| `weighted_flash_decoding` | `0.1` |
| `triton` | `3.1.0` |
| `pybind11` | `2.12.0` |
| `retroinfer_kernels` | imports from `/home/v-xuchuanluo/RetroInfer/library/retroinfer/retroinfer_kernels/__init__.py` |
| GPU | NVIDIA A100 80GB PCIe, 81920 MiB, driver 580.173.02, compute capability 8.0 |

The existing `.venv` is the runtime to reuse for package/runtime facts. It is not an exact README environment because README asks for Python 3.10.16 while the audited `.venv` uses Python 3.11.15, but it imports the pinned RetroInfer stack and installed kernels successfully.

## Host and pinned-memory resource facts

The requested CPU full block-cache K/V owner should prefer pinned memory. Current host capacity probe:

```bash
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
ulimit -l
```

Observed:

| Item | Observation |
| --- | ---: |
| `MemTotal` | 226,765,912 kB |
| `MemAvailable` | 220,147,364 kB |
| `ulimit -l` | 28,345,736 kB |

For the canonical 120K x batch 8 x cache ratio 5% shape, the full logical block-cache K/V body is 5.8125 GiB. That is below the current memlock limit before counting existing pinned `list_keys/list_values`, index metadata, cluster-id buffers, and transfer staging. If future implementation cannot allocate the CPU owner as pinned memory, classify it as `execution_status=blocked` or `failure_class=host_pinned_memory_resource_limit`, not as a failed slot-rotation idea.

## Profiler and toolchain availability

Observed toolchain/profiler facts:

| Tool | Observation |
| --- | --- |
| `nsys` | `/usr/local/bin/nsys`, NVIDIA Nsight Systems 2026.1.3 |
| `ncu` | not found on `PATH` in the fresh probe |
| default `nvcc` | `/usr/bin/nvcc`, CUDA 12.0 (`V12.0.140`) |
| additional `nvcc` | `/usr/local/cuda-13.3/bin/nvcc`, CUDA 13.3 (`V13.3.73`) |
| `CUDA_HOME` | unset |
| `torch.utils.cpp_extension.CUDA_HOME` | `/usr` |

This proves runtime reuse and Nsight Systems availability, but it does **not** prove source rebuild compatibility for the README/PyTorch CUDA 12.4 stack. The CUDA/C++ rebuild blocker remains: no CUDA 12.4-compatible `CUDA_HOME`/`nvcc` path has been proven. Rebuild failures must be classified as `environment_rebuild_toolchain_cuda_version_mismatch`, separate from CPU/GPU slot-rotation idea status.

## Official package/toolchain frontier check

The fresh frontier search checked official PyTorch previous-version installation docs and CUDA documentation. PyTorch's official matrix still lists `torch==2.5.1` CUDA 12.4 wheels/conda packages, matching the project `.venv` runtime. Newer PyTorch releases list newer CUDA wheel families, but this scope does not authorize replacing the project stack. CUDA documentation confirms that asynchronous host/device copies need pinned/page-locked host buffers for real overlap and that CUDA graph workflows require stable captured resources and valid stream/event dependencies.

## Existing benchmark and measurement setup

- Correctness/metrics entry point: `throughput_eval/test.py`.
- Block-cache matrix runner: `throughput_eval/reproduce_block_cache.py`.
- Existing final cache-ratio bundle: `experiments/cache_ratio/`.
- Existing reusable 120K x batch 8 rows are in `experiments/cache_ratio/results.csv`; these are not to be overwritten.
- Existing rerun script uses `experiments/cache_ratio/run.sh`, defaulting to `$ROOT/.venv/bin/python` if present, otherwise `python`.

## Required proof before implementation/build stages

Keep Argus verifier tooling separate from the project runtime. Use project `.venv/bin/python` for RetroInfer runtime and package facts. If a check needs Argus modules, run it from the project root with the Argus path/environment, not by installing Argus into `.venv`.

Before any source rebuild:

```bash
cd /home/v-xuchuanluo/RetroInfer/library/retroinfer
<verified-cuda-12.4-root>/bin/nvcc --version
CUDA_HOME=<verified-cuda-12.4-root> /home/v-xuchuanluo/RetroInfer/.venv/bin/python -m pip install -v .
```

Before any performance claim:

```bash
cd /home/v-xuchuanluo/RetroInfer
CUDA_VISIBLE_DEVICES=0 .venv/bin/python throughput_eval/reproduce_block_cache.py --help >/tmp/retroinfer_reproduce_block_cache_help.txt
```

Only after runtime import, correctness, and mechanism telemetry pass should baseline reproduction, profiling, or optimize experiments begin.
