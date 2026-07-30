# opt034 environment audit

- Project root: `/home/v-xuchuanluo/RetroInfer`
- Git HEAD: `7096eab9190da389bbc75c1992140d1432d9d8ec`
- Python: `.venv/bin/python`, Python 3.11.15
- Runtime packages: torch `2.5.1+cu124`, Triton `3.1.0`, transformers `4.49.0`, numpy `1.26.4`, vLLM `0.6.5`, flash-attn `2.7.3`, flashinfer `0.2.4+cu124torch2.5`, weighted_flash_decoding `0.1`, pybind11 `2.12.0`; `cache_hub.retroinfer_cache` and required `retroinfer_kernels` symbols import successfully.
- GPU: one `NVIDIA A100 80GB PCIe`, compute capability 8.0, 85093777408 bytes reported by PyTorch. Corrected `nvidia-smi` probe showed 0 MiB used and no running processes before measurements.
- Driver/runtime visibility: `nvidia-smi` driver `580.173.02`, CUDA driver display `13.0`; PyTorch CUDA runtime `12.4`.
- Rebuild status: source rebuild not attempted because CUDA/C++ rebuild is a non-goal. Fresh probe still has `CUDA_HOME` unset, `torch.utils.cpp_extension.CUDA_HOME == /usr`, `/usr/bin/nvcc` CUDA 12.0, and `/usr/local/cuda-13.3/bin/nvcc` CUDA 13.3, so no CUDA 12.4-compatible rebuild compiler was proven.

Raw audit log: `attempts/opt_attempt_034_capacity_reallocation/logs/env_audit.log`.
