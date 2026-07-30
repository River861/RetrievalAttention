# opt012 environment audit

- Project runtime: `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`, Python 3.11.15.
- CUDA runtime: `torch==2.5.1+cu124`, `torch.cuda.is_available=True`, `torch.version.cuda=12.4`.
- GPU: NVIDIA A100 80GB PCIe, compute capability 8.0, driver 580.173.02.
- Required packages imported in the project runtime: `triton==3.1.0`, `transformers==4.49.0`, `vllm==0.6.5`, `flash_attn==2.7.3`, `flashinfer==0.2.4+cu124torch2.5`, `pybind11==2.12.0`, `retroinfer_kernels`, `weighted_flash_decoding==0.1`, `numpy==1.26.4`.
- Attempt-local capability check: a small pinned CPU tensor reported `is_pinned=True` and copied to GPU on a side CUDA stream with `non_blocking=True`; event synchronization completed.
- Source rebuild status: blocked/non-goal. `/usr/bin/nvcc` is CUDA 12.0, not a proven CUDA 12.4 compiler for the cu124 runtime; opt012 used installed project extensions and did not rebuild CUDA/C++ code.

Raw command output: `ENV_AUDIT.raw.txt`.
