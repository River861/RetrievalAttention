# opt013 environment audit

- Project runtime: `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`, Python 3.11.15.
- CUDA runtime: `torch==2.5.1+cu124`, `torch.cuda.is_available=True`, `torch.version.cuda=12.4`.
- GPU: NVIDIA A100 80GB PCIe, compute capability 8.0, driver 580.173.02.
- Required runtime packages imported in the project runtime: `triton==3.1.0`, `transformers==4.49.0`, `vllm==0.6.5`, `flash_attn==2.7.3`, `flashinfer==0.2.4+cu124torch2.5`, `pybind11==2.12.0`, `retroinfer_kernels`, `weighted_flash_decoding==0.1`, `numpy==1.26.4`.
- Installed RetroInfer kernel symbols imported: `ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, `batch_gemm_softmax`.
- Attempt-local CUDA copy probe passed: pinned CPU tensor copied to GPU on a side CUDA stream with `non_blocking=True`; synchronized sum was 8.0.
- Source rebuild status: blocked/non-goal. `CUDA_HOME` is unset, `torch.utils.cpp_extension.CUDA_HOME` resolves to `/usr`, `/usr/bin/nvcc` is CUDA 12.0, and `/usr/local/cuda-13.3/bin/nvcc` is CUDA 13.3; no CUDA 12.4-compatible rebuild compiler was proven.

Raw command output: `ENV_AUDIT.raw.txt`.
