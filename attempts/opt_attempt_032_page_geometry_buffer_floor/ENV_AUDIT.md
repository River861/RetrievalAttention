# opt032 environment audit

- Fresh audit before edits used `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`: Python 3.11.15, `torch==2.5.1+cu124`, CUDA available, GPU0 `NVIDIA A100 80GB PCIe`, capability `(8, 0)`.
- Project runtime imports succeeded for `vllm==0.6.5`, `transformers==4.49.0`, `flash_attn==2.7.3`, `flashinfer==0.2.4+cu124torch2.5`, `weighted_flash_decoding==0.1`, `triton==3.1.0`, `pybind11==2.12.0`, and installed `retroinfer_kernels` symbols (`ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, `batch_gemm_softmax`).
- A pinned host-to-device smoke copy to CUDA completed in the venv runtime, so the installed cu124 runtime path is usable for the Python-only opt032 change.
- Source rebuild remains blocked and was not attempted: `CUDA_HOME` is unset, `torch.utils.cpp_extension.CUDA_HOME=/usr`, `/usr/bin/nvcc` is CUDA 12.0, and `/usr/local/cuda-13.3/bin/nvcc` is CUDA 13.3, not a proven CUDA 12.4 rebuild compiler.
- Raw audit: `attempts/opt_attempt_032_page_geometry_buffer_floor/ENV_AUDIT.raw.txt`.
