# Environment audit: opt028

- Runtime decision: reused project-native .venv on A100.
- Python: `/home/v-xuchuanluo/RetroInfer/.venv/bin/python` (3.11.15); torch `2.5.1+cu124` CUDA `12.4`; GPU0 `NVIDIA A100 80GB PCIe`.
- RetroInfer kernels: ThreadPool, WaveBufferCPU, gather_copy_and_concat, gather_copy_and_scatter, gather_copy_vectors, batch_gemm_softmax.
- Rebuild status: source rebuild remains blocked unless a CUDA 12.4 nvcc/CUDA_HOME is proven; this no-source mission will use installed cu124 extensions.
