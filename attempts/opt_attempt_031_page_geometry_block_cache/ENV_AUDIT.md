# opt031 环境审计

- 运行环境使用项目 `.venv`：Python 3.11.15、`torch==2.5.1+cu124`、CUDA runtime 12.4，GPU 为 NVIDIA A100 80GB PCIe。
- `retroinfer_kernels` 已安装且可导入 `ThreadPool`、`WaveBufferCPU`、`gather_copy_and_concat`、`gather_copy_and_scatter`、`gather_copy_vectors`、`batch_gemm_softmax`。
- CUDA/C++ 源码重编译仍被环境阻塞：`CUDA_HOME` 未设置，默认 `/usr/bin/nvcc` 为 CUDA 12.0，另有 `/usr/local/cuda-13.3/bin/nvcc`，未证明 CUDA 12.4 编译器；本 attempt 未重编译。
- 原始审计日志：`attempts/opt_attempt_031_page_geometry_block_cache/ENV_AUDIT.raw.txt`。
