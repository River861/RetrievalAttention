# opt017 环境与能力审计

- 运行时：`/home/v-xuchuanluo/RetroInfer/.venv/bin/python`，Python 3.11.15，`torch==2.5.1+cu124`，`torch.version.cuda==12.4`，GPU0 为 NVIDIA A100 80GB PCIe。
- 已安装扩展：`retroinfer_kernels`、`weighted_flash_decoding`、`flash_attn`、`flashinfer` 均可导入；`side_stream_cudagraph_replay=ok` 证明本机制需要的 side-stream CUDA graph replay + event wait 可执行。
- 隔离 PATH 修复：审计追加记录显示将 `.venv/bin` 放到 PATH 后可发现 `.venv/bin/ninja`。
- 源码重编译状态：仍阻塞，`CUDA_HOME` 未设，`torch.utils.cpp_extension.CUDA_HOME=/usr`，`/usr/bin/nvcc` 是 CUDA 12.0；因此本 attempt 只使用 Python/runtime 编排和已安装 cu124 扩展，不进行 CUDA/C++ rebuild。

原始审计：`ENV_AUDIT.raw.txt`。
