# opt019 environment audit

Project runtime `/home/v-xuchuanluo/RetroInfer/.venv/bin/python` imports torch 2.5.1+cu124, vllm 0.6.5, transformers 4.49.0, flash-attn 2.7.3, flashinfer 0.2.4+cu124torch2.5, triton 3.1.0, pybind11 2.12.0, numpy 1.26.4, `retroinfer_kernels`, and `weighted_flash_decoding` on NVIDIA A100 80GB.

Runtime is valid for installed cu124 extensions. Source rebuild capability remains blocked: default `/usr/bin/nvcc` is CUDA 12.0 and `/usr/local/cuda*` is CUDA 13.3, not a proven CUDA 12.4 compiler. No source rebuild was attempted.

Raw audit: `ENV_AUDIT.raw.txt`.
