# opt016 environment audit

- Generated: `2026-07-29T22:43:25+00:00`
- Runtime: `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`, Python `3.11.15`
- GPU: NVIDIA A100 80GB PCIe, driver `580.173.02`, compute capability `8.0`
- Python/cu124 stack: `torch==2.5.1+cu124`, CUDA runtime `12.4`, `vllm==0.6.5`, `transformers==4.49.0`, `flash_attn==2.7.3`, `flashinfer==0.2.4+cu124torch2.5`, `triton==3.1.0`, `pybind11==2.12.0`, `numpy==1.26.4`, `weighted_flash_decoding==0.1`
- Installed RetroInfer kernels import successfully: `ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, `batch_gemm_softmax`
- opt016 mechanism capability: a fresh A100 probe captured a tiny `torch.cuda.CUDAGraph`, replayed it on a side stream, and ordered the default stream with `wait_event`; result `side_stream_cudagraph_replay=ok`
- Source rebuild remains blocked and was not attempted: `torch.utils.cpp_extension.CUDA_HOME=/usr`, `/usr/bin/nvcc` is CUDA `12.0`, and `/usr/local/cuda-13.3/bin/nvcc` is CUDA `13.3`, so no CUDA 12.4-compatible rebuild compiler is proven

Raw audit: `ENV_AUDIT.raw.txt`.
