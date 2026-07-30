# RetroInfer opt018 environment audit
2026-07-29T23:33:35Z

## Repository
cwd=/home/v-xuchuanluo/RetroInfer
git_head=7096eab9190da389bbc75c1992140d1432d9d8ec
 M cache_hub/retroinfer_cache.py
 M model_hub/llama.py
 M research/PIPELINE_STATE.json
 M throughput_eval/reproduce_block_cache.py
?? RESULTS.md
?? attempts/
?? research/BASELINE_PROTOCOL.md
?? research/BASELINE_RESULT.json
?? research/ENVIRONMENT_AUDIT.json
?? research/ENVIRONMENT_AUDIT.md
?? research/FRONTIER_WATCH.jsonl
?? research/GROUND_TRUTH.md
?? research/INFRASTRUCTURE_REUSE_PLAN.md
?? research/KERNEL_SCOPE.md
?? research/PROJECT_NATIVE_SETUP.md
?? research/TOOLCHAIN_CANDIDATES.md
?? research/VALIDATION_MATRIX.md
?? research/VALIDATION_RESULT.json
?? research/baseline_block_cache_single_a100_120k_b8_cr0p05/
?? research/frontier/
?? research/validation_fractional_layer_capacity_120k_b8_cr0p05/
?? research/validation_layer_cache_residency_120k_b8_cr0p05/
?? research/validation_low6_async_buffer_mult_3p0_120k_b8_cr0p05/
?? research/validation_low6_async_cluster_id_copy_120k_b8_cr0p05/
?? research/validation_low6_multilayer_fractional_capacity_120k_b8_cr0p05/

## GPU
NVIDIA A100 80GB PCIe, 81920 MiB, 580.173.02

## Python/runtime
{
  "CUDA_HOME_env": null,
  "capability0": "8.0",
  "cuda_available": true,
  "flash_attn": "2.7.3",
  "flashinfer": "0.2.4+cu124torch2.5",
  "gpu0": "NVIDIA A100 80GB PCIe",
  "ninja_path_with_venv_path": "/home/v-xuchuanluo/RetroInfer/.venv/bin/ninja",
  "ninja_version_with_venv_path": [
    "1.11.1.git.kitware.jobserver-1"
  ],
  "nsys_path_with_venv_path": "/usr/local/bin/nsys",
  "nsys_version_with_venv_path": [
    "NVIDIA Nsight Systems version 2026.1.3.425-261338342291v0"
  ],
  "nvcc_path_with_venv_path": "/usr/bin/nvcc",
  "nvcc_version_with_venv_path": [
    "nvcc: NVIDIA (R) Cuda compiler driver",
    "Copyright (c) 2005-2023 NVIDIA Corporation",
    "Built on Fri_Jan__6_16:45:21_PST_2023",
    "Cuda compilation tools, release 12.0, V12.0.140",
    "Build cuda_12.0.r12.0/compiler.32267302_0"
  ],
  "platform": "Linux-6.17.0-1020-azure-x86_64-with-glibc2.39",
  "ptxas_path_with_venv_path": "/usr/bin/ptxas",
  "ptxas_version_with_venv_path": [
    "ptxas: NVIDIA (R) Ptx optimizing assembler",
    "Copyright (c) 2005-2023 NVIDIA Corporation",
    "Built on Fri_Jan__6_16:43:29_PST_2023",
    "Cuda compilation tools, release 12.0, V12.0.140",
    "Build cuda_12.0.r12.0/compiler.32267302_0"
  ],
  "pybind11": "2.12.0",
  "python_executable": "/home/v-xuchuanluo/RetroInfer/.venv/bin/python",
  "python_version": "3.11.15",
  "retroinfer_kernels": "import_ok",
  "side_stream_cudagraph_replay": "ok",
  "torch": "2.5.1+cu124",
  "torch_cpp_extension_CUDA_HOME": "/usr",
  "torch_cuda": "12.4",
  "transformers": "4.49.0",
  "triton": "3.1.0",
  "vllm": "0.6.5",
  "weighted_flash_decoding": "0.1"
}
