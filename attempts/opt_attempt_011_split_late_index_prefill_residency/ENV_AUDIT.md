## repo
/home/v-xuchuanluo/RetroInfer
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
7096eab9190da389bbc75c1992140d1432d9d8ec

## setup files
./benchmark/reasoning/requirements.txt
./library/cutlass/pyproject.toml
./library/cutlass/setup.cfg
./library/retroinfer/setup.py
./requirements.txt

## python
/usr/bin/python
Python 3.12.3

## cuda tools
/usr/bin/nvidia-smi
NVIDIA A100 80GB PCIe, 580.173.02, 81920 MiB
/usr/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0

## python packages
torch: missing
triton: missing
transformers: missing
accelerate: missing
datasets: missing
numpy: missing
flash_attn: missing
tilelang: missing
## selected project runtime
drwx------ 6 v-xuchuanluo@microsoft.com users 4096 Jul 29 07:04 .venv
lrwxrwxrwx 1 v-xuchuanluo@microsoft.com users   10 Jul 29 07:03 .venv/bin/python -> python3.11
Python 3.11.15
/home/v-xuchuanluo/RetroInfer/.venv/lib/python3.11/site-packages/transformers/utils/hub.py:106: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
executable checked via .venv/bin/python
platform= Linux-6.17.0-1020-azure-x86_64-with-glibc2.39
torch: FOUND
  version=2.5.1+cu124
triton: FOUND
  version=3.1.0
transformers: FOUND
  version=4.49.0
vllm: FOUND
  version=0.6.5
flash_attn: FOUND
  version=2.7.3
flashinfer: FOUND
  version=0.2.4+cu124torch2.5
pybind11: FOUND
  version=2.12.0
retroinfer_kernels: FOUND
  version=unknown
weighted_flash_decoding: FOUND
  version=0.1
numpy: FOUND
  version=1.26.4
torch.cuda.is_available= True
torch.version.cuda= 12.4
torch.backends.cuda.is_built= True
torch.cuda.device_count= 1
cuda:0: name=NVIDIA A100 80GB PCIe total_memory=85093777408 capability=8.0
