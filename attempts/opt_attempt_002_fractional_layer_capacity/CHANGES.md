# opt_attempt_002_fractional_layer_capacity

## 变更

- 在 `cache_hub/retroinfer_cache.py` 增加默认关闭的 `RETROINFER_LAYER_CACHE_CAPACITY_SCALE` 运行时控制；默认未设置时逐层容量仍为原 `cache_ratio=0.05` 计算值，API 不变。
- 新控制接受 `1=0.25`、`layers:1=25%`、`2-4=0.5` 这类显式层选择，按 `pages_per_cluster` 对齐缩放每层 `WaveBufferCPU` capacity、GPU `cache_keys/cache_values` 张量大小和 gather/scatter stride。无效层、重复层、非数值、`<=0` 或 `>1` 的 scale 会直接抛出 `ValueError`。
- `block_cache_metadata()` / `RETROINFER_RESULT_JSON` 现在输出 `block_cache_capacity_scale_spec`、`block_cache_layer_capacity_scales`，并保留逐层 `block_cache_layer_pages` / `block_cache_layer_bytes`，用于复核候选真实显存容量。
- `throughput_eval/reproduce_block_cache.py` 的 block-cache 字段表同步保留这些逐层 metadata；未新建 benchmark harness，仍复用项目原 harness。

## 环境与边界

- 使用项目 `.venv`、已安装 `retroinfer_kernels`、`throughput_eval/reproduce_block_cache.py` 和单张 `NVIDIA A100 80GB PCIe`。
- PyTorch runtime 为 CUDA 12.4；`CUDA_HOME` 为空且 `nvcc` 为 CUDA 12.0.140，不满足 CUDA 12.4 source rebuild 条件，因此本次没有尝试 CUDA/C++ rebuild。
- 基线来自 `research/BASELINE_RESULT.json`：block cache `5.812500 GiB`，进程峰值 `41960 MiB`，decode `198.836414 tok/s`，正确性通过。
- prior 对照来自 `attempts/opt_attempt_001_layer_cache_residency/`：关闭 layer 1 的 block cache 为 `5.630859 GiB` / `41772 MiB` / `194.096747 tok/s`。

## 120000x8 / cache_ratio=0.05 结果

| candidate | env spec | correctness | layer1 pages | layer1 scale | block cache GiB | peak GPU MiB | decode tok/s | decode vs baseline | classification |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | default | pass | 744 | 1.00 | 5.812500 | 41960 | 198.836414 | +0.000% | canonical reference |
| layer1_scale0p25 | 1=0.25 | pass | 186 | 0.25 | 5.676270 | 41820 | 195.737446 | -1.559% | pareto_tradeoff_correctness_passed |
| layer1_scale0p50 | 1=0.50 | pass | 372 | 0.50 | 5.721680 | 41868 | 176.529888 | -11.219% | non_competitive_dominated_by_layer1_scale0p25 |
| layer1_scale0p75 | 1=0.75 | pass | 558 | 0.75 | 5.767090 | 41912 | 201.359835 | +1.269% | pareto_tradeoff_correctness_passed_recommended |
| prior_layer_off | `RETROINFER_LAYER_CACHE_RESIDENCY=layers:0,2-31` | pass | 0 | 0.00 | 5.630859 | 41772 | 194.096747 | -2.384% | prior Pareto tradeoff |

## 结论

- `layer1_scale0p75` 是本轮推荐的 fractional Pareto 点：它正确性通过，block-cache 显存比 baseline 少 `0.045410 GiB`，进程峰值少 `48 MiB`，单轮实测 decode 为 `201.359835 tok/s`。这是实硬件单轮结果，应在后续更长重复轮中确认稳定性后再扩大吞吐主张。
- `layer1_scale0p25` 是更偏显存节省的 Pareto tradeoff：block-cache 少 `0.136230 GiB`、进程峰值少 `140 MiB`，decode 较 baseline 低 `1.559%`，但高于 prior layer-off，说明 fractional capacity 比直接关闭 layer 1 更温和。
- `layer1_scale0p50` 正确性通过但非竞争：它比 `layer1_scale0p25` 显存更高且 decode 更低，标记为 dominated/non-competitive，不作为保留候选。

## 原始证据

- `attempts/opt_attempt_002_fractional_layer_capacity/layer1_scale0p25_120k_b8_cr0p05/`
- `attempts/opt_attempt_002_fractional_layer_capacity/layer1_scale0p50_120k_b8_cr0p05/`
- `attempts/opt_attempt_002_fractional_layer_capacity/layer1_scale0p75_120k_b8_cr0p05/`
- 结构化汇总：`attempts/opt_attempt_002_fractional_layer_capacity/OUTCOME.json`

每个执行候选的 raw log 都包含 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`。
