# opt_attempt_001_layer_cache_residency

## 变更

- 在 `cache_hub/retroinfer_cache.py` 增加默认关闭的 `RETROINFER_CACHE_TELEMETRY=1` 统计，直接复用 `hit_num_units`、`miss_num_units`、`update_num_units`，在 `block_cache_metadata()`/`RETROINFER_RESULT_JSON` 中输出逐层 page 级 hit/miss/update 计数。
- 增加默认兼容的 `RETROINFER_LAYER_CACHE_RESIDENCY` Python-only 驻留开关；默认 `all` 与原行为一致，也支持 `layers:...`、`first:N`、`last:N`、`even/odd` 等逐层 block-cache 容量掩码。该实现只改变 Python 侧每层 `WaveBufferCPU` capacity、GPU cache tensor 形状和 gather/scatter cache stride，不触碰 CUDA/C++ rebuild。

## 瓶颈证据

`telemetry_probe_120k_b8_cr0p05` 使用 canonical `120000x8`、`gen_len=100`、`cache_ratio=0.05`、CUDA graph，正确性通过。总 page 级命中率为 `0.928702`，`update_num_units == miss_num_units`，说明 miss 后的 admission 是当前 block-cache 驻留压力的直接来源。最低命中层为 layer 1 (`0.737696`)，其次包括 layer 25/29/2/3/28/31/27，因此候选按逐层冷度而不是全局降低 `cache_ratio`。

## 结果

最佳候选为 `RETROINFER_LAYER_CACHE_RESIDENCY=layers:0,2-31`，仅关闭 layer 1 的 GPU block cache：

| run | correctness | active layers | block cache GiB | peak GPU MiB | decode tok/s | e2e gen tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| certified baseline (`research/BASELINE_RESULT.json`) | pass | 32 | 5.812500 | 41960 | 198.836414 | 3.277046 |
| candidate31 | pass | 31 | 5.630859 | 41772 | 194.096747 | 3.274087 |

相对 baseline，candidate31 降低 block-cache bytes `3.125%`，降低进程峰值 GPU memory `188 MiB`，decode throughput 下降 `2.384%`，end-to-end generated throughput 下降 `0.090%`。更激进的 28/24 层候选也正确，但 decode 损失分别为 `13.321%`/`25.445%`，不作为推荐候选。

## 原始证据

- Telemetry probe: `attempts/opt_attempt_001_layer_cache_residency/telemetry_probe_120k_b8_cr0p05/`
- Candidate31: `attempts/opt_attempt_001_layer_cache_residency/candidate31_120k_b8_cr0p05/`
- Negative candidates: `attempts/opt_attempt_001_layer_cache_residency/candidate28_120k_b8_cr0p05/`, `attempts/opt_attempt_001_layer_cache_residency/candidate24_120k_b8_cr0p05/`
- Baseline reference: `research/BASELINE_RESULT.json`

## 限制

该增量证明了逐层驻留可以提供一个高吞吐、低一点显存的真实 Pareto tradeoff，但不是对默认 `cache_ratio=0.05` 的吞吐支配；它以小幅 decode 损失换取小幅 block-cache/峰值显存下降。后续更有希望的方向是使用逐层容量缩放而非二值关闭，或把 layer 1 的 capacity 降到最小 admissible pages，以避免完全失去该层命中。
