# opt019 latest-stack telemetry-guided capacity

## 结果

本轮没有保留新方案。已复用 opt013 作为参考，未重跑健康 baseline 或 Figure 13；先在 opt013 栈上跑了一次 `RETROINFER_CACHE_TELEMETRY=1` 的 120000x8/cache_ratio=0.05 遥测 profile，仅用于 per-layer hit/miss/update 选层，不用它声明吞吐。

## 选层依据

遥测总计 hit_rate=0.922837，最低 hit-rate 层为 1/2/25/29/28/3；但 layer-1 降容量和 stream-only 已在 opt014/opt018 负结果中显示吞吐敏感。因此本轮只尝试非重复的 full-capacity 低 hit-rate 层：layer31 单层 0.75，以及 layers27/30/31 轻量 0.875。完整层表见 `attempts/opt_attempt_019_latest_stack_telemetry_guided_capacity/telemetry_layer_stats.csv`。

## 实测候选

| candidate | block cache MiB | peak GPU MiB | decode tok/s | e2e tok/s | vs opt013 | decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| layer31_0p75 | 5626.5 | 29836.0 | 198.996261 | 3.268093267 | bc -46.5, peak -48.0, decode +0.959689, e2e -0.000794258 | fail_one_round (e2e_regressed_vs_opt013) |
| full_low3_0p875 | 5604.0 | 29812.0 | 190.781388 | 3.256874698 | bc -69.0, peak -72.0, decode -7.255184, e2e -0.012012826 | fail_one_round (decode_regressed_vs_opt013,e2e_regressed_vs_opt013) |

两个候选均 returncode=0 且 raw `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`。`layer31_0p75` 降低 block cache/peak memory 且 decode 高于 opt013，但 e2e generated throughput 比 opt013 低 0.000794 tok/s；`full_low3_0p875` 降低更多显存但 decode/e2e 明显回退。因此没有触发 two-round rerun。

## 代码与工具链

没有 opt019 源码改动；继续使用现有 Python env-gated capacity surface。项目 `.venv` 的 cu124 runtime 可用，installed extensions 可导入；source rebuild 仍因未证明 CUDA 12.4 nvcc/CUDA_HOME 而阻塞，且本节点非 CUDA/C++ rebuild。
