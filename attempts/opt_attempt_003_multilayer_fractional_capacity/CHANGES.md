# opt_attempt_003_multilayer_fractional_capacity

## 变更

- 本轮没有改动 CUDA/C++、没有 rebuild、没有新增 benchmark harness；复用现有默认关闭的 `RETROINFER_LAYER_CACHE_CAPACITY_SCALE` 和 `throughput_eval/reproduce_block_cache.py`。
- 在运行测量前追加了 optimize-stage frontier watch：`research/frontier/optimize.json` 与 `research/FRONTIER_WATCH.jsonl`，记录从 single-layer fractional capacity 转向 multi-layer fractional capacity schedule 的机制 pivot。
- 使用项目 `.venv` 和已安装 `retroinfer_kernels` 做 fresh environment audit：runtime 为 `torch==2.5.1+cu124`、A100 80GB、关键 kernel symbols import 成功；source rebuild 仍因未证明 CUDA 12.4-compatible `nvcc`/`CUDA_HOME` 而保持阻塞。

## 候选依据

候选来自 `attempts/opt_attempt_001_layer_cache_residency/telemetry_probe_120k_b8_cr0p05/results.jsonl` 的真实低命中层：`1,25,29,2,3,28`。为记录 cache-hit impact，本轮 3 个 1-round optimize 测量均设置 `RETROINFER_CACHE_TELEMETRY=1`；因此吞吐与未开 telemetry 的 `research/BASELINE_RESULT.json`、`research/VALIDATION_RESULT.json` 对比只作为 optimize-stage 定向证据，后续 validation 应关闭 telemetry 重复测量。

## 120000x8 / gen_len=100 / cache_ratio=0.05 结果

| candidate | capacity spec | correctness | block cache GiB | peak GPU MiB | decode tok/s | e2e gen tok/s | total hit-rate delta vs telemetry baseline | classification |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | default | pass | 5.812500 | 41960 | 198.836414 | 3.277046 | 0.000000 | canonical reference |
| low3_scale0p75 | `1=0.75,25=0.75,29=0.75` | pass | 5.676270 | 41816 | 193.784409 | 3.286009 | -0.002569 | e2e-conservative tradeoff |
| low6_scale0p75 | `1=0.75,2-3=0.75,25=0.75,28-29=0.75` | pass | 5.540039 | 41672 | 194.410435 | 3.268599 | -0.005885 | retained memory/decode Pareto tradeoff |
| low3_0p75_secondary_0p875 | `1=0.75,25=0.75,29=0.75,2-3=0.875,28=0.875` | pass | 5.608887 | 41744 | 192.109237 | 3.274679 | -0.003898 | non-competitive decode tradeoff |

`low6_scale0p75` 是本轮保留候选：相对 canonical baseline，block-cache 显存少 `0.272461 GiB` (`-4.6875%`)，进程峰值 GPU memory 少 `288 MiB`，decode throughput 低 `2.226%`；相对 prior layer-off candidate，它同时降低 block-cache/峰值显存并略高 decode throughput。它不支持 async H2D overlap 声明，结论仅限本轮 telemetry-enabled optimize 探索。

## 原始证据

- Frontier snapshot: `research/frontier/optimize.json`
- Environment audit: `attempts/opt_attempt_003_multilayer_fractional_capacity/environment_audit.log`
- `low3_scale0p75`: `attempts/opt_attempt_003_multilayer_fractional_capacity/low3_scale0p75_120k_b8_cr0p05/`
- `low6_scale0p75`: `attempts/opt_attempt_003_multilayer_fractional_capacity/low6_scale0p75_120k_b8_cr0p05/`
- `low3_0p75_secondary_0p875`: `attempts/opt_attempt_003_multilayer_fractional_capacity/low3_0p75_secondary_0p875_120k_b8_cr0p05/`
- Structured outcome: `attempts/opt_attempt_003_multilayer_fractional_capacity/OUTCOME.json`

每个执行候选的 raw log 都包含 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true` 且 harness returncode 为 `0`。
