# opt_attempt_006_telemetry_protected_capacity_frontier

## 变更

- 本次没有新增或修改源代码；复用现有默认关闭的 `RETROINFER_LAYER_CACHE_CAPACITY_SCALE` 与 `RETROINFER_ASYNC_CLUSTER_ID_COPY` 运行时开关，保持 API/default 行为不变。
- 新增本 attempt 的 fresh frontier binding、环境审计、三组 telemetry-disabled + async-enabled 120000x8 单轮测量、raw logs/results/memory samples，以及结构化 `OUTCOME.json`。
- 未 CUDA/C++ rebuild，未新增 benchmark harness，未重跑 baseline/Figure 13/cache-ratio matrix。

## 环境与 frontier

- 环境审计: `attempts/opt_attempt_006_telemetry_protected_capacity_frontier/environment_audit.log`；`.venv` runtime 为 PyTorch CUDA 12.4 + A100 80GB，installed `retroinfer_kernels` symbols 可导入。CUDA 12.4 rebuild compiler 仍未证明。
- Frontier binding: `attempts/opt_attempt_006_telemetry_protected_capacity_frontier/frontier_binding.json`，对应 snapshot `research/frontier/optimize_telemetry_protected_capacity_frontier.json`，已追加 `research/FRONTIER_WATCH.jsonl`。

## 120000x8 / gen_len=100 / cache_ratio=0.05 实测结果

| candidate | capacity spec | correctness | block cache GiB | peak GPU MiB | decode tok/s | e2e gen tok/s | vs validated low6+async | conclusion |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `stable_low3_0p5_async` | `2-3=0.5,28=0.5` | pass | 5.540039 | 41684 | 194.572679 | 3.285282 | same block cache, +12 MiB peak, -6.607 tok/s decode | not retained |
| `l1_0p875_low3_0p5_async` | `1=0.875,2-3=0.5,28=0.5` | pass | 5.517578 | 41660 | 194.692875 | 3.271978 | -0.022461 GiB, -12 MiB peak, -6.487 tok/s decode | not retained |
| `l1_0p875_low3_0p375_async` | `1=0.875,2-3=0.375,28=0.375` | pass | 5.450195 | 41588 | 193.181476 | 3.270082 | -0.089844 GiB, -84 MiB peak, -7.998 tok/s decode | not retained |

结论：三组 bounded candidate 均 returncode 0 且 raw log 包含 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`。更强 cuts 能继续降低 block cache/峰值显存，但 decode throughput 均低于 `research/VALIDATION_RESULT.json` 的 validated low6+async；本轮不保留新候选。

## 原始证据

- `stable_low3_0p5_async`: `attempts/opt_attempt_006_telemetry_protected_capacity_frontier/stable_low3_0p5_async_120k_b8_cr0p05/`
- `l1_0p875_low3_0p5_async`: `attempts/opt_attempt_006_telemetry_protected_capacity_frontier/l1_0p875_low3_0p5_async_120k_b8_cr0p05/`
- `l1_0p875_low3_0p375_async`: `attempts/opt_attempt_006_telemetry_protected_capacity_frontier/l1_0p875_low3_0p375_async_120k_b8_cr0p05/`
- Structured outcome: `attempts/opt_attempt_006_telemetry_protected_capacity_frontier/OUTCOME.json`
