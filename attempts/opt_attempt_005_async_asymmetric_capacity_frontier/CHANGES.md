# opt_attempt_005_async_asymmetric_capacity_frontier

## 变更

- 本次没有新增/修改源代码；复用现有默认关闭的 `RETROINFER_LAYER_CACHE_CAPACITY_SCALE` 与 `RETROINFER_ASYNC_CLUSTER_ID_COPY` 运行时开关，保持 API/default 行为不变。
- 新增本 attempt 的 fresh optimize frontier binding、环境审计、三组 telemetry-disabled + async-enabled 120000x8 单轮测量、raw logs/results/memory samples，以及结构化 `OUTCOME.json`。
- 未 CUDA/C++ rebuild，未新增 benchmark harness，未运行 baseline/Figure 13/cache-ratio matrix。

## 环境与 frontier

- 环境审计: `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/environment_audit.log`；`.venv` runtime 为 PyTorch CUDA 12.4 + A100 80GB，installed `retroinfer_kernels` symbols 可导入。CUDA 12.4 rebuild compiler 仍未证明。
- Frontier binding: `research/frontier/optimize_async_asymmetric_capacity_frontier.json` (`sha256=c40ee5cdf0eb366a36c572ab37dac65c933fb2bd81ac803018dd909bb2a99190`)，已追加 `research/FRONTIER_WATCH.jsonl`。

## 120000x8 / gen_len=100 / cache_ratio=0.05 实测结果

| candidate | correctness | block cache GiB | peak GPU MiB | decode tok/s | e2e gen tok/s | vs low6+async block GiB | vs low6+async peak MiB | vs low6+async decode | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `l1_0p25_low5_0p75_async` | pass | 5.449219 | 41580 | 193.844307 | 3.292937 | -0.090820 | -92 | -7.335661 | correctness_passed_more_memory_saving_but_decode_regressed_vs_low6_async |
| `mixed_0p5_0p75_async` | pass | 5.403809 | 41540 | 195.216069 | 3.272248 | -0.136230 | -132 | -5.963900 | correctness_passed_more_memory_saving_but_decode_regressed_vs_low6_async |
| `low6_0p625_async` | pass | 5.402344 | 41528 | 195.759420 | 3.272763 | -0.137695 | -144 | -5.420549 | correctness_passed_more_memory_saving_but_decode_regressed_vs_low6_async |

结论：本 bounded frontier exhausting。三组 schedule 均 raw oracle 正确并进一步省显存，但 decode 均低于 validated low6+async reference，因此不保留为下一阶段候选。

## 原始证据

- `l1_0p25_low5_0p75_async`: results `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/l1_0p25_low5_0p75_async_120k_b8_cr0p05/results.jsonl`; raw log `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/l1_0p25_low5_0p75_async_120k_b8_cr0p05/raw_logs/opt_attempt_005_l1_0p25_low5_0p75_async_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`; memory samples `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/l1_0p25_low5_0p75_async_120k_b8_cr0p05/memory_samples/opt_attempt_005_l1_0p25_low5_0p75_async_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`.
- `mixed_0p5_0p75_async`: results `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/mixed_0p5_0p75_async_120k_b8_cr0p05/results.jsonl`; raw log `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/mixed_0p5_0p75_async_120k_b8_cr0p05/raw_logs/opt_attempt_005_mixed_0p5_0p75_async_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`; memory samples `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/mixed_0p5_0p75_async_120k_b8_cr0p05/memory_samples/opt_attempt_005_mixed_0p5_0p75_async_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`.
- `low6_0p625_async`: results `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/low6_0p625_async_120k_b8_cr0p05/results.jsonl`; raw log `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/low6_0p625_async_120k_b8_cr0p05/raw_logs/opt_attempt_005_low6_0p625_async_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`; memory samples `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/low6_0p625_async_120k_b8_cr0p05/memory_samples/opt_attempt_005_low6_0p625_async_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`.
- Structured outcome: `attempts/opt_attempt_005_async_asymmetric_capacity_frontier/OUTCOME.json`.

所有执行候选的 harness returncode 均为 `0`，raw log 均包含 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`。
