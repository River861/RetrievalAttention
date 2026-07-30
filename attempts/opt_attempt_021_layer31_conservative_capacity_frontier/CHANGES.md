# opt021 layer31 conservative capacity frontier

## 变更

- 未改动源代码；本轮只使用现有 opt013/latest Python 运行时开关。
- 复用 opt013 的 late block-cache allocation、uninitialized late init、pinned side-stream index metadata migration、async cluster-id copy、`RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`，并保持 `RETROINFER_CACHE_TELEMETRY` unset。
- 按任务边界只测两个候选：先测 `31=0.875`，因正确性通过但 decode 保留门失败，再测唯一允许 fallback `31=0.9375`。

## 结果

| candidate | block cache MiB | peak GPU MiB | decode tok/s | e2e latency s | e2e tok/s | overflow warnings | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opt013 reference | 5673.0 | 29884.0 | 198.036572 | 244.734262 | 3.268888 | 0 | reference |
| layer31_0p875 | 5650.0 | 29860.0 | 196.805201 | 244.464769 | 3.272455 | 0 | fail: decode_regressed_vs_opt013 |
| layer31_0p9375 | 5661.5 | 29872.0 | 199.188527 | 245.526919 | 3.258299 | 0 | fail: e2e_regressed_vs_opt013 |

## 结论

两个候选都 returncode=0、raw `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`、raw `use_cuda_graph=true`、有完整 memory samples 且无 buffer-overflow warning，并都降低 block cache 与进程峰值 GPU 显存。`layer31_0p875` 降低 block cache 23.0 MiB、峰值 GPU 24.0 MiB，但 decode 低于 opt013；`layer31_0p9375` 降低 block cache 11.5 MiB、峰值 GPU 12.0 MiB，decode 高于 opt013，但 e2e generated throughput 低于 opt013。因此没有候选满足相对 opt013 的 decode/e2e 双无回归保留门槛，不触发两轮复测，不保留。

## 证据

- Frontier: `frontier_check.raw.txt`, `frontier_optimize_input.json`
- Environment audit: `ENV_AUDIT.raw.txt`
- Raw runs: `raw/layer31_0p875_120k_b8_cr0p05_r1/`, `raw/layer31_0p9375_120k_b8_cr0p05_r1/`
- Structured summary: `candidate_screen_results.json`, `candidate_run_manifest.csv`, `OUTCOME.json`
- Source provenance: `NO_SOURCE_CHANGE.md`, `SOURCE_PROVENANCE.json`, `SOURCE_DIFF.patch`
