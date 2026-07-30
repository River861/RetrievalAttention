# opt030 scratch buffer + layer31 capacity recovery 技术报告

## 变更

- 本 attempt 未修改源代码；只使用现有默认关闭 runtime knobs：late block-cache allocation、late block-cache uninitialized、pinned side-stream index metadata migration、async cluster-id copy、RETROINFER_SCRATCH_BUFFER_INIT=uninitialized、RETROINFER_BUFFER_NPROBE_MULTIPLIER=2.75。
- 按 bounded contract 在 canonical 120000x8/cache_ratio=0.05/gen_len=100/seed=2025 口径下先筛 layer31=0.8125；该一轮筛选通过后做 exact 2-round confirmation。确认失败后才运行 layer31=0.875 一轮 fallback screen；未运行 Figure13、healthy baseline、stream-only/async-wave rerun 或 CUDA/C++ rebuild。
- Fresh runtime/frontier/source provenance: `ENV_AUDIT.raw.txt`, `ENV_AUDIT.json`, `ENV_AUDIT.md`, `frontier_check.raw.txt`, `SOURCE_PROVENANCE.json`, `SOURCE_DIFF.patch`。

## 对照与结果

opt013 reference：block cache `5673.0` MiB，peak GPU `29884.0` MiB，decode `198.036572` tok/s，e2e `3.268888` tok/s。

| candidate | rounds | block cache MiB | peak GPU MiB | decode tok/s | e2e tok/s | block Δ MiB | peak Δ MiB | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c1_layer31_0p8125_screen | 1 | 5638.0 | 29832.0 | 200.104186 | 3.270718 | -35.0 | -52.0 | screen_passed_confirm_required: all_one_round_gates_passed |
| c1_layer31_0p8125_confirm2 | 2 | 5638.0 | 29832.0 | 189.169124 | 3.257162 | -35.0 | -52.0 | fail_confirmation: decode_no_worse_than_opt013,e2e_no_worse_than_opt013 |
| c2_layer31_0p875_screen | 1 | 5650.0 | 29844.0 | 198.646276 | 3.261603 | -23.0 | -40.0 | fail_one_round: e2e_no_worse_than_opt013 |

## 结论

- `c1_layer31_0p8125_screen` 一轮结果通过 correctness、use_cuda_graph、memory、decode、e2e gates，因此按 contract 启动 exact 2-round confirmation。
- `c1_layer31_0p8125_confirm2` 两轮 correctness 均通过，block cache `5638.0` MiB、peak GPU `29832.0` MiB，仍比 opt013 低；但 decode mean `189.169124` tok/s 和 e2e mean `3.257162` tok/s 均低于 opt013，因此不能 retained。
- `c2_layer31_0p875_screen` 保持 memory 与 decode one-round gates，但 e2e `3.261603` tok/s 低于 opt013 `3.268888`，按 bounded gate 不允许二轮确认。

## 证据路径

- c1 screen: `attempts/opt_attempt_030_scratch_buffer_layer31_capacity_recovery/raw/c1_layer31_0p8125_120k_b8_cr0p05_r1`
- c1 confirmation: `attempts/opt_attempt_030_scratch_buffer_layer31_capacity_recovery/raw/c1_layer31_0p8125_confirm2_120k_b8_cr0p05_r2`
- c2 fallback: `attempts/opt_attempt_030_scratch_buffer_layer31_capacity_recovery/raw/c2_layer31_0p875_120k_b8_cr0p05_r1`
- Structured: `candidate_screen_results.json`, `candidate_run_manifest.csv`, `OUTCOME.json`
- Command log: `command_manifest.jsonl`
