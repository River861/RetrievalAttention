# opt023 scratch empty + buffer 2.75 layer31 screen

## 变更

- 本 attempt 未改源代码；使用现有默认关闭 knobs 组合验证：late block-cache allocation、late block-cache uninitialized、pinned side-stream index metadata migration、async cluster-id copy、层 1/2-3/25/28-29/31 capacity 0.75、buffer nprobe multiplier 2.75、scratch buffer uninitialized。
- 保留当前源代码 provenance 与 diff：`SOURCE_PROVENANCE.json`、`SOURCE_DIFF.patch`；未 CUDA/C++ rebuild、未新增 native kernel、未新增 benchmark harness。
- Fresh env/frontier audit 写入 `ENV_AUDIT.raw.txt` 与 `frontier_check.raw.txt`。

## 实测结果

| candidate | rounds | block cache MiB | peak GPU MiB | decode tok/s | e2e tok/s | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| opt013 reference | 2 | 5673.0 | 29884.0 | 198.036572 | 3.268888 | reference |
| scratch_empty_buffer_2p75_layer31_0p75 | 1 | 5626.5 | 29820.0 | 200.406653 | 3.267889 | fail_one_round: e2e_regressed_vs_opt013 |

Round 1 `returncode=0`、`use_cuda_graph=true`、`RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`、memory samples 完整、zero buffer-overflow warnings、`scratch_buffer_init_effective=true`、`buffer_nprobe_multiplier=2.75`。相对 opt013，block cache -46.5 MiB、peak GPU -64.0 MiB、decode +2.370081 tok/s，但 e2e -0.000999 tok/s，因此按 contract 不允许二轮 rerun，记录为 negative performance outcome。

## 证据

- Raw measurement: `attempts/opt_attempt_023_scratch_empty_buffer_2p75_layer31_screen/raw/scratch_empty_buffer_2p75_layer31_0p75_120k_b8_cr0p05_r1/`
- Structured summary: `candidate_screen_results.json`, `candidate_run_manifest.csv`, `OUTCOME.json`
- Raw log: `attempts/opt_attempt_023_scratch_empty_buffer_2p75_layer31_screen/raw/scratch_empty_buffer_2p75_layer31_0p75_120k_b8_cr0p05_r1/raw_logs/opt023_scratch_empty_buffer_2p75_layer31_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`
- Memory samples: `attempts/opt_attempt_023_scratch_empty_buffer_2p75_layer31_screen/raw/scratch_empty_buffer_2p75_layer31_0p75_120k_b8_cr0p05_r1/memory_samples/opt023_scratch_empty_buffer_2p75_layer31_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`
