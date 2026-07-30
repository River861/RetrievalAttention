# opt022 scratch empty layer31 recovery

## 变更

- 在 `cache_hub/retroinfer_cache.py` 增加默认关闭的 `RETROINFER_SCRATCH_BUFFER_INIT`：默认/未设置仍走 `torch.zeros`；设置为 `uninitialized` 时，仅对 decode 路径中已由 kernel/PyTorch 输出覆盖后才读取的 scratch tensors 使用 `torch.empty`。
- 在 `throughput_eval/reproduce_block_cache.py` 增加 scratch init policy/env/bytes/count telemetry 字段，便于 raw `RETROINFER_RESULT_JSON`、`results.jsonl`、`summary.csv` 证明该路径生效。
- 未改 CUDA/C++、未新增 native kernel、未新增 benchmark harness；保留现有 opt013/latest late block-cache、pinned side-stream metadata、async cluster-id copy 栈。

## 实测结果

| candidate | rounds | block cache MiB | peak GPU MiB | decode tok/s mean | e2e tok/s mean | scratch empty | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opt013 reference | 2 | 5673.0 | 29884.0 | 198.036572 | 3.268888 | n/a | reference |
| scratch_empty_layer31_0p75 | 2 | 5626.5 | 29836.0 | 198.172430 | 3.267173 | 269.9 MiB / 17 tensors | fail_two_round_mean: e2e_regressed_vs_opt013 |

两轮 raw log 均 `returncode=0`、`use_cuda_graph=true`、`RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`，memory samples 完整且没有 buffer-overflow warning。候选相对 opt013 降低 block cache -46.5 MiB、进程峰值 GPU 显存 -48.0 MiB，decode mean 提高 0.135858 tok/s；但两轮 e2e generated throughput 低 0.001715 tok/s，因此不保留。

## 证据

- Fresh frontier: `frontier_check.raw.txt`, `frontier_optimize_input.json`
- Fresh env audit: `ENV_AUDIT.raw.txt`
- Source provenance/diff: `SOURCE_PROVENANCE.json`, `SOURCE_DIFF.patch`
- Raw measurement: `raw/scratch_empty_layer31_0p75_120k_b8_cr0p05/`
- Structured summary: `candidate_screen_results.json`, `candidate_run_manifest.csv`, `OUTCOME.json`
