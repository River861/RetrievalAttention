# opt014 latest-stack block-cache capacity frontier

## 结论

本轮没有保留新的低容量 schedule。三个候选都在 A100 120000x8 / gen_len=100 / cache_ratio=0.05 / seed=2025 下正确性通过，raw log 中 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true` 且进程退出码为 0；但三个候选相对 opt013 都降低了 decode throughput，因此未满足“block cache/peak memory 更低且 decode/e2e 不差于 opt013”的保留门槛。

## 代码与运行栈

- 本轮没有新增源代码改动；复用现有环境变量控制面。
- 固定栈：`RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late`、`RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream`、`RETROINFER_LATE_BLOCK_CACHE_INIT=uninitialized`、`RETROINFER_ASYNC_CLUSTER_ID_COPY=1`、`RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`。
- 复用源 diff 身份：`SOURCE_DIFF.patch`，sha256 `3e66848a8981311e7619d6567547afba0dd5d675097c25a8c071627a7952f215`。

## 与 opt013 对比

opt013 参考点：block cache 5673.0 MiB、peak process GPU memory 29884 MiB、decode 198.036572 tok/s、e2e generated throughput 3.268888 tok/s。

| schedule | block cache MiB | peak GPU MiB | decode tok/s | e2e gen tok/s | gate |
|---|---:|---:|---:|---:|---|
| `1=0.625,2-3=0.625,25=0.625,28-29=0.625` | 5532.0 | 29740 | 195.456719 | 3.269190 | refuted: decode worse |
| `1=0.25,2-3=0.75,25=0.75,28-29=0.75` | 5580.0 | 29800 | 173.804211 | 3.252068 | refuted: decode/e2e worse |
| `1=0.5,2-3=0.75,25=0.75,28-29=0.75` | 5626.5 | 29820 | 193.712211 | 3.259187 | refuted: decode/e2e worse |

## 证据

- 环境审计：`ENV_AUDIT.raw.txt`，证明 `.venv` / torch cu124 / A100 可用；CUDA 12.4 rebuild compiler 仍未证明，因此未做 CUDA/C++ rebuild。
- Frontier：`frontier_optimize_input.json`、`frontier_record.raw.txt`、`frontier_check.raw.txt`。
- Raw harness artifacts：`raw/*_120k_b8_cr0p05/{results.jsonl,summary.csv,per_round_table.csv,raw_logs/,memory_samples/,harness_stdout.log,harness_returncode.txt}`。
- 结构化结论：`OUTCOME.json`。
