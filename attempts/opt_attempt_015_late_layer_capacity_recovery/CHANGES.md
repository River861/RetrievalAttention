# opt015 late-layer capacity recovery

## 结论

本轮不保留新的 capacity schedule。三组 1-round A100 120000x8 / gen_len=100 / cache_ratio=0.05 候选均正确性通过，raw log 中 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true` 且 returncode=0。唯一 1-round 同时降低显存且 decode/e2e 不差于 opt013 的 `1=0.75,2-3=0.75,25=0.625,28-29=0.625` 已按要求 rerun 到 2 rounds；确认结果仍正确性 2/2 pass，但 decode 和 e2e generated throughput 均低于 opt013，因此 refuted。

## 代码与运行栈

- opt015 没有新增源代码改动；复用 opt013/opt014 已存在的环境变量控制面。
- 固定栈：`RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late`、`RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream`、`RETROINFER_LATE_BLOCK_CACHE_INIT=uninitialized`、`RETROINFER_ASYNC_CLUSTER_ID_COPY=1`、`RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`。
- source provenance：`SOURCE_DIFF.patch`，sha256 `3e66848a8981311e7619d6567547afba0dd5d675097c25a8c071627a7952f215`；`SOURCE_PROVENANCE.json` 记录 HEAD 与相关 source surfaces。
- 环境审计：`.venv` Python 3.11.15、torch 2.5.1+cu124、A100 80GB、`retroinfer_kernels`/`weighted_flash_decoding` import OK；CUDA/C++ rebuild 仍因缺少已证明 CUDA 12.4 nvcc/CUDA_HOME 而阻塞，未尝试 rebuild。

## 与 opt013 对比

opt013 参考点：block cache 5673.0 MiB、peak process GPU memory 29884 MiB、decode 198.036572 tok/s、e2e generated throughput 3.268888 tok/s。

| schedule | rounds | block cache MiB | peak GPU MiB | decode tok/s | e2e latency s | e2e gen tok/s | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `1=0.75,2-3=0.75,25=0.625,28-29=0.625` | 1 | 5602.5 | 29812 | 198.448903 | 244.422221 | 3.273025 | screen pass; rerun required |
| `1=0.75,2-3=0.75,25=0.75,28-29=0.625` | 1 | 5626.0 | 29836 | 180.400476 | 245.917653 | 3.253121 | decode throughput worse than opt013; e2e generated throughput worse than opt013 |
| `1=0.75,2-3=0.75,25=0.625,28-29=0.75` | 1 | 5649.5 | 29860 | 193.642122 | 245.709864 | 3.255873 | decode throughput worse than opt013; e2e generated throughput worse than opt013 |
| `1=0.75,2-3=0.75,25=0.625,28-29=0.625` | 2 | 5602.5 | 29812 | 197.210521 | 245.640371 | 3.256794 | refuted: decode/e2e worse in 2-round confirmation |

## 证据

- Frontier：`frontier_optimize_input.json`、`frontier_record.raw.txt`、`frontier_check.raw.txt`。
- Raw harness artifacts：`raw/*_120k_b8_cr0p05/{results.jsonl,summary.csv,per_round_table.csv,raw_logs/,memory_samples/,harness_stdout.log,harness_returncode.txt}`。
- 结构化结论：`OUTCOME.json`。
- 关键 raw latency source：每个 candidate 的 `results.jsonl`/`summary.csv` 记录 `e2e_latency_s`；表内 e2e generated throughput 按 `batch_size * gen_len / mean_e2e_latency_s` 计算。
