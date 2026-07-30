# opt012：pinned late metadata migration

## 环境与前沿约束

- 使用项目固定运行时：`.venv/bin/python`、Python 3.11.15、Torch 2.5.1+cu124、A100 80GB；`retroinfer_kernels` 与 `weighted_flash_decoding` 可导入。
- 小型 pinned host tensor + side CUDA stream + non_blocking H2D copy 在本机通过；CUDA/C++ source rebuild 仍因未证明 CUDA 12.4 `nvcc/CUDA_HOME` 而阻塞，本轮未重编。
- fresh optimize frontier record：`research/frontier/optimize.json`，record_id `1faa0539da74a3a6`。边界不变：没有 profiler/timeline，所以不声明 transfer/compute overlap。

## 代码改动

- `cache_hub/retroinfer_cache.py`
  - 新增默认关闭的 `RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY`。
  - 默认 `unset/default/pageable_blocking` 保留原 late path：CPU pageable metadata，`prepare_cache()` 中 blocking current-stream H2D。
  - 新增候选策略：
    - `pinned_blocking`：prefill metadata 分配到 pinned CPU，decode 前 blocking current-stream H2D。
    - `pinned_non_blocking`：prefill metadata 分配到 pinned CPU，decode 前 current-stream `non_blocking=True` H2D 并同步。
    - `pinned_side_stream`：prefill metadata 分配到 pinned CPU，decode 前 per-device side stream `non_blocking=True` H2D，`allocate_computation_buffer()` 后、decode 前同步。
  - 保持 forced-late 语义：prefill 阶段 `index_metadata_prefill_gpu_bytes=0`，`cache_keys/cache_values` 和 computation buffer 仍在 `prepare_cache()` 分配。
  - 记录 late metadata migration policy、copy mode、copy bytes、source pinned bytes、stream/sync counts、host copy-launch + sync-wait elapsed ms、prepare-window elapsed ms 与 residency bytes。`elapsed_ms` 不再把交错的 cache/buffer allocation 误标成纯 migration 时间。
- `throughput_eval/reproduce_block_cache.py`
  - 将新 env 与 migration metrics 加入 raw result、summary、per-round CSV 字段。

## 实验命令

主验证未重跑 baseline，复用 `research/BASELINE_RESULT.json`、opt010 和 opt011 artifacts。

```bash
env -u RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY \
  -u RETROINFER_STAGE_INDEX_METADATA_LAYERS \
  -u RETROINFER_STREAM_ONLY_LAYERS \
  -u RETROINFER_LAYER_CACHE_RESIDENCY \
  -u RETROINFER_CACHE_TELEMETRY \
  RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late \
  RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream \
  RETROINFER_ASYNC_CLUSTER_ID_COPY=1 \
  RETROINFER_LAYER_CACHE_CAPACITY_SCALE='1=0.75,2-3=0.75,25=0.75,28-29=0.75' \
  RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0 \
  .venv/bin/python throughput_eval/reproduce_block_cache.py \
    --suite opt012_pinned_late_metadata_side_stream \
    --context-batch-groups 120000x8 \
    --cache-ratios 0.05 \
    --baseline-cache-ratio 0.05 \
    --rounds 2 \
    --gen-len 100 \
    --retrieval-budget 0.018 \
    --estimation-budget 0.232 \
    --seed 2025 \
    --cuda-visible-devices 0 \
    --require-idle-gpu \
    --stop-on-error \
    --output-dir attempts/opt_attempt_012_pinned_late_metadata_migration/raw/harness_120k_b8_cr0p05
```

早期 side-stream 候选一度因 timing scope 被独立 review 指出含混；修正后保留 intended side-stream prepare window，但新增 scoped telemetry。bounded budget 额外跑过 1-round `pinned_blocking` 和 `pinned_non_blocking` probe；二者 e2e 均低于最终 side-stream 主候选，未提升为 2-round retained result。

## 结果

| 指标 | opt010 | opt011 | opt012 retained |
| --- | ---: | ---: | ---: |
| correctness | 2/2 pass | 2/2 pass | 2/2 pass |
| block cache | 5673.0 MiB | 5673.0 MiB | 5673.0 MiB |
| prefill index metadata GPU bytes | not captured | 7,847,116,800 | 0 |
| prefill index metadata pinned CPU bytes | not captured | 0 | 7,847,116,800 |
| late metadata H2D bytes | not captured | 0 | 7,847,116,800 |
| late migration host launch + sync wait | not captured | 0 | 24.50 ms mean |
| late migration prepare window | not captured | 0 | 590.50 ms mean |
| peak process GPU memory | 29884.0 MiB | 35610.0 MiB | 29884.0 MiB |
| decode throughput | 198.5853 tok/s | 198.5628 tok/s | 201.1317 tok/s |
| e2e latency | 245.2668 s | 244.1727 s | 245.0673 s |
| e2e generated throughput | 3.261779 tok/s | 3.276369 tok/s | 3.264410 tok/s |

结论：opt012 保留 opt010 的低 peak GPU memory，并把 forced-late e2e throughput 从 3.261779 小幅提高到 3.264410 tok/s，同时 decode 从 198.5853 提高到 201.1317 tok/s；相对 opt011 则少 5726 MiB peak memory，但 e2e 仍更低。因此这是“opt010-like memory + 小幅吞吐回收”的 bounded tradeoff，不是 profiler-proven overlap 或广泛 speedup。

## 证据路径

- 主结果：`raw/harness_120k_b8_cr0p05/results.jsonl`、`summary.csv`、`per_round_table.csv`、`raw_logs/`、`memory_samples/`
- 变体 probe：`raw/pin_only_probe_120k_b8_cr0p05/`、`raw/non_blocking_probe_120k_b8_cr0p05/`
- 结构化结果：`OUTCOME.json`
- 源码 diff：`SOURCE_DIFF.patch`
- 环境审计：`ENV_AUDIT.raw.txt`、`ENV_AUDIT.md`
- Frontier binding：`frontier_optimize_input.json`、`research/frontier/optimize.json`
