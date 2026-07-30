# opt013：uninitialized late block-cache allocation

## 环境与前沿约束

- fresh optimize frontier record：`research/frontier/optimize.json`，record_id `48ad2d2ac2538fd5`；`frontier_watch check --stage optimize` 通过。
- 项目运行时复核通过：`.venv/bin/python`、Python 3.11.15、Torch 2.5.1+cu124、A100 80GB；`retroinfer_kernels` 的 `ThreadPool`、`WaveBufferCPU`、gather/scatter 与 `batch_gemm_softmax` 符号可导入。
- source rebuild 仍是环境阻塞/非本轮目标：`CUDA_HOME` 未设置，`torch.utils.cpp_extension.CUDA_HOME=/usr`，`/usr/bin/nvcc` 是 CUDA 12.0，另有 `/usr/local/cuda-13.3/bin/nvcc`，未证明 CUDA 12.4 编译器。

## 代码改动

- `cache_hub/retroinfer_cache.py`
  - 新增默认关闭的 `RETROINFER_LATE_BLOCK_CACHE_INIT`。
  - 默认 `unset/default/zero/zero_fill` 保持原有 `torch.zeros` 行为。
  - `empty/uninitialized/no_zero_fill` 仅在 forced-late `prepare_cache()` 路径中对 `cache_keys/cache_values` 使用 `torch.empty`；prefill/preallocate、index metadata、execution buffer 等其它张量仍保持原行为。
  - 记录 policy/env/effective/mode/safety、uninitialized tensor count/bytes，以及 PyTorch deterministic/uninitialized-fill 状态。
- `throughput_eval/reproduce_block_cache.py`
  - 将新 env 与 late block-cache init telemetry 加入 raw result、summary 和 CSV 字段。

安全依据：`WaveBufferCPU` 初始将所有 cluster 置为 miss，只有 `batch_update` 分配 block id 后才将 `inBlockCache=true`；Python 侧随后用 `gather_copy_and_scatter()` 把 execution buffer 写入 GPU cache。因此 late block-cache K/V tensor 的 cache-hit 读取发生在 admission 写入之后。

## 实验命令

未重跑 baseline；复用 `research/BASELINE_RESULT.json`、opt010、opt011、opt012 artifacts。

```bash
env -u RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY \
  -u RETROINFER_STAGE_INDEX_METADATA_LAYERS \
  -u RETROINFER_STREAM_ONLY_LAYERS \
  -u RETROINFER_LAYER_CACHE_RESIDENCY \
  -u RETROINFER_CACHE_TELEMETRY \
  RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late \
  RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream \
  RETROINFER_LATE_BLOCK_CACHE_INIT=uninitialized \
  RETROINFER_ASYNC_CLUSTER_ID_COPY=1 \
  RETROINFER_LAYER_CACHE_CAPACITY_SCALE='1=0.75,2-3=0.75,25=0.75,28-29=0.75' \
  RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0 \
  .venv/bin/python throughput_eval/reproduce_block_cache.py \
    --suite opt013_uninitialized_late_block_cache \
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
    --output-dir attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/raw/harness_120k_b8_cr0p05
```

## 结果

| 指标 | opt012 | opt013 |
| --- | ---: | ---: |
| correctness | 2/2 pass | 2/2 pass |
| block cache | 5673.0 MiB | 5673.0 MiB |
| late uninitialized block-cache bytes | 0 | 5,948,571,648 |
| peak process GPU memory | 29884.0 MiB | 29884.0 MiB |
| decode throughput | 201.1317 tok/s | 198.0366 tok/s |
| e2e latency | 245.0673 s | 244.7343 s |
| e2e generated throughput | 3.264410 tok/s | 3.268888 tok/s |

结论：opt013 保持 opt012 的 block-cache/peak-memory 口径并小幅改善 e2e，但 decode 明显回退。因此该方案只作为“同显存、e2e 小幅改善、decode 负面”的 bounded candidate，等待独立 review；不声明 profiler 级 overlap 或通用加速。

## 证据路径

- 主结果：`raw/harness_120k_b8_cr0p05/results.jsonl`、`summary.csv`、`per_round_table.csv`、`raw_logs/`、`memory_samples/`
- 结构化结果：`OUTCOME.json`
- 源码 diff：`SOURCE_DIFF.patch`
- 环境审计：`ENV_AUDIT.raw.txt`、`ENV_AUDIT.md`
- Frontier binding：`frontier_optimize_input.json`、`frontier_record.raw.txt`、`frontier_check.raw.txt`
