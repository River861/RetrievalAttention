# opt009 stream index metadata staging

## 变更

- 在 `cache_hub/retroinfer_cache.py` 新增默认关闭的 `RETROINFER_STAGE_INDEX_METADATA_LAYERS`。unset/empty/off 时保持原行为；设置为 `2`、`2-3`、`layers:2,3` 或 `all` 时，所选层的 `centroids`、`value_sum`、`centroids_mask`、`cluster_size`（以及 stream-only 所需 `cluster_size_cumsum`）保留在 pinned CPU host memory，不再作为逐层常驻 GPU tensor。
- 为每个相关 GPU 分配一个共享 staging buffer，使用独立 CUDA stream 做 non-blocking H2D prefetch；eager 与 CUDA graph 路径都在 topk 前同步 staged buffer，CUDA graph 捕获固定 buffer 地址。该实现不改 CUDA/C++ 源码、不重建 vendor/kernel 库。
- 在 `block_cache_metadata()` 与 `throughput_eval/reproduce_block_cache.py` 中记录 staged-index metadata：staged layers、nominal/persistent/stage/resident GPU bytes、host-pinned bytes、prefetch/sync/copy 计数、copy/sync policy、CUDA graph 行为等。原有 block-cache KV bytes 口径保持不变。

## 实验口径

复用 canonical `120000x8`、`gen_len=100`、`cache_ratio=0.05`、`retrieval_budget=0.018`、`estimation_budget=0.232`、seed 2025、GPU0 A100 80GB；叠加保留候选 `RETROINFER_ASYNC_CLUSTER_ID_COPY=1`、`RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`、`RETROINFER_LAYER_CACHE_CAPACITY_SCALE=1=0.75,2-3=0.75,25=0.75,28-29=0.75`。baseline 复用 `research/BASELINE_RESULT.json`，当前验证参考复用 `research/VALIDATION_RESULT.json`，未重跑健康 baseline，未运行 Figure 13。

## 结果

| candidate | staged layers | correctness | block cache MiB | peak GPU MiB | decode tok/s | e2e tok/s | staged-index GPU bytes saved | H2D staged bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stage_l2` | `[2]` | pass | 5673.0 | 41604 | 196.4327 | 3.2858 | 0 | 245222400 |
| `stage_l2_l3` | `[2, 3]` | pass | 5673.0 | 41366 | 143.0606 | 3.2514 | 245222400 | 49044480000 |

`stage_l2` 是机制可行性点：CUDA graph 可读取 staging buffer，正确性通过，但单层 staging 只有一个 per-device buffer 替代一个常驻层，resident index metadata 省 0 bytes，decode 比 retained validation 低 2.17 tok/s。

`stage_l2_l3` 是首个实际省显存点：相比 retained validation，峰值进程 GPU 显存少 238 MiB，index metadata resident GPU bytes 少 245222400；但每个 decode token 需要在两层之间重复搬运约 245 MB metadata，整轮 H2D staged bytes 为 49044480000，decode 比 validation 低 55.54 tok/s，e2e 低 0.03365 tok/s。该机制不形成优于现有 `low6+async+buffer_mult3.0` 的显存-吞吐 Pareto，按负结果停止扩展。

## 证据路径

- 环境审计：`attempts/opt_attempt_009_stream_index_metadata_staging/environment_audit.json`、`environment_audit.log`
- Frontier binding：`attempts/opt_attempt_009_stream_index_metadata_staging/frontier_binding.json`
- 源码 diff：`attempts/opt_attempt_009_stream_index_metadata_staging/source_diff.patch`
- 候选原始结果：`attempts/opt_attempt_009_stream_index_metadata_staging/stage_l2_120k_b8_cr0p05/`、`stage_l2_l3_120k_b8_cr0p05/`
- 结构化汇总：`attempts/opt_attempt_009_stream_index_metadata_staging/candidate_summary.json`、`candidate_summary.csv`
- 结论：`attempts/opt_attempt_009_stream_index_metadata_staging/OUTCOME.json`
