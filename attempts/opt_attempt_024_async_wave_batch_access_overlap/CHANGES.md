# opt024 async wave batch access overlap implementation notes

## 改动
- 在 `cache_hub/retroinfer_cache.py` 增加默认关闭的 `RETROINFER_ASYNC_WAVE_BATCH_ACCESS` Python/runtime 路径；默认未设置时保持原有串行顺序不变。
- 开启后，非 stream-only sparse attention 在 top-k 后启动已有 pinned `cluster_ids` D2H copy，并立即提交一个单线程 Python worker；worker 等待 copy event 后执行 `WaveBufferCPU.batch_access()`，主线程继续调度 estimation-zone GPU work，并在 `gather_copy_and_concat` 读取 wave-buffer 输出前显式 join。
- 在 `throughput_eval/reproduce_block_cache.py` 增加 async wave batch access 环境与 launch/sync/wait/batch-access 计时字段，原始日志可复核路径是否执行。

## 结果摘要
- opt013-capacity 隔离 1 轮：正确性通过，block cache 5673.0 MiB、峰值 29884.0 MiB、decode 199.855955 tok/s、e2e generated 3.270202 tok/s；无显存收益，仅作为 lower-memory gate。
- lower-memory screen 1 轮：正确性通过，block cache 5626.5 MiB、峰值 29820.0 MiB、decode 199.562963 tok/s、e2e generated 3.282219 tok/s，触发二轮确认。
- lower-memory 二轮确认：2/2 正确性通过，block cache 5626.5 MiB、峰值 29820.0 MiB、decode 199.644112 tok/s，但 e2e generated 3.258567 tok/s 低于 opt013 3.268888 tok/s；因此不保留为 Pareto 改进。

## 证据
- Raw artifacts: `attempts/opt_attempt_024_async_wave_batch_access_overlap/raw/`
- Outcome: `attempts/opt_attempt_024_async_wave_batch_access_overlap/OUTCOME.json`
- Source diff: `attempts/opt_attempt_024_async_wave_batch_access_overlap/SOURCE_DIFF.patch`
