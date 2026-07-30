# opt032 page-geometry buffer floor

## 代码改动

- 在 `cache_hub/retroinfer_cache.py` 增加默认关闭的 `RETROINFER_BUFFER_PAGES_PER_CLUSTER_FLOOR`。未设置或空值时，buffer 仍按当前 `pages_per_cluster` 计算；设置正整数时，仅 `buffer_min_pages` 和 `buffer_probe_pages` 使用 `max(pages_per_cluster, floor)`。
- 该 floor 不参与 `compute_retroinfer_block_cache_capacity()`，因此 `RETROINFER_PAGES_PER_CLUSTER_OVERRIDE=1` 仍保持 block-cache pages-per-layer=372、block cache=2836.5 MiB。
- 在 `throughput_eval/reproduce_block_cache.py` 增加结果字段和环境记录：`buffer_pages_per_cluster*`、floor env、是否生效、来源，便于复核 buffer floor 与 block-cache page geometry 被解耦。

## 验证结果

- `raw/logs/buffer_floor_preflight.log` 证明：unset 保持 `pages_per_cluster=2/config_default`；`RETROINFER_PAGES_PER_CLUSTER_OVERRIDE=1` 使 block-cache pages=372；再设置 floor=2 后 block-cache pages 仍为 372，但 buffer formula 从 402 恢复到 804；`0`、`-1`、`abc` 均显式 `ValueError`。
- C1（page=1 + floor=2）在 120000x8/cache_ratio=0.05 一轮通过正确性，`use_cuda_graph=true`，block cache=2836.5 MiB，峰值进程 GPU 显存=28010 MiB，buffer warnings=0；但 decode=145.05 tok/s、e2e generated=3.2522 tok/s，低于 opt013 的 198.04 tok/s、3.2689 tok/s。
- 唯一允许 recovery（只额外设置 `RETROINFER_SCRATCH_BUFFER_INIT=uninitialized`）同样通过正确性、零 warning、block cache=2836.5 MiB、峰值=28010 MiB；decode=145.92 tok/s、e2e generated=3.2435 tok/s，仍低于 opt013。

## 结论

buffer floor 机制成功消除了 opt031 page=1 的 WaveBuffer max-consider warnings，并保留 block-cache 显存减半与峰值显存下降；但吞吐未恢复，未达到 no decode/e2e regression gate，因此不运行 2-round confirmation，不作为 Pareto 改善保留。该结果说明 opt031 的 warning 不是 page=1 吞吐损失的唯一原因，后续若继续应转向 page=1 下的 cache hit/miss、cluster-size 分布或 execution-buffer gather/copy 开销分析。
