# opt033 page-geometry compensated cache-ratio screen

## 变更

本轮没有改源码；只使用现有默认关闭旋钮做有界实验。配置保持 opt013 stack：late block-cache allocation、`RETROINFER_LATE_BLOCK_CACHE_INIT=uninitialized`、`RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY=pinned_side_stream`、`RETROINFER_ASYNC_CLUSTER_ID_COPY=1`、低 6 层 capacity scale、`RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`。在此基础上固定 `RETROINFER_PAGES_PER_CLUSTER_OVERRIDE=1` 和 `RETROINFER_BUFFER_PAGES_PER_CLUSTER_FLOOR=2`，筛选 `cache_ratio=0.075`，若吞吐仍负再筛选 `0.0875`。

## 结果

opt013 retained reference：block cache 5673.0 MiB、peak process GPU memory 29884 MiB、decode 198.0366 tok/s、e2e generated 3.2689 tok/s。

- C1 `cache_ratio=0.075`：正确性通过、`use_cuda_graph=true`、zero WaveBuffer max-consider warnings、block cache 4254.0 MiB、peak 28438 MiB；但 decode 185.9305 tok/s，e2e generated 约 3.2687 tok/s，均低于 opt013。
- C2 `cache_ratio=0.0875`：正确性通过、`use_cuda_graph=true`、zero WaveBuffer max-consider warnings、block cache 4963.5 MiB、peak 29154 MiB；但 decode 192.0930 tok/s，e2e generated 约 3.2609 tok/s，仍低于 opt013。

## 结论

提高 page=1 的 cache ratio 可以部分恢复 decode（C2 优于 C1），并且仍低于 opt013 的 block-cache 和峰值显存；但两个一轮候选都没有达到“decode/e2e 不回退”门槛，因此不运行 2-round confirmation，不作为 Pareto 改善保留。负结果指向：page=1 的吞吐损失不是单纯 cache residency 数量不足，后续应优先看 page=1 下的访问粒度、hit/miss 分布、WaveBuffer batch access 和 gather/copy 开销。

