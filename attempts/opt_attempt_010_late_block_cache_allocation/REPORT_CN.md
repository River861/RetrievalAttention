# opt010：强制 late block cache allocation 技术报告

## 摘要

本次实现了一个默认不改变现有行为的运行时控制：`RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY`。默认 `auto` 保留原先 A100 上的自动预分配策略；强制 `late` 时，RetroInfer 在 prefill 阶段不提前分配 GPU block cache/index/compute buffer，而是沿用已有 `prepare_cache()` 生命周期在 prefill 后、decode 前分配。

## 代码证据与机制

现有链路中，`cache_hub/retroinfer_cache.py` 构造函数先计算 `cache_sizes`、`buffer_size`，再调用 `pre_allocate_decision()`。若 `self.allocated=True`，构造期即分配 `cache_keys/cache_values`、index metadata 和 computation buffer；若为 False，则 index metadata 先保留在 CPU，`prepare_cache()` 再把 cache 和 metadata/buffer 搬到 GPU。opt010 没有新增 kernel，而是把这条既有 late allocation 分支显式暴露为 default-off policy，并记录可审计 metadata。

## 实验设置

- 配置：120000x8，cache_ratio=0.05，gen_len=100，retrieval_budget=0.018，estimation_budget=0.232，seed=2025。
- 继承 retained 方案：`RETROINFER_ASYNC_CLUSTER_ID_COPY=1`、`RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`、`RETROINFER_LAYER_CACHE_CAPACITY_SCALE=1=0.75,2-3=0.75,25=0.75,28-29=0.75`。
- 新增变量：`RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late`。
- 对照：baseline 为 `research/BASELINE_RESULT.json`；当前 retained 优化点为 `research/VALIDATION_RESULT.json`。

## 结果

两轮均通过 NIAH groundtruth `9879991` 正确性检查。候选平均峰值进程 GPU 显存为 29884.0 MiB，block cache 为 5673.0 MiB，decode throughput 为 198.585 tok/s，e2e throughput 为 3.261779 tok/s。

相对 retained validation，峰值进程 GPU 显存降低 11720.0 MiB；block cache bytes 不变，说明收益来自生命周期晚分配避免 prefill 峰值，而不是减少 decode 阶段常驻 block cache 容量。decode throughput 基本持平（-0.01%），e2e throughput 低 0.71%。相对原始 baseline，峰值进程 GPU 显存降低 12076.0 MiB，block cache 仍保持 low6 的 5.540 GiB。

## 结论与限制

该机制给出了一个显存-吞吐 Pareto 改善点：在保留 RetroInfer decode block cache 容量和正确性的前提下，显著降低进程峰值 GPU 显存，decode 不退化到 opt009 staging 的负结果。限制是本次只验证 forced-late 一种 bounded candidate；没有 profiler 证据支持 H2D/compute overlap 速度声明；CUDA/C++ rebuild 仍因 CUDA 12.4 编译器不可证而不尝试。
