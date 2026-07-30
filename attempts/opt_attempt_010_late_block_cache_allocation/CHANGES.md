# opt010 late block cache allocation

## 变更

- 在 `cache_hub/retroinfer_cache.py` 新增默认保持 `auto` 的 `RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY`：unset/empty/default/auto 继续使用原先 `free_memory > estimated_gpu_memory * 1.5` 的预分配决策；`late`/`after_prefill` 强制走已有 post-prefill `prepare_cache()` 分配路径；`preallocate` 强制 prefill 前分配。
- 将分配策略、决策、free memory、estimated GPU memory、auto 判定、是否 prefill 前已分配、是否 prepare_cache 后分配等字段纳入 `block_cache_metadata()`，并扩展 `throughput_eval/reproduce_block_cache.py` 的 block-cache 字段与环境记录。
- 没有改 CUDA/C++ 源码、没有重建 vendor/kernel 库、没有复用 opt009 的 whole-layer index metadata staging 作为主机制。

## 实验口径

复用 retained `low6+async+buffer_mult3.0` 配置：`120000x8`、`gen_len=100`、`cache_ratio=0.05`、`retrieval_budget=0.018`、`estimation_budget=0.232`、seed 2025、GPU0 A100 80GB、CUDA graph 开启；叠加 `RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late`。baseline 使用 `research/BASELINE_RESULT.json`，当前优化参考使用 `research/VALIDATION_RESULT.json`，未重跑健康 baseline，未运行 Figure 13。

## 结果

| candidate | correctness | allocation decision | preallocated before prefill | block cache MiB | peak GPU MiB | decode tok/s | e2e tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| forced_late_low6_async_buffer_mult_3p0 | pass | allocate_after_prefill | False | 5673.0 | 29884.0 | 198.585 | 3.261779 |

相比 `research/VALIDATION_RESULT.json`：峰值进程 GPU 显存 -11720.0 MiB (-28.17%)，block cache bytes 不变，decode throughput -0.014 tok/s (-0.01%)，e2e throughput -0.023278 tok/s (-0.71%)。

## 证据路径

- 环境审计：`attempts/opt_attempt_010_late_block_cache_allocation/environment_audit.txt`、`environment_audit.json`
- Frontier binding：`attempts/opt_attempt_010_late_block_cache_allocation/frontier_binding.json`
- 源码 diff：`attempts/opt_attempt_010_late_block_cache_allocation/source_diff.patch`
- 原始 120000x8 结果：`attempts/opt_attempt_010_late_block_cache_allocation/late_allocation_low6_async_buffer3_120k_b8_cr0p05/`
- 结构化结果：`attempts/opt_attempt_010_late_block_cache_allocation/candidate_summary.json`、`candidate_summary.csv`、`OUTCOME.json`
