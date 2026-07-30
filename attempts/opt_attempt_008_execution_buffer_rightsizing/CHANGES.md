# opt008 execution buffer right-sizing

## 变更

- 在 `cache_hub/retroinfer_cache.py` 新增默认保持 4.0 的 `RETROINFER_BUFFER_NPROBE_MULTIPLIER`，用于把原有 `(nprobe + nprobe_new) * 4` execution-buffer pages 公式变为可调；unset/empty 时保持默认行为，非有限或 `<=0` 值直接报错。
- 在 `block_cache_metadata()` 和 `throughput_eval/reproduce_block_cache.py` 中记录 `buffer_nprobe_multiplier`、`buffer_pages`、`buffer_min_pages`、`buffer_probe_pages`、`buffer_retrieval_clusters`、`execution_stride`，使 raw logs、`results.jsonl/csv`、`summary.csv` 和 attempt summary 都能复核执行缓冲区尺寸。
- 未触碰 CUDA/C++ 源码，未做 CUDA/C++ rebuild；环境审计仍将 source rebuild 分类为 CUDA 12.4 nvcc/CUDA_HOME blocker。

## 实验口径

固定 `120000x8`、`gen_len=100`、`cache_ratio=0.05`、`retrieval_budget=0.018`、`estimation_budget=0.232`、seed 2025、GPU0 A100 80GB；保留 `RETROINFER_ASYNC_CLUSTER_ID_COPY=1` 和低 6 层容量缩放 `1=0.75,2-3=0.75,25=0.75,28-29=0.75`。基线复用 `research/BASELINE_RESULT.json`，验证参考复用 `research/VALIDATION_RESULT.json`，未重跑健康 baseline。

## 结果

| candidate | multiplier | buffer pages | execution stride | block cache GiB | peak GPU MiB | decode tok/s | e2e tok/s | correctness | buffer-capacity warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `buffer_mult_3p0` | 3.0 | 804 | 6599 | 5.5400 | 41604 | 200.5168 | 3.2815 | pass | 0 |
| `buffer_mult_2p0` | 2.0 | 536 | 4455 | 5.5400 | 41536 | 200.1177 | 3.2758 | pass | 160 |
| `buffer_mult_1p0` | 1.0 | 268 | 2311 | 5.5400 | 41472 | 114.2990 | 3.2316 | pass | 168258 |

`buffer_mult_3p0` 是保留候选：相对 canonical baseline，block cache 仍为低 6 层方案的 5.5400 GiB（-4.6875%），峰值进程 GPU 显存 41604 MiB（-356 MiB），decode 为 200.5168 tok/s（+0.845%），e2e 为 3.2815 tok/s（+0.136%）。相对 validated low6+async，block cache 不变，峰值 GPU 显存少 68 MiB，e2e 高 0.052%，decode 低 0.330%（仍高于 canonical baseline）；本轮是单轮有界测量，不声称 profiler overlap。

`buffer_mult_2p0` 进一步降到 536 buffer pages，峰值 GPU 显存比 `buffer_mult_3p0` 再低 68 MiB，但 e2e 低于 validated low6+async 且 raw log 有 160 条 retrieved-pages buffer-capacity warning；`buffer_mult_1p0` 虽继续降显存但 decode 明显回退，并有 168258 条 retrieved-pages buffer-capacity warning，作为负结果保留。

## 证据路径

- 环境审计：`attempts/opt_attempt_008_execution_buffer_rightsizing/environment_audit.json`、`environment_audit.log`
- 候选原始日志/结果/显存采样：`attempts/opt_attempt_008_execution_buffer_rightsizing/buffer_mult_{3p0,2p0,1p0}_120k_b8_cr0p05/`
- 结构化汇总：`attempts/opt_attempt_008_execution_buffer_rightsizing/candidate_summary.json`、`candidate_summary.csv`
- 源码 diff：`attempts/opt_attempt_008_execution_buffer_rightsizing/source_diff.patch`
- 结论：`attempts/opt_attempt_008_execution_buffer_rightsizing/OUTCOME.json`
