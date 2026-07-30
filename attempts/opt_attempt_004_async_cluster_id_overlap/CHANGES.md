# opt_attempt_004_async_cluster_id_overlap

## 变更

- 在 `cache_hub/retroinfer_cache.py` 增加默认关闭的 `RETROINFER_ASYNC_CLUSTER_ID_COPY`：unset 时仍走原同步 `cluster_ids.copy_`；设置为 `1/true/yes/on` 时，在 top-k 后立刻用独立 CUDA stream 将 GPU `cI[..., :nprobe]` 非阻塞拷贝到 pinned CPU `cluster_ids`。
- 异步路径把 copy 与 estimation-zone `gather_copy_vectors` + `weighted_flash_decoding`（CUDA graph 模式下为 estimation graph replay）重叠，并在两个 sparse attention 路径的 `WaveBufferCPU.batch_access()` 前同步 copy event。
- `block_cache_metadata()`、`results.jsonl`、`config.json`、`commands.jsonl` 和 raw log 现在记录 `RETROINFER_ASYNC_CLUSTER_ID_COPY`、low6 capacity scale 等复现实验环境；未改 CUDA/C++，未 rebuild，未新增 harness。

## 环境与 frontier

- Fresh environment audit: `attempts/opt_attempt_004_async_cluster_id_overlap/environment_audit.log`；`.venv` runtime 为 PyTorch CUDA 12.4 + A100 80GB，`retroinfer_kernels` symbols 在先 import `torch` 后可用。CUDA 12.4 rebuild compiler 仍未证明，所以没有 CUDA/C++ rebuild。
- Fresh optimize frontier binding: `research/frontier/optimize_async_cluster_id_overlap.json` (`sha256=b83d3088fe3f482cf18bf29848c812a736abd84d390543ef4ffec57f5391075b`)，并追加到 `research/FRONTIER_WATCH.jsonl`。

## 120000x8 / gen_len=100 / cache_ratio=0.05 实测结果

| candidate | reference | correctness | block cache GiB | peak GPU MiB | decode tok/s | e2e gen tok/s | decode delta vs ref | e2e delta vs ref | memory delta vs ref |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| async-only | baseline | pass | 5.812500 | 41960 | 200.817331 | 3.289914 | +1.980917 (+0.996%) | +0.012869 (+0.393%) | +0 MiB |
| low6+async | baseline | pass | 5.540039 | 41672 | 199.402850 | 3.272344 | +0.566436 (+0.285%) | -0.004701 (-0.143%) | -288 MiB |
| low6+async | low6 validation | pass | 5.540039 | 41672 | 199.402850 | 3.272344 | +3.310816 (+1.688%) | -0.005338 (-0.163%) | +0 MiB |

结论限于这两次 isolated A100 实测：async-only 不改变 block-cache/峰值显存，decode 比 canonical baseline 均值高；low6+async 保留 low6 的 block-cache 与峰值显存下降，decode 同时高于 canonical baseline 均值和 low6 validation 均值，但按 `batch_size * gen_len / e2e_latency_s` 计算的 e2e generated throughput 略低于 baseline/low6 reference，因此不声明端到端加速或 profiler 级 overlap 证明。

## 原始证据

- async-only: `attempts/opt_attempt_004_async_cluster_id_overlap/async_only_120k_b8_cr0p05`；raw log `attempts/opt_attempt_004_async_cluster_id_overlap/async_only_120k_b8_cr0p05/raw_logs/async_cluster_id_copy_opt004_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`；memory samples `attempts/opt_attempt_004_async_cluster_id_overlap/async_only_120k_b8_cr0p05/memory_samples/async_cluster_id_copy_opt004_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`。
- low6+async: `attempts/opt_attempt_004_async_cluster_id_overlap/low6_async_120k_b8_cr0p05`；raw log `attempts/opt_attempt_004_async_cluster_id_overlap/low6_async_120k_b8_cr0p05/raw_logs/low6_async_cluster_id_copy_opt004_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`；memory samples `attempts/opt_attempt_004_async_cluster_id_overlap/low6_async_120k_b8_cr0p05/memory_samples/low6_async_cluster_id_copy_opt004_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`。
- Structured outcome: `attempts/opt_attempt_004_async_cluster_id_overlap/OUTCOME.json`。

两个执行候选的 harness returncode 均为 `0`，raw log 均包含 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true`。
