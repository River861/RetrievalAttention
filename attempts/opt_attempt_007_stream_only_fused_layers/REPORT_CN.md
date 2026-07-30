# opt007 stream-only fused layers 技术结果

## 实现

新增默认关闭的 `RETROINFER_STREAM_ONLY_LAYERS`。被选中的 offload RetroInfer 层不再分配持久 GPU block-cache tensor，也不进入 `WaveBufferCPU.batch_access()` / LRU admission / `gather_copy_and_scatter`；decode 时直接复用已安装的 `gather_copy_cluster_and_concat_fuse`，从 steady-zone GPU KV 与 organized CPU/list KV 拼接 execution buffer。默认未设置环境变量时路径保持原状。

## 环境

`.venv` 运行时为 Python 3.11.15、torch 2.5.1+cu124、torch CUDA 12.4，GPU0 为 NVIDIA A100 80GB PCIe。`retroinfer_kernels.gather_copy_cluster_and_concat_fuse` 已安装可导入。源码 CUDA/C++ rebuild 仍因未证明 CUDA 12.4 nvcc/CUDA_HOME 而阻塞。

## 验证

低层 probe `logs/stream_only_fused_gather.log` 对 pinned CPU/list KV 输入验证 key/value exact 和 valid_lengths exact，结果 `pass`。三个 120000x8/gen_len=100/cache_ratio=0.05 单轮 candidate 均设置 `RETROINFER_ASYNC_CLUSTER_ID_COPY=1`、telemetry unset，raw log 中均有 `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true` 和 `# returncode: 0`。

## 结果

| candidate | stream-only / capacity | block cache GiB | peak GPU MiB | decode tok/s |
| --- | --- | ---: | ---: | ---: |
| canonical baseline | off / default | 5.812500 | 41960.0 | 198.836414 |
| validated low6+async | off / low6 0.75 | 5.540039 | 41672.0 | 201.179969 |
| stream_low3_async | 2-3,28 / default | 5.267578 | 41416.0 | 178.740008 |
| stream_l28_low6_async | 28 / low6 excluding 28 | 5.403809 | 41532.0 | 188.941983 |
| stream_l2_low6_async | 2 / low6 excluding 2 | 5.403809 | 41532.0 | 192.509438 |

## 结论

该实现真实降低了持久 GPU block cache，并且三个 bounded candidate 都通过正确性；但即使缩小到单层 stream-only，最佳 decode 仍为 `stream_l2_low6_async` 的 192.509 tok/s，低于 baseline 198.836 tok/s 和 validated low6+async 201.180 tok/s。因此 opt007 try budget 已用尽，不保留任何 stream-only candidate；下一步应 replan，而不是进入 validation/report。
