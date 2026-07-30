# opt011：split-late index metadata prefill residency

## 环境与约束

- 新审计写入 `ENV_AUDIT.md`：系统 `/usr/bin/python` 未装项目依赖，但项目固定 `.venv/bin/python` 可用，Python 3.11.15，Torch 2.5.1+cu124，Triton 3.1.0，vLLM 0.6.5，FlashAttention 2.7.3，FlashInfer 0.2.4+cu124torch2.5，`retroinfer_kernels`/`weighted_flash_decoding` import OK。
- GPU：单卡 NVIDIA A100 80GB PCIe，driver 580.173.02，compute capability 8.0。独立 audit 看到 `/usr/bin/nvcc` 为 CUDA 12.0，harness `environment.json` 后续记录到 PATH 上的 `nvcc` 为 CUDA 13.3；二者都不是已证明匹配 Torch cu124 的 source rebuild 工具链。本轮 non-goal 是 CUDA/C++ rebuild，因此复用已安装 wheel/extension，不重编 vendor/kernel。
- 前沿绑定见 `FRONTIER_BINDING.md`：PyTorch H2D overlap 仍依赖 pinned host + `non_blocking=True` + side stream；CUDA Graph 仍要求 replay tensor 地址稳定。本轮因此不走新的 host staging 分支，而是验证 prefill 阶段 GPU 常驻 index metadata 是否能减少 late mode 的 e2e 损失。

## 代码改动

- `cache_hub/retroinfer_cache.py`
  - 新增 `RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY`，默认 unset 等价 `cpu`，保持 forced-late 旧行为：metadata 先在 CPU，`prepare_cache()` 再迁移到 GPU。
  - 当显式设为 `gpu` 且 block cache allocation policy 为 `late` 时，只让 `centroids/value_sum/centroids_mask/cluster_size` 等 index metadata 在 prefill 阶段直接分配到 GPU；`cache_keys/cache_values` 和 computation buffer 仍在 `prepare_cache()` 才分配。
  - 增加 metadata residency 计量字段：prefill/current GPU bytes、CPU bytes、host pinned bytes，以及 effective reason。
- `throughput_eval/reproduce_block_cache.py`
  - 将新 env 纳入 raw log/config 捕获。
  - 将新 metadata residency 字段加入 CSV/summary 字段，保证 canonical harness 原始结果可复核。

## 验证命令

```bash
RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY=late \
RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY=gpu \
RETROINFER_ASYNC_CLUSTER_ID_COPY=1 \
RETROINFER_LAYER_CACHE_CAPACITY_SCALE='1=0.75,2-3=0.75,25=0.75,28-29=0.75' \
RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0 \
.venv/bin/python throughput_eval/reproduce_block_cache.py \
  --suite opt011_split_late_index_prefill_residency \
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
  --output-dir attempts/opt_attempt_011_split_late_index_prefill_residency/raw/harness_120k_b8_cr0p05
```

## 结果

两轮 120000x8 均通过 NIAH groundtruth 检查。原始 artifacts 在 `raw/harness_120k_b8_cr0p05/`，结构化结论在 `OUTCOME.json`。

| 指标 | opt011 |
|---|---:|
| block cache | 5673.0 MiB |
| index metadata prefill GPU residency | 7,847,116,800 bytes = 7483.59375 MiB |
| peak process GPU memory mean/max | 35610.0 / 35610.0 MiB |
| decode throughput mean | 198.562825 tok/s |
| e2e latency mean | 244.172720 s |
| e2e generated throughput | 3.276369 tok/s |

对比 `research/BASELINE_RESULT.json`：block cache 少 279.0 MiB，进程峰值少 6350.0 MiB（-15.13%）；decode 低 0.14%，e2e generated throughput 低 0.02%，属于近似持平，不能宣称速度提升。

对比 `research/VALIDATION_RESULT.json` / opt010：block cache 相同；因为 metadata prefill 常驻 GPU，峰值多 5726.0 MiB（+19.16%）；decode 低 0.011%，但 e2e latency 少 1.094 s，e2e generated throughput 高 0.447%。结论是：split-late GPU metadata 回收了 opt010 的一部分 e2e 损失，同时仍显著低于 baseline 峰值，但不在 memory 上支配 opt010，也没有 decode speedup。

## 限制

- 只按任务要求跑了 120000x8、cache_ratio=0.05、2 rounds；未扩展 Figure 13 或额外矩阵。
- 本方案把 index metadata 的约 7.31 GiB 重新放回 GPU prefill residency，适合作为“低于 baseline 峰值但更接近 baseline e2e”的中间 Pareto 点，而不是 opt010 的严格替代。
