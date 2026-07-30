# opt029：stream-only layer31 low-hit screen

## 变更范围

本尝试不修改源码，不新增 benchmark harness，也不触发 CUDA/C++ rebuild。候选只通过现有环境变量启用项目内已有机制：`RETROINFER_STREAM_ONLY_LAYERS=31` 让第 31 层 block cache 页数变为 0，并保留 opt013 的 late allocation、uninitialized late block cache、pinned side-stream metadata migration、async cluster-id copy、low6 capacity scale 和 `RETROINFER_BUFFER_NPROBE_MULTIPLIER=3.0`。

## 测量口径

参考基线复用 `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/OUTCOME.json`；没有重跑 healthy baseline 或 Figure 13。两次 screen 均使用 canonical `120000x8`、`cache_ratio=0.05`、`gen_len=100`、默认 `use_cuda_graph=true` harness，并以 raw `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth` 作为正确性门槛。

## 结果

| candidate | block cache MiB | peak GPU MiB | decode tok/s | e2e gen tok/s | raw correctness | warnings | decision |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| opt013 reference | 5673.0 | 29884.0 | 198.036572 | 3.268888 | pass | 0 | retained reference |
| C1 stream-only layer31 | 5487.0 | 29696.0 | 190.548259 | 3.271099 | pass | 0 | fail: decode regression |
| C2 stream-only layer31 + async-wave | 5487.0 | 29696.0 | 177.335506 | 3.255615 | pass | 0 | fail: decode/e2e regression |

C1 使 block cache 降低 186.0 MiB、peak process GPU memory 降低 188.0 MiB，且 e2e 未回退，但 decode 比 opt013 低 7.49 tok/s，因此只触发允许的 C2；C2 的 async-wave 没有恢复吞吐，decode 与 e2e 均回退。没有候选满足晋级门槛，因此未运行 exact two-round confirmation。

## 结论

`RETROINFER_STREAM_ONLY_LAYERS=31` 作为 layer31 block-cache residency removal 机制在本口径下是负结果：内存收益明确但吞吐 Pareto 不优于 opt013。
