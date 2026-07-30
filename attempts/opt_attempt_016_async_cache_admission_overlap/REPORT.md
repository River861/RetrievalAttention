# opt016：异步 block-cache admission overlap

本轮实现了默认关闭的 `RETROINFER_ASYNC_CACHE_ADMISSION`。打开后，`wave_buffer.sync()` 完成 CPU 侧 update/LRU 后，`gather_copy_and_scatter`（`--use_cuda_graph` 下为已有 `update_cudagraphs[layer].replay()`）被提交到每个 device 的 side stream；在同 device 下一次执行 buffer 写入或 cache read 前，用 current stream `wait_event` 显式排序，避免 shared execution buffer 被过早覆盖。

环境复核显示 `.venv`/cu124/A100 运行栈可用，`retroinfer_kernels` 符号可导入，且小型 CUDAGraph 可在 side stream replay 并通过 `wait_event` 排序。CUDA/C++ source rebuild 仍因缺少已证明的 CUDA 12.4 `nvcc`/`CUDA_HOME` 阻塞，未尝试 rebuild。

| candidate | correctness | block cache MiB | peak GPU MiB | decode tok/s | e2e latency s | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| opt013 capacity + async admission | pass | 5673.0 | 29884.0 | 195.417338 | 243.845076 | decode 低于 opt013，不保留 |
| `1=0.75,2-3=0.75,25=0.625,28-29=0.625` + async admission | pass | 5602.5 | 29812.0 | 192.108815 | 245.498946 | 显存更低但 decode/e2e 均低于 opt013，不保留 |

结论：该 Python/runtime side-stream admission 机制在目标 A100 120000x8 / gen_len=100 / cache_ratio=0.05 配置下正确性通过，但没有恢复低显存 schedule 的吞吐；按 retention gate 记录为 bounded negative result。没有运行 Figure 13、没有重跑健康 baseline，也没有对失败 candidate 做 2-round rerun。
