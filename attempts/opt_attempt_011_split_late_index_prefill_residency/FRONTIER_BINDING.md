# opt011 frontier binding

机制触发：本轮从 forced-late block-cache allocation 转向“late KV/cache buffer + prefill GPU index metadata”的分离驻留机制，因此刷新与 H2D/stream overlap、CUDA graph 地址稳定性相关的前沿约束。

绑定结论：

1. PyTorch 官方 pinned-memory/non_blocking 指南和近期 stream overlap 讨论仍要求：真正可重叠的 H2D 传输需要 pinned host tensor、`non_blocking=True`，并显式放到 side stream；否则默认流上通常会串行化。参考：<https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>、<https://discuss.pytorch.org/t/how-to-properly-run-cuda-ops-asynchronously-across-multiple-streams-in-pytorch/224120>。
2. PyTorch CUDA Graph 仍要求 replay 时参与图的 tensor 地址和 shape 稳定；推荐预分配固定 GPU buffer，再在 replay 前 copy 数据。参考：<https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html>、<https://github.com/pytorch/pytorch/wiki/CUDA-graph-best-practice>。
3. 对本代码路径的约束：opt010 的 CPU metadata + prepare/decode H2D/stage 路线有机会节省峰值，但会把 metadata 传输和 graph/stage buffer 地址管理放到关键路径。opt011 因此选择一个更小的机制 pivot：只在 forced-late 时允许 index metadata 在 prefill 阶段直接常驻 GPU，保持 CUDA graph 捕获看到稳定 metadata tensor，同时继续把 GPU block-cache KV 和 computation buffer 推迟到 `prepare_cache()`。

本轮没有重写 vendor kernels、CUDA/C++ extension 或 harness；机制完全绑定到 `cache_hub/retroinfer_cache.py` 的 metadata allocation/prepare path 和 `throughput_eval/reproduce_block_cache.py` 的 metadata capture。
