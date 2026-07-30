# opt012 frontier binding

Fresh optimize-stage record: `research/frontier/optimize.json`, record_id `1faa0539da74a3a6`.

机制触发：从 opt011 的 prefill GPU metadata residency 转向 opt012 的 forced-late pinned CPU metadata + one-time before-decode migration，因此刷新 pinned memory、non_blocking transfer、side stream 和 overlap claim 边界。

绑定结论：

1. 官方 CUDA/PyTorch 语义仍要求 pinned/page-locked host memory、`non_blocking=True`、非默认 stream 和显式同步，才具备异步 H2D transfer 的机制条件。
2. 仅有代码路径、copy bytes 和 elapsed time 不能证明 transfer/compute overlap；没有 profiler/timeline，本轮不声明 overlap 或 overlap-derived speedup。
3. opt009 已经否定 per-decode metadata staging；opt012 只做一次 late migration，目标是保留 opt010 低 peak memory 并降低 opt010 prepare/decode 前 metadata migration cost。
