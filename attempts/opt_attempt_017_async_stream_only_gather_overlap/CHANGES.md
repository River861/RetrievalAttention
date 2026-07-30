# opt017 async stream-only gather overlap

## 实现改动

- 新增默认关闭的 `RETROINFER_ASYNC_STREAM_ONLY_GATHER`。未设置或为 `0/off/false` 时，stream-only 层继续使用原有 inline gather；API 和默认行为不变。
- eager 路径中，启用后在 top-k 写出 `cI` 后立即把 `gather_copy_cluster_and_concat_fuse` 投递到每设备 side stream，并在 `weighted_flash_decoding` 读取 execution buffer 前用 event wait 排序，从而尝试与 estimation-zone gather/weighted decoding 重叠。
- `--use_cuda_graph` 路径中，启用后把 stream-only gather 从原 `attn_cudagraph` 中拆出为 gather-only CUDA graph；运行时在 side stream replay gather graph，当前流继续 replay estimation graph，然后在 attention-only graph 前 wait。
- `throughput_eval/reproduce_block_cache.py` 增加新环境变量和 telemetry 字段，raw `RETROINFER_RESULT_JSON`/CSV 可证明 async path 是否执行。

## 测量结果（A100, 120000x8, gen_len=100, cache_ratio=0.05, --use_cuda_graph）

| 配置 | 正确性 | block cache MiB | peak GPU MiB | decode tok/s | e2e tok/s(800/latency) | async gather telemetry |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| opt013 reference | pass | 5673.0 | 29884.0 | 198.036572 | 3.268888 | off |
| sync stream-only control | True | 5533.5 | 29744.0 | 192.552511 | 3.269219 | launch=0 |
| async stream-only gather | True | 5533.5 | 29744.0 | 187.173264 | 3.257293 | launch=99, cudagraph=99, sync=99 |

## 结论

async candidate 正确性通过，且因 stream-only layer 2 使 block cache/peak memory 低于 opt013；但 decode 和 e2e throughput 均低于 opt013，并且明显慢于 sync stream-only control。因此该机制在本 bounded 点不保留，不触发 retained-candidate 2-round rerun。

## 限制

- telemetry 证明 side-stream CUDA graph gather path 执行（99 次 launch/sync，pending=0），但未采集 Nsight timeline，因此不声称真实 overlap 幅度。
- 结果只覆盖指定 120000x8/gen_len=100/cache_ratio=0.05 A100 点。
