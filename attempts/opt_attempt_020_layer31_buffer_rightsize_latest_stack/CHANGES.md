# opt020 layer31 buffer rightsize latest stack

## 变更

- 未改动源代码；本轮只使用现有 opt013/latest Python 运行时开关。
- 复用 opt013 的 late block-cache allocation、uninitialized late init、pinned side-stream index metadata migration、async cluster-id copy 栈。
- 测量两个有界候选：`RETROINFER_LAYER_CACHE_CAPACITY_SCALE=1=0.75,2-3=0.75,25=0.75,28-29=0.75,31=0.75`，分别设置 `RETROINFER_BUFFER_NPROBE_MULTIPLIER=2.75` 和 `2.5`。

## 结果

| candidate | block cache MiB | peak GPU MiB | decode tok/s | e2e tok/s | buffer pages | overflow warnings | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opt013 reference | 5673.0 | 29884.0 | 198.036572 | 3.268888 | 804 | 0 | reference |
| buffer_mult_2p75 | 5626.5 | 29820.0 | 197.748459 | 3.269961 | 738 | 0 | fail: decode regression |
| buffer_mult_2p5 | 5626.5 | 29804.0 | 195.265572 | 3.259166 | 670 | 0 | fail: decode/e2e regression |

## 结论

两个候选都 returncode=0、raw `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth=true` 且无 buffer-overflow warning，并降低 block cache 与进程峰值 GPU 显存；但没有候选满足相对 opt013 的 decode/e2e 无回归保留门槛，因此不触发两轮复测，不保留。

## 证据

- Frontier: `frontier_check.raw.txt`
- Environment audit: `ENV_AUDIT.raw.txt`
- Raw runs: `raw/buffer_mult_2p75_120k_b8_cr0p05_r1/`, `raw/buffer_mult_2p5_120k_b8_cr0p05_r1/`
- Structured summary: `candidate_screen_results.json`, `OUTCOME.json`
- Source provenance: `SOURCE_PROVENANCE.json`, `SOURCE_DIFF.patch`
