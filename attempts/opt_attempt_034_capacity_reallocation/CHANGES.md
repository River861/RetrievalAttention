# opt034 capacity reallocation

## 改动

- 未修改源代码；复用 opt013 的 late/uninitialized block-cache allocation、pinned side-stream index metadata migration、async cluster-id copy，以及现有 `RETROINFER_LAYER_CACHE_CAPACITY_SCALE`。
- C1: `3=0.75,8=0.875,12-13=0.875,15-16=0.875,25=0.75,28-29=0.75`。
- C2: `2-3=0.75,8=0.875,12-13=0.875,25=0.75,28-29=0.75`。

## 结果

| candidate | correctness | use_cuda_graph | buffer warnings | block cache MiB | peak GPU MiB | decode tok/s | e2e generated tok/s | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| opt013 ref | pass | true | 0 | 5673.0 | 29884.0 | 198.036572 | 3.268888 | retained reference |
| C1 | true | true | 0 | 5651.0 | 29860.0 | 188.805696 | 3.272616 | reject: decode regression |
| C2 | true | true | 0 | 5650.5 | 29860.0 | 194.760313 | 3.262294 | reject: decode/e2e regression |

C1 and C2 both lowered block-cache and peak GPU memory versus opt013 and passed raw `RETROINFER_OUTPUT_JSON.all_outputs_contain_groundtruth == true`, but neither met the decode no-regression gate. C2 also missed e2e no-regression. No two-round confirmation was run.
