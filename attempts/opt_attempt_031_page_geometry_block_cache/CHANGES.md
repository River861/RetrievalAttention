# opt031 page-geometry block cache

## 代码改动

- 在 `config/config.py` 增加默认关闭的 `RETROINFER_PAGES_PER_CLUSTER_OVERRIDE`。未设置或空值保持 `pages_per_cluster=2`；设置正整数时覆盖为该值；`0`、负数或非整数会显式抛出 `ValueError`。
- 在 `throughput_eval/reproduce_block_cache.py` 记录该环境变量，并把 `pages_per_cluster` 的默认值、override env、是否生效和来源写入 block-cache 结果字段，便于结果元数据复核。

## 验证结果

- `raw/logs/config_override_preflight.log` 证明：unset 为 `2/config_default`，override `1` 为 `1/env_override`，invalid `0` 明确失败。
- 120000x8、cache_ratio=0.05、复用 opt013 stack 的 base page-geometry 一轮通过正确性，block cache 从 5673.0 MiB 降到 2836.5 MiB，峰值进程 GPU 显存从 29884.0 MiB 降到 28010.0 MiB；但 decode 从 198.04 降到 145.66 tok/s，e2e generated 从 3.2689 降到 3.2565 tok/s，且有 984 条 buffer warning。
- recovery 候选（page=1 + async wave + scratch uninitialized + buffer 2.75）同样通过正确性并保持 2836.5 MiB / 28010.0 MiB，但 decode 仅 146.63 tok/s，e2e generated 3.2431 tok/s，buffer warning 增至 1451 条。

## 结论

`pages_per_cluster=1` 的机制真实降低了 block cache 和进程峰值显存，但在当前 120000x8/0.05 口径下造成大量 buffer overflow warnings 和明显 decode/e2e 吞吐回退；未满足 confirmation gate，因此没有运行 2-round confirmation。该分支作为负结果保留，不能作为 Pareto 改善声明。
