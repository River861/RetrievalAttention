# Notes

- Original source: successful measured rows from `research/block_cache_single_a100_pressure_final/summary.csv` before cleanup.
- Supplemental source: successful 120K batch 8 cache ratio 2.5% run at `/home/v-xuchuanluo/.argus-skill/copilot-home/session-state/7253a517-eda5-4749-81b1-9df3fa71cead/files/cache_ratio_2p5_run`.
- Supplemental raw log: `/home/v-xuchuanluo/.argus-skill/copilot-home/session-state/7253a517-eda5-4749-81b1-9df3fa71cead/files/cache_ratio_2p5_run/raw_logs/block_cache_single_a100_retroinfer_ctx120000_bsz8_cr0p025_r1.txt`.
- Hardware: one A100 80GB PCIe, `CUDA_VISIBLE_DEVICES=0`.
- Model/task: `gradientai/Llama-3-8B-Instruct-Gradient-1048k`, NIAH, `gen_len=100`, seed 2025.
- Cache ratio means GPU KV block-cache capacity as a fraction of total KV vectors; 5% is the paper/default setting.
- Metrics: throughput is mean successful decode tokens/s; block-cache size is derived from configured cached KV pages; memory is mean peak process GPU memory.
- Included measured configurations: original 120K batch 1/8, 240K batch 1, and 480K batch 1 for cache ratios 0.5%, 5%, and 10%, plus the supplemental 120K batch 8 cache ratio 2.5% row.
- Reproduction entry: `./run.sh` reruns only the supplemental 120K batch 8 cache ratio 2.5% point and writes `experiments/cache_ratio/rerun`; it requires the full model, CUDA stack, and an idle A100.
