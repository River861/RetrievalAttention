# Notes

- Source: successful measured rows from `research/block_cache_single_a100_pressure_final/summary.csv` before cleanup.
- Hardware: one A100 80GB PCIe, `CUDA_VISIBLE_DEVICES=0`.
- Model/task: `gradientai/Llama-3-8B-Instruct-Gradient-1048k`, NIAH, `gen_len=100`, seed 2025.
- Cache ratio means GPU KV block-cache capacity as a fraction of total KV vectors; 5% is the paper/default setting.
- Metrics: throughput is mean successful decode tokens/s; block-cache size is derived from configured cached KV pages; memory is mean peak process GPU memory.
- Included measured configurations: 120K batch 1/8, 240K batch 1, and 480K batch 1 for cache ratios 0.5%, 5%, and 10%.
- Reproduction entry: `./run.sh` reruns the configured matrix and writes a fresh output directory; it requires the full model, CUDA stack, and an idle A100.
