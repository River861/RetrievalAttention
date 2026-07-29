# Notes

- Source: successful measured rows from `research/fig13_single_gpu_consolidated/paper_cache_peak_throughput.csv` before cleanup.
- Hardware: one A100 80GB PCIe, `CUDA_VISIBLE_DEVICES=0`.
- Model/task: `gradientai/Llama-3-8B-Instruct-Gradient-1048k`, NIAH, `gen_len=100`.
- Metrics: decode throughput is peak successful decode tokens/s over the measured batch grid; memory is peak process GPU memory.
- RetroInfer uses cache ratio 5%, retrieval budget 0.018, estimation budget 0.232. Full_Flash_Attn has no block-cache ratio.
- Comparability: this is a single-GPU local reproduction surface, not the paper's full hardware setting. Only successful measured rows are included. The local 1,024K attempts had no successful decode measurement, so they are not included in `results.csv` or the report.
- Reproduction entry: `./run.sh` reruns the configured measurement and writes a fresh output directory; it requires the full model, CUDA stack, and an idle A100.
