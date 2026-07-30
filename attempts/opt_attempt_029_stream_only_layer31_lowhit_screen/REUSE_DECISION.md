# Reuse decision

Generated before any opt029 measurement.

- Reuse the project-pinned runtime: `/home/v-xuchuanluo/RetroInfer/.venv/bin/python`.
- Runtime proof: Python 3.11.15, torch 2.5.1+cu124, CUDA runtime 12.4, `torch.cuda.is_available() == true`, GPU0 NVIDIA A100 80GB PCIe, and installed `retroinfer_kernels` symbols import successfully.
- Do not rebuild CUDA/C++ or edit native WaveBuffer code in this attempt. The audited compiler remains `/usr/bin/nvcc` CUDA 12.0, not a proven CUDA 12.4-compatible toolchain for this cu124 project.
- Execute opt029 as a no-source, environment-gated screen using the existing `RETROINFER_STREAM_ONLY_LAYERS`, per-layer capacity-scale, late block-cache allocation, pinned-side-stream metadata migration, and async-cluster-id-copy paths.
- Reuse `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/OUTCOME.json` as the retained reference; do not rerun the healthy baseline or Figure 13.
