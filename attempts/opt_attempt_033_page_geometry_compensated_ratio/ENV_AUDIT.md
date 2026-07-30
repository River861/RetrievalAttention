# opt033 environment audit

- Project root: `/home/v-xuchuanluo/RetroInfer`
- Git HEAD: `7096eab9190da389bbc75c1992140d1432d9d8ec`
- Python: `.venv/bin/python`, Python 3.11.15
- Runtime packages: torch `2.5.1+cu124`, Triton `3.1.0`, transformers `4.49.0`, numpy `1.26.4`; `cache_hub.retroinfer_cache` imports successfully.
- GPU: one `NVIDIA A100 80GB PCIe`, compute capability 8.0, 85093777408 bytes reported by PyTorch.
- Driver/runtime visibility: `nvidia-smi` driver `580.173.02`, CUDA `13.0`; PyTorch CUDA runtime `12.4`.
- GPU state before measurements: no running GPU processes, 6 MiB used.
- Non-blocker: `cupy` is not installed; the canonical RetroInfer harness and installed torch/Triton/CUDA extension path do not require it for this attempt.

Raw audit log: `attempts/opt_attempt_033_page_geometry_compensated_ratio/logs/env_audit.log`.

