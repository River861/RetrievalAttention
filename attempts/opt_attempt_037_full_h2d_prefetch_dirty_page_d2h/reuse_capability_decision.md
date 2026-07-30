# opt037 reuse/capability decision

- Project `.venv`: reused for runtime work. Live preflight used `/home/v-xuchuanluo/RetroInfer/.venv/bin/python` (`Python 3.11.15`, symlink target `/usr/bin/python3.11`) with `torch==2.5.1+cu124`, `torch.version.cuda==12.4`, CUDA available on `NVIDIA A100 80GB PCIe` (cc 8.0).
- Installed RetroInfer kernels: reused unchanged; imports passed for `ThreadPool`, `WaveBufferCPU`, `gather_copy_and_concat`, `gather_copy_and_scatter`, `gather_copy_vectors`, and `batch_gemm_softmax`.
- Pinned memory: small pinned allocation smoke test passed (`4096` bytes, `is_pinned()==true`). Host facts at preflight: `MemTotal=226765912 kB`, `MemAvailable=219653176 kB`, `RLIMIT_MEMLOCK soft/hard=29026033664` bytes.
- CUDA 12.4 rebuild blocker: source rebuild remains blocked and is not an idea failure. `CUDA_HOME` is unset, `torch.utils.cpp_extension.CUDA_HOME` resolves to `/usr`, `/usr/bin/nvcc` is CUDA 12.0, `/usr/local/cuda-12.4/bin/nvcc` and `/usr/local/cuda-12/bin/nvcc` are absent, and `/usr/local/cuda-13.3/bin/nvcc` is CUDA 13.3.
- opt035 reuse decision: opt035 is reused only as a correctness/mechanism base for CPU full-KV ownership, fixed `<=m` GPU slot addresses, full-layer async H2D prefetch, D2H/H2D streams/events, generation ownership, and telemetry shape. Its measured speed result is not reused as a speed claim because opt035 was performance-refuted on the canonical 120K/batch8/cache_ratio=5%/gen_len=100 run (`25.0386 tok/s` decode vs `198.8364 tok/s` baseline).
- opt036 reuse decision: opt036 provides the audited dirty-page D2H implementation over `WaveBuffer.update_cache_indices`, but its page-H2D behavior and performance result are not reused for opt037 because this mission requires full-layer H2D prefetch and zero page-H2D transfers.

