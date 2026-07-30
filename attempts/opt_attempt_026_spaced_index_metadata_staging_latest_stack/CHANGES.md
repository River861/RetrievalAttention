# opt_attempt_026_spaced_index_metadata_staging_latest_stack

## Scope

- No source code was changed. This attempt exercises the existing default-off `RETROINFER_STAGE_INDEX_METADATA_LAYERS` path on top of the opt013 stack.
- Candidate layers: `4,9,14,19,24,29`.
- Shape: `context_len=120000`, `batch_size=8`, `gen_len=100`, `cache_ratio=0.05`, `seed=2025`, single A100 80GB.

## Environment/frontier evidence

- Fresh environment proof: `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/ENV_AUDIT.raw.txt`. The project `.venv` imports torch 2.5.1+cu124, vLLM/FlashInfer/FlashAttention, and installed `retroinfer_kernels`; source rebuild remains blocked by missing CUDA 12.4-compatible `nvcc` and was not attempted.
- Fresh frontier note/check: `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/frontier_check.raw.txt`. The bounded pivot reuses opt009 staging prior art and opt013 as the retained reference; no new harness, CUDA/C++ rebuild, Figure 13 rerun, or healthy baseline rerun.

## Screen result

| Metric | opt013 reference | opt026 one-round screen | Delta |
| --- | ---: | ---: | ---: |
| Correctness | pass | pass | -- |
| Block cache | 5673.0 MiB | 5673.0 MiB | 0.0 MiB |
| Peak process GPU memory | 29884.0 MiB | 28696.0 MiB | -1188.0 MiB |
| Decode throughput | 198.036572 tok/s | 117.523710 tok/s | -80.512861 tok/s |
| E2E generated throughput | 3.268888 tok/s | 3.235919 tok/s | -0.032969 tok/s |
| E2E latency | 244.734262 s | 247.224991 s | 2.490729 s |
| Nominal metadata GPU bytes saved | -- | 1226112000 | -- |
| Stage prefetch/sync count | -- | 600 / 600 | -- |

## Decision

Stopped after the one-round screen. Correctness, block cache memory, and peak process memory passed the gate, but decode and e2e throughput regressed versus opt013, so the predeclared exact two-round confirmation was not run and the candidate is not retained.

## Evidence paths

- Raw screen artifacts: `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/raw/screen_120k_b8_cr0p05`
- Raw log: `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/raw/screen_120k_b8_cr0p05/raw_logs/opt026_spaced_index_metadata_stage_screen_retroinfer_ctx120000_bsz8_cr0p05_r1.txt`
- Memory samples: `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/raw/screen_120k_b8_cr0p05/memory_samples/opt026_spaced_index_metadata_stage_screen_retroinfer_ctx120000_bsz8_cr0p05_r1.jsonl`
- Structured outcome: `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/OUTCOME.json`
- Source provenance/no-source note: `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/SOURCE_PROVENANCE.json`, `attempts/opt_attempt_026_spaced_index_metadata_staging_latest_stack/NO_SOURCE_CHANGES.md`
