# opt028 async-wave conservative layer31 recovery

## Code changes

- No source edits were made for opt028.
- The attempt reused existing default-off env gates: `RETROINFER_ASYNC_WAVE_BATCH_ACCESS=1` and `RETROINFER_LAYER_CACHE_CAPACITY_SCALE` on the opt013 late/uninitialized/pinned-side-stream/async-cluster-copy stack.

## Measurement contract

- Canonical harness only: `120000x8`, `cache_ratio=0.05`, `gen_len=100`, default `--use_cuda_graph`, A100 GPU0, seed 2025.
- Baseline gate: opt013 from `attempts/opt_attempt_013_uninitialized_late_block_cache_allocation/OUTCOME.json`; no healthy baseline or Figure 13 rerun.

## Result

- `c1_layer31_0p875` (one_round_screen_primary, 1 round): block cache 5650.0 MiB (-23.000 vs opt013), peak 29860.0 MiB (-24.000), decode 198.135517 tok/s (+0.098946), e2e generated 3.271084 tok/s (+0.002197); gate `qualifies_for_two_round_confirmation`, warnings=0, use_cuda_graph=True, raw correctness=True.
- `c2_layer31_0p9375` (extra_screen_not_decision_input, 1 round): block cache 5661.5 MiB (-11.500 vs opt013), peak 29872.0 MiB (-12.000), decode 199.298387 tok/s (+1.261815), e2e generated 3.257793 tok/s (-0.011095); gate `e2e_regressed_vs_opt013`, warnings=0, use_cuda_graph=True, raw correctness=True.
- `c1_layer31_0p875_confirm_2round` (two_round_confirmation, 2 round): block cache 5650.0 MiB (-23.000 vs opt013), peak 29860.0 MiB (-24.000), decode 199.457158 tok/s (+1.420586), e2e generated 3.261647 tok/s (-0.007241); gate `e2e_regressed_vs_opt013`, warnings=0, use_cuda_graph=True, raw correctness=True.

- Final status: `not_retained`; C1 one-round screen qualified, but exact two-round confirmation failed at least one retention gate.
- C2 is preserved as an extra non-decision artifact from the initial local gate-reference bug; C1 and its confirmation are the retained-decision evidence.
