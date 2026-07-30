# opt037 changes

Implemented a default-off opt-in slot-rotation mode:

- New env: `RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DIRTY_PAGE_D2H=1`.
- Required base env: `RETROINFER_BLOCK_CACHE_SLOT_ROTATION=1`.
- Mutually exclusive with opt036's `RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DELTA=1`.
- New telemetry copy mode: `block_cache_slot_rotation_copy_mode="full_h2d_dirty_page_d2h"`.
- H2D path: preserves opt035 full-layer H2D wait/prefetch and CUDA graph fixed slot addresses.
- D2H path: after layer admission, reuses the dirty-page flush path driven by `WaveBuffer.update_cache_indices`.
- Opt036 page-H2D materialization remains gated only by `RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DELTA=1`, so the new mode should report zero page-H2D transfers.
- `throughput_eval/reproduce_block_cache.py` now propagates the new mode/env telemetry fields.

Build check:

```bash
.venv/bin/python -m py_compile cache_hub/retroinfer_cache.py throughput_eval/reproduce_block_cache.py
```

