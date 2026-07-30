# opt021 no-source-change note

- opt021 did not edit repository source code.
- Measurements used the existing dirty opt013/latest Python stack captured in `SOURCE_PROVENANCE.json` and `SOURCE_DIFF.patch`.
- Exercised surface: existing `RETROINFER_LAYER_CACHE_CAPACITY_SCALE` in `cache_hub/retroinfer_cache.py` with the canonical `throughput_eval/reproduce_block_cache.py` harness.
