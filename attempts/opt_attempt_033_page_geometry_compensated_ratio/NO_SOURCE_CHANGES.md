# opt033 source note

opt033 made no source edits. It reused the already-present default-off runtime knobs from opt031/opt032:

- `RETROINFER_PAGES_PER_CLUSTER_OVERRIDE=1`
- `RETROINFER_BUFFER_PAGES_PER_CLUSTER_FLOOR=2`
- opt013 late/uninitialized block-cache allocation and pinned side-stream index-metadata migration stack

The worktree was already dirty before opt033. The measured source provenance is recorded in `SOURCE_PROVENANCE.json`; implementation provenance for the reused page-geometry and buffer-floor knobs is in opt032 artifacts.

