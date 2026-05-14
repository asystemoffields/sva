# Refreshed Summon-Verify Attention Artifact

This folder contains an SVA artifact bundle with calibration-refreshed identity coarse codebooks.

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Profile: `sva-smollm2-135m-2x256-longctx-refresh-v1`
- Base profile: `sva-smollm2-135m-2x256-v1`
- Context length: `32768`
- Refresh method: `calibration_identity_kmeans`
- Route source: `qk`
- Rank dim: `64`
- Coarse code: `2x256`
- Default shortlist/budget: `2048/512`
- Layers: `30`

Files:

- `sva_config.json`: artifact metadata and serving defaults.
- `sva_artifacts.pt`: per-layer low-rank projections, logit scales, and refreshed coarse codebooks.

The loader is `sva/artifacts.py`.
