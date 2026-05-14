# Summon-Verify Attention Artifact

This folder contains a frozen SVA artifact bundle.

- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Profile: `sva-smollm2-135m-2x256-v1`
- Context length: `8192`
- Route source: `qk`
- Rank dim: `64`
- Coarse code: `2x256`
- Default shortlist/budget: `2048/512`
- Layers: `30`

Files:

- `sva_config.json`: artifact metadata and serving defaults.
- `sva_artifacts.pt`: per-layer low-rank projections, logit scales, and coarse codebooks.

The loader is `experiments/sva_artifact_io.py`.
