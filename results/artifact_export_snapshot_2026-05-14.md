# Artifact Export Snapshot - 2026-05-14

This snapshot records the first local deployable SVA artifact bundle.

## Artifact

- Profile: `sva-smollm2-135m-2x256-v1`
- Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Local path: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Modal volume: `sva-artifacts`
- Remote path: `/sva-smollm2-135m-2x256-v1`
- Weights file: `sva_artifacts.pt`
- Manifest file: `sva_config.json`
- Artifact size: about `13.3 MB`

## Contents

- 30 layers
- Route source: Q/K
- Rank dim: `64`
- Coarse code: `2x256`
- Default shortlist/budget: `2048/512`
- Tensor dtype: `bfloat16`

## Verification

The exporter trained all 30 layers on H100, saved the bundle, and reloaded it before exiting. The downloaded local bundle was then reloaded with `load_sva_artifact_bundle`:

```text
profile sva-smollm2-135m-2x256-v1
layers 30
shape 2 256
default 2048 512
layer0_q_proj (9, 64, 64) torch.bfloat16
layer0_codebooks (9, 2, 256, 32) torch.bfloat16
```

## Publication Notes

The folder is ready to publish as a small Hugging Face artifact repo or as a GitHub release asset. The repo-side loader lives at `experiments/sva_artifact_io.py`, and the exporter lives at `experiments/export_sva_artifact.py`.
