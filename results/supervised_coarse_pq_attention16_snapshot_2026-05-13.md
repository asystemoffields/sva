# Supervised Coarse PQ Attention-Label Snapshot - 2026-05-13

## Run

- Commit: `5eb10b5`
- Modal app: `sva-supervised-coarse-pq-attention16-h100`
- Function call: `fc-01KRHYGGX2S4FN6M5EERBEVJ6J`
- Dashboard: https://modal.com/id/fc-01KRHYGGX2S4FN6M5EERBEVJ6J
- Full log: `results/modal_runs/sva-h100-supervised-coarse-pq-attention16-20260513-203442.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage: learned rank-64 scorer, `16x256` fine PQ
- Coarse supervision target: attention top-16 keys

## Aggregate Results

| Method | Coarse rank | Coarse PQ | Shortlist | Top-16 recall |
| --- | ---: | --- | ---: | ---: |
| exact ranker | 0 | none | 512 | 0.839332 |
| full fine PQ | 0 | none | 512 | 0.802021 |
| unsupervised coarse-to-fine | 64 | 4x64, 24 bits/key | 4096 | 0.800967 |
| unsupervised coarse-to-fine | 64 | 8x16, 32 bits/key | 4096 | 0.800022 |
| unsupervised coarse-to-fine | 64 | 4x64, 24 bits/key | 2048 | 0.792085 |
| unsupervised coarse-to-fine | 64 | 4x64, 24 bits/key | 1024 | 0.764881 |
| supervised coarse-to-fine | 64 | 4x64, 24 bits/key | 4096 | 0.802688 |
| supervised coarse-to-fine | 64 | 8x16, 32 bits/key | 4096 | 0.802502 |
| supervised coarse-to-fine | 32 | 4x64, 24 bits/key | 4096 | 0.801944 |
| supervised coarse-to-fine | 64 | 4x64, 24 bits/key | 2048 | 0.797464 |
| supervised coarse-to-fine | 32 | 4x64, 24 bits/key | 2048 | 0.794813 |
| supervised coarse-to-fine | 64 | 4x64, 24 bits/key | 1024 | 0.769128 |

## Interpretation

Sharp supervision recovers the coarse stage. The broad top-512 fine-PQ target was diffuse; attention top-16 labels make the trained coarse ranker competitive with the unsupervised coarse PQ code.

The gain is real but small. At shortlist `4096`, supervised `4x64` reached `0.802688`, just above full fine PQ's `0.802021` and above the unsupervised `4x64` coarse-to-fine row at `0.800967`. At shortlist `2048`, supervised `4x64` reached `0.797464` versus `0.792085` unsupervised. At shortlist `1024`, it reached `0.769128` versus `0.764881` unsupervised.

The result says target sharpness matters, while separate coarse-ranker supervision alone is a refinement rather than the next large jump. The next risk to test is whether the coarse code should be optimized inside the fine-ranker space itself: learn code assignments or coarse centroids for candidate survival, then let the existing `16x256` fine PQ handle the final shortlist ordering.
