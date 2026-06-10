# Attention-Weighted Coarse PQ Snapshot - 2026-05-13

## Run

- Commit: `ed22795`
- Modal app: `sva-weighted-coarse-pq-h100`
- Modal app id: `ap-UOC6jJrskHdM3bIfTSYDdo`
- Function call: `fc-01KRHYVPJK5FMNWA18P9FWGT0J`
- Dashboard: https://modal.com/id/fc-01KRHYVPJK5FMNWA18P9FWGT0J
- Full log: `results/modal_runs/sva-h100-weighted-coarse-pq-20260513-204047.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage: learned rank-64 scorer, `16x256` fine PQ
- Coarse space: same learned rank-64 fine-ranker space
- Coarse codebook training: weighted k-means with attention top-16 key boosts `4,16,64`

## Aggregate Results

| Method | Coarse PQ | Shortlist | Top-16 recall |
| --- | --- | ---: | ---: |
| exact ranker | none | 512 | 0.839332 |
| full fine PQ | none | 512 | 0.802021 |
| unsupervised coarse-to-fine | 4x64, 24 bits/key | 4096 | 0.800967 |
| weighted coarse, boost 16 | 4x64, 24 bits/key | 4096 | 0.800952 |
| weighted coarse, boost 4 | 4x64, 24 bits/key | 2048 | 0.794999 |
| unsupervised coarse-to-fine | 4x64, 24 bits/key | 2048 | 0.792147 |
| weighted coarse, boost 4 | 4x64, 24 bits/key | 1024 | 0.773717 |
| weighted coarse, boost 16 | 4x64, 24 bits/key | 1024 | 0.772104 |
| unsupervised coarse-to-fine | 4x64, 24 bits/key | 1024 | 0.764865 |

## Comparison To Separate Coarse Ranker

The previous attention-label separate coarse ranker reached:

| Method | Coarse PQ | Shortlist | Top-16 recall |
| --- | --- | ---: | ---: |
| supervised separate coarse ranker | 4x64, 24 bits/key | 4096 | 0.802688 |
| supervised separate coarse ranker | 4x64, 24 bits/key | 2048 | 0.797464 |
| supervised separate coarse ranker | 4x64, 24 bits/key | 1024 | 0.769128 |

## Interpretation

Attention-weighted codebooks are useful in the tightest shortlist range. At shortlist `1024`, the best weighted codebook reached `0.773717`, above the unsupervised `0.764865` and the separate supervised coarse ranker's `0.769128`.

At shortlist `2048`, weighting gives a smaller lift over unsupervised, `0.794999` versus `0.792147`, while the separate supervised ranker remains higher at `0.797464`. At shortlist `4096`, weighting essentially ties unsupervised and stays below the separate supervised ranker.

This points to a split design: keep the same fine-ranker space for very tight candidate survival, and use a sharper learned coarse projection when the shortlist is larger. The next test should combine the two: train the separate coarse ranker, then fit attention-weighted codebooks in that coarse space.
