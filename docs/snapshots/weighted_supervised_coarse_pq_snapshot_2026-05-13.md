# Weighted Supervised Coarse PQ Snapshot - 2026-05-13

## Run

- Commit: `ced601d`
- Modal app: `sva-weighted-supervised-coarse-pq-h100`
- Modal app id: `ap-J3LpMKEG3ksbPspYRGifqX`
- Function call: `fc-01KRHZ36NJAYAGDN8X07E28W30`
- Dashboard: https://modal.com/id/fc-01KRHZ36NJAYAGDN8X07E28W30
- Full log: `results/modal_runs/sva-h100-weighted-supervised-coarse-pq-20260513-204454.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage: learned rank-64 scorer, `16x256` fine PQ
- Coarse stage: supervised rank-64 coarse scorer trained on attention top-16 labels
- Coarse codebook training: weighted k-means in supervised coarse space with attention top-16 key boosts `4,16,64`

## Aggregate Results

| Method | Coarse PQ | Shortlist | Top-16 recall |
| --- | --- | ---: | ---: |
| exact ranker | none | 512 | 0.839332 |
| full fine PQ | none | 512 | 0.802037 |
| unsupervised coarse-to-fine | 4x64, 24 bits/key | 4096 | 0.800998 |
| supervised coarse-to-fine | 4x64, 24 bits/key | 4096 | 0.802750 |
| weighted supervised, boost 16 | 4x64, 24 bits/key | 4096 | 0.803184 |
| weighted supervised, boost 4 | 4x64, 24 bits/key | 2048 | 0.799185 |
| weighted supervised, boost 16 | 4x64, 24 bits/key | 2048 | 0.799882 |
| supervised coarse-to-fine | 4x64, 24 bits/key | 2048 | 0.797464 |
| weighted supervised, boost 4 | 4x64, 24 bits/key | 1024 | 0.776445 |
| weighted supervised, boost 16 | 4x64, 24 bits/key | 1024 | 0.775391 |
| supervised coarse-to-fine | 4x64, 24 bits/key | 1024 | 0.769113 |
| unsupervised coarse-to-fine | 4x64, 24 bits/key | 1024 | 0.764881 |

## Interpretation

The combined design stacked the two partial gains. A supervised coarse projection gives the coarse scorer a sharper geometry, and attention-weighted codebook fitting improves candidate survival inside that geometry.

The strongest `4096` row reached `0.803184`, slightly above full fine PQ's `0.802037` and above the unweighted supervised row's `0.802750`. The `2048` row improved from `0.797464` to `0.799882`. The tight `1024` row improved from `0.769113` to `0.776445`.

This makes weighted supervised coarse PQ the current best serving candidate for the learned-ranker branch. It preserves the same scan structure and bit budget as the earlier `4x64 -> 16x256` design, so the million-token throughput result should carry over; the main open question is how low the shortlist can go before recall drops too far.

## Next Test

Run the same combined design with shorter shortlists: `512,768,1024`, budget `512`, and the best `4x64` coarse code. This tests whether SVA can move from the low-thousands shortlist band toward the few-hundreds band without losing the recovered recall.
