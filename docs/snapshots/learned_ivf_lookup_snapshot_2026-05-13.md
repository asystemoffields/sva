# Learned IVF Lookup Snapshot

Date: 2026-05-13

## Question

Random sign-LSH over the learned rank-64 Q/K score lost too much dot-product order. This test asks whether score-aware centroid routing can summon better candidates for the same learned ranker.

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: SmolLM2 configured window, `8192` tokens
- Train text: generated long stream
- Eval text: reversed generated stream
- Layers: `0,1,5,10,18,24,29`
- Ranker: asymmetric learned Q/K projection, rank `64`
- Lookup: per-head k-means centroids over learned low-rank keys
- Query routing: top learned-score centroids
- Target context projection: `1,000,000` tokens
- Verifier budgets: `256,512`
- Full-attention target: top-16 keys per sampled query/head

Run:

```text
ap-be4O2j4YbnrzE2V1oml2DL
results/modal_runs/sva-h100-learned-ivf-lookup-20260513-184357.modal.log
```

## Aggregate Result

| centroids | probes | budget | avg 8192 candidates | avg projected 1M candidates | p95 projected 1M candidates | verified top-16 recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | 512 | 28.5 | 3,484.5 | 13,183.6 | 0.234422 |
| 512 | 4 | 512 | 13.6 | 1,665.6 | 7,080.1 | 0.166574 |
| 1024 | 8 | 512 | 12.3 | 1,504.6 | 6,347.7 | 0.158234 |
| 512 | 2 | 512 | 6.4 | 783.1 | 3,784.2 | 0.102477 |
| 2048 | 8 | 512 | 5.4 | 664.9 | 2,929.7 | 0.095827 |
| 1024 | 4 | 512 | 5.6 | 687.9 | 3,418.0 | 0.094169 |
| 2048 | 4 | 512 | 2.4 | 297.6 | 1,464.8 | 0.050208 |
| 1024 | 2 | 512 | 2.5 | 302.6 | 1,709.0 | 0.050084 |
| 2048 | 2 | 512 | 1.1 | 132.8 | 732.4 | 0.024569 |
| 2048 | 1 | 512 | 0.5 | 59.0 | 366.2 | 0.012308 |

The best aggregate row matches the best random sign-LSH recall almost exactly, `0.234422` versus `0.233429`, while using about `3.5k` projected million-token candidates instead of about `38.6k`. In the few-hundred-candidate band, IVF reaches about `0.095-0.102` recall, far above the sign-LSH band around `0.013`.

## Interpretation

Score-aware routing is a real improvement over sign buckets, but single-write k-means cells are still too blunt. The learned ranker can recover above `0.8` recall when it ranks all keys down to 512 candidates; centroid routing at similar projected candidate counts is still around `0.1`.

This points toward a stronger catalog rather than abandoning the learned-ranker path. The next lookup should give each key multiple chances to be found, or train the routing cells directly against top-key recall instead of relying on unsupervised Euclidean key clusters.

## Next Test

The sharp follow-up is multi-write or supervised routing:

1. Keep the same learned rank-64 Q/K scorer.
2. Let each key write to multiple high-scoring cells.
3. Keep query probes small enough to stay in the projected `128-1024` candidate range.
4. Compare recall per projected candidate against the single-write IVF run.

