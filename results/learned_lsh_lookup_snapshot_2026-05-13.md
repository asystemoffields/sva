# Learned LSH Lookup Snapshot

Date: 2026-05-13

## Question

The learned rank-64 Q/K score can recover much of full attention's top-16 key set when it is allowed to rank every key. This test asks whether random-hyperplane LSH over that learned low-rank space can make the lookup sublinear before exact verification.

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: SmolLM2 configured window, `8192` tokens
- Train text: generated long stream
- Eval text: reversed generated stream
- Layers: `0,1,5,10,18,24,29`
- Ranker: asymmetric learned Q/K projection, rank `64`
- Lookup: random-hyperplane LSH over learned low-rank Q/K vectors
- Target context projection: `1,000,000` tokens
- Verifier budgets: `256,512`
- Full-attention target: top-16 keys per sampled query/head

Run:

```text
ap-ACYCgwmcnZYt7SthFdDu9a
results/modal_runs/sva-h100-learned-lsh-lookup-v2-20260513-183151.modal.log
```

## Aggregate Result

| bits | tables | radius | budget | avg 8192 candidates | avg projected 1M candidates | p95 projected 1M candidates | verified top-16 recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 128 | 2 | 512 | 316.1 | 38,586.7 | 140,741.0 | 0.233429 |
| 18 | 128 | 2 | 256 | 316.1 | 38,586.7 | 140,741.0 | 0.229740 |
| 18 | 64 | 2 | 512 | 180.0 | 21,968.6 | 86,145.0 | 0.158002 |
| 20 | 128 | 2 | 512 | 132.1 | 16,128.4 | 71,020.5 | 0.133867 |
| 20 | 64 | 2 | 512 | 74.3 | 9,073.3 | 42,413.3 | 0.090061 |
| 22 | 128 | 2 | 512 | 57.7 | 7,047.0 | 33,746.3 | 0.080233 |
| 24 | 128 | 2 | 512 | 26.3 | 3,216.1 | 16,290.3 | 0.050394 |
| 22 | 64 | 1 | 512 | 5.1 | 625.0 | 2,929.7 | 0.013083 |
| 24 | 128 | 1 | 512 | 4.1 | 505.0 | 2,374.3 | 0.012137 |
| 24 | 64 | 1 | 512 | 2.3 | 279.2 | 1,342.8 | 0.007084 |

The strongest aggregate LSH row reaches `0.233429` verified top-16 recall, but it projects to about `38.6k` candidates on average at a million-token context and p95 about `140.7k`. In the rough million-token candidate band we care about, the best aggregate recall is about `0.013`.

Layer 0 is much friendlier to this lookup than the deeper layers. For example, layer 0 at `18 bits / 128 tables / radius 2 / budget 512` reaches `0.780165` recall, but the aggregate across the sampled stack falls to `0.233429`. The serving mechanism has to work through the whole stack, so this is a kill for random sign-LSH as the learned-ranker address.

## Interpretation

The learned ranker remains a strong signal. The previous held-out-text ranker result reached `0.749752` top-16 recall at a 256-candidate verifier budget and `0.835488` at 512. This run says the missing piece is not the compact score; it is the lookup geometry.

Random sign buckets throw away too much order information from the learned dot-product space. The next lookup should use the score structure directly: centroid routing, IVF-style learned cells, product-quantized asymmetric scoring, or another address that lets the query ask for high inner-product regions instead of nearby sign codes.

## Next Test

Run a score-aware lookup:

1. Train the same rank-64 Q/K scorer.
2. Build per-head centroids over low-rank keys.
3. Route each query to the highest learned-score centroids.
4. Verify only keys in those routed cells.
5. Compare recall and projected million-token candidate counts against this LSH result.

