# Product-Quantized Lookup Snapshot

Date: 2026-05-13

## Question

Hard routing cells lost the learned-score signal when the catalog became selective. This test asks whether product quantization can preserve the learned rank-64 Q/K score well enough to choose exact-verifier candidates from compressed key codes.

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: SmolLM2 configured window, `8192` tokens
- Train text: generated long stream
- Eval text: reversed generated stream
- Layers: `0,1,5,10,18,24,29`
- Ranker: asymmetric learned Q/K projection, rank `64`
- PQ codebooks: per-head k-means over learned low-rank train keys
- PQ score: asymmetric query-to-codebook lookup summed across subspaces
- Subspaces: `4,8,16`
- Codewords per subspace: `16,64,256`
- Verifier budgets: `128,256,512`
- Full-attention target: top-16 keys per sampled query/head

Run:

```text
ap-OMgr32qnpNoF7W0CnMI7JT
fc-01KRHTQZ5RRG67KATJKAV9N120
results/modal_runs/sva-h100-pq-lookup-20260513-192851.modal.log
```

## Aggregate Result

Exact learned-ranker baseline:

| method | budget | avg candidates | top-16 recall |
| --- | ---: | ---: | ---: |
| exact rank-64 score | 128 | 128.0 | 0.655661 |
| exact rank-64 score | 256 | 254.0 | 0.754046 |
| exact rank-64 score | 512 | 500.0 | 0.839084 |

PQ rows, sorted by `512`-budget recall:

| subspaces | codewords | bits/key | budget 128 recall | budget 256 recall | budget 512 recall | score cosine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 128 | 0.598524 | 0.704985 | 0.803184 | 0.962802 |
| 16 | 64 | 96 | 0.553354 | 0.668108 | 0.775995 | 0.932386 |
| 8 | 256 | 64 | 0.535714 | 0.647166 | 0.755937 | 0.903463 |
| 4 | 256 | 32 | 0.515485 | 0.624752 | 0.732871 | 0.865555 |
| 8 | 64 | 48 | 0.491846 | 0.608243 | 0.724346 | 0.871316 |
| 16 | 16 | 64 | 0.465650 | 0.590340 | 0.715665 | 0.869227 |
| 4 | 64 | 24 | 0.472827 | 0.588216 | 0.704241 | 0.837629 |
| 8 | 16 | 32 | 0.420449 | 0.543759 | 0.675750 | 0.810606 |
| 4 | 16 | 16 | 0.403646 | 0.524042 | 0.653196 | 0.792769 |

The best PQ row preserves about `95.7%` of the exact learned-ranker recall at budget `512` (`0.803184 / 0.839084`) and about `93.5%` at budget `256` (`0.704985 / 0.754046`). The `64`-bit row, `8 subspaces / 256 codewords`, still preserves about `90.1%` of the exact recall at budget `512`.

## Interpretation

This is a strong go for score-preserving compressed lookup. Unlike hard routing cells, PQ keeps the learned ranker order well enough that the exact verifier sees most of the full-attention top keys.

The cost shape changes from ranking full low-rank key vectors to scanning compact codes. At `16 x 256`, each key needs `128` ideal bits per head/layer; at `8 x 256`, each key needs `64` ideal bits. This is still a scan over the context, but it is a much cheaper scan than exact full-dimensional QK or exact learned rank-64 scoring.

## Next Test

The next risk is speed and sublinearity:

1. Benchmark PQ scan throughput at million-token scale.
2. Try a coarse-to-fine PQ path: use fewer subspaces or coarse codebooks to shortlist, then higher-resolution PQ, then exact QK.
3. Test whether the `64`-bit row is enough for a socketed model-quality run, since it is the more plausible deployment target.

