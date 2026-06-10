# Learned Multi-Write IVF Lookup Snapshot

Date: 2026-05-13

## Question

Single-write IVF improved the learned-ranker lookup geometry, but it still lost most of the full learned-score signal. This test asks whether giving each key multiple centroid writes can recover more top-attention keys in the projected million-token candidate band.

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: SmolLM2 configured window, `8192` tokens
- Train text: generated long stream
- Eval text: reversed generated stream
- Layers: `0,1,5,10,18,24,29`
- Ranker: asymmetric learned Q/K projection, rank `64`
- Lookup: per-head k-means centroids over learned low-rank keys
- Key writes: nearest `2,4,8` centroids by Euclidean distance
- Query routing: top learned-score centroids
- Target context projection: `1,000,000` tokens
- Verifier budgets: `256,512`
- Full-attention target: top-16 keys per sampled query/head

Run:

```text
ap-1R8BFnkmOT0YdjHqDIQqsl
fc-01KRHRZ74BBP5HAVQTBQM65REJ
results/modal_runs/sva-h100-learned-multiwrite-ivf-lookup-20260513-185751.modal.log
```

## Aggregate Result

Budgets `256` and `512` were identical in this sweep because the raw candidate sets were usually smaller than `256`.

| centroids | writes | probes | budget | avg 8192 candidates | avg projected 1M candidates | p95 projected 1M candidates | verified top-16 recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 8 | 4 | 256 | 12.8 | 1,564.4 | 7,324.2 | 0.147647 |
| 2048 | 4 | 4 | 256 | 7.4 | 898.1 | 4,150.4 | 0.105422 |
| 4096 | 8 | 4 | 256 | 6.7 | 811.8 | 3,784.2 | 0.098431 |
| 2048 | 8 | 2 | 256 | 6.4 | 781.5 | 4,028.3 | 0.092262 |
| 2048 | 2 | 4 | 256 | 4.2 | 514.5 | 2,441.4 | 0.071227 |
| 4096 | 4 | 4 | 256 | 3.9 | 471.0 | 2,252.2 | 0.067677 |
| 2048 | 4 | 2 | 256 | 3.5 | 427.9 | 2,319.3 | 0.059384 |
| 4096 | 8 | 2 | 256 | 3.3 | 403.1 | 2,075.2 | 0.056935 |
| 2048 | 8 | 1 | 256 | 3.2 | 389.8 | 2,441.4 | 0.054998 |
| 4096 | 2 | 4 | 256 | 2.2 | 262.6 | 1,220.7 | 0.042628 |
| 4096 | 4 | 2 | 256 | 1.8 | 225.6 | 1,220.7 | 0.037063 |
| 2048 | 2 | 2 | 256 | 1.9 | 234.9 | 1,342.8 | 0.037032 |
| 2048 | 4 | 1 | 256 | 1.7 | 204.8 | 1,220.7 | 0.032614 |
| 4096 | 8 | 1 | 256 | 1.6 | 199.8 | 1,220.7 | 0.031219 |
| 4096 | 2 | 2 | 256 | 1.0 | 121.5 | 610.4 | 0.022228 |
| 2048 | 2 | 1 | 256 | 0.9 | 110.2 | 732.4 | 0.019609 |
| 4096 | 4 | 1 | 256 | 0.9 | 108.6 | 610.4 | 0.019066 |
| 4096 | 2 | 1 | 256 | 0.5 | 58.0 | 366.2 | 0.011331 |

The best few-hundred to low-thousand candidate row is `2048 centroids / 4 writes / 4 probes`, with `0.105422` recall at about `898` projected million-token candidates. That is close to single-write IVF's `0.102477` recall at about `783` projected candidates.

The highest-recall multi-write row is `2048 centroids / 8 writes / 4 probes`, with `0.147647` recall at about `1,564` projected candidates. The comparable single-write row at `512 centroids / 4 probes` reached `0.166574` recall at about `1,666` projected candidates.

## Interpretation

Multi-write IVF improves over some same-centroid single-write settings, but it does not beat the existing single-write frontier in recall per projected candidate. More Euclidean writes mostly widen the same catalog geometry.

The bottleneck is now the routing objective. K-means cells are organized around key geometry, while the retrieval target is query-key score under the learned ranker. The next catalog should be trained or constructed against top-key recall directly.

## Next Test

Move from unsupervised cells to a supervised router:

1. Keep the learned rank-64 Q/K scorer.
2. Train or optimize route vectors/cells using the full-attention top-key labels.
3. Let keys write to cells that are useful for the queries that need them.
4. Measure recall per projected million-token candidate against both single-write and multi-write IVF.

