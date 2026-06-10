# PQ Scan Benchmark Snapshot

Date: 2026-05-13

## Question

Product-quantized learned-score lookup preserves most of the learned ranker signal, but it scans compressed key codes. This benchmark asks whether a simple GPU implementation has plausible million-token throughput.

## Setup

- Hardware: Modal H100
- Context: `1,000,000` synthetic keys
- Heads: `9`
- Queries: `1,4`
- PQ shapes: `8 x 256` and `16 x 256`
- Verifier budget: top `512`
- Implementation: stock PyTorch gather over code tables plus `torch.topk`

Run:

```text
ap-O5kn0XOaaeZmZm0vAodTYT
fc-01KRHV08VFBXNB4SZJRK78GV6B
results/modal_runs/sva-h100-pq-scan-benchmark-v2-20260513-193323.modal.log
```

## Result

| queries | subspaces | bits/key | scanned code lookups | scan ms avg | topk ms avg | total ms avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 64 | 72,000,000 | 1.959 | 0.235 | 2.195 |
| 1 | 16 | 128 | 144,000,000 | 4.240 | 0.237 | 4.476 |
| 4 | 8 | 64 | 288,000,000 | 5.671 | 0.656 | 6.327 |
| 4 | 16 | 128 | 576,000,000 | 14.139 | 0.655 | 14.794 |

For one query, the `8 x 256` row is about `2.2 ms` per layer for a million-token scan over 9 heads. The `16 x 256` row is about `4.5 ms` per layer. Batched queries improve per-query cost, but autoregressive decoding usually cares most about the single-query row.

## Interpretation

The primitive is plausible but not finished. Applying `8 x 256` PQ scanning in every layer of a 30-layer model would be about `66 ms/token` before the rest of the model compute and before any custom-kernel optimization. The recall result says PQ is worth pursuing; the benchmark says the next engineering problem is reducing how often and how broadly the scan runs.

## Next Test

The sharp follow-up is coarse-to-fine PQ:

1. Use a cheap coarse PQ score to shortlist from the full million-token code stream.
2. Re-score the shortlist with the stronger `8 x 256` or `16 x 256` PQ score.
3. Then run exact QK over the final `256-512` candidates.
4. Measure recall and latency together.

