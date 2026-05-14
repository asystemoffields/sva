# Coarse-to-Fine PQ Scan Benchmark Snapshot - 2026-05-13

## Run

- Commit: `04e1a52`
- Modal app: `sva-coarse-to-fine-pq-scan-benchmark-h100`
- Function call: `fc-01KRHWR9WPWPJ8Y3W2GSESBKR1`
- Dashboard: https://modal.com/id/fc-01KRHWR9WPWPJ8Y3W2GSESBKR1
- Full log: `results/modal_runs/sva-h100-coarse-to-fine-pq-scan-benchmark-20260513-200356.modal.log`
- Context: `1,000,000` keys
- Heads: `9`
- Queries: `1,4`
- Fine PQ: `16x256`
- Coarse PQ: `4x16`, `4x64`, `8x16`
- Shortlists: `1024,2048,4096`
- Verifier budget: `512`
- Hardware: H100 through Modal

This benchmark measures the staged serving path directly:

1. scan cheap coarse PQ codes over all one million keys;
2. select a shortlist;
3. rescore the shortlist with `16x256` fine PQ;
4. select the final verifier budget.

## Results

| Queries | Coarse PQ | Shortlist | Coarse scan ms | Shortlist top-k ms | Fine rescore ms | Final top-k ms | Total ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4x64, 24 bits/key | 1024 | 0.549 | 0.245 | 0.994 | 0.052 | 1.840 |
| 1 | 4x16, 16 bits/key | 4096 | 0.548 | 0.262 | 0.975 | 0.070 | 1.854 |
| 1 | 4x16, 16 bits/key | 1024 | 0.550 | 0.245 | 1.007 | 0.053 | 1.855 |
| 1 | 4x64, 24 bits/key | 2048 | 0.550 | 0.247 | 1.050 | 0.059 | 1.907 |
| 1 | 4x64, 24 bits/key | 4096 | 0.549 | 0.263 | 1.036 | 0.071 | 1.917 |
| 4 | 4x16, 16 bits/key | 1024 | 1.274 | 0.677 | 1.001 | 0.051 | 3.004 |
| 4 | 4x64, 24 bits/key | 2048 | 1.276 | 0.683 | 0.993 | 0.062 | 3.014 |
| 4 | 4x64, 24 bits/key | 4096 | 1.276 | 0.689 | 1.001 | 0.072 | 3.038 |
| 4 | 8x16, 32 bits/key | 1024 | 2.588 | 0.688 | 1.006 | 0.063 | 4.345 |

## Comparison To Full PQ Scan

Previous full-scan PQ benchmark over one million keys and 9 heads:

| Path | Queries | Total ms |
| --- | ---: | ---: |
| full PQ 8x256 | 1 | 2.195 |
| full PQ 16x256 | 1 | 4.476 |
| full PQ 8x256 | 4 | 6.327 |
| full PQ 16x256 | 4 | 14.794 |

Best recall-preserving coarse-to-fine rows from the lookup test:

| Recall path | Queries measured here | Total ms | Aggregate recall |
| --- | ---: | ---: | ---: |
| 4x64 coarse -> 16x256 fine, shortlist 2048 | 1 | 1.907 | 0.789078 |
| 4x64 coarse -> 16x256 fine, shortlist 4096 | 1 | 1.917 | 0.799541 |
| 4x64 coarse -> 16x256 fine, shortlist 2048 | 4 | 3.014 | 0.789078 |
| 4x64 coarse -> 16x256 fine, shortlist 4096 | 4 | 3.038 | 0.799541 |

The `4x64 -> 16x256` staged path is faster than full `8x256` scanning in this stock PyTorch H100 benchmark while preserving nearly all of the full `16x256` fine-PQ recall at shortlist `4096`.

## Interpretation

This is the first point where the two halves meet:

- Recall: coarse-to-fine PQ can keep the fine-PQ winners alive.
- Throughput: the staged path can beat full lower-quality PQ scanning at one million keys.

The measured fine-rescore block is around `1 ms` across shortlists, which suggests the current benchmark is dominated by gather/kernel overhead. A fused kernel or packed-code implementation should improve that stage. The next research risk is quality at smaller shortlists, especially `1024-2048`, because the fastest path is already operationally attractive.

## Next Test

Train the coarse stage against the fine-PQ winners instead of using independently fit coarse PQ. The goal is to keep `16x256` fine-PQ recall with a `1024-2048` shortlist, making the fastest measured path also the strongest recall path.
