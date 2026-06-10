# Coarse Exact-Rescore Benchmark Snapshot - 2026-05-13

## Run

- Commit: working tree after `ee799b2`
- Modal app: `sva-coarse-exact-rescore-benchmark-h100`
- Modal app id: `ap-pmy4xJ0rZhcsH2NXOjwiDz`
- Function call: `fc-01KRJ1GQC22QFKBD1BAG80PYQH`
- Dashboard: https://modal.com/id/fc-01KRJ1GQC22QFKBD1BAG80PYQH
- Full log: `results/modal_runs/sva-h100-coarse-exact-rescore-benchmark-20260513-212611.modal.log`
- Context: `1,000,000` keys
- Heads: `9`
- Queries: `1,4`
- Coarse PQ: `4x64`, `24` bits/key
- Exact middle stage: rank-64 bf16 keys and queries
- Shortlists: `1024,1536,2048,4096`
- Verifier budget: `512`
- Hardware: H100 through Modal

This benchmark measures the proposed serving path after the handoff diagnostic:

1. scan cheap coarse PQ codes over all one million keys;
2. select a coarse shortlist;
3. gather rank-64 low-rank keys for that shortlist;
4. score them exactly against the rank-64 query;
5. select the final verifier budget.

It measures scorer and top-k time, not value aggregation.

## Results

| Queries | Shortlist | Coarse scan ms | Shortlist top-k ms | Exact rank-64 rescore ms | Final top-k ms | Total ms | Rank-key memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 0.571 | 0.251 | 0.123 | 0.051 | 0.996 | 1.15 GB |
| 1 | 1536 | 0.571 | 0.250 | 0.123 | 0.060 | 1.005 | 1.15 GB |
| 1 | 2048 | 0.569 | 0.252 | 0.121 | 0.061 | 1.004 | 1.15 GB |
| 1 | 4096 | 0.569 | 0.258 | 0.121 | 0.071 | 1.018 | 1.15 GB |
| 4 | 1024 | 1.300 | 0.680 | 0.125 | 0.057 | 2.162 | 1.15 GB |
| 4 | 1536 | 1.299 | 0.677 | 0.125 | 0.060 | 2.161 | 1.15 GB |
| 4 | 2048 | 1.299 | 0.679 | 0.124 | 0.062 | 2.164 | 1.15 GB |
| 4 | 4096 | 1.298 | 0.685 | 0.191 | 0.073 | 2.246 | 1.15 GB |

## Comparison

The previous `4x64 -> 16x256` coarse-to-fine PQ benchmark took about `1.91 ms` for one query and about `3.03 ms` for four queries at shortlist `2048`. The exact rank-64 middle stage measured here took about `1.00 ms` for one query and `2.16 ms` for four queries at the same shortlist.

Combined with the handoff diagnostic, this makes the exact middle stage the strongest current serving candidate: it is faster than the fine-PQ middle stage in this synthetic benchmark, and it preserved much more recall in the real SmolLM2 held-out-text diagnostic.

## Interpretation

The cost is dominated by the coarse scan and shortlist top-k. The exact rank-64 rescore itself is around `0.12 ms` for one query through shortlist `2048`, and still only `0.19 ms` for four queries at shortlist `4096`.

The memory trade is the important one. Rank-64 bf16 keys for `9` heads and one million tokens take about `1.15 GB` per layer cache in this synthetic layout. The next risk is whether that memory is acceptable, compressible, or only needed for a subset of layers. The next quality test should socket this three-stage path into the SmolLM2 attention replacement harness: hard-negative coarse PQ summon, exact rank-64 rescore, then exact attention verification.
