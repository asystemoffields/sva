# Million Cached Decode Benchmark Snapshot - 2026-05-14

This snapshot records the first synthetic million-token cached-decode throughput benchmark for SVA.

## Run

- Modal app: `ap-GnVcXZMDcC6cxNwcUZubrR`
- Function call: `fc-01KRJDGEQ4S4A38TZ2Z0AHNCZQ`
- Dashboard: https://modal.com/id/fc-01KRJDGEQ4S4A38TZ2Z0AHNCZQ
- Hardware: H100
- Heads: `9`
- Head dim: `64`
- Rank dim: `64`
- Coarse code: `4x64`
- Contexts: `8192`, `65536`, `262144`, `1000000`
- Query counts: `1`, `4`, `16`
- Shortlists: `1024`, `2048`
- Budgets: `256`, `512`
- Variants: full attention, loop-style SVA, vectorized PyTorch SVA

The benchmark uses synthetic cached Q/K/V, low-rank keys, and product codes. It measures serving-shaped decode lookup after the key-side SVA catalog already exists.

## Key Results

| context | queries | shortlist | budget | full ms avg | SVA vectorized ms avg | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 16 | 2048 | 512 | 0.205 | 0.931 | 0.22x |
| 65536 | 16 | 2048 | 512 | 0.287 | 1.236 | 0.23x |
| 262144 | 1 | 2048 | 512 | 0.580 | 0.704 | 0.82x |
| 262144 | 16 | 2048 | 512 | 0.966 | 2.531 | 0.38x |
| 1000000 | 1 | 1024 | 512 | 2.086 | 1.012 | 2.06x |
| 1000000 | 1 | 2048 | 512 | 2.092 | 1.024 | 2.04x |
| 1000000 | 4 | 1024 | 512 | 2.362 | 2.306 | 1.02x |
| 1000000 | 4 | 2048 | 512 | 2.355 | 2.324 | 1.01x |
| 1000000 | 16 | 1024 | 512 | 3.446 | 6.983 | 0.49x |
| 1000000 | 16 | 2048 | 512 | 3.443 | 7.131 | 0.48x |

The vectorized PyTorch scoring path is a major improvement over the loop-style path. At one million keys and one query, loop-style SVA was around `3.7-3.8 ms`; vectorized SVA was around `1.0 ms`.

## Component Readout

At one million keys, `queries=1`, `shortlist=2048`, and `budget=512`, vectorized SVA spent about:

| component | ms |
|---|---:|
| coarse scan | 0.667 |
| shortlist top-k | 0.247 |
| exact low-rank rescore | 0.172 |
| verifier attention | 0.201 |
| component total | 1.287 |

At one million keys, `queries=16`, `shortlist=2048`, and `budget=512`, vectorized SVA spent about:

| component | ms |
|---|---:|
| coarse scan | 4.821 |
| shortlist top-k | 1.860 |
| exact low-rank rescore | 0.383 |
| verifier attention | 0.208 |
| component total | 7.273 |

## Readout

The no-custom-kernel path has a real opening for single-token million-context decode: vectorized PyTorch SVA is about `2x` faster than full attention in this synthetic serving shape.

The grouped-query case is the current speed problem. The bottleneck is no longer exact verification; it is the summon stage itself, especially coarse score construction plus top-k. That makes the next optimization target clear: train and measure SVA around smaller shortlists, adaptive summon budgets, and score/top-k paths that avoid materializing as much `heads x queries x context` work.

## Next Step

The next decisive test should combine quality and speed pressure: retrain the summon stage for `512-1024` shortlist survival, then rerun the million cached-decode benchmark with `512`, `768`, `1024`, and `2048` shortlists. If `512-768` preserves enough recall, SVA gains speed without requiring a custom kernel.
