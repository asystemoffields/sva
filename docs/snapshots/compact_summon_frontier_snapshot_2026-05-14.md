# Compact Summon Frontier Snapshot - 2026-05-14

This snapshot records the first compact coarse-code quality and million-token speed frontier for SVA.

## Run

- Modal app: `ap-nYAAXBJmg2c0e2MPZPtYhO`
- Function call: `fc-01KRJG72XH1KS5XAF9FKFQ1SD4`
- Dashboard: https://modal.com/id/fc-01KRJG72XH1KS5XAF9FKFQ1SD4
- Hardware: H100
- Quality model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Quality context: `8192`
- Quality eval: 4 held-out built-in documents
- Layers: all 30
- Artifact training: teacher
- Ranker: rank `64`
- Quality shortlists: `512`, `768`, `1024`, `1536`, `2048`
- Quality budgets: `128`, `256`, `512`
- Speed context: synthetic `1,000,000` key cache
- Speed query counts: `1`, `4`, `16`
- Coarse-code shapes: `1x256`, `2x64`, `2x128`, `2x256`, `4x64`

## Quality Frontier

Representative verified top-16 recall:

| coarse code | 1024/256 | 1536/256 | 2048/256 | 2048/512 |
|---|---:|---:|---:|---:|
| 1x256 | 0.978194 | 0.990606 | 0.995388 | 0.995985 |
| 2x64 | 0.977494 | 0.990406 | 0.995278 | 0.995880 |
| 2x128 | 0.984053 | 0.993473 | 0.996805 | 0.997432 |
| 2x256 | 0.988336 | 0.995258 | 0.997623 | 0.998271 |
| 4x64 | 0.989731 | 0.995916 | 0.997869 | 0.998524 |

## Million-Token Speed Frontier

Representative vectorized SVA latency in milliseconds:

| coarse code | q=1, 2048/512 | q=4, 2048/512 | q=16, 2048/512 |
|---|---:|---:|---:|
| 1x256 | 0.649 | 1.411 | 4.003 |
| 2x64 | 0.776 | 1.723 | 5.042 |
| 2x128 | 0.780 | 1.722 | 5.054 |
| 2x256 | 0.776 | 1.727 | 5.054 |
| 4x64 | 1.028 | 2.332 | 7.129 |

Full attention on the same synthetic setup was about `2.08 ms` for `q=1`, `2.35-2.38 ms` for `q=4`, and `3.45 ms` for `q=16`.

## Readout

Compact coarse codes are the first clear no-custom-kernel speed lever.

`1x256` is the fastest branch: about `3.2x` faster than full attention for one-query million-token decode and faster than full attention for `q=4`. It still loses at `q=16`, but the gap is much smaller than with `4x64`.

`2x256` is the stronger quality/speed compromise. It nearly matches the `4x64` quality frontier while reducing one-query latency from about `1.03 ms` to about `0.78 ms` and grouped `q=16` latency from about `7.13 ms` to about `5.05 ms`.

The main bottleneck remains coarse scan plus top-k, but this result says the coarse code shape is a real systems lever. The deployable artifact should make coarse-code shape explicit metadata rather than baking in `4x64`.

## Next Step

The next deployability step is to package a frozen SVA artifact format for SmolLM2: per-layer low-rank projections, logit scales, coarse codebooks, shape metadata, shortlist/budget defaults, and a tiny loader that can run cached decode from those artifacts.
