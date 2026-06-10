# Tight Summon Frontier Snapshot - 2026-05-14

This snapshot records the first combined tight-shortlist quality and million-token speed frontier for SVA.

## Run

- Modal app: `ap-nNolz3O0hFu4T7UuB1wth6`
- Function call: `fc-01KRJDWSB0ZXG23BYKEVT052JN`
- Dashboard: https://modal.com/id/fc-01KRJDWSB0ZXG23BYKEVT052JN
- Hardware: H100
- Quality model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Quality context: `8192`
- Quality eval: 4 held-out built-in documents
- Layers: all 30
- Artifact training: teacher
- Ranker: rank `64`, `4x64` coarse PQ
- Tight training: `384` train query samples, `280` ranker steps, `160` hard-negative steps, pool `512`, `96` hard negatives
- Quality shortlists: `256`, `384`, `512`, `768`, `1024`, `1536`, `2048`
- Quality budgets: `64`, `128`, `256`, `384`, `512`
- Speed context: synthetic `1,000,000` key cache
- Speed query counts: `1`, `4`, `16`
- Speed variants: full attention and vectorized PyTorch SVA

The first launch failed early because the cached-decode benchmark did not support an empty timing-query list. The parser now allows quality-only sweeps, and the rerun completed.

## Quality Frontier

| shortlist | budget | candidate top16 recall | verified top16 recall |
|---:|---:|---:|---:|
| 256 | 64 | 0.904604 | 0.897267 |
| 256 | 128 | 0.904604 | 0.904166 |
| 256 | 256 | 0.904604 | 0.904604 |
| 384 | 128 | 0.943061 | 0.941897 |
| 384 | 256 | 0.943061 | 0.943026 |
| 512 | 128 | 0.962860 | 0.961043 |
| 512 | 256 | 0.962860 | 0.962761 |
| 512 | 512 | 0.962860 | 0.962860 |
| 768 | 128 | 0.981772 | 0.978932 |
| 768 | 256 | 0.981772 | 0.981497 |
| 768 | 512 | 0.981772 | 0.981765 |
| 1024 | 128 | 0.990155 | 0.986606 |
| 1024 | 256 | 0.990155 | 0.989732 |
| 1024 | 512 | 0.990155 | 0.990131 |
| 1536 | 128 | 0.996543 | 0.992219 |
| 1536 | 256 | 0.996543 | 0.995918 |
| 1536 | 512 | 0.996543 | 0.996487 |
| 2048 | 128 | 0.998602 | 0.993940 |
| 2048 | 256 | 0.998602 | 0.997870 |
| 2048 | 512 | 0.998602 | 0.998525 |

## Million-Token Speed Frontier

| queries | shortlist | budget | full ms avg | SVA vectorized ms avg | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 128 | 2.099 | 0.962 | 2.18x |
| 1 | 512 | 256 | 2.071 | 0.961 | 2.15x |
| 1 | 768 | 256 | 2.062 | 0.958 | 2.15x |
| 1 | 1024 | 256 | 2.051 | 0.958 | 2.14x |
| 1 | 2048 | 512 | 2.055 | 0.970 | 2.12x |
| 4 | 512 | 128 | 2.317 | 2.190 | 1.06x |
| 4 | 512 | 256 | 2.324 | 2.217 | 1.05x |
| 4 | 1024 | 256 | 2.363 | 2.252 | 1.05x |
| 4 | 2048 | 512 | 2.354 | 2.303 | 1.02x |
| 16 | 512 | 128 | 3.480 | 6.805 | 0.51x |
| 16 | 512 | 256 | 3.452 | 6.839 | 0.50x |
| 16 | 1024 | 256 | 3.452 | 6.925 | 0.50x |
| 16 | 2048 | 512 | 3.454 | 7.114 | 0.49x |

## Readout

The tight-shortlist frontier says `512` is too small for the current quality target, `768` may be useful for a lower-fidelity mode, `1024` is the first plausible tight setting, and `1536-2048` restores the high-recall regime.

Shrinking the shortlist does not buy much speed in the current vectorized PyTorch path. At one million keys and one query, `512/128` is `0.962 ms` while `2048/512` is `0.970 ms`. The dominant cost is the coarse scan and score/top-k path over the full cache, not the exact low-rank rescore or verifier.

## Next Step

The next optimization target is the summon stage before shortlist size: reduce or avoid the full coarse score matrix. The most promising no-custom-kernel branches are fewer coarse subspaces/codewords, staged coarse routing, or an adaptive/head-pruned summon pass that keeps the one-query speedup while improving grouped-query decode.
