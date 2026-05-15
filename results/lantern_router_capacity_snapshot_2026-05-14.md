# Lantern Router Capacity Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: train/eval at `8192`, evaluated on held-out reversed text
- Layers: `0,1,5,10,18,24,29`
- Low-rank ranker: rank `64`, trained for `160` steps per layer
- Router: supervised page-side key writes plus query probes, trained for `240` steps per layer/cell setting
- Labels: full-attention top-16 keys
- Cells: `512,1024,2048`
- Writes per key: `2,4,8`
- Query probes: `1,2,4`
- Verifier budgets: `256,512`
- Target projection: empirical candidate density scaled to `1,000,000` tokens
- Modal app: `ap-Mh1oAnqECaIPETMRm4f3mS`
- Function call: `fc-01KRMGJMR1BR3JP9YS9X6H6J4A`
- Full log: `results/modal_runs/sva-h100-lantern-router-capacity-20260514-202853.full.log`
- Runner: `modal_h100_lantern_router.py`

Two earlier launches were stopped intentionally. The first unbalanced Lantern run put too many keys into the same route cells and produced projected million-token candidate counts above `150k`. The second run added a soft key-write balance term, but the early rows were still too broad. The capacity run added hard per-cell write caps during assignment, so candidate counts became narrow enough to measure the recall tradeoff.

## Aggregate Result

The table reports aggregate held-out top-16 recall across the measured layers using verifier budget `512`. The `256` budget matched these rows except for the widest `512 cells / 8 writes / 4 probes` row, where verified recall was `0.475384` instead of `0.475524`.

| Cells | Writes | Probes | Budget | Avg Candidates | Projected 1M Candidates | Raw Recall | Verified Recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 1 | 512 | 63.4 | 7,733.4 | 0.088046 | 0.088046 |
| 512 | 2 | 2 | 512 | 106.1 | 12,953.7 | 0.195669 | 0.195669 |
| 512 | 2 | 4 | 512 | 174.5 | 21,295.9 | 0.306610 | 0.306610 |
| 512 | 4 | 1 | 512 | 124.9 | 15,248.4 | 0.134425 | 0.134425 |
| 512 | 4 | 2 | 512 | 200.1 | 24,423.7 | 0.264896 | 0.264896 |
| 512 | 4 | 4 | 512 | 314.1 | 38,343.2 | 0.387850 | 0.387850 |
| 512 | 8 | 1 | 512 | 245.5 | 29,963.7 | 0.188554 | 0.188554 |
| 512 | 8 | 2 | 512 | 377.2 | 46,046.5 | 0.338433 | 0.338433 |
| 512 | 8 | 4 | 512 | 564.0 | 68,848.4 | 0.475524 | 0.475524 |
| 1024 | 2 | 1 | 512 | 31.9 | 3,891.9 | 0.041450 | 0.041450 |
| 1024 | 2 | 2 | 512 | 54.9 | 6,698.0 | 0.114614 | 0.114614 |
| 1024 | 2 | 4 | 512 | 96.1 | 11,726.1 | 0.223168 | 0.223168 |
| 1024 | 4 | 1 | 512 | 63.5 | 7,753.7 | 0.082434 | 0.082434 |
| 1024 | 4 | 2 | 512 | 105.0 | 12,811.7 | 0.185020 | 0.185020 |
| 1024 | 4 | 4 | 512 | 175.8 | 21,456.3 | 0.298999 | 0.298999 |
| 1024 | 8 | 1 | 512 | 125.5 | 15,317.9 | 0.130007 | 0.130007 |
| 1024 | 8 | 2 | 512 | 200.3 | 24,445.8 | 0.243955 | 0.243955 |
| 1024 | 8 | 4 | 512 | 324.0 | 39,550.4 | 0.371652 | 0.371652 |
| 2048 | 2 | 1 | 512 | 15.9 | 1,942.0 | 0.024290 | 0.024290 |
| 2048 | 2 | 2 | 512 | 27.9 | 3,401.5 | 0.065755 | 0.065755 |
| 2048 | 2 | 4 | 512 | 51.6 | 6,294.4 | 0.142997 | 0.142997 |
| 2048 | 4 | 1 | 512 | 31.7 | 3,875.5 | 0.041636 | 0.041636 |
| 2048 | 4 | 2 | 512 | 53.4 | 6,523.4 | 0.107282 | 0.107282 |
| 2048 | 4 | 4 | 512 | 93.9 | 11,456.4 | 0.205869 | 0.205869 |
| 2048 | 8 | 1 | 512 | 63.2 | 7,714.7 | 0.080280 | 0.080280 |
| 2048 | 8 | 2 | 512 | 103.2 | 12,599.6 | 0.176603 | 0.176603 |
| 2048 | 8 | 4 | 512 | 173.4 | 21,168.0 | 0.285435 | 0.285435 |

## Readout

Capacity-balanced Lantern proves that page-side write hooks can produce controlled candidate sets: actual candidate counts are tens to hundreds at 8K, projecting to roughly `1.9k-68.8k` candidates at one million tokens. The widest row reaches `0.475524` aggregate top-16 recall.

The recall-per-candidate curve is not yet better than the earlier IVF/PQ mainline. Single-write learned IVF already reached `0.234422` recall at about `3.5k` projected candidates, while Lantern reaches `0.065755` at about `3.4k`, `0.142997` at about `6.3k`, and `0.306610` at about `21.3k`. The page-side idea is operational, but the current routing objective loses too many relevant keys once hard capacity assignment keeps the cells narrow.

The literary version: the lanterns have learned to leave crowded hooks and spread through the library. The missing trick is that they must spread to the hooks the future reader will actually open.

## Next Implementation

Train the routing geometry against the capacity-constrained behavior instead of training a broad soft overlap and applying hard capacity afterward:

- add a stricter route-alignment loss that makes positive keys write to the query's top route cells, not merely overlap softly;
- sweep hard capacity factors and stronger balance weights together, because recall and candidate width are now coupled;
- if the CE-style alignment improves the curve, fold it into a fixed candidate-tensor lookup benchmark and compare against the current static inverted path.

The pass condition for the next Lantern run is clear: exceed the single-write IVF recall-per-candidate frontier near the `3k-10k` projected-candidate band, or show a new high-recall point that is cheap enough to become a long-context candidate source.
