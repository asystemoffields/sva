# Lantern Router Alignment Snapshot - 2026-05-14

## Setup

- Base harness: `experiments/sva_lantern_router_test.py`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: train/eval at `8192`, held-out reversed text
- Layers: `0,1,5,10,18,24,29`
- Low-rank ranker: rank `64`, `160` steps
- Router: `240` steps
- New objective terms:
  - key alignment weight: `0.5`
  - query alignment weight: `0.25`
  - balance weight: `1.0`
- Cells: `512,1024,2048`
- Writes: `2,4,8`
- Probes: `1,2,4`
- Budgets: `256,512`
- Modal app: `ap-SnbKzEnvKI4Zy96xWkaFh7`
- Function call: `fc-01KRMH0QYKMJCBSH3NPGXTY2EM`
- Full log: `results/modal_runs/sva-h100-lantern-router-align-20260514-203636.full.log`

This run tested the immediate follow-up from the capacity snapshot: make positive keys align more strictly to the query route cells, instead of relying only on soft overlap and then imposing hard capacity at assignment time.

## Aggregate Result

Verifier budget `512`:

| Cells | Writes | Probes | Avg Candidates | Projected 1M Candidates | Verified Recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 1 | 63.6 | 7,761.7 | 0.092758 |
| 512 | 2 | 2 | 111.3 | 13,585.8 | 0.198971 |
| 512 | 2 | 4 | 180.2 | 22,001.7 | 0.307462 |
| 512 | 4 | 1 | 125.9 | 15,366.1 | 0.139245 |
| 512 | 4 | 2 | 208.9 | 25,500.7 | 0.264431 |
| 512 | 4 | 4 | 324.4 | 39,600.1 | 0.397848 |
| 512 | 8 | 1 | 247.9 | 30,257.4 | 0.191050 |
| 512 | 8 | 2 | 392.8 | 47,952.0 | 0.340216 |
| 512 | 8 | 4 | 585.1 | 71,420.0 | 0.504433 |
| 1024 | 2 | 1 | 32.0 | 3,902.2 | 0.055230 |
| 1024 | 2 | 2 | 57.8 | 7,056.3 | 0.125093 |
| 1024 | 2 | 4 | 100.4 | 12,251.1 | 0.231368 |
| 1024 | 4 | 1 | 63.7 | 7,780.2 | 0.090014 |
| 1024 | 4 | 2 | 111.0 | 13,545.8 | 0.186074 |
| 1024 | 4 | 4 | 184.1 | 22,475.4 | 0.309973 |
| 1024 | 8 | 1 | 126.4 | 15,423.8 | 0.134921 |
| 1024 | 8 | 2 | 212.3 | 25,916.7 | 0.245210 |
| 1024 | 8 | 4 | 338.9 | 41,367.1 | 0.393555 |
| 2048 | 2 | 1 | 16.0 | 1,955.4 | 0.035590 |
| 2048 | 2 | 2 | 29.3 | 3,577.9 | 0.075118 |
| 2048 | 2 | 4 | 53.7 | 6,560.6 | 0.143152 |
| 2048 | 4 | 1 | 32.0 | 3,908.5 | 0.055881 |
| 2048 | 4 | 2 | 56.2 | 6,856.0 | 0.116955 |
| 2048 | 4 | 4 | 97.3 | 11,879.1 | 0.207729 |
| 2048 | 8 | 1 | 63.8 | 7,783.0 | 0.091301 |
| 2048 | 8 | 2 | 108.6 | 13,260.5 | 0.180199 |
| 2048 | 8 | 4 | 179.4 | 21,901.7 | 0.285962 |

## Comparison To Capacity Baseline

The CE-style alignment gave small improvements but did not change the shape:

| Row | Capacity Recall | Alignment Recall | Candidate Change |
| --- | ---: | ---: | ---: |
| `2048 / 2 / 1` | 0.024290 | 0.035590 | `1,942.0 -> 1,955.4` |
| `2048 / 2 / 2` | 0.065755 | 0.075118 | `3,401.5 -> 3,577.9` |
| `2048 / 2 / 4` | 0.142997 | 0.143152 | `6,294.4 -> 6,560.6` |
| `1024 / 2 / 4` | 0.223168 | 0.231368 | `11,726.1 -> 12,251.1` |
| `512 / 8 / 4` | 0.475524 | 0.504433 | `68,848.4 -> 71,420.0` |

## Readout

Alignment is directionally useful but not sufficient. It improves the widest high-recall point and slightly lifts the narrowest `2048` rows, but the low-thousands candidate band remains below the single-write IVF frontier. The closest row to the prior `0.234422` IVF recall is `1024 cells / 2 writes / 4 probes`, which reaches `0.231368` at `12.3k` projected candidates, not the roughly `3.5k` candidate scale where IVF achieved that recall.

The failure mode is now sharper: a fixed number of learned hooks with hard capacity can be balanced, and it can be trained to agree with query routes, but the assignment is still too local and myopic. It decides where each page lives without optimizing the whole catalog's future retrieval coverage.

## Next Implementation

The next Lantern test should make write assignment part of the objective:

- use a balanced assignment layer or Sinkhorn-style relaxation so capacity is visible during training;
- train against route coverage under `top_p` query probes rather than only single-key overlap;
- compare directly against the current IVF/PQ frontier in the `2k-12k` projected-candidate band.

If that does not move recall-per-candidate substantially, Lantern should remain a side branch and the mainline should return to the static inverted/PQ serving path.
