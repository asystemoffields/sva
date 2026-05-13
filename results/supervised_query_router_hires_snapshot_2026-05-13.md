# High-Resolution Supervised Query Router Snapshot

Date: 2026-05-13

## Question

The low-resolution supervised query-cell router recovered much more recall when it could summon broadly, but it was far too dense for million-token serving. This test asks whether increasing the cell count can keep the supervised signal while moving into the target projected-candidate band.

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: SmolLM2 configured window, `8192` tokens
- Train text: generated long stream
- Eval text: reversed generated stream
- Layers: `0,1,5,10,18,24,29`
- Ranker: asymmetric learned Q/K projection, rank `64`
- Read cells: k-means over `4096` learned low-rank train queries
- Write model: per-head key-to-cell vectors trained from full-attention top-key labels
- Query writes during training: `1`
- Cells: `2048,4096`
- Key writes: `1,2,4`
- Query probes: `1,2`
- Target context projection: `1,000,000` tokens
- Verifier budgets: `256,512`
- Full-attention target: top-16 keys per sampled query/head

Run:

```text
ap-3pG3Uzgkb1iEruukL7fhcs
fc-01KRHSSHTRTCPXXYHJD6MYN60B
results/modal_runs/sva-h100-supervised-query-router-hires-20260513-191215.modal.log
```

## Aggregate Result

Budgets `256` and `512` were identical in this sweep because the raw candidate sets were smaller than both budgets. The table shows the `512` rows.

| cells | key writes | probes | avg 8192 candidates | avg projected 1M candidates | p95 projected 1M candidates | verified top-16 recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 4 | 2 | 29.2 | 3,568.1 | 14,770.5 | 0.042721 |
| 2048 | 4 | 1 | 15.2 | 1,850.0 | 7,812.5 | 0.025143 |
| 2048 | 2 | 2 | 15.2 | 1,852.8 | 8,056.6 | 0.022662 |
| 2048 | 2 | 1 | 7.8 | 953.4 | 4,150.4 | 0.013563 |
| 4096 | 4 | 2 | 8.2 | 999.5 | 4,150.4 | 0.012680 |
| 2048 | 1 | 2 | 7.7 | 945.4 | 4,272.5 | 0.011316 |
| 4096 | 4 | 1 | 4.1 | 497.8 | 2,252.2 | 0.007285 |
| 4096 | 2 | 2 | 4.1 | 506.1 | 2,197.3 | 0.006433 |
| 2048 | 1 | 1 | 3.9 | 480.2 | 2,075.2 | 0.006479 |
| 4096 | 2 | 1 | 2.1 | 250.5 | 1,098.6 | 0.003643 |
| 4096 | 1 | 2 | 2.1 | 256.0 | 1,098.6 | 0.003162 |
| 4096 | 1 | 1 | 1.0 | 127.6 | 610.4 | 0.001891 |

## Interpretation

This is a kill for the current supervised query-cell router. It reaches the intended projected million-token band, but recall is only about `0.002-0.013` inside `128-1024` projected candidates. At `3.6k` projected candidates it reaches `0.042721`, still well below single-write IVF's `0.234422` at about `3.5k`.

The low-resolution version proved that top-key supervision can increase recall when cells are broad. The high-resolution version shows that this particular read-cell/write-cell factorization does not preserve the useful signal when cells become selective.

## Next Test

Move away from hard query-cell routing and toward score-preserving compressed lookup:

1. Keep the learned rank-64 Q/K scorer.
2. Quantize the low-rank key space into product codebooks or another asymmetric score-preserving catalog.
3. Let the query score compressed key codes directly before exact verification.
4. Measure whether approximate learned-score retrieval can reach the `128-1024` projected-candidate band without collapsing recall.

