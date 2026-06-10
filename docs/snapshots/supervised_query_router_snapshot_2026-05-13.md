# Supervised Query Router Snapshot

Date: 2026-05-13

## Question

The multi-write IVF result suggested that unsupervised key geometry is the bottleneck. This test asks whether a supervised catalog can do better by learning cells from query contexts that actually request full-attention top keys.

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: SmolLM2 configured window, `8192` tokens
- Train text: generated long stream
- Eval text: reversed generated stream
- Layers: `0,1,5,10,18,24,29`
- Ranker: asymmetric learned Q/K projection, rank `64`
- Read cells: k-means over learned low-rank train queries
- Write model: per-head key-to-cell vectors trained from full-attention top-key labels
- Query writes during training: `1`
- Cells: `256,512`
- Key writes: `4,8,16`
- Query probes: `1,2,4`
- Target context projection: `1,000,000` tokens
- Verifier budgets: `256,512`
- Full-attention target: top-16 keys per sampled query/head

Run:

```text
ap-inChP8iHpkN3bsHX5lYDf8
fc-01KRHSGKW3RX8Z25MSGCWVCTPC
results/modal_runs/sva-h100-supervised-query-router-20260513-190721.modal.log
```

## Aggregate Result

The table uses the `512` verifier budget rows.

| cells | key writes | probes | avg 8192 candidates | avg projected 1M candidates | p95 projected 1M candidates | verified top-16 recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16 | 4 | 1,367.9 | 166,978.7 | 411,828.6 | 0.655816 |
| 256 | 16 | 2 | 820.5 | 100,154.5 | 288,195.8 | 0.485677 |
| 256 | 8 | 4 | 840.8 | 102,635.4 | 276,733.4 | 0.482670 |
| 512 | 16 | 4 | 447.4 | 54,608.8 | 133,288.6 | 0.336697 |
| 256 | 8 | 2 | 473.9 | 57,843.8 | 183,343.5 | 0.328203 |
| 256 | 4 | 4 | 490.7 | 59,901.6 | 180,651.9 | 0.320390 |
| 256 | 16 | 1 | 444.3 | 54,234.6 | 184,448.2 | 0.319615 |
| 512 | 16 | 2 | 235.5 | 28,753.2 | 80,377.2 | 0.214813 |
| 512 | 8 | 4 | 232.7 | 28,400.1 | 75,293.0 | 0.202164 |
| 256 | 4 | 2 | 264.7 | 32,311.4 | 110,760.5 | 0.201823 |
| 256 | 8 | 1 | 244.2 | 29,814.3 | 114,166.3 | 0.200676 |
| 512 | 16 | 1 | 121.4 | 14,816.5 | 47,241.2 | 0.129449 |
| 512 | 8 | 2 | 119.1 | 14,539.4 | 43,933.1 | 0.123760 |
| 256 | 4 | 1 | 132.0 | 16,117.4 | 67,193.6 | 0.117854 |
| 512 | 4 | 4 | 118.9 | 14,519.6 | 41,992.2 | 0.114490 |
| 512 | 8 | 1 | 60.5 | 7,383.1 | 26,123.0 | 0.072529 |
| 512 | 4 | 2 | 59.9 | 7,307.2 | 23,736.6 | 0.068669 |
| 512 | 4 | 1 | 30.2 | 3,682.7 | 13,726.8 | 0.039109 |

## Interpretation

The supervised objective moves recall much more aggressively than unsupervised multi-write IVF when it is allowed to summon broadly. For example, `256 cells / 16 writes / 4 probes` reaches `0.655816` aggregate recall, far above the IVF rows, but it projects to about `167k` candidates at a million-token context.

At the smallest candidate setting in this run, `512 cells / 4 writes / 1 probe`, recall is only `0.039109` at about `3.7k` projected candidates. Single-write IVF had already reached `0.234422` at about `3.5k` projected candidates. This version learned a useful supervised signal, but its cells are too coarse and too dense.

## Next Test

The next sharp test is a high-resolution supervised router:

1. Increase read cells to `2048-4096`.
2. Increase train query samples so the read cells have enough query coverage.
3. Sweep smaller key-write/probe settings such as `1,2,4` writes and `1,2` probes.
4. Keep the target projected million-token candidate band near `128-1024`.

