# Full Deployment 8192 Snapshot - 2026-05-14

This snapshot records the first completed held-out all-layer socket benchmark at SmolLM2's configured `8192` token window.

## Run

- Modal app: `ap-EMJVWsLhmodMDR7rHuWEP3`
- Function call: `fc-01KRJCDR4QB7SVQVZTBEDDBT7Z`
- Dashboard: https://modal.com/id/fc-01KRJCDR4QB7SVQVZTBEDDBT7Z
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Hardware: H100
- Calibration: 6 built-in calibration documents, repeated to `8192` tokens
- Eval: 4 held-out built-in documents, `8192` tokens each
- Layers: all 30 socketed
- Route source: Q/K
- Artifact training: teacher
- Lookup: `4x64` coarse PQ, hard-negative trained rank-64 scorer, attention-weighted codebooks
- Shortlists: `512`, `1024`, `2048`
- Verifier budgets: `128`, `256`, `512`

The first attempt at `8192` hit OOM at shortlist `2048` because the reference socket built the query-by-shortlist low-rank rescore tensor for the full sequence at once. The socket now processes long query windows in chunks, and the rerun completed.

## Aggregate Results

| context | shortlist | budget | loss_delta | KL | top1 agreement | logit cosine | verified top16 recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 512 | 128 | 0.003601 | 0.003918 | 0.999634 | 0.909783 | 0.964981 |
| 8192 | 512 | 256 | 0.002808 | 0.002823 | 0.999634 | 0.936224 | 0.967109 |
| 8192 | 512 | 512 | 0.002075 | 0.002098 | 0.999573 | 0.948567 | 0.966662 |
| 8192 | 1024 | 128 | 0.002442 | 0.002399 | 0.999786 | 0.941799 | 0.986941 |
| 8192 | 1024 | 256 | 0.001587 | 0.001256 | 0.999756 | 0.967090 | 0.991203 |
| 8192 | 1024 | 512 | 0.000732 | 0.000564 | 0.999756 | 0.984127 | 0.991471 |
| 8192 | 2048 | 128 | 0.002259 | 0.002235 | 0.999786 | 0.949199 | 0.992517 |
| 8192 | 2048 | 256 | 0.001526 | 0.001155 | 0.999786 | 0.971767 | 0.997809 |
| 8192 | 2048 | 512 | 0.000794 | 0.000481 | 0.999786 | 0.989240 | 0.998707 |

## Readout

The 8192 deployment result is still a go. The smaller `512` shortlist degrades under the longer context, `1024` remains usable, and `2048/512` restores very high top-key survival while preserving the next-token distribution closely.

The all-layer socket quality is now tested at the model's full configured context window with frozen artifacts and held-out documents. The reference socket timing is still slow because it is a Python/PyTorch implementation optimized for measurement clarity, not serving.

## Next Step

The next combined test should move beyond full-sequence prefill timing and into serving mechanics: cached key-side codes, decode lookup, and eventually a fused or custom lookup path that avoids the current PyTorch loop overhead.
