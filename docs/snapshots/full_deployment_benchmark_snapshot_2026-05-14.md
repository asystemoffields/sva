# Full Deployment Benchmark Snapshot - 2026-05-14

This snapshot records the first held-out deployment benchmark for socketed Summon-Verify Attention.

## Run

- Modal app: `ap-J4NiU8j8O8YJ6pl9QRDOv2`
- Function call: `fc-01KRJBA8FZVYWR8JC3Y6CK0PEK`
- Dashboard: https://modal.com/id/fc-01KRJBA8FZVYWR8JC3Y6CK0PEK
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Hardware: H100
- Calibration: 6 built-in calibration documents, repeated to `4096` tokens
- Eval: 4 held-out built-in documents
- Layers: all 30 socketed
- Route source: Q/K
- Artifact training: teacher
- Lookup: `4x64` coarse PQ, hard-negative trained rank-64 scorer, attention-weighted codebooks
- Contexts: `2048`, `4096`
- Shortlists: `512`, `1024`
- Verifier budgets: `128`, `256`, `512`

## Aggregate Results

| context | shortlist | budget | loss_delta | KL | top1 agreement | logit cosine | verified top16 recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 512 | 128 | 0.000977 | 0.001193 | 0.999145 | 0.929682 | 0.983658 |
| 2048 | 512 | 256 | 0.000000 | 0.000381 | 0.999145 | 0.979829 | 0.988596 |
| 2048 | 512 | 512 | 0.000000 | 0.000230 | 0.999145 | 0.990611 | 0.989009 |
| 2048 | 1024 | 128 | 0.000489 | 0.000718 | 0.999145 | 0.965610 | 0.991925 |
| 2048 | 1024 | 256 | 0.000000 | 0.000302 | 0.999145 | 0.988006 | 0.998169 |
| 2048 | 1024 | 512 | 0.000000 | 0.000165 | 0.999145 | 0.998157 | 0.999083 |
| 4096 | 512 | 128 | 0.002442 | 0.001992 | 0.999390 | 0.913694 | 0.973875 |
| 4096 | 512 | 256 | 0.001343 | 0.001118 | 0.999451 | 0.945508 | 0.978512 |
| 4096 | 512 | 512 | 0.001221 | 0.000805 | 0.999451 | 0.960045 | 0.978896 |
| 4096 | 1024 | 128 | 0.001343 | 0.001035 | 0.999511 | 0.950904 | 0.988142 |
| 4096 | 1024 | 256 | 0.000855 | 0.000494 | 0.999511 | 0.976958 | 0.994829 |
| 4096 | 1024 | 512 | 0.000488 | 0.000244 | 0.999511 | 0.990862 | 0.995900 |

## Readout

This is a cleaner test than the earlier deployment proxy because the artifacts are frozen from calibration documents and evaluated on held-out documents. The result is still positive: quality remains close to full attention across the tested contexts, and the `1024/512` setting is very strong.

The timing columns from the run show the current Python socket path is much slower than full attention. That is expected for this reference harness because it rebuilds product codes and performs coarse scoring inside each attention forward. The quality result is the scientific signal here; a serving implementation would cache key-side codes and move the lookup into a fused or specialized path before the timing result becomes meaningful.

## Next Step

Run the same held-out benchmark at SmolLM2's configured `8192` window, then add a cached-key benchmark that separates prefill construction cost from per-token decode lookup cost.
