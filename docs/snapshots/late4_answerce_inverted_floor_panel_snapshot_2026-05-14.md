# Late4 Answer-KL+CE Inverted Overhead-Floor Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Adapter: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket: late4, replacing layers `26-29`
- Budget: `summon_topk=512`, `verify_topk=128`
- Summon mode: `inverted`
- Context: `32768`
- Cases: 24 held-out passkey cases, matching the broad panel
- Modal app: `ap-HBV1RDvV2uTxOcbc0SUzYI`
- Function calls: `fc-01KRMDH445XMR2WGFVEDHR2MB8` and `fc-01KRMDH4B4P6V7ED17D5KMP3W2`
- Runner: `modal_h100_late4_answerce_inverted_panel.py --cells 1,2`

## Result

| Cells/Subspace | Variant | Cases | Answer NLL Delta | Answer KL To Full | Top-1 Agreement | Logit Cosine | Prefill Slowdown | Decode Slowdown |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Unadapted late4 SVA | 24 | `-0.618893` | `0.236993` | `0.732143` | `0.997155` | `25.712892` | `3.029818` |
| 1 | Answer-KL+CE adapter | 24 | `-0.933832` | `0.166606` | `0.821429` | `0.997689` | `25.787808` | `3.005931` |
| 2 | Unadapted late4 SVA | 24 | `-0.426227` | `0.126494` | `0.875000` | `0.997700` | `26.537996` | `3.342966` |
| 2 | Answer-KL+CE adapter | 24 | `-0.710121` | `0.088524` | `0.886905` | `0.998575` | `26.612105` | `3.329559` |

## Readout

This is the overhead-floor result. The candidate pool is much smaller than the `4/8/16/32` indexed settings, but decode remains about `3x` slower than full attention and slower than the scan-summon broad panel (`1.823752x`). Quality also drifts: `2` cells/subspace has much worse KL than `8` while `1` cells/subspace is clearly off-distribution relative to full attention.

The result isolates the wall-clock issue. Simply reducing cells/candidates does not make the current inverted path competitive. The fixed cost is the implementation shape: per-head Python loops, posting-list union, duplicate handling, and gather/scatter-like candidate construction before the verifier. The next productive step is to replace that path with a vectorized/static candidate builder, then retest the `8` or `16` cells quality setting.

## Next Implementation

Build a vectorized inverted decode path that:

- computes top cells for all heads at once,
- gathers fixed padded posting slots without per-head Python loops,
- avoids full candidate-list `unique` before ranking,
- masks duplicate verified slots after the final top-k,
- keeps the exact verifier unchanged.

Then rerun the same 24-case panel at `8` and `16` cells/subspace. Those are the quality-relevant indexed settings; `1/2` are useful only as an overhead diagnostic.
