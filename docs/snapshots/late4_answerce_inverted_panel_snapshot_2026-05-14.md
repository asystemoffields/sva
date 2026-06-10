# Late4 Answer-KL+CE Inverted Panel Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Adapter: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket: late4, replacing layers `26-29`
- Budget: `summon_topk=512`, `verify_topk=128`
- Summon mode: `inverted`
- Context: `32768`
- Cases: 24 held-out passkey cases, matching `results/late4_answerce_broad_panel_snapshot_2026-05-14.md`
- Modal app: `ap-p29Bx7ikVB8Kjb2d55Abo6`
- Function calls: `fc-01KRM7W9EFC5V89BYFDPX9S57Q` and `fc-01KRM7W9JR2CX7ZM10FQ27HRM4`
- Runner: `modal_h100_late4_answerce_inverted_panel.py`

## Result

| Cells/Subspace | Variant | Cases | Answer NLL Delta | Answer KL To Full | Top-1 Agreement | Logit Cosine | Prefill Slowdown | Decode Slowdown |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Unadapted late4 SVA | 24 | `-0.014172` | `0.073774` | `0.821429` | `0.998223` | `25.578618` | `3.316947` |
| 16 | Answer-KL+CE adapter | 24 | `-0.434830` | `0.032604` | `0.922619` | `0.999239` | `25.519320` | `3.296342` |
| 32 | Unadapted late4 SVA | 24 | `0.091863` | `0.067197` | `0.857143` | `0.998340` | `27.551360` | `3.377197` |
| 32 | Answer-KL+CE adapter | 24 | `-0.320555` | `0.028585` | `0.898810` | `0.999453` | `27.580174` | `3.340533` |

For comparison, the prior scan-summon broad-panel adapter row was answer KL `0.031916`, top-1 `0.946429`, cosine `0.999103`, NLL delta `-0.474124`, prefill slowdown `21.768739`, and decode slowdown `1.823752`.

## Readout

The indexed decode path preserved most of the CE001 adapter quality at 16 cells/subspace: KL was nearly identical to the scan result, cosine was slightly higher, and NLL improvement stayed large. Top-1 agreement dropped from `0.946429` to `0.922619`.

The current implementation did not improve wall-clock. Both inverted settings were slower than scan in this harness, with decode slowdown around `3.3x` versus scan's `1.82x`. The likely reason is visible in the runtime stats: inverted decode verifies the same ~128 tokens, but it summons the union of posting lists before exact verification. At 16 cells/subspace the per-row decode summoned count is roughly in the low thousands; at 32 it is roughly twice that. This says the path is quality-credible, but the candidate pool is still too wide for the present PyTorch implementation.

## Next Test

Run the same 24-case panel with tighter indexed budgets, `4` and `8` cells/subspace. If quality remains close, the indexed path becomes a practical target for optimization. If quality breaks, the next systems step should be a capped posting-list policy that preserves high-score cells while bounding candidate count before exact verification.
