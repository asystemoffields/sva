# Late4 Answer-KL+CE Static Inverted Panel Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Adapter: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket: late4, replacing layers `26-29`
- Budget: `summon_topk=512`, `verify_topk=128`
- Summon mode: `inverted_static`
- Context: `32768`
- Cases: 24 held-out passkey cases, matching the broad panel
- Modal app: `ap-UbybE4XIeBmOA52suapzcD`
- Function calls: `fc-01KRMFBNY3KMXC2Z77AT154JC1` and `fc-01KRMFBP262DDGYECK6XE678QS`
- Runner: `modal_h100_late4_answerce_inverted_panel.py --cells 8,16 --summon-mode inverted_static`

## Result

| Cells/Subspace | Variant | Cases | Answer NLL Delta | Answer KL To Full | Top-1 Agreement | Logit Cosine | Prefill Slowdown | Decode Slowdown | Decode Avg Summoned | Decode Avg Verified |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | Unadapted late4 SVA | 24 | `-0.027485` | `0.064254` | `0.886905` | `0.998844` | `27.318287` | `2.340806` | `1483.235532` | `108.065972` |
| 8 | Answer-KL+CE adapter | 24 | `-0.421458` | `0.040815` | `0.916667` | `0.991645` | `26.717886` | `2.276378` | `1481.547840` | `107.866319` |
| 16 | Unadapted late4 SVA | 24 | `-0.003146` | `0.080441` | `0.845238` | `0.998056` | `26.522149` | `1.936994` | `3151.603588` | `97.258873` |
| 16 | Answer-KL+CE adapter | 24 | `-0.435238` | `0.034456` | `0.910714` | `0.999215` | `26.108347` | `1.872479` | `3145.725887` | `97.369985` |

For comparison:

| Summon | Cells/Subspace | Adapter KL | Adapter Top-1 | Adapter Cosine | Adapter NLL Delta | Decode Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scan | n/a | `0.031916` | `0.946429` | `0.999103` | `-0.474124` | `1.823752` |
| Inverted | 8 | `0.038735` | `0.910714` | `0.992045` | `-0.425858` | `3.109223` |
| Inverted | 16 | `0.032604` | `0.922619` | `0.999239` | `-0.434830` | `3.296342` |
| Static inverted | 8 | `0.040815` | `0.916667` | `0.991645` | `-0.421458` | `2.276378` |
| Static inverted | 16 | `0.034456` | `0.910714` | `0.999215` | `-0.435238` | `1.872479` |

## Readout

The vectorized/static candidate builder is a real wall-clock improvement. At the quality-relevant `16` cells/subspace setting, adapted decode slowdown dropped from `3.296342x` in the old inverted path to `1.872479x`, while KL, cosine, and NLL delta stayed close to the old indexed result. This puts static inverted decode near the scan-summon row (`1.823752x`) while avoiding the old per-head Python candidate-union path.

The remaining bottleneck is still candidate construction and gather shape. Static inverted verifies fewer than the nominal `128` tokens on average after duplicate masking (`~97` at `16` cells), which is acceptable for this speed probe but should be tightened before treating this as the stable architecture. Prefill is still slow because this implementation only changes single-token decode; prefill still uses the broader scan-shaped path.

## Next Implementation

Turn static inverted from a speed probe into a verifier-ready retrieval path:

- replace post-top-k duplicate masking with capped unique plus refill, so each head can reliably verify the intended budget,
- keep postings in a layout that directly emits a fixed candidate tensor per head,
- add a decode-only benchmark that separates catalog lookup, exact scoring, value gather, and softmax/value aggregation,
- retest `16` cells/subspace first, because it is the current best quality/speed point.

The target is a summon output already shaped like the verifier input, so decode avoids building, sorting, and cleaning a pile before exact attention runs.
