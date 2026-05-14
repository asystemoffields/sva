# Late4 Answer-KL+CE Tight Inverted Panel Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Adapter: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket: late4, replacing layers `26-29`
- Budget: `summon_topk=512`, `verify_topk=128`
- Summon mode: `inverted`
- Context: `32768`
- Cases: 24 held-out passkey cases, matching the broad panel
- Modal app: `ap-8AiHOzevA5V4MFHX7ENk7x`
- Function calls: `fc-01KRMAQ3NW81M79MRESQ62MTSG` and `fc-01KRMAQ3V0Q20HA8ZFKJPCWDXD`
- Runner: `modal_h100_late4_answerce_inverted_panel.py --cells 4,8`

## Result

| Cells/Subspace | Variant | Cases | Answer NLL Delta | Answer KL To Full | Top-1 Agreement | Logit Cosine | Prefill Slowdown | Decode Slowdown |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | Unadapted late4 SVA | 24 | `-0.115141` | `0.090830` | `0.875000` | `0.990243` | `25.848621` | `3.131101` |
| 4 | Answer-KL+CE adapter | 24 | `-0.452635` | `0.082317` | `0.904762` | `0.989791` | `25.723788` | `3.103107` |
| 8 | Unadapted late4 SVA | 24 | `-0.031427` | `0.062724` | `0.869048` | `0.995300` | `28.045587` | `3.149548` |
| 8 | Answer-KL+CE adapter | 24 | `-0.425858` | `0.038735` | `0.910714` | `0.992045` | `27.853850` | `3.109223` |

For comparison:

| Summon | Cells/Subspace | Adapter KL | Adapter Top-1 | Adapter Cosine | Adapter NLL Delta | Decode Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scan | n/a | `0.031916` | `0.946429` | `0.999103` | `-0.474124` | `1.823752` |
| Inverted | 4 | `0.082317` | `0.904762` | `0.989791` | `-0.452635` | `3.103107` |
| Inverted | 8 | `0.038735` | `0.910714` | `0.992045` | `-0.425858` | `3.109223` |
| Inverted | 16 | `0.032604` | `0.922619` | `0.999239` | `-0.434830` | `3.296342` |
| Inverted | 32 | `0.028585` | `0.898810` | `0.999453` | `-0.320555` | `3.340533` |

## Readout

Tighter indexed summon reduces the candidate pool, but the current wall-clock barely responds. Runtime rows show decode candidates around hundreds for `4` cells/subspace and around low thousands for `8`, versus several thousand at `16` and `32`. Decode slowdown stayed near `3.1x` across `4` and `8`, so the present bottleneck is no longer simply candidate count. It is the Python/posting-list/gather path used to construct and score candidates.

Quality has a clear knee. `8` cells/subspace is the tightest indexed setting that stays near the scan adapter on KL; `4` cells/subspace preserves answer NLL but drifts on KL and cosine. The best quality/cost target among these indexed runs is `8` or `16`, depending on whether the next implementation can make posting-list candidate construction cheap.

## Next Test

Run `1` and `2` cells/subspace as an overhead floor. If decode slowdown remains near `3x`, a custom/vectorized posting path is required before this inverted route can compete wall-clock. If slowdown finally drops, then a capped or adaptive cell policy becomes the next method-level target.
