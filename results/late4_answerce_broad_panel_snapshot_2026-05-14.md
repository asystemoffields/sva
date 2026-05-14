# Late4 Answer-KL+CE Broad Held-Out Panel Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Adapter: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- Socket: late4, replacing layers `26-29`
- Budget: `summon_topk=512`, `verify_topk=128`
- Context: `32768`
- Held-out keys: `219384`, `407615`, `592806`, `638174`, `750291`, `826430`, `319057`, `460128`
- Placements: `start`, `middle`, `end`
- Cases: `24`
- Modal call: `fc-01KRM4XQYG14B956G2B5ZC6N6N`
- Runner: `modal_h100_late4_answerce_broad_panel.py`

## Result

| Variant | Cases | Answer NLL Delta | Answer KL To Full | Top-1 Agreement | Logit Cosine | Prefill Slowdown | Decode Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unadapted late4 SVA | 24 | `-0.029969` | `0.096806` | `0.851190` | `0.998147` | `21.883706` | `1.880294` |
| Answer-KL+CE adapter | 24 | `-0.474124` | `0.031916` | `0.946429` | `0.999103` | `21.768739` | `1.823752` |

The combined answer-token KL plus `0.01 * gold_CE` adapter generalized to a larger unseen 32K passkey panel. Compared with the unadapted `512/128` late4 SVA control, it cut answer KL by about `67%`, improved top-1 agreement by about `9.5` points, improved logit cosine, and made the gold-answer NLL delta substantially better while preserving the same `256x` decode exact-read reduction in SVA layers.

The remaining production blocker is wall-clock: this scan harness still traverses the context during summon, so prefill remains about `21.8x` slower than full attention and decode remains slower despite the value-read reduction. The quality-side result is strong enough to keep pursuing indexed/cached summon and serving-shaped kernels instead of spending the next step on a larger version of the same validation panel.

## Next Test

Run a focused CE-weight sweep around `0.01` only if needed to lock the adapter objective. The higher-information production step is now an indexed/cached summon benchmark for this same late4 `512/128` adapted path, because the broad-panel result says the adapter is good enough to measure systems wins against a credible quality target.
