# Late4 Saved Adapter Full Answer-Decode Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Frozen SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket layers: `26-29`
- Tight SVA budget: `shortlist=512`, `budget=128`
- Adapter: rank-16 residual output adapter on top of SVA attention
- Adapter params: `110592`
- Adapter bundle: `results/hf_artifacts/sva-late4-512x128-adapter-v1`
- Modal distill call: `fc-01KRM0EX1E01857RH5BME82ZNT`
- Modal answer-decode call with unadapted control: `fc-01KRM0XYY3HHK97KDQVKA66ECD`

## Final-Prompt Distillation Result

The adapter was trained for 24 steps against full-attention final-prompt logits at 32K.

| Phase | KL to full | Top-1 agreement | Logit cosine |
|---|---:|---:|---:|
| baseline train | `0.060001` | `1.000000` | `0.998997` |
| distilled train | `0.011410` | `1.000000` | `0.999713` |
| baseline eval | `0.045040` | `0.666667` | `0.998986` |
| distilled eval | `0.009248` | `1.000000` | `0.999712` |

## Full Answer-Decode Result

The saved adapter was evaluated on 9 held-out 32K passkey cases: three keys times start/middle/end placement. The same run measured unadapted SVA at the same tight budget.

| Variant | Answer NLL delta | KL to full | Top-1 agreement | Logit cosine | Decode exact-read reduction |
|---|---:|---:|---:|---:|---:|
| SVA unadapted `512/128` | `-0.014381` | `0.080839` | `0.809524` | `0.998957` | `256x` |
| SVA adapted `512/128` | `-0.170994` | `0.070306` | `0.825397` | `0.999289` | `256x` |

Wall-clock in this Python scan harness remains dominated by prefill summon. Mean prefill slowdown was about `20.5x` for both SVA variants; mean decode slowdown was `1.81x` unadapted and `1.91x` adapted. These timings should be read as harness pressure, while the value-read reduction is the method signal.

## Interpretation

SVA-active adaptation improved the tight-budget full answer-decode panel, not just final-prompt logits. The improvement is modest on answer-token KL/top-1 compared with the much larger final-prompt gain, which points to the next training target: distill or train on the answer-token decode distribution directly.

The current result supports continued fine-tuning work at tight budgets. The strongest next run is answer-token distillation over multiple keys and placements, then a repeat of this 9-case panel.
