# Late4 Answer-KL Plus Gold-CE Distillation Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Frozen SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket layers: `26-29`
- Tight SVA budget: `shortlist=512`, `budget=128`
- Adapter: rank-16 residual output adapter on top of SVA attention
- Adapter params: `110592`
- Objective: answer-token KL to full attention plus `0.01 *` gold-answer CE
- Adapter bundle: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- Modal distill call: `fc-01KRM22MVB6RQQKYTXFCDXCN4B`
- Modal answer-decode call: `fc-01KRM2CXQ018YMN1WCXH35AYBE`

## Distillation Result

| Phase | KL to full | Top-1 agreement | Logit cosine |
|---|---:|---:|---:|
| baseline train | `0.076646` | `0.821429` | `0.998917` |
| distilled train | `0.029528` | `0.928571` | `0.999374` |
| baseline eval | `0.052571` | `0.904762` | `0.998111` |
| distilled eval | `0.020634` | `0.952381` | `0.998561` |

The CE term was large enough to matter while preserving most of the answer-token KL gain from pure answer distillation. Held-out KL landed close to the pure answer-distilled adapter (`0.020634` versus `0.019472`).

## Full Answer-Decode Result

The saved adapter was evaluated on 9 held-out 32K passkey cases: three keys times start/middle/end placement. The same run measured unadapted SVA at the same tight budget.

| Variant | Answer NLL delta | KL to full | Top-1 agreement | Logit cosine | Decode exact-read reduction |
|---|---:|---:|---:|---:|---:|
| SVA unadapted `512/128` | `-0.013716` | `0.080711` | `0.809524` | `0.998966` | `256x` |
| SVA answer-KL+CE `512/128` | `-0.505290` | `0.030835` | `0.936508` | `0.999417` | `256x` |

## Interpretation

This is the strongest tight-budget late4 adaptation result so far. Compared with unadapted `512/128` SVA, the combined objective cut answer KL by about `62%`, improved top-1 agreement by about `12.7` points, and improved gold-answer NLL by about `0.49` while preserving the `256x` decode exact-read reduction inside the SVA layers.

Compared with pure answer-token distillation, the combined objective kept similar distribution fidelity and recovered strong retrieval pressure:

| Adapter | Answer NLL delta | KL to full | Top-1 agreement |
|---|---:|---:|---:|
| final-prompt adapter | `-0.170994` | `0.070306` | `0.825397` |
| answer-token adapter | `0.065083` | `0.027706` | `0.920635` |
| answer-KL+CE adapter | `-0.505290` | `0.030835` | `0.936508` |

The next sharp test is a small CE-weight sweep around this point, likely `0.003`, `0.01`, and `0.03`, evaluated on the same 9-case panel and then on a larger held-out set if the shape holds.
