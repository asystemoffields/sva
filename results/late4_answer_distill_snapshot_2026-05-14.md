# Late4 Answer-Token Distillation Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Frozen SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket layers: `26-29`
- Tight SVA budget: `shortlist=512`, `budget=128`
- Adapter: rank-16 residual output adapter on top of SVA attention
- Adapter params: `110592`
- Adapter bundle: `results/hf_artifacts/sva-late4-512x128-answerdistill-v1`
- Modal distill call: `fc-01KRM17WY59D9H8T968J7KETTF`
- Modal answer-decode call: `fc-01KRM1KH47V40PK9TVCS55GRHQ`

## What Changed

The previous adapter trained on the final prompt logit only. This run trained on all answer-token logits by appending the gold answer prefix and gathering the causal logits for each answer token position.

That turns the target from "match the doorway into decode" into "match the distribution through the answer decode path."

## Answer-Token Distillation Result

| Phase | KL to full | Top-1 agreement | Logit cosine |
|---|---:|---:|---:|
| baseline train | `0.076646` | `0.821429` | `0.998917` |
| distilled train | `0.026147` | `0.928571` | `0.999433` |
| baseline eval | `0.052571` | `0.904762` | `0.998111` |
| distilled eval | `0.019472` | `0.952381` | `0.998908` |

## Full Answer-Decode Result

The saved answer-distilled adapter was evaluated on 9 held-out 32K passkey cases: three keys times start/middle/end placement. The same run measured unadapted SVA at the same tight budget.

| Variant | Answer NLL delta | KL to full | Top-1 agreement | Logit cosine | Decode exact-read reduction |
|---|---:|---:|---:|---:|---:|
| SVA unadapted `512/128` | `-0.012770` | `0.080789` | `0.809524` | `0.998958` | `256x` |
| SVA answer-distilled `512/128` | `0.065083` | `0.027706` | `0.920635` | `0.999429` | `256x` |

## Interpretation

Answer-token distillation gives the strongest drop-in fidelity result so far at the tight `512/128` late4 budget. It cuts answer KL by about `66%` versus unadapted SVA and materially improves top-1 agreement.

The tradeoff is gold-answer NLL: the answer-distilled adapter is closer to the full-attention distribution, while the final-prompt adapter assigned more probability to the gold passkey. That suggests the next objective should combine answer-token KL with a small gold-answer CE term, and compare two separately useful modes:

- fidelity mode: preserve full-attention logits as closely as possible;
- retrieval mode: preserve or improve gold-answer probability under long-context passkey stress.
