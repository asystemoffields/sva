# Three-Stage Socket Snapshot - 2026-05-13

This snapshot records the first SmolLM2 three-stage SVA socket tests:

1. coarse product-quantized summon,
2. exact rank-64 rescore inside the shortlist,
3. exact full-dimensional QK/V attention over the final budget.

The model is `HuggingFaceTB/SmolLM2-135M-Instruct` at `seq_len=2048`, using `4x64` coarse PQ, `coarse_shortlist=1024`, and `budget=512` unless noted.

## All-Layer First Run

| seq_len | shortlist | budget | loss_delta | KL | top1 agreement | logit cosine | candidate top16 recall | verified top16 recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 512 | 256 | 1.015625 | 1.014542 | 0.791789 | 0.600218 | 0.726999 | 0.721794 |
| 2048 | 1024 | 512 | 1.078125 | 1.072883 | 0.804104 | 0.563977 | 0.728127 | 0.722376 |

The verifier is not the main failure point in this run: candidate recall and verified recall are nearly identical. The missing attention targets are being lost during coarse summon.

## Layer-Isolation Follow-Up

This run used the stronger hard-negative settings from the offline lookup tests: `ranker_train_steps=160`, `coarse_hard_steps=80`.

| socketed layers | artifacts trained | loss_delta | KL | top1 agreement | logit cosine | candidate top16 recall | verified top16 recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.000000 | 0.000303 | 0.994626 | 0.999132 | 0.995712 | 0.995280 |
| 10 | 1 | 0.000000 | 0.000343 | 0.996580 | 0.996232 | 0.854464 | 0.854298 |
| 18 | 1 | 0.000000 | 0.000821 | 0.997557 | 0.984490 | 0.785752 | 0.785347 |
| 29 | 1 | 0.000000 | 0.000788 | 0.997069 | 0.988675 | 0.736012 | 0.735420 |
| 0,1,2,3 | 4 | 0.007812 | 0.009403 | 0.993161 | 0.831774 | 0.840575 | 0.839930 |
| 0,5,10,18,24,29 | 6 | 0.003906 | 0.003095 | 0.992672 | 0.963463 | 0.842612 | 0.841965 |
| all | 30 | 1.562500 | 1.563977 | 0.718124 | 0.520247 | 0.720604 | 0.718463 |

## Interpretation

The layer-isolated result is positive. A single socketed layer stays almost indistinguishable from full attention at the logit level, even when the router's local top-16 recall is imperfect. A sparse six-layer socket also stays close: `loss_delta=0.003906`, `KL=0.003095`, and `top1_agreement=0.992672`.

The all-layer failure is therefore most likely a compounding representation-shift problem. Each socketed layer changes the hidden-state distribution slightly, and later learned catalogs are then queried outside the teacher-state distribution they were trained on. The effect is visible in late-layer recall: layer `29` has `0.736012` candidate recall when socketed alone, `0.721756` in the sparse six-layer setting, and `0.582462` when every layer is socketed.

## Next Step

The next test should train the router in the state distribution it will actually serve:

- patch a prefix or selected set of layers,
- run the model to collect socketed hidden states,
- train the next layer's SVA catalog on those live states,
- expand the socket progressively.

This progressive socket training directly tests whether the all-layer collapse is a training-distribution issue rather than a fundamental SVA limit.
