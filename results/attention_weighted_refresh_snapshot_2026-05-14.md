# Attention-Weighted Refresh Snapshot

Date: 2026-05-14

## Question

Plain calibration refresh improved long-context aggregate recall, but the profile-router passkey benchmark showed that balanced codebooks did not improve language-level exact evidence preservation. This run tests whether fitting refreshed codebooks with attention top-k weights helps the summoner preserve the evidence full attention actually uses.

Entropy and max bucket load are recorded as diagnostics only.

## Run

```text
app: ap-uMatZedZL30QSYyv4N9ymq
function: fc-01KRKJT9BC9G2Z53FSY3FCMQB0
log: results/modal_runs/sva-h100-attention-weighted-refresh-20260514-1120.full.log
rows: 540 result rows, 45 summary rows
exit: 0
```

Settings:

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Base artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Contexts: `8192,16384,32768`
- Calibration length: `32768`
- Layers: `0,15,29`
- Budgets: `512,1024,2048`
- Targets: teacher top-16 recall
- Variants: frozen artifact, plain calibration refresh, attention-weighted refresh, strong attention-weighted refresh, eval-key refit ceiling

## Teacher Top-16 Recall

| Context | Budget | Frozen artifact | Plain refresh | Attention weighted | Strong attention weighted | Eval refit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 512 | 0.972367 | 0.952257 | 0.957709 | 0.952176 | 0.985352 |
| 8192 | 1024 | 0.993978 | 0.988643 | 0.989900 | 0.987784 | 0.996573 |
| 8192 | 2048 | 0.999295 | 0.998635 | 0.998834 | 0.998535 | 0.999602 |
| 16384 | 512 | 0.782534 | 0.832746 | 0.853561 | 0.853000 | 0.864538 |
| 16384 | 1024 | 0.870018 | 0.893292 | 0.909008 | 0.910816 | 0.909749 |
| 16384 | 2048 | 0.939435 | 0.934842 | 0.944399 | 0.947980 | 0.944924 |
| 32768 | 512 | 0.563169 | 0.635887 | 0.669515 | 0.677083 | 0.645526 |
| 32768 | 1024 | 0.657407 | 0.726002 | 0.755281 | 0.762297 | 0.734565 |
| 32768 | 2048 | 0.752450 | 0.808793 | 0.829825 | 0.836887 | 0.816081 |

## Catalog Diagnostics

At `32768`, attention weighting improved teacher recall while reducing score-cosine and entropy versus plain refresh:

| Variant | Score cosine | Score MSE | Norm. code entropy | Max code fraction |
| --- | ---: | ---: | ---: | ---: |
| Frozen artifact | 0.809707 | 72.254091 | 0.718895 | 0.230200 |
| Plain refresh | 0.945655 | 19.475662 | 0.978364 | 0.014335 |
| Attention weighted | 0.945231 | 19.624035 | 0.965287 | 0.015558 |
| Strong attention weighted | 0.941867 | 20.860638 | 0.928906 | 0.021504 |
| Eval refit | 0.954126 | 14.401466 | 0.986270 | 0.011481 |

## Interpretation

This is the clearest evidence so far that the next catalog objective should be evidence-aware, not entropy-first.

At `32768`, strong attention-weighted refresh improves over plain refresh by `+0.041196`, `+0.036295`, and `+0.028094` at budgets `512/1024/2048`. It also beats the identity eval-refit ceiling on teacher recall at all three budgets. That means the useful axis is not just better score reconstruction; it is preserving the keys that matter to full attention.

The tradeoff is visible: strong weighting lowers entropy and score cosine, but raises teacher recall. That is exactly why entropy should stay a diagnostic. The pass/fail target is evidence survival and language behavior.

## Next Step

Export an all-layer strong attention-weighted long-context profile and rerun the passkey profile-router benchmark. The recall proxy is now positive; the next question is whether it converts into answer NLL and logit preservation.
