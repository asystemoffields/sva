# Output Distillation Socket Snapshot - 2026-05-14

This snapshot records the first output/logit-preserving probe for early and all-layer SVA sockets.

## Run

- Runner: `modal_h100_output_distill_socket.py`
- Harness: `experiments/sva_output_distill_socket_test.py`
- Modal call: `fc-01KRKRERSS1BGNKY52Y564M9RD`
- Local log: `results/modal_runs/sva-h100-output-distill-socket-fixed-20260514-132721.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `1024`
- SVA profile: rank `64`, `4x64` coarse PQ, shortlist `1024`, verifier budget `512`
- Adapter: residual output adapter after SVA attention, rank `16`, source `both`
- Training: 60 distillation steps on `base,reverse`; held-out eval on `rotate,odds_evens`

## Mean Results

```text
case                  phase            loss_delta  KL_to_full  top1_agree  logit_cos
early26_teacher       baseline_eval     0.000000   0.000552    0.992180    0.999450
early26_teacher       distilled_eval    0.000000   0.000624    0.990225    0.999407
early26_progressive   baseline_eval     0.000000   0.000542    0.992669    0.999420
early26_progressive   distilled_eval   -0.003906   0.000718    0.995601    0.999434
all30_teacher         baseline_eval     0.000000   0.000562    0.993646    0.999424
all30_teacher         distilled_eval   -0.003906   0.000686    0.991202    0.999421
```

## Interpretation

The harness works: SVA was socketed, the adapters trained on H100, and metrics were emitted for baseline and post-distillation splits.

The test is too easy to answer the early-layer failure question. At `1024`, frozen SVA already preserves the full-attention distribution closely for both `0-25` and all 30 layers. Adapter training reduced train KL, but it did not produce a clean held-out KL improvement.

The next sharp version should target the actual failure regime: long-context/passkey prefill or answer-logit preservation, where early-layer SVA previously drifted despite good local recall.
