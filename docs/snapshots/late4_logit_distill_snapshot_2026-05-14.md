# Late4 Logit Distillation Snapshot - 2026-05-14

This snapshot records the first SVA-active fine-tuning probe for the tight-budget late4 socket.

## Run

- Runner: `modal_h100_late4_logit_distill.py`
- Harness: `experiments/sva_late4_logit_distill.py`
- Modal call: `fc-01KRKY3NQ281AN66WXSQBX4XJS`
- Local log: `results/modal_runs/sva-h100-late4-logit-distill-20260514-1506.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `32768`
- Socket: layers `26-29`
- Artifact: strong attention-weighted long-context profile
- Budget: shortlist `512`, verifier budget `128`
- Trainable parameters: `110592`
- Training target: full-attention final-prompt logits
- Train set: keys `731942,184029`, placements `start,middle`
- Held-out eval: key `905317`, placements `start,middle,end`
- Steps: `24`

## Mean Results

| split | baseline KL | distilled KL | baseline top-1 | distilled top-1 | baseline cosine | distilled cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.060001 | 0.012359 | 1.000000 | 0.750000 | 0.998997 | 0.999677 |
| held-out | 0.045040 | 0.009811 | 0.666667 | 1.000000 | 0.998986 | 0.999657 |

Held-out per-case KL:

```text
placement  baseline_kl  distilled_kl
start      0.078199     0.016622
middle     0.052806     0.008562
end        0.004115     0.004250
```

## Interpretation

This is the first positive fine-tuning signal for making SVA cheaper. With only tiny residual adapters on top of the late4 SVA outputs, the tight `512/128` budget moved much closer to the full-attention final-prompt distribution on a held-out passkey. The win transferred across key value and placement, with the largest gains on the long-range start and middle placements.

This does not yet prove full answer-decode preservation, because the objective and evaluation here are final-prompt logits. It does show that SVA-active training can recover distribution closeness at a much smaller verifier budget than the zero-shot late4 setting.

## Next Step

Save the trained adapters as an artifact and run the full passkey answer benchmark with the adapted `512/128` late4 socket. If that transfers, repeat across the 9-case late4 robustness panel.
