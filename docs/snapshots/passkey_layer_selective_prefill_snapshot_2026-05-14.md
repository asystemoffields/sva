# Passkey Layer-Selective Prefill Snapshot - 2026-05-14

## Question

The profile prefill benchmark showed that all-layer SVA accumulates long-context drift before answer decoding. This sweep asks whether keeping some layers on full attention while socketing SVA into others reduces 32K final-prompt drift.

## Run

- Modal app: `ap-BWAy8Rh5aNuSI1W6dXB7Xj`
- Log: `results/modal_runs/sva-h100-passkey-layer-selective-prefill-20260514-1245.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `32768`
- Placement: passkey at start, query at end
- Profile: strong attention-weighted long-context artifact
- Policy: scan summon, shortlist `8192`, verifier budget `2048`

## Result

| Layer group | Socket layers | First-token NLL delta | Prefill KL | Logit cosine | Top-1 agreement | SVA prefill ms | Slowdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | all | 0.992831 | 2.148224 | 0.970458 | 0.000000 | 104030.648 | 52.66x |
| selective9 | `0,1,3,4,5,6,7,8,10` | 1.674210 | 3.836644 | 0.973374 | 0.000000 | 31047.524 | 45.62x |
| sparse6 | `0,5,10,18,24,29` | 0.944199 | 0.234555 | 0.989136 | 1.000000 | 20902.272 | 28.53x |
| early4 | `0,1,2,3` | 1.540283 | 1.231577 | 0.982503 | 0.000000 | 14104.789 | 17.68x |
| early10 | `0,1,2,3,4,5,6,7,8,9` | 0.482875 | 0.227857 | 0.997476 | 0.000000 | 34385.166 | 47.93x |
| late10 | `20,21,22,23,24,25,26,27,28,29` | 0.035455 | 0.019850 | 0.999362 | 1.000000 | 34420.971 | 50.58x |

Full attention first-token NLL was `3.372235`. The `late10` row reached `3.407691`, kept the same top token as full attention, and reduced KL by about `108x` versus all-layer SVA.

## Interpretation

Layer placement matters more than the catalog profile in this test. At 32K, early and mixed replacement still corrupts the hidden-state stream enough to hurt the final prompt logits. Replacing only the last ten layers is much closer to full attention, which suggests a late-layer SVA socket can use a mostly full-attention-formed representation while reducing exact reads in the final stack.

The wall-clock result remains dominated by the scan implementation. `late10` improved quality sharply, but the current PyTorch scan still takes about `34.4s` for 32K prefill. The result is therefore a quality/socketing signal, with speed still assigned to the indexed summon path.

## Next Step

Run the full passkey answer benchmark with the strong attention-weighted profile and `late10` layer socket. Compare against all-layer and the earlier `sparse6` set at `16384` and `32768`. The pass condition is answer-level NLL/KL/top-1 agreement improving in the same direction as the prefill probe.
