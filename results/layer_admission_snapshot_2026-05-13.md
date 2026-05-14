# Layer Admission Snapshot - 2026-05-13

This run tested a first automatic admission screen for selective SVA socketing.

It starts from the stable 15-layer set:

```text
0,1,3,4,5,6,7,8,9,10,13,15,17,18,21
```

It then screens candidate layers against that base set. A layer is admitted if teacher-state socketing stays below `loss_delta <= 0.050`; borderline layers are retried with progressive artifact training and admitted if they stay below `loss_delta <= 0.060`.

Important caveat: this runner screens each candidate against the base 15-layer set, then validates the final combined admitted set. It is a first-order admission screen, not a fully greedy compiler pass that re-tests each candidate against the current accumulated admitted set.

## Candidate Decisions

| layer | decision | training used | loss_delta | KL | top1 agreement | verified top16 recall |
|---:|---|---|---:|---:|---:|---:|
| 2 | admit | teacher | 0.023438 | 0.024017 | 0.988764 | 0.807500 |
| 11 | reject | progressive | 0.085938 | 0.085733 | 0.975574 | 0.796605 |
| 12 | admit | progressive | 0.058594 | 0.058974 | 0.982413 | 0.799060 |
| 14 | admit | teacher | 0.031250 | 0.030324 | 0.986321 | 0.807448 |
| 16 | reject | progressive | 0.105469 | 0.105799 | 0.971666 | 0.803705 |
| 19 | reject | teacher | 0.562500 | 0.558711 | 0.900831 | 0.797861 |
| 20 | admit | teacher | 0.027344 | 0.027379 | 0.987298 | 0.810411 |
| 22 | admit | teacher | 0.019531 | 0.018823 | 0.990230 | 0.808364 |
| 23 | admit | teacher | 0.019531 | 0.020930 | 0.989741 | 0.813921 |
| 24 | admit | teacher | 0.019531 | 0.021031 | 0.987787 | 0.812084 |
| 25 | admit | teacher | 0.019531 | 0.019507 | 0.990718 | 0.797209 |
| 26 | admit | teacher | 0.023438 | 0.022739 | 0.987298 | 0.803744 |
| 27 | admit | teacher | 0.019531 | 0.019880 | 0.989741 | 0.803119 |
| 28 | admit | teacher | 0.023438 | 0.022031 | 0.990230 | 0.803476 |
| 29 | admit | teacher | 0.023438 | 0.024320 | 0.989741 | 0.803758 |

## Final Combined Set

Admitted socket layers:

```text
0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18,20,21,22,23,24,25,26,27,28,29
```

Full-attention fallback layers:

```text
11,16,19
```

| final condition | socketed count | training | loss_delta | KL | top1 agreement | verified top16 recall |
|---|---:|---|---:|---:|---:|---:|
| admitted set | 27 | teacher | 0.117188 | 0.114144 | 0.971666 | 0.750432 |
| admitted set | 27 | progressive | 0.054688 | 0.054555 | 0.983390 | 0.786364 |

## Readout

The admission screen is directionally useful. It found a 27-layer socket set with only layers `11`, `16`, and `19` kept as full attention. Progressive artifact training brought the combined 27-layer set down to `loss_delta=0.054688`, much better than the earlier 20-layer progressive frontier at `0.179688`.

The result also strengthens the mechanism concern. The rejected layers do not have terrible verified top-16 recall: layer `19` still reached `0.797861`, and layer `16` reached `0.803705`. The model-level failure is therefore not adequately explained by missing teacher top keys.

The probable missing diagnostic is downstream state preservation: whether the layer's attention output and residual update land in the same subspace, with the same scale and downstream effect, as full attention.

## Next Step

Run a mechanism diagnostic on accepted layers versus rejected layers:

1. Compare full-attention output and SVA output at the layer/head level.
2. Measure output cosine, output norm error, residual update norm, softmax KL, attention entropy, value-weighted error, and downstream logit delta.
3. Run patch tests: replace only the suspect layer output with the teacher output and measure how much final loss is repaired.
4. Use those measurements to decide whether the next SVA objective should train for top-key recall, attention distribution KL, value-output reconstruction, or downstream logit preservation.

This is the path from an odd fix to an understood mechanism.
