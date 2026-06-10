# Layer Fallback Snapshot - 2026-05-13

This run tested whether the 20-layer selective socket frontier can be rescued by leaving fragile layers as full attention.

The stable baseline is the 15-layer socket:

```text
0,1,3,4,5,6,7,8,9,10,13,15,17,18,21
```

The full 20-layer frontier adds `14`, `16`, `19`, `20`, and `23`. The cliff map identified layer `19` as the main fault line and layer `16` as a secondary pressure point.

All runs used `HuggingFaceTB/SmolLM2-135M-Instruct`, `seq_len=2048`, QK routing, `4x64` coarse PQ, `coarse_shortlist=1024`, `budget=512`, `ranker_train_steps=160`, `coarse_hard_steps=80`, hard pool `512`, and attention-weighted codebooks with boost `4`.

## Results

| condition | training | socketed count | socketed addition beyond base 15 | full-attention fallback layers | loss_delta | KL | top1 agreement | logit cosine | candidate top16 recall | verified top16 recall |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| base_15 | teacher | 15 | none | 14,16,19,20,23 | 0.019531 | 0.018087 | 0.991207 | 0.796962 | 0.813215 | 0.810951 |
| add_19 | progressive | 16 | 19 | 14,16,20,23 | 0.156250 | 0.154671 | 0.977040 | 0.594932 | 0.806288 | 0.804819 |
| no_19 | teacher | 19 | 14,16,20,23 | 19 | 0.097656 | 0.097367 | 0.967758 | 0.733643 | 0.800643 | 0.798275 |
| no_19 | progressive | 19 | 14,16,20,23 | 19 | 0.074219 | 0.071809 | 0.977528 | 0.752908 | 0.797847 | 0.796193 |
| no_16_19 | teacher | 18 | 14,20,23 | 16,19 | 0.042969 | 0.042852 | 0.984367 | 0.787485 | 0.812182 | 0.809804 |
| no_16_19 | progressive | 18 | 14,20,23 | 16,19 | 0.039062 | 0.040822 | 0.985344 | 0.810712 | 0.812036 | 0.810279 |

## Readout

Per-layer fallback works as a practical control knob. The failed 20-layer frontier had `loss_delta=0.757812` under teacher training and `0.179688` under progressive training. Keeping layer `19` as full attention improved the 19-layer progressive socket to `loss_delta=0.074219`. Keeping both layers `16` and `19` as full attention improved the 18-layer progressive socket to `loss_delta=0.039062`.

This is also a scaling warning. The right future-facing fix is not a hand-maintained list of fragile layers. A large model would need an automatic socketing protocol:

1. train or fit the SVA path for a candidate layer,
2. evaluate downstream distribution preservation under a fixed quality budget,
3. route the layer to one of three outcomes: socket now, progressive/state-aware training, or full-attention fallback.

The local recall metrics again understate the model-level issue. The 18-layer fallback and 19-layer fallback have similar verified top-16 recall, but the 18-layer version is much better at preserving logits. The gate should measure downstream state or logit preservation, not just attention-target recall.

## Next Step

Turn the manual fallback result into an automatic layer admission test:

- build a runner that scores every candidate layer by incremental loss/KL when added to a stable socket set,
- admit layers whose incremental damage stays below a threshold,
- send borderline layers to progressive training,
- keep failed layers as full attention until the router objective improves.

For a frontier-scale model, this becomes a model-surgery compiler pass rather than a hand-designed architecture.
