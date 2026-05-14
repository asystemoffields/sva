# Layer Cliff Snapshot - 2026-05-13

This run mapped the 15-to-20 selective socket cliff from `results/layer_frontier_snapshot_2026-05-13.md`.

The baseline was the stable 15-layer set:

```text
0,1,3,4,5,6,7,8,9,10,13,15,17,18,21
```

The five layers added in the failed 20-layer frontier were `14`, `16`, `19`, `20`, and `23`. All runs used `HuggingFaceTB/SmolLM2-135M-Instruct`, `seq_len=2048`, QK routing, `4x64` coarse PQ, `coarse_shortlist=1024`, `budget=512`, `ranker_train_steps=160`, `coarse_hard_steps=80`, hard pool `512`, and attention-weighted codebooks with boost `4`.

## Results

| condition | training | socketed count | added layers | loss_delta | KL | top1 agreement | logit cosine | candidate top16 recall | verified top16 recall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| base_15 | teacher | 15 | none | 0.023438 | 0.024391 | 0.988764 | 0.790866 | 0.813287 | 0.811023 |
| add_14 | teacher | 16 | 14 | 0.027344 | 0.028440 | 0.986810 | 0.769730 | 0.809935 | 0.807540 |
| add_16 | teacher | 16 | 16 | 0.078125 | 0.078464 | 0.974597 | 0.715870 | 0.801950 | 0.799713 |
| add_19 | teacher | 16 | 19 | 0.562500 | 0.562174 | 0.901808 | 0.544017 | 0.800148 | 0.797899 |
| add_20 | teacher | 16 | 20 | 0.023438 | 0.024428 | 0.989253 | 0.798291 | 0.813097 | 0.810775 |
| add_23 | teacher | 16 | 23 | 0.019531 | 0.018521 | 0.990718 | 0.789244 | 0.816197 | 0.813999 |
| add_14_16 | teacher | 17 | 14,16 | 0.082031 | 0.081203 | 0.973131 | 0.702270 | 0.798805 | 0.796423 |
| add_19_20 | teacher | 17 | 19,20 | 0.468750 | 0.463211 | 0.917929 | 0.555637 | 0.795645 | 0.793249 |
| frontier_20 | teacher | 20 | 14,16,19,20,23 | 0.757812 | 0.751164 | 0.862726 | 0.542447 | 0.782007 | 0.779580 |
| add_14_16 | progressive | 17 | 14,16 | 0.066406 | 0.065821 | 0.982413 | 0.739046 | 0.795191 | 0.793548 |
| add_19_20 | progressive | 17 | 19,20 | 0.128906 | 0.127494 | 0.977528 | 0.679911 | 0.791349 | 0.789265 |
| frontier_20 | progressive | 20 | 14,16,19,20,23 | 0.179688 | 0.179911 | 0.963850 | 0.662632 | 0.783030 | 0.781464 |

## Readout

Layer `19` is the main cliff trigger in this run. Adding layer `19` alone to the stable 15-layer set raised `loss_delta` from `0.023438` to `0.562500`, while adding layer `20` or `23` alone stayed near baseline and layer `14` was mild.

Layer `16` is a secondary pressure point. It raised `loss_delta` to `0.078125` alone and `0.082031` with layer `14`.

Progressive artifact training helped substantially where the distribution shift was real. It improved the 20-layer frontier from `loss_delta=0.757812` to `0.179688`, and improved the `19,20` addition from `0.468750` to `0.128906`. That says the cliff is partly trainable state drift, with layer `19` as the most sensitive layer.

The local recall metrics did not predict the model-level cliff. `add_19` still had `0.797899` verified top-16 recall, close to `add_16` at `0.799713`, but its model loss was much worse. The issue is therefore not just whether the verifier finds enough teacher top keys. It is whether the socketed layer preserves the downstream hidden-state distribution.

## Next Step

Run the direct per-layer fallback test:

1. socket the 20-layer frontier minus layer `19`,
2. repeat with progressive artifact training,
3. if that is close to the 15-layer frontier, treat layer `19` as a full-attention fallback while continuing to expand other layers.

The practical hypothesis is now:

```text
SVA can replace a large selective subset of attention layers;
some layers need either progressive state training, a better router objective, or a full-attention fallback.
```
