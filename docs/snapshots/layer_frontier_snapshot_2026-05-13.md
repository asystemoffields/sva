# Layer Frontier Snapshot - 2026-05-13

This run expanded the strongest selective-hybrid socket from the three-condition test into larger layer sets.

All runs used `HuggingFaceTB/SmolLM2-135M-Instruct`, `seq_len=2048`, QK routing, teacher-state artifact training, `4x64` coarse PQ, `coarse_shortlist=1024`, `budget=512`, `ranker_train_steps=160`, `coarse_hard_steps=80`, hard pool `512`, and attention-weighted codebooks with boost `4`.

## Results

| frontier | socketed layers | layer count | loss_delta | KL | top1 agreement | logit cosine | candidate top16 recall | verified top16 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| frontier_6 | 0,1,3,4,7,10 | 6 | 0.007812 | 0.008817 | 0.994138 | 0.835482 | 0.875299 | 0.874180 |
| frontier_9 | 0,1,3,4,5,6,7,8,10 | 9 | 0.011719 | 0.010025 | 0.992672 | 0.801478 | 0.853670 | 0.850978 |
| frontier_12 | 0,1,3,4,5,6,7,8,9,10,15,18 | 12 | 0.015625 | 0.014993 | 0.992672 | 0.770225 | 0.831344 | 0.828815 |
| frontier_15 | 0,1,3,4,5,6,7,8,9,10,13,15,17,18,21 | 15 | 0.019531 | 0.018851 | 0.991695 | 0.794052 | 0.812975 | 0.810719 |
| frontier_20 | 0,1,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,23 | 20 | 0.757812 | 0.757605 | 0.860772 | 0.542104 | 0.782229 | 0.779792 |

## Readout

The selective socket frontier is real through 15 layers. Replacing half of SmolLM2's attention layers kept the next-token distribution close to full attention: `loss_delta=0.019531`, `KL=0.018851`, and `top1_agreement=0.991695`.

The first large break is between 15 and 20 socketed layers. The 20-layer set still has only moderately lower local recall (`0.779792` verified top-16 recall), but the model-level loss jumps to `0.757812`. That points to compounding representation drift and fragile mid/late layers rather than a simple local-recall threshold.

The worst-head diagnostics in the failed 20-layer run put pressure on layers `19`, `20`, `21`, `14`, and `16`. Those layers should be treated as suspects before blaming the whole frontier.

## Next Step

Map the cliff directly:

1. Run ablations around the 15-to-20 boundary by adding one suspect layer at a time to the 15-layer set.
2. Try the same suspect additions with progressive artifact training.
3. If one or two layers account for most of the cliff, test per-layer fallback: keep those layers full-attention while socketing the rest of the frontier.

This keeps the theory subordinate to the facts: the current object is selective SVA socketing plus explicit fragile-layer handling.
