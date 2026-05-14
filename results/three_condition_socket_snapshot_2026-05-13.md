# Three-Condition Socket Snapshot - 2026-05-13

This run compared three answers to the all-layer three-stage socket collapse:

1. hidden-state hierarchical routing,
2. progressive socket training,
3. selective hybrid socketing.

All conditions used `HuggingFaceTB/SmolLM2-135M-Instruct`, `seq_len=2048`, `4x64` coarse PQ, `coarse_shortlist=1024`, `budget=512`, `ranker_train_steps=160`, and `coarse_hard_steps=80`.

## Results

| condition | socketed layers | artifacts | route source | artifact training | loss_delta | KL | top1 agreement | logit cosine | candidate top16 recall | verified top16 recall |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| hidden-state hierarchy | all | 30 | hidden | teacher | 1.609375 | 1.611989 | 0.701026 | 0.405951 | 0.700970 | 0.657011 |
| progressive QK | all | 30 | qk | progressive | 1.257812 | 1.254928 | 0.778701 | 0.548340 | 0.740070 | 0.738783 |
| selective hybrid QK | 0,1,3,4,5,6,7,8,10 | 9 | qk | teacher | 0.011719 | 0.010180 | 0.992672 | 0.801694 | 0.853443 | 0.850742 |

## Readout

The hidden-state routing prototype was a kill in this form. It used layer input hidden states as the route source while keeping the coarse-to-rank-to-verify hierarchy. It underperformed QK routing and lost many targets during the exact-rank stage as well as the coarse stage.

Progressive training helped but did not solve all-layer socketing. It improved the previous all-layer full-step result from `loss_delta=1.562500` and candidate recall `0.720604` to `loss_delta=1.257812` and candidate recall `0.740070`. That supports the distribution-shift diagnosis, but the effect is too small by itself.

The selective hybrid is the first strong socketing shape after the all-layer collapse. Replacing 9 of 30 layers kept loss and KL near the full-attention baseline while preserving the same candidate budget. This gives a practical path for surgical socketing: expand only through layers that tolerate the replacement, and leave fragile layers as full attention until their routers improve.

## Next Step

The next sharp test is a layer-selection frontier:

- sweep increasingly large socket layer sets,
- rank layers by observed tolerance and cost impact,
- measure quality per replaced-layer count,
- use that frontier to decide whether the next model-surgery target is 30%, 50%, or full-layer replacement.

The all-layer branch should continue, but the immediate reliable path is selective expansion rather than forcing every layer through the same router.
