# Evidence Haystack Snapshot - 2026-05-14

## Purpose

This benchmark measures whether SVA keeps the exact passkey evidence alive as context grows. It separates three questions:

- Does full-attention teacher score the key evidence highly at the final query?
- Does the SVA summoner include the evidence in its candidate set?
- Does the exact verifier keep the evidence after reranking the summoned candidates?

## Run

- Commit: `e65bc5e`
- Modal log: `results/modal_runs/sva-h100-evidence-haystack-20260514-095330.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Layers: `0,15,29`
- Contexts: `4096,8192,16384,32768`
- Placements: `start,middle,end`
- Shortlist and verifier budget: `8192/2048`
- Anchor counts: `1,4,8,16`

The CPU smoke test and adapter unit tests passed before launch.

## Best Evidence Survival

Best row per context and placement, choosing highest verified key survival, then needle survival, then fewer value reads.

| Context | Placement | Policy | Anchors | Verified key | Summoned key | Verified needle | Avg verified | Read reduction | Teacher top-64 key | Teacher best rank |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | start | full | 8 | 0.777778 | 0.925926 | 1.000000 | 2048.0 | 2.0x | 0.001157 | 1502.1 |
| 4096 | middle | full | 1 | 0.962963 | 0.962963 | 1.000000 | 2048.0 | 2.0x | 0.005787 | 601.5 |
| 4096 | end | split | 16 | 1.000000 | 1.000000 | 1.000000 | 431.5 | 9.5x | 0.074653 | 28.6 |
| 8192 | start | full | 8 | 0.740741 | 0.962963 | 0.925926 | 2048.0 | 4.0x | 0.000000 | 1611.5 |
| 8192 | middle | full | 16 | 0.666667 | 0.814815 | 0.888889 | 2048.0 | 4.0x | 0.001157 | 1829.3 |
| 8192 | end | split | 16 | 1.000000 | 1.000000 | 1.000000 | 434.6 | 18.8x | 0.078704 | 19.2 |
| 16384 | start | full | 8 | 0.703704 | 0.814815 | 0.888889 | 2048.0 | 8.0x | 0.009838 | 984.3 |
| 16384 | middle | full | 1 | 0.148148 | 0.148148 | 0.370370 | 2048.0 | 8.0x | 0.000000 | 5387.9 |
| 16384 | end | full | 4 | 0.814815 | 0.888889 | 0.888889 | 2048.0 | 8.0x | 0.017361 | 1172.1 |
| 32768 | start | full | 8 | 0.259259 | 0.333333 | 0.444444 | 2048.0 | 16.0x | 0.000579 | 10683.9 |
| 32768 | middle | full | 1 | 0.333333 | 0.333333 | 0.407407 | 2048.0 | 16.0x | 0.000579 | 8539.1 |
| 32768 | end | full | 16 | 0.444444 | 0.777778 | 0.518519 | 2048.0 | 16.0x | 0.001736 | 5124.1 |

## Interpretation

Multi-anchor summon helps. At `8192` start placement, full-budget anchors lift summoned key survival from `0.592593` with one anchor to `0.962963` with eight anchors. At `32768` end placement, it lifts summoned key survival from `0.222222` with one anchor to `0.777778` with sixteen anchors.

The final verifier is now a visible bottleneck. At `32768` end placement with sixteen full-budget anchors, the key is summoned in `0.777778` of head/layer cases, but survives final verification in `0.444444`. The broad needle span also survives better than the exact key. This points to a reranking objective problem after the summoner has already brought useful evidence into view.

Split-budget multi-anchor is attractive when the evidence is close to the query. At `8192` end placement, sixteen split anchors keep exact key survival at `1.000000` while verifying about `435` tokens, an `18.8x` read reduction. For distant evidence, split budgets become too tight.

The teacher statistics are an important warning signal for this diagnostic: full attention often assigns low final-query top-64 rank to the exact key when the passkey is far from the query. The benchmark is best paired with language-quality runs after each policy change.

## Next Step

The next high-information test is an evidence-aware rerank after multi-anchor summon. Keep the same candidate-union setup, then compare:

- current-query exact verifier,
- best-anchor exact verifier,
- max-over-anchor verifier,
- span/block statement verifier that lets nearby evidence tokens contribute together.

The target is to close the gap between summoned key survival and verified key survival at `32768`, then retest the winning policy in the language passkey benchmark.
