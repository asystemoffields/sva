# Refreshed Long-Context Profile Snapshot

Date: 2026-05-14

## Question

The held-out refresh benchmark showed that calibration-fit codebooks improve long-context summon quality. This step exports those refreshed identity codebooks as a real production-loadable SVA artifact and verifies that the exported artifact behaves like the benchmark result.

## Artifact

Local artifact:

```text
results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1
```

Remote Modal volume artifact:

```text
sva-artifacts:/sva-smollm2-135m-2x256-longctx-refresh-v1
```

Export run:

```text
app: ap-tiZPdmqCW6zQJSkFsU4J2u
function: fc-01KRKFTH6KM43SRPRKQFQN4YDH
log: results/modal_runs/sva-h100-export-refreshed-artifact-20260514-1056.full.log
exit: 0
```

The exported artifact reloads through the production loader:

```text
loaded_layers: 30
profile: sva-smollm2-135m-2x256-longctx-refresh-v1
context: 32768
refresh: calibration_identity_kmeans
```

During export, all 30 layer codebooks were fitted on a `32768` token calibration stream. Fit-time normalized code entropy ranged from about `0.978347` to `0.990571`; max code fraction ranged from about `0.008984` to `0.018940`.

## Recall Sanity Run

Recall run:

```text
app: ap-mCfxDsh3KEfv66l0aMIEjv
function: fc-01KRKG3DFHNE1DQAASZNDHY00X
log: results/modal_runs/sva-h100-refreshed-profile-recall-20260514-1100.full.log
rows: 216 result rows, 18 summary rows
exit: 0
```

The sanity run evaluates the exported artifact directly as `artifact_identity`, with `eval_refit_identity` as the ceiling.

| Context | Budget | Refreshed artifact | Eval refit upper bound |
| ---: | ---: | ---: | ---: |
| 8192 | 512 | 0.952637 | 0.985352 |
| 8192 | 1024 | 0.988589 | 0.996573 |
| 8192 | 2048 | 0.998788 | 0.999602 |
| 16384 | 512 | 0.832501 | 0.864538 |
| 16384 | 1024 | 0.893925 | 0.909776 |
| 16384 | 2048 | 0.935258 | 0.944942 |
| 32768 | 512 | 0.630588 | 0.645535 |
| 32768 | 1024 | 0.725071 | 0.734556 |
| 32768 | 2048 | 0.809860 | 0.816090 |

## Catalog Quality

| Context | Variant | Score cosine | Score MSE | Norm. code entropy | Max code fraction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8192 | refreshed artifact | 0.907399 | 28.729431 | 0.756394 | 0.032471 |
| 8192 | eval refit | 0.952261 | 11.823350 | 0.984260 | 0.009880 |
| 16384 | refreshed artifact | 0.930350 | 24.147602 | 0.872928 | 0.017891 |
| 16384 | eval refit | 0.953336 | 13.071882 | 0.987055 | 0.009347 |
| 32768 | refreshed artifact | 0.945643 | 19.481562 | 0.978694 | 0.014597 |
| 32768 | eval refit | 0.954126 | 14.401495 | 0.986271 | 0.011481 |

## Interpretation

The exported profile reproduces the useful part of the calibration-refresh diagnostic. Compared with the original frozen artifact, it improves the `32768` rows from `0.563169/0.657407/0.752450` to `0.630588/0.725071/0.809860` at budgets `512/1024/2048`.

At `8192`, the refreshed profile gives back the original artifact's advantage. That is now the concrete reason to implement a profile router: use the original 8k profile for ordinary context, then switch to the refreshed long-context profile when the catalog entropy of the original profile starts collapsing.

## Next Step

Socket the refreshed profile into the language-facing long-context tests:

- original profile at 8k,
- refreshed profile at 16k/32k,
- simple context-length routing first,
- later routing by measured entropy/max-code-load per layer.

The first target is passkey language recall and answer NLL at `16384` and `32768`, where the refreshed profile should preserve evidence better without changing the exact verifier.
