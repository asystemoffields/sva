# Rotation Diagnostic Snapshot - 2026-05-14

## Purpose

This run tests whether SVA's low-rank product-code catalog is limited by axis geometry or codebook fit.

Exact low-rank Q/K scores are invariant under an orthogonal rotation. Product quantization is not invariant because it splits the low-rank vector into fixed subspaces. The diagnostic compares:

- `artifact_identity`: the frozen exported artifact codebooks.
- `refit_identity`: identity low-rank space with codebooks refit to the evaluated key bank.
- `hadamard`: Hadamard-rotated low-rank space with codebooks refit to the evaluated key bank.
- `signed_hadamard`: signed and permuted Hadamard rotation with codebooks refit to the evaluated key bank.

The refit variants are diagnostic upper bounds for catalog/codebook quality. They fit codebooks to the evaluated key bank.

## Run

- Commit: pending at run launch
- Modal log: `results/modal_runs/sva-h100-rotation-diagnostic-20260514-103258.full.log`
- Modal function: `fc-01KRKEFED8WZVR5SBQRK80VBA8`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Contexts: `8192,16384,32768`
- Placements: `middle,end`
- Layers: `0,15,29`
- Budgets: `512,1024,2048`
- Top-k target: `16`
- Query samples per case: `64`
- K-means iterations: `8`
- Result volume: `216` result rows and `12` summary rows, exit `0`

## Aggregate Summary

| Variant | Budget | Teacher top-16 recall | Low-rank top-16 recall | Score cosine | Score MSE | Code entropy | Max code fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| artifact_identity | 512 | 0.771888 | 0.805465 | 0.870095 | 49.155 | 0.812974 | 0.138368 |
| refit_identity | 512 | 0.837637 | 0.932973 | 0.957629 | 13.163 | 0.986205 | 0.009986 |
| hadamard | 512 | 0.836197 | 0.931966 | 0.957368 | 13.230 | 0.986252 | 0.009961 |
| signed_hadamard | 512 | 0.837963 | 0.931225 | 0.957594 | 13.160 | 0.986145 | 0.009956 |
| artifact_identity | 1024 | 0.837511 | 0.866048 | 0.870095 | 49.155 | 0.812974 | 0.138368 |
| refit_identity | 1024 | 0.884145 | 0.968491 | 0.957629 | 13.163 | 0.986205 | 0.009986 |
| hadamard | 1024 | 0.883632 | 0.967978 | 0.957368 | 13.230 | 0.986252 | 0.009961 |
| signed_hadamard | 1024 | 0.885031 | 0.968129 | 0.957594 | 13.160 | 0.986145 | 0.009956 |
| artifact_identity | 2048 | 0.893808 | 0.917263 | 0.870095 | 49.155 | 0.812974 | 0.138368 |
| refit_identity | 2048 | 0.923472 | 0.987727 | 0.957629 | 13.163 | 0.986205 | 0.009986 |
| hadamard | 2048 | 0.922948 | 0.987829 | 0.957368 | 13.230 | 0.986252 | 0.009961 |
| signed_hadamard | 2048 | 0.923792 | 0.988221 | 0.957594 | 13.160 | 0.986145 | 0.009956 |

## Context Breakdown

Budget `1024`, averaged over placements `middle,end` and layers `0,15,29`.

| Context | Variant | Teacher top-16 recall | Low-rank top-16 recall | Score cosine | Code entropy | Max code fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | artifact_identity | 0.994575 | 0.996003 | 0.924515 | 0.927667 | 0.018047 |
| 8192 | refit_identity | 0.998517 | 0.999276 | 0.958647 | 0.984796 | 0.009708 |
| 8192 | hadamard | 0.998535 | 0.999439 | 0.958099 | 0.984729 | 0.009602 |
| 8192 | signed_hadamard | 0.998264 | 0.998897 | 0.958431 | 0.984676 | 0.009496 |
| 16384 | artifact_identity | 0.868182 | 0.875018 | 0.876423 | 0.804293 | 0.151961 |
| 16384 | refit_identity | 0.912778 | 0.990976 | 0.956531 | 0.987653 | 0.009257 |
| 16384 | hadamard | 0.910464 | 0.990759 | 0.956410 | 0.987595 | 0.009378 |
| 16384 | signed_hadamard | 0.909903 | 0.989348 | 0.956643 | 0.987430 | 0.009511 |
| 32768 | artifact_identity | 0.649776 | 0.727123 | 0.809347 | 0.706961 | 0.245096 |
| 32768 | refit_identity | 0.741138 | 0.915220 | 0.957707 | 0.986168 | 0.010994 |
| 32768 | hadamard | 0.741898 | 0.913737 | 0.957596 | 0.986434 | 0.010904 |
| 32768 | signed_hadamard | 0.746926 | 0.916142 | 0.957708 | 0.986330 | 0.010859 |

## Interpretation

The large gain comes from codebook fit and code balance. Refit codebooks lift score cosine from `0.870095` to about `0.9576`, reduce score MSE by about `3.7x`, and make code usage much more uniform: normalized entropy rises from `0.812974` to about `0.986`, while the average largest-code fraction drops from `0.138368` to about `0.010`.

Hadamard rotation is viable but not the main win in this run. Plain refit, Hadamard refit, and signed-Hadamard refit are tightly clustered. The signed-Hadamard variant is slightly best on teacher recall at budgets `512`, `1024`, and `2048`, but the differences are small.

The context breakdown explains the long-context decay seen in the span benchmark. At `32768` with budget `1024`, frozen artifact codebooks average only `0.649776` teacher top-16 recall and `0.809347` score cosine. Refit codebooks raise those to `0.741138` and `0.957707`.

## Next Step

Turn this diagnostic into a deployable catalog test:

- train identity and signed-Hadamard codebooks on a separate calibration stream,
- evaluate on held-out long-context streams,
- compare against the current frozen artifact at the same `512/1024/2048` budgets,
- if the held-out result keeps most of the refit gain, export a refreshed artifact profile and rerun evidence rerank plus span statements.
