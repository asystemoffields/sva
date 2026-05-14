# Held-Out Codebook Refresh Snapshot

Date: 2026-05-14

## Question

The rotation diagnostic showed that refitting product codebooks on the eval keys sharply improved long-context recall, score cosine, and code entropy. This benchmark removes that transductive advantage: codebooks are fitted on a separate calibration stream, then evaluated on held-out documents at `8192`, `16384`, and `32768` tokens.

The run uses the local artifact:

```text
results/hf_artifacts/sva-smollm2-135m-2x256-v1
```

The corrected H100 run is:

```text
name: sva-h100-codebook-refresh-fixed-20260514-104644
app: ap-EfpLdVhvREEQoz8CyJlcmW
function: fc-01KRKF8NBQ9TAT78NTHASFY4QZ
log: results/modal_runs/sva-h100-codebook-refresh-fixed-20260514-104644.full.log
rows: 540 result rows, 45 summary rows
exit: 0
```

An earlier run from `sva-h100-codebook-refresh-20260514-104407` was superseded because some intended `32768` token eval documents tokenized to about `25k` tokens. The harness now repeats calibration and eval streams until they fill the requested token length.

## Variants

- `artifact_identity`: frozen artifact codebooks.
- `calib_identity`: identity low-rank space, codebooks fitted on separate calibration text.
- `calib_hadamard`: Hadamard-rotated low-rank space, codebooks fitted on separate calibration text.
- `calib_signed_hadamard`: signed/permuted Hadamard low-rank space, codebooks fitted on separate calibration text.
- `eval_refit_identity`: eval-key refit upper bound.

All rows use layers `0,15,29`, budgets `512,1024,2048`, top-k target `16`, `64` query samples per document, and held-out eval documents.

## Teacher Top-16 Recall

| Context | Budget | Frozen artifact | Best calibration refresh | Eval refit upper bound |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 512 | 0.972367 | 0.953423 (`calib_hadamard`) | 0.985352 |
| 8192 | 1024 | 0.993978 | 0.988923 (`calib_signed_hadamard`) | 0.996573 |
| 8192 | 2048 | 0.999295 | 0.998987 (`calib_hadamard`) | 0.999602 |
| 16384 | 512 | 0.782534 | 0.833867 (`calib_hadamard`) | 0.864538 |
| 16384 | 1024 | 0.870018 | 0.893537 (`calib_signed_hadamard`) | 0.909749 |
| 16384 | 2048 | 0.939435 | 0.934842 (`calib_identity`) | 0.944924 |
| 32768 | 512 | 0.563169 | 0.635887 (`calib_identity`) | 0.645553 |
| 32768 | 1024 | 0.657407 | 0.726002 (`calib_identity`) | 0.734592 |
| 32768 | 2048 | 0.752450 | 0.809995 (`calib_hadamard`) | 0.816090 |

## Catalog Quality

At `32768`, the refreshed calibration codebooks almost close the gap to eval-key refit:

| Variant | Score cosine | Score MSE | Norm. code entropy | Max code fraction |
| --- | ---: | ---: | ---: | ---: |
| `artifact_identity` | 0.809707 | 72.254091 | 0.718895 | 0.230200 |
| `calib_identity` | 0.945655 | 19.475674 | 0.978364 | 0.014335 |
| `calib_hadamard` | 0.945707 | 19.408481 | 0.978560 | 0.014418 |
| `calib_signed_hadamard` | 0.945974 | 19.379005 | 0.978849 | 0.014356 |
| `eval_refit_identity` | 0.954126 | 14.401471 | 0.986271 | 0.011481 |

At `16384`, calibration refresh also improves the tight-budget rows. At `8192`, the frozen artifact remains best, especially at budget `512`, so the deployable shape should likely keep context-matched profiles rather than replacing the 8k artifact globally.

## Interpretation

This is a real held-out improvement for long-context SVA. The strongest diagnostic in this run is Shannon-style: when the product-code catalog loses entropy, many keys collide into a few overloaded codes and the summoner stops preserving enough evidence. At `32768`, the frozen artifact's largest average code bucket carries about `23%` of traffic in a subspace. Calibration refresh reduces that to about `1.4%` and raises teacher recall by `0.057545-0.072718` across budgets.

Entropy alone is a state variable, not the full objective. The useful formula seems to need at least:

```text
recall ~= f(context, verifier_budget, score_distortion, normalized_code_entropy, max_code_load)
```

The 8k rows show that entropy must be measured against the served distribution. The 32k calibration stream produces balanced codebooks for long contexts, but those codebooks are less matched to the 8k distribution than the frozen artifact.

## Next Step

Export a refreshed long-context profile instead of treating codebook refresh as only a diagnostic. The next benchmark should compare:

- current `2x256` artifact profile for 8k,
- calibration-refreshed `2x256` long-context profile for 16k and 32k,
- a simple context-length router that selects the profile before lookup.

The pass condition is improved 16k/32k held-out recall without regressing the 8k production sanity check.
