# Passkey Attention-Weighted Profile Router Snapshot - 2026-05-14

## Question

The attention-weighted codebook refresh improved held-out teacher top-key recall at `32768`. This run tests whether that proxy gain converts into passkey language behavior when the exported profile is used in the production-style SVA adapter.

## Run

- Modal app: `ap-u6lXLo9fiFIKAnxOxkop7v`
- Function log: `results/modal_runs/sva-h100-passkey-attnweighted-router-20260514-1158.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Base artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Long artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Router: base artifact below `16384`, attention-weighted long artifact at `16384+`
- Policy: scan summon, shortlist `8192`, verifier budget `2048`
- Placement: passkey at start, query at end
- Dtype/device: `bfloat16` on H100

## Result

| Context | SVA profile | Full NLL | SVA NLL | NLL delta | Answer KL | Logit cosine | Decode read reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | original | 2.993736 | 3.064683 | 0.070947 | 0.011096 | 0.979466 | 4.00x |
| 16384 | attention-weighted long | 3.843979 | 3.868731 | 0.024752 | 0.118649 | 0.997162 | 8.00x |
| 32768 | attention-weighted long | 6.378009 | 6.530252 | 0.152243 | 1.560773 | 0.744593 | 16.00x |

For comparison, the prior plain refreshed profile router reached:

| Context | Plain refreshed NLL delta | Attention-weighted NLL delta |
| --- | ---: | ---: |
| 16384 | 0.042013 | 0.024752 |
| 32768 | 0.138533 | 0.152243 |

The earlier original-profile scale-out row reached:

| Context | Original-profile NLL delta | Attention-weighted NLL delta |
| --- | ---: | ---: |
| 16384 | -0.016004 | 0.024752 |
| 32768 | 0.116894 | 0.152243 |

## Interpretation

Attention-weighted refresh partially transfers from proxy recall to language behavior. At `16384`, it improves over the plain refreshed profile and keeps very high answer logit cosine. At `32768`, the stronger evidence weighting regresses versus both the plain refreshed profile and the original-profile scale-out row.

The method lesson is narrow and useful: evidence survival is a better catalog objective than entropy, but the weighting policy is now too blunt. A single all-layer strong profile can overfit the calibration attention pattern at the longest tested context.

The systems lesson is unchanged. The current scan prefill is far from deployable: `32768` prefill took `94.7s` for SVA versus `0.189s` for full attention in this PyTorch harness. Decode is closer, but still `3.16x` slower at `32768` while reading `16x` fewer values.

## Next Step

Run a mixed-strength profile sweep that keeps the evidence-aware objective but reduces brittleness:

- compare base attention-weighted boost `1/2/4` instead of only strong effective `16`
- route by context length and possibly layer group
- track passkey key survival alongside answer NLL so we can see whether the `32768` failure is summon miss, verifier miss, or wrong evidence emphasis
- keep speed work focused on replacing scan prefill with an indexed/elevator summon path
