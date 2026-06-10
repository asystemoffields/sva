# Pretrained Socket Prefilter Snapshot

Date: 2026-05-13

## Question

Can a cheap low-dimensional prefilter shrink the exact-scored candidate set after the successful `64 tables / 10 bits / probe_radius=2` SVA lookup?

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Harness: `experiments/sva_pretrained_socket_test.py`
- Runner: `modal_h100_socket_prefilter.py`
- Hardware: Modal H100
- Contexts: 256 and 512 tokens
- Fixed SVA lookup: `64 tables`, `10 bits`, `probe_radius=2`
- Post-score value budget: 32
- Prefilter: random low-dimensional projected QK score, applied only inside the SVA-summoned candidate set

`avg_summoned` is the broad lookup set. `avg_exact_scored` is the set that receives full-dimensional exact QK scoring after the prefilter. `avg_postscore_attended` is the final top-k value aggregation set.

## Result

| seq | prefilter_dim | prefilter_budget | loss_delta | KL to full | top1 agreement | logit cosine | candidate top16 recall | exact top16 recall | avg summoned | avg exact scored | avg attended |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0 | 0 | 0.031250 | 0.036275 | 0.896732 | 0.992801 | 0.973725 | 0.973725 | 113.128 | 113.128 | 29.895 |
| 256 | 8 | 64 | 0.562500 | 0.579936 | 0.661438 | 0.964519 | 0.972904 | 0.781297 | 112.093 | 55.209 | 29.895 |
| 256 | 16 | 32 | 0.875000 | 0.951934 | 0.501961 | 0.937864 | 0.972438 | 0.634531 | 111.519 | 29.895 | 29.895 |
| 256 | 16 | 64 | 0.250000 | 0.243318 | 0.735948 | 0.980459 | 0.972958 | 0.830280 | 112.094 | 55.214 | 29.896 |
| 256 | 32 | 32 | 0.531250 | 0.560132 | 0.637909 | 0.958351 | 0.971756 | 0.725233 | 111.751 | 29.894 | 29.894 |
| 256 | 32 | 64 | 0.156250 | 0.132987 | 0.801307 | 0.987144 | 0.972654 | 0.884595 | 112.158 | 55.215 | 29.895 |
| 256 | 32 | 96 | 0.093750 | 0.059560 | 0.870588 | 0.991074 | 0.973234 | 0.939075 | 112.604 | 75.769 | 29.895 |
| 256 | 48 | 64 | 0.062500 | 0.075598 | 0.841830 | 0.991180 | 0.972978 | 0.910769 | 112.469 | 55.219 | 29.894 |
| 512 | 0 | 0 | 0.031250 | 0.032305 | 0.935421 | 0.958685 | 0.976851 | 0.976851 | 222.574 | 222.574 | 30.947 |
| 512 | 8 | 64 | 2.531250 | 2.483641 | 0.470320 | 0.764629 | 0.973071 | 0.629927 | 214.319 | 59.604 | 30.947 |
| 512 | 16 | 32 | 3.218750 | 3.189851 | 0.339204 | 0.728417 | 0.974272 | 0.518191 | 214.577 | 30.947 | 30.947 |
| 512 | 16 | 64 | 1.093750 | 1.058421 | 0.660143 | 0.786191 | 0.975936 | 0.697439 | 218.474 | 59.607 | 30.948 |
| 512 | 32 | 32 | 1.187500 | 1.179435 | 0.628832 | 0.780718 | 0.975137 | 0.620869 | 219.654 | 30.947 | 30.947 |
| 512 | 32 | 64 | 0.406250 | 0.396143 | 0.824527 | 0.823078 | 0.976652 | 0.788216 | 221.596 | 59.607 | 30.947 |
| 512 | 32 | 96 | 0.250000 | 0.219804 | 0.896282 | 0.838756 | 0.976940 | 0.863820 | 222.115 | 85.876 | 30.947 |
| 512 | 48 | 64 | 0.125000 | 0.111417 | 0.899543 | 0.886814 | 0.976605 | 0.833913 | 222.102 | 59.609 | 30.947 |

## Interpretation

The broad SVA lookup remains strong. Across both contexts, candidate top-16 recall stays around `0.97`, confirming that `64 tables / 10 bits / probe_radius=2` reliably summons the full-attention keys.

The random prefilter is the new weak point:

- At 256 tokens, `prefilter_dim=48`, `prefilter_budget=64` cuts exact scoring from about `113` candidates to `55` with `loss_delta=0.062500`.
- At 512 tokens, the same setting cuts exact scoring from about `223` to `60`, but loss delta rises to `0.125000`.
- More aggressive exact-score budgets around `31` are too lossy with this random projected prefilter.

This means the next useful invention target is a better cheap ranker, not a broader lookup. The current summon stage finds the right neighborhood; the prefilter needs a more faithful local ranking signal.

## Next Risk

Try prefilters that use structure already present in the model rather than a fresh random projection:

- shared subset of the true head dimensions
- learned or calibrated low-rank projection initialized from Q/K covariance
- two-stage prefilter: cheap lexical/address score plus low-dimensional QK
- layer/head-specific prefilter budgets, especially around the recurring weak pocket near layer 18

