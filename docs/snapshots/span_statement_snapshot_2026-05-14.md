# Span Statement Snapshot - 2026-05-14

## Purpose

This run tests whether summoned evidence should be verified as local statements instead of isolated tokens.

The policy is:

- summon seed tokens from recent anchor states,
- score those seeds cheaply with exact current-query or max-anchor scores,
- open local spans around the best seeds,
- run exact attention over the selected span union,
- compare the selected-span output with full attention for the final query.

## Run

- Commit: `9485c63`
- Modal log: `results/modal_runs/sva-h100-span-statement-20260514-102409.full.log`
- Modal function: `fc-01KRKDZ9Y39ZFPAR5JPXTM1GN2`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Layers: `0,15,29`
- Contexts: `8192,16384,32768`
- Placements: `start,middle,end`
- Seed shortlist and anchor budget: `8192/2048`
- Anchor counts: `4,8,16`
- Seed score modes: `current,max_anchor`
- Span radii: `0,8,32`
- Span token budgets: `1024,2048,4096`
- Result volume: `1458` layer rows and `486` summary rows, exit `0`

## Best Efficient Rows

Best row per context and placement with `score_reduction >= 4.0`.

| Context | Placement | Mode | Anchors | Radius | Span budget | Output cosine | Relative error | Score reduction | Read reduction | Avg segments | Span key hit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | start | current | 16 | 32 | 1024 | 0.997496 | 0.0559 | 5.54x | 8.01x | 4.4 | 0.667 |
| 8192 | middle | current | 8 | 8 | 1024 | 0.995961 | 0.0697 | 4.87x | 8.00x | 33.1 | 0.333 |
| 8192 | end | current | 8 | 32 | 1024 | 0.997710 | 0.0497 | 4.95x | 8.00x | 4.8 | 1.000 |
| 16384 | start | current | 4 | 8 | 4096 | 0.992012 | 0.0799 | 4.08x | 5.42x | 67.1 | 0.630 |
| 16384 | middle | current | 8 | 32 | 4096 | 0.992140 | 0.0759 | 4.21x | 5.11x | 16.1 | 0.185 |
| 16384 | end | current | 8 | 32 | 4096 | 0.993407 | 0.0717 | 4.22x | 5.13x | 16.4 | 0.889 |
| 32768 | start | current | 4 | 8 | 4096 | 0.966933 | 0.2132 | 6.32x | 8.26x | 109.6 | 0.259 |
| 32768 | middle | current | 16 | 32 | 4096 | 0.971550 | 0.2575 | 7.05x | 8.20x | 24.7 | 0.259 |
| 32768 | end | current | 4 | 32 | 4096 | 0.951055 | 0.3291 | 6.28x | 8.03x | 21.8 | 0.333 |

## Radius Comparison

Aggregate over efficient rows with `score_reduction >= 1.0`.

| Context | Radius | Rows | Output cosine | Relative error | Score reduction | Read reduction | Avg segments | Span key hit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 0 | 54 | 0.991612 | 0.1230 | 4.10x | 13.48x | 272.6 | 0.551 |
| 8192 | 8 | 46 | 0.997589 | 0.0464 | 2.71x | 5.28x | 52.7 | 0.659 |
| 8192 | 32 | 43 | 0.998145 | 0.0427 | 2.64x | 5.09x | 7.8 | 0.739 |
| 16384 | 0 | 54 | 0.962010 | 0.3306 | 7.51x | 24.85x | 353.8 | 0.374 |
| 16384 | 8 | 54 | 0.985230 | 0.1301 | 4.70x | 10.65x | 45.3 | 0.474 |
| 16384 | 32 | 54 | 0.982172 | 0.1299 | 4.43x | 9.85x | 11.2 | 0.534 |
| 32768 | 0 | 54 | 0.890605 | 0.6079 | 11.74x | 39.00x | 532.3 | 0.212 |
| 32768 | 8 | 54 | 0.925220 | 0.3996 | 7.70x | 18.91x | 73.7 | 0.257 |
| 32768 | 32 | 54 | 0.907434 | 0.4521 | 7.61x | 18.70x | 15.7 | 0.192 |

## Interpretation

Span statements are a useful verifier shape. At `8192`, radius `32` improves aggregate output cosine from `0.991612` to `0.998145`, cuts relative error from `0.1230` to `0.0427`, and reduces scattered segments from about `273` to about `8`. At `16384`, span statements still lift output quality substantially while turning hundreds of token reads into a few dozen local statements.

The `32768` rows expose the current bottleneck. Spans improve the output over isolated token verification, but key survival stays low: the best efficient rows hit the key in only about `0.259-0.333` of sampled head/layer cases. Once the summoner misses the evidence, a statement-shaped verifier cannot recover it.

`current` seed scoring is the better default in this sweep. It averaged `output_cosine=0.965102` with `9.18x` exact-score reduction, versus `0.951254` and `2.63x` for `max_anchor`. Max-anchor scoring spends many more exact scores and only wins a few individual rows.

## Next Step

Keep span statements as part of the verifier/serving shape, but move the main pressure back to summon catalog quality.

The active follow-up is the rotation diagnostic: compare the frozen artifact codebooks against refit identity and Hadamard-style low-rank rotations. If rotation improves score alignment or top-key survival at the same budget, the next artifact export should absorb the rotation into the low-rank projections and train codebooks in the rotated space.
