# Evidence Rerank Snapshot - 2026-05-14

## Purpose

This run tests the next bottleneck from the evidence-haystack benchmark: when SVA summons evidence into the candidate set, can a better rerank keep it through the final verifier budget?

The sweep compares:

- `current`: exact rerank with the final query state.
- `max_anchor`: exact rerank with the maximum score over recent anchor states.
- `expand_radius`: neighborhood expansion around summoned candidates before rerank.

## Run

- Commit: `911ebf9`
- Modal log: `results/modal_runs/sva-h100-evidence-rerank-20260514-095956.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Layers: `0,15,29`
- Contexts: `8192,16384,32768`
- Placements: `start,middle,end`
- Shortlist and verifier budget: `8192/2048`
- Anchor counts: `1,4,8,16`
- Rerank modes: `current,max_anchor`
- Expansion radii: `0,8,32`

## Best Efficient Rows

Best row per context and placement with `score_reduction >= 1.0`.

| Context | Placement | Policy | Anchors | Rerank | Radius | Verified key | Verified needle | Score reduction | Read reduction |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | start | full | 8 | current | 0 | 0.740741 | 0.925926 | 2.32x | 4.00x |
| 8192 | middle | full | 16 | current | 0 | 0.666667 | 0.888889 | 2.02x | 4.00x |
| 8192 | end | split | 16 | current | 0 | 1.000000 | 1.000000 | 18.85x | 18.85x |
| 16384 | start | full | 4 | current | 32 | 0.888889 | 0.962963 | 1.83x | 8.00x |
| 16384 | middle | full | 4 | current | 8 | 0.259259 | 0.407407 | 2.31x | 8.00x |
| 16384 | end | split | 8 | current | 8 | 0.888889 | 0.888889 | 5.99x | 9.27x |
| 32768 | start | split | 8 | current | 32 | 0.296296 | 0.370370 | 4.01x | 16.00x |
| 32768 | middle | full | 8 | current | 32 | 0.370370 | 0.444444 | 1.69x | 16.00x |
| 32768 | end | full | 4 | max_anchor | 0 | 0.481481 | 0.518519 | 2.19x | 16.00x |

## Baseline Comparison

Baseline is best `current` rerank with `expand_radius=0`.

| Context | Placement | Baseline key | Best efficient key | Delta |
| ---: | --- | ---: | ---: | ---: |
| 8192 | start | 0.740741 | 0.740741 | 0.000000 |
| 8192 | middle | 0.666667 | 0.666667 | 0.000000 |
| 8192 | end | 1.000000 | 1.000000 | 0.000000 |
| 16384 | start | 0.703704 | 0.888889 | 0.185185 |
| 16384 | middle | 0.148148 | 0.259259 | 0.111111 |
| 16384 | end | 0.814815 | 0.888889 | 0.074074 |
| 32768 | start | 0.259259 | 0.296296 | 0.037037 |
| 32768 | middle | 0.333333 | 0.370370 | 0.037037 |
| 32768 | end | 0.444444 | 0.481481 | 0.037037 |

## Aggregate Signal

| Context | Mode | Radius | Verified key | Candidate key | Avg score reduction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8192 | current | 0 | 0.685185 | 0.725309 | 6.956x |
| 8192 | current | 32 | 0.773148 | 0.916667 | 1.521x |
| 8192 | max_anchor | 32 | 0.782408 | 0.916667 | 0.480x |
| 16384 | current | 0 | 0.461420 | 0.507716 | 12.868x |
| 16384 | current | 32 | 0.614198 | 0.746914 | 2.712x |
| 16384 | max_anchor | 32 | 0.601852 | 0.746914 | 0.913x |
| 32768 | current | 0 | 0.265432 | 0.353395 | 20.803x |
| 32768 | current | 32 | 0.307098 | 0.601852 | 2.961x |
| 32768 | max_anchor | 32 | 0.310185 | 0.601852 | 1.110x |

## Interpretation

Neighborhood expansion is a real method-level lever. At `16384`, current-query rerank plus radius `32` raises aggregate verified key survival from `0.461420` to `0.614198` while keeping average exact score work below full attention.

At `32768`, expansion exposes the next failure mode. Current-query radius `32` raises candidate key coverage to `0.601852`, but verified key survival reaches only `0.307098`. The evidence often reaches the candidate set and then loses the final individual-token rerank.

`max_anchor` rerank helps some individual rows, but its score cost grows quickly with anchor count and expansion size. The best `32768` end row improves exact key survival from `0.444444` to `0.481481` at `2.19x` score reduction, while many high-anchor expanded rows spend more exact scores than full attention.

## Next Step

The next test should move from individual-token rerank to span/block statements:

- summon sparse evidence tokens,
- merge them into small contiguous spans or fixed blocks,
- let each selected span compute a local exact statement,
- merge those statements for the final answer.

This directly targets the `candidate_key_hit` to `verified_key_hit` gap by preserving neighborhood structure after summon.
