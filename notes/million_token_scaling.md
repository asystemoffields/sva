# Million-Token Scaling Target

Date: 2026-05-13

## Current Signal

The SmolLM2 socket tests show that pretrained Q/K structure is addressable. SVA can find the same neighborhoods full attention uses when the summoned set is broad enough.

At 512 tokens, the strongest setting was:

- `64 tables`
- `10 bits`
- `probe_radius=2`
- `budget=128`
- `loss_delta=0.015625`
- `top16 full-attention key recall=0.976191`

That setting summons about `221` candidates per query at 512 tokens. In a causal sequence, the average available prefix is about half the context length, so this is still a broad read.

## Million-Token Constraint

At a 1,000,000-token context, average prefix length is about 500,000. A usable replacement should keep exact full-dimensional QK scoring in the rough range of 128 to 1024 candidates per query.

That means the address stage must become much more selective as context grows. Fixed 10-bit addresses cannot be the final million-token shape.

For random independent binary addresses, the expected candidate probability is approximately:

```text
p = 1 - (1 - ball(bits, radius) / 2^bits)^tables
ball(bits, 2) = 1 + bits + bits * (bits - 1) / 2
expected_candidates = average_prefix * p
```

For `tables=64`, `radius=2`, and average prefix `500,000`:

| bits | expected candidates |
| ---: | ---: |
| 20 | ~6400 |
| 21 | ~3500 |
| 22 | ~1900 |
| 23 | ~1050 |
| 24 | ~580 |
| 25 | ~310 |

The million-token address needs to live around 23-25 bits for radius-2 probing if the target exact-scored set is hundreds to about a thousand candidates.

## Candidate Architecture

The likely million-token shape is a three-stage SVA stack:

1. High-resolution summon
   - Use 20-25 bit addresses, not 10-bit addresses.
   - Keep multiple tables and modest probing for robustness.
   - Include a fixed local window and special/global tokens outside the indexed path.

2. Model-aware cheap ranker
   - Rank within the summoned set before exact QK.
   - Use structure from the model rather than a fresh random projection.
   - Good candidates: selected true head dimensions, learned low-rank Q/K projection, product-quantized QK, or a tiny distillation-trained ranker.

3. Exact verifier
   - Run full QK only over the reduced candidate set.
   - Use the original V/O path so the socket remains close to a drop-in attention replacement.

## Key Risk

The 512-token prefilter sweep showed that random low-dimensional ranking is too blunt when pushed hard. It can cut exact scoring, but it loses top-key recall before the quality loss is acceptable at longer context.

So the next invention target is the cheap ranker. The summon stage already finds the right broad neighborhood; the ranker has to preserve the top full-attention keys while shrinking exact scoring by another factor.

## Next Verification Step

Run a high-bit address sweep on real SmolLM2 Q/K activations:

- contexts: 512 and 1024 if feasible
- bits: 14, 16, 18, 20, 22, 24
- tables: 64 and 128
- probe radii: 1 and 2
- metric: top-16 full-attention key recall versus exact-scored candidate count

Then run a million-token retrieval simulation using real SmolLM2 key/query samples plus streamed distractor keys. That test does not need full language-model forward passes over a million tokens; it only needs to answer whether high-resolution addresses can preserve the full-QK top keys at million-token scale.
