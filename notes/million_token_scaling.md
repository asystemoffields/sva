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

The first full-window address sweep used SmolLM2's actual configured context window:

- `model_max_position_embeddings=8192`
- `seq_len=8192`
- sampled layers: `0,1,5,10,18,24,29`
- full-QK target: top 16 keys per sampled query/head

Aggregate result: random high-bit binary addresses split the tradeoff sharply. `14 bits / 128 tables / radius 2` reached `0.838557` top-16 recall, but projects to about `282k` random candidates at a million tokens. `24 bits / 128 tables / radius 2` projects to about `1.1k` random candidates, but reached only `0.092231` top-16 recall.

The completed million-token pressure simulation used the same real 8192-token SmolLM2 Q/K samples and projected empirical address hit density to a million-token prefix. The best aggregate recall was `20 bits / 256 tables / radius 2` at `0.384905`, but it projected to about `39.6k` average empirical candidates at a million tokens, with p95 about `129k`. In the rough `128-1024` average candidate band, aggregate recall stayed around 1-2%.

The learned compressed-ranker test gives the first strong positive signal after that kill. A trained 64-dimensional Q/K score reached `0.759781` aggregate top-16 recall at a 256-candidate verifier budget and `0.848338` at a 512-candidate budget, using held-out query positions from the same 8192-token sample.

The working conclusion is now sharper: the broad SVA socket works, and the next invention has to be the address code. Million-token retrieval needs a richer compressed catalog than random binary addresses, and a learned low-rank Q/K score is now the first useful catalog target.

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
   - Current best target: learned low-rank Q/K projection.
   - Good serving candidates: learned binary code over the low-rank space, product-quantized QK, or an ANN index over compressed keys.

3. Exact verifier
   - Run full QK only over the reduced candidate set.
   - Use the original V/O path so the socket remains close to a drop-in attention replacement.

## Key Risk

The 512-token prefilter sweep showed that random low-dimensional ranking is too blunt when pushed hard. It can cut exact scoring, but it loses top-key recall before the quality loss is acceptable at longer context.

So the next invention target is the cheap ranker. The summon stage already finds the right broad neighborhood; the ranker has to preserve the top full-attention keys while shrinking exact scoring by another factor.

## Next Verification Step

The next architectural test is held-out text generalization for the learned ranker:

- train on one or more 8192-token samples
- evaluate on separate text samples
- keep the exact verifier unchanged
- keep the target at rank-64 or cheaper, top-16 recall above `0.75`, and verifier budget at or below `512`

If that holds, convert the learned low-rank score into a true addressable lookup and rerun the million-token pressure simulation.
