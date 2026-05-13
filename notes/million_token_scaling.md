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

The held-out text test preserved the signal. Training on one 8192-token stream and evaluating on a reversed 8192-token stream reached `0.749752` aggregate top-16 recall at a 256-candidate verifier budget and `0.835488` at a 512-candidate budget.

The learned-score LSH serving test answered the next question. Random-hyperplane LSH over the rank-64 Q/K space reached only `0.233429` aggregate top-16 recall in its best row, while projecting to about `38.6k` average candidates at a million-token context. In the rough few-hundred-candidate band, aggregate recall stayed around `0.013`.

The score-aware IVF serving test improved the tradeoff. Single-write k-means centroids over learned low-rank keys reached `0.234422` recall at about `3.5k` projected million-token candidates, and about `0.095-0.102` recall in the few-hundred-candidate band. That is a large gain over sign-LSH at the same candidate scale, but still well below the learned ranker's all-key score.

The multi-write IVF follow-up narrowed the branch. Giving each key several nearest-centroid writes reached `0.105422` recall at about `898` projected million-token candidates and `0.147647` at about `1,564` projected candidates. Those rows are close to, or below, comparable single-write IVF rows. The useful conclusion is that unsupervised centroid geometry is not the missing catalog by itself.

The first supervised query-cell router showed that direct top-key supervision can move recall, reaching `0.655816` at about `167k` projected million-token candidates. The low-resolution cells were too dense: the smallest setting reached only `0.039109` recall at about `3.7k` projected candidates. The next version needs high-resolution cells and smaller write/probe settings.

The high-resolution supervised router reached the target density but lost the signal. In the `128-1024` projected-candidate band, recall was only about `0.002-0.013`. At about `3.6k` projected candidates, recall was `0.042721`, far below single-write IVF's `0.234422` at a similar candidate count.

Product-quantized learned-score lookup is the first strong serving result after the learned ranker. The exact learned rank-64 scorer reached `0.754046` recall at a 256-candidate verifier budget and `0.839084` at 512. PQ with `16 subspaces / 256 codewords` reached `0.704985` at 256 and `0.803184` at 512. A more compact `8 subspaces / 256 codewords` version reached `0.647166` at 256 and `0.755937` at 512.

The working conclusion is now sharper: the broad SVA socket works, and the learned low-rank Q/K score works as a compact ranking signal. Hard lookup cells have been weak, but score-preserving compressed scans can keep most of the learned-ranker signal.

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
   - Current ranking signal: learned low-rank Q/K projection.
   - Current best serving target: product-quantized learned Q/K scoring.
   - Good serving candidates: coarse-to-fine PQ, asymmetric compressed scoring, or an ANN index over compressed keys.

3. Exact verifier
   - Run full QK only over the reduced candidate set.
   - Use the original V/O path so the socket remains close to a drop-in attention replacement.

## Key Risk

The 512-token prefilter sweep showed that random low-dimensional ranking is too blunt when pushed hard. It can cut exact scoring, but it loses top-key recall before the quality loss is acceptable at longer context.

So the next invention target is the cheap ranker. The summon stage already finds the right broad neighborhood; the ranker has to preserve the top full-attention keys while shrinking exact scoring by another factor.

## Next Verification Step

The next architectural test is efficient serving for the learned ranker:

- keep the exact verifier unchanged
- convert the rank-64 score into a true addressable lookup without random sign buckets or unsupervised centroid cells
- test PQ scan throughput and coarse-to-fine PQ over compressed keys
- keep the target at top-16 recall above `0.75` and verifier budget at or below `512`
- rerun the million-token pressure simulation with empirical candidate density
