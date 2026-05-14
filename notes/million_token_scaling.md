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

The synthetic million-token PQ scan benchmark showed plausible but nontrivial throughput. On H100 with stock PyTorch gather plus top-k, `8 x 256` PQ over 9 heads scanned one million keys in about `2.2 ms` for one query; `16 x 256` took about `4.5 ms`. Used in every layer of a 30-layer model, that is too much without more structure or a custom kernel, but it is fast enough to justify a coarse-to-fine serving test.

The working conclusion is now sharper: the broad SVA socket works, and the learned low-rank Q/K score works as a compact ranking signal. Hard lookup cells have been weak, but score-preserving compressed scans can keep most of the learned-ranker signal at plausible primitive speed.

The span-statement branch is now measured. At `8192`, radius `32` improved aggregate selected-output cosine from `0.991612` to `0.998145` while reducing scattered segments from about `273` to about `8`. At `16384`, the best efficient rows stayed near `0.992-0.993` cosine. At `32768`, span statements improved over isolated token verification but did not preserve enough evidence: the best efficient rows had key survival around `0.259-0.333`. The serving shape remains useful, but the main bottleneck is again summon catalog quality.

The rotation diagnostic found a large codebook-quality opening. At budgets `512/1024/2048`, frozen artifact codebooks reached aggregate teacher top-16 recall `0.771888/0.837511/0.893808`; refit codebooks reached `0.837637/0.884145/0.923472`. PQ score cosine rose from `0.870095` to about `0.9576`, and code entropy rose from `0.812974` to about `0.986`. Hadamard and signed-Hadamard refits were close to plain refit, so the immediate win is better codebook fit and balance. The next test should train identity and signed-Hadamard codebooks on separate calibration streams, then evaluate held-out long-context recall.

Held-out codebook refresh confirmed the codebook-quality opening. At `32768`, the frozen artifact reached teacher top-16 recall `0.563169/0.657407/0.752450` at budgets `512/1024/2048`; calibration-fit codebooks lifted those to `0.635887/0.726002/0.809995`, close to the eval-key refit upper bound `0.645553/0.734592/0.816090`. The Shannon-style diagnostic is clear: normalized code entropy rose from `0.718895` to about `0.978`, and the largest average code bucket fell from `0.230200` to about `0.0144`. At `8192`, the frozen artifact remains best, which points to context-matched catalog profiles rather than one global codebook.

The first refreshed long-context artifact now exists locally at `results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1`. It preserves the long-context gain as a deployable profile: at `32768`, it reaches teacher top-16 recall `0.630588/0.725071/0.809860` at budgets `512/1024/2048`, with score cosine `0.945643`, normalized code entropy `0.978694`, and max code fraction `0.014597`. At `8192`, it reaches `0.952637/0.988589/0.998788`, below the original artifact's `0.972367/0.993978/0.999295`, so the next production path is profile routing.

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
   - Track catalog capacity with normalized code entropy, max code load, and score distortion.

3. Exact verifier
   - Run full QK only over the reduced candidate set.
   - Use the original V/O path so the socket remains close to a drop-in attention replacement.

4. Context-matched catalog profiles
   - Keep the strong 8k artifact profile for ordinary context.
   - Use calibration-refreshed long-context profiles when the served context crosses the 16k/32k range.
   - Route by context length first, then later by measured per-layer catalog statistics.

## Key Risk

The 512-token prefilter sweep showed that random low-dimensional ranking is too blunt when pushed hard. It can cut exact scoring, but it loses top-key recall before the quality loss is acceptable at longer context.

So the next invention target is the cheap ranker and its catalog. The summon stage has to preserve the top full-attention keys while shrinking exact scoring by another factor, and the catalog needs enough effective entropy that increasing context length does not collapse many useful keys into the same overloaded codes.

The current working formula is:

```text
teacher_recall = f(context_length, verifier_budget, score_cosine, score_mse, normalized_code_entropy, max_code_fraction)
```

The important refinement is distribution matching. High code entropy is useful when it is measured on the distribution being served; a 32k-calibrated catalog can improve 16k/32k recall while giving back some of the 8k artifact's advantage.

## Next Verification Step

The next architectural test is language-facing context-matched serving:

- keep the exact verifier unchanged
- compare the current 8k artifact, the refreshed long-context profile, and a context-length router
- track recall, score distortion, normalized code entropy, max code load, value reads, and wall time
- rerun the passkey and long-context recall benchmarks with the selected profile
