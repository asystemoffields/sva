# Learned Compressed-Ranker Snapshot

Date: 2026-05-13

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`

Run artifact: `results/modal_runs/sva-h100-learned-ranker-20260513-181549.modal.log`

## Setup

This run tested whether a trained low-rank Q/K score can summon the full-attention top keys before exact verification.

- `model_max_position_embeddings=8192`
- `effective_max_length=8192`
- `seq_len=8192`
- layers: `0,1,5,10,18,24,29`
- rank dimensions: `16,32,64`
- exact verifier budgets tested: `64,128,256,512,1024`
- train query positions: 128 held-in positions
- eval query positions: 64 held-out positions
- full-QK target: top 16 keys per sampled query/head
- heads: 9 attention heads
- total key targets per aggregate row: 64,512

For each sampled layer and head, the test trained small asymmetric projections `q -> rank_dim` and `k -> rank_dim` with a multi-positive top-16 objective. Evaluation used only held-out query positions from the same 8192-token sample. Candidate sets were the top budget keys under the learned compressed score, then measured against full-QK top-16 recall.

## Aggregate Results

| phase | rank dim | budget | avg candidates | top-16 recall |
| --- | ---: | ---: | ---: | ---: |
| random | 16 | 64 | 64.0 | 0.069 |
| random | 16 | 128 | 128.0 | 0.108 |
| random | 16 | 256 | 256.0 | 0.177 |
| random | 16 | 512 | 505.8 | 0.293 |
| random | 16 | 1024 | 958.0 | 0.433 |
| random | 32 | 64 | 64.0 | 0.102 |
| random | 32 | 128 | 128.0 | 0.144 |
| random | 32 | 256 | 256.0 | 0.217 |
| random | 32 | 512 | 505.8 | 0.333 |
| random | 32 | 1024 | 958.0 | 0.474 |
| random | 64 | 64 | 64.0 | 0.086 |
| random | 64 | 128 | 128.0 | 0.126 |
| random | 64 | 256 | 256.0 | 0.196 |
| random | 64 | 512 | 505.8 | 0.310 |
| random | 64 | 1024 | 958.0 | 0.453 |
| trained | 16 | 64 | 64.0 | 0.417 |
| trained | 16 | 128 | 128.0 | 0.524 |
| trained | 16 | 256 | 256.0 | 0.638 |
| trained | 16 | 512 | 505.8 | 0.753 |
| trained | 16 | 1024 | 958.0 | 0.850 |
| trained | 32 | 64 | 64.0 | 0.482 |
| trained | 32 | 128 | 128.0 | 0.591 |
| trained | 32 | 256 | 256.0 | 0.700 |
| trained | 32 | 512 | 505.8 | 0.804 |
| trained | 32 | 1024 | 958.0 | 0.888 |
| trained | 64 | 64 | 64.0 | 0.549 |
| trained | 64 | 128 | 128.0 | 0.658 |
| trained | 64 | 256 | 256.0 | 0.760 |
| trained | 64 | 512 | 505.8 | 0.848 |
| trained | 64 | 1024 | 958.0 | 0.915 |

## Interpretation

This is the first strong go signal after the random-address kill. A learned 64-dimensional compressed score recovered `0.759781` aggregate top-16 recall at a 256-key verifier budget and `0.848338` at a 512-key verifier budget. The random baselines at the same budgets were `0.196491` and `0.309787`.

The result says that the real SmolLM2 attention neighborhoods can be described by a much smaller learned score than full Q/K. That gives the address mechanism a concrete target: serve approximate top keys under the learned score, then let exact QK verify.

Layer difficulty still matters. Layer 0 and layer 5 were easy; layers 18 and 24 were harder. For example, trained rank-64 at budget 512 reached `0.982313` recall on layer 0, `0.933268` on layer 5, `0.776584` on layer 18, and `0.714952` on layer 24.

## Caveat

The train/eval split held out query positions, not documents. The key set and text distribution were shared within the same 8192-token sample. This is enough to justify the next test, but the result needs a held-out text run before it can be treated as a general address function.

## Decision

Learned compressed ranking is a go.

The next SVA shape is:

1. learned compact Q/K score to summon a few hundred to about a thousand candidates
2. exact full-QK verifier over that set
3. a sublinear serving structure for the learned score, such as learned binary codes, product quantization, or an ANN index over compressed keys

## Next Risk To Test

Train the low-rank ranker on one or more text samples and evaluate on held-out text. If recall survives, convert the learned score into an addressable lookup:

- learned binary code over the low-rank space
- product-quantized compressed keys
- multi-probe ANN over compressed keys

The next kill/go threshold should stay close to:

- rank-64 or cheaper
- top-16 recall above `0.75`
- verifier budget at or below `512`
- held-out text evaluation
