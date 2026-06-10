# SVA Token/Block Hybrid Snapshot

This snapshot records the first dynamic hybrid test between scattered token SVA and block-elevator SVA.

## Question

Can SVA route each head/query to the right kind of witness?

- Token SVA preserves exact-token retrieval by verifying scattered token candidates.
- Block SVA preserves diffuse value output by reading contiguous blocks and merging local softmax statements.

The hybrid test asks whether a cheap selector can use token SVA for peaky/exact cases and block SVA for diffuse cases.

## Run

- Modal app: `sva-block-hybrid-h100`
- Function call: `fc-01KRK924RRS6GAR448F9YG1C7V`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Base activations: real SmolLM2 Q/K/V from an 8192-token held-out document
- Target contexts: `8192`, `32768`, `131072`
- Layers: `0`, `15`, `29`
- Queries: 8 late-context positions per layer
- Teacher: exact full attention output and top-16 keys
- Synthetic extension: sample real base keys/values with `0.01` Gaussian noise

The equal-read comparison below uses `2048` value reads:

- token SVA: `2048` scattered token segments
- block SVA `64 x 32`: `32` contiguous block segments
- block SVA `128 x 16`: `16` contiguous block segments

## Baselines

| method | context | segments | top-16 recall | output cosine | relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| token | 8192 | 2048 | 1.000000 | 0.995610 | 0.081846 |
| block `64 x 32` | 8192 | 32 | 0.349248 | 0.980028 | 0.169727 |
| block `128 x 16` | 8192 | 16 | 0.320023 | 0.977871 | 0.176530 |
| token | 32768 | 2048 | 0.984664 | 0.979632 | 0.243745 |
| block `64 x 32` | 32768 | 32 | 0.104167 | 0.975399 | 0.217723 |
| block `128 x 16` | 32768 | 16 | 0.098958 | 0.968733 | 0.235360 |
| token | 131072 | 2048 | 0.890914 | 0.944623 | 0.457266 |
| block `64 x 32` | 131072 | 32 | 0.026910 | 0.969223 | 0.270862 |
| block `128 x 16` | 131072 | 16 | 0.027199 | 0.965356 | 0.259839 |

## Oracle Hybrid

The oracle selector chooses token or block per head/query using the lower output error against the teacher. This is an upper bound on how much a learned selector could win.

| context | block | token fraction | avg segments | segment reduction | top-16 recall | output cosine | relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | `64 x 32` | 0.791667 | 1628.00 | 1.46x | 0.867766 | 0.998782 | 0.036986 |
| 8192 | `128 x 16` | 0.777778 | 1596.44 | 1.51x | 0.846354 | 0.998826 | 0.034765 |
| 32768 | `64 x 32` | 0.569444 | 1180.00 | 2.87x | 0.610243 | 0.994715 | 0.094606 |
| 32768 | `128 x 16` | 0.597222 | 1229.56 | 2.50x | 0.630498 | 0.994938 | 0.089417 |
| 131072 | `64 x 32` | 0.407407 | 853.33 | 5.12x | 0.409722 | 0.986559 | 0.165284 |
| 131072 | `128 x 16` | 0.407407 | 843.85 | 5.50x | 0.406829 | 0.985461 | 0.161267 |

The upper bound is strong. At `131072`, the oracle hybrid keeps the same `2048` average value reads, cuts scattered segments by about `5x`, and improves relative error from token SVA's `0.457266` and block SVA's `0.270862` to `0.165284`.

## Cheap Entropy Selector

The first deployable selector uses normalized coarse-score entropy. It chooses token SVA when entropy is at or below `0.55`, and block SVA otherwise.

| context | block | token fraction | avg segments | segment reduction | top-16 recall | output cosine | relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | `64 x 32` | 0.865741 | 1777.33 | 1.16x | 0.911748 | 0.994431 | 0.088338 |
| 8192 | `128 x 16` | 0.865741 | 1775.19 | 1.16x | 0.905961 | 0.994721 | 0.087112 |
| 32768 | `64 x 32` | 0.509259 | 1058.67 | 1.98x | 0.553530 | 0.980996 | 0.212533 |
| 32768 | `128 x 16` | 0.509259 | 1050.81 | 2.00x | 0.550926 | 0.978469 | 0.220915 |
| 131072 | `64 x 32` | 0.296296 | 629.33 | 3.92x | 0.296296 | 0.967016 | 0.278122 |
| 131072 | `128 x 16` | 0.296296 | 618.07 | 4.04x | 0.293692 | 0.965908 | 0.276191 |

The cheap selector is informative but underpowered. It captures enough signal to improve the `32768` row and greatly raises top-16 recall over block-only at `131072`, but it does not reach the oracle's output quality. The gap says the next object should be a learned dispatcher, not only a hand threshold.

## Reading

This is a go for the hybrid branch. Token and block SVA fail in different ways:

- token SVA keeps top-key recall but can preserve the wrong value mixture at long synthetic contexts;
- block SVA preserves the diffuse value mixture but loses exact top keys;
- an oracle mixture beats both, so the two paths contain complementary information.

The next decisive test is a learned selector trained on cheap features:

- coarse entropy and margin
- block centroid margin
- token/block output disagreement proxy
- layer and head id
- context length

Then rerun the passkey language benchmark with hybrid mode. The target is to preserve exact-string behavior while using block statements when the output is diffuse enough to benefit.
