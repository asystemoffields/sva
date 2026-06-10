# Longer Pretrained Socket Snapshot

Date: 2026-05-13

## Question

Does the pretrained SmolLM2 socket result survive longer contexts, and do misses come from SVA lookup or from the post-lookup budget?

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Harness: `experiments/sva_pretrained_socket_test.py`
- Runner: `modal_h100_socket_long.py`
- Hardware: Modal H100
- Dtype: bfloat16
- Text count: 3 generated long samples
- Effective context lengths: 128, 256, 512 tokens
- Diagnostic: top-16 full-attention key recall per layer/head

The diagnostic harness uses the model's full QK scores to measure how well SVA finds the keys full attention would have favored. `avg_summoned` is the exact-scored candidate set in this harness. `avg_verified` is the post-score top-k/value-aggregation set.

## Result

| seq | tables | bits | probe | budget | loss_delta | KL to full | top1 agreement | logit cosine | top16 recall | avg summoned | avg verified |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 8 | 1 | 64 | 0.156250 | 0.285056 | 0.698163 | 0.937450 | 0.777949 | 39.859 | 36.314 |
| 128 | 32 | 10 | 2 | 64 | 0.062500 | 0.127609 | 0.805774 | 0.981270 | 0.870413 | 47.351 | 41.134 |
| 128 | 64 | 8 | 1 | 64 | 0.031250 | 0.109176 | 0.855643 | 0.984255 | 0.923974 | 52.593 | 44.142 |
| 128 | 64 | 10 | 1 | 64 | 0.343750 | 0.541630 | 0.603675 | 0.882516 | 0.651813 | 29.806 | 28.214 |
| 128 | 64 | 10 | 2 | 64 | 0.000000 | 0.007494 | 0.950131 | 0.998202 | 0.970278 | 58.024 | 46.474 |
| 128 | 64 | 10 | 2 | 32 | 0.000000 | 0.017676 | 0.931759 | 0.995601 | 0.970100 | 58.093 | 27.790 |
| 128 | 64 | 10 | 2 | 128 | 0.000000 | 0.005820 | 0.955381 | 0.998350 | 0.970245 | 58.020 | 58.020 |
| 256 | 32 | 8 | 1 | 64 | 0.187500 | 0.251438 | 0.701961 | 0.954502 | 0.788904 | 74.622 | 48.554 |
| 256 | 32 | 10 | 2 | 64 | 0.062500 | 0.103294 | 0.813072 | 0.985445 | 0.880120 | 90.010 | 52.085 |
| 256 | 64 | 8 | 1 | 64 | 0.031250 | 0.079383 | 0.856209 | 0.988235 | 0.930826 | 101.189 | 53.989 |
| 256 | 64 | 10 | 1 | 64 | 0.468750 | 0.600137 | 0.541176 | 0.912885 | 0.670554 | 54.685 | 40.791 |
| 256 | 64 | 10 | 2 | 64 | 0.031250 | 0.012986 | 0.930719 | 0.997062 | 0.973465 | 112.734 | 55.216 |
| 256 | 64 | 10 | 2 | 32 | 0.031250 | 0.036275 | 0.896732 | 0.992801 | 0.973725 | 113.128 | 29.895 |
| 256 | 64 | 10 | 2 | 128 | 0.000000 | 0.006273 | 0.950327 | 0.998653 | 0.973300 | 112.570 | 91.427 |
| 512 | 32 | 8 | 1 | 64 | 0.406250 | 0.403896 | 0.784083 | 0.807004 | 0.803493 | 144.114 | 56.080 |
| 512 | 32 | 10 | 2 | 64 | 0.203125 | 0.206748 | 0.872146 | 0.840406 | 0.889917 | 175.074 | 58.001 |
| 512 | 64 | 8 | 1 | 64 | 0.109375 | 0.122997 | 0.903457 | 0.879576 | 0.936442 | 197.039 | 58.993 |
| 512 | 64 | 10 | 1 | 64 | 1.078125 | 1.115355 | 0.598174 | 0.745144 | 0.689022 | 103.232 | 50.939 |
| 512 | 64 | 10 | 2 | 64 | 0.031250 | 0.018961 | 0.960209 | 0.958182 | 0.976451 | 221.688 | 59.608 |
| 512 | 64 | 10 | 2 | 32 | 0.031250 | 0.032305 | 0.935421 | 0.958685 | 0.976851 | 222.574 | 30.947 |
| 512 | 64 | 10 | 2 | 128 | 0.015625 | 0.009582 | 0.970646 | 0.973771 | 0.976191 | 221.044 | 109.673 |

Best quality by context:

- 128 tokens: `64 tables`, `10 bits`, `probe_radius=2`, `budget=128`; `loss_delta=0.000000`, `KL=0.005820`, `top1=0.955381`.
- 256 tokens: `64 tables`, `10 bits`, `probe_radius=2`, `budget=128`; `loss_delta=0.000000`, `KL=0.006273`, `top1=0.950327`.
- 512 tokens: `64 tables`, `10 bits`, `probe_radius=2`, `budget=128`; `loss_delta=0.015625`, `KL=0.009582`, `top1=0.970646`.

The smaller `budget=32` variant also held up:

- 128 tokens: `loss_delta=0.000000`, `top1=0.931759`, `avg_verified=27.790`.
- 256 tokens: `loss_delta=0.031250`, `top1=0.896732`, `avg_verified=29.895`.
- 512 tokens: `loss_delta=0.031250`, `top1=0.935421`, `avg_verified=30.947`.

## Interpretation

This is a strong go signal for socket compatibility. Existing pretrained Q/K structure is rich enough for SVA-style lookup across 512-token contexts in SmolLM2-135M.

The important failure boundary is now clearer:

- `64 tables / 10 bits / probe 1` is too selective and misses the full-attention top keys.
- `64 tables / 10 bits / probe 2` recovers top-key recall near `0.97` across 128, 256, and 512 tokens.
- Lowering the post-score budget from 128 to 32 barely changes top-16 key recall, which points away from the post-score budget as the immediate bottleneck.
- The exact-scored summoned set is still broad at 512 tokens: about `221` candidates for the best `64x10 probe 2` setting.

Layer/head diagnostics repeatedly identify the weakest pocket around layer 18, especially head 8. In the best 512-token `64x10 probe 2 budget 128` run, the worst reported head was layer 18 head 8 with top-16 recall `0.701685`, while aggregate recall stayed `0.976191`.

## Next Risk

The next target is to reduce exact-scored candidates without losing the top keys:

- add a cheap prefilter inside the pretrained socket harness
- compare `avg_summoned`, `avg_prefiltered`, and final top-16 recall
- test whether the weak layer/head pockets need wider probing, more tables, or layer-specific addressing

The promising path is now `64 tables / 10 bits / probe radius 2`, with a cheap prefilter to turn the broad summoned set into a much smaller exact-score set.

