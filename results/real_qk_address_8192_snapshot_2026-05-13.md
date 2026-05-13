# Real-QK Address Sweep At SmolLM2 Window

Date: 2026-05-13

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`

Run artifact: `results/modal_runs/sva-h100-real-qk-address-20260513-173816.modal.log`

## Setup

This run used SmolLM2's configured context window:

- `model_max_position_embeddings=8192`
- `effective_max_length=8192`
- `seq_len=8192`
- layers: `0,1,5,10,18,24,29`
- top keys: full-QK top 16 per sampled query
- sampled queries: 64 positions from token 128 through token 8191
- heads: 9 attention heads
- total key targets per aggregate row: 64,512

The lookup used random binary address projections over the model's real post-RoPE Q/K vectors. The metric is whether the address lookup includes the keys that full attention would rank in its top 16.

## Aggregate Results

| bits | tables | radius | avg candidates at 8192 | top-16 recall | random expected candidates at 1M |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 64 | 1 | 160.6 | 0.292 | 28,467.7 |
| 14 | 64 | 2 | 781.2 | 0.693 | 169,964.5 |
| 14 | 128 | 1 | 291.5 | 0.442 | 55,314.7 |
| 14 | 128 | 2 | 1,236.0 | 0.839 | 282,153.1 |
| 16 | 64 | 1 | 56.6 | 0.154 | 8,233.3 |
| 16 | 64 | 2 | 322.4 | 0.469 | 62,674.0 |
| 16 | 128 | 1 | 106.6 | 0.257 | 16,331.1 |
| 16 | 128 | 2 | 550.4 | 0.630 | 117,491.9 |
| 18 | 64 | 1 | 20.1 | 0.079 | 2,314.0 |
| 18 | 64 | 2 | 129.2 | 0.290 | 20,568.0 |
| 18 | 128 | 1 | 39.0 | 0.141 | 4,617.4 |
| 18 | 128 | 2 | 231.0 | 0.429 | 40,289.9 |
| 20 | 64 | 1 | 7.3 | 0.040 | 640.5 |
| 20 | 64 | 2 | 51.7 | 0.170 | 6,398.6 |
| 20 | 128 | 1 | 14.6 | 0.074 | 1,280.1 |
| 20 | 128 | 2 | 96.0 | 0.272 | 12,715.2 |
| 22 | 64 | 1 | 2.8 | 0.020 | 175.4 |
| 22 | 64 | 2 | 21.1 | 0.097 | 1,934.2 |
| 22 | 128 | 1 | 5.7 | 0.038 | 350.8 |
| 22 | 128 | 2 | 40.3 | 0.164 | 3,860.9 |
| 24 | 64 | 1 | 1.1 | 0.010 | 47.7 |
| 24 | 64 | 2 | 8.8 | 0.052 | 573.8 |
| 24 | 128 | 1 | 2.3 | 0.019 | 95.4 |
| 24 | 128 | 2 | 17.2 | 0.092 | 1,146.9 |

## Interpretation

The low-bit settings can recover meaningful full-attention neighborhoods, but their projected million-token candidate counts are far above the target band. The high-bit settings land near the desired million-token selectivity, but they lose most of the full-QK top keys.

The strongest aggregate recall was `14 bits / 128 tables / radius 2` at `0.839`, with about `1,236` candidates at 8192 tokens and a random million-token candidate estimate of about `282k`.

The most million-selective practical setting in this sweep was around `24 bits / radius 2`, but `24 bits / 128 tables / radius 2` only reached `0.092` aggregate top-16 recall.

Layer 0 was much easier than later layers. For example, `16 bits / 128 tables / radius 2` reached `0.933` recall in layer 0, but the aggregate across layers was `0.630`.

## Decision

Random high-bit binary addresses are a kill for the million-token version of this exact address function.

SVA remains a go as an architecture because the earlier socket test showed that broad summon plus exact verify can preserve the pretrained model's behavior. The next invention target is the address function: it needs to be model-aware, trained, or multi-scale so that it compresses the QK neighborhood structure instead of slicing it with random hyperplanes.

## Next Risk To Test

Train or fit a compact address projection per layer/head against full-QK top-key labels, then rerun this same 8192-window address sweep. The kill/go criterion should be:

- top-16 recall above `0.75` across sampled layers
- empirical million-token candidate count in the rough `128-1024` range after the million-stream simulation
- no change to the verifier path: exact QK still resolves the final attended keys
