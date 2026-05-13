# Million-Token Address-Pressure Snapshot

Date: 2026-05-13

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`

Run artifact: `results/modal_runs/sva-h100-million-stream-20260513-174843.modal.log`

## Setup

This run kept the actual SmolLM2 functional window and extrapolated address density to a million-token prefix:

- `model_max_position_embeddings=8192`
- `effective_max_length=8192`
- `seq_len=8192`
- `target_context=1000000`
- layers: `0,1,5,10,18,24,29`
- full-QK target: top 16 keys per sampled query/head
- sampled queries: 64 positions from token 128 through token 8191
- heads: 9 attention heads
- total key targets per aggregate row: 64,512

The lookup used random binary address projections over real post-RoPE Q/K vectors. For each configuration, the run measured actual candidate counts at 8192 tokens, then estimated empirical million-token candidates from the observed real-QK address hit density.

## Aggregate Results

| bits | tables | radius | avg candidates at 8192 | empirical avg candidates at 1M | empirical p95 at 1M | top-16 recall | random expected candidates at 1M |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 64 | 1 | 7.2 | 2,099.5 | 8,859.7 | 0.038 | 640.5 |
| 20 | 64 | 2 | 50.5 | 13,046.0 | 48,891.9 | 0.168 | 6,398.6 |
| 20 | 128 | 1 | 14.2 | 4,013.5 | 16,653.5 | 0.070 | 1,280.1 |
| 20 | 128 | 2 | 93.7 | 23,124.5 | 81,921.1 | 0.266 | 12,715.2 |
| 20 | 256 | 1 | 27.7 | 7,636.1 | 31,068.1 | 0.120 | 2,556.9 |
| 20 | 256 | 2 | 168.5 | 39,565.1 | 129,423.2 | 0.385 | 25,107.1 |
| 22 | 64 | 1 | 2.8 | 841.0 | 3,861.3 | 0.019 | 175.4 |
| 22 | 64 | 2 | 20.4 | 5,750.8 | 23,777.7 | 0.094 | 1,934.2 |
| 22 | 128 | 1 | 5.4 | 1,655.9 | 7,349.1 | 0.035 | 350.8 |
| 22 | 128 | 2 | 38.7 | 10,609.7 | 42,468.5 | 0.158 | 3,860.9 |
| 22 | 256 | 1 | 10.7 | 3,221.0 | 14,425.1 | 0.063 | 701.4 |
| 22 | 256 | 2 | 72.0 | 18,997.5 | 72,925.5 | 0.243 | 7,691.9 |
| 24 | 64 | 1 | 1.1 | 354.2 | 1,771.7 | 0.009 | 47.7 |
| 24 | 64 | 2 | 8.4 | 2,572.9 | 11,525.0 | 0.052 | 573.8 |
| 24 | 128 | 1 | 2.1 | 704.7 | 3,374.0 | 0.017 | 95.4 |
| 24 | 128 | 2 | 16.3 | 4,911.4 | 21,471.7 | 0.092 | 1,146.9 |
| 24 | 256 | 1 | 4.2 | 1,370.2 | 6,543.5 | 0.033 | 190.7 |
| 24 | 256 | 2 | 30.7 | 8,991.5 | 38,638.9 | 0.147 | 2,291.2 |
| 26 | 64 | 1 | 0.4 | 150.8 | 775.2 | 0.005 | 12.9 |
| 26 | 64 | 2 | 3.6 | 1,153.1 | 5,547.5 | 0.027 | 167.8 |
| 26 | 128 | 1 | 0.9 | 298.3 | 1,502.2 | 0.009 | 25.7 |
| 26 | 128 | 2 | 7.0 | 2,231.6 | 10,450.8 | 0.050 | 335.6 |
| 26 | 256 | 1 | 1.7 | 584.8 | 2,957.2 | 0.016 | 51.5 |
| 26 | 256 | 2 | 13.4 | 4,191.5 | 19,486.7 | 0.084 | 670.9 |

## Interpretation

The real-QK hit density is much clumpier than the independent random estimate. In several practical-looking settings, empirical million-token candidate counts are many times larger than the formula predicts.

The best aggregate recall in this sweep was `20 bits / 256 tables / radius 2` at `0.384905`, but it projected to about `39.6k` average empirical candidates at a million tokens, with p95 about `129k`.

The practical candidate band did not recover enough keys. In the rough `128-1024` average candidate range, the best aggregate recall was `22 bits / 64 tables / radius 1` at `0.018617` with about `841` empirical candidates. `24 bits / 128 tables / radius 1` landed at about `705` candidates with `0.017222` recall, and `26 bits / 256 tables / radius 1` landed at about `585` candidates with `0.015858` recall.

Layer 0 remains much easier than later layers. The same `20 bits / 256 tables / radius 2` setting reached `0.763238` recall in layer 0, but only `0.384905` across all sampled layers.

## Decision

Random binary addresses are a kill for the million-token version of this address function.

SVA remains a go as a decomposition: broad summon plus exact verify already preserves pretrained behavior in the socket tests. The address function now has a clear target: it must be learned or model-aware enough to preserve full-QK top keys while keeping empirical million-token candidates in the hundreds to low thousands.

## Next Risk To Test

Train a compact address code per layer/head against full-QK top-key labels, then rerun both tests:

- 8192-window full-QK address recall
- million-token empirical address-pressure simulation

The kill/go criterion should be:

- top-16 recall above `0.75` across sampled layers
- empirical million-token average candidates in the rough `128-1024` range, or a clearly useful staged path through a few thousand candidates
- exact QK remains the verifier over the final candidate set
