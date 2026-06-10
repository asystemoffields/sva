# Learned Ranker Held-Out Text Snapshot

Date: 2026-05-13

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`

Run artifact: `results/modal_runs/sva-h100-learned-ranker-generalize-20260513-182036.modal.log`

## Setup

This run trained the learned compressed Q/K ranker on one 8192-token stream and evaluated it on a separate reversed-order 8192-token stream.

- `model_max_position_embeddings=8192`
- `train_seq_len=8192`
- `eval_seq_len=8192`
- `eval_text_mode=reverse`
- layers: `0,1,5,10,18,24,29`
- rank dimensions: `16,32,64`
- verifier budgets: `64,128,256,512,1024`
- train query positions: 128
- eval query positions: 64
- full-QK target: top 16 keys per sampled query/head
- total key targets per aggregate row: 64,512

The ranker parameters were trained on train-text Q/K top-key labels and evaluated against full-QK top-key labels on the reversed eval text. The exact verifier budget is the number of keys the compressed score would hand to exact QK.

## Aggregate Results

| phase | rank dim | budget | avg candidates | top-16 recall |
| --- | ---: | ---: | ---: | ---: |
| random | 16 | 64 | 64.0 | 0.069 |
| random | 16 | 128 | 128.0 | 0.109 |
| random | 16 | 256 | 254.0 | 0.170 |
| random | 16 | 512 | 500.0 | 0.265 |
| random | 16 | 1024 | 968.0 | 0.401 |
| random | 32 | 64 | 64.0 | 0.096 |
| random | 32 | 128 | 128.0 | 0.139 |
| random | 32 | 256 | 254.0 | 0.204 |
| random | 32 | 512 | 500.0 | 0.294 |
| random | 32 | 1024 | 968.0 | 0.426 |
| random | 64 | 64 | 64.0 | 0.084 |
| random | 64 | 128 | 128.0 | 0.125 |
| random | 64 | 256 | 254.0 | 0.186 |
| random | 64 | 512 | 500.0 | 0.279 |
| random | 64 | 1024 | 968.0 | 0.412 |
| trained | 16 | 64 | 64.0 | 0.415 |
| trained | 16 | 128 | 128.0 | 0.522 |
| trained | 16 | 256 | 254.0 | 0.632 |
| trained | 16 | 512 | 500.0 | 0.738 |
| trained | 16 | 1024 | 968.0 | 0.838 |
| trained | 32 | 64 | 64.0 | 0.479 |
| trained | 32 | 128 | 128.0 | 0.589 |
| trained | 32 | 256 | 254.0 | 0.697 |
| trained | 32 | 512 | 500.0 | 0.795 |
| trained | 32 | 1024 | 968.0 | 0.880 |
| trained | 64 | 64 | 64.0 | 0.543 |
| trained | 64 | 128 | 128.0 | 0.649 |
| trained | 64 | 256 | 254.0 | 0.750 |
| trained | 64 | 512 | 500.0 | 0.835 |
| trained | 64 | 1024 | 968.0 | 0.906 |

## Interpretation

The held-out text result preserves almost all of the same signal as the held-out query-position run. Trained rank-64 recall at budget 256 was `0.749752`, compared with `0.759781` in the same-text held-out-query run. At budget 512 it was `0.835488`, compared with `0.848338`.

This strongly suggests that a model-aware compressed Q/K score can describe SmolLM2's full-attention neighborhoods in a reusable way. It also gives a concrete operating point: rank-64, 512 candidates, full-QK top-16 recall around `0.84` on this test.

Layer 24 remains the hardest sampled layer. Trained rank-64 at budget 512 reached `0.707682` on layer 24, while layer 18 reached `0.783312` and layer 29 reached `0.839627`.

## Decision

Learned compressed ranking remains a go after held-out text evaluation.

The next invention target is no longer "can a compact score find the right keys?" It is "can we serve that compact score sublinearly at million-token scale?"

## Next Risk To Test

Convert the learned rank-64 score into an addressable lookup and measure candidate recall:

- product-quantized compressed keys
- learned binary code over the rank-64 space
- multi-probe ANN over compressed keys

The next kill/go threshold:

- held-out text
- top-16 recall at or above `0.75`
- verifier budget at or below `512`
- lookup cost that scales sublinearly with context
