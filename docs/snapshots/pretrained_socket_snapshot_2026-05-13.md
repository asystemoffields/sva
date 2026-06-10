# Pretrained Socket Snapshot

Date: 2026-05-13

## Question

Can Summon-Verify Attention be socketed into a pretrained modern decoder and approximate the model's own full-attention behavior using the existing Q/K/V/O weights?

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Harness: `experiments/sva_pretrained_socket_test.py`
- Runner: `modal_h100_socket.py`
- Hardware: Modal H100
- Dtype: bfloat16
- Text count: 6
- Effective sequence length: 53 tokens
- SVA mechanism: each Llama attention layer keeps the pretrained projections, RoPE, value path, output projection, norms, MLPs, and logits. The attention score matrix is restricted to a mask produced by SVA candidate lookup, then exact QK attention runs over the summoned candidates.

## Result

Full-attention reference loss: `4.937500`

| tables | bits | probe_radius | budget | sva_loss | loss_delta | KL to full | top1 agreement | logit cosine | avg summoned |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 1 | 64 | 7.656250 | 2.718750 | 3.431300 | 0.201465 | 0.725322 | 7.932 |
| 16 | 8 | 1 | 64 | 5.812500 | 0.875000 | 1.227824 | 0.435897 | 0.834199 | 12.774 |
| 32 | 8 | 1 | 64 | 5.156250 | 0.218750 | 0.360491 | 0.659341 | 0.947996 | 17.562 |
| 16 | 10 | 1 | 64 | 8.437500 | 3.500000 | 4.206571 | 0.131868 | 0.695492 | 5.991 |
| 32 | 10 | 1 | 64 | 6.781250 | 1.843750 | 2.107814 | 0.293040 | 0.792790 | 9.125 |
| 32 | 10 | 2 | 64 | 5.031250 | 0.093750 | 0.188110 | 0.783883 | 0.974020 | 20.432 |
| 32 | 12 | 1 | 64 | 8.875000 | 3.937500 | 4.551042 | 0.142857 | 0.689338 | 5.320 |

Best setting in this sweep: `32 tables`, `10 bits`, `probe_radius=2`, `budget=64`.

## Interpretation

This is the strongest result so far for the pretrained-weight path. The best SVA socket run stays close to the full-attention model on the same prompts:

- loss delta: `+0.093750`
- KL to full logits: `0.188110`
- top-1 token agreement: `0.783883`
- logit cosine: `0.974020`
- average verified candidates: `20.432` out of `53` positions

The table shape matters sharply. With 10 or 12 address bits and only radius-1 probing, the lookup becomes too selective and misses useful pages. Increasing the probe radius to 2 fixes much of that for 10-bit addresses.

## Next Risk

Run the same socket test at longer contexts and on more text:

- context lengths: 128, 256, and 512 effective tokens
- tables: 32 and 64
- bits: 8 and 10
- probe radius: 1 and 2
- budgets: 32, 64, and 128

Add per-layer and per-head candidate recall against the full-attention top keys. That will show whether misses concentrate in specific layers or heads, and whether the candidate budget can shrink where attention is naturally local or redundant.

