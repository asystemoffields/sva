# Passkey Layer-Selective Language Snapshot - 2026-05-14

## Question

The prefill probe showed that socketing only late layers nearly matches full attention at the final prompt position. This benchmark tests whether that transfers to full passkey answer scoring across all answer tokens.

## Run

- Modal app: `ap-LvnPih2wnsrsiuozkPUknE`
- Log: `results/modal_runs/sva-h100-passkey-layer-selective-language-20260514-1300.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Contexts: `16384,32768`
- Placement: passkey at start, query at end
- Profile: original artifact below `16384`, strong attention-weighted long-context artifact at `16384+`
- Policy: scan summon, shortlist `8192`, verifier budget `2048`

## Result

| Layer group | Context | Answer NLL delta | Answer KL | Logit cosine | Top-1 agreement | Prefill ms | Decode slowdown | Exact read reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 16384 | 0.024752 | 0.118649 | 0.997162 | 0.714286 | 46045.932 | 1.28x | 8.00x |
| all | 32768 | 0.152243 | 1.560773 | 0.744593 | 0.428571 | 95492.275 | 3.30x | 16.00x |
| sparse6 | 16384 | -0.197931 | 0.040245 | 0.997672 | 0.857143 | 9450.837 | 0.66x | 8.00x |
| sparse6 | 32768 | 0.157752 | 1.493731 | 0.542861 | 0.571429 | 19154.740 | 0.83x | 16.00x |
| late10 | 16384 | 0.069254 | 0.010561 | 0.999667 | 1.000000 | 15655.531 | 0.94x | 8.00x |
| late10 | 32768 | -0.001995 | 0.028690 | 0.999246 | 0.714286 | 31972.038 | 1.06x | 16.00x |

## Interpretation

The 32K result validates the prefill signal at answer level. `late10` brings answer NLL essentially to full attention, with low KL and high cosine. It also keeps the decode path close to full attention wall-clock in the current unfused PyTorch implementation, while reading `16x` fewer values in the SVA layers.

At 16K, the picture splits by metric. `sparse6` improves gold answer NLL, while `late10` has the lowest KL and perfect top-1 agreement with full attention. For a drop-in replacement, the distribution-preservation metric points to `late10`; for this exact passkey target, sparse and late sockets both deserve a boundary sweep.

The systems result is still prefill-limited by scan summon. The quality result is the main news: late-layer socketing avoids the large 32K drift seen in all-layer replacement.

## Next Step

Run a late-boundary sweep over `late4`, `late6`, `late8`, `late10`, `late12`, and `late15` at `16384` and `32768`. The goal is to find the smallest late-layer socket that preserves full-attention answer distribution at 32K, then carry that boundary into indexed-summon speed work.
