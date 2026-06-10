# Passkey Late-Boundary Language Snapshot - 2026-05-14

## Question

The `late10` socket matched full attention at 32K. This sweep asks how much of that late block is actually needed, and whether a smaller late-layer socket preserves full-attention answer distribution more cleanly.

## Run

- Modal app: `ap-QMlq2py3bcnb0e0NRRMTwt`
- Log: `results/modal_runs/sva-h100-passkey-late-boundary-language-20260514-1310.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Contexts: `16384,32768`
- Placement: passkey at start, query at end
- Profile: strong attention-weighted long-context artifact
- Policy: scan summon, shortlist `8192`, verifier budget `2048`

## Result

| Layer group | Socket layers | Context | Answer NLL delta | Answer KL | Logit cosine | Top-1 agreement | Prefill ms | Decode slowdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| late4 | `26-29` | 16384 | -0.023765 | 0.002857 | 0.999770 | 0.857143 | 6635.585 | 0.39x |
| late4 | `26-29` | 32768 | 0.023928 | 0.005527 | 0.999898 | 1.000000 | 13622.432 | 0.59x |
| late6 | `24-29` | 16384 | -0.017408 | 0.005114 | 0.999836 | 0.857143 | 9883.963 | 0.65x |
| late6 | `24-29` | 32768 | 0.035770 | 0.007220 | 0.999366 | 1.000000 | 20533.179 | 0.73x |
| late8 | `22-29` | 16384 | -0.010888 | 0.005089 | 0.999897 | 0.857143 | 13198.451 | 0.75x |
| late8 | `22-29` | 32768 | -0.011626 | 0.016420 | 0.999564 | 0.857143 | 27664.447 | 0.86x |
| late10 | `20-29` | 16384 | 0.069254 | 0.010561 | 0.999667 | 1.000000 | 16634.529 | 0.81x |
| late10 | `20-29` | 32768 | -0.015913 | 0.028511 | 0.999308 | 0.714286 | 34978.647 | 1.01x |
| late12 | `18-29` | 16384 | 0.045964 | 0.012211 | 0.999617 | 0.857143 | 19994.004 | 0.94x |
| late12 | `18-29` | 32768 | -0.069515 | 0.067570 | 0.998795 | 0.857143 | 42110.454 | 1.24x |
| late15 | `15-29` | 16384 | 0.183972 | 0.024227 | 0.999308 | 1.000000 | 25025.272 | 1.09x |
| late15 | `15-29` | 32768 | -0.187885 | 0.095193 | 0.998396 | 0.857143 | 52757.804 | 1.47x |

## Interpretation

`late4` is the cleanest drop-in candidate in this sweep. It is the smallest tested socket, has the lowest answer KL at both contexts, reaches perfect top-1 agreement at 32K, and keeps answer logit cosine closest to full attention. Larger late sockets sometimes improve the gold passkey NLL, but they drift farther from the full-attention distribution.

This changes the near-term socketing target from `late10` to `late4`. It also narrows the production story: keep full attention for layers `0-25`, replace only layers `26-29` with SVA, and focus speed work on making those late-layer summons indexed rather than scan-based.

## Next Step

Run a `late4` robustness check at `32768` across start/middle/end placements and multiple passkeys. If that holds, the next quality test should leave passkeys and move to a broader language or retrieval benchmark with the same late4 socket.
