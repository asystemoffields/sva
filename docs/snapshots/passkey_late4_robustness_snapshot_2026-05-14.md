# Passkey Late4 Robustness Snapshot - 2026-05-14

## Question

The late-boundary sweep made `late4` the cleanest socket candidate. This run tests whether that result holds across multiple passkeys and passkey placements at 32K.

## Run

- Modal app: `ap-5t2pP7pRXeeqzRoj3DuCZ1`
- Log: `results/modal_runs/sva-h100-passkey-late4-robustness-20260514-1320.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `32768`
- Placements: `start,middle,end`
- Passkeys: `731942,184207,905613`
- Profile: strong attention-weighted long-context artifact
- Socket: layers `26,27,28,29`
- Policy: scan summon, shortlist `8192`, verifier budget `2048`

## Result

| Key | Placement | Answer NLL delta | Answer KL | Top-1 agreement | Logit cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| 731942 | start | 0.009820 | 0.005536 | 1.000000 | 0.999887 |
| 731942 | middle | 0.035450 | 0.005278 | 0.857143 | 0.999933 |
| 731942 | end | 0.042883 | 0.006311 | 1.000000 | 0.999736 |
| 184207 | start | -0.044725 | 0.007405 | 1.000000 | 0.999929 |
| 184207 | middle | 0.003261 | 0.004988 | 1.000000 | 0.999961 |
| 184207 | end | 0.054583 | 0.004354 | 1.000000 | 0.999916 |
| 905613 | start | -0.023678 | 0.008842 | 0.857143 | 0.999919 |
| 905613 | middle | 0.019558 | 0.003072 | 1.000000 | 0.999762 |
| 905613 | end | 0.000726 | 0.004135 | 1.000000 | 0.999954 |

Aggregate over 9 cases:

| Metric | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| Answer NLL delta | 0.010875 | -0.044725 | 0.054583 |
| Answer KL | 0.005547 | 0.003072 | 0.008842 |
| Top-1 agreement | 0.968254 | 0.857143 | 1.000000 |
| Logit cosine | 0.999889 | 0.999736 | 0.999961 |
| Prefill slowdown | 80.43x | 11.32x | 91.70x |
| Decode slowdown | 1.90x | 0.36x | 2.23x |

## Interpretation

`late4` is robust on this passkey panel. The answer distribution stays close to full attention across key values and placements: KL remains below `0.009`, logit cosine remains above `0.9997`, and answer NLL deltas stay inside about `+/-0.055`.

The design has become simpler:

1. Full attention builds the early and middle hidden-state stream.
2. SVA replaces only the last four attention layers.
3. The verifier remains exact attention over summoned tokens.
4. The remaining production problem is making summon indexed instead of scan-based.

## Next Step

Move beyond passkeys with the same `late4` socket. The next quality target should be a broader language or retrieval benchmark at 32K, while the systems target remains indexed summon for layers `26-29`.
