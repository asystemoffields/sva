# Passkey Prefill-Drift Profile Snapshot - 2026-05-14

## Question

The final-query key-survival diagnostic did not explain the full language gap between profiles at `32768`. This benchmark measures whether the drift is already present at the final prompt position before answer-token decoding begins.

## Run

- Modal app: `ap-hURmwSJvEPcLRvdXTW3qQu`
- Log: `results/modal_runs/sva-h100-passkey-prefill-drift-profiles-20260514-1231.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Contexts: `16384,32768`
- Placement: passkey at start, query at end
- Profiles: original, plain refreshed, attention-weighted boost2, strong attention-weighted effective boost16
- Policy: scan summon, shortlist `8192`, verifier budget `2048`

## Result

| Profile | Context | First-token NLL delta | Prefill KL | Logit cosine | Top-1 agreement | Avg verified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 16384 | -0.446989 | 0.648889 | 0.879943 | 0.000000 | 1920.062500 |
| plain refresh | 16384 | 0.113845 | 0.242081 | 0.971266 | 0.000000 | 1920.062500 |
| attention boost2 | 16384 | 0.275620 | 0.219007 | 0.967401 | 0.000000 | 1920.062500 |
| strong attention | 16384 | -0.013569 | 0.242276 | 0.982664 | 0.000000 | 1920.062500 |
| original | 32768 | 1.037624 | 2.211099 | -0.593741 | 0.000000 | 1984.031250 |
| plain refresh | 32768 | 1.684878 | 2.381831 | -0.788812 | 0.000000 | 1984.031250 |
| attention boost2 | 32768 | 2.515614 | 2.228631 | -0.461123 | 0.000000 | 1984.031250 |
| strong attention | 32768 | 0.992831 | 2.148224 | 0.970458 | 0.000000 | 1984.031250 |

## Interpretation

At `32768`, prefill drift is already large before answer decoding starts. Every SVA profile increases first-token NLL by about `1.0+`, and KL to full attention is above `2.1`. This explains why final-query key survival alone was not enough to predict passkey answer quality.

Strong attention-weighted refresh is the best prefill profile at `32768` by first-token NLL and logit cosine, but it still trails full attention substantially. The later full-answer benchmark also shows that later decode tokens can change the profile ordering, so the fix needs to address both accumulated prompt drift and decode-step evidence retrieval.

The result points back to layer selectivity/progressive socketing. Codebook refresh can improve local summon quality, but all-layer replacement accumulates representation drift over long prompts.

## Next Step

Add layer-selective runtime patching and run a 32K passkey prefill-drift sweep over layer groups. The immediate hypothesis is that a small set of fragile layers should stay full-attention while SVA handles tolerant layers.
