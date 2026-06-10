# Passkey Key-Survival Profile Snapshot - 2026-05-14

## Question

The attention-weighted router sweep showed that profile choice changes passkey language NLL at `32768`, while aggregate recall proxies did not predict the ordering. This diagnostic asks whether the final answer query is losing the early passkey evidence during summon or verifier selection.

## Run

- Modal app: `ap-3YWSxMLeKWJmEmysn8fuQI`
- Log: `results/modal_runs/sva-h100-passkey-key-survival-profiles-20260514-1222.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Contexts: `16384,32768`
- Placement: passkey at start, query at end
- Layers: all 30
- Policy: one final-query anchor, shortlist `8192`, verifier budget `2048`, no span expansion
- Profiles compared: original, plain refreshed, attention-weighted boost2, strong attention-weighted effective boost16

## Aggregate Result

| Profile | Context | Key summoned | Key verified | Needle summoned | Needle verified | Teacher top-16 key hit | Teacher best key rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 16384 | 0.707407 | 0.707407 | 0.818519 | 0.818519 | 0.005556 | 909.085185 |
| original | 32768 | 0.148148 | 0.148148 | 0.244444 | 0.244444 | 0.001852 | 11291.611255 |
| plain refresh | 16384 | 0.692593 | 0.692593 | 0.807407 | 0.807407 | 0.005556 | 909.085185 |
| plain refresh | 32768 | 0.137037 | 0.137037 | 0.222222 | 0.222222 | 0.001852 | 11291.611255 |
| attention boost2 | 16384 | 0.692593 | 0.692593 | 0.814815 | 0.814815 | 0.005556 | 909.085185 |
| attention boost2 | 32768 | 0.133333 | 0.133333 | 0.233333 | 0.233333 | 0.001852 | 11291.611255 |
| strong attention | 16384 | 0.696296 | 0.696296 | 0.814815 | 0.814815 | 0.005556 | 909.085185 |
| strong attention | 32768 | 0.140741 | 0.140741 | 0.225926 | 0.225926 | 0.001852 | 11291.611255 |

Teacher-mass-weighted key survival:

| Profile | 16384 | 32768 |
| --- | ---: | ---: |
| original | 0.769627 | 0.333384 |
| plain refresh | 0.756026 | 0.329946 |
| attention boost2 | 0.749888 | 0.329891 |
| strong attention | 0.756443 | 0.331316 |

## Interpretation

The final-query diagnostic separates summon and verifier loss: in this setup, key verified equals key summoned because the verifier budget is the same size as the summoned candidate set. The loss is therefore in the summon stage for the final query.

The profile differences are too small to explain the full language gap. Original has the best key survival at `32768`, and strong attention is close. The larger passkey NLL differences likely come from accumulated prefill drift: SVA changes hidden states while reading the prompt, so the final query may already be operating on a different representation before the last summon happens.

Teacher top-16 key hit is extremely low at `32768`; the passkey tokens are often not among the highest exact-QK keys at the final query. That means a simple final-query top-k evidence metric is also an incomplete target for exact-string retrieval. We need to measure representation drift through the whole prefill path.

## Next Step

Run a passkey prefill-drift diagnostic:

- compare full vs SVA logits at the final prompt position before answer decoding
- separate first-answer-token NLL from later decode-token NLL
- compare original, plain refreshed, boost2, and strong attention profiles at `16384/32768`
- record prefill stats and answer logit cosine/KL for each profile

If prefill drift dominates, the next fix is progressive/socket training or profile routing by fragile layers, not another codebook-only refresh.
