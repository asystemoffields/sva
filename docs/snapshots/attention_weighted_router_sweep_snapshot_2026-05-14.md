# Attention-Weighted Router Sweep Snapshot - 2026-05-14

## Question

The strong attention-weighted routed profile improved passkey NLL at `16384` but regressed at `32768`. This sweep tests whether gentler attention weighting keeps the `16384` gain while recovering the `32768` row.

## Run

- Modal app: `ap-bHv0IbVP12ZMzn4FNLZFLZ`
- Log: `results/modal_runs/sva-h100-attnweighted-router-sweep-20260514-1205.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Base artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Exported profiles: attention-weighted boost `1`, `2`, `4`
- Router: long profile at `16384+`
- Policy: scan summon, shortlist `8192`, verifier budget `2048`
- Placement: passkey at start, query at end
- Device/dtype: H100, `bfloat16`

## Result

| Profile | Effective boost | Context | NLL delta | Answer KL | Logit cosine | Top-1 agreement | Decode read reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention-weighted boost1 | 1 | 16384 | 0.022352 | 0.118648 | 0.996080 | 0.714286 | 8.00x |
| attention-weighted boost1 | 1 | 32768 | 0.334068 | 1.450383 | 0.744229 | 0.285714 | 16.00x |
| attention-weighted boost2 | 2 | 16384 | 0.021849 | 0.118947 | 0.994988 | 0.714286 | 8.00x |
| attention-weighted boost2 | 2 | 32768 | 0.480951 | 3.615740 | -0.104913 | 0.428571 | 16.00x |
| attention-weighted boost4 | 4 | 16384 | 0.108811 | 0.124450 | 0.993268 | 0.857143 | 8.00x |
| attention-weighted boost4 | 4 | 32768 | 0.593440 | 2.076446 | 0.083490 | 0.428571 | 16.00x |
| strong attention-weighted | 16 | 16384 | 0.024752 | 0.118649 | 0.997162 | 0.714286 | 8.00x |
| strong attention-weighted | 16 | 32768 | 0.152243 | 1.560773 | 0.744593 | 0.428571 | 16.00x |

Prior reference rows:

| Profile | Context | NLL delta |
| --- | ---: | ---: |
| plain refreshed routed profile | 16384 | 0.042013 |
| plain refreshed routed profile | 32768 | 0.138533 |
| original-profile scale-out | 16384 | -0.016004 |
| original-profile scale-out | 32768 | 0.116894 |

## Interpretation

Gentler attention weighting helps the `16384` row slightly. Boost `2` is the best attention-weighted result there, improving over the strong profile by about `0.0029` NLL and over the plain refreshed routed profile by about `0.0202`.

At `32768`, lowering the boost makes the result worse. The strong effective boost `16` remains the best attention-weighted profile at that context, but it still trails the plain refreshed profile and the original-profile scale-out row. This points away from a simple over-strong-weighting explanation.

The sharper hypothesis is that the aggregate teacher top-k recall proxy is now misaligned with this exact passkey deployment case. Attention-weighted refresh can preserve more generic top-attended evidence while harming the particular early passkey evidence needed at the final query.

## Next Step

Run a passkey-specific key-survival diagnostic:

- compare original, plain refreshed, boost2, and strong profiles
- record whether the exact passkey token positions are summoned and verified by layer/head at the answer query
- separate summon miss from verifier miss
- keep the same `8192/2048` budget so the diagnostic matches the language benchmark

If the passkey evidence is summoned but not verified, tune the verifier/rerank shape. If it is not summoned, the catalog objective needs evidence-position or query-conditioned training rather than stronger generic attention top-k weighting.
