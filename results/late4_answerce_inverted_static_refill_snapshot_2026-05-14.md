# Late4 Answer-KL+CE Static Inverted Refill Snapshot - 2026-05-14

## Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Adapter: `results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1`
- SVA artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1`
- Socket: late4, replacing layers `26-29`
- Budget: `summon_topk=512`, `verify_topk=128`
- Summon mode: `inverted_static`
- Change under test: duplicate refill before exact verification
- Context: `32768`
- Cases: 24 held-out passkey cases, matching the broad/static panels
- Modal app: `ap-xO9WcWOY6MddADsvytRP6v`
- Function calls:
  - `8` cells/subspace: `fc-01KRMK4TQF97EBS4V5MEEDV0VM`
  - `16` cells/subspace: `fc-01KRMK4TW6PS7YD4QQAP8MC0NK`
- Full log: `results/modal_runs/sva-h100-late4-answerce-inverted-static-refill-20260514-211346.full.log`
- Runner: `modal_h100_late4_answerce_inverted_panel.py --cells 8,16 --summon-mode inverted_static`

The implementation now takes a larger low-rank refill pool, drops duplicate token ids before exact QK verification, and emits a fixed-width unique candidate tensor for the verifier.

## Result

| Cells/Subspace | Variant | Cases | Answer NLL Delta | Answer KL To Full | Top-1 Agreement | Logit Cosine | Prefill Slowdown | Decode Slowdown | Decode Avg Summoned | Decode Avg Verified | Exact Read Reduction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | Unadapted late4 SVA | 24 | `-0.030152` | `0.062601` | `0.869048` | `0.995300` | `27.236153` | `2.067015` | `1483.133` | `126.605` | `258.824` |
| 8 | Answer-KL+CE adapter | 24 | `-0.427092` | `0.038739` | `0.910714` | `0.992043` | `26.986163` | `2.006075` | `1481.725` | `126.589` | `258.857` |
| 16 | Unadapted late4 SVA | 24 | `-0.014295` | `0.073703` | `0.821429` | `0.998212` | `26.383055` | `2.381932` | `3153.892` | `127.545` | `256.914` |
| 16 | Answer-KL+CE adapter | 24 | `-0.432021` | `0.032605` | `0.922619` | `0.999225` | `26.333829` | `2.319209` | `3146.624` | `127.543` | `256.918` |

For comparison, the previous static inverted panel verified fewer tokens after duplicate masking:

| Cells/Subspace | Adapter KL | Adapter Top-1 | Adapter Cosine | Adapter NLL Delta | Decode Slowdown | Decode Avg Summoned | Decode Avg Verified |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8, old static | `0.040815` | `0.916667` | `0.991645` | `-0.421458` | `2.276378` | `1481.548` | `107.866` |
| 8, refill static | `0.038739` | `0.910714` | `0.992043` | `-0.427092` | `2.006075` | `1481.725` | `126.589` |
| 16, old static | `0.034456` | `0.910714` | `0.999215` | `-0.435238` | `1.872479` | `3145.726` | `97.370` |
| 16, refill static | `0.032605` | `0.922619` | `0.999225` | `-0.432021` | `2.319209` | `3146.624` | `127.543` |

## Readout

Duplicate refill does exactly what it was meant to do. It restores the verifier to roughly the intended `128` unique exact-read tokens per decoded answer token. That makes the cost accounting cleaner: the adapted rows now show about `257-259x` exact-read reduction in the SVA layers, rather than silently verifying only `~97-108` unique tokens.

The best current production point is now `8` cells/subspace with refill. It improves the old `8`-cell static row on KL, cosine, NLL delta, decode verified count, and decode slowdown, while losing a small amount of top-1 agreement. The `16`-cell refill row improves KL and top-1 but loses wall-clock versus old static, so it is a quality option rather than the speed target.

The remaining wall-clock bottleneck is still candidate construction and low-rank refill/top-k, not exact verification. The next implementation should split decode timing into catalog lookup, refill/top-k, exact score, value gather, and value aggregation, then optimize the largest slice.

## Next Implementation

- Add component timing for static inverted decode.
- Try a smaller refill factor or adaptive refill: refill only when duplicate pressure is high.
- Keep `8` cells/subspace as the main static-inverted speed target unless a timing split shows an easy `16`-cell win.
