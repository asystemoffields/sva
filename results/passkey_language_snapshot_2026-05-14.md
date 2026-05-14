# SVA Passkey Language Snapshot

This snapshot records the first language-level passkey stress test for the production SVA adapter.

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Hardware: Modal H100
- Adaptive inverted app: `ap-7Zu1lSCUZCztU03ngUVfWz`
- Fixed scan app: `ap-qWlT1dcDXozluDjffxgyk1`
- Fixed scan scale-out app: `ap-KE9alEKf0mJFtTqrgO54zN`
- Prompt: passkey at the beginning, filler to the target context, question at the end
- Key: `731942`
- Answer tokens: `7`

The harness scores the correct answer tokens by cached decode after the prompt. Full attention is the teacher at every tested length.

## Full-Attention Baseline

The baseline model itself weakens on this exact-string task as context grows:

| context | full answer NLL | full answer PPL | peak memory GB |
| ---: | ---: | ---: | ---: |
| `4096` | `0.757377` | `2.132675` | `0.749661` |
| `8192` | `2.993736` | `19.960115` | `1.216519` |
| `16384` | `3.843979` | `46.710974` | `2.151212` |
| `32768` | `6.363328` | `580.173954` | `4.020597` |

This makes the benchmark mainly a full-vs-SVA preservation test for this tiny model, especially past 8k.

## Adaptive Inverted Decode

The adaptive inverted path used far fewer exact reads but hurt exact-string preservation.

| context | NLL delta vs full | answer KL | logit cosine | decode verified | read reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `4096` | `3.776614` | `3.633467` | `0.801508` | `32.31` | `126.76x` |
| `8192` | `0.523940` | `0.149961` | `0.967730` | `113.89` | `71.93x` |
| `16384` | `2.152642` | `1.534646` | `0.968482` | `128.00` | `128.00x` |
| `32768` | `0.981283` | `2.599723` | `0.229150` | `168.30` | `194.70x` |

This is a useful negative for aggressive adaptive budgets: exact lookup tasks need more reliable verification than average next-token agreement suggests.

## Fixed Scan Decode

Holding decode to the fixed `2048/512` scan path recovered much of the passkey behavior.

| context | NLL delta vs full | answer KL | logit cosine | decode verified | read reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `4096` | `0.076570` | `0.014368` | `0.993062` | `512.00` | `8.00x` |
| `8192` | `0.163233` | `0.060347` | `0.971666` | `512.00` | `16.00x` |
| `16384` | `1.583940` | `0.950786` | `0.970610` | `512.00` | `32.00x` |
| `32768` | `0.629711` | `2.709807` | `0.467468` | `512.00` | `64.00x` |

The fixed path is the better production default for exact-string safety. At `4k` and `8k`, it stays close to full attention while keeping sparse value reads. At `16k` and `32k`, fixed `2048/512` begins to lose passkey preservation, which matches the long-context recall pressure tests.

## Fixed Scan Scale-Out

Increasing the shortlist and verifier budget recovered the longer-context passkey rows.

| context | shortlist | budget | NLL delta vs full | answer KL | logit cosine | read reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `16384` | `8192` | `2048` | `-0.016004` | `0.209053` | `0.982420` | `8.00x` |
| `32768` | `8192` | `2048` | `0.116894` | `1.902723` | `0.184094` | `16.00x` |

This turns the 16k/32k passkey result from a hard negative into a budget-policy result. Exact-string preservation can be recovered with larger sparse verification, at least through 32k in this benchmark.

## Interpretation

The previous held-out document benchmarks measured broad next-token agreement; this passkey test probes rare exact-token survival. The result says:

1. Adaptive inverted decode is too aggressive for exact retrieval in its current form.
2. Fixed scan `2048/512` is much safer through 8k.
3. Past 8k, a fixed `2048` shortlist is too small for rare exact-token tasks.
4. Larger sparse budgets recover answer NLL through 32k, but the current prefill path is far too slow.

The next decisive test is a serving-shaped version of this benchmark: preserve the `8192/2048` quality while reducing prefill and decode cost. That points to block-first SVA, shared candidate sets, or a catalog trained for exact-token survival at lower shortlist.
