# Passkey Profile Router Snapshot

Date: 2026-05-14

## Question

The refreshed long-context profile improves aggregate top-key recall at `16384` and `32768`. This benchmark tests whether that improvement converts into language-level exact-string preservation on the passkey task.

## Run

```text
app: ap-fhxdJwE6kvwNW0nt4vQRVx
function: fc-01KRKGE9RQ07RD4EECNCPE04MX
log: results/modal_runs/sva-h100-passkey-profile-router-20260514-1110.full.log
exit: 0
```

Settings:

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Base artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Long artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1`
- Router: base artifact below `16384`, long artifact at `16384+`
- Prompt: passkey at the beginning, question at the end
- Key: `731942`
- SVA policy: scan mode, shortlist `8192`, verifier budget `2048`

## Result

| Context | SVA profile | Full NLL | SVA NLL | NLL delta | Answer KL | Logit cosine | Decode read reduction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | base | 2.993736 | 3.064683 | 0.070947 | 0.011096 | 0.979466 | 4.00x |
| 16384 | long refresh | 3.843979 | 3.885993 | 0.042013 | 0.122472 | 0.995564 | 8.00x |
| 32768 | long refresh | 6.378009 | 6.516541 | 0.138533 | 1.209262 | 0.698508 | 16.00x |

For comparison, the earlier original-profile scale-out run with the same `8192/2048` scan policy reached:

| Context | Original-profile NLL delta | Routed long-profile NLL delta |
| ---: | ---: | ---: |
| 16384 | -0.016004 | 0.042013 |
| 32768 | 0.116894 | 0.138533 |

## Interpretation

This is a useful negative. The refreshed profile improves aggregate teacher top-16 recall, score cosine, and code entropy, but that does not directly improve this exact-string language benchmark at the large `8192/2048` sparse budget.

The likely lesson is that catalog balance can help long-context recall, but passkey preservation needs a more targeted objective. A globally better codebook can still move answer-sensitive keys, layers, or value mass in ways the aggregate recall proxy does not see.

The prefill path is also still far from production-shaped. At `32768`, routed SVA prefill took about `93.6s` in this PyTorch scan harness while full attention took about `0.18s`; decode was closer, with about `2.94x` slowdown and `16x` fewer value reads.

## Next Step

The next sharp test should be an evidence-aware refresh rather than another plain entropy refresh:

- compare original and refreshed profiles on passkey key survival by layer/head/query,
- weight calibration codebooks toward attention top-k and passkey/evidence neighborhoods,
- record code entropy and max-code-load as diagnostics,
- rerun the profile-router passkey test only after the evidence-weighted profile improves key survival.
