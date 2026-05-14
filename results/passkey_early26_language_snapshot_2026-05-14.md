# Passkey Early26 Language Snapshot - 2026-05-14

This snapshot records a direct 32K passkey-language pressure check on the current layer boundary.

## Run

- Runner: `modal_h100_passkey_early26_language.py`
- Harness: `experiments/sva_passkey_language_benchmark.py`
- Modal call: `fc-01KRKTM4ACPPQKN0VCMKAHJJDD`
- Local log: `results/modal_runs/sva-h100-passkey-early26-language-20260514-1405.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `32768`
- Placement: passkey at start, query at end
- Artifact: strong attention-weighted long-context profile
- Policy: scan summon, shortlist `8192`, verifier budget `2048`

## Results

| socket | layers | answer NLL delta | KL to full | top-1 agreement | logit cosine | prefill slowdown | decode slowdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| early26 | `0-25` | 0.146059 | 1.531219 | 0.285714 | 0.745349 | 53.41x | 1.19x |
| all30 | all | 0.166924 | 1.562648 | 0.428571 | 0.743596 | 90.43x | 2.08x |
| late4 | `26-29` | 0.023928 | 0.005527 | 1.000000 | 0.999898 | 20.42x | 0.55x |

All SVA rows used `16x` fewer decode exact/value reads than full attention in the SVA-socketed layers.

## Interpretation

This confirms the late-boundary diagnosis. Replacing `0-25` alone produces almost the same answer-logit distribution drift as replacing all layers. The current drift is therefore not mainly caused by the late readout layers; it is introduced by early/mid representation construction.

`late4` remains the clean production candidate: it preserves the answer distribution and, because it sockets only four layers, can already reduce decode wall-clock in this scan-based harness. Prefill is still much slower because scan summon is expensive, so the systems target remains indexed/cached summon for the late4 path.

## Next Step

Do not spend more cycles on `1024`-token early-layer distillation. If early-layer replacement remains a research branch, train it directly in this 32K passkey/logit-preservation regime. For production progress, prioritize making `late4` faster: indexed summon, cached catalogs, shared candidate sets, or a local-full plus remote-SVA split.
