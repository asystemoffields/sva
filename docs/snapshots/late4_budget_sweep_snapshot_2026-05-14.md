# Late4 Budget Sweep Snapshot - 2026-05-14

This snapshot records a 32K passkey-language budget squeeze for the current `late4` production socket.

## Run

- Runner: `modal_h100_late4_budget_sweep.py`
- Harness: `experiments/sva_passkey_language_benchmark.py`
- Modal call: `fc-01KRKXQJ8H247MNEV4HEWN2M4T`
- Local log: `results/modal_runs/sva-h100-late4-budget-sweep-20260514-1459.full.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `32768`
- Placement: passkey at start, query at end
- Socket: layers `26-29`
- Artifact: strong attention-weighted long-context profile
- Summon mode: scan

## Results

| shortlist / budget | answer NLL delta | KL to full | top-1 agreement | logit cosine | decode exact-read reduction | prefill slowdown | decode slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8192 / 2048` | 0.023928 | 0.005527 | 1.000000 | 0.999898 | 16x | 11.68x | 0.34x |
| `4096 / 1024` | -0.014591 | 0.013036 | 1.000000 | 0.999346 | 32x | 12.70x | 0.47x |
| `2048 / 512` | 0.002096 | 0.022243 | 0.857143 | 0.998889 | 64x | 7.56x | 0.48x |
| `1024 / 256` | -0.021138 | 0.053690 | 0.857143 | 0.998632 | 128x | 6.56x | 0.52x |
| `512 / 128` | 0.017761 | 0.085656 | 0.857143 | 0.998725 | 256x | 4.59x | 0.41x |

## Interpretation

Late4 has more budget headroom than expected. `4096/1024` is still very close to full attention, and `2048/512` is plausible for a production-oriented target with `64x` fewer exact/value reads in the SVA layers. The `1024/256` and `512/128` rows preserve gold-answer NLL on this case but show enough distribution drift to be useful adaptation targets.

Prefill remains slower because this benchmark still uses scan summon. Smaller budgets reduce verifier work and memory, but the full-cache coarse scan remains the dominant prefill cost. Decode is already faster than full attention in this harness for all tested late4 budgets.

## Next Step

Use `512/128` or `1024/256` as the fine-tuning target. The training question is whether SVA-active logit distillation can recover the `8192/2048` distribution closeness while keeping the much smaller verifier budget.
