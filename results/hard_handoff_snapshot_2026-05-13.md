# Hard-Negative Handoff Snapshot - 2026-05-13

## Run

- Commit: working tree after `ee799b2`
- Modal app: `sva-hard-handoff-h100`
- Modal app id: `ap-zCzoARcyAfS0SrOEVAbWg1`
- Function call: `fc-01KRJ181YK37MQF3N0EXNXRMGW`
- Dashboard: https://modal.com/id/fc-01KRJ181YK37MQF3N0EXNXRMGW
- Full log: `results/modal_runs/sva-h100-hard-handoff-20260513-212228.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage baseline: learned rank-64 scorer, `16x256` fine PQ
- Coarse stage: hard-negative supervised rank-64 scorer, weighted `4x64` coarse PQ
- Hard-negative pass: pool `512`, `80` steps, `64` negatives, margin `1.0`
- Diagnostic setting: compare the same coarse shortlist under coarse-only survival, exact rank-64 rescoring, and fine-PQ rescoring.

## Baselines

| Method | Budget | Top-16 recall |
| --- | ---: | ---: |
| full fine PQ | 512 | 0.802021 |
| exact learned ranker | 512 | 0.839332 |

## Handoff Diagnostic

| Coarse shortlist | Coarse-only survival | Exact rank-64 rescore, budget 512 | Fine-PQ rescore, budget 512 | Exact minus fine-PQ |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.827164 | 0.827164 | 0.827164 | 0.000000 |
| 768 | 0.875496 | 0.837798 | 0.831256 | 0.006542 |
| 1024 | 0.906095 | 0.847873 | 0.834077 | 0.013796 |
| 1536 | 0.940817 | 0.850555 | 0.828559 | 0.021996 |
| 2048 | 0.959279 | 0.847811 | 0.820669 | 0.027142 |
| 3072 | 0.979322 | 0.843362 | 0.810516 | 0.032846 |
| 4096 | 0.989568 | 0.841564 | 0.806176 | 0.035388 |

## Interpretation

The hard-negative coarse stage is summoning the right keys. Coarse-only survival climbs from `0.906095` at shortlist `1024` to `0.959279` at `2048` and `0.989568` at `4096`.

The `2048` dip comes from the handoff. Exact rank-64 rescoring stays strong, peaking at `0.850555` around shortlist `1536`, while fine-PQ rescoring falls as the shortlist widens. The gap between exact rank-64 and fine-PQ rescoring grows from `0.013796` at shortlist `1024` to `0.035388` at `4096`.

This changes the mainline target. The coarse catalog is good enough for the current toy serving setup; the next target is the verifier side. The most direct branch is coarse PQ summon, exact rank-64 rescore over roughly `1024-2048` candidates, then full attention verification over the final `512`. The next measurement should test whether that exact rank-64 middle stage is cheap enough at million-token scale.
