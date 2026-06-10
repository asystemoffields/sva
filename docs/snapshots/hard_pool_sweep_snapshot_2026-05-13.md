# Hard-Negative Pool Sweep Snapshot - 2026-05-13

## Run

- Commit: `ee799b2`
- Modal app: `sva-hard-pool-sweep-h100`
- Modal app id: `ap-gqeFzPDYQWtzfAbFQjkREL`
- Function call: `fc-01KRJ0RQV3VHH48E2AJT58M27N`
- Dashboard: https://modal.com/id/fc-01KRJ0RQV3VHH48E2AJT58M27N
- Full log: `results/modal_runs/sva-h100-hard-pool-sweep-20260513-211408.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage: learned rank-64 scorer, `16x256` fine PQ
- Coarse stage: supervised rank-64 coarse scorer, `4x64` coarse PQ
- Hard-negative pass: `80` steps, `64` negatives, margin `1.0`
- Swept hard-negative mining pools: `512,768,1024,2048`

## Baselines

| Method | Shortlist | Top-16 recall |
| --- | ---: | ---: |
| exact learned ranker | 512 | 0.839332 |
| full fine PQ | 512 | 0.802021 |

## Weighted Hard-Negative Matrix

| Pool | Boost | Shortlist 512 | Shortlist 768 | Shortlist 1024 | Shortlist 2048 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4 | 0.827179 | 0.831303 | 0.834108 | 0.820685 |
| 512 | 16 | 0.824095 | 0.828063 | 0.831861 | 0.820545 |
| 768 | 4 | 0.826575 | 0.830931 | 0.832372 | 0.820080 |
| 768 | 16 | 0.824792 | 0.828978 | 0.831582 | 0.819723 |
| 1024 | 4 | 0.826156 | 0.827954 | 0.832031 | 0.819429 |
| 1024 | 16 | 0.825505 | 0.829892 | 0.832233 | 0.820483 |
| 2048 | 4 | 0.826311 | 0.829009 | 0.830900 | 0.819351 |
| 2048 | 16 | 0.823568 | 0.828761 | 0.830683 | 0.819413 |

## Best Rows

| Method | Shortlist | Top-16 recall | Gap to exact ranker |
| --- | ---: | ---: | ---: |
| weighted hard supervised, pool 512, boost 4 | 512 | 0.827179 | 0.012153 |
| weighted hard supervised, pool 512, boost 4 | 768 | 0.831303 | 0.008029 |
| weighted hard supervised, pool 512, boost 4 | 1024 | 0.834108 | 0.005224 |
| weighted hard supervised, pool 512, boost 4 | 2048 | 0.820685 | 0.018647 |

## Interpretation

The smaller hard-negative mining pool won every aggregate shortlist in this sweep. The best row, pool `512` with boost `4`, reached `0.834108` top-16 recall at shortlist `1024`, within `0.005224` of the exact learned ranker and `0.032087` above full fine-PQ scoring.

That suggests the current objective works best when the coarse scorer learns to protect positives against the most immediate shortlist competitors. Expanding the mining pool adds harder global structure, but it does not improve this local survival problem in the present setup.

The recurring oddity is the `2048` shortlist dip. It appears across pool sizes, so it is probably a handoff or ranking-shape issue rather than a single mining-pool artifact. The next mainline test should isolate the handoff: compare coarse-only survival, fine-PQ rescoring survival, and exact low-rank rescoring survival for the same hard-negative candidate sets.
