# Hard-Negative Supervised Coarse PQ Snapshot - 2026-05-13

## Run

- Commit: `c2e981c`
- Modal app: `sva-hard-supervised-coarse-pq-h100`
- Modal app id: `ap-thbANO6dvkNPa33rhGH5aR`
- Function call: `fc-01KRHZXMBNR0WPDEE656CMN9YP`
- Dashboard: https://modal.com/id/fc-01KRHZXMBNR0WPDEE656CMN9YP
- Full log: `results/modal_runs/sva-h100-hard-supervised-coarse-pq-20260513-205919.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage: learned rank-64 scorer, `16x256` fine PQ
- Coarse stage: supervised rank-64 coarse scorer, `4x64` coarse PQ
- Hard-negative pass: `80` steps, pool `1024`, `64` negatives, margin `1.0`

## Aggregate Results

| Method | Shortlist | Top-16 recall |
| --- | ---: | ---: |
| exact ranker | 512 | 0.839332 |
| full fine PQ | 512 | 0.802021 |
| weighted hard supervised, boost 16 | 512 | 0.825505 |
| weighted hard supervised, boost 4 | 512 | 0.826187 |
| hard supervised | 512 | 0.813833 |
| weighted hard supervised, boost 16 | 768 | 0.829861 |
| weighted hard supervised, boost 4 | 768 | 0.827970 |
| hard supervised | 768 | 0.822049 |
| weighted hard supervised, boost 16 | 1024 | 0.832217 |
| weighted hard supervised, boost 4 | 1024 | 0.832062 |
| hard supervised | 1024 | 0.827319 |
| weighted hard supervised, boost 16 | 2048 | 0.820468 |
| weighted hard supervised, boost 4 | 2048 | 0.819429 |
| hard supervised | 2048 | 0.819305 |

## Comparison

The prior best tight-shortlist rows were:

| Method | Shortlist | Top-16 recall |
| --- | ---: | ---: |
| weighted supervised, boost 4 | 512 | 0.713759 |
| weighted supervised, boost 4 | 768 | 0.757239 |
| weighted supervised, boost 4 | 1024 | 0.776445 |
| weighted supervised, boost 16 | 2048 | 0.799882 |

Hard-negative training is the first result that makes the few-hundreds band look plausible. It raised shortlist `512` from about `0.714` to `0.826`, and shortlist `1024` from about `0.776` to `0.832`, close to the exact learned-ranker ceiling of `0.839332`.

## Interpretation

The main failure was objective mismatch. The earlier supervised ranker and weighted codebooks improved the coarse catalog, but they still trained against generic positive labels. Mining the coarse ranker's own high-scoring wrong keys creates the right local pressure: top attention keys must survive against the specific competitors that would otherwise occupy the shortlist.

The unusual shape is that `2048` is lower than `1024` in the hard-negative rows. That suggests the hard-negative scorer is being optimized for survival inside the mined `1024` band and that the fine-PQ handoff can still reorder candidates in ways that lose some positives. The next mainline test should sweep hard-negative pool sizes and train pools, especially `512,768,1024,2048`, to see whether the survival peak can be moved or flattened.
