# Supervised Coarse PQ Snapshot - 2026-05-13

## Run

- Commit: `8368c2d`
- Modal app: `sva-supervised-coarse-pq-h100`
- Function call: `fc-01KRHYA1M695XHWFFRH2HZZSPC`
- Full log: `results/modal_runs/sva-h100-supervised-coarse-pq-20260513-202944.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage: learned rank-64 scorer, `16x256` fine PQ
- Coarse supervision target: top-512 full fine-PQ winners

## Aggregate Results

| Method | Coarse rank | Coarse PQ | Shortlist | Top-16 recall |
| --- | ---: | --- | ---: | ---: |
| exact ranker | 0 | none | 512 | 0.841859 |
| full fine PQ | 0 | none | 512 | 0.805137 |
| unsupervised coarse-to-fine | 64 | 8x16, 32 bits/key | 4096 | 0.803556 |
| unsupervised coarse-to-fine | 64 | 4x64, 24 bits/key | 4096 | 0.803509 |
| unsupervised coarse-to-fine | 64 | 4x64, 24 bits/key | 2048 | 0.794116 |
| unsupervised coarse-to-fine | 64 | 4x64, 24 bits/key | 1024 | 0.768586 |
| supervised coarse-to-fine | 64 | 4x64, 24 bits/key | 4096 | 0.758293 |
| supervised coarse-to-fine | 64 | 4x64, 24 bits/key | 2048 | 0.665752 |
| supervised coarse-to-fine | 64 | 4x64, 24 bits/key | 1024 | 0.551107 |

## Interpretation

The naive supervised coarse stage is a regression. Training a separate coarse ranker to imitate the top-512 fine-PQ candidate set made the shortlist much worse than unsupervised PQ over the same fine-ranker space.

The likely issue is target width. Top-512 supervision asks the coarse ranker to spread probability over a broad set, while the serving need is sharper: keep the few keys that matter for the final top-16 attention recall. The next test uses the same harness with attention top-16 labels for the coarse stage.
