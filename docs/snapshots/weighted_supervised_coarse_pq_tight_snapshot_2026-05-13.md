# Weighted Supervised Coarse PQ Tight-Shortlist Snapshot - 2026-05-13

## Run

- Commit: `c06836a`
- Modal app: `sva-weighted-supervised-coarse-pq-tight-h100`
- Modal app id: `ap-6WsSOJtKDUA7esXINqokvT`
- Function call: `fc-01KRHZ8DC76X6KM12TSQS235A7`
- Dashboard: https://modal.com/id/fc-01KRHZ8DC76X6KM12TSQS235A7
- Full log: `results/modal_runs/sva-h100-weighted-supervised-coarse-pq-tight-20260513-204745.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Fine stage: learned rank-64 scorer, `16x256` fine PQ
- Coarse stage: supervised rank-64 coarse scorer, `4x64` coarse PQ
- Shortlists: `512,768,1024`
- Budget: `512`

## Aggregate Results

| Method | Shortlist | Top-16 recall |
| --- | ---: | ---: |
| exact ranker | 512 | 0.839332 |
| full fine PQ | 512 | 0.802021 |
| weighted supervised, boost 4 | 1024 | 0.776445 |
| weighted supervised, boost 16 | 1024 | 0.775375 |
| supervised coarse-to-fine | 1024 | 0.769097 |
| unsupervised coarse-to-fine | 1024 | 0.764943 |
| weighted supervised, boost 4 | 768 | 0.757239 |
| weighted supervised, boost 16 | 768 | 0.754449 |
| supervised coarse-to-fine | 768 | 0.747179 |
| unsupervised coarse-to-fine | 768 | 0.741939 |
| weighted supervised, boost 4 | 512 | 0.713759 |
| weighted supervised, boost 16 | 512 | 0.711201 |
| supervised coarse-to-fine | 512 | 0.705063 |
| unsupervised coarse-to-fine | 512 | 0.699250 |

## Interpretation

The weighted supervised coarse stage keeps improving candidate survival as the shortlist tightens, but the curve is steep below `1024`.

At `1024`, boost `4` holds `0.776445`, which is the best low-shortlist result so far. At `768`, it keeps `0.757239`. At `512`, it falls to `0.713759`, still above the unweighted and unsupervised rows but far below full fine PQ's `0.802021`.

The practical band for this version is `1024-2048`: `1024` is the best aggressive point, while `2048` from the previous run reached `0.799882`, nearly matching full fine PQ. The next invention target is a shortlist-aware coarse objective: train the coarse stage to maximize top-key survival specifically at `512-1024`, rather than fitting weighted k-means after the ranker is trained.
