# Coarse-to-Fine PQ Lookup Snapshot - 2026-05-13

## Run

- Commit: `8e63d1a`
- Modal app: `sva-coarse-to-fine-pq-h100`
- Function call: `fc-01KRHWFXKHVD1TS5FH4T5RH21F`
- Dashboard: https://modal.com/id/fc-01KRHWFXKHVD1TS5FH4T5RH21F
- Full log: `results/modal_runs/sva-h100-coarse-to-fine-pq-20260513-195924.modal.log`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Context: `8192` tokens, matching `max_position_embeddings=8192`
- Eval text: reversed held-out stream
- Layers: `0,1,5,10,18,24,29`
- Ranker: learned asymmetric rank-64 scorer trained for `160` steps per layer

## Aggregate Baselines

| Method | Budget | Fine PQ bits/key | Avg final candidates | Top-16 recall |
| --- | ---: | ---: | ---: | ---: |
| exact rank-64 scorer | 256 | 0 | 254.0 | 0.750403 |
| exact rank-64 scorer | 512 | 0 | 500.0 | 0.835798 |
| fine PQ 8x256 | 256 | 64 | 254.0 | 0.641013 |
| fine PQ 8x256 | 512 | 64 | 500.0 | 0.750698 |
| fine PQ 16x256 | 256 | 128 | 254.0 | 0.700645 |
| fine PQ 16x256 | 512 | 128 | 500.0 | 0.801169 |

The `fine PQ` rows score the full compressed key cache, then keep the verifier budget. The `exact rank-64 scorer` rows score all low-rank keys directly.

## Best Coarse-to-Fine Rows

| Coarse PQ | Fine PQ | Shortlist | Budget | Avg fine rescored | Top-16 recall |
| --- | --- | ---: | ---: | ---: | ---: |
| 4x64, 24 bits/key | 16x256, 128 bits/key | 4096 | 512 | 3104.0 | 0.799541 |
| 8x16, 32 bits/key | 16x256, 128 bits/key | 4096 | 512 | 3104.0 | 0.798100 |
| 4x16, 16 bits/key | 16x256, 128 bits/key | 4096 | 512 | 3104.0 | 0.797247 |
| 4x64, 24 bits/key | 16x256, 128 bits/key | 2048 | 512 | 1808.0 | 0.789078 |
| 8x16, 32 bits/key | 16x256, 128 bits/key | 2048 | 512 | 1808.0 | 0.784087 |
| 4x16, 16 bits/key | 16x256, 128 bits/key | 2048 | 512 | 1808.0 | 0.777902 |
| 4x64, 24 bits/key | 16x256, 128 bits/key | 4096 | 256 | 3104.0 | 0.700149 |
| 4x64, 24 bits/key | 16x256, 128 bits/key | 2048 | 256 | 1808.0 | 0.696351 |

Best `4096 -> 512` coarse-to-fine recall preserved `0.799541 / 0.801169 = 99.8%` of full fine-PQ recall. Best `2048 -> 512` preserved `98.5%`. At budget `256`, best coarse-to-fine preserved `99.9%` of full fine-PQ recall.

## Layer Check

For the strongest shape, `4x64 coarse -> 16x256 fine`, budget `512`:

| Layer | Fine PQ full scan | Coarse-to-fine 2048 | Coarse-to-fine 4096 | Exact ranker |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.966037 | 0.960069 | 0.965929 | 0.977539 |
| 1 | 0.622070 | 0.567708 | 0.610677 | 0.675781 |
| 5 | 0.896484 | 0.894531 | 0.896810 | 0.924805 |
| 10 | 0.935221 | 0.932183 | 0.935764 | 0.944227 |
| 18 | 0.740885 | 0.744249 | 0.741536 | 0.775065 |
| 24 | 0.673937 | 0.661133 | 0.674045 | 0.726780 |
| 29 | 0.773546 | 0.763672 | 0.772027 | 0.826389 |

## Interpretation

Coarse-to-fine PQ works as a recall-preserving serving shape for the learned rank-64 score. The coarse code only has to keep the fine-PQ winners in the candidate pool; the fine score handles ordering inside that pool.

The 4096 shortlist is large at an 8192-token context, but this is the wrong place for fixed shortlist economics to shine. At million-token context, a fixed 1024-4096 shortlist turns the expensive fine stage into a bounded operation. The next risk is direct throughput: cheap coarse scan over one million keys plus fine rescoring of a fixed shortlist.

## Next Test

Run a synthetic H100 benchmark for:

1. coarse PQ scan over one million keys;
2. shortlist top-k;
3. fine PQ rescoring only within `1024,2048,4096` shortlisted keys;
4. final verifier-budget top-k.

This should be compared against the previous full fine-PQ scan benchmark, where `8x256` took about `2.2 ms` and `16x256` took about `4.5 ms` per one-query layer over one million keys.
