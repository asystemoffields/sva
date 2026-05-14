# SVA Block Elevator Snapshot

This snapshot records the first block-first SVA "elevator" test.

## Question

Can SVA summon contiguous blocks, then let selected blocks compute their own exact local attention statements in place?

That serving shape has two parts:

1. Summon blocks instead of scattered individual token positions.
2. For each selected block, compute local exact QK scores, local softmax max/sum statistics, and a local weighted value; merge those partials into the final output with the usual streaming softmax recurrence.

In the metaphor, the witnesses give their statements from where they are already sitting. The court receives compact sworn summaries from each selected row, then merges them into one verdict.

## Run

- Modal app: `sva-block-elevator-h100`
- Function call: `fc-01KRK8DDDY8DB39DJERRNJ4REC`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Base activations: real SmolLM2 Q/K/V from an 8192-token held-out document
- Target contexts: `8192`, `32768`, `131072`
- Layers: `0`, `15`, `29`
- Queries: 8 late-context positions per layer
- Teacher: exact full attention output and top-16 keys, computed by streaming over the full KV bank
- Synthetic extension: sample real base keys/values with `0.01` Gaussian noise

The token baseline uses the existing SVA shape: compact-code shortlist, exact low-rank rerank, then exact attention over selected individual tokens. The block variants choose blocks by either token-code max score (`coarse_max`) or block low-rank centroids (`centroid`), then compute exact local block statements.

## Token Baseline

The token baseline used `8192/2048`, so it always verified up to `2048` individual tokens.

| context | tokens read | read reduction | top-16 recall | output cosine | relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 2048 | 4x | 0.999421 | 0.995650 | 0.081375 |
| 32768 | 2048 | 16x | 0.989873 | 0.979064 | 0.245475 |
| 131072 | 2048 | 64x | 0.896701 | 0.943327 | 0.456380 |

## Block Centroid Summary

These rows average layers `0`, `15`, and `29`. `segments` equals selected blocks per head/query. A production kernel would work over these contiguous segments rather than thousands of scattered gathers.

| context | block | segments | tokens read | read reduction | top-16 recall | output cosine | relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 64 x 32 | 32 | 2048 | 4x | 0.355324 | 0.968464 | 0.225213 |
| 8192 | 64 x 64 | 64 | 4096 | 2x | 0.612558 | 0.992352 | 0.102814 |
| 32768 | 64 x 32 | 32 | 2048 | 16x | 0.111690 | 0.966077 | 0.243210 |
| 32768 | 128 x 64 | 64 | 8192 | 4x | 0.338831 | 0.992719 | 0.115157 |
| 131072 | 64 x 32 | 32 | 2048 | 64x | 0.032118 | 0.966790 | 0.276183 |
| 131072 | 64 x 64 | 64 | 4096 | 32x | 0.064525 | 0.982508 | 0.196981 |
| 131072 | 128 x 64 | 64 | 8192 | 16x | 0.105613 | 0.990151 | 0.135000 |

## Reading

The block path is a real serving idea. A selected block can compute a compact local statement:

- local max score
- local denominator contribution
- local weighted value vector

Those statements merge exactly into attention over the selected blocks. This avoids materializing a token-level candidate pile and gives the kernel contiguous work.

The surprising result is that top-key recall and output preservation diverge. Centroid block selection misses most exact full-attention top-16 keys at long context, yet preserves the value output much better than that recall suggests. At `131072` tokens with the same `2048` value reads, token SVA had output cosine `0.943327` and relative error `0.456380`; centroid blocks had output cosine `0.966790` and relative error `0.276183`.

The risk is exact-token survival. For passkey-style retrieval, low top-16 recall can still matter even when the value output looks close on average. The promising shape is hybrid:

1. Use token SVA when the query appears peaky or exact-token sensitive.
2. Use block elevator SVA when attention is diffuse enough that preserving the value output matters more than recovering the exact top keys.
3. Learn or calibrate that dispatch by layer, head, query margin, and context length.

## Timing Note

The current `method_ms` values are diagnostic only. The block statement function loops in Python over heads, queries, and selected blocks. The useful systems numbers in this run are `tokens_read` and `segments_per_query`: block SVA turns `2048` scattered token segments into `16-64` contiguous block segments.

## Next Test

Run a hybrid deployment benchmark that chooses token SVA or block SVA per layer/head/query using cheap confidence features:

- coarse-score margin
- centroid entropy
- expected attention diffuseness
- layer index
- target context length

The benchmark should report both language-level quality and serving-shaped cost on the passkey setup, because that is where exact survival and long-context utility collide.
