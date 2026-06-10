# SVA Learned Hybrid Selector Snapshot

This snapshot records the first learned dispatcher between token SVA and block-elevator SVA.

## Question

Can cheap pre-verifier features predict when a query should use scattered token SVA versus contiguous block SVA?

This matters because the long-context problem has split into two failure modes:

- token SVA keeps exact-key recall but can preserve the wrong value mixture at long synthetic context;
- block SVA preserves diffuse value output but loses exact top keys;
- an oracle mixture beats both, so the missing part is routing.

## Run

- Modal app: `sva-learned-hybrid-selector-h100`
- Function call: `fc-01KRKAVNZMAAK6M6F1MG4PE7GR`
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Artifact: `results/hf_artifacts/sva-smollm2-135m-2x256-v1`
- Train document: held-out eval doc `0`
- Test document: held-out eval doc `1`
- Contexts: `8192`, `32768`, `131072`
- Layers: `0`, `15`, `29`
- Query positions: 8 late positions per layer
- Block configs: `64 x 16`, `64 x 32`, `128 x 16`, `128 x 32`
- Token baseline: `8192/2048`

The selector is a tiny two-layer MLP trained on `2592` examples. Labels choose token SVA when token output has lower relative error than block output. Features are available before exact verification:

- token coarse-score entropy and margin
- low-rank query norm
- block-score entropy, margin, and spread
- head/query position
- layer id
- context length
- block geometry
- block/token read ratio

Training accuracy was `0.943673`; token-positive rate was `0.599923`.

## Held-Out Baselines

These rows average layers `0`, `15`, and `29` on the held-out test document.

| method | context | value reads | segments | output cosine | relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| token | 8192 | 2048 | 2048 | 0.990090 | 0.118552 |
| block `64 x 32` | 8192 | 2048 | 32 | 0.976566 | 0.223182 |
| block `128 x 16` | 8192 | 2048 | 16 | 0.975248 | 0.237744 |
| token | 32768 | 2048 | 2048 | 0.968438 | 0.320695 |
| block `64 x 32` | 32768 | 2048 | 32 | 0.971523 | 0.227765 |
| block `128 x 16` | 32768 | 2048 | 16 | 0.969560 | 0.244531 |
| token | 131072 | 2048 | 2048 | 0.928426 | 0.570985 |
| block `64 x 32` | 131072 | 2048 | 32 | 0.968925 | 0.273731 |
| block `128 x 16` | 131072 | 2048 | 16 | 0.968674 | 0.250723 |

## Learned Selector

Representative equal-read rows:

| selector | context | block | token fraction | avg segments | top-16 recall | output cosine | relative error |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| entropy `0.55` | 8192 | `64 x 32` | 0.814815 | 1674.67 | 0.879919 | 0.994326 | 0.094537 |
| learned `0.50` | 8192 | `64 x 32` | 0.712963 | 1469.33 | 0.817419 | 0.995820 | 0.076963 |
| oracle | 8192 | `64 x 32` | 0.768519 | 1581.33 | 0.853588 | 0.998845 | 0.038614 |
| entropy `0.55` | 32768 | `128 x 16` | 0.379630 | 787.41 | 0.440683 | 0.981582 | 0.234377 |
| learned `0.35` | 32768 | `128 x 16` | 0.564815 | 1163.70 | 0.605324 | 0.992115 | 0.106382 |
| oracle | 32768 | `128 x 16` | 0.564815 | 1163.70 | 0.600694 | 0.993224 | 0.096472 |
| entropy `0.55` | 131072 | `128 x 16` | 0.277778 | 580.44 | 0.292824 | 0.973833 | 0.241209 |
| learned `0.50` | 131072 | `128 x 16` | 0.287037 | 599.26 | 0.294271 | 0.981577 | 0.179351 |
| oracle | 131072 | `128 x 16` | 0.370370 | 768.59 | 0.376447 | 0.987161 | 0.147850 |

The learned selector is a clear improvement over the hand entropy rule. It also gets surprisingly close to the oracle on the held-out `32768` row while keeping `2048` value reads.

At `131072`, learned `128 x 16` improves relative error from token SVA's `0.570985` and block-only `0.250723` to `0.179351`, with about `599` average segments instead of `2048` scattered token segments.

## Reading

This is a go for a learned dispatcher. The selector learned a meaningful routing rule from cheap signals and transferred to a different held-out document and synthetic key/value sample.

This does not solve long-context by itself. The remaining gap is:

- exact-token tasks still need high-recall token survival;
- diffuse-output tasks benefit from block statements;
- production needs a selector trained against language loss, not only layer-output relative error.

## Why SVA Does Not Automatically Extend Context

SVA reduces exact attention work after lookup. It does not guarantee that the right keys are in the candidate set.

As context grows, several things happen:

- a fixed verifier budget covers a smaller fraction of the sequence;
- approximate catalogs get more collisions and near misses;
- RoPE and the pretrained model's representations may be outside their trained regime;
- passkey-style tasks punish one missed token even when average output cosine looks good;
- current no-custom-kernel paths still scan compact codes over the whole cache, so some work remains linear in context.

So SVA expands the feasible context only when summon recall and catalog capacity scale with the window. The block and hybrid results are promising because they show another route: preserve the value output using contiguous summaries when exact top-key identity is less important.

## Next Test

Socket the learned selector into a language-facing passkey benchmark:

1. token mode for exact/peaky queries,
2. block mode for diffuse queries,
3. report answer NLL, logit KL, value reads, segment count, and wall time.

That is the next sharp test for whether this routing actually helps long-context use, not only synthetic layer-output reconstruction.
