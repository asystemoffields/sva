# Experiments

Each script here is a standalone, argparse-driven harness. Small settings run on CPU;
the published numbers were produced on Modal H100s through the `modal_h100_*.py`
runners at the repository root, which mount this directory and call these scripts'
entry points (launch commands in [`../docs/h100_runbook.md`](../docs/h100_runbook.md)).
Full configurations and readouts for every result below are in
[`../docs/snapshots/`](../docs/snapshots/).

Several runners reuse one harness with different flags (e.g. the weighted/hard coarse-PQ
runners all drive `sva_supervised_coarse_pq_test.py`, the evidence-rerank runner drives
`sva_evidence_haystack_benchmark.py`).

## Mechanism and lookup-geometry tests

| Script | What it tests | Key result |
| --- | --- | --- |
| `sva_kill_test.py` | Toy binding/retrieval task: content-addressed summon + exact verify vs full attention | `sva_24x10` top-1 `0.9995` on 8192 pages reading 16 candidates |
| `sva_causal_sequence_test.py` | Incremental causal cache: each step writes a page, later queries recover it | top-1 `0.9957` at 1024 tokens, ~54 pages read vs ~512 for full causal attention |
| `sva_trainable_recall_test.py` | Trainable modern-decoder recall benchmark | early positive; superseded by the pretrained socket |
| `sva_real_qk_address_sweep.py` | Random high-bit binary addresses over real SmolLM2 Q/K at 8192 | kill at million-token selectivity (recall 0.84 only at ~282k candidates) |
| `sva_million_stream_sim.py` | Million-token address pressure from empirical hit densities | kill: 1-2% recall in the 128-1024 candidate band |
| `sva_address_scaling.py` | Address selectivity calculator for long contexts | utility |
| `sva_learned_ranker_test.py` | Learned per-layer/head rank-64 asymmetric Q/K ranker (also held-out-text generalization) | top-16 recall `0.76`/`0.85` at 256/512 candidates, holds on held-out text |
| `sva_learned_lsh_lookup_test.py` | Sign-LSH serving of the learned score | kill (`0.23` recall at ~38.6k projected candidates) |
| `sva_learned_ivf_lookup_test.py` | IVF/centroid routing over learned keys | beats LSH; below ranker ceiling |
| `sva_learned_multiwrite_ivf_lookup_test.py` | Multi-write IVF (2/4/8 writes per key) | marginal over single-write |
| `sva_supervised_query_router_test.py` | Supervised query-cell router | recall collapses in the useful candidate band |
| `sva_lantern_router_test.py` | Supervised page-side write hooks ("Lantern") | trails IVF/PQ frontier; parked |
| `sva_pq_lookup_test.py` | Product-quantized learned-score lookup | `0.70`/`0.80` recall at 256/512 (16x256 codes) |
| `sva_coarse_to_fine_pq_test.py` | Coarse PQ shortlist, fine PQ rescore | `0.7995` vs `0.8012` full fine-PQ recall at shortlist 4096 |
| `sva_supervised_coarse_pq_test.py` | Supervised / attention-weighted / hard-negative coarse codebooks | hard-negative pool 512, boost 4: recall `0.827` at shortlist 512 (vs `0.714` before) |

## Model socketing and deployment benchmarks

| Script | What it tests | Key result |
| --- | --- | --- |
| `sva_pretrained_socket_test.py` | Replace Llama attention score matrices in SmolLM2-135M, keep everything else | all 30 layers, 2k: `KL=0.000362`, top-1 `0.9946` (after the input-layernorm fix) |
| `sva_output_distill_socket_test.py` | Output/logit-preserving probe for early and all-layer sockets | early-layer replacement needs direct training |
| `sva_deployment_socket_test.py` | Frozen artifacts on paragraph-order shifts (leakage audit) | effectively lossless on rotate/reverse/odds-evens |
| `sva_full_deployment_benchmark.py` | Frozen artifacts, held-out documents, context/budget sweeps | 8192 ctx, 2048/512: `KL=0.000481`, top-1 `0.99979`, top-16 recall `0.9987` |
| `sva_8k_head_to_head_benchmark.py` | Full attention vs SVA wall-clock + quality at 8k | `KL=0.000446` with 16x fewer exact scores/value reads; stock-PyTorch adapter slower wall-clock at 8k |
| `sva_cached_decode_benchmark.py` | Decode-time lookup quality with precomputed key catalogs | top-16 recall `0.998` at 8192 ctx, 2048/512 |
| `sva_inverted_adaptive_decode_benchmark.py` | Inverted-code (indexed) decode policies | adaptive variant too aggressive on passkeys; static inverted path is the production shape |

## Speed benchmarks (synthetic million-token caches, H100)

| Script | What it tests | Key result |
| --- | --- | --- |
| `sva_pq_scan_benchmark.py` | Full PQ scan over 1M keys | 2.2-4.5 ms/query |
| `sva_coarse_to_fine_pq_scan_benchmark.py` | Staged coarse-to-fine scan | 1.91 ms/query |
| `sva_coarse_exact_rescore_benchmark.py` | Coarse PQ + exact rank-64 rescore | ~1.0 ms/query at shortlist 2048; rescore itself ~0.12 ms |
| `sva_million_cached_decode_benchmark.py` | Cached decode vs full attention over 1M keys | SVA 1.02 ms vs full attention 2.09 ms (q=1, 2048/512); `1x256` codes reach 0.65 ms |

## Long-context and passkey benchmarks

| Script | What it tests | Key result |
| --- | --- | --- |
| `sva_long_context_recall_sim.py` | Fixed 8k artifact recall as the key bank grows (8k to 1M) | `16384/2048` recovers 128k (`0.943` recall, 64x fewer reads); 1M needs more catalog capacity |
| `sva_passkey_language_benchmark.py` | Passkey retrieval NLL/KL vs full attention after cached prefill | late4 socket at 32k: aggregate `KL=0.005547`, top-1 `0.968` over 9 cases |
| `sva_passkey_prefill_drift_benchmark.py` | Where long-context drift enters (prefill, per-profile) | drift is early-layer prefill; motivated layer-selective socketing |
| `sva_evidence_haystack_benchmark.py` | Does the needed evidence survive summon as context grows (plus rerank/expansion flags) | multi-anchor summon lifts survival `0.59` to `0.96` at 8k |
| `sva_span_statement_benchmark.py` | Verify local spans around summoned evidence | output cosine `0.991` to `0.998` at 8k, radius 32 |
| `sva_block_elevator_benchmark.py` | Summon contiguous blocks, merge local softmax partials | blocks beat scattered tokens at 131k for the same read budget |
| `sva_block_hybrid_benchmark.py` | Route each head/query between token and block SVA | oracle selector: relative error `0.165` vs `0.457` token-only at 131k |
| `sva_learned_hybrid_selector_benchmark.py` | Tiny MLP selector for token/block routing | relative error `0.179` at 131k, near the `0.165` oracle |

## Catalog quality and adaptation

| Script | What it tests | Key result |
| --- | --- | --- |
| `sva_rotation_diagnostic.py` | Frozen vs refit coarse codebooks | refit lifts teacher recall `0.772` to `0.838` at budget 512 |
| `sva_codebook_refresh_benchmark.py` | Held-out calibration-time codebook refresh at 32k | recall `0.563` to `0.636` at budget 512, near the eval-refit ceiling |
| `sva_late4_logit_distill.py` | Train a 110k-param residual adapter on top of late4 SVA at 512/128 | held-out 32k answer-token KL `0.053` to `0.019` (answer-KL+CE objective) |
| `sva_late4_adapter_answer_benchmark.py` | Full answer-decode validation of saved adapters with unadapted control | 24-case panel: `KL=0.032`, top-1 `0.946` at 512/128, 256x decode read reduction |

## Artifact tooling

| Script | What it does |
| --- | --- |
| `sva_artifact_io.py` | Save/load helpers for frozen artifact bundles (the `sva/` package has the production loader) |
| `export_sva_artifact.py` | Train and export an HF/GitHub-ready artifact folder |
| `export_refreshed_sva_artifact.py` | Export with calibration-refreshed coarse codebooks, preserving trained projections |
