# Result Snapshots

Every experiment in this repo ends in a dated snapshot markdown file recording the run
configuration, the numbers, a readout, and the next step. This directory holds all 80 of
them. They were written in real time (2026-05-13 to 2026-05-14) and are kept verbatim;
some interpretations were later revised by follow-up runs (most notably the
attention-input normalization fix, see `normfix_socket_audit_snapshot_2026-05-13.md`).

For the chronological narrative that ties these together, see
[`../research_log.md`](../research_log.md). For the headline numbers, see the
top-level [README](../../README.md).

## The arc, in brief

1. SVA worked on synthetic binding tasks: near-full-attention top-1 with 16 reads out of 8192 pages.
2. It then survived first contact with a pretrained model (SmolLM2-135M) as a drop-in score-matrix replacement.
3. Random high-bit addresses died at scale; a learned low-rank ranker plus coarse/fine product quantization became the lookup geometry.
4. An apparent "fragile layer" failure turned out to be a harness bug (artifacts trained on pre-norm hidden states, served post-norm). After the fix, all 30 layers socket at `KL=0.000362` at 2k.
5. Frozen artifacts held on held-out documents up to the model's full 8192 window, and million-token lookups ran in ~0.65-1.0 ms on H100 with stock PyTorch.
6. Long contexts (32k+) exposed prefill drift in early layers; codebook refresh and attention-weighted catalogs helped recall but not language quality.
7. Layer-selective socketing resolved it: full attention through layer 25, SVA in layers 26-29 ("late4") stays at `KL≈0.006` on 32k passkeys.
8. A tiny distilled residual adapter recovered tight budgets (512/128, 256x decode read reduction, answer KL 0.032), and a static inverted index replaced scan summon for decode.
9. A supervised "Lantern" router probe for page-side writes underperformed the IVF/PQ frontier and was parked.

## Phase 1 — Mechanism validation on synthetic tasks (2026-05-13)

| Snapshot | Records |
| --- | --- |
| `verification_snapshot_2026-05-13.md` | Kill-test results: binding task, 8192 pages, 16-candidate verifier near full-attention top-1. |
| `causal_sequence_snapshot_2026-05-13.md` | Incremental causal-cache test: timesteps write pages, later queries recover them. |
| `trainable_recall_snapshot_2026-05-13.md` | H100 trainable-representation recall checkpoint. |

## Phase 2 — First contact with a pretrained model (2026-05-13)

| Snapshot | Records |
| --- | --- |
| `pretrained_socket_snapshot_2026-05-13.md` | SmolLM2-135M attention-socket first sweep. |
| `pretrained_long_socket_snapshot_2026-05-13.md` | Longer-context (512-token) socket sweep. |
| `pretrained_prefilter_socket_snapshot_2026-05-13.md` | Cheap random-projection prefilter inside the socket. |

## Phase 3 — Address scaling and lookup geometry (2026-05-13)

| Snapshot | Records |
| --- | --- |
| `real_qk_address_8192_snapshot_2026-05-13.md` | Real-QK high-bit binary address sweep at the full 8192 window; random addresses are a kill at million-token scale. |
| `million_stream_snapshot_2026-05-13.md` | Million-token address-pressure simulation from real Q/K hit densities. |
| `learned_ranker_snapshot_2026-05-13.md` | Learned low-rank (rank-64) Q/K ranker: 0.76-0.85 top-16 recall at 256-512 candidates. |
| `learned_ranker_generalization_snapshot_2026-05-13.md` | Held-out-text generalization of the learned ranker. |
| `learned_lsh_lookup_snapshot_2026-05-13.md` | Sign-LSH over the learned space: a kill. |
| `learned_ivf_lookup_snapshot_2026-05-13.md` | IVF/centroid routing over learned keys. |
| `learned_multiwrite_ivf_lookup_snapshot_2026-05-13.md` | Multi-write IVF: marginal over single-write. |
| `supervised_query_router_snapshot_2026-05-13.md` | Supervised query-cell router, low resolution. |
| `supervised_query_router_hires_snapshot_2026-05-13.md` | High-resolution supervised router: recall collapses in the useful candidate band. |
| `pq_lookup_snapshot_2026-05-13.md` | Product-quantized learned-score lookup: preserves most ranker recall. |
| `pq_scan_benchmark_snapshot_2026-05-13.md` | Million-token PQ scan throughput (2.2-4.5 ms/query). |
| `coarse_to_fine_pq_snapshot_2026-05-13.md` | Coarse-to-fine PQ staging preserves fine-PQ recall. |
| `coarse_to_fine_pq_scan_benchmark_snapshot_2026-05-13.md` | Staged scan throughput: 1.91 ms/query over 1M keys. |
| `coarse_exact_rescore_benchmark_snapshot_2026-05-13.md` | Coarse PQ + exact rank-64 rescore: ~1.0 ms/query over 1M keys. |
| `supervised_coarse_pq_snapshot_2026-05-13.md` | Supervised coarse stage vs broad fine-PQ targets: a regression. |
| `supervised_coarse_pq_attention16_snapshot_2026-05-13.md` | Attention top-16 labels recover the supervised coarse signal. |
| `weighted_coarse_pq_snapshot_2026-05-13.md` | Attention-weighted coarse codebooks in the fine-ranker space. |
| `weighted_supervised_coarse_pq_snapshot_2026-05-13.md` | Weighted codebooks inside a supervised coarse space: gains stack. |
| `weighted_supervised_coarse_pq_tight_snapshot_2026-05-13.md` | Tight-shortlist (512-1024) pressure test. |
| `hard_supervised_coarse_pq_snapshot_2026-05-13.md` | Hard-negative coarse training for shortlist survival. |
| `hard_pool_sweep_snapshot_2026-05-13.md` | Mining-pool sweep: pool 512 / boost 4 best (recall 0.827 at shortlist 512). |
| `hard_handoff_snapshot_2026-05-13.md` | Handoff diagnostic isolating the fine-PQ rescore dip at shortlist 2048. |

## Phase 4 — Socket debugging and the normalization fix (2026-05-13)

| Snapshot | Records |
| --- | --- |
| `three_stage_socket_snapshot_2026-05-13.md` | Three-stage socket and layer-isolation tests; first sighting of apparent fragile layers. |
| `three_condition_socket_snapshot_2026-05-13.md` | Hidden-state vs progressive vs selective-hybrid socket conditions. |
| `layer_frontier_snapshot_2026-05-13.md` | Selective socket layer-frontier sweep. |
| `layer_cliff_snapshot_2026-05-13.md` | Cliff-mapping of layers that appeared to reject SVA. |
| `layer_fallback_snapshot_2026-05-13.md` | Per-layer fallback tests. |
| `layer_admission_snapshot_2026-05-13.md` | Automatic admission screening. |
| `normfix_socket_audit_snapshot_2026-05-13.md` | **The fix.** Artifacts were trained on pre-norm hidden states but served on `input_layernorm` outputs. After correction, all 30 layers socket at `KL=0.000362` (2k). The fragile-layer interpretation in the four snapshots above is stale. |

## Phase 5 — Deployment benchmarks, artifacts, production adapter (2026-05-14)

| Snapshot | Records |
| --- | --- |
| `full_deployment_benchmark_snapshot_2026-05-14.md` | Frozen artifacts on held-out documents at 2k/4k: effectively lossless. |
| `full_deployment_8192_snapshot_2026-05-14.md` | Full-window 8192 held-out benchmark: `KL=0.000481`, top-1 `0.999786` at 2048/512. |
| `cached_decode_benchmark_snapshot_2026-05-14.md` | Cached-key decode quality split from harness overhead. |
| `million_cached_decode_benchmark_snapshot_2026-05-14.md` | Synthetic 1M-key cached decode: first no-custom-kernel speed opening. |
| `tight_summon_frontier_snapshot_2026-05-14.md` | Tight-shortlist quality/speed frontier; cost dominated by coarse scan, not verifier. |
| `compact_summon_frontier_snapshot_2026-05-14.md` | Compact coarse codes: `1x256` at 0.65 ms/query over 1M keys vs 2.08 ms full attention. |
| `artifact_export_snapshot_2026-05-14.md` | First deployable artifact bundle (`sva-smollm2-135m-2x256-v1`). |
| `production_adapter_snapshot_2026-05-14.md` | The `sva/` package: reversible Llama patching, cached decode, chat demo, smoke tests. |
| `long_context_extension_snapshot_2026-05-14.md` | 8k head-to-head plus 128k/1M fixed-artifact recall proxy. |

## Phase 6 — Long-context stress and catalog quality (2026-05-14)

| Snapshot | Records |
| --- | --- |
| `passkey_language_snapshot_2026-05-14.md` | First passkey language stress test of decode policy. |
| `block_elevator_snapshot_2026-05-14.md` | Contiguous-block summon with local softmax statements. |
| `block_hybrid_snapshot_2026-05-14.md` | Token/block hybrid: oracle selector shows complementarity. |
| `learned_hybrid_selector_snapshot_2026-05-14.md` | Tiny MLP selector approaches the oracle at 32k. |
| `evidence_haystack_snapshot_2026-05-14.md` | Multi-anchor summon lifts evidence survival as context grows. |
| `evidence_rerank_snapshot_2026-05-14.md` | Evidence-aware rerank and neighborhood expansion. |
| `span_statement_snapshot_2026-05-14.md` | Span-statement verification preserves local evidence neighborhoods. |
| `rotation_diagnostic_snapshot_2026-05-14.md` | Codebook refit opens a large recall gap over frozen codebooks. |
| `codebook_refresh_snapshot_2026-05-14.md` | Held-out calibration-time codebook refresh generalizes the refit win at 32k. |
| `refreshed_profile_snapshot_2026-05-14.md` | Exported long-context refreshed artifact and recall sanity check. |
| `passkey_profile_router_snapshot_2026-05-14.md` | Context-routed profiles: recall gains do not convert to passkey gains (useful negative). |
| `attention_weighted_refresh_snapshot_2026-05-14.md` | Attention-weighted refresh beats the identity refit ceiling on teacher recall. |
| `passkey_attention_weighted_router_snapshot_2026-05-14.md` | Language-facing test of the attention-weighted profile: mixed. |
| `attention_weighted_router_sweep_snapshot_2026-05-14.md` | Boost-strength sweep sharpening the 16k/32k tradeoff. |
| `passkey_key_survival_profiles_snapshot_2026-05-14.md` | Final-query key survival is similar across profiles; loss is summon-side. |
| `passkey_prefill_drift_profiles_snapshot_2026-05-14.md` | Prefill drift identified as the long-context failure mode. |
| `output_distill_socket_snapshot_2026-05-14.md` | Output/logit-preserving probe for early and all-layer sockets. |

## Phase 7 — Layer-selective socketing: the late4 finding (2026-05-14)

| Snapshot | Records |
| --- | --- |
| `passkey_layer_selective_prefill_snapshot_2026-05-14.md` | Socketing only layers 20-29 cuts 32k prefill KL from 2.148 to 0.020. |
| `passkey_layer_selective_language_snapshot_2026-05-14.md` | Full-answer validation of late-layer sockets. |
| `passkey_late_boundary_language_snapshot_2026-05-14.md` | Boundary sweep lands on `late4` (layers 26-29): KL 0.0055 at 32k. |
| `passkey_late4_robustness_snapshot_2026-05-14.md` | 9-case multi-key/placement panel: aggregate KL 0.005547, top-1 0.968. |
| `passkey_early26_language_snapshot_2026-05-14.md` | Replacing layers 0-25 alone reproduces the all-layer drift; the damage is early. |
| `late4_budget_sweep_snapshot_2026-05-14.md` | Budget squeeze: 512/128 keeps gold-answer NLL with 256x fewer reads (KL 0.086, the adaptation target). |

## Phase 8 — Tight-budget adaptation and indexed summon (2026-05-14)

| Snapshot | Records |
| --- | --- |
| `late4_logit_distill_snapshot_2026-05-14.md` | 110k-parameter residual adapter cuts held-out 32k KL 0.045 to 0.009 at the final prompt. |
| `late4_adapter_answer_snapshot_2026-05-14.md` | Saved-adapter answer-decode validation with unadapted control. |
| `late4_answer_distill_snapshot_2026-05-14.md` | Answer-token distillation: answer KL 0.081 to 0.028 on the 9-case panel. |
| `late4_answer_ce_distill_snapshot_2026-05-14.md` | Answer-KL + 0.01 gold-CE objective: best tight-budget adapter. |
| `late4_answerce_broad_panel_snapshot_2026-05-14.md` | 24-case held-out panel: KL 0.032, top-1 0.946 at 512/128 with 256x read reduction. |
| `late4_answerce_inverted_panel_snapshot_2026-05-14.md` | First indexed (inverted posting) summon: quality-credible, wall-clock poor. |
| `late4_answerce_inverted_tight_panel_snapshot_2026-05-14.md` | 4/8 cells per subspace: bottleneck is the Python posting path, not reads. |
| `late4_answerce_inverted_floor_panel_snapshot_2026-05-14.md` | 1/2 cells overhead floor confirms the implementation wall. |
| `late4_answerce_inverted_static_panel_snapshot_2026-05-14.md` | Vectorized static inverted decode: slowdown 3.3x to 1.87x. |
| `late4_answerce_inverted_static_refill_snapshot_2026-05-14.md` | Duplicate refill restores honest 128-token verification accounting. |
| `late4_answerce_static_tail_profile_snapshot_2026-05-14.md` | Component profiling + tail buffering: decode slowdown to 1.29x; prefill remains 25x. |

## Phase 9 — Alternative lookup geometry probes (2026-05-14)

| Snapshot | Records |
| --- | --- |
| `lantern_router_capacity_snapshot_2026-05-14.md` | Supervised page-side write hooks with capacity-balanced cells. |
| `lantern_router_alignment_snapshot_2026-05-14.md` | Route-alignment follow-up; still trails the IVF/PQ frontier, parked. |
