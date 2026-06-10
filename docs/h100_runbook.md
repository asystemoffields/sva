# H100 Run Commands (Modal)

Every `modal_h100_*.py` file at the repository root is a Modal H100 runner that wraps
one experiment from `experiments/` (the runner pins the image, mounts the repo, and
calls the experiment's main entry point on an H100). The launcher script below uses
`modal run --detach` and writes run metadata, stdout, stderr, and result files under
`results/modal_runs/`. The launch scripts are PowerShell because this work was run
from a Windows host.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-trainable
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-socket -ModalFile modal_h100_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-three-stage-socket -ModalFile modal_h100_three_stage_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-million-stream -ModalFile modal_h100_million_stream.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ranker -ModalFile modal_h100_learned_ranker.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ranker-generalize -ModalFile modal_h100_learned_ranker_generalize.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-lsh-lookup -ModalFile modal_h100_learned_lsh_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ivf-lookup -ModalFile modal_h100_learned_ivf_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-multiwrite-ivf-lookup -ModalFile modal_h100_learned_multiwrite_ivf_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-query-router -ModalFile modal_h100_supervised_query_router.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-query-router-hires -ModalFile modal_h100_supervised_query_router_hires.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-pq-lookup -ModalFile modal_h100_pq_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-pq-scan-benchmark -ModalFile modal_h100_pq_scan_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-coarse-to-fine-pq -ModalFile modal_h100_coarse_to_fine_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-coarse-to-fine-pq-scan-benchmark -ModalFile modal_h100_coarse_to_fine_pq_scan_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-coarse-exact-rescore-benchmark -ModalFile modal_h100_coarse_exact_rescore_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-coarse-pq -ModalFile modal_h100_supervised_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-supervised-coarse-pq-attention16 -ModalFile modal_h100_supervised_coarse_pq_attention16.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-weighted-coarse-pq -ModalFile modal_h100_weighted_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-weighted-supervised-coarse-pq -ModalFile modal_h100_weighted_supervised_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-weighted-supervised-coarse-pq-tight -ModalFile modal_h100_weighted_supervised_coarse_pq_tight.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-hard-supervised-coarse-pq -ModalFile modal_h100_hard_supervised_coarse_pq.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-hard-pool-sweep -ModalFile modal_h100_hard_pool_sweep.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-hard-handoff -ModalFile modal_h100_hard_handoff.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-three-stage-socket-layers -ModalFile modal_h100_three_stage_socket_layers.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-three-condition-socket -ModalFile modal_h100_three_condition_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-frontier -ModalFile modal_h100_layer_frontier.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-cliff -ModalFile modal_h100_layer_cliff.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-fallback -ModalFile modal_h100_layer_fallback.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-layer-admission -ModalFile modal_h100_layer_admission.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-normfix-all-layers -ModalFile modal_h100_normfix_all_layers.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-deployment-socket -ModalFile modal_h100_deployment_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-full-deployment-benchmark -ModalFile modal_h100_full_deployment_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-full-deployment-8192 -ModalFile modal_h100_full_deployment_8192.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-cached-decode-benchmark -ModalFile modal_h100_cached_decode_benchmark.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-million-cached-decode -ModalFile modal_h100_million_cached_decode.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-tight-summon-frontier -ModalFile modal_h100_tight_summon_frontier.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-compact-summon-frontier -ModalFile modal_h100_compact_summon_frontier.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-export-artifact -ModalFile modal_h100_export_sva_artifact.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-export-refreshed-artifact -ModalFile modal_h100_export_refreshed_artifact.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-block-elevator -ModalFile modal_h100_block_elevator.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-block-hybrid -ModalFile modal_h100_block_hybrid.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-hybrid-selector -ModalFile modal_h100_learned_hybrid_selector.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-evidence-haystack -ModalFile modal_h100_evidence_haystack.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-evidence-rerank -ModalFile modal_h100_evidence_rerank.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-profile-router -ModalFile modal_h100_passkey_profile_router.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-span-statement -ModalFile modal_h100_span_statement.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-rotation-diagnostic -ModalFile modal_h100_rotation_diagnostic.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-codebook-refresh -ModalFile modal_h100_codebook_refresh.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-attention-weighted-refresh -ModalFile modal_h100_attention_weighted_refresh.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-refreshed-profile-recall -ModalFile modal_h100_refreshed_profile_recall.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-export-attnweighted-artifact -ModalFile modal_h100_export_attention_weighted_artifact.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-attnweighted-router -ModalFile modal_h100_passkey_attention_weighted_router.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-attnweighted-router-sweep -ModalFile modal_h100_attention_weighted_router_sweep.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-key-survival-profiles -ModalFile modal_h100_passkey_key_survival_profiles.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-prefill-drift-profiles -ModalFile modal_h100_passkey_prefill_drift_profiles.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-layer-selective-prefill -ModalFile modal_h100_passkey_layer_selective_prefill.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-layer-selective-language -ModalFile modal_h100_passkey_layer_selective_language.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-late-boundary-language -ModalFile modal_h100_passkey_late_boundary_language.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-passkey-late4-robustness -ModalFile modal_h100_passkey_late4_robustness.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-logit-distill -ModalFile modal_h100_late4_logit_distill.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-adapter-answer -ModalFile modal_h100_late4_adapter_answer.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-answer-distill -ModalFile modal_h100_late4_answer_distill.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-answerdistill-adapter-answer -ModalFile modal_h100_late4_answerdistill_adapter_answer.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-answer-ce-distill -ModalFile modal_h100_late4_answer_ce_distill.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-answerce-adapter-answer -ModalFile modal_h100_late4_answerce_adapter_answer.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-answerce-broad-panel -ModalFile modal_h100_late4_answerce_broad_panel.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-late4-answerce-inverted-panel -ModalFile modal_h100_late4_answerce_inverted_panel.py
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\scripts\start_modal_h100_background.ps1' -Name 'sva-h100-late4-answerce-inverted-tight-panel' -ModalFile 'modal_h100_late4_answerce_inverted_panel.py' -ModalArgs '--cells','4,8'"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\scripts\start_modal_h100_background.ps1' -Name 'sva-h100-late4-answerce-inverted-static-panel' -ModalFile 'modal_h100_late4_answerce_inverted_panel.py' -ModalArgs '--cells','8,16','--summon-mode','inverted_static'"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\scripts\start_modal_h100_background.ps1' -Name 'sva-h100-static-tail-full-panel' -ModalFile 'modal_h100_late4_answerce_inverted_panel.py' -ModalArgs '--cells','8','--summon-mode','inverted_static'"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-lantern-router -ModalFile modal_h100_lantern_router.py
```

The launcher uses `modal run --detach` and writes local metadata, stdout, stderr, and result files under `results/modal_runs/`.

Live progress is visible through Modal logs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_modal_h100.ps1 -Tail 200
```
