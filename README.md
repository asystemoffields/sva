# Summon-Verify Attention

Summon-Verify Attention (SVA) is a candidate sparse replacement for transformer attention.

The idea is simple:

1. Each page writes itself into several cheap content-addressed lookup tables.
2. A query activates the same addresses and summons a small candidate set.
3. A verifier runs exact dot-product attention over only the summoned candidates.

In the current toy tests, the write address and read address are the same object. That is the key design move: the memory does not need a separate librarian to learn where every page went.

## Quick Start

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python experiments\\sva_kill_test.py --task binding --trials 2 --tables 8 16 24 --bits 10 --budget 16 --query-noise 0.05 --logit-scale 16
```

## Current Result

On the 8192-page binding task with a 16-candidate verifier budget, SVA reaches near-full-attention top-1 recovery while reading only 16 candidates:

```text
method              top1    cos_teacher  avg_candidates
full_attention      1.0000  1.0000       8192.0
coarse_bank_verify  0.5552  0.5820       15.7
sva_16x10           0.9907  0.9721       16.0
sva_24x10           0.9995  0.9859       16.0
```

The newest robustness result is adjacent-bucket probing plus a cheap prefilter. Under noisy binding lookup, plain `sva_16x10` reached `top1=0.7124`; `sva_probe1_16x10` reached `top1=0.9263`, matching the full-attention teacher's `0.9287` on that setup while verifying 16 candidates.

With a 32-dimensional prefilter, `sva_probe1_prefilter32d128_16x10` kept `top1=0.9233` while reducing exact full-dimensional scoring from about 968 summoned pages to 128. That makes the working shape: summon broadly, cheap prefilter, exact verify.

The causal-cache test is now positive too. At 1024 tokens, `sva_causal_probe1_prefilter32d128_24x12` reached `top1=0.9957` with about 54 summoned prior pages on average, while full causal attention read about 512 prior pages on average in the same setup.

The pretrained socket test is now the sharpest signal. In `HuggingFaceTB/SmolLM2-135M-Instruct`, SVA can replace every Llama attention layer's score matrix while keeping the pretrained Q/K/V/O projections, RoPE, norms, MLPs, and logits. The best first H100 sweep matched full attention closely on short prompts:

```text
setting                         loss_delta  KL_to_full  top1_agree  logit_cos  avg_verified
32 tables, 10 bits, probe 2     0.093750    0.188110    0.783883    0.974020   20.432 / 53
```

The longer-context H100 sweep strengthened that result. At 512 tokens, `64 tables / 10 bits / probe 2 / budget 128` reached `loss_delta=0.015625`, `KL=0.009582`, `top1_agreement=0.970646`, and top-16 full-attention key recall `0.976191`.

The next research step is to make the summoned candidate set smaller. In the current socket harness, `avg_summoned` is the broad lookup set; without a prefilter, that is also the exact-scored set. At 512 tokens the strongest setting summons about 221 candidates before the post-score top-k budget.

The first random-projection prefilter reduced exact scoring but exposed the next bottleneck. At 256 tokens, `prefilter_dim=48 / prefilter_budget=64` cut exact scoring from about 113 candidates to 55 with `loss_delta=0.062500`. At 512 tokens, the same shape cut exact scoring from about 223 to 60 with `loss_delta=0.125000`. The next invention target is a better cheap ranker inside the summoned set.

The full-window real-QK address sweep now matches SmolLM2's configured context window: `max_position_embeddings=8192`, `seq_len=8192`. Random high-bit binary addresses are a kill for the million-token version of that exact address function. Aggregate top-16 recall at `14 bits / 128 tables / radius 2` was `0.838557`, but the random million-token candidate estimate was about `282k`. At `24 bits / 128 tables / radius 2`, the estimate falls to about `1.1k`, but top-16 recall was only `0.092231`.

The million-token pressure simulation sharpened that result using empirical hit density from real 8192-token SmolLM2 Q/K samples. The best aggregate recall was `20 bits / 256 tables / radius 2` at `0.384905`, but it projected to about `39.6k` average candidates at a million tokens, with p95 about `129k`. In the rough 128-1024 candidate band, aggregate recall stayed around 1-2%. The next work is a learned or model-aware address code.

The learned compressed-ranker test is the first strong follow-up. Training a small asymmetric Q/K ranker per layer/head on held-in query positions and evaluating held-out query positions reached aggregate top-16 recall `0.759781` with a 64-dimensional score and 256 verifier candidates, and `0.848338` with 512 verifier candidates. The next risk is held-out text generalization, then serving the learned score through an addressable lookup.

The held-out text test preserved the signal. Training on one 8192-token stream and evaluating on a reversed 8192-token stream reached aggregate top-16 recall `0.749752` with rank 64 and 256 verifier candidates, and `0.835488` with 512 verifier candidates. The next invention target is sublinear lookup for that learned compact score.

The first learned-score serving attempt tested random-hyperplane LSH over the rank-64 space. It is a kill for that specific lookup geometry. The strongest aggregate row reached only `0.233429` verified top-16 recall while projecting to about `38.6k` average candidates at a million-token context; in the rough few-hundred-candidate band, recall stayed around `0.013`.

Score-aware IVF routing improved the serving shape. Single-write k-means centroids over learned low-rank keys reached `0.234422` recall at about `3.5k` projected million-token candidates, and about `0.095-0.102` recall in the few-hundred-candidate band. That is much better than sign-LSH at the same scale, but still far below the learned ranker's all-key recall. The next target is multi-write or supervised routing: give each key more than one good way to be summoned, or train the catalog cells directly against top-key recall.

Multi-write IVF answered the first half of that branch. Giving each key `2,4,8` nearest-centroid writes modestly improved some local settings, but the best few-hundred to low-thousand candidate row reached `0.105422` recall at about `898` projected million-token candidates, close to single-write IVF's `0.102477` recall at about `783`. The highest-recall multi-write row reached `0.147647` at about `1,564` projected candidates, below single-write IVF's `0.166574` at about `1,666`. The current target is supervised routing or asymmetric compressed scoring: make the catalog optimize for top-key recall directly.

## Files

- `experiments/sva_kill_test.py`: standalone toy benchmark.
- `experiments/sva_causal_sequence_test.py`: incremental causal-cache benchmark.
- `experiments/sva_trainable_recall_test.py`: trainable modern-decoder recall benchmark.
- `experiments/sva_pretrained_socket_test.py`: pretrained SmolLM2 attention-socket benchmark.
- `experiments/sva_real_qk_address_sweep.py`: real-QK high-bit address sweep at the model's configured context window.
- `experiments/sva_million_stream_sim.py`: million-token address-pressure simulation from real SmolLM2 8192-token Q/K samples.
- `experiments/sva_learned_ranker_test.py`: learned compressed Q/K ranker test.
- `experiments/sva_learned_lsh_lookup_test.py`: learned-ranker random-hyperplane LSH serving test.
- `experiments/sva_learned_ivf_lookup_test.py`: learned-ranker IVF/centroid routing serving test.
- `experiments/sva_learned_multiwrite_ivf_lookup_test.py`: learned-ranker multi-write IVF serving test.
- `experiments/sva_address_scaling.py`: address selectivity calculator for long contexts.
- `modal_h100_trainable.py`: Modal H100 runner for the trainable benchmark.
- `modal_h100_socket.py`: Modal H100 runner for the pretrained socket sweep.
- `modal_h100_million_stream.py`: Modal H100 runner for the million-token address-pressure simulation.
- `modal_h100_learned_ranker.py`: Modal H100 runner for the learned compressed-ranker test.
- `modal_h100_learned_ranker_generalize.py`: Modal H100 runner for the held-out-text ranker test.
- `modal_h100_learned_lsh_lookup.py`: Modal H100 runner for learned-ranker LSH serving.
- `modal_h100_learned_ivf_lookup.py`: Modal H100 runner for learned-ranker IVF serving.
- `modal_h100_learned_multiwrite_ivf_lookup.py`: Modal H100 runner for learned-ranker multi-write IVF serving.
- `scripts/start_modal_h100_background.ps1`: detached Modal launcher that writes run logs under `results/modal_runs/`.
- `results/verification_snapshot_2026-05-13.md`: current kill-test results.
- `results/trainable_recall_snapshot_2026-05-13.md`: H100 trainable-representation checkpoint.
- `results/pretrained_socket_snapshot_2026-05-13.md`: SmolLM2 pretrained socket checkpoint.
- `results/pretrained_long_socket_snapshot_2026-05-13.md`: longer-context SmolLM2 socket checkpoint.
- `results/pretrained_prefilter_socket_snapshot_2026-05-13.md`: cheap-prefilter socket checkpoint.
- `results/real_qk_address_8192_snapshot_2026-05-13.md`: SmolLM2 full-window real-QK address sweep.
- `results/million_stream_snapshot_2026-05-13.md`: million-token address-pressure snapshot.
- `results/learned_ranker_snapshot_2026-05-13.md`: learned compressed-ranker snapshot.
- `results/learned_ranker_generalization_snapshot_2026-05-13.md`: held-out-text learned ranker snapshot.
- `results/learned_lsh_lookup_snapshot_2026-05-13.md`: learned-ranker LSH lookup snapshot.
- `results/learned_ivf_lookup_snapshot_2026-05-13.md`: learned-ranker IVF lookup snapshot.
- `results/learned_multiwrite_ivf_lookup_snapshot_2026-05-13.md`: learned-ranker multi-write IVF lookup snapshot.
- `notes/attention_replacement_findings.md`: broader research log leading to SVA.
- `notes/million_token_scaling.md`: scaling target for million-token contexts.

## H100 Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-trainable
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-socket -ModalFile modal_h100_socket.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-million-stream -ModalFile modal_h100_million_stream.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ranker -ModalFile modal_h100_learned_ranker.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ranker-generalize -ModalFile modal_h100_learned_ranker_generalize.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-lsh-lookup -ModalFile modal_h100_learned_lsh_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-ivf-lookup -ModalFile modal_h100_learned_ivf_lookup.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-learned-multiwrite-ivf-lookup -ModalFile modal_h100_learned_multiwrite_ivf_lookup.py
```

The launcher uses `modal run --detach` and writes local metadata, stdout, stderr, and result files under `results/modal_runs/`.

Live progress is visible through Modal logs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_modal_h100.ps1 -Tail 200
```
