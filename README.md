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

## Files

- `experiments/sva_kill_test.py`: standalone toy benchmark.
- `experiments/sva_causal_sequence_test.py`: incremental causal-cache benchmark.
- `experiments/sva_trainable_recall_test.py`: trainable modern-decoder recall benchmark.
- `experiments/sva_pretrained_socket_test.py`: pretrained SmolLM2 attention-socket benchmark.
- `experiments/sva_address_scaling.py`: address selectivity calculator for long contexts.
- `modal_h100_trainable.py`: Modal H100 runner for the trainable benchmark.
- `modal_h100_socket.py`: Modal H100 runner for the pretrained socket sweep.
- `scripts/start_modal_h100_background.ps1`: detached Modal launcher that writes run logs under `results/modal_runs/`.
- `results/verification_snapshot_2026-05-13.md`: current kill-test results.
- `results/trainable_recall_snapshot_2026-05-13.md`: H100 trainable-representation checkpoint.
- `results/pretrained_socket_snapshot_2026-05-13.md`: SmolLM2 pretrained socket checkpoint.
- `results/pretrained_long_socket_snapshot_2026-05-13.md`: longer-context SmolLM2 socket checkpoint.
- `results/pretrained_prefilter_socket_snapshot_2026-05-13.md`: cheap-prefilter socket checkpoint.
- `notes/attention_replacement_findings.md`: broader research log leading to SVA.
- `notes/million_token_scaling.md`: scaling target for million-token contexts.

## H100 Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-trainable
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-socket -ModalFile modal_h100_socket.py
```

The launcher uses `modal run --detach` and writes local metadata, stdout, stderr, and result files under `results/modal_runs/`.

Live progress is visible through Modal logs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_modal_h100.ps1 -Tail 200
```
