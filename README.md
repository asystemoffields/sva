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

The next research step is to test SVA in a tiny trainable sequence model where keys and queries are learned rather than synthetic.

## Files

- `experiments/sva_kill_test.py`: standalone toy benchmark.
- `experiments/sva_causal_sequence_test.py`: incremental causal-cache benchmark.
- `experiments/sva_trainable_recall_test.py`: trainable modern-decoder recall benchmark.
- `modal_h100_trainable.py`: Modal H100 runner for the trainable benchmark.
- `scripts/start_modal_h100_background.ps1`: detached Modal launcher that writes run logs under `results/modal_runs/`.
- `results/verification_snapshot_2026-05-13.md`: current kill-test results.
- `notes/attention_replacement_findings.md`: broader research log leading to SVA.

## H100 Run

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-trainable
```

The launcher uses `modal run --detach` and writes local metadata, stdout, stderr, and result files under `results/modal_runs/`.

Live progress is visible through Modal logs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_modal_h100.ps1 -Tail 200
```
