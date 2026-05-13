# Compute Runbook

Date: 2026-05-13

## Hardware Fit Rule

Use H100/A100 when the hot path is dense tensor work:

- large matmuls
- fused attention kernels
- torch compile or Triton kernels
- high GPU memory use
- sustained high GPU utilization

Use high-clock CPU or many CPU workers when the hot path is algorithmic:

- Python loops over tokens, heads, tables, or candidates
- hash-table lookup
- per-query branching
- small scattered tensor ops
- low GPU memory use with modest GPU utilization

The trainable SVA H100 run revealed this sharply. Training a tiny modern decoder belongs on GPU, but the current SVA replacement eval uses Python loops over candidates. That makes the full job CPU/Python-bound during the most important measurement.

## Before Spending On A Long Run

Run a short profiling pilot first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_modal_h100_background.ps1 -Name sva-h100-pilot
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_modal_h100.ps1 -Tail 200
```

During the pilot, check:

- Does it emit progress every few minutes?
- Is GPU memory meaningfully occupied?
- Is GPU utilization high and sustained?
- Is the Python process burning one CPU core?
- Is the slow phase training or evaluation?

## Current SVA Implication

The next H100-worthy SVA run should first move the SVA eval hot path out of Python. Good options:

- vectorize candidate generation for fixed-size sequence windows
- implement a torch/Triton candidate verifier
- separate training from SVA eval, then run SVA eval with smaller batches and streaming progress
- run algorithmic SVA lookup sweeps on CPU while reserving GPU for dense training

Until then, H100 can answer the full-attention training part quickly, but it will not automatically accelerate the Python hash-table retrieval path.
