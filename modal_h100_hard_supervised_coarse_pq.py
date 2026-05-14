"""Modal H100 runner for hard-negative supervised coarse PQ."""

from __future__ import annotations

import modal


app = modal.App("sva-hard-supervised-coarse-pq-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--max-length",
    "8192",
    "--text-repeats",
    "320",
    "--eval-text-mode",
    "reverse",
    "--layers",
    "0,1,5,10,18,24,29",
    "--fine-rank-dim",
    "64",
    "--coarse-rank-dims",
    "64",
    "--coarse-label-topk",
    "16",
    "--coarse-label-source",
    "attention",
    "--weighted-coarse-boosts",
    "4,16",
    "--weighted-coarse-space",
    "supervised",
    "--coarse-configs",
    "4x64",
    "--fine-configs",
    "16x256",
    "--shortlists",
    "512,768,1024,2048",
    "--budgets",
    "512",
    "--topk",
    "16",
    "--train-query-samples",
    "128",
    "--eval-query-samples",
    "64",
    "--min-query-pos",
    "128",
    "--fine-train-steps",
    "160",
    "--coarse-train-steps",
    "160",
    "--coarse-hard-steps",
    "80",
    "--coarse-hard-pool",
    "1024",
    "--coarse-hard-negatives",
    "64",
    "--coarse-hard-margin",
    "1.0",
    "--coarse-hard-lr-scale",
    "0.5",
    "--batch-queries",
    "16",
    "--lr",
    "0.003",
    "--weight-decay",
    "0.0001",
    "--kmeans-iters",
    "8",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=45 * 60)
def run_hard_supervised_coarse_pq() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_supervised_coarse_pq_test.py", *ARGS]
    print("hard_supervised_coarse_pq_start," + " ".join(ARGS), flush=True)
    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Hard-negative supervised coarse PQ run failed with exit code {return_code}")
    print("hard_supervised_coarse_pq_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_hard_supervised_coarse_pq.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
