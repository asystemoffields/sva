"""Modal H100 runner for coarse-to-fine PQ lookup test."""

from __future__ import annotations

import modal


app = modal.App("sva-coarse-to-fine-pq-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.45", "huggingface_hub>=0.24", "numpy>=1.26")
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
    "--rank-dim",
    "64",
    "--coarse-configs",
    "4x16,8x16,4x64",
    "--fine-configs",
    "8x256,16x256",
    "--shortlists",
    "1024,2048,4096",
    "--budgets",
    "256,512",
    "--topk",
    "16",
    "--train-query-samples",
    "128",
    "--eval-query-samples",
    "64",
    "--min-query-pos",
    "128",
    "--train-steps",
    "160",
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


@app.function(image=image, gpu="H100", timeout=120 * 60)
def run_coarse_to_fine_pq() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_coarse_to_fine_pq_test.py", *ARGS]
    print("coarse_to_fine_pq_start," + " ".join(ARGS), flush=True)
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
        raise RuntimeError(f"coarse-to-fine PQ test failed with exit code {return_code}")
    print("coarse_to_fine_pq_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_coarse_to_fine_pq.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id

