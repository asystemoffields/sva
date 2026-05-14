"""Modal H100 runner for long-context SVA recall simulation."""

from __future__ import annotations

import modal


app = modal.App("sva-long-context-recall-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--artifact-dir",
    "results/hf_artifacts/sva-smollm2-135m-2x256-v1",
    "--base-context",
    "8192",
    "--target-contexts",
    "8192,32768,131072,1000000",
    "--layers",
    "0,7,15,23,29",
    "--eval-repeats",
    "320",
    "--query-samples",
    "16",
    "--topk",
    "16",
    "--shortlists",
    "1024,2048",
    "--budgets",
    "256,512",
    "--synthetic-noise-std",
    "0.01",
    "--teacher-chunk-size",
    "65536",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_long_context_recall() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_long_context_recall_sim.py", *ARGS]
    print("long_context_recall_h100_start", flush=True)
    print("command," + " ".join(cmd), flush=True)
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
    print(f"long_context_recall_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Long-context recall benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_long_context_recall.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
