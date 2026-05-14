"""Modal H100 runner for the full held-out SVA deployment benchmark."""

from __future__ import annotations

import modal


app = modal.App("sva-full-deployment-benchmark-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--calibration-repeats",
    "160",
    "--eval-repeats",
    "160",
    "--eval-doc-limit",
    "4",
    "--context-lengths",
    "2048,4096",
    "--calibration-length",
    "4096",
    "--route-source",
    "qk",
    "--artifact-training",
    "teacher",
    "--rank-dim",
    "64",
    "--coarse-subspaces",
    "4",
    "--coarse-codewords",
    "64",
    "--coarse-label-topk",
    "16",
    "--train-query-samples",
    "192",
    "--min-query-pos",
    "128",
    "--ranker-train-steps",
    "200",
    "--coarse-hard-steps",
    "100",
    "--coarse-hard-pool",
    "512",
    "--coarse-hard-negatives",
    "64",
    "--coarse-hard-margin",
    "1.0",
    "--coarse-hard-lr-scale",
    "0.5",
    "--weighted-boost",
    "4",
    "--batch-queries",
    "16",
    "--ranker-lr",
    "0.003",
    "--ranker-weight-decay",
    "0.0001",
    "--kmeans-iters",
    "8",
    "--coarse-shortlists",
    "512,1024",
    "--budgets",
    "128,256,512",
    "--diagnose-topk",
    "16",
    "--head-report-limit",
    "0",
    "--timing-repeats",
    "1",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=120 * 60)
def run_full_deployment_benchmark() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_full_deployment_benchmark.py", *BASE_ARGS]
    print("full_deployment_benchmark_start", flush=True)
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
    print(f"full_deployment_benchmark_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"full deployment benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_full_deployment_benchmark.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
