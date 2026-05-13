"""Modal H100 runner for high-resolution supervised query-cell router test."""

from __future__ import annotations

import modal


app = modal.App("sva-supervised-query-router-hires-h100")

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
    "--cells",
    "2048,4096",
    "--query-writes",
    "1",
    "--key-writes",
    "1,2,4",
    "--probes",
    "1,2",
    "--budgets",
    "256,512",
    "--topk",
    "16",
    "--train-query-samples",
    "4096",
    "--eval-query-samples",
    "64",
    "--min-query-pos",
    "128",
    "--ranker-steps",
    "160",
    "--ranker-batch-queries",
    "16",
    "--ranker-lr",
    "0.003",
    "--ranker-weight-decay",
    "0.0001",
    "--kmeans-iters",
    "8",
    "--writer-steps",
    "240",
    "--writer-batch-pairs",
    "1024",
    "--writer-negatives",
    "32",
    "--writer-lr",
    "0.002",
    "--writer-weight-decay",
    "0.0001",
    "--target-context",
    "1000000",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=180 * 60)
def run_supervised_query_router_hires() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_supervised_query_router_test.py", *ARGS]
    print("supervised_query_router_hires_start," + " ".join(ARGS), flush=True)
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
        raise RuntimeError(f"high-resolution supervised query router test failed with exit code {return_code}")
    print("supervised_query_router_hires_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_supervised_query_router_hires.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id

