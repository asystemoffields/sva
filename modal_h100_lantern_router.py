"""Modal H100 runner for supervised Lantern SVA routing."""

from __future__ import annotations

import modal


app = modal.App("sva-lantern-router-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
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
    "512,1024,2048",
    "--writes",
    "2,4,8",
    "--probes",
    "1,2,4",
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
    "--ranker-steps",
    "160",
    "--router-steps",
    "240",
    "--batch-queries",
    "16",
    "--negative-samples",
    "32",
    "--ranker-lr",
    "0.003",
    "--router-lr",
    "0.002",
    "--weight-decay",
    "0.0001",
    "--router-temperature",
    "0.07",
    "--negative-weight",
    "0.5",
    "--balance-samples",
    "1024",
    "--balance-weight",
    "1.0",
    "--key-alignment-weight",
    "0.5",
    "--query-alignment-weight",
    "0.25",
    "--write-candidates",
    "64",
    "--max-load-factor",
    "2.0",
    "--target-context",
    "1000000",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=150 * 60)
def run_lantern_router() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_lantern_router_test.py", *ARGS]
    print("lantern_router_h100_start", flush=True)
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
    print(f"lantern_router_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Lantern router test failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_lantern_router.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
