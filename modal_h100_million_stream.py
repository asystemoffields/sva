"""Modal H100 runner for million-token SVA address-pressure simulation."""

from __future__ import annotations

import modal


app = modal.App("sva-million-stream-h100")

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
    "--layers",
    "0,1,5,10,18,24,29",
    "--bits",
    "20,22,24,26",
    "--tables",
    "64,128,256",
    "--radii",
    "1,2",
    "--topk",
    "16",
    "--query-samples",
    "64",
    "--min-query-pos",
    "128",
    "--target-context",
    "1000000",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_million_stream() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_million_stream_sim.py", *ARGS]
    print("million_stream_start," + " ".join(ARGS), flush=True)
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
        raise RuntimeError(f"million stream simulation failed with exit code {return_code}")
    print("million_stream_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_million_stream.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
