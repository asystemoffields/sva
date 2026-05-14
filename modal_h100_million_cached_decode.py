"""Modal H100 runner for synthetic million-token cached SVA decode."""

from __future__ import annotations

import modal


app = modal.App("sva-million-cached-decode-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy>=1.26")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


ARGS = [
    "--contexts",
    "8192,65536,262144,1000000",
    "--heads",
    "9",
    "--queries",
    "1,4,16",
    "--head-dim",
    "64",
    "--rank-dim",
    "64",
    "--coarse-subspaces",
    "4",
    "--coarse-codewords",
    "64",
    "--shortlists",
    "1024,2048",
    "--budgets",
    "256,512",
    "--variants",
    "full,sva_loop,sva_vectorized",
    "--warmup",
    "3",
    "--repeats",
    "10",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_million_cached_decode() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_million_cached_decode_benchmark.py", *ARGS]
    print("million_cached_decode_h100_start", flush=True)
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
    print(f"million_cached_decode_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"million cached decode benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_million_cached_decode.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
