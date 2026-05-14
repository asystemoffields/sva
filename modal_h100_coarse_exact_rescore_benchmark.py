"""Modal H100 runner for coarse PQ plus exact low-rank rescore benchmark."""

from __future__ import annotations

import modal


app = modal.App("sva-coarse-exact-rescore-benchmark-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy>=1.26")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


ARGS = [
    "--context",
    "1000000",
    "--heads",
    "9",
    "--queries",
    "1,4",
    "--rank-dims",
    "64",
    "--coarse-configs",
    "4x64",
    "--shortlists",
    "1024,1536,2048,4096",
    "--budgets",
    "512",
    "--warmup",
    "3",
    "--repeats",
    "10",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=30 * 60)
def run_coarse_exact_rescore_benchmark() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_coarse_exact_rescore_benchmark.py", *ARGS]
    print("coarse_exact_rescore_benchmark_start," + " ".join(ARGS), flush=True)
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
        raise RuntimeError(f"Coarse exact-rescore benchmark failed with exit code {return_code}")
    print("coarse_exact_rescore_benchmark_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_coarse_exact_rescore_benchmark.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
