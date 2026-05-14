"""Modal H100 runner for the tight-shortlist SVA quality/speed frontier."""

from __future__ import annotations

import modal


app = modal.App("sva-tight-summon-frontier-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


QUALITY_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--calibration-repeats",
    "320",
    "--eval-repeats",
    "320",
    "--eval-doc-limit",
    "4",
    "--context-length",
    "8192",
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
    "384",
    "--min-query-pos",
    "128",
    "--ranker-train-steps",
    "280",
    "--coarse-hard-steps",
    "160",
    "--coarse-hard-pool",
    "512",
    "--coarse-hard-negatives",
    "96",
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
    "256,384,512,768,1024,1536,2048",
    "--budgets",
    "64,128,256,384,512",
    "--quality-query-samples",
    "192",
    "--timing-query-counts",
    "",
    "--warmup",
    "0",
    "--repeats",
    "1",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


SPEED_ARGS = [
    "--contexts",
    "1000000",
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
    "512,768,1024,1536,2048",
    "--budgets",
    "128,256,384,512",
    "--variants",
    "full,sva_vectorized",
    "--warmup",
    "3",
    "--repeats",
    "10",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


def run_command(label: str, args: list[str]) -> str:
    import os
    import subprocess
    import sys

    cmd = [sys.executable, "-u", *args]
    print(f"{label}_start", flush=True)
    print(f"{label}_command," + " ".join(cmd), flush=True)
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
    print(f"{label}_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}")
    return "".join(lines)


@app.function(image=image, gpu="H100", timeout=180 * 60)
def run_tight_summon_frontier() -> str:
    import os

    os.chdir("/root/sva")
    quality = run_command(
        "tight_summon_quality",
        ["experiments/sva_cached_decode_benchmark.py", *QUALITY_ARGS],
    )
    speed = run_command(
        "tight_summon_speed",
        ["experiments/sva_million_cached_decode_benchmark.py", *SPEED_ARGS],
    )
    return quality + "\n" + speed


@app.local_entrypoint()
def main() -> str:
    call = run_tight_summon_frontier.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
