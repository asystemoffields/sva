"""Modal H100 runner for exported refreshed-profile recall sanity checks."""

from __future__ import annotations

import modal


app = modal.App("sva-refreshed-profile-recall-h100")

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
    "results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1",
    "--contexts",
    "8192,16384,32768",
    "--calibration-length",
    "32768",
    "--allow-beyond-model-context",
    "--eval-doc-limit",
    "4",
    "--eval-repeats",
    "320",
    "--layers",
    "0,15,29",
    "--variants",
    "artifact_identity,eval_refit_identity",
    "--budgets",
    "512,1024,2048",
    "--topk",
    "16",
    "--query-samples",
    "64",
    "--min-query-pos",
    "128",
    "--kmeans-iters",
    "8",
    "--assign-chunk-size",
    "8192",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_refreshed_profile_recall() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_codebook_refresh_benchmark.py", *ARGS]
    print("refreshed_profile_recall_h100_start", flush=True)
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
    print(f"refreshed_profile_recall_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Refreshed profile recall benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_refreshed_profile_recall.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
