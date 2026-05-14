"""Modal H100 runner for focused posting-list inverted SVA decode."""

from __future__ import annotations

import modal


app = modal.App("sva-inverted-posting-decode-h100")

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
    "--context-length",
    "8192",
    "--eval-repeats",
    "320",
    "--eval-doc-limit",
    "2",
    "--shortlist",
    "2048",
    "--max-budget",
    "512",
    "--scan-budget",
    "512",
    "--adaptive-min-budgets",
    "128,256",
    "--adaptive-mid-budget",
    "256",
    "--cells-per-subspace",
    "16,32",
    "--repeats",
    "10",
    "--warmup",
    "2",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=60 * 60)
def run_inverted_posting_decode() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_inverted_adaptive_decode_benchmark.py", *ARGS]
    print("inverted_posting_decode_h100_start", flush=True)
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
    print(f"inverted_posting_decode_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Inverted posting decode benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_inverted_posting_decode.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
