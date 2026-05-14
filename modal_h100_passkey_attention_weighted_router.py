"""Modal H100 runner for passkey routing with an attention-weighted long profile."""

from __future__ import annotations

import modal


app = modal.App("sva-passkey-attention-weighted-router-h100")

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
    "--long-artifact-dir",
    "results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1",
    "--long-artifact-min-context",
    "16384",
    "--contexts",
    "8192,16384,32768",
    "--teacher-context-max",
    "32768",
    "--placements",
    "start",
    "--key",
    "731942",
    "--shortlist",
    "8192",
    "--budget",
    "2048",
    "--query-chunk-size",
    "128",
    "--summon-mode",
    "scan",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_passkey_attention_weighted_router() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_passkey_language_benchmark.py", *ARGS]
    print("passkey_attention_weighted_router_h100_start", flush=True)
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
    print(f"passkey_attention_weighted_router_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Passkey attention-weighted router benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_passkey_attention_weighted_router.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
