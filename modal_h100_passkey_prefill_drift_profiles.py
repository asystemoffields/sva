"""Modal H100 runner for passkey prefill-drift profile comparisons."""

from __future__ import annotations

import modal


app = modal.App("sva-passkey-prefill-drift-profiles-h100")
volume = modal.Volume.from_name("sva-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


PROFILES = ";".join(
    [
        "original=results/hf_artifacts/sva-smollm2-135m-2x256-v1",
        "plain_refresh=results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1",
        "attn_boost2=/artifacts/sva-smollm2-135m-2x256-attnweighted-boost2-v1",
        "attn_strong16=results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1",
    ]
)

ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--profiles",
    PROFILES,
    "--contexts",
    "16384,32768",
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


@app.function(image=image, gpu="H100", timeout=90 * 60, volumes={"/artifacts": volume})
def run_passkey_prefill_drift_profiles() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_passkey_prefill_drift_benchmark.py", *ARGS]
    print("passkey_prefill_drift_profiles_h100_start", flush=True)
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
    print(f"passkey_prefill_drift_profiles_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Passkey prefill-drift benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_passkey_prefill_drift_profiles.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
