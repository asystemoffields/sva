"""Modal H100 runner for passkey key-survival comparisons across SVA profiles."""

from __future__ import annotations

import modal


app = modal.App("sva-passkey-key-survival-profiles-h100")
volume = modal.Volume.from_name("sva-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


LAYERS = ",".join(str(layer_idx) for layer_idx in range(30))

PROFILES = [
    ("original", "results/hf_artifacts/sva-smollm2-135m-2x256-v1"),
    ("plain_refresh", "results/hf_artifacts/sva-smollm2-135m-2x256-longctx-refresh-v1"),
    ("attn_boost2", "/artifacts/sva-smollm2-135m-2x256-attnweighted-boost2-v1"),
    ("attn_strong16", "results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1"),
]

BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--contexts",
    "16384,32768",
    "--placements",
    "start",
    "--layers",
    LAYERS,
    "--key",
    "731942",
    "--shortlist",
    "8192",
    "--budget",
    "2048",
    "--anchor-counts",
    "1",
    "--rerank-modes",
    "current",
    "--expand-radii",
    "0",
    "--topk",
    "16",
    "--assign-chunk-size",
    "8192",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60, volumes={"/artifacts": volume})
def run_passkey_key_survival_profiles() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    print("passkey_key_survival_profiles_h100_start", flush=True)
    lines: list[str] = []
    for profile_name, artifact_dir in PROFILES:
        print(f"profile_start,name={profile_name},artifact_dir={artifact_dir}", flush=True)
        cmd = [
            sys.executable,
            "-u",
            "experiments/sva_evidence_haystack_benchmark.py",
            *BASE_ARGS,
            "--artifact-dir",
            artifact_dir,
        ]
        print("command," + " ".join(cmd), flush=True)
        process = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                lines.append(f"profile={profile_name}," + line)
        return_code = process.wait()
        print(f"profile_exit,name={profile_name},exit={return_code}", flush=True)
        if return_code != 0:
            raise RuntimeError(f"Passkey key-survival profile {profile_name} failed with exit code {return_code}")
    print("passkey_key_survival_profiles_h100_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_passkey_key_survival_profiles.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
