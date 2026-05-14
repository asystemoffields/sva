"""Modal H100 runner for an attention-weighted profile-strength sweep."""

from __future__ import annotations

import modal


app = modal.App("sva-attention-weighted-router-sweep-h100")
volume = modal.Volume.from_name("sva-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


PROFILES = [
    {
        "name": "boost1",
        "refresh_method": "attention_weighted",
        "attention_boost": "1",
        "remote_dir": "/artifacts/sva-smollm2-135m-2x256-attnweighted-boost1-v1",
        "profile_name": "sva-smollm2-135m-2x256-attnweighted-boost1-v1",
    },
    {
        "name": "boost2",
        "refresh_method": "attention_weighted",
        "attention_boost": "2",
        "remote_dir": "/artifacts/sva-smollm2-135m-2x256-attnweighted-boost2-v1",
        "profile_name": "sva-smollm2-135m-2x256-attnweighted-boost2-v1",
    },
    {
        "name": "boost4",
        "refresh_method": "attention_weighted",
        "attention_boost": "4",
        "remote_dir": "/artifacts/sva-smollm2-135m-2x256-attnweighted-boost4-v1",
        "profile_name": "sva-smollm2-135m-2x256-attnweighted-boost4-v1",
    },
]


EXPORT_BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--artifact-dir",
    "results/hf_artifacts/sva-smollm2-135m-2x256-v1",
    "--calibration-length",
    "32768",
    "--attention-topk",
    "16",
    "--calibration-query-samples",
    "128",
    "--allow-beyond-model-context",
    "--calibration-doc-limit",
    "6",
    "--calibration-repeats",
    "320",
    "--kmeans-iters",
    "8",
    "--assign-chunk-size",
    "8192",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--model-dtype",
    "bfloat16",
    "--artifact-dtype",
    "bfloat16",
]


PASSKEY_BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--artifact-dir",
    "results/hf_artifacts/sva-smollm2-135m-2x256-v1",
    "--long-artifact-min-context",
    "16384",
    "--contexts",
    "16384,32768",
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


def run_streaming(cmd: list[str], failure_label: str) -> str:
    import os
    import subprocess

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
    print(f"{failure_label}_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"{failure_label} failed with exit code {return_code}")
    return "".join(lines)


@app.function(image=image, gpu="H100", timeout=90 * 60, volumes={"/artifacts": volume})
def run_attention_weighted_router_sweep() -> str:
    import os
    import sys

    os.chdir("/root/sva")
    print("attention_weighted_router_sweep_h100_start", flush=True)
    all_lines: list[str] = []

    for profile in PROFILES:
        print(
            "sweep_profile_start,"
            f"name={profile['name']},refresh_method={profile['refresh_method']},"
            f"attention_boost={profile['attention_boost']}",
            flush=True,
        )
        export_cmd = [
            sys.executable,
            "-u",
            "experiments/export_refreshed_sva_artifact.py",
            *EXPORT_BASE_ARGS,
            "--output-dir",
            profile["remote_dir"],
            "--profile-name",
            profile["profile_name"],
            "--refresh-method",
            profile["refresh_method"],
            "--attention-boost",
            profile["attention_boost"],
        ]
        all_lines.append(run_streaming(export_cmd, f"export_{profile['name']}"))
        volume.commit()

        passkey_cmd = [
            sys.executable,
            "-u",
            "experiments/sva_passkey_language_benchmark.py",
            *PASSKEY_BASE_ARGS,
            "--long-artifact-dir",
            profile["remote_dir"],
        ]
        all_lines.append(run_streaming(passkey_cmd, f"passkey_{profile['name']}"))
        print(f"sweep_profile_done,name={profile['name']},remote_dir={profile['remote_dir']}", flush=True)

    print("attention_weighted_router_sweep_h100_done", flush=True)
    return "".join(all_lines)


@app.local_entrypoint()
def main() -> str:
    call = run_attention_weighted_router_sweep.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    print(
        "download_commands,"
        "modal volume get sva-artifacts /sva-smollm2-135m-2x256-attnweighted-boost1-v1 results/hf_artifacts ; "
        "modal volume get sva-artifacts /sva-smollm2-135m-2x256-attnweighted-boost2-v1 results/hf_artifacts ; "
        "modal volume get sva-artifacts /sva-smollm2-135m-2x256-attnweighted-boost4-v1 results/hf_artifacts"
    )
    return call.object_id
