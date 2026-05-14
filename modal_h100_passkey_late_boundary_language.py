"""Modal H100 runner for late-layer passkey answer boundary sweeps."""

from __future__ import annotations

import modal


app = modal.App("sva-passkey-late-boundary-language-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


GROUPS = [
    ("late4", "26,27,28,29"),
    ("late6", "24,25,26,27,28,29"),
    ("late8", "22,23,24,25,26,27,28,29"),
    ("late10", "20,21,22,23,24,25,26,27,28,29"),
    ("late12", "18,19,20,21,22,23,24,25,26,27,28,29"),
    ("late15", "15,16,17,18,19,20,21,22,23,24,25,26,27,28,29"),
]

BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--artifact-dir",
    "results/hf_artifacts/sva-smollm2-135m-2x256-v1",
    "--long-artifact-dir",
    "results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1",
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


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_passkey_late_boundary_language() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    print("passkey_late_boundary_language_h100_start", flush=True)
    lines: list[str] = []
    for group_name, socket_layers in GROUPS:
        print(f"layer_group_start,name={group_name},socket_layers={socket_layers}", flush=True)
        cmd = [
            sys.executable,
            "-u",
            "experiments/sva_passkey_language_benchmark.py",
            *BASE_ARGS,
            "--socket-layers",
            socket_layers,
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
                lines.append(f"layer_group={group_name}," + line)
        return_code = process.wait()
        print(f"layer_group_exit,name={group_name},exit={return_code}", flush=True)
        if return_code != 0:
            raise RuntimeError(f"Layer group {group_name} failed with exit code {return_code}")
    print("passkey_late_boundary_language_h100_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_passkey_late_boundary_language.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
