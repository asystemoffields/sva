"""Modal H100 runner for late4 passkey budget-squeeze sweeps."""

from __future__ import annotations

import modal


app = modal.App("sva-late4-budget-sweep-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


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
    "32768",
    "--teacher-context-max",
    "32768",
    "--placements",
    "start",
    "--key",
    "731942",
    "--socket-layers",
    "26,27,28,29",
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


CASES = [
    ("8192_2048", 8192, 2048),
    ("4096_1024", 4096, 1024),
    ("2048_512", 2048, 512),
    ("1024_256", 1024, 256),
    ("512_128", 512, 128),
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_late4_budget_sweep() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    print("late4_budget_sweep_h100_start", flush=True)
    lines: list[str] = []
    for label, shortlist, budget in CASES:
        print(f"budget_case_start,label={label},shortlist={shortlist},budget={budget}", flush=True)
        cmd = [
            sys.executable,
            "-u",
            "experiments/sva_passkey_language_benchmark.py",
            *BASE_ARGS,
            "--shortlist",
            str(shortlist),
            "--budget",
            str(budget),
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
                lines.append(f"budget_case={label}," + line)
        return_code = process.wait()
        print(f"budget_case_exit,label={label},exit={return_code}", flush=True)
        if return_code != 0:
            raise RuntimeError(f"Budget case {label} failed with exit code {return_code}")
    print("late4_budget_sweep_h100_done", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_late4_budget_sweep.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
