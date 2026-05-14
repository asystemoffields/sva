"""Modal H100 runner for tight-budget late4 logit distillation."""

from __future__ import annotations

import modal


app = modal.App("sva-late4-logit-distill-h100")

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
    "results/hf_artifacts/sva-smollm2-135m-2x256-attnweighted-v1",
    "--contexts",
    "32768",
    "--train-keys",
    "731942,184029",
    "--eval-keys",
    "905317",
    "--train-placements",
    "start,middle",
    "--eval-placements",
    "start,middle,end",
    "--socket-layers",
    "26-29",
    "--shortlist",
    "512",
    "--budget",
    "128",
    "--query-chunk-size",
    "128",
    "--summon-mode",
    "scan",
    "--adapter-rank",
    "16",
    "--distill-steps",
    "24",
    "--lr",
    "0.001",
    "--temperature",
    "1.0",
    "--grad-clip",
    "1.0",
    "--log-every",
    "4",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=120 * 60)
def run_late4_logit_distill() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_late4_logit_distill.py", *ARGS]
    print("late4_logit_distill_h100_start", flush=True)
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
    print(f"late4_logit_distill_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Late4 logit distillation failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_late4_logit_distill.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
