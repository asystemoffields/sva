"""Modal H100 runner for full answer-decode validation of the answer-distilled late4 adapter."""

from __future__ import annotations

import modal


app = modal.App("sva-late4-answerdistill-adapter-answer-h100")
volume = modal.Volume.from_name("sva-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


REMOTE_ADAPTER_DIR = "/artifacts/sva-late4-512x128-answerdistill-v1"

ARGS = [
    "--adapter-dir",
    REMOTE_ADAPTER_DIR,
    "--contexts",
    "32768",
    "--keys",
    "731942,184207,905613",
    "--placements",
    "start,middle,end",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=120 * 60, volumes={"/artifacts": volume})
def run_late4_answerdistill_adapter_answer() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/sva_late4_adapter_answer_benchmark.py", *ARGS]
    print("late4_answerdistill_adapter_answer_h100_start", flush=True)
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
    print(f"late4_answerdistill_adapter_answer_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Late4 answer-distilled adapter answer benchmark failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_late4_answerdistill_adapter_answer.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    print(f"remote_adapter_dir,{REMOTE_ADAPTER_DIR}")
    return call.object_id
