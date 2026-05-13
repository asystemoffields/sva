"""Modal H100 runner for the trainable SVA recall benchmark."""

from __future__ import annotations

import modal


app = modal.App("sva-trainable-recall-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "torch>=2.0")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


DEFAULT_ARGS = [
    "--steps",
    "2500",
    "--log-every",
    "250",
    "--batch-size",
    "256",
    "--eval-batch-size",
    "64",
    "--eval-batches",
    "8",
    "--n-pairs",
    "32",
    "--n-keys",
    "128",
    "--n-values",
    "128",
    "--d-model",
    "128",
    "--n-heads",
    "4",
    "--n-layers",
    "2",
    "--sva-tables",
    "8",
    "16",
    "24",
    "32",
    "--sva-bits",
    "10",
    "--sva-budget",
    "16",
    "--probe-radius",
    "1",
    "--lr",
    "4e-4",
]


@app.function(image=image, gpu="H100", timeout=60 * 60)
def run_trainable_recall(extra_args: list[str] | None = None) -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    args = DEFAULT_ARGS if extra_args is None else extra_args
    cmd = [sys.executable, "-u", "experiments/sva_trainable_recall_test.py", *args]
    print("command," + " ".join(cmd), flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )
    lines = []
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"trainable recall run failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = run_trainable_recall.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
