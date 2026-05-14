"""Modal H100 runner that exports a portable SVA artifact bundle."""

from __future__ import annotations

import modal


app = modal.App("sva-export-artifact-h100")
volume = modal.Volume.from_name("sva-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


REMOTE_ARTIFACT_DIR = "/artifacts/sva-smollm2-135m-2x256-v1"

ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--output-dir",
    REMOTE_ARTIFACT_DIR,
    "--profile-name",
    "sva-smollm2-135m-2x256-v1",
    "--calibration-repeats",
    "320",
    "--context-length",
    "8192",
    "--route-source",
    "qk",
    "--artifact-training",
    "teacher",
    "--rank-dim",
    "64",
    "--coarse-subspaces",
    "2",
    "--coarse-codewords",
    "256",
    "--coarse-shortlist",
    "2048",
    "--default-budget",
    "512",
    "--coarse-label-topk",
    "16",
    "--train-query-samples",
    "384",
    "--min-query-pos",
    "128",
    "--ranker-train-steps",
    "280",
    "--coarse-hard-steps",
    "160",
    "--coarse-hard-pool",
    "512",
    "--coarse-hard-negatives",
    "96",
    "--coarse-hard-margin",
    "1.0",
    "--coarse-hard-lr-scale",
    "0.5",
    "--weighted-boost",
    "4",
    "--batch-queries",
    "16",
    "--ranker-lr",
    "0.003",
    "--ranker-weight-decay",
    "0.0001",
    "--kmeans-iters",
    "8",
    "--device",
    "cuda",
    "--model-dtype",
    "bfloat16",
    "--artifact-dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60, volumes={"/artifacts": volume})
def export_sva_artifact() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/export_sva_artifact.py", *ARGS]
    print("sva_artifact_export_start", flush=True)
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
    print(f"sva_artifact_export_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"SVA artifact export failed with exit code {return_code}")
    volume.commit()
    print(f"sva_artifact_remote_dir,{REMOTE_ARTIFACT_DIR}", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = export_sva_artifact.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    print(f"remote_artifact_dir,{REMOTE_ARTIFACT_DIR}")
    print("download_command,modal volume get sva-artifacts /sva-smollm2-135m-2x256-v1 results/hf_artifacts/sva-smollm2-135m-2x256-v1")
    return call.object_id
