"""Modal H100 runner that exports a strong attention-weighted SVA artifact."""

from __future__ import annotations

import modal


app = modal.App("sva-export-attention-weighted-artifact-h100")
volume = modal.Volume.from_name("sva-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


REMOTE_ARTIFACT_DIR = "/artifacts/sva-smollm2-135m-2x256-attnweighted-v1"

ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--artifact-dir",
    "results/hf_artifacts/sva-smollm2-135m-2x256-v1",
    "--output-dir",
    REMOTE_ARTIFACT_DIR,
    "--profile-name",
    "sva-smollm2-135m-2x256-attnweighted-v1",
    "--calibration-length",
    "32768",
    "--refresh-method",
    "attention_weighted_strong",
    "--attention-topk",
    "16",
    "--calibration-query-samples",
    "128",
    "--attention-boost",
    "4",
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


@app.function(image=image, gpu="H100", timeout=90 * 60, volumes={"/artifacts": volume})
def export_attention_weighted_artifact() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    cmd = [sys.executable, "-u", "experiments/export_refreshed_sva_artifact.py", *ARGS]
    print("attention_weighted_artifact_h100_start", flush=True)
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
    print(f"attention_weighted_artifact_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Attention-weighted SVA artifact export failed with exit code {return_code}")
    volume.commit()
    print(f"attention_weighted_artifact_remote_dir,{REMOTE_ARTIFACT_DIR}", flush=True)
    return "".join(lines)


@app.local_entrypoint()
def main() -> str:
    call = export_attention_weighted_artifact.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    print(f"remote_artifact_dir,{REMOTE_ARTIFACT_DIR}")
    print(
        "download_command,"
        "modal volume get sva-artifacts /sva-smollm2-135m-2x256-attnweighted-v1 "
        "results/hf_artifacts"
    )
    return call.object_id
