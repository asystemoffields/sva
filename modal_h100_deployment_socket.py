"""Modal H100 runner for deployment-proxy pretrained SVA socket tests."""

from __future__ import annotations

import modal


app = modal.App("sva-deployment-socket-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--text-repeats",
    "320",
    "--max-length",
    "2048",
    "--route-source",
    "qk",
    "--artifact-training",
    "teacher",
    "--rank-dim",
    "64",
    "--coarse-subspaces",
    "4",
    "--coarse-codewords",
    "64",
    "--coarse-label-topk",
    "16",
    "--train-query-samples",
    "128",
    "--min-query-pos",
    "128",
    "--ranker-train-steps",
    "160",
    "--coarse-hard-steps",
    "80",
    "--coarse-hard-pool",
    "512",
    "--coarse-hard-negatives",
    "64",
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
    "--coarse-shortlist",
    "1024",
    "--budget",
    "512",
    "--diagnose-topk",
    "16",
    "--head-report-limit",
    "8",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


CASES = [
    ("transductive_base", "base", "base"),
    ("deploy_rotate", "base", "rotate"),
    ("deploy_reverse", "base", "reverse"),
    ("deploy_odds_evens", "base", "odds_evens"),
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_deployment_socket() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    all_output: list[str] = []
    for index, (label, train_mode, eval_mode) in enumerate(CASES, start=1):
        args = ["--train-text-mode", train_mode, "--eval-text-mode", eval_mode]
        cmd = [sys.executable, "-u", "experiments/sva_deployment_socket_test.py", *BASE_ARGS, *args]
        print(f"deployment_socket_start,{index},{label},{train_mode},{eval_mode}", flush=True)
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
        all_output.append(f"# deployment {index} {label} {train_mode} {eval_mode}\n" + "".join(lines))
        if return_code != 0:
            print(f"deployment_socket_failed,{index},{label},{return_code}", flush=True)
            all_output.append(f"\ndeployment_failed,{index},{label},{return_code}\n")
            continue
        print(f"deployment_socket_done,{index},{label}", flush=True)
    return "\n".join(all_output)


@app.local_entrypoint()
def main() -> str:
    call = run_deployment_socket.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
