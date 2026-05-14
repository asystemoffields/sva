"""Modal H100 runner for output-distilled socketed SVA adapters."""

from __future__ import annotations

import modal


app = modal.App("sva-output-distill-socket-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--text-repeats",
    "160",
    "--max-length",
    "1024",
    "--route-source",
    "qk",
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
    "120",
    "--coarse-hard-steps",
    "60",
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
    "--adapter-rank",
    "16",
    "--adapter-source",
    "both",
    "--adapter-scale",
    "1.0",
    "--distill-steps",
    "60",
    "--adapter-lr",
    "0.001",
    "--temperature",
    "1.0",
    "--grad-clip",
    "1.0",
    "--log-every",
    "10",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


CASES = [
    ("early26_teacher", "0-25", "teacher"),
    ("early26_progressive", "0-25", "progressive"),
    ("all30_teacher", "", "teacher"),
]


@app.function(image=image, gpu="H100", timeout=180 * 60)
def run_output_distill_socket() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    all_output: list[str] = []
    for index, (label, layers, artifact_training) in enumerate(CASES, start=1):
        args = [
            "--socket-layers",
            layers,
            "--artifact-training",
            artifact_training,
        ]
        cmd = [sys.executable, "-u", "experiments/sva_output_distill_socket_test.py", *BASE_ARGS, *args]
        print(f"output_distill_start,{index},{label},{artifact_training},{layers or 'all'}", flush=True)
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
        all_output.append(f"# output_distill {index} {label} {artifact_training} {layers or 'all'}\n" + "".join(lines))
        if return_code != 0:
            print(f"output_distill_failed,{index},{label},{return_code}", flush=True)
            all_output.append(f"\noutput_distill_failed,{index},{label},{return_code}\n")
            continue
        print(f"output_distill_done,{index},{label}", flush=True)
    return "\n".join(all_output)


@app.local_entrypoint()
def main() -> str:
    call = run_output_distill_socket.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
