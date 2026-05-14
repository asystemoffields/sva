"""Modal H100 runner for the selective SVA socket layer frontier."""

from __future__ import annotations

import modal


app = modal.App("sva-layer-frontier-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--mode",
    "three_stage",
    "--route-source",
    "qk",
    "--artifact-training",
    "teacher",
    "--long-texts",
    "--n-texts",
    "1",
    "--text-repeats",
    "320",
    "--max-length",
    "2048",
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


FRONTIER = [
    ("frontier_6", "0,1,3,4,7,10"),
    ("frontier_9", "0,1,3,4,5,6,7,8,10"),
    ("frontier_12", "0,1,3,4,5,6,7,8,9,10,15,18"),
    ("frontier_15", "0,1,3,4,5,6,7,8,9,10,13,15,17,18,21"),
    ("frontier_20", "0,1,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,23"),
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_layer_frontier() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    all_output: list[str] = []
    for index, (label, layers) in enumerate(FRONTIER, start=1):
        args = ["--socket-layers", layers]
        cmd = [sys.executable, "-u", "experiments/sva_pretrained_socket_test.py", *BASE_ARGS, *args]
        print(f"layer_frontier_start,{index},{label},{layers}", flush=True)
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
        all_output.append(f"# frontier {index} {label} {layers}\n" + "".join(lines))
        if return_code != 0:
            print(f"layer_frontier_failed,{index},{label},{return_code}", flush=True)
            all_output.append(f"\nfrontier_failed,{index},{label},{return_code}\n")
            continue
        print(f"layer_frontier_done,{index},{label}", flush=True)
    return "\n".join(all_output)


@app.local_entrypoint()
def main() -> str:
    call = run_layer_frontier.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
