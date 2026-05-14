"""Modal H100 runner for mapping the selective SVA socket cliff."""

from __future__ import annotations

import modal


app = modal.App("sva-layer-cliff-h100")

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


BASE_15 = "0,1,3,4,5,6,7,8,9,10,13,15,17,18,21"

CASES = [
    ("base_15_teacher", BASE_15, "teacher"),
    ("add_14_teacher", f"{BASE_15},14", "teacher"),
    ("add_16_teacher", f"{BASE_15},16", "teacher"),
    ("add_19_teacher", f"{BASE_15},19", "teacher"),
    ("add_20_teacher", f"{BASE_15},20", "teacher"),
    ("add_23_teacher", f"{BASE_15},23", "teacher"),
    ("add_14_16_teacher", f"{BASE_15},14,16", "teacher"),
    ("add_19_20_teacher", f"{BASE_15},19,20", "teacher"),
    ("frontier_20_teacher", f"{BASE_15},14,16,19,20,23", "teacher"),
    ("add_14_16_progressive", f"{BASE_15},14,16", "progressive"),
    ("add_19_20_progressive", f"{BASE_15},19,20", "progressive"),
    ("frontier_20_progressive", f"{BASE_15},14,16,19,20,23", "progressive"),
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_layer_cliff() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    all_output: list[str] = []
    for index, (label, layers, artifact_training) in enumerate(CASES, start=1):
        args = [
            "--artifact-training",
            artifact_training,
            "--socket-layers",
            layers,
        ]
        cmd = [sys.executable, "-u", "experiments/sva_pretrained_socket_test.py", *BASE_ARGS, *args]
        print(f"layer_cliff_start,{index},{label},{artifact_training},{layers}", flush=True)
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
        all_output.append(f"# cliff {index} {label} {artifact_training} {layers}\n" + "".join(lines))
        if return_code != 0:
            print(f"layer_cliff_failed,{index},{label},{return_code}", flush=True)
            all_output.append(f"\ncliff_failed,{index},{label},{return_code}\n")
            continue
        print(f"layer_cliff_done,{index},{label}", flush=True)
    return "\n".join(all_output)


@app.local_entrypoint()
def main() -> str:
    call = run_layer_cliff.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
