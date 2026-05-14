"""Modal H100 runner for layer-isolated three-stage SVA socket tests."""

from __future__ import annotations

import modal


app = modal.App("sva-three-stage-socket-layers-h100")

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
    "12",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


SWEEP = [
    ["--socket-layers", "0"],
    ["--socket-layers", "10"],
    ["--socket-layers", "18"],
    ["--socket-layers", "29"],
    ["--socket-layers", "0,1,2,3"],
    ["--socket-layers", "0,5,10,18,24,29"],
    [],
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_three_stage_socket_layers() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    all_output: list[str] = []
    for index, args in enumerate(SWEEP, start=1):
        cmd = [sys.executable, "-u", "experiments/sva_pretrained_socket_test.py", *BASE_ARGS, *args]
        label = " ".join(args) if args else "--socket-layers all"
        print(f"three_stage_socket_layers_start,{index},{label}", flush=True)
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
        if return_code != 0:
            raise RuntimeError(f"three-stage socket layer item {index} failed with exit code {return_code}")
        all_output.append(f"# sweep_item {index} {label}\n" + "".join(lines))
        print(f"three_stage_socket_layers_done,{index}", flush=True)
    return "\n".join(all_output)


@app.local_entrypoint()
def main() -> str:
    call = run_three_stage_socket_layers.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
