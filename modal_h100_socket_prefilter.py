"""Modal H100 runner for cheap-prefilter SVA socket sweeps."""

from __future__ import annotations

import modal


app = modal.App("sva-pretrained-socket-prefilter-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.45", "huggingface_hub>=0.24")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
)


CONTEXT_LENGTHS = [256, 512]

CONFIGS = [
    ["--prefilter-dim", "0", "--prefilter-budget", "0", "--budget", "32"],
    ["--prefilter-dim", "8", "--prefilter-budget", "64", "--budget", "32"],
    ["--prefilter-dim", "16", "--prefilter-budget", "32", "--budget", "32"],
    ["--prefilter-dim", "16", "--prefilter-budget", "64", "--budget", "32"],
    ["--prefilter-dim", "32", "--prefilter-budget", "32", "--budget", "32"],
    ["--prefilter-dim", "32", "--prefilter-budget", "64", "--budget", "32"],
    ["--prefilter-dim", "32", "--prefilter-budget", "96", "--budget", "32"],
    ["--prefilter-dim", "48", "--prefilter-budget", "64", "--budget", "32"],
]


BASE_ARGS = [
    "--model-id",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "--long-texts",
    "--n-texts",
    "3",
    "--text-repeats",
    "8",
    "--tables",
    "64",
    "--bits",
    "10",
    "--probe-radius",
    "2",
    "--diagnose-topk",
    "16",
    "--head-report-limit",
    "6",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


@app.function(image=image, gpu="H100", timeout=90 * 60)
def run_socket_sweep() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    all_output: list[str] = []
    sweep_items = [
        ["--max-length", str(length), *config]
        for length in CONTEXT_LENGTHS
        for config in CONFIGS
    ]
    for index, args in enumerate(sweep_items, start=1):
        cmd = [sys.executable, "-u", "experiments/sva_pretrained_socket_test.py", *BASE_ARGS, *args]
        print(f"sweep_start,{index},{len(sweep_items)}," + " ".join(args), flush=True)
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
            raise RuntimeError(f"prefilter sweep item {index} failed with exit code {return_code}")
        all_output.append(f"# sweep_item {index} {' '.join(args)}\n" + "".join(lines))
        print(f"sweep_done,{index},{len(sweep_items)}", flush=True)
    return "\n".join(all_output)


@app.local_entrypoint()
def main() -> str:
    call = run_socket_sweep.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
