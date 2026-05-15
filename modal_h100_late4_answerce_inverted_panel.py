"""Modal H100 runner for indexed-summon validation of the answer-KL+CE late4 adapter."""

from __future__ import annotations

import modal


app = modal.App("sva-late4-answerce-inverted-panel-h100")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "transformers>=4.48", "huggingface_hub>=0.24", "numpy>=1.26", "sentencepiece>=0.2")
    .add_local_dir("experiments", remote_path="/root/sva/experiments")
    .add_local_dir("sva", remote_path="/root/sva/sva")
    .add_local_dir("results/hf_artifacts", remote_path="/root/sva/results/hf_artifacts")
)


ADAPTER_DIR = "results/hf_artifacts/sva-late4-512x128-answerdistill-ce001-v1"

BASE_ARGS = [
    "--adapter-dir",
    ADAPTER_DIR,
    "--contexts",
    "32768",
    "--keys",
    "219384,407615,592806,638174,750291,826430,319057,460128",
    "--placements",
    "start,middle,end",
    "--attn-implementation",
    "sdpa",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


def parse_cells(value: str) -> list[int]:
    cells = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cells:
        raise ValueError("Expected at least one inverted cell budget.")
    return cells


@app.function(image=image, gpu="H100", timeout=180 * 60)
def run_late4_answerce_inverted_panel(cells_per_subspace: int, summon_mode: str) -> str:
    import os
    import subprocess
    import sys

    if summon_mode not in {"inverted", "inverted_static"}:
        raise ValueError(f"Unsupported summon mode: {summon_mode}")
    os.chdir("/root/sva")
    cmd = [
        sys.executable,
        "-u",
        "experiments/sva_late4_adapter_answer_benchmark.py",
        *BASE_ARGS,
        "--summon-mode",
        summon_mode,
        "--inverted-cells-per-subspace",
        str(cells_per_subspace),
    ]
    print("late4_answerce_inverted_panel_h100_start", flush=True)
    print(f"summon_mode,{summon_mode}", flush=True)
    print(f"cells_per_subspace,{cells_per_subspace}", flush=True)
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
    print(f"late4_answerce_inverted_panel_h100_exit,{return_code}", flush=True)
    if return_code != 0:
        raise RuntimeError(f"Late4 answer-CE inverted panel failed with exit code {return_code}")
    return "".join(lines)


@app.local_entrypoint()
def main(cells: str = "16,32", summon_mode: str = "inverted") -> str:
    call_ids: list[str] = []
    for cells_per_subspace in parse_cells(cells):
        call = run_late4_answerce_inverted_panel.spawn(cells_per_subspace, summon_mode)
        call_ids.append(call.object_id)
        print(f"function_call_id,cells={cells_per_subspace},{call.object_id}")
        print(f"dashboard,cells={cells_per_subspace},{call.get_dashboard_url()}")
    print(f"adapter_dir,{ADAPTER_DIR}")
    print(f"summon_mode,{summon_mode}")
    return ",".join(call_ids)
