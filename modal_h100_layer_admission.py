"""Modal H100 runner for an automatic selective SVA layer-admission pass."""

from __future__ import annotations

import modal


app = modal.App("sva-layer-admission-h100")

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
    "4",
    "--device",
    "cuda",
    "--dtype",
    "bfloat16",
]


BASE_LAYERS = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 17, 18, 21]
CANDIDATE_LAYERS = [2, 11, 12, 14, 16, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29]

TEACHER_LOSS_THRESHOLD = 0.050
PROGRESSIVE_RETRY_THRESHOLD = 0.150
PROGRESSIVE_LOSS_THRESHOLD = 0.060


def layers_arg(layers: list[int]) -> str:
    return ",".join(str(layer) for layer in layers)


def parse_metrics(lines: list[str]) -> dict[str, str]:
    wanted = {
        "socket_layer_count",
        "artifact_training",
        "loss_delta",
        "kl_to_full",
        "top1_agreement",
        "logit_cosine",
        "candidate_top16_recall",
        "verified_top16_recall",
        "full_loss",
        "sva_loss",
    }
    metrics: dict[str, str] = {}
    for line in lines:
        if "," not in line:
            continue
        key, value = line.strip().split(",", 1)
        if key in wanted:
            metrics[key] = value
    return metrics


@app.function(image=image, gpu="H100", timeout=120 * 60)
def run_layer_admission() -> str:
    import os
    import subprocess
    import sys

    os.chdir("/root/sva")
    all_output: list[str] = []
    admitted = list(BASE_LAYERS)
    decisions: list[dict[str, str]] = []

    def run_case(label: str, case_layers: list[int], artifact_training: str) -> dict[str, str]:
        args = [
            "--artifact-training",
            artifact_training,
            "--socket-layers",
            layers_arg(case_layers),
        ]
        cmd = [sys.executable, "-u", "experiments/sva_pretrained_socket_test.py", *BASE_ARGS, *args]
        print(f"layer_admission_start,{label},{artifact_training},{layers_arg(case_layers)}", flush=True)
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
        all_output.append(f"# admission {label} {artifact_training} {layers_arg(case_layers)}\n" + "".join(lines))
        metrics = parse_metrics(lines)
        metrics["return_code"] = str(return_code)
        print(f"layer_admission_done,{label},{artifact_training},{return_code}", flush=True)
        return metrics

    base_metrics = run_case("base", BASE_LAYERS, "teacher")
    print(
        "admission_baseline,"
        + ",".join(
            [
                base_metrics.get("loss_delta", ""),
                base_metrics.get("kl_to_full", ""),
                base_metrics.get("top1_agreement", ""),
                base_metrics.get("verified_top16_recall", ""),
            ]
        ),
        flush=True,
    )

    for layer in CANDIDATE_LAYERS:
        candidate_layers = sorted(set(BASE_LAYERS + [layer]))
        teacher = run_case(f"candidate_{layer}", candidate_layers, "teacher")
        teacher_loss = float(teacher.get("loss_delta", "inf"))
        decision = "reject"
        source = "teacher"
        final = teacher

        if teacher_loss <= TEACHER_LOSS_THRESHOLD:
            decision = "admit"
        elif teacher_loss <= PROGRESSIVE_RETRY_THRESHOLD:
            progressive = run_case(f"candidate_{layer}_progressive", candidate_layers, "progressive")
            progressive_loss = float(progressive.get("loss_delta", "inf"))
            source = "progressive"
            final = progressive
            if progressive_loss <= PROGRESSIVE_LOSS_THRESHOLD:
                decision = "admit"

        if decision == "admit":
            admitted = sorted(set(admitted + [layer]))

        decisions.append(
            {
                "layer": str(layer),
                "decision": decision,
                "source": source,
                "loss_delta": final.get("loss_delta", ""),
                "kl_to_full": final.get("kl_to_full", ""),
                "top1_agreement": final.get("top1_agreement", ""),
                "verified_top16_recall": final.get("verified_top16_recall", ""),
            }
        )
        print(
            "admission_decision,"
            + ",".join(
                [
                    str(layer),
                    decision,
                    source,
                    final.get("loss_delta", ""),
                    final.get("kl_to_full", ""),
                    final.get("top1_agreement", ""),
                    final.get("verified_top16_recall", ""),
                    layers_arg(admitted),
                ]
            ),
            flush=True,
        )

    combined_teacher = run_case("combined_admitted", admitted, "teacher")
    combined_progressive = run_case("combined_admitted_progressive", admitted, "progressive")
    print(f"admission_final_layers,{layers_arg(admitted)}", flush=True)
    print(
        "admission_final_teacher,"
        + ",".join(
            [
                combined_teacher.get("loss_delta", ""),
                combined_teacher.get("kl_to_full", ""),
                combined_teacher.get("top1_agreement", ""),
                combined_teacher.get("verified_top16_recall", ""),
            ]
        ),
        flush=True,
    )
    print(
        "admission_final_progressive,"
        + ",".join(
            [
                combined_progressive.get("loss_delta", ""),
                combined_progressive.get("kl_to_full", ""),
                combined_progressive.get("top1_agreement", ""),
                combined_progressive.get("verified_top16_recall", ""),
            ]
        ),
        flush=True,
    )
    all_output.append("# admission decisions\n")
    for decision in decisions:
        all_output.append(str(decision) + "\n")
    all_output.append(f"# final admitted {layers_arg(admitted)}\n")
    return "\n".join(all_output)


@app.local_entrypoint()
def main() -> str:
    call = run_layer_admission.spawn()
    print(f"function_call_id,{call.object_id}")
    print(f"dashboard,{call.get_dashboard_url()}")
    return call.object_id
