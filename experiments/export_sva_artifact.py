"""Export a portable frozen SVA artifact bundle."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sva_artifact_io import MANIFEST_NAME, WEIGHTS_NAME, load_sva_artifact_bundle, save_sva_artifact_bundle
from sva_full_deployment_benchmark import CALIBRATION_DOCS, calibration_stream, load_documents
from sva_pretrained_socket_test import (
    build_artifacts_for_hidden_states,
    build_progressive_three_stage_artifacts,
    encode_batch,
    format_layer_list,
    parse_layer_list,
)


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def write_artifact_readme(output_dir: Path, manifest: dict) -> None:
    text = f"""# Summon-Verify Attention Artifact

This folder contains a frozen SVA artifact bundle.

- Base model: `{manifest["model_id"]}`
- Profile: `{manifest["profile_name"]}`
- Context length: `{manifest["context_length"]}`
- Route source: `{manifest["route_source"]}`
- Rank dim: `{manifest["rank_dim"]}`
- Coarse code: `{manifest["coarse_subspaces"]}x{manifest["coarse_codewords"]}`
- Default shortlist/budget: `{manifest["default_shortlist"]}/{manifest["default_budget"]}`
- Layers: `{manifest["layer_count"]}`

Files:

- `{MANIFEST_NAME}`: artifact metadata and serving defaults.
- `{WEIGHTS_NAME}`: per-layer low-rank projections, logit scales, and coarse codebooks.

The loader is `experiments/sva_artifact_io.py`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen SVA artifacts for a model/profile.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-name", default="sva-smollm2-135m-2x256")
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--calibration-doc-limit", type=int, default=None)
    parser.add_argument("--calibration-repeats", type=int, default=320)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--socket-layers", default="")
    parser.add_argument("--route-source", choices=["qk", "hidden"], default="qk")
    parser.add_argument("--artifact-training", choices=["teacher", "progressive"], default="teacher")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--coarse-subspaces", type=int, default=2)
    parser.add_argument("--coarse-codewords", type=int, default=256)
    parser.add_argument("--coarse-shortlist", type=int, default=2048)
    parser.add_argument("--default-budget", type=int, default=512)
    parser.add_argument("--coarse-label-topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=384)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--ranker-train-steps", type=int, default=280)
    parser.add_argument("--coarse-hard-steps", type=int, default=160)
    parser.add_argument("--coarse-hard-pool", type=int, default=512)
    parser.add_argument("--coarse-hard-negatives", type=int, default=96)
    parser.add_argument("--coarse-hard-margin", type=float, default=1.0)
    parser.add_argument("--coarse-hard-lr-scale", type=float, default=0.5)
    parser.add_argument("--weighted-boost", type=float, default=4.0)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--ranker-lr", type=float, default=0.003)
    parser.add_argument("--ranker-weight-decay", type=float, default=0.0001)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--artifact-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()
    args.mode = "three_stage"
    args.tables = 16
    args.bits = 10
    args.budget = args.default_budget
    args.probe_radius = 1
    args.prefilter_dim = 0
    args.prefilter_budget = 0
    args.diagnose_topk = 0
    args.head_report_limit = 0

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    model_dtype = dtype_from_name(args.model_dtype, device)
    artifact_dtype = dtype_from_name(args.artifact_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=model_dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    calibration_docs = load_documents(
        args.calibration_file,
        CALIBRATION_DOCS,
        args.calibration_doc_limit,
        "calibration",
    )
    calibration_batch = encode_batch(
        tokenizer,
        [calibration_stream(calibration_docs, args.calibration_repeats)],
        args.context_length,
        device,
    )

    if args.artifact_training == "progressive":
        artifacts = build_progressive_three_stage_artifacts(model, calibration_batch, socket_layers, args, device)
    else:
        with torch.no_grad():
            output = model(**calibration_batch, use_cache=False, output_hidden_states=True)
        if output.hidden_states is None:
            raise ValueError("Artifact export requires hidden states.")
        artifacts = build_artifacts_for_hidden_states(model, output.hidden_states, socket_layers, args, device)

    manifest = {
        "artifact_type": "summon_verify_attention",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "model_id": args.model_id,
        "profile_name": args.profile_name,
        "context_length": args.context_length,
        "socket_layers": format_layer_list(socket_layers),
        "layer_count": len(artifacts),
        "route_source": args.route_source,
        "artifact_training": args.artifact_training,
        "rank_dim": args.rank_dim,
        "coarse_subspaces": args.coarse_subspaces,
        "coarse_codewords": args.coarse_codewords,
        "default_shortlist": args.coarse_shortlist,
        "default_budget": args.default_budget,
        "coarse_label_topk": args.coarse_label_topk,
        "train_query_samples": args.train_query_samples,
        "ranker_train_steps": args.ranker_train_steps,
        "coarse_hard_steps": args.coarse_hard_steps,
        "coarse_hard_pool": args.coarse_hard_pool,
        "coarse_hard_negatives": args.coarse_hard_negatives,
        "weighted_boost": args.weighted_boost,
        "artifact_dtype": args.artifact_dtype,
    }

    output_dir = Path(args.output_dir)
    save_sva_artifact_bundle(output_dir, manifest, artifacts, artifact_dtype)
    write_artifact_readme(output_dir, manifest)
    loaded_manifest, loaded_artifacts = load_sva_artifact_bundle(output_dir)
    print("sva_artifact_exported")
    print(f"output_dir,{output_dir}")
    print(f"manifest,{json.dumps(loaded_manifest, sort_keys=True)}")
    print(f"loaded_layers,{len(loaded_artifacts)}")
    print(f"weights_file,{output_dir / WEIGHTS_NAME}")


if __name__ == "__main__":
    main()
