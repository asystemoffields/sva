"""Export an SVA artifact with calibration-refreshed coarse codebooks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import load_sva_artifact_bundle as load_runtime_bundle
from sva_artifact_io import MANIFEST_NAME, WEIGHTS_NAME, load_sva_artifact_bundle, save_sva_artifact_bundle
from sva_block_elevator_benchmark import layer_qkv_from_hidden
from sva_codebook_refresh_benchmark import calibration_stream_to_length, emit
from sva_full_deployment_benchmark import CALIBRATION_DOCS, load_documents
from sva_pq_lookup_test import encode_product_keys, fit_product_codebooks
from sva_pretrained_socket_test import encode_batch, format_layer_list, parse_layer_list
from sva_rotation_diagnostic import mean_code_max_fraction, normalized_code_entropy


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def write_refreshed_readme(output_dir: Path, manifest: dict) -> None:
    text = f"""# Refreshed Summon-Verify Attention Artifact

This folder contains an SVA artifact bundle with calibration-refreshed identity coarse codebooks.

- Base model: `{manifest["model_id"]}`
- Profile: `{manifest["profile_name"]}`
- Base profile: `{manifest.get("base_profile_name")}`
- Context length: `{manifest["context_length"]}`
- Refresh method: `{manifest.get("codebook_refresh_method")}`
- Route source: `{manifest["route_source"]}`
- Rank dim: `{manifest["rank_dim"]}`
- Coarse code: `{manifest["coarse_subspaces"]}x{manifest["coarse_codewords"]}`
- Default shortlist/budget: `{manifest["default_shortlist"]}/{manifest["default_budget"]}`
- Layers: `{manifest["layer_count"]}`

Files:

- `{MANIFEST_NAME}`: artifact metadata and serving defaults.
- `{WEIGHTS_NAME}`: per-layer low-rank projections, logit scales, and refreshed coarse codebooks.

The loader is `sva/artifacts.py`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


@torch.no_grad()
def fit_refreshed_codebooks(
    *,
    model,
    runtime_bundle,
    calibration_batch: dict[str, torch.Tensor],
    layers: list[int],
    kmeans_iters: int,
    assign_chunk_size: int,
    seed: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    seq_len = int(calibration_batch["input_ids"].shape[1])
    output = model(**calibration_batch, use_cache=False, output_hidden_states=True)
    if output.hidden_states is None:
        raise ValueError("Expected calibration hidden states.")
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    refreshed: dict[int, torch.Tensor] = {}

    for layer_idx in layers:
        _, key_all, _, _ = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
        artifact = runtime_bundle.layers[layer_idx]
        k_proj = artifact.k_proj.to(device=device, dtype=torch.float32)
        k_low = torch.einsum("hkd,hdr->hkr", key_all.float(), k_proj)
        codebooks = fit_product_codebooks(
            k_low,
            runtime_bundle.coarse_subspaces,
            runtime_bundle.coarse_codewords,
            kmeans_iters,
            seed + layer_idx * 9973,
            assign_chunk_size,
        )
        codes = encode_product_keys(k_low, codebooks, assign_chunk_size)
        emit(
            "refreshed_artifact_fit",
            {
                "layer": layer_idx,
                "calibration_seq_len": seq_len,
                "code_entropy": normalized_code_entropy(codes, runtime_bundle.coarse_codewords),
                "code_max_fraction": mean_code_max_fraction(codes, runtime_bundle.coarse_codewords),
            },
        )
        refreshed[layer_idx] = codebooks.detach().cpu()
        del key_all, k_low, codebooks, codes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del output
    return refreshed


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an SVA artifact with refreshed coarse codebooks.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-name", default="sva-smollm2-135m-2x256-longctx-refresh-v1")
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--calibration-doc-limit", type=int, default=None)
    parser.add_argument("--calibration-repeats", type=int, default=320)
    parser.add_argument("--calibration-length", type=int, default=32768)
    parser.add_argument("--socket-layers", default="")
    parser.add_argument("--allow-beyond-model-context", action="store_true")
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--artifact-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model_dtype = dtype_from_name(args.model_dtype, device)
    artifact_dtype = dtype_from_name(args.artifact_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    model_context = getattr(model.config, "max_position_embeddings", None)
    if model_context is not None and args.calibration_length > int(model_context) and not args.allow_beyond_model_context:
        raise ValueError(
            f"Requested calibration length {args.calibration_length} exceeds model max_position_embeddings "
            f"{model_context}. Pass --allow-beyond-model-context to override."
        )

    runtime_bundle = load_runtime_bundle(args.artifact_dir, map_location=device)
    base_manifest, base_artifacts = load_sva_artifact_bundle(args.artifact_dir, map_location="cpu")
    requested_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    layers = requested_layers if requested_layers is not None else sorted(runtime_bundle.layers)

    calibration_docs = load_documents(args.calibration_file, CALIBRATION_DOCS, args.calibration_doc_limit, "calibration")
    calibration_text = calibration_stream_to_length(
        tokenizer,
        calibration_docs,
        args.calibration_length,
        args.calibration_repeats,
    )
    calibration_batch = encode_batch(tokenizer, [calibration_text], args.calibration_length, device)

    print("refreshed_artifact_export_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"base_artifact_dir,{args.artifact_dir}", flush=True)
    print(f"base_profile,{runtime_bundle.manifest.get('profile_name')}", flush=True)
    print(f"output_dir,{args.output_dir}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"model_dtype,{model_dtype}", flush=True)
    print(f"artifact_dtype,{artifact_dtype}", flush=True)
    print(f"calibration_docs,{len(calibration_docs)}", flush=True)
    print(f"calibration_seq_len,{calibration_batch['input_ids'].shape[1]}", flush=True)
    print(f"layers,{format_layer_list(layers)}", flush=True)

    refreshed_codebooks = fit_refreshed_codebooks(
        model=model,
        runtime_bundle=runtime_bundle,
        calibration_batch=calibration_batch,
        layers=layers,
        kmeans_iters=args.kmeans_iters,
        assign_chunk_size=args.assign_chunk_size,
        seed=args.seed,
        device=device,
    )

    refreshed_artifacts = {}
    for layer_idx, artifact in base_artifacts.items():
        if layer_idx in refreshed_codebooks:
            refreshed_artifacts[layer_idx] = replace(
                artifact,
                coarse_codebooks=refreshed_codebooks[layer_idx].to(dtype=artifact_dtype),
            )
        else:
            refreshed_artifacts[layer_idx] = artifact

    manifest = dict(base_manifest)
    manifest.update(
        {
            "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "profile_name": args.profile_name,
            "base_profile_name": base_manifest.get("profile_name"),
            "base_artifact_dir": str(args.artifact_dir),
            "context_length": args.calibration_length,
            "codebook_refresh_method": "calibration_identity_kmeans",
            "codebook_refresh_layers": format_layer_list(layers),
            "codebook_refresh_calibration_doc_count": len(calibration_docs),
            "codebook_refresh_calibration_length": args.calibration_length,
            "codebook_refresh_kmeans_iters": args.kmeans_iters,
            "codebook_refresh_seed": args.seed,
            "artifact_dtype": args.artifact_dtype,
            "layer_count": len(refreshed_artifacts),
        }
    )

    output_dir = Path(args.output_dir)
    save_sva_artifact_bundle(output_dir, manifest, refreshed_artifacts, artifact_dtype)
    write_refreshed_readme(output_dir, manifest)
    loaded_manifest, loaded_artifacts = load_sva_artifact_bundle(output_dir)
    print("refreshed_artifact_exported", flush=True)
    print(f"manifest,{json.dumps(loaded_manifest, sort_keys=True)}", flush=True)
    print(f"loaded_layers,{len(loaded_artifacts)}", flush=True)
    print(f"weights_file,{output_dir / WEIGHTS_NAME}", flush=True)


if __name__ == "__main__":
    main()
