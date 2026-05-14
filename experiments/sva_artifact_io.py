"""Save and load portable SVA artifact bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from sva_pretrained_socket_test import ThreeStageLayerArtifacts


SCHEMA_VERSION = 1
WEIGHTS_NAME = "sva_artifacts.pt"
MANIFEST_NAME = "sva_config.json"


def _json_float(value: float) -> float | None:
    if value != value:
        return None
    return float(value)


def save_sva_artifact_bundle(
    output_dir: Path,
    manifest: dict[str, Any],
    artifacts: dict[int, ThreeStageLayerArtifacts],
    tensor_dtype: torch.dtype,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_meta: dict[str, dict[str, Any]] = {}
    tensors: dict[str, torch.Tensor] = {}

    for layer_idx, artifact in sorted(artifacts.items()):
        key = str(layer_idx)
        prefix = f"layers.{layer_idx}"
        layer_meta[key] = {
            "train_loss": _json_float(float(artifact.train_loss)),
            "hard_loss": _json_float(float(artifact.hard_loss)),
            "route_source": artifact.route_source,
        }
        tensors[f"{prefix}.q_proj"] = artifact.q_proj.detach().cpu().to(tensor_dtype)
        tensors[f"{prefix}.k_proj"] = artifact.k_proj.detach().cpu().to(tensor_dtype)
        tensors[f"{prefix}.logit_scale"] = artifact.logit_scale.detach().cpu().to(torch.float32)
        tensors[f"{prefix}.coarse_codebooks"] = artifact.coarse_codebooks.detach().cpu().to(tensor_dtype)

    weights_payload = {
        "schema_version": SCHEMA_VERSION,
        "layer_meta": layer_meta,
        "tensors": tensors,
    }
    torch.save(weights_payload, output_dir / WEIGHTS_NAME)

    manifest = dict(manifest)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["weights_file"] = WEIGHTS_NAME
    manifest["layers"] = sorted(int(layer_idx) for layer_idx in artifacts)
    with (output_dir / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_sva_artifact_bundle(
    artifact_dir: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[int, ThreeStageLayerArtifacts]]:
    with (artifact_dir / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported SVA artifact schema: {manifest.get('schema_version')!r}")

    weights_file = artifact_dir / manifest.get("weights_file", WEIGHTS_NAME)
    payload = torch.load(weights_file, map_location=map_location, weights_only=False)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported SVA weights schema: {payload.get('schema_version')!r}")

    tensors: dict[str, torch.Tensor] = payload["tensors"]
    layer_meta: dict[str, dict[str, Any]] = payload.get("layer_meta", {})
    artifacts: dict[int, ThreeStageLayerArtifacts] = {}
    for layer_idx in manifest["layers"]:
        layer_idx = int(layer_idx)
        prefix = f"layers.{layer_idx}"
        meta = layer_meta.get(str(layer_idx), {})
        artifacts[layer_idx] = ThreeStageLayerArtifacts(
            q_proj=tensors[f"{prefix}.q_proj"],
            k_proj=tensors[f"{prefix}.k_proj"],
            logit_scale=tensors[f"{prefix}.logit_scale"],
            coarse_codebooks=tensors[f"{prefix}.coarse_codebooks"],
            train_loss=float(meta["train_loss"]) if meta.get("train_loss") is not None else float("nan"),
            hard_loss=float(meta["hard_loss"]) if meta.get("hard_loss") is not None else float("nan"),
            route_source=meta.get("route_source", manifest.get("route_source", "qk")),
        )

    return manifest, artifacts
