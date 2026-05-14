"""Portable SVA artifact loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
WEIGHTS_NAME = "sva_artifacts.pt"
MANIFEST_NAME = "sva_config.json"


@dataclass(frozen=True)
class SVALayerArtifacts:
    """Frozen per-layer tensors needed by the SVA lookup path."""

    q_proj: torch.Tensor
    k_proj: torch.Tensor
    logit_scale: torch.Tensor
    coarse_codebooks: torch.Tensor
    train_loss: float
    hard_loss: float
    route_source: str = "qk"


@dataclass(frozen=True)
class SVAArtifactBundle:
    """Loaded SVA artifact bundle plus manifest metadata."""

    manifest: dict[str, Any]
    layers: dict[int, SVALayerArtifacts]

    @property
    def model_id(self) -> str | None:
        value = self.manifest.get("model_id")
        return str(value) if value is not None else None

    @property
    def rank_dim(self) -> int:
        return int(self.manifest["rank_dim"])

    @property
    def default_shortlist(self) -> int:
        return int(self.manifest.get("default_shortlist", 2048))

    @property
    def default_budget(self) -> int:
        return int(self.manifest.get("default_budget", 512))

    @property
    def coarse_subspaces(self) -> int:
        return int(self.manifest["coarse_subspaces"])

    @property
    def coarse_codewords(self) -> int:
        return int(self.manifest["coarse_codewords"])

    @property
    def layer_count(self) -> int:
        return int(self.manifest.get("layer_count", len(self.layers)))

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SVAArtifactBundle":
        """Return a copy with artifact tensors moved to `device` and optional `dtype`."""

        moved = {}
        for layer_idx, layer in self.layers.items():
            tensor_dtype = dtype if dtype is not None else layer.q_proj.dtype
            moved[layer_idx] = SVALayerArtifacts(
                q_proj=layer.q_proj.to(device=device, dtype=tensor_dtype),
                k_proj=layer.k_proj.to(device=device, dtype=tensor_dtype),
                logit_scale=layer.logit_scale.to(device=device, dtype=torch.float32),
                coarse_codebooks=layer.coarse_codebooks.to(device=device, dtype=tensor_dtype),
                train_loss=layer.train_loss,
                hard_loss=layer.hard_loss,
                route_source=layer.route_source,
            )
        return SVAArtifactBundle(dict(self.manifest), moved)


def _load_torch_payload(weights_file: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(weights_file, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(weights_file, map_location=map_location)


def load_sva_artifact_bundle(
    artifact_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> SVAArtifactBundle:
    """Load a frozen SVA artifact bundle from a local directory."""

    artifact_path = Path(artifact_dir)
    with (artifact_path / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported SVA artifact schema: {manifest.get('schema_version')!r}")

    weights_file = artifact_path / manifest.get("weights_file", WEIGHTS_NAME)
    payload = _load_torch_payload(weights_file, map_location)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported SVA weights schema: {payload.get('schema_version')!r}")

    tensors: dict[str, torch.Tensor] = payload["tensors"]
    layer_meta: dict[str, dict[str, Any]] = payload.get("layer_meta", {})
    layers: dict[int, SVALayerArtifacts] = {}
    for raw_layer_idx in manifest["layers"]:
        layer_idx = int(raw_layer_idx)
        prefix = f"layers.{layer_idx}"
        meta = layer_meta.get(str(layer_idx), {})
        layers[layer_idx] = SVALayerArtifacts(
            q_proj=tensors[f"{prefix}.q_proj"],
            k_proj=tensors[f"{prefix}.k_proj"],
            logit_scale=tensors[f"{prefix}.logit_scale"],
            coarse_codebooks=tensors[f"{prefix}.coarse_codebooks"],
            train_loss=float(meta["train_loss"]) if meta.get("train_loss") is not None else float("nan"),
            hard_loss=float(meta["hard_loss"]) if meta.get("hard_loss") is not None else float("nan"),
            route_source=str(meta.get("route_source", manifest.get("route_source", "qk"))),
        )

    bundle = SVAArtifactBundle(manifest, layers)
    validate_artifact_bundle(bundle)
    return bundle


def validate_artifact_bundle(bundle: SVAArtifactBundle) -> None:
    """Validate manifest/tensor shape consistency before patching a model."""

    if not bundle.layers:
        raise ValueError("SVA artifact bundle contains no layers.")

    rank_dim = bundle.rank_dim
    subspaces = bundle.coarse_subspaces
    codewords = bundle.coarse_codewords
    if rank_dim <= 0:
        raise ValueError(f"rank_dim must be positive, got {rank_dim}.")
    if subspaces <= 0 or rank_dim % subspaces != 0:
        raise ValueError(f"rank_dim={rank_dim} must be divisible by coarse_subspaces={subspaces}.")
    if codewords <= 0:
        raise ValueError(f"coarse_codewords must be positive, got {codewords}.")

    for layer_idx, layer in bundle.layers.items():
        if layer.q_proj.shape != layer.k_proj.shape:
            raise ValueError(f"Layer {layer_idx} q_proj/k_proj shape mismatch: {layer.q_proj.shape} vs {layer.k_proj.shape}.")
        if layer.q_proj.ndim != 3:
            raise ValueError(f"Layer {layer_idx} q_proj must have shape [heads, head_dim, rank_dim].")
        if int(layer.q_proj.shape[-1]) != rank_dim:
            raise ValueError(f"Layer {layer_idx} rank_dim mismatch: {layer.q_proj.shape[-1]} vs {rank_dim}.")
        if layer.logit_scale.ndim != 1 or int(layer.logit_scale.shape[0]) != int(layer.q_proj.shape[0]):
            raise ValueError(f"Layer {layer_idx} logit_scale must have one value per attention head.")
        expected_sub_dim = rank_dim // subspaces
        expected_codebook = (int(layer.q_proj.shape[0]), subspaces, codewords, expected_sub_dim)
        if tuple(layer.coarse_codebooks.shape) != expected_codebook:
            raise ValueError(
                f"Layer {layer_idx} coarse_codebooks shape mismatch: "
                f"{tuple(layer.coarse_codebooks.shape)} vs {expected_codebook}."
            )
        if layer.route_source != "qk":
            raise ValueError(
                f"Layer {layer_idx} uses route_source={layer.route_source!r}. "
                "The production adapter currently supports qk artifacts."
            )
