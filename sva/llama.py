"""Llama-family drop-in attention adapter for Summon-Verify Attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .artifacts import SVAArtifactBundle, SVALayerArtifacts, load_sva_artifact_bundle
from .ops import encode_product_keys, product_quantized_scores
from .stats import SVAStats


@dataclass(frozen=True)
class SVALlamaServingConfig:
    """Serving settings for a patched Llama attention module."""

    rank_dim: int
    coarse_shortlist: int
    budget: int
    assign_chunk_size: int = 8192
    query_chunk_size: int | None = None


class SVALlamaAttention(nn.Module):
    """Drop-in replacement for a Llama attention layer using frozen SVA artifacts."""

    def __init__(
        self,
        original: nn.Module,
        layer_artifacts: SVALayerArtifacts,
        serving: SVALlamaServingConfig,
        stats: SVAStats | None = None,
    ) -> None:
        super().__init__()
        self.original = original
        self.config = original.config
        self.layer_idx = original.layer_idx
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.scaling = original.scaling
        self.attention_dropout = original.attention_dropout
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.serving = serving
        self.stats = stats

        self.register_buffer("sva_q_proj", layer_artifacts.q_proj.detach().clone(), persistent=False)
        self.register_buffer("sva_k_proj", layer_artifacts.k_proj.detach().clone(), persistent=False)
        self.register_buffer("sva_logit_scale", layer_artifacts.logit_scale.detach().float().clone(), persistent=False)
        self.register_buffer("sva_coarse_codebooks", layer_artifacts.coarse_codebooks.detach().clone(), persistent=False)
        self._cached_k_low: torch.Tensor | None = None
        self._cached_coarse_codes: torch.Tensor | None = None
        self._cached_key_len = 0
        self._cached_signature: tuple[torch.device, torch.dtype, int, int, int] | None = None

    def reset_catalog(self) -> None:
        self._cached_k_low = None
        self._cached_coarse_codes = None
        self._cached_key_len = 0
        self._cached_signature = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if self.training:
            raise RuntimeError("SVA Llama adapter is inference-only. Call model.eval() before running it.")
        if kwargs.get("output_attentions"):
            raise RuntimeError("SVA Llama adapter does not materialize dense attention weights.")
        if position_embeddings is None:
            raise ValueError("SVA Llama adapter requires model-provided RoPE position embeddings.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output = self.sva_attention(query_states, key_states, value_states, attention_mask)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output), None

    def sva_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)
        batch, n_heads, q_len, head_dim = query.shape
        k_len = key_states.shape[2]
        if batch != 1:
            raise ValueError("SVA Llama adapter currently expects batch size 1.")

        if attention_mask is not None:
            allowed = attention_mask[..., :q_len, :k_len] > -1e4
            allowed = allowed.expand(batch, n_heads, q_len, k_len)
        else:
            key_positions = torch.arange(k_len, device=query.device)
            if q_len == k_len:
                query_positions = torch.arange(q_len, device=query.device)
            else:
                query_positions = torch.arange(k_len - q_len, k_len, device=query.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            allowed = allowed[None, None, :, :].expand(batch, n_heads, q_len, k_len)

        q_proj = self.sva_q_proj.to(device=query.device, dtype=torch.float32)
        k_proj = self.sva_k_proj.to(device=query.device, dtype=torch.float32)
        scale = self.sva_logit_scale.to(device=query.device, dtype=torch.float32).exp().clamp(0.01, 100.0)
        codebooks = self.sva_coarse_codebooks.to(device=query.device, dtype=torch.float32)
        q_low = torch.einsum("bhtd,hdr->bhtr", query.float(), q_proj) * scale[None, :, None, None]
        k_low, coarse_codes = self._key_catalog(key_states, k_proj, codebooks, q_len)
        actual_shortlist = min(self.serving.coarse_shortlist, k_len)
        actual_budget = min(self.serving.budget, actual_shortlist)
        query_chunk_size = self.serving.query_chunk_size
        if query_chunk_size is None:
            query_chunk_size = 128 if (q_len >= 4096 or actual_shortlist >= 2048) else q_len

        output_chunks: list[torch.Tensor] = []
        for q_start in range(0, q_len, query_chunk_size):
            q_end = min(q_start + query_chunk_size, q_len)
            chunk_len = q_end - q_start
            query_chunk = query[:, :, q_start:q_end, :]
            q_low_chunk = q_low[:, :, q_start:q_end, :]
            allowed_chunk = allowed[:, :, q_start:q_end, :]

            coarse_scores = product_quantized_scores(
                q_low_chunk[0],
                codebooks,
                coarse_codes,
                self.serving.rank_dim,
            )[None, :, :, :]
            coarse_scores = coarse_scores.masked_fill(~allowed_chunk, torch.finfo(coarse_scores.dtype).min)

            coarse_idx = coarse_scores.topk(actual_shortlist, dim=-1).indices
            candidate_counts = torch.minimum(
                allowed_chunk.sum(dim=-1),
                torch.tensor(actual_shortlist, dtype=torch.long, device=query.device),
            )
            candidate_valid = torch.arange(actual_shortlist, device=query.device)
            candidate_valid = candidate_valid[None, None, None, :] < candidate_counts[..., None]

            source_low = k_low[:, :, None, :, :].expand(batch, n_heads, chunk_len, k_len, self.serving.rank_dim)
            shortlist_low = torch.gather(
                source_low,
                dim=3,
                index=coarse_idx[..., None].expand(batch, n_heads, chunk_len, actual_shortlist, self.serving.rank_dim),
            )
            rank_scores = (shortlist_low * q_low_chunk[..., None, :]).sum(dim=-1) / math.sqrt(self.serving.rank_dim)
            rank_scores = rank_scores.masked_fill(~candidate_valid, torch.finfo(rank_scores.dtype).min)

            _, rank_keep = rank_scores.topk(actual_budget, dim=-1)
            final_idx = coarse_idx.gather(dim=-1, index=rank_keep)
            verified_counts = torch.minimum(
                candidate_counts,
                torch.tensor(actual_budget, dtype=torch.long, device=query.device),
            )
            verified_valid = torch.arange(actual_budget, device=query.device)
            verified_valid = verified_valid[None, None, None, :] < verified_counts[..., None]

            selected_keys = torch.gather(
                key_states[:, :, None, :, :].expand(batch, n_heads, chunk_len, k_len, head_dim),
                dim=3,
                index=final_idx[..., None].expand(batch, n_heads, chunk_len, actual_budget, head_dim),
            )
            selected_values = torch.gather(
                value_states[:, :, None, :, :].expand(batch, n_heads, chunk_len, k_len, head_dim),
                dim=3,
                index=final_idx[..., None].expand(batch, n_heads, chunk_len, actual_budget, head_dim),
            )
            selected_scores = (selected_keys * query_chunk[..., None, :]).sum(dim=-1) * self.scaling
            selected_scores = selected_scores.masked_fill(~verified_valid, torch.finfo(selected_scores.dtype).min)
            weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            output_chunks.append((weights[..., None] * selected_values).sum(dim=-2))

            if self.stats is not None:
                counts = {
                    "summoned": float(candidate_counts.float().sum().item()),
                    "exact_scored": float(verified_counts.float().sum().item()),
                    "verified": float(verified_counts.float().sum().item()),
                    "queries": float(candidate_counts.numel()),
                }
                self.stats.add(int(self.layer_idx or 0), counts)

        return torch.cat(output_chunks, dim=2)

    def _key_catalog(
        self,
        key_states: torch.Tensor,
        k_proj: torch.Tensor,
        codebooks: torch.Tensor,
        q_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, n_heads, k_len, head_dim = key_states.shape
        signature = (key_states.device, key_states.dtype, n_heads, head_dim, self.serving.rank_dim)
        can_append = (
            self._cached_signature == signature
            and self._cached_k_low is not None
            and self._cached_coarse_codes is not None
            and self._cached_key_len > 0
            and k_len > self._cached_key_len
            and k_len - self._cached_key_len <= max(q_len, 1)
            and q_len < k_len
        )
        if can_append:
            start = self._cached_key_len
            new_k_low = torch.einsum("bhsd,hdr->bhsr", key_states[:, :, start:, :].float(), k_proj)
            new_codes = encode_product_keys(new_k_low[0], codebooks, self.serving.assign_chunk_size)
            k_low = torch.cat([self._cached_k_low, new_k_low], dim=2)
            coarse_codes = torch.cat([self._cached_coarse_codes, new_codes], dim=1)
        else:
            k_low = torch.einsum("bhsd,hdr->bhsr", key_states.float(), k_proj)
            coarse_codes = encode_product_keys(k_low[0], codebooks, self.serving.assign_chunk_size)

        self._cached_k_low = k_low.detach()
        self._cached_coarse_codes = coarse_codes.detach()
        self._cached_key_len = k_len
        self._cached_signature = signature
        return k_low, coarse_codes


class SVALlamaPatcher:
    """Reversible patch handle for Llama-family Hugging Face models."""

    def __init__(
        self,
        model: nn.Module,
        bundle: SVAArtifactBundle,
        shortlist: int | None = None,
        budget: int | None = None,
        assign_chunk_size: int = 8192,
    ) -> None:
        self.model = model
        self.bundle = bundle
        self.stats = SVAStats()
        self.originals: dict[int, nn.Module] = {}
        self.serving = SVALlamaServingConfig(
            rank_dim=bundle.rank_dim,
            coarse_shortlist=bundle.default_shortlist if shortlist is None else int(shortlist),
            budget=bundle.default_budget if budget is None else int(budget),
            assign_chunk_size=assign_chunk_size,
        )

    def patch(self) -> "SVALlamaPatcher":
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Expected a Hugging Face Llama-style model with model.layers.")
        if self.originals:
            return self

        if self.bundle.layer_count != len(layers):
            raise ValueError(f"Artifact has {self.bundle.layer_count} layers but model has {len(layers)}.")
        model_id = self.bundle.model_id
        model_name = getattr(getattr(self.model, "config", None), "_name_or_path", None)
        if model_id and model_name and str(model_name) not in {"", model_id}:
            raise ValueError(f"Artifact model_id={model_id!r} does not match loaded model {model_name!r}.")

        for layer_idx, layer in enumerate(layers):
            artifacts = self.bundle.layers.get(layer_idx)
            if artifacts is None:
                raise ValueError(f"Missing SVA artifacts for layer {layer_idx}.")
            self._validate_layer_shapes(layer_idx, layer.self_attn, artifacts)
            self.originals[layer_idx] = layer.self_attn
            replacement = SVALlamaAttention(layer.self_attn, artifacts, self.serving, self.stats)
            replacement.train(layer.self_attn.training)
            layer.self_attn = replacement
        return self

    def unpatch(self) -> None:
        layers = self.model.model.layers
        for layer_idx, original in self.originals.items():
            layers[layer_idx].self_attn = original
        self.originals.clear()

    def reset_stats(self) -> None:
        self.stats.reset()

    def reset_catalogs(self) -> None:
        for layer in self.model.model.layers:
            attention = layer.self_attn
            if isinstance(attention, SVALlamaAttention):
                attention.reset_catalog()

    def __enter__(self) -> "SVALlamaPatcher":
        return self.patch()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.unpatch()

    def _validate_layer_shapes(self, layer_idx: int, attention: nn.Module, artifacts: SVALayerArtifacts) -> None:
        n_heads = int(attention.config.num_attention_heads)
        head_dim = int(attention.head_dim)
        expected = (n_heads, head_dim, self.bundle.rank_dim)
        if tuple(artifacts.q_proj.shape) != expected:
            raise ValueError(f"Layer {layer_idx} artifact q_proj shape {tuple(artifacts.q_proj.shape)} != {expected}.")
        if tuple(artifacts.k_proj.shape) != expected:
            raise ValueError(f"Layer {layer_idx} artifact k_proj shape {tuple(artifacts.k_proj.shape)} != {expected}.")


def patch_llama_attention(
    model: nn.Module,
    artifact_dir_or_bundle: str | Path | SVAArtifactBundle,
    shortlist: int | None = None,
    budget: int | None = None,
    assign_chunk_size: int = 8192,
) -> SVALlamaPatcher:
    """Patch a Llama-family model with SVA attention and return a reversible handle."""

    bundle = (
        artifact_dir_or_bundle
        if isinstance(artifact_dir_or_bundle, SVAArtifactBundle)
        else load_sva_artifact_bundle(artifact_dir_or_bundle)
    )
    return SVALlamaPatcher(
        model=model,
        bundle=bundle,
        shortlist=shortlist,
        budget=budget,
        assign_chunk_size=assign_chunk_size,
    ).patch()
