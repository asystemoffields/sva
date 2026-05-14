"""Socket Summon-Verify Attention into a pretrained Llama-style LM.

The test keeps pretrained Q/K/V/O projections, RoPE, norms, MLPs, and logits.
Only the attention lookup changes: full causal attention is replaced by an SVA
candidate mask followed by exact QK scoring over the summoned candidates.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from sva_learned_ranker_test import LowRankRanker, train_ranker
from sva_pq_lookup_test import encode_product_keys, product_quantized_scores
from sva_real_qk_address_sweep import sample_query_positions, topk_indices_for_queries
from sva_supervised_coarse_pq_test import fit_weighted_product_codebooks, key_label_weights, train_ranker_hard_negatives


def make_stats() -> defaultdict[str, float]:
    return defaultdict(float)


def make_nested_stats() -> defaultdict[int | tuple[int, int], defaultdict[str, float]]:
    return defaultdict(make_stats)


@dataclass
class ThreeStageLayerArtifacts:
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    logit_scale: torch.Tensor
    coarse_codebooks: torch.Tensor
    train_loss: float
    hard_loss: float
    route_source: str = "qk"


TEXTS = [
    (
        "Summon-Verify Attention asks a query to summon a small set of candidate memories, "
        "then verifies them exactly. The important question is whether a pretrained model's "
        "own keys and queries already contain enough address structure for that lookup to work."
    ),
    (
        "A tiny language model is useful here because its learned keys and queries are real, "
        "but the model is still inspectable. If the socket test works on a small Llama-style "
        "decoder, the next experiment can move to a larger model without changing the premise."
    ),
    (
        "The river moved quietly through the city while the old library kept its lights on late "
        "into the evening. A courier crossed the bridge, checked the address twice, and found "
        "the right door by matching small details that nobody else seemed to notice."
    ),
    (
        "Sparse attention methods must preserve the important behavior of full attention while "
        "reading far fewer cached tokens. The failure mode is simple: if the retrieval stage misses "
        "one crucial earlier token, the exact verifier can only be precisely wrong."
    ),
    (
        "Modern decoder blocks usually combine grouped-query attention, rotary position embeddings, "
        "RMS normalization, and a gated feedforward network. A socketed replacement should leave those "
        "learned components intact and change only the way candidate keys are selected."
    ),
    (
        "In an engineering notebook, the first result is rarely the final answer. A useful experiment "
        "separates the scientific risk from the systems risk, records what actually happened, and then "
        "makes the next measurement sharper."
    ),
]


@dataclass
class SVAConfig:
    mode: str = "lsh"
    tables: int = 16
    bits: int = 10
    budget: int = 64
    probe_radius: int = 1
    seed: int = 17
    prefilter_dim: int = 0
    prefilter_budget: int = 0
    rank_dim: int = 64
    coarse_subspaces: int = 4
    coarse_codewords: int = 64
    coarse_shortlist: int = 1024
    assign_chunk_size: int = 8192
    diagnose_topk: int = 0
    head_report_limit: int = 0
    three_stage_artifacts: dict[int, ThreeStageLayerArtifacts] = field(default_factory=dict)
    stats: defaultdict[str, float] = field(default_factory=make_stats)
    layer_stats: defaultdict[int | tuple[int, int], defaultdict[str, float]] = field(default_factory=make_nested_stats)
    head_stats: defaultdict[int | tuple[int, int], defaultdict[str, float]] = field(default_factory=make_nested_stats)


class SVALlamaAttention(nn.Module):
    def __init__(self, original: nn.Module, cfg: SVAConfig) -> None:
        super().__init__()
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
        self.cfg = cfg

        n_heads = self.config.num_attention_heads
        generator = torch.Generator(device=self.q_proj.weight.device)
        generator.manual_seed(cfg.seed + 10_000 * int(self.layer_idx or 0))
        projections = torch.randn(
            n_heads,
            cfg.tables,
            cfg.bits,
            self.head_dim,
            generator=generator,
            device=self.q_proj.weight.device,
            dtype=self.q_proj.weight.dtype,
        )
        projections = projections / math.sqrt(self.head_dim)
        self.register_buffer("sva_projections", projections, persistent=False)
        if cfg.prefilter_dim > 0:
            prefilter_projections = torch.randn(
                n_heads,
                cfg.prefilter_dim,
                self.head_dim,
                generator=generator,
                device=self.q_proj.weight.device,
                dtype=self.q_proj.weight.dtype,
            )
            prefilter_projections = prefilter_projections / math.sqrt(self.head_dim)
        else:
            prefilter_projections = None
        self.register_buffer("prefilter_projections", prefilter_projections, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("SVA socket test requires model-provided RoPE position embeddings.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_output = self.sva_attention(query_states, key_states, value_states, attention_mask, hidden_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o_proj(attn_output), None

    def sva_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        if cfg.mode == "three_stage":
            return self.three_stage_attention(query, key, value, attention_mask, hidden_states)

        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)
        batch, n_heads, q_len, head_dim = query.shape
        k_len = key_states.shape[2]
        projections = self.sva_projections[:n_heads, : cfg.tables, : cfg.bits, :].to(dtype=query.dtype)

        q_bits = torch.einsum("bhtd,hrmd->bhtrm", query, projections) > 0
        k_bits = torch.einsum("bhsd,hrmd->bhsrm", key_states, projections) > 0
        hamming = (q_bits[:, :, :, None, :, :] != k_bits[:, :, None, :, :, :]).sum(dim=-1)
        candidate_mask = (hamming <= cfg.probe_radius).any(dim=-1)

        if attention_mask is not None:
            allowed = attention_mask[..., :q_len, :k_len] > -1e4
            allowed = allowed.expand(batch, n_heads, q_len, k_len)
            candidate_mask = candidate_mask & allowed
        else:
            causal = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril()
            allowed = causal[None, None, :, :].expand(batch, n_heads, q_len, k_len)
            candidate_mask = candidate_mask & allowed

        if q_len == k_len:
            eye = torch.eye(q_len, dtype=torch.bool, device=query.device)
            candidate_mask = candidate_mask | eye[None, None, :, :]

        full_scores = torch.matmul(query, key_states.transpose(2, 3)) * self.scaling
        top_idx = None
        top_valid = None
        if cfg.diagnose_topk > 0:
            topk = min(cfg.diagnose_topk, k_len)
            full_masked = full_scores.masked_fill(~allowed, torch.finfo(full_scores.dtype).min)
            top_idx = full_masked.topk(topk, dim=-1).indices
            rank = torch.arange(topk, device=query.device)
            top_valid = rank[None, None, None, :] < allowed.sum(dim=-1)[..., None]

        candidate_counts = candidate_mask.sum(dim=-1)
        exact_mask = candidate_mask

        if cfg.prefilter_dim > 0 and cfg.prefilter_budget > 0 and cfg.prefilter_budget < k_len:
            prefilter_projections = self.prefilter_projections[:n_heads, : cfg.prefilter_dim, :].to(dtype=query.dtype)
            q_low = torch.einsum("bhtd,hmd->bhtm", query, prefilter_projections)
            k_low = torch.einsum("bhsd,hmd->bhsm", key_states, prefilter_projections)
            prefilter_scores = torch.matmul(q_low, k_low.transpose(2, 3)) / math.sqrt(cfg.prefilter_dim)
            prefilter_scores = prefilter_scores.masked_fill(~candidate_mask, torch.finfo(prefilter_scores.dtype).min)
            prefilter_k = min(cfg.prefilter_budget, k_len)
            _, prefilter_idx = torch.topk(prefilter_scores, prefilter_k, dim=-1)
            prefilter_counts = torch.minimum(
                candidate_counts,
                torch.tensor(prefilter_k, dtype=candidate_counts.dtype, device=candidate_counts.device),
            )
            prefilter_valid = torch.arange(prefilter_k, device=query.device)
            prefilter_valid = prefilter_valid[None, None, None, :] < prefilter_counts[..., None]
            exact_mask = torch.zeros_like(candidate_mask)
            exact_mask.scatter_(-1, prefilter_idx, prefilter_valid)

        exact_counts = exact_mask.sum(dim=-1)
        scores = full_scores.masked_fill(~exact_mask, torch.finfo(full_scores.dtype).min)

        if cfg.budget > 0 and cfg.budget < k_len:
            chosen_scores, chosen_idx = torch.topk(scores, cfg.budget, dim=-1)
            source = value_states[:, :, None, :, :].expand(batch, n_heads, q_len, k_len, head_dim)
            chosen_values = torch.gather(
                source,
                dim=3,
                index=chosen_idx[..., None].expand(batch, n_heads, q_len, cfg.budget, head_dim),
            )
            weights = F.softmax(chosen_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            output = (weights[..., None] * chosen_values).sum(dim=-2)
            verified_counts = torch.minimum(
                exact_counts,
                torch.tensor(cfg.budget, dtype=candidate_counts.dtype, device=candidate_counts.device),
            )
            verified_mask = None
            if cfg.diagnose_topk > 0:
                chosen_valid = torch.arange(cfg.budget, device=query.device)
                chosen_valid = chosen_valid[None, None, None, :] < verified_counts[..., None]
                verified_mask = torch.zeros_like(candidate_mask)
                verified_mask.scatter_(-1, chosen_idx, chosen_valid)
        else:
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            output = torch.matmul(weights, value_states)
            verified_counts = exact_counts
            verified_mask = exact_mask

        record_attention_stats(
            cfg,
            int(self.layer_idx or 0),
            candidate_counts,
            exact_counts,
            verified_counts,
            candidate_mask,
            exact_mask,
            verified_mask,
            top_idx,
            top_valid,
        )
        return output

    def three_stage_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        cfg = self.cfg
        layer_idx = int(self.layer_idx or 0)
        artifacts = cfg.three_stage_artifacts.get(layer_idx)
        if artifacts is None:
            raise ValueError(f"Missing three-stage SVA artifacts for layer {layer_idx}.")

        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)
        batch, n_heads, q_len, head_dim = query.shape
        k_len = key_states.shape[2]
        if batch != 1:
            raise ValueError("Three-stage socket mode currently expects batch size 1.")

        if attention_mask is not None:
            allowed = attention_mask[..., :q_len, :k_len] > -1e4
            allowed = allowed.expand(batch, n_heads, q_len, k_len)
        else:
            causal = torch.ones(q_len, k_len, dtype=torch.bool, device=query.device).tril()
            allowed = causal[None, None, :, :].expand(batch, n_heads, q_len, k_len)

        q_proj = artifacts.q_proj.to(device=query.device, dtype=torch.float32)
        k_proj = artifacts.k_proj.to(device=query.device, dtype=torch.float32)
        scale = artifacts.logit_scale.to(device=query.device, dtype=torch.float32).exp().clamp(0.01, 100.0)
        if artifacts.route_source == "hidden":
            if hidden_states is None:
                raise ValueError("Hidden-state route source requires hidden_states.")
            route = hidden_states.float()
            q_low = torch.einsum("btd,hdr->bhtr", route, q_proj) * scale[None, :, None, None]
            k_low = torch.einsum("bsd,hdr->bhsr", route, k_proj)
        else:
            q_low = torch.einsum("bhtd,hdr->bhtr", query.float(), q_proj) * scale[None, :, None, None]
            k_low = torch.einsum("bhsd,hdr->bhsr", key_states.float(), k_proj)

        codebooks = artifacts.coarse_codebooks.to(device=query.device, dtype=torch.float32)
        coarse_codes = encode_product_keys(k_low[0], codebooks, cfg.assign_chunk_size)
        actual_shortlist = min(cfg.coarse_shortlist, k_len)
        actual_budget = min(cfg.budget, actual_shortlist)
        query_chunk_size = 128 if (q_len >= 4096 or actual_shortlist >= 2048) else q_len
        output_chunks: list[torch.Tensor] = []

        for q_start in range(0, q_len, query_chunk_size):
            q_end = min(q_start + query_chunk_size, q_len)
            chunk_len = q_end - q_start
            query_chunk = query[:, :, q_start:q_end, :]
            q_low_chunk = q_low[:, :, q_start:q_end, :]
            allowed_chunk = allowed[:, :, q_start:q_end, :]

            coarse_scores = product_quantized_scores(q_low_chunk[0], codebooks, coarse_codes, cfg.rank_dim)[
                None, :, :, :
            ]
            coarse_scores = coarse_scores.masked_fill(~allowed_chunk, torch.finfo(coarse_scores.dtype).min)

            coarse_idx = coarse_scores.topk(actual_shortlist, dim=-1).indices
            candidate_counts = torch.minimum(
                allowed_chunk.sum(dim=-1),
                torch.tensor(actual_shortlist, dtype=torch.long, device=query.device),
            )
            candidate_valid = torch.arange(actual_shortlist, device=query.device)
            candidate_valid = candidate_valid[None, None, None, :] < candidate_counts[..., None]
            candidate_mask = torch.zeros_like(allowed_chunk)
            candidate_mask.scatter_(-1, coarse_idx, candidate_valid)

            source_low = k_low[:, :, None, :, :].expand(batch, n_heads, chunk_len, k_len, cfg.rank_dim)
            shortlist_low = torch.gather(
                source_low,
                dim=3,
                index=coarse_idx[..., None].expand(batch, n_heads, chunk_len, actual_shortlist, cfg.rank_dim),
            )
            rank_scores = (shortlist_low * q_low_chunk[..., None, :]).sum(dim=-1) / math.sqrt(cfg.rank_dim)
            rank_scores = rank_scores.masked_fill(~candidate_valid, torch.finfo(rank_scores.dtype).min)

            _, rank_keep = rank_scores.topk(actual_budget, dim=-1)
            final_idx = coarse_idx.gather(dim=-1, index=rank_keep)
            verified_counts = torch.minimum(
                candidate_counts,
                torch.tensor(actual_budget, dtype=torch.long, device=query.device),
            )
            verified_valid = torch.arange(actual_budget, device=query.device)
            verified_valid = verified_valid[None, None, None, :] < verified_counts[..., None]
            verified_mask = torch.zeros_like(allowed_chunk)
            verified_mask.scatter_(-1, final_idx, verified_valid)

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

            top_idx = None
            top_valid = None
            if cfg.diagnose_topk > 0:
                topk = min(cfg.diagnose_topk, k_len)
                full_scores = torch.matmul(query_chunk, key_states.transpose(2, 3)) * self.scaling
                full_masked = full_scores.masked_fill(~allowed_chunk, torch.finfo(full_scores.dtype).min)
                top_idx = full_masked.topk(topk, dim=-1).indices
                rank = torch.arange(topk, device=query.device)
                top_valid = rank[None, None, None, :] < allowed_chunk.sum(dim=-1)[..., None]

            record_attention_stats(
                cfg,
                layer_idx,
                candidate_counts,
                verified_counts,
                verified_counts,
                candidate_mask,
                verified_mask,
                verified_mask,
                top_idx,
                top_valid,
            )

        return torch.cat(output_chunks, dim=2)


def add_to_stats(target: defaultdict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        target[key] += value


def record_attention_stats(
    cfg: SVAConfig,
    layer_idx: int,
    candidate_counts: torch.Tensor,
    exact_counts: torch.Tensor,
    verified_counts: torch.Tensor,
    candidate_mask: torch.Tensor,
    exact_mask: torch.Tensor,
    verified_mask: torch.Tensor | None,
    top_idx: torch.Tensor | None,
    top_valid: torch.Tensor | None,
) -> None:
    candidate_counts_f = candidate_counts.float()
    exact_counts_f = exact_counts.float()
    verified_counts_f = verified_counts.float()
    total = {
        "summoned": float(candidate_counts_f.sum().item()),
        "exact_scored": float(exact_counts_f.sum().item()),
        "verified": float(verified_counts_f.sum().item()),
        "queries": float(candidate_counts.numel()),
    }
    add_to_stats(cfg.stats, total)
    add_to_stats(cfg.layer_stats[layer_idx], total)

    if cfg.head_report_limit > 0:
        per_head_summoned = candidate_counts_f.sum(dim=(0, 2)).detach().cpu()
        per_head_exact = exact_counts_f.sum(dim=(0, 2)).detach().cpu()
        per_head_verified = verified_counts_f.sum(dim=(0, 2)).detach().cpu()
        head_queries = float(candidate_counts.shape[0] * candidate_counts.shape[2])
        for head_idx in range(candidate_counts.shape[1]):
            add_to_stats(
                cfg.head_stats[(layer_idx, head_idx)],
                {
                    "summoned": float(per_head_summoned[head_idx].item()),
                    "exact_scored": float(per_head_exact[head_idx].item()),
                    "verified": float(per_head_verified[head_idx].item()),
                    "queries": head_queries,
                },
            )

    if top_idx is None or top_valid is None or verified_mask is None:
        return

    candidate_hits = candidate_mask.gather(dim=-1, index=top_idx) & top_valid
    exact_hits = exact_mask.gather(dim=-1, index=top_idx) & top_valid
    verified_hits = verified_mask.gather(dim=-1, index=top_idx) & top_valid
    top_stats = {
        "topk_items": float(top_valid.sum().item()),
        "candidate_topk_hits": float(candidate_hits.sum().item()),
        "exact_topk_hits": float(exact_hits.sum().item()),
        "verified_topk_hits": float(verified_hits.sum().item()),
    }
    add_to_stats(cfg.stats, top_stats)
    add_to_stats(cfg.layer_stats[layer_idx], top_stats)

    if cfg.head_report_limit > 0:
        per_head_items = top_valid.float().sum(dim=(0, 2, 3)).detach().cpu()
        per_head_candidate_hits = candidate_hits.float().sum(dim=(0, 2, 3)).detach().cpu()
        per_head_exact_hits = exact_hits.float().sum(dim=(0, 2, 3)).detach().cpu()
        per_head_verified_hits = verified_hits.float().sum(dim=(0, 2, 3)).detach().cpu()
        for head_idx in range(candidate_counts.shape[1]):
            add_to_stats(
                cfg.head_stats[(layer_idx, head_idx)],
                {
                    "topk_items": float(per_head_items[head_idx].item()),
                    "candidate_topk_hits": float(per_head_candidate_hits[head_idx].item()),
                    "exact_topk_hits": float(per_head_exact_hits[head_idx].item()),
                    "verified_topk_hits": float(per_head_verified_hits[head_idx].item()),
                },
            )


def parse_layer_list(value: str, n_layers: int) -> list[int] | None:
    if not value.strip():
        return None

    layers: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    invalid = sorted(layer for layer in layers if layer < 0 or layer >= n_layers)
    if invalid:
        raise ValueError(f"Layer indices out of range for {n_layers} layers: {invalid}")
    return sorted(layers)


def format_layer_list(layers: list[int] | None) -> str:
    if layers is None:
        return "all"
    return ",".join(str(layer) for layer in layers)


def patch_model_with_sva(model: nn.Module, cfg: SVAConfig, layer_indices: list[int] | None = None) -> None:
    selected = None if layer_indices is None else set(layer_indices)
    for layer_idx, layer in enumerate(model.model.layers):
        if selected is None or layer_idx in selected:
            layer.self_attn = SVALlamaAttention(layer.self_attn, cfg)


@torch.no_grad()
def layer_qk_from_hidden(
    model: nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    layer_idx: int,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    layer = model.model.layers[layer_idx]
    # HF Llama hidden_states are layer-boundary states; attention sees the input-layernormed state.
    hidden = layer.input_layernorm(hidden_states[layer_idx])
    hidden_shape = (hidden.shape[0], hidden.shape[1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = repeat_kv(key, layer.self_attn.num_key_value_groups)
    return query.float(), key.float(), float(layer.self_attn.scaling)


def build_three_stage_artifacts(
    model: nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    layer_indices: list[int] | None,
    route_source: str,
    seq_len: int,
    rank_dim: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    coarse_label_topk: int,
    train_query_samples: int,
    min_query_pos: int,
    train_steps: int,
    hard_steps: int,
    hard_pool: int,
    hard_negatives: int,
    hard_margin: float,
    hard_lr_scale: float,
    weighted_boost: float,
    batch_queries: int,
    lr: float,
    weight_decay: float,
    kmeans_iters: int,
    assign_chunk_size: int,
    seed: int,
    device: torch.device,
) -> dict[int, ThreeStageLayerArtifacts]:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    positions = sample_query_positions(seq_len, coarse_label_topk, train_query_samples, min_query_pos)
    position_t = torch.tensor(positions, device=device, dtype=torch.long)
    artifacts: dict[int, ThreeStageLayerArtifacts] = {}

    train_layers = layer_indices if layer_indices is not None else list(range(len(model.model.layers)))
    for layer_idx in train_layers:
        target_query_all, target_key_all, scaling = layer_qk_from_hidden(model, hidden_states, layer_idx, position_ids)
        if target_query_all.shape[0] != 1:
            raise ValueError("Three-stage artifact training currently expects batch size 1.")

        target_query = target_query_all[0]
        target_key = target_key_all[0]
        if route_source == "hidden":
            # Match the live self-attention input, not the pre-norm layer boundary state.
            hidden = model.model.layers[layer_idx].input_layernorm(hidden_states[layer_idx])[0].float()
            route = hidden[None, :, :].expand(target_query.shape[0], -1, -1).contiguous()
            query = route
            key = route
        elif route_source == "qk":
            query = target_query
            key = target_key
        else:
            raise ValueError(f"Unknown route source: {route_source}")

        top_idx, top_valid = topk_indices_for_queries(target_query, target_key, positions, coarse_label_topk, scaling)
        train_query = query[:, position_t, :].contiguous()

        torch.manual_seed(seed + layer_idx * 2000 + rank_dim)
        ranker = LowRankRanker(query.shape[0], query.shape[-1], rank_dim).to(device)
        train_loss = train_ranker(
            ranker,
            key,
            train_query,
            positions,
            top_idx,
            top_valid,
            train_steps,
            batch_queries,
            lr,
            weight_decay,
            seed + layer_idx * 2000 + rank_dim,
        )
        hard_loss = float("nan")
        if hard_steps > 0:
            hard_loss = train_ranker_hard_negatives(
                ranker,
                key,
                train_query,
                positions,
                top_idx,
                top_valid,
                hard_steps,
                batch_queries,
                lr * hard_lr_scale,
                weight_decay,
                seed + layer_idx * 3000 + rank_dim,
                hard_pool,
                hard_negatives,
                hard_margin,
            )

        with torch.no_grad():
            k_low = torch.einsum("hkd,hdr->hkr", key, ranker.k_proj)
            weights = key_label_weights(top_idx, top_valid, seq_len, weighted_boost, device)
            codebooks = fit_weighted_product_codebooks(
                k_low,
                weights,
                coarse_subspaces,
                coarse_codewords,
                kmeans_iters,
                seed + layer_idx * 1000 + rank_dim * 23 + int(weighted_boost * 1000),
                assign_chunk_size,
            )
            artifacts[layer_idx] = ThreeStageLayerArtifacts(
                q_proj=ranker.q_proj.detach().clone(),
                k_proj=ranker.k_proj.detach().clone(),
                logit_scale=ranker.logit_scale.detach().clone(),
                coarse_codebooks=codebooks.detach().clone(),
                train_loss=train_loss,
                hard_loss=hard_loss,
                route_source=route_source,
            )
        print(
            "three_stage_artifact,"
            f"{layer_idx},{rank_dim},{coarse_subspaces},{int(codebooks.shape[2])},"
            f"{train_loss:.6f},{hard_loss:.6f},{route_source}",
            flush=True,
        )
        del target_query_all, target_key_all, target_query, target_key, query, key, train_query, ranker, k_low, weights, codebooks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return artifacts


def make_long_texts(n_texts: int, repeats: int) -> list[str]:
    texts = []
    for sample_idx in range(n_texts):
        paragraphs = [
            TEXTS[(sample_idx + offset) % len(TEXTS)]
            for offset in range(max(repeats, 1) * len(TEXTS))
        ]
        texts.append(" ".join(paragraphs))
    return texts


def load_texts(path: str | None, long_texts: bool, n_texts: int | None, text_repeats: int) -> list[str]:
    if path is None:
        if long_texts:
            return make_long_texts(n_texts or 4, text_repeats)
        return TEXTS[:n_texts] if n_texts is not None else TEXTS
    with open(path, "r", encoding="utf-8") as handle:
        texts = [line.strip() for line in handle if line.strip()]
    if not texts:
        raise ValueError(f"No non-empty texts found in {path}")
    return texts[:n_texts] if n_texts is not None else texts


def encode_batch(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in batch.items()}


def shifted_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compare_logits(
    full_logits: torch.Tensor,
    sva_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, float]:
    valid = attention_mask[:, 1:].bool()
    full = full_logits[:, :-1, :][valid]
    sva = sva_logits[:, :-1, :][valid]
    full_log_probs = F.log_softmax(full.float(), dim=-1)
    sva_log_probs = F.log_softmax(sva.float(), dim=-1)
    full_probs = full_log_probs.exp()
    kl = (full_probs * (full_log_probs - sva_log_probs)).sum(dim=-1).mean()
    top1_agreement = (full.argmax(dim=-1) == sva.argmax(dim=-1)).float().mean()
    cosine = F.cosine_similarity(full.float(), sva.float(), dim=-1).mean()
    return {
        "kl_to_full": float(kl.item()),
        "top1_agreement": float(top1_agreement.item()),
        "logit_cosine": float(cosine.item()),
    }


@torch.no_grad()
def run_model(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    output = model(**batch, use_cache=False)
    return output.logits


def config_from_args(
    args: argparse.Namespace,
    three_stage_artifacts: dict[int, ThreeStageLayerArtifacts],
    diagnose_topk: int | None = None,
    head_report_limit: int | None = None,
) -> SVAConfig:
    return SVAConfig(
        mode=args.mode,
        tables=args.tables,
        bits=args.bits,
        budget=args.budget,
        probe_radius=args.probe_radius,
        seed=args.seed,
        prefilter_dim=args.prefilter_dim,
        prefilter_budget=args.prefilter_budget,
        rank_dim=args.rank_dim,
        coarse_subspaces=args.coarse_subspaces,
        coarse_codewords=args.coarse_codewords,
        coarse_shortlist=args.coarse_shortlist,
        assign_chunk_size=args.assign_chunk_size,
        diagnose_topk=args.diagnose_topk if diagnose_topk is None else diagnose_topk,
        head_report_limit=args.head_report_limit if head_report_limit is None else head_report_limit,
        three_stage_artifacts=three_stage_artifacts,
    )


def build_artifacts_for_hidden_states(
    model: nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    layer_indices: list[int] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, ThreeStageLayerArtifacts]:
    return build_three_stage_artifacts(
        model,
        hidden_states,
        layer_indices,
        args.route_source,
        int(hidden_states[0].shape[1]),
        args.rank_dim,
        args.coarse_subspaces,
        args.coarse_codewords,
        args.coarse_label_topk,
        args.train_query_samples,
        args.min_query_pos,
        args.ranker_train_steps,
        args.coarse_hard_steps,
        args.coarse_hard_pool,
        args.coarse_hard_negatives,
        args.coarse_hard_margin,
        args.coarse_hard_lr_scale,
        args.weighted_boost,
        args.batch_queries,
        args.ranker_lr,
        args.ranker_weight_decay,
        args.kmeans_iters,
        args.assign_chunk_size,
        args.seed,
        device,
    )


def build_progressive_three_stage_artifacts(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    layer_indices: list[int] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, ThreeStageLayerArtifacts]:
    train_layers = layer_indices if layer_indices is not None else list(range(len(model.model.layers)))
    artifacts: dict[int, ThreeStageLayerArtifacts] = {}
    training_cfg = config_from_args(args, artifacts, diagnose_topk=0, head_report_limit=0)
    original_attn: dict[int, nn.Module] = {}

    for layer_idx in train_layers:
        with torch.no_grad():
            output = model(**batch, use_cache=False, output_hidden_states=True)
        if output.hidden_states is None:
            raise ValueError("Progressive artifact training requires hidden states.")

        new_artifacts = build_artifacts_for_hidden_states(model, output.hidden_states, [layer_idx], args, device)
        artifacts.update(new_artifacts)
        original_attn[layer_idx] = model.model.layers[layer_idx].self_attn
        model.model.layers[layer_idx].self_attn = SVALlamaAttention(original_attn[layer_idx], training_cfg)
        print(
            "progressive_socket_patch,"
            f"{layer_idx},{args.route_source},{len(artifacts)}",
            flush=True,
        )
        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for layer_idx, original in original_attn.items():
        model.model.layers[layer_idx].self_attn = original

    return artifacts


def ratio(stats: dict[str, float], numerator: str, denominator: str) -> float:
    total = stats.get(denominator, 0.0)
    if total <= 0:
        return float("nan")
    return stats.get(numerator, 0.0) / total


def print_diagnostics(cfg: SVAConfig) -> None:
    if cfg.diagnose_topk <= 0 or cfg.stats["topk_items"] <= 0:
        return

    print(f"candidate_top{cfg.diagnose_topk}_recall,{ratio(cfg.stats, 'candidate_topk_hits', 'topk_items'):.6f}")
    print(f"exact_top{cfg.diagnose_topk}_recall,{ratio(cfg.stats, 'exact_topk_hits', 'topk_items'):.6f}")
    print(f"verified_top{cfg.diagnose_topk}_recall,{ratio(cfg.stats, 'verified_topk_hits', 'topk_items'):.6f}")
    print(
        "layer_metric_header,layer,avg_summoned,avg_exact_scored,avg_verified,"
        "candidate_topk_recall,exact_topk_recall,verified_topk_recall"
    )
    for layer_idx in sorted(key for key in cfg.layer_stats if isinstance(key, int)):
        stats = cfg.layer_stats[layer_idx]
        avg_summoned = ratio(stats, "summoned", "queries")
        avg_exact = ratio(stats, "exact_scored", "queries")
        avg_verified = ratio(stats, "verified", "queries")
        candidate_recall = ratio(stats, "candidate_topk_hits", "topk_items")
        exact_recall = ratio(stats, "exact_topk_hits", "topk_items")
        verified_recall = ratio(stats, "verified_topk_hits", "topk_items")
        print(
            "layer_metric,"
            f"{layer_idx},{avg_summoned:.3f},{avg_exact:.3f},{avg_verified:.3f},"
            f"{candidate_recall:.6f},{exact_recall:.6f},{verified_recall:.6f}"
        )

    if cfg.head_report_limit <= 0:
        return

    rows = []
    for key, stats in cfg.head_stats.items():
        layer_idx, head_idx = key
        rows.append(
            (
                ratio(stats, "verified_topk_hits", "topk_items"),
                ratio(stats, "candidate_topk_hits", "topk_items"),
                ratio(stats, "exact_topk_hits", "topk_items"),
                layer_idx,
                head_idx,
                ratio(stats, "summoned", "queries"),
                ratio(stats, "exact_scored", "queries"),
                ratio(stats, "verified", "queries"),
            )
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    print(
        "worst_head_header,layer,head,avg_summoned,avg_exact_scored,avg_verified,"
        "candidate_topk_recall,exact_topk_recall,verified_topk_recall"
    )
    for (
        verified_recall,
        candidate_recall,
        exact_recall,
        layer_idx,
        head_idx,
        avg_summoned,
        avg_exact,
        avg_verified,
    ) in rows[: cfg.head_report_limit]:
        print(
            "worst_head,"
            f"{layer_idx},{head_idx},{avg_summoned:.3f},{avg_exact:.3f},{avg_verified:.3f},"
            f"{candidate_recall:.6f},{exact_recall:.6f},{verified_recall:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained LLM SVA socket test.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--long-texts", action="store_true")
    parser.add_argument("--n-texts", type=int, default=None)
    parser.add_argument("--text-repeats", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--mode", choices=["lsh", "three_stage"], default="lsh")
    parser.add_argument(
        "--socket-layers",
        default="",
        help="Comma-separated layers or ranges to replace, for example '0,4,8' or '0-3'. Empty means all layers.",
    )
    parser.add_argument("--route-source", choices=["qk", "hidden"], default="qk")
    parser.add_argument("--artifact-training", choices=["teacher", "progressive"], default="teacher")
    parser.add_argument("--tables", type=int, default=16)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prefilter-dim", type=int, default=0)
    parser.add_argument("--prefilter-budget", type=int, default=0)
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--coarse-subspaces", type=int, default=4)
    parser.add_argument("--coarse-codewords", type=int, default=64)
    parser.add_argument("--coarse-shortlist", type=int, default=1024)
    parser.add_argument("--coarse-label-topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--ranker-train-steps", type=int, default=160)
    parser.add_argument("--coarse-hard-steps", type=int, default=80)
    parser.add_argument("--coarse-hard-pool", type=int, default=512)
    parser.add_argument("--coarse-hard-negatives", type=int, default=64)
    parser.add_argument("--coarse-hard-margin", type=float, default=1.0)
    parser.add_argument("--coarse-hard-lr-scale", type=float, default=0.5)
    parser.add_argument("--weighted-boost", type=float, default=4.0)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--ranker-lr", type=float, default=0.003)
    parser.add_argument("--ranker-weight-decay", type=float, default=0.0001)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--diagnose-topk", type=int, default=0)
    parser.add_argument("--head-report-limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    dtype_map = {
        "auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    texts = load_texts(args.text_file, args.long_texts, args.n_texts, args.text_repeats)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    batch = encode_batch(tokenizer, texts, args.max_length, device)
    if args.mode == "three_stage" and batch["input_ids"].shape[0] != 1:
        raise ValueError("Three-stage socket mode currently expects one text. Pass --n-texts 1.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    with torch.no_grad():
        full_output = model(
            **batch,
            use_cache=False,
            output_hidden_states=args.mode == "three_stage",
        )
    full_logits = full_output.logits
    full_loss = shifted_loss(full_logits, batch["input_ids"], batch["attention_mask"])
    three_stage_artifacts: dict[int, ThreeStageLayerArtifacts] = {}
    if args.mode == "three_stage":
        if args.artifact_training == "progressive":
            three_stage_artifacts = build_progressive_three_stage_artifacts(model, batch, socket_layers, args, device)
        else:
            if full_output.hidden_states is None:
                raise ValueError("Three-stage mode requires hidden states from the full model pass.")
            three_stage_artifacts = build_artifacts_for_hidden_states(
                model,
                full_output.hidden_states,
                socket_layers,
                args,
                device,
            )

    cfg = config_from_args(args, three_stage_artifacts)
    patch_model_with_sva(model, cfg, socket_layers)
    sva_logits = run_model(model, batch)
    sva_loss = shifted_loss(sva_logits, batch["input_ids"], batch["attention_mask"])
    metrics = compare_logits(full_logits, sva_logits, batch["attention_mask"])

    avg_summoned = cfg.stats["summoned"] / max(cfg.stats["queries"], 1.0)
    avg_exact = cfg.stats["exact_scored"] / max(cfg.stats["queries"], 1.0)
    avg_verified = cfg.stats["verified"] / max(cfg.stats["queries"], 1.0)
    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"n_texts,{len(texts)}")
    print(f"seq_len,{batch['input_ids'].shape[1]}")
    print(f"mode,{args.mode}")
    print(f"socket_layers,{format_layer_list(socket_layers)}")
    print(f"socket_layer_count,{len(socket_layers) if socket_layers is not None else len(model.model.layers)}")
    print(f"socket_layers_text,{format_layer_list(socket_layers).replace(',', ';')}")
    print(f"route_source,{args.route_source}")
    print(f"artifact_training,{args.artifact_training}")
    print(f"tables,{args.tables}")
    print(f"bits,{args.bits}")
    print(f"budget,{args.budget}")
    print(f"probe_radius,{args.probe_radius}")
    print(f"prefilter_dim,{args.prefilter_dim}")
    print(f"prefilter_budget,{args.prefilter_budget}")
    if args.mode == "three_stage":
        print(f"rank_dim,{args.rank_dim}")
        print(f"coarse_subspaces,{args.coarse_subspaces}")
        print(f"coarse_codewords,{args.coarse_codewords}")
        print(f"coarse_shortlist,{args.coarse_shortlist}")
        print(f"coarse_label_topk,{args.coarse_label_topk}")
        print(f"ranker_train_steps,{args.ranker_train_steps}")
        print(f"coarse_hard_steps,{args.coarse_hard_steps}")
        print(f"coarse_hard_pool,{args.coarse_hard_pool}")
        print(f"weighted_boost,{args.weighted_boost:g}")
    print(f"full_loss,{full_loss.item():.6f}")
    print(f"sva_loss,{sva_loss.item():.6f}")
    print(f"loss_delta,{(sva_loss - full_loss).item():.6f}")
    for key, value in metrics.items():
        print(f"{key},{value:.6f}")
    print(f"avg_summoned,{avg_summoned:.3f}")
    print(f"avg_verified,{avg_verified:.3f}")
    print(f"avg_exact_scored,{avg_exact:.3f}")
    print(f"avg_postscore_attended,{avg_verified:.3f}")
    print_diagnostics(cfg)


if __name__ == "__main__":
    main()
