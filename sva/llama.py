"""Llama-family drop-in attention adapter for Summon-Verify Attention."""

from __future__ import annotations

import math
import time
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
    summon_mode: str = "scan"
    inverted_cells_per_subspace: int = 8
    adaptive_min_budget: int | None = None
    adaptive_mid_budget: int | None = None
    adaptive_low_margin: float = 0.35
    adaptive_high_margin: float = 0.70
    profile_components: bool = False
    static_tail_rebuild_interval: int = 64


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

        self.register_buffer("sva_q_proj", layer_artifacts.q_proj.detach().float().clone(), persistent=False)
        self.register_buffer("sva_k_proj", layer_artifacts.k_proj.detach().float().clone(), persistent=False)
        self.register_buffer("sva_logit_scale", layer_artifacts.logit_scale.detach().float().clone(), persistent=False)
        self.register_buffer(
            "sva_scale",
            layer_artifacts.logit_scale.detach().float().exp().clamp(0.01, 100.0).clone(),
            persistent=False,
        )
        self.register_buffer("sva_coarse_codebooks", layer_artifacts.coarse_codebooks.detach().float().clone(), persistent=False)
        self._cached_k_low: torch.Tensor | None = None
        self._cached_coarse_codes: torch.Tensor | None = None
        self._cached_postings: torch.Tensor | None = None
        self._cached_posting_counts: torch.Tensor | None = None
        self._cached_postings_key_len = 0
        self._cached_key_len = 0
        self._cached_signature: tuple[torch.device, torch.dtype, int, int, int] | None = None

    @staticmethod
    def _top_unique_candidates(
        candidate_idx: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_valid: torch.Tensor,
        budget: int,
        refill_factor: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        candidate_width = int(candidate_idx.shape[1])
        if candidate_width == 0 or budget <= 0:
            empty_idx = candidate_idx[:, :0]
            empty_scores = candidate_scores[:, :0]
            empty_valid = candidate_valid[:, :0]
            return empty_idx, empty_scores, empty_valid, 0

        refill_count = min(candidate_width, max(budget, budget * refill_factor))
        masked_scores = candidate_scores.masked_fill(~candidate_valid, torch.finfo(candidate_scores.dtype).min)
        refill_scores, refill_order = masked_scores.topk(refill_count, dim=-1)
        refill_idx = candidate_idx.gather(dim=1, index=refill_order)
        refill_valid = candidate_valid.gather(dim=1, index=refill_order)

        same_idx = refill_idx[:, :, None] == refill_idx[:, None, :]
        prior = torch.triu(torch.ones(refill_count, refill_count, device=candidate_idx.device, dtype=torch.bool), diagonal=1)
        duplicate = (same_idx & prior[None, :, :]).any(dim=1)
        unique_valid = refill_valid & ~duplicate

        selected_count = min(budget, refill_count)
        unique_scores = refill_scores.masked_fill(~unique_valid, torch.finfo(refill_scores.dtype).min)
        selected_scores, selected_order = unique_scores.topk(selected_count, dim=-1)
        selected_idx = refill_idx.gather(dim=1, index=selected_order)
        selected_valid = unique_valid.gather(dim=1, index=selected_order)
        return selected_idx, selected_scores, selected_valid, refill_count

    @staticmethod
    def _profile_now(device: torch.device) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    @staticmethod
    def _profile_elapsed_ms(device: torch.device, start: float) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - start) * 1000.0

    def reset_catalog(self) -> None:
        self._cached_k_low = None
        self._cached_coarse_codes = None
        self._cached_postings = None
        self._cached_posting_counts = None
        self._cached_postings_key_len = 0
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

        profile_static = self.serving.profile_components and self.serving.summon_mode == "inverted_static" and q_len == 1
        outer_timings: dict[str, float] = {}

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

        outer_start = self._profile_now(query.device) if profile_static else 0.0
        stage_start = outer_start
        q_proj = self.sva_q_proj
        k_proj = self.sva_k_proj
        scale = self.sva_scale
        codebooks = self.sva_coarse_codebooks
        if q_proj.device != query.device:
            q_proj = q_proj.to(device=query.device)
            k_proj = k_proj.to(device=query.device)
            scale = scale.to(device=query.device)
            codebooks = codebooks.to(device=query.device)
        q_low = torch.einsum("bhtd,hdr->bhtr", query.float(), q_proj) * scale[None, :, None, None]
        if profile_static:
            outer_timings["static_projection_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            stage_start = self._profile_now(query.device)
        k_low, coarse_codes = self._key_catalog(key_states, k_proj, codebooks, q_len)
        if profile_static:
            outer_timings["static_key_catalog_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            stage_start = self._profile_now(query.device)
        actual_shortlist = min(self.serving.coarse_shortlist, k_len)
        actual_budget = min(self.serving.budget, actual_shortlist)
        query_chunk_size = self.serving.query_chunk_size
        if query_chunk_size is None:
            query_chunk_size = 128 if (q_len >= 4096 or actual_shortlist >= 2048) else q_len

        if self.serving.summon_mode == "inverted_static" and q_len == 1:
            output = self._inverted_static_decode_attention(
                query,
                key_states,
                value_states,
                q_low,
                k_low,
                coarse_codes,
                codebooks,
                allowed,
                actual_budget,
            )
            if profile_static and self.stats is not None:
                outer_timings["static_outer_total_ms"] = self._profile_elapsed_ms(query.device, outer_start)
                self.stats.add(
                    int(self.layer_idx or 0),
                    {
                        "profile_outer_calls": 1.0,
                        **outer_timings,
                    },
                )
            return output
        if self.serving.summon_mode == "inverted" and q_len == 1:
            return self._inverted_decode_attention(
                query,
                key_states,
                value_states,
                q_low,
                k_low,
                coarse_codes,
                codebooks,
                allowed,
                actual_budget,
            )
        if self.serving.summon_mode not in {"scan", "inverted", "inverted_static"}:
            raise ValueError(f"Unknown SVA summon_mode: {self.serving.summon_mode!r}")

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

    def _inverted_decode_attention(
        self,
        query: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_low: torch.Tensor,
        k_low: torch.Tensor,
        coarse_codes: torch.Tensor,
        codebooks: torch.Tensor,
        allowed: torch.Tensor,
        max_budget: int,
    ) -> torch.Tensor:
        batch, n_heads, q_len, head_dim = query.shape
        k_len = key_states.shape[2]
        if batch != 1 or q_len != 1:
            raise ValueError("Inverted SVA decode expects batch size 1 and q_len 1.")

        min_budget = min(self.serving.adaptive_min_budget or max_budget, max_budget, k_len)
        mid_budget = min(self.serving.adaptive_mid_budget or max(min_budget, max_budget // 2), max_budget, k_len)
        max_budget = min(max_budget, k_len)
        cells_per_subspace = min(self.serving.inverted_cells_per_subspace, int(codebooks.shape[2]))
        subspaces = int(codebooks.shape[1])
        sub_dim = self.serving.rank_dim // subspaces
        if self._cached_postings is None or self._cached_posting_counts is None:
            self._rebuild_postings(coarse_codes, int(codebooks.shape[2]))
        assert self._cached_postings is not None
        assert self._cached_posting_counts is not None
        output_heads: list[torch.Tensor] = []

        total_summoned = 0.0
        total_verified = 0.0
        total_cell_visits = 0.0

        for head_idx in range(n_heads):
            qh_low = q_low[0, head_idx, 0].float()
            q_parts = qh_low.reshape(subspaces, sub_dim)
            code_scores = torch.einsum("sd,scd->sc", q_parts, codebooks[head_idx].float()) / math.sqrt(
                self.serving.rank_dim
            )
            top_cells = code_scores.topk(cells_per_subspace, dim=-1).indices
            subspace_ids = torch.arange(subspaces, device=query.device)[:, None]
            posting_lists = self._cached_postings[head_idx][subspace_ids, top_cells]
            posting_counts = self._cached_posting_counts[head_idx][subspace_ids, top_cells]
            slots = torch.arange(posting_lists.shape[-1], device=query.device)
            posting_valid = slots[None, None, :] < posting_counts[..., None]
            candidate_idx = posting_lists[posting_valid]

            allowed_head = allowed[0, head_idx, 0]
            if int(candidate_idx.numel()) > 0:
                candidate_idx = candidate_idx[allowed_head[candidate_idx]]
            current_idx = torch.tensor([k_len - 1], device=query.device, dtype=torch.long)
            if bool(allowed_head[-1].item()):
                candidate_idx = torch.cat([candidate_idx, current_idx])
            if int(candidate_idx.numel()) > 0:
                candidate_idx = torch.unique(candidate_idx)
            if int(candidate_idx.numel()) == 0:
                candidate_idx = allowed_head.nonzero(as_tuple=False).flatten()[-1:]

            candidate_scores = (
                k_low[0, head_idx, candidate_idx].float() * qh_low[None, :]
            ).sum(dim=-1) / math.sqrt(self.serving.rank_dim)
            rank_count = min(max_budget, int(candidate_idx.numel()))
            rank_scores, rank_order = candidate_scores.topk(rank_count, dim=-1)
            margin_ref = min(min_budget, rank_count) - 1
            margin = float((rank_scores[0] - rank_scores[margin_ref]).detach().item()) if rank_count > 1 else float("inf")
            if margin >= self.serving.adaptive_high_margin:
                budget = min(min_budget, rank_count)
            elif margin >= self.serving.adaptive_low_margin:
                budget = min(mid_budget, rank_count)
            else:
                budget = rank_count

            selected_idx = candidate_idx[rank_order[:budget]]
            selected_keys = key_states[0, head_idx, selected_idx]
            selected_values = value_states[0, head_idx, selected_idx]
            selected_scores = (selected_keys.float() * query[0, head_idx, 0].float()[None, :]).sum(dim=-1) * self.scaling
            weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query.dtype)
            output_heads.append((weights[:, None] * selected_values).sum(dim=0))

            total_summoned += float(candidate_idx.numel())
            total_verified += float(budget)
            total_cell_visits += float(subspaces * cells_per_subspace)

        if self.stats is not None:
            self.stats.add(
                int(self.layer_idx or 0),
                {
                    "summoned": total_summoned,
                    "exact_scored": total_verified,
                    "verified": total_verified,
                    "queries": float(n_heads),
                    "cell_visits": total_cell_visits,
                },
            )

        return torch.stack(output_heads, dim=0)[None, :, None, :]

    def _inverted_static_decode_attention(
        self,
        query: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_low: torch.Tensor,
        k_low: torch.Tensor,
        coarse_codes: torch.Tensor,
        codebooks: torch.Tensor,
        allowed: torch.Tensor,
        max_budget: int,
    ) -> torch.Tensor:
        batch, n_heads, q_len, head_dim = query.shape
        k_len = key_states.shape[2]
        if batch != 1 or q_len != 1:
            raise ValueError("Static inverted SVA decode expects batch size 1 and q_len 1.")

        profile = self.serving.profile_components
        timings: dict[str, float] = {}
        total_start = self._profile_now(query.device) if profile else 0.0
        stage_start = total_start

        min_budget = min(self.serving.adaptive_min_budget or max_budget, max_budget, k_len)
        mid_budget = min(self.serving.adaptive_mid_budget or max(min_budget, max_budget // 2), max_budget, k_len)
        max_budget = min(max_budget, k_len)
        cells_per_subspace = min(self.serving.inverted_cells_per_subspace, int(codebooks.shape[2]))
        subspaces = int(codebooks.shape[1])
        sub_dim = self.serving.rank_dim // subspaces
        if self._cached_postings is None or self._cached_posting_counts is None:
            self._rebuild_postings(coarse_codes, int(codebooks.shape[2]))
        assert self._cached_postings is not None
        assert self._cached_posting_counts is not None

        qh_low = q_low[0, :, 0].float()
        q_parts = qh_low.reshape(n_heads, subspaces, sub_dim)
        code_scores = torch.einsum("hsd,hscd->hsc", q_parts, codebooks.float()) / math.sqrt(self.serving.rank_dim)
        top_cells = code_scores.topk(cells_per_subspace, dim=-1).indices

        head_ids = torch.arange(n_heads, device=query.device)[:, None, None]
        subspace_ids = torch.arange(subspaces, device=query.device)[None, :, None]
        posting_lists = self._cached_postings[head_ids, subspace_ids, top_cells]
        posting_counts = self._cached_posting_counts[head_ids, subspace_ids, top_cells]
        slots = torch.arange(posting_lists.shape[-1], device=query.device)
        posting_valid = slots[None, None, None, :] < posting_counts[..., None]

        candidate_idx = posting_lists.reshape(n_heads, -1)
        candidate_valid = posting_valid.reshape(n_heads, -1)
        allowed_heads = allowed[0, :, 0]
        candidate_valid = candidate_valid & allowed_heads.gather(dim=1, index=candidate_idx.clamp(0, k_len - 1))

        tail_start = min(max(int(self._cached_postings_key_len), 0), k_len)
        if tail_start < k_len:
            tail_idx = torch.arange(tail_start, k_len, device=query.device)[None, :].expand(n_heads, -1)
            candidate_idx = torch.cat([candidate_idx, tail_idx], dim=1)
            candidate_valid = torch.cat([candidate_valid, allowed_heads[:, tail_start:k_len]], dim=1)
        else:
            current_idx = torch.full((n_heads, 1), k_len - 1, device=query.device, dtype=torch.long)
            candidate_idx = torch.cat([candidate_idx, current_idx], dim=1)
            candidate_valid = torch.cat([candidate_valid, allowed_heads[:, -1:]], dim=1)

        if profile:
            timings["static_catalog_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            stage_start = self._profile_now(query.device)

        candidate_low = k_low[0].float().gather(
            dim=1,
            index=candidate_idx[..., None].expand(n_heads, candidate_idx.shape[1], self.serving.rank_dim),
        )
        candidate_scores = (candidate_low * qh_low[:, None, :]).sum(dim=-1) / math.sqrt(self.serving.rank_dim)
        selected_idx, rank_scores, selected_valid, refill_count = self._top_unique_candidates(
            candidate_idx,
            candidate_scores,
            candidate_valid,
            max_budget,
        )
        rank_count = int(selected_idx.shape[1])

        if profile:
            timings["static_refill_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            stage_start = self._profile_now(query.device)

        if rank_count > 1:
            margin_ref = min(min_budget, rank_count) - 1
            margins = rank_scores[:, 0] - rank_scores[:, margin_ref]
            budgets = torch.full((n_heads,), max_budget, device=query.device, dtype=torch.long)
            budgets = torch.where(
                margins >= self.serving.adaptive_high_margin,
                torch.full_like(budgets, min_budget),
                budgets,
            )
            budgets = torch.where(
                (margins < self.serving.adaptive_high_margin) & (margins >= self.serving.adaptive_low_margin),
                torch.full_like(budgets, mid_budget),
                budgets,
            ).clamp(max=rank_count)
        else:
            budgets = torch.ones(n_heads, device=query.device, dtype=torch.long)

        budget_valid = torch.arange(rank_count, device=query.device)[None, :] < budgets[:, None]
        selected_valid = selected_valid & budget_valid

        if profile:
            timings["static_budget_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            stage_start = self._profile_now(query.device)

        selected_keys = key_states[0].gather(
            dim=1,
            index=selected_idx[..., None].expand(n_heads, rank_count, head_dim),
        )
        selected_values = value_states[0].gather(
            dim=1,
            index=selected_idx[..., None].expand(n_heads, rank_count, head_dim),
        )

        if profile:
            timings["static_gather_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            stage_start = self._profile_now(query.device)

        selected_scores = (selected_keys.float() * query[0, :, 0, None, :].float()).sum(dim=-1) * self.scaling
        selected_scores = selected_scores.masked_fill(~selected_valid, torch.finfo(selected_scores.dtype).min)

        if profile:
            timings["static_exact_score_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            stage_start = self._profile_now(query.device)

        weights = F.softmax(selected_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        output = (weights[..., None] * selected_values).sum(dim=1)

        if profile:
            timings["static_aggregate_ms"] = self._profile_elapsed_ms(query.device, stage_start)
            timings["static_total_ms"] = self._profile_elapsed_ms(query.device, total_start)

        if self.stats is not None:
            values = {
                "summoned": float(candidate_valid.float().sum().item()),
                "refill_pool": float(refill_count * n_heads),
                "exact_scored": float(selected_valid.float().sum().item()),
                "verified": float(selected_valid.float().sum().item()),
                "queries": float(n_heads),
                "cell_visits": float(n_heads * subspaces * cells_per_subspace),
            }
            if profile:
                values["profile_calls"] = 1.0
                values.update(timings)
            self.stats.add(
                int(self.layer_idx or 0),
                values,
            )

        return output[None, :, None, :]

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
            k_low = torch.cat([self._cached_k_low, new_k_low], dim=2)
            if self.serving.summon_mode == "inverted":
                new_codes = encode_product_keys(new_k_low[0], codebooks, self.serving.assign_chunk_size)
                coarse_codes = torch.cat([self._cached_coarse_codes, new_codes], dim=1)
                self._append_postings(new_codes, start, int(codebooks.shape[2]))
            elif self.serving.summon_mode == "inverted_static" and q_len == 1:
                coarse_codes = self._cached_coarse_codes
                rebuild_interval = max(1, int(self.serving.static_tail_rebuild_interval))
                tail_len = k_len - self._cached_postings_key_len
                if self._cached_postings is None or tail_len >= rebuild_interval:
                    coarse_codes = encode_product_keys(k_low[0], codebooks, self.serving.assign_chunk_size)
                    self._rebuild_postings(coarse_codes, int(codebooks.shape[2]))
            else:
                new_codes = encode_product_keys(new_k_low[0], codebooks, self.serving.assign_chunk_size)
                coarse_codes = torch.cat([self._cached_coarse_codes, new_codes], dim=1)
        else:
            k_low = torch.einsum("bhsd,hdr->bhsr", key_states.float(), k_proj)
            coarse_codes = encode_product_keys(k_low[0], codebooks, self.serving.assign_chunk_size)
            if self.serving.summon_mode in {"inverted", "inverted_static"}:
                self._rebuild_postings(coarse_codes, int(codebooks.shape[2]))

        self._cached_k_low = k_low.detach()
        self._cached_coarse_codes = coarse_codes.detach()
        self._cached_key_len = k_len
        self._cached_signature = signature
        return k_low, coarse_codes

    @torch.no_grad()
    def _rebuild_postings(self, coarse_codes: torch.Tensor, codewords: int) -> None:
        n_heads, k_len, subspaces = coarse_codes.shape
        counts = torch.zeros(n_heads, subspaces, codewords, device=coarse_codes.device, dtype=torch.long)
        for head_idx in range(n_heads):
            for subspace_idx in range(subspaces):
                labels = coarse_codes[head_idx, :, subspace_idx].long()
                counts[head_idx, subspace_idx] = torch.bincount(labels, minlength=codewords)

        max_bucket = max(1, int(counts.max().item()))
        postings = torch.zeros(n_heads, subspaces, codewords, max_bucket, device=coarse_codes.device, dtype=torch.long)
        for head_idx in range(n_heads):
            for subspace_idx in range(subspaces):
                labels = coarse_codes[head_idx, :, subspace_idx].long()
                order = torch.argsort(labels)
                head_counts = counts[head_idx, subspace_idx]
                offsets = torch.cat(
                    [
                        torch.zeros(1, device=coarse_codes.device, dtype=torch.long),
                        head_counts.cumsum(dim=0),
                    ]
                )
                for codeword in range(codewords):
                    count = int(head_counts[codeword].item())
                    if count:
                        start = int(offsets[codeword].item())
                        end = start + count
                        postings[head_idx, subspace_idx, codeword, :count] = order[start:end]

        self._cached_postings = postings
        self._cached_posting_counts = counts
        self._cached_postings_key_len = k_len

    @torch.no_grad()
    def _append_postings(self, new_codes: torch.Tensor, start: int, codewords: int) -> None:
        if self._cached_postings is None or self._cached_posting_counts is None:
            self._rebuild_postings(self._cached_coarse_codes, codewords)
            return

        n_heads, new_len, subspaces = new_codes.shape
        needed = self._cached_posting_counts.clone()
        for offset in range(new_len):
            for head_idx in range(n_heads):
                for subspace_idx in range(subspaces):
                    codeword = int(new_codes[head_idx, offset, subspace_idx].item())
                    needed[head_idx, subspace_idx, codeword] += 1

        max_needed = int(needed.max().item())
        if max_needed > int(self._cached_postings.shape[-1]):
            next_width = max(max_needed, int(self._cached_postings.shape[-1]) * 2)
            expanded = torch.zeros(
                *self._cached_postings.shape[:-1],
                next_width,
                device=self._cached_postings.device,
                dtype=self._cached_postings.dtype,
            )
            expanded[..., : self._cached_postings.shape[-1]] = self._cached_postings
            self._cached_postings = expanded

        for offset in range(new_len):
            position = start + offset
            for head_idx in range(n_heads):
                for subspace_idx in range(subspaces):
                    codeword = int(new_codes[head_idx, offset, subspace_idx].item())
                    count = int(self._cached_posting_counts[head_idx, subspace_idx, codeword].item())
                    self._cached_postings[head_idx, subspace_idx, codeword, count] = position
                    self._cached_posting_counts[head_idx, subspace_idx, codeword] += 1
        self._cached_postings_key_len = start + new_len


class SVALlamaPatcher:
    """Reversible patch handle for Llama-family Hugging Face models."""

    def __init__(
        self,
        model: nn.Module,
        bundle: SVAArtifactBundle,
        shortlist: int | None = None,
        budget: int | None = None,
        assign_chunk_size: int = 8192,
        query_chunk_size: int | None = None,
        summon_mode: str = "scan",
        inverted_cells_per_subspace: int = 8,
        adaptive_min_budget: int | None = None,
        adaptive_mid_budget: int | None = None,
        adaptive_low_margin: float = 0.35,
        adaptive_high_margin: float = 0.70,
        profile_components: bool = False,
        static_tail_rebuild_interval: int = 64,
        layers: list[int] | None = None,
    ) -> None:
        self.model = model
        self.bundle = bundle
        self.stats = SVAStats()
        self.originals: dict[int, nn.Module] = {}
        self.layers = None if layers is None else list(dict.fromkeys(int(layer_idx) for layer_idx in layers))
        self.serving = SVALlamaServingConfig(
            rank_dim=bundle.rank_dim,
            coarse_shortlist=bundle.default_shortlist if shortlist is None else int(shortlist),
            budget=bundle.default_budget if budget is None else int(budget),
            assign_chunk_size=assign_chunk_size,
            query_chunk_size=query_chunk_size,
            summon_mode=summon_mode,
            inverted_cells_per_subspace=inverted_cells_per_subspace,
            adaptive_min_budget=adaptive_min_budget,
            adaptive_mid_budget=adaptive_mid_budget,
            adaptive_low_margin=adaptive_low_margin,
            adaptive_high_margin=adaptive_high_margin,
            profile_components=profile_components,
            static_tail_rebuild_interval=static_tail_rebuild_interval,
        )

    def patch(self) -> "SVALlamaPatcher":
        model_layers = getattr(getattr(self.model, "model", None), "layers", None)
        if model_layers is None:
            raise ValueError("Expected a Hugging Face Llama-style model with model.layers.")
        if self.originals:
            return self

        if self.bundle.layer_count != len(model_layers):
            raise ValueError(f"Artifact has {self.bundle.layer_count} layers but model has {len(model_layers)}.")
        model_id = self.bundle.model_id
        model_name = getattr(getattr(self.model, "config", None), "_name_or_path", None)
        if model_id and model_name and str(model_name) not in {"", model_id}:
            raise ValueError(f"Artifact model_id={model_id!r} does not match loaded model {model_name!r}.")

        patch_layers = self.layers if self.layers is not None else list(range(len(model_layers)))
        for layer_idx in patch_layers:
            if layer_idx < 0 or layer_idx >= len(model_layers):
                raise ValueError(f"Patch layer index {layer_idx} is outside model layer range 0..{len(model_layers) - 1}.")
            layer = model_layers[layer_idx]
            artifacts = self.bundle.layers.get(layer_idx)
            if artifacts is None:
                raise ValueError(f"Missing SVA artifacts for layer {layer_idx}.")
            self._validate_layer_shapes(layer_idx, layer.self_attn, artifacts)
            self.originals[layer_idx] = layer.self_attn
            replacement = SVALlamaAttention(layer.self_attn, artifacts, self.serving, self.stats)
            try:
                replacement_device = next(layer.self_attn.parameters()).device
            except StopIteration:
                replacement_device = artifacts.q_proj.device
            replacement.to(device=replacement_device)
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
            attention = self._unwrap_sva_attention(layer.self_attn)
            if isinstance(attention, SVALlamaAttention):
                attention.reset_catalog()

    def _unwrap_sva_attention(self, attention: nn.Module) -> nn.Module:
        while hasattr(attention, "sva_attn"):
            attention = getattr(attention, "sva_attn")
        return attention

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
    query_chunk_size: int | None = None,
    summon_mode: str = "scan",
    inverted_cells_per_subspace: int = 8,
    adaptive_min_budget: int | None = None,
    adaptive_mid_budget: int | None = None,
    adaptive_low_margin: float = 0.35,
    adaptive_high_margin: float = 0.70,
    profile_components: bool = False,
    static_tail_rebuild_interval: int = 64,
    layers: list[int] | None = None,
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
        query_chunk_size=query_chunk_size,
        summon_mode=summon_mode,
        inverted_cells_per_subspace=inverted_cells_per_subspace,
        adaptive_min_budget=adaptive_min_budget,
        adaptive_mid_budget=adaptive_mid_budget,
        adaptive_low_margin=adaptive_low_margin,
        adaptive_high_margin=adaptive_high_margin,
        profile_components=profile_components,
        static_tail_rebuild_interval=static_tail_rebuild_interval,
        layers=layers,
    ).patch()
