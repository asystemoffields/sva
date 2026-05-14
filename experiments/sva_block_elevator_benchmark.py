"""Block-first SVA benchmark.

This tests the "elevator" idea: summon contiguous blocks, then run exact
attention inside those blocks. It also tests a "statements from seats" serving
shape: selected blocks produce local softmax partials that are merged into the
final output, avoiding a token-level gather into a separate candidate pile.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import SVAArtifactBundle, load_sva_artifact_bundle
from sva.ops import encode_product_keys, product_quantized_scores
from sva_full_deployment_benchmark import EVAL_DOCS, repeated_document
from sva_pretrained_socket_test import encode_batch, parse_layer_list


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def comma_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def emit(prefix: str, row: dict[str, float | int | str]) -> None:
    print(prefix + "," + ",".join(f"{key}={row_value(value)}" for key, value in row.items()), flush=True)


def sample_query_positions(seq_len: int, samples: int, min_pos: int) -> torch.Tensor:
    start = min(max(min_pos, 0), seq_len - 1)
    positions = np.linspace(start, seq_len - 1, num=min(samples, seq_len - start), dtype=np.int64)
    return torch.tensor(np.unique(positions), dtype=torch.long)


@torch.no_grad()
def layer_qkv_from_hidden(
    model: Any,
    hidden_states: tuple[torch.Tensor, ...],
    layer_idx: int,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    layer = model.model.layers[layer_idx]
    hidden = layer.input_layernorm(hidden_states[layer_idx])
    hidden_shape = (hidden.shape[0], hidden.shape[1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = repeat_kv(key, layer.self_attn.num_key_value_groups)
    value = repeat_kv(value, layer.self_attn.num_key_value_groups)
    return query[0].float(), key[0].float(), value[0].float(), float(layer.self_attn.scaling)


@torch.no_grad()
def make_synthetic_kv_bank(
    base_key: torch.Tensor,
    base_value: torch.Tensor,
    context: int,
    noise_std: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, base_len, _ = base_key.shape
    generator = torch.Generator(device=base_key.device)
    generator.manual_seed(seed + context * 17 + heads * 101)
    source_idx = torch.randint(base_len, (context,), device=base_key.device, generator=generator)
    key = base_key[:, source_idx, :].clone()
    value = base_value[:, source_idx, :].clone()
    if noise_std > 0:
        key = key + torch.randn(key.shape, device=key.device, dtype=key.dtype, generator=generator) * noise_std
        value = value + torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator) * noise_std
    return key, value


@torch.no_grad()
def exact_attention_streaming(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, queries, head_dim = query.shape
    running_max = torch.full((heads, queries), -float("inf"), device=query.device)
    running_sum = torch.zeros((heads, queries), device=query.device)
    running_out = torch.zeros((heads, queries, head_dim), device=query.device)
    top_scores = torch.full((heads, queries, 0), -float("inf"), device=query.device)
    top_indices = torch.empty((heads, queries, 0), device=query.device, dtype=torch.long)
    topk = 16

    for start in range(0, key.shape[1], chunk_size):
        end = min(start + chunk_size, key.shape[1])
        scores = torch.einsum("hqd,hkd->hqk", query.float(), key[:, start:end, :].float()) * scaling
        local_max = scores.max(dim=-1).values
        new_max = torch.maximum(running_max, local_max)
        old_scale = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max[..., None])
        local_sum = weights.sum(dim=-1)
        local_out = torch.einsum("hqk,hkd->hqd", weights, value[:, start:end, :].float())
        running_out = running_out * old_scale[..., None] + local_out
        running_sum = running_sum * old_scale + local_sum
        running_max = new_max

        chunk_k = min(topk, scores.shape[-1])
        chunk_scores, chunk_idx = scores.topk(chunk_k, dim=-1)
        chunk_idx = chunk_idx + start
        merged_scores = torch.cat([top_scores, chunk_scores], dim=-1)
        merged_idx = torch.cat([top_indices, chunk_idx], dim=-1)
        keep_k = min(topk, merged_scores.shape[-1])
        top_scores, order = merged_scores.topk(keep_k, dim=-1)
        top_indices = merged_idx.gather(dim=-1, index=order)

    return running_out / running_sum.clamp_min(1e-20)[..., None], top_indices


@torch.no_grad()
def project_catalog(
    query: torch.Tensor,
    key: torch.Tensor,
    bundle: SVAArtifactBundle,
    layer_idx: int,
    assign_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    artifact = bundle.layers[layer_idx]
    q_proj = artifact.q_proj.to(device=query.device, dtype=torch.float32)
    k_proj = artifact.k_proj.to(device=query.device, dtype=torch.float32)
    scale = artifact.logit_scale.to(device=query.device, dtype=torch.float32).exp().clamp(0.01, 100.0)
    q_low = torch.einsum("hqd,hdr->hqr", query.float(), q_proj) * scale[:, None, None]
    k_low = torch.einsum("hkd,hdr->hkr", key.float(), k_proj)
    codebooks = artifact.coarse_codebooks.to(device=query.device, dtype=torch.float32)
    codes = encode_product_keys(k_low, codebooks, assign_chunk_size)
    return q_low, k_low, codebooks, codes


def pad_to_blocks(tensor: torch.Tensor, block_size: int, pad_value: float) -> tuple[torch.Tensor, int]:
    context = tensor.shape[-1]
    blocks = (context + block_size - 1) // block_size
    padded = blocks * block_size
    if padded == context:
        return tensor, blocks
    pad_shape = (*tensor.shape[:-1], padded - context)
    pad = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1), blocks


@torch.no_grad()
def block_scores(
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
    block_size: int,
    mode: str,
) -> torch.Tensor:
    heads, queries, _ = q_low.shape
    if mode == "coarse_max":
        token_scores = product_quantized_scores(q_low, codebooks, codes, rank_dim)
        padded, blocks = pad_to_blocks(token_scores, block_size, -float("inf"))
        return padded.view(heads, queries, blocks, block_size).max(dim=-1).values
    if mode == "centroid":
        context = k_low.shape[1]
        blocks = (context + block_size - 1) // block_size
        padded_len = blocks * block_size
        if padded_len != context:
            pad = torch.zeros(k_low.shape[0], padded_len - context, k_low.shape[2], device=k_low.device, dtype=k_low.dtype)
            k_work = torch.cat([k_low, pad], dim=1)
            mask = torch.cat(
                [
                    torch.ones(context, device=k_low.device, dtype=torch.float32),
                    torch.zeros(padded_len - context, device=k_low.device, dtype=torch.float32),
                ],
            )
        else:
            k_work = k_low
            mask = torch.ones(context, device=k_low.device, dtype=torch.float32)
        block_keys = k_work.view(k_low.shape[0], blocks, block_size, k_low.shape[2])
        block_mask = mask.view(blocks, block_size)
        counts = block_mask.sum(dim=-1).clamp_min(1.0)
        centroids = (block_keys * block_mask[None, :, :, None]).sum(dim=2) / counts[None, :, None]
        return torch.einsum("hqr,hbr->hqb", q_low.float(), centroids.float()) / math.sqrt(rank_dim)
    raise ValueError(f"Unknown block score mode: {mode}")


@torch.no_grad()
def token_sva_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
    scaling: float,
    shortlist: int,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = key.shape[1]
    actual_shortlist = min(shortlist, context)
    actual_budget = min(budget, actual_shortlist)
    coarse_scores = product_quantized_scores(q_low, codebooks, codes, rank_dim)
    coarse_idx = coarse_scores.topk(actual_shortlist, dim=-1).indices
    heads, queries, _ = coarse_idx.shape
    selected_low = k_low[:, None, :, :].expand(heads, queries, context, rank_dim).gather(
        dim=2,
        index=coarse_idx[..., None].expand(heads, queries, actual_shortlist, rank_dim),
    )
    rank_scores = (selected_low.float() * q_low[:, :, None, :].float()).sum(dim=-1) / math.sqrt(rank_dim)
    rank_keep = rank_scores.topk(actual_budget, dim=-1).indices
    final_idx = coarse_idx.gather(dim=-1, index=rank_keep)
    output = torch.empty_like(query)
    for head_idx in range(heads):
        for query_idx in range(queries):
            idx = final_idx[head_idx, query_idx]
            scores = (key[head_idx, idx].float() * query[head_idx, query_idx].float()).sum(dim=-1) * scaling
            weights = torch.softmax(scores, dim=-1)
            output[head_idx, query_idx] = torch.einsum("k,kd->d", weights, value[head_idx, idx].float())
    return output, final_idx


@torch.no_grad()
def block_statement_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_size: int,
    scaling: float,
) -> torch.Tensor:
    heads, queries, head_dim = query.shape
    output = torch.empty_like(query)
    context = key.shape[1]
    for head_idx in range(heads):
        for query_idx in range(queries):
            running_max = torch.tensor(-float("inf"), device=query.device)
            running_sum = torch.zeros((), device=query.device)
            running_out = torch.zeros(head_dim, device=query.device)
            for block_id in selected_blocks[head_idx, query_idx]:
                start = int(block_id.item()) * block_size
                end = min(start + block_size, context)
                if start >= context:
                    continue
                scores = (key[head_idx, start:end].float() * query[head_idx, query_idx].float()).sum(dim=-1) * scaling
                local_max = scores.max()
                new_max = torch.maximum(running_max, local_max)
                old_scale = torch.exp(running_max - new_max)
                weights = torch.exp(scores - new_max)
                running_out = running_out * old_scale + torch.einsum("k,kd->d", weights, value[head_idx, start:end].float())
                running_sum = running_sum * old_scale + weights.sum()
                running_max = new_max
            output[head_idx, query_idx] = running_out / running_sum.clamp_min(1e-20)
    return output


def recall_at_topk(candidate_idx: torch.Tensor, teacher_idx: torch.Tensor) -> float:
    hits = 0
    total = teacher_idx.numel()
    for head_idx in range(teacher_idx.shape[0]):
        for query_idx in range(teacher_idx.shape[1]):
            candidate = set(int(item) for item in candidate_idx[head_idx, query_idx].detach().cpu().tolist())
            for target in teacher_idx[head_idx, query_idx].detach().cpu().tolist():
                hits += int(int(target) in candidate)
    return hits / max(total, 1)


def block_recall(selected_blocks: torch.Tensor, teacher_idx: torch.Tensor, block_size: int) -> float:
    hits = 0
    total = teacher_idx.numel()
    block_ids = teacher_idx // block_size
    for head_idx in range(teacher_idx.shape[0]):
        for query_idx in range(teacher_idx.shape[1]):
            candidate = set(int(item) for item in selected_blocks[head_idx, query_idx].detach().cpu().tolist())
            for target in block_ids[head_idx, query_idx].detach().cpu().tolist():
                hits += int(int(target) in candidate)
    return hits / max(total, 1)


def output_metrics(candidate: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
    cand = candidate.float().reshape(-1, candidate.shape[-1])
    truth = teacher.float().reshape(-1, teacher.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(cand, truth, dim=-1)
    mse = (cand - truth).square().mean(dim=-1)
    rel = (cand - truth).norm(dim=-1) / truth.norm(dim=-1).clamp_min(1e-8)
    return {
        "output_cosine": float(cosine.mean().item()),
        "output_mse": float(mse.mean().item()),
        "relative_error": float(rel.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-first SVA elevator benchmark.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--base-context", type=int, default=8192)
    parser.add_argument("--target-contexts", default="8192,32768,131072")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--eval-doc-index", type=int, default=0)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--query-samples", type=int, default=8)
    parser.add_argument("--min-query-pos", type=int, default=512)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--token-shortlist", type=int, default=8192)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--block-sizes", default="64,128")
    parser.add_argument("--block-budgets", default="32,64")
    parser.add_argument("--score-modes", default="coarse_max,centroid")
    parser.add_argument("--synthetic-noise-std", type=float, default=0.01)
    parser.add_argument("--teacher-chunk-size", type=int, default=65536)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype_map = {
        "auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    bundle = load_sva_artifact_bundle(args.artifact_dir, map_location=device)
    target_contexts = comma_ints(args.target_contexts)
    block_sizes = comma_ints(args.block_sizes)
    block_budgets = comma_ints(args.block_budgets)
    score_modes = comma_strings(args.score_modes)
    layers = parse_layer_list(args.layers, len(model.model.layers))
    layers = layers if layers is not None else list(range(len(model.model.layers)))

    doc = EVAL_DOCS[args.eval_doc_index]
    batch = encode_batch(tokenizer, [repeated_document(doc, args.eval_repeats)], args.base_context, device)
    seq_len = int(batch["input_ids"].shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    query_positions = sample_query_positions(seq_len, args.query_samples, args.min_query_pos).to(device)

    print("block_elevator_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"base_context,{seq_len}", flush=True)
    print(f"target_contexts,{args.target_contexts}", flush=True)
    print(f"layers,{args.layers}", flush=True)
    print(f"artifact_profile,{bundle.manifest.get('profile_name')}", flush=True)

    with torch.no_grad():
        output = model(**batch, use_cache=False, output_hidden_states=True)
    if output.hidden_states is None:
        raise ValueError("Expected hidden states from model forward.")

    token_aggregate: dict[int, dict[str, float]] = {}
    aggregate: dict[tuple[str, int, int, int], dict[str, float]] = {}
    for layer_idx in layers:
        query_all, key_base, value_base, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
        query = query_all[:, query_positions, :]
        for context in target_contexts:
            key_bank, value_bank = make_synthetic_kv_bank(
                key_base,
                value_base,
                context,
                args.synthetic_noise_std,
                args.seed + layer_idx * 1000,
            )
            sync_if_needed(device)
            start = time.perf_counter()
            teacher_out, teacher_topk = exact_attention_streaming(
                query,
                key_bank,
                value_bank,
                scaling,
                args.teacher_chunk_size,
            )
            sync_if_needed(device)
            teacher_ms = (time.perf_counter() - start) * 1000
            q_low, k_low, codebooks, codes = project_catalog(query, key_bank, bundle, layer_idx, args.assign_chunk_size)

            sync_if_needed(device)
            start = time.perf_counter()
            token_out, token_idx = token_sva_output(
                query,
                key_bank,
                value_bank,
                q_low,
                k_low,
                codebooks,
                codes,
                bundle.rank_dim,
                scaling,
                args.token_shortlist,
                args.token_budget,
            )
            sync_if_needed(device)
            token_ms = (time.perf_counter() - start) * 1000
            token_row = {
                "variant": "token_sva",
                "layer": layer_idx,
                "context": context,
                "score_mode": "token",
                "block_size": 0,
                "block_budget": 0,
                "tokens_read": min(args.token_budget, context),
                "read_reduction": context / max(min(args.token_budget, context), 1),
                "top16_recall": recall_at_topk(token_idx, teacher_topk),
                "teacher_ms": teacher_ms,
                "method_ms": token_ms,
                "segments_per_query": min(args.token_budget, context),
                **output_metrics(token_out, teacher_out),
            }
            emit("block_elevator_row", token_row)
            token_bucket = token_aggregate.setdefault(
                context,
                {
                    "count": 0.0,
                    "top16_recall": 0.0,
                    "output_cosine": 0.0,
                    "relative_error": 0.0,
                    "method_ms": 0.0,
                    "tokens_read": float(token_row["tokens_read"]),
                    "read_reduction": float(token_row["read_reduction"]),
                },
            )
            token_bucket["count"] += 1.0
            token_bucket["top16_recall"] += float(token_row["top16_recall"])
            token_bucket["output_cosine"] += float(token_row["output_cosine"])
            token_bucket["relative_error"] += float(token_row["relative_error"])
            token_bucket["method_ms"] += float(token_row["method_ms"])

            for score_mode in score_modes:
                scores = block_scores(q_low, k_low, codebooks, codes, bundle.rank_dim, max(block_sizes), score_mode)
                # Recompute per block size when needed, because block boundaries change.
                for block_size in block_sizes:
                    if block_size == max(block_sizes):
                        block_score = scores
                    else:
                        block_score = block_scores(q_low, k_low, codebooks, codes, bundle.rank_dim, block_size, score_mode)
                    blocks = block_score.shape[-1]
                    for block_budget in block_budgets:
                        actual_blocks = min(block_budget, blocks)
                        selected_blocks = block_score.topk(actual_blocks, dim=-1).indices
                        sync_if_needed(device)
                        start = time.perf_counter()
                        block_out = block_statement_output(query, key_bank, value_bank, selected_blocks, block_size, scaling)
                        sync_if_needed(device)
                        block_ms = (time.perf_counter() - start) * 1000
                        tokens_read = min(context, actual_blocks * block_size)
                        row = {
                            "variant": "block_sva",
                            "layer": layer_idx,
                            "context": context,
                            "score_mode": score_mode,
                            "block_size": block_size,
                            "block_budget": actual_blocks,
                            "tokens_read": tokens_read,
                            "read_reduction": context / max(tokens_read, 1),
                            "top16_recall": block_recall(selected_blocks, teacher_topk, block_size),
                            "teacher_ms": teacher_ms,
                            "method_ms": block_ms,
                            "segments_per_query": actual_blocks,
                            **output_metrics(block_out, teacher_out),
                        }
                        emit("block_elevator_row", row)
                        key = (score_mode, context, block_size, actual_blocks)
                        bucket = aggregate.setdefault(
                            key,
                            {
                                "count": 0.0,
                                "top16_recall": 0.0,
                                "output_cosine": 0.0,
                                "relative_error": 0.0,
                                "method_ms": 0.0,
                                "tokens_read": float(tokens_read),
                                "read_reduction": context / max(tokens_read, 1),
                            },
                        )
                        bucket["count"] += 1.0
                        bucket["top16_recall"] += row["top16_recall"]
                        bucket["output_cosine"] += row["output_cosine"]
                        bucket["relative_error"] += row["relative_error"]
                        bucket["method_ms"] += row["method_ms"]

            del key_bank, value_bank, q_low, k_low, codebooks, codes, teacher_out, teacher_topk, token_out, token_idx
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del query_all, key_base, value_base, query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for context, bucket in sorted(token_aggregate.items()):
        count = max(bucket["count"], 1.0)
        emit(
            "block_elevator_token_summary",
            {
                "context": context,
                "tokens_read": int(bucket["tokens_read"]),
                "read_reduction": bucket["read_reduction"],
                "top16_recall": bucket["top16_recall"] / count,
                "output_cosine": bucket["output_cosine"] / count,
                "relative_error": bucket["relative_error"] / count,
                "method_ms": bucket["method_ms"] / count,
                "layers": int(count),
            },
        )

    for (score_mode, context, block_size, block_budget), bucket in sorted(aggregate.items()):
        count = max(bucket["count"], 1.0)
        emit(
            "block_elevator_summary",
            {
                "score_mode": score_mode,
                "context": context,
                "block_size": block_size,
                "block_budget": block_budget,
                "tokens_read": int(bucket["tokens_read"]),
                "read_reduction": bucket["read_reduction"],
                "top16_recall": bucket["top16_recall"] / count,
                "output_cosine": bucket["output_cosine"] / count,
                "relative_error": bucket["relative_error"] / count,
                "method_ms": bucket["method_ms"] / count,
                "layers": int(count),
            },
        )

    print("block_elevator_done", flush=True)


if __name__ == "__main__":
    main()
