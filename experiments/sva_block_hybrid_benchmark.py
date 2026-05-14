"""Token/block hybrid SVA benchmark.

This tests whether a cheap confidence rule can route each head/query to either
token SVA or block-elevator SVA. The target is a production shape where exact
queries keep scattered token precision, while diffuse queries use contiguous
block statements.
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
from sva_block_elevator_benchmark import (
    block_recall,
    block_scores,
    block_statement_output,
    comma_ints,
    exact_attention_streaming,
    layer_qkv_from_hidden,
    make_synthetic_kv_bank,
    output_metrics,
    project_catalog,
    recall_at_topk,
    sample_query_positions,
    sync_if_needed,
    token_sva_output,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import load_sva_artifact_bundle
from sva.ops import product_quantized_scores
from sva_full_deployment_benchmark import EVAL_DOCS, repeated_document
from sva_pretrained_socket_test import encode_batch, parse_layer_list


def comma_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def emit(prefix: str, row: dict[str, float | int | str]) -> None:
    print(prefix + "," + ",".join(f"{key}={row_value(value)}" for key, value in row.items()), flush=True)


def per_item_relative_error(candidate: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (candidate.float() - teacher.float()).norm(dim=-1) / teacher.float().norm(dim=-1).clamp_min(1e-8)


def recall_matrix_token(candidate_idx: torch.Tensor, teacher_idx: torch.Tensor) -> torch.Tensor:
    values = torch.empty(teacher_idx.shape[:2], device=teacher_idx.device, dtype=torch.float32)
    topk = max(int(teacher_idx.shape[-1]), 1)
    for head_idx in range(teacher_idx.shape[0]):
        for query_idx in range(teacher_idx.shape[1]):
            candidate = set(int(item) for item in candidate_idx[head_idx, query_idx].detach().cpu().tolist())
            hits = sum(int(int(target) in candidate) for target in teacher_idx[head_idx, query_idx].detach().cpu().tolist())
            values[head_idx, query_idx] = hits / topk
    return values


def recall_matrix_block(selected_blocks: torch.Tensor, teacher_idx: torch.Tensor, block_size: int) -> torch.Tensor:
    values = torch.empty(teacher_idx.shape[:2], device=teacher_idx.device, dtype=torch.float32)
    topk = max(int(teacher_idx.shape[-1]), 1)
    block_ids = teacher_idx // block_size
    for head_idx in range(teacher_idx.shape[0]):
        for query_idx in range(teacher_idx.shape[1]):
            candidate = set(int(item) for item in selected_blocks[head_idx, query_idx].detach().cpu().tolist())
            hits = sum(int(int(target) in candidate) for target in block_ids[head_idx, query_idx].detach().cpu().tolist())
            values[head_idx, query_idx] = hits / topk
    return values


@torch.no_grad()
def coarse_entropy_and_margin(
    q_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
    shortlist: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = product_quantized_scores(q_low, codebooks, codes, rank_dim)
    k = min(shortlist, scores.shape[-1])
    top_scores = scores.topk(k, dim=-1).values.float()
    if k == 1:
        margin = torch.full(top_scores.shape[:2], float("inf"), device=top_scores.device)
        entropy = torch.zeros(top_scores.shape[:2], device=top_scores.device)
        return entropy, margin
    margin = top_scores[..., 0] - top_scores[..., 1]
    probs = torch.softmax(top_scores - top_scores[..., :1], dim=-1)
    entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1) / math.log(k)
    return entropy, margin


def hybrid_output(token_out: torch.Tensor, block_out: torch.Tensor, choose_token: torch.Tensor) -> torch.Tensor:
    return torch.where(choose_token[..., None], token_out, block_out)


def summarize_hybrid(
    *,
    selector: str,
    context: int,
    layer_idx: int,
    block_size: int,
    block_budget: int,
    token_budget: int,
    choose_token: torch.Tensor,
    token_out: torch.Tensor,
    block_out: torch.Tensor,
    teacher_out: torch.Tensor,
    token_recall: torch.Tensor,
    block_recall_values: torch.Tensor,
) -> dict[str, float | int | str]:
    mixed = hybrid_output(token_out, block_out, choose_token)
    token_fraction = float(choose_token.float().mean().item())
    block_tokens = min(context, block_size * block_budget)
    avg_tokens_read = token_fraction * min(token_budget, context) + (1.0 - token_fraction) * block_tokens
    avg_segments = token_fraction * min(token_budget, context) + (1.0 - token_fraction) * block_budget
    hybrid_recall = torch.where(choose_token, token_recall, block_recall_values).mean().item()
    metrics = output_metrics(mixed, teacher_out)
    return {
        "selector": selector,
        "layer": layer_idx,
        "context": context,
        "block_size": block_size,
        "block_budget": block_budget,
        "token_fraction": token_fraction,
        "avg_tokens_read": avg_tokens_read,
        "read_reduction": context / max(avg_tokens_read, 1e-9),
        "avg_segments": avg_segments,
        "segment_reduction": min(token_budget, context) / max(avg_segments, 1e-9),
        "top16_recall": float(hybrid_recall),
        **metrics,
    }


def add_summary(
    aggregate: dict[tuple[str, int, int, int], dict[str, float]],
    row: dict[str, float | int | str],
) -> None:
    key = (str(row["selector"]), int(row["context"]), int(row["block_size"]), int(row["block_budget"]))
    bucket = aggregate.setdefault(
        key,
        {
            "count": 0.0,
            "token_fraction": 0.0,
            "avg_tokens_read": 0.0,
            "read_reduction": 0.0,
            "avg_segments": 0.0,
            "segment_reduction": 0.0,
            "top16_recall": 0.0,
            "output_cosine": 0.0,
            "relative_error": 0.0,
        },
    )
    bucket["count"] += 1.0
    for metric in bucket:
        if metric != "count":
            bucket[metric] += float(row[metric])


def main() -> None:
    parser = argparse.ArgumentParser(description="Token/block hybrid SVA benchmark.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--base-context", type=int, default=8192)
    parser.add_argument("--target-contexts", default="8192,32768,131072")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--eval-doc-index", type=int, default=0)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--query-samples", type=int, default=8)
    parser.add_argument("--min-query-pos", type=int, default=512)
    parser.add_argument("--token-shortlist", type=int, default=8192)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--block-sizes", default="64,128")
    parser.add_argument("--block-budgets", default="16,32")
    parser.add_argument("--entropy-thresholds", default="0.55,0.70,0.85,0.95")
    parser.add_argument("--synthetic-noise-std", type=float, default=0.01)
    parser.add_argument("--teacher-chunk-size", type=int, default=65536)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=23)
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
    entropy_thresholds = comma_floats(args.entropy_thresholds)
    layers = parse_layer_list(args.layers, len(model.model.layers))
    layers = layers if layers is not None else list(range(len(model.model.layers)))

    doc = EVAL_DOCS[args.eval_doc_index]
    batch = encode_batch(tokenizer, [repeated_document(doc, args.eval_repeats)], args.base_context, device)
    seq_len = int(batch["input_ids"].shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    query_positions = sample_query_positions(seq_len, args.query_samples, args.min_query_pos).to(device)

    print("block_hybrid_start", flush=True)
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
            entropy, margin = coarse_entropy_and_margin(q_low, codebooks, codes, bundle.rank_dim, args.token_shortlist)

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
            token_error = per_item_relative_error(token_out, teacher_out)
            token_recall_values = recall_matrix_token(token_idx, teacher_topk)
            token_row = {
                "variant": "token",
                "layer": layer_idx,
                "context": context,
                "tokens_read": min(args.token_budget, context),
                "read_reduction": context / max(min(args.token_budget, context), 1),
                "segments": min(args.token_budget, context),
                "top16_recall": recall_at_topk(token_idx, teacher_topk),
                "teacher_ms": teacher_ms,
                **output_metrics(token_out, teacher_out),
            }
            emit("block_hybrid_baseline", token_row)

            for block_size in block_sizes:
                block_score = block_scores(q_low, k_low, codebooks, codes, bundle.rank_dim, block_size, "centroid")
                blocks = block_score.shape[-1]
                for block_budget in block_budgets:
                    actual_blocks = min(block_budget, blocks)
                    selected_blocks = block_score.topk(actual_blocks, dim=-1).indices
                    block_out = block_statement_output(query, key_bank, value_bank, selected_blocks, block_size, scaling)
                    block_error = per_item_relative_error(block_out, teacher_out)
                    block_recall_values = recall_matrix_block(selected_blocks, teacher_topk, block_size)
                    block_row = {
                        "variant": "block_centroid",
                        "layer": layer_idx,
                        "context": context,
                        "block_size": block_size,
                        "block_budget": actual_blocks,
                        "tokens_read": min(context, block_size * actual_blocks),
                        "read_reduction": context / max(min(context, block_size * actual_blocks), 1),
                        "segments": actual_blocks,
                        "top16_recall": block_recall(selected_blocks, teacher_topk, block_size),
                        **output_metrics(block_out, teacher_out),
                    }
                    emit("block_hybrid_baseline", block_row)

                    oracle_choose_token = token_error <= block_error
                    oracle_row = summarize_hybrid(
                        selector="oracle_error",
                        context=context,
                        layer_idx=layer_idx,
                        block_size=block_size,
                        block_budget=actual_blocks,
                        token_budget=args.token_budget,
                        choose_token=oracle_choose_token,
                        token_out=token_out,
                        block_out=block_out,
                        teacher_out=teacher_out,
                        token_recall=token_recall_values,
                        block_recall_values=block_recall_values,
                    )
                    emit("block_hybrid_row", oracle_row)
                    add_summary(aggregate, oracle_row)

                    for threshold in entropy_thresholds:
                        choose_token = entropy <= threshold
                        row = summarize_hybrid(
                            selector=f"entropy_le_{threshold:.2f}",
                            context=context,
                            layer_idx=layer_idx,
                            block_size=block_size,
                            block_budget=actual_blocks,
                            token_budget=args.token_budget,
                            choose_token=choose_token,
                            token_out=token_out,
                            block_out=block_out,
                            teacher_out=teacher_out,
                            token_recall=token_recall_values,
                            block_recall_values=block_recall_values,
                        )
                        emit("block_hybrid_row", row)
                        add_summary(aggregate, row)

                    margin_threshold = torch.quantile(margin.float().reshape(-1), 0.75)
                    choose_token = margin >= margin_threshold
                    row = summarize_hybrid(
                        selector="margin_top_quartile",
                        context=context,
                        layer_idx=layer_idx,
                        block_size=block_size,
                        block_budget=actual_blocks,
                        token_budget=args.token_budget,
                        choose_token=choose_token,
                        token_out=token_out,
                        block_out=block_out,
                        teacher_out=teacher_out,
                        token_recall=token_recall_values,
                        block_recall_values=block_recall_values,
                    )
                    emit("block_hybrid_row", row)
                    add_summary(aggregate, row)

                    del block_out, block_error, block_recall_values, selected_blocks
            del key_bank, value_bank, q_low, k_low, codebooks, codes, teacher_out, teacher_topk
            del token_out, token_idx, token_error, token_recall_values, entropy, margin
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del query_all, key_base, value_base, query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (selector, context, block_size, block_budget), bucket in sorted(aggregate.items()):
        count = max(bucket["count"], 1.0)
        emit(
            "block_hybrid_summary",
            {
                "selector": selector,
                "context": context,
                "block_size": block_size,
                "block_budget": block_budget,
                "token_fraction": bucket["token_fraction"] / count,
                "avg_tokens_read": bucket["avg_tokens_read"] / count,
                "read_reduction": bucket["read_reduction"] / count,
                "avg_segments": bucket["avg_segments"] / count,
                "segment_reduction": bucket["segment_reduction"] / count,
                "top16_recall": bucket["top16_recall"] / count,
                "output_cosine": bucket["output_cosine"] / count,
                "relative_error": bucket["relative_error"] / count,
                "layers": int(count),
            },
        )

    print("block_hybrid_done", flush=True)


if __name__ == "__main__":
    main()
