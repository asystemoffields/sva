"""Span-statement SVA benchmark for passkey evidence.

This benchmark tests whether sparse summoned tokens should open local spans
instead of competing as individual tokens in the final verifier. Each selected
span computes exact attention where it sits; the selected span union is compared
with full attention for the final query.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sva_block_elevator_benchmark import comma_ints, layer_qkv_from_hidden, output_metrics, project_catalog
from sva_evidence_haystack_benchmark import build_evidence_case, exact_teacher_stats, span_hit, union_candidates_for_head
from sva_passkey_language_benchmark import comma_strings

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import load_sva_artifact_bundle
from sva_pretrained_socket_test import parse_layer_list


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value.replace("\n", " ").replace(",", ";")
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def emit(prefix: str, row: dict[str, float | int | str]) -> None:
    print(prefix + "," + ",".join(f"{key}={row_value(value)}" for key, value in row.items()), flush=True)


def exact_attention_current(
    query_current: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    scores = torch.einsum("hd,hkd->hk", query_current.float(), key.float()) * scaling
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hk,hkd->hd", weights, value.float())


def score_seed_candidates(
    query_all: torch.Tensor,
    key: torch.Tensor,
    head_idx: int,
    candidates: torch.Tensor,
    anchor_positions: list[int],
    mode: str,
    scaling: float,
) -> tuple[torch.Tensor, float]:
    if candidates.numel() == 0:
        return torch.empty(0, device=key.device, dtype=torch.float32), 0.0
    if mode == "current":
        scores = (key[head_idx, candidates].float() * query_all[head_idx, -1].float()[None, :]).sum(dim=-1) * scaling
        return scores, float(candidates.numel())
    if mode == "max_anchor":
        anchor_query = query_all[head_idx, anchor_positions].float()
        scores = torch.einsum("ad,kd->ak", anchor_query, key[head_idx, candidates].float()) * scaling
        return scores.max(dim=0).values, float(candidates.numel() * len(anchor_positions))
    raise ValueError(f"Unknown seed score mode: {mode}")


def add_partial_window(mask: torch.Tensor, seed: int, token_budget: int) -> int:
    context = mask.numel()
    width = min(token_budget, context)
    start = max(0, min(seed - width // 2, context - width))
    end = start + width
    before = int(mask.sum().item())
    mask[start:end] = True
    return int(mask.sum().item()) - before


def select_span_tokens(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    radius: int,
    token_budget: int,
    context: int,
) -> torch.Tensor:
    if candidates.numel() == 0 or token_budget <= 0:
        return torch.empty(0, device=candidates.device, dtype=torch.long)
    order = scores.argsort(descending=True)
    mask = torch.zeros(context, device=candidates.device, dtype=torch.bool)
    selected = 0
    width = 2 * radius + 1
    for item in order:
        seed = int(candidates[item].item())
        start = max(0, seed - radius)
        end = min(context, seed + radius + 1)
        new_tokens = int((~mask[start:end]).sum().item())
        if new_tokens == 0:
            continue
        if selected + new_tokens > token_budget:
            if selected == 0:
                selected += add_partial_window(mask, seed, token_budget)
            continue
        mask[start:end] = True
        selected += new_tokens
        if selected >= token_budget or (radius == 0 and selected >= min(token_budget, candidates.numel())):
            break
        if width <= 1 and selected >= token_budget:
            break
    return mask.nonzero(as_tuple=False).flatten()


def segment_count(indices: torch.Tensor) -> int:
    if indices.numel() == 0:
        return 0
    ordered = indices.sort().values
    return int(1 + (ordered[1:] != ordered[:-1] + 1).sum().item())


def attention_over_tokens(
    query_head: torch.Tensor,
    key_head: torch.Tensor,
    value_head: torch.Tensor,
    indices: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.zeros_like(query_head)
    scores = (key_head[indices].float() * query_head.float()[None, :]).sum(dim=-1) * scaling
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("k,kd->d", weights, value_head[indices].float())


def evaluate_span_policy(
    *,
    query_all: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    teacher_output: torch.Tensor,
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
    scaling: float,
    anchor_positions: list[int],
    shortlist: int,
    per_anchor_budget: int,
    seed_score_mode: str,
    span_radius: int,
    span_token_budget: int,
    key_start: int,
    key_end: int,
    needle_start: int,
    needle_end: int,
) -> dict[str, float]:
    head_count = key.shape[0]
    context = key.shape[1]
    seed_counts = []
    seed_score_counts = []
    span_token_counts = []
    exact_score_counts = []
    segment_counts = []
    seed_key_hits = []
    seed_needle_hits = []
    span_key_hits = []
    span_needle_hits = []
    outputs = []

    for head_idx in range(head_count):
        seeds = union_candidates_for_head(
            q_low=q_low,
            k_low=k_low,
            codebooks=codebooks,
            codes=codes,
            rank_dim=rank_dim,
            head_idx=head_idx,
            anchor_positions=anchor_positions,
            shortlist=shortlist,
            per_anchor_budget=per_anchor_budget,
        )
        seed_scores, seed_score_count = score_seed_candidates(query_all, key, head_idx, seeds, anchor_positions, seed_score_mode, scaling)
        span_tokens = select_span_tokens(seeds, seed_scores, span_radius, span_token_budget, context)
        output = attention_over_tokens(query_all[head_idx, -1], key[head_idx], value[head_idx], span_tokens, scaling)

        seed_counts.append(float(seeds.numel()))
        seed_score_counts.append(seed_score_count)
        span_token_counts.append(float(span_tokens.numel()))
        exact_score_counts.append(seed_score_count + float(span_tokens.numel()))
        segment_counts.append(float(segment_count(span_tokens)))
        seed_key_hits.append(span_hit(seeds, key_start, key_end))
        seed_needle_hits.append(span_hit(seeds, needle_start, needle_end))
        span_key_hits.append(span_hit(span_tokens, key_start, key_end))
        span_needle_hits.append(span_hit(span_tokens, needle_start, needle_end))
        outputs.append(output)

    span_output = torch.stack(outputs, dim=0)
    metrics = output_metrics(span_output[:, None, :], teacher_output[:, None, :])
    return {
        "avg_seeds": float(torch.tensor(seed_counts).mean().item()),
        "avg_seed_scores": float(torch.tensor(seed_score_counts).mean().item()),
        "avg_span_tokens": float(torch.tensor(span_token_counts).mean().item()),
        "avg_exact_scores": float(torch.tensor(exact_score_counts).mean().item()),
        "avg_segments": float(torch.tensor(segment_counts).mean().item()),
        "seed_key_hit": float(torch.tensor(seed_key_hits).mean().item()),
        "seed_needle_hit": float(torch.tensor(seed_needle_hits).mean().item()),
        "span_key_hit": float(torch.tensor(span_key_hits).mean().item()),
        "span_needle_hit": float(torch.tensor(span_needle_hits).mean().item()),
        **metrics,
        "output_cosine_flat": float(F.cosine_similarity(span_output.reshape(1, -1), teacher_output.reshape(1, -1), dim=-1).item()),
    }


def add_summary(
    aggregate: dict[tuple[str, int, str, int, int, int], dict[str, float]],
    row: dict[str, float | int | str],
) -> None:
    key = (
        str(row["seed_score_mode"]),
        int(row["context"]),
        str(row["placement"]),
        int(row["anchor_count"]),
        int(row["span_radius"]),
        int(row["span_token_budget"]),
    )
    bucket = aggregate.setdefault(
        key,
        {
            "count": 0.0,
            "avg_seeds": 0.0,
            "avg_seed_scores": 0.0,
            "avg_span_tokens": 0.0,
            "avg_exact_scores": 0.0,
            "avg_segments": 0.0,
            "seed_key_hit": 0.0,
            "seed_needle_hit": 0.0,
            "span_key_hit": 0.0,
            "span_needle_hit": 0.0,
            "output_cosine": 0.0,
            "output_cosine_flat": 0.0,
            "output_mse": 0.0,
            "relative_error": 0.0,
            "teacher_key_mass": 0.0,
            "teacher_topk_key_hit": 0.0,
            "teacher_best_key_rank": 0.0,
        },
    )
    bucket["count"] += 1.0
    for metric in bucket:
        if metric != "count":
            bucket[metric] += float(row[metric])


def main() -> None:
    parser = argparse.ArgumentParser(description="SVA span-statement evidence benchmark.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--placements", default="start,middle,end")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--key", default="731942")
    parser.add_argument("--shortlist", type=int, default=8192)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--anchor-counts", default="4,8,16")
    parser.add_argument("--seed-score-modes", default="current,max_anchor")
    parser.add_argument("--span-radii", default="0,8,32")
    parser.add_argument("--span-token-budgets", default="1024,2048,4096")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--attn-implementation", default="eager")
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    bundle = load_sva_artifact_bundle(args.artifact_dir, map_location=device)

    contexts = comma_ints(args.contexts)
    placements = comma_strings(args.placements)
    anchor_counts = comma_ints(args.anchor_counts)
    seed_score_modes = comma_strings(args.seed_score_modes)
    span_radii = comma_ints(args.span_radii)
    span_token_budgets = comma_ints(args.span_token_budgets)
    layers = parse_layer_list(args.layers, len(model.model.layers))
    layers = layers if layers is not None else list(range(len(model.model.layers)))

    print("span_statement_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"placements,{args.placements}", flush=True)
    print(f"layers,{args.layers}", flush=True)
    print(f"seed_score_modes,{args.seed_score_modes}", flush=True)
    print(f"span_radii,{args.span_radii}", flush=True)
    print(f"span_token_budgets,{args.span_token_budgets}", flush=True)
    print(f"artifact_profile,{bundle.manifest.get('profile_name')}", flush=True)

    aggregate: dict[tuple[str, int, str, int, int, int], dict[str, float]] = {}
    for context in contexts:
        for placement in placements:
            case = build_evidence_case(tokenizer, context, args.key, placement, device)
            with torch.no_grad():
                output = model(
                    input_ids=case.input_ids,
                    attention_mask=case.attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
            if output.hidden_states is None:
                raise ValueError("Expected hidden states from model forward.")
            position_ids = torch.arange(context, device=device).unsqueeze(0)
            key_positions = torch.arange(case.key_start, case.key_end, device=device)
            for layer_idx in layers:
                query_all, key_all, value_all, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
                query_current = query_all[:, -1, :]
                teacher_output = exact_attention_current(query_current, key_all, value_all, scaling)
                teacher = exact_teacher_stats(query_current, key_all, scaling, key_positions, args.topk)
                q_low, k_low, codebooks, codes = project_catalog(query_all, key_all, bundle, layer_idx, args.assign_chunk_size)

                for anchor_count in anchor_counts:
                    actual_anchor_count = min(anchor_count, context)
                    anchor_positions = list(range(context - actual_anchor_count, context))
                    per_anchor_budget = max(1, args.budget // actual_anchor_count)
                    for seed_score_mode in seed_score_modes:
                        for span_radius in span_radii:
                            for span_token_budget in span_token_budgets:
                                metrics = evaluate_span_policy(
                                    query_all=query_all,
                                    key=key_all,
                                    value=value_all,
                                    teacher_output=teacher_output,
                                    q_low=q_low,
                                    k_low=k_low,
                                    codebooks=codebooks,
                                    codes=codes,
                                    rank_dim=bundle.rank_dim,
                                    scaling=scaling,
                                    anchor_positions=anchor_positions,
                                    shortlist=args.shortlist,
                                    per_anchor_budget=per_anchor_budget,
                                    seed_score_mode=seed_score_mode,
                                    span_radius=span_radius,
                                    span_token_budget=span_token_budget,
                                    key_start=case.key_start,
                                    key_end=case.key_end,
                                    needle_start=case.needle_start,
                                    needle_end=case.needle_end,
                                )
                                row = {
                                    "context": context,
                                    "placement": placement,
                                    "layer": layer_idx,
                                    "anchor_count": actual_anchor_count,
                                    "per_anchor_budget": per_anchor_budget,
                                    "seed_score_mode": seed_score_mode,
                                    "span_radius": span_radius,
                                    "span_token_budget": span_token_budget,
                                    "read_reduction": context / max(metrics["avg_span_tokens"], 1e-9),
                                    "score_reduction": context / max(metrics["avg_exact_scores"], 1e-9),
                                    "key_start": case.key_start,
                                    "key_end": case.key_end,
                                    **teacher,
                                    **metrics,
                                }
                                emit("span_statement_row", row)
                                add_summary(aggregate, row)
                del query_all, key_all, value_all, q_low, k_low, codebooks, codes, teacher_output
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for (seed_score_mode, context, placement, anchor_count, span_radius, span_token_budget), bucket in sorted(aggregate.items()):
        count = max(bucket["count"], 1.0)
        avg_span_tokens = bucket["avg_span_tokens"] / count
        avg_exact_scores = bucket["avg_exact_scores"] / count
        emit(
            "span_statement_summary",
            {
                "context": context,
                "placement": placement,
                "anchor_count": anchor_count,
                "seed_score_mode": seed_score_mode,
                "span_radius": span_radius,
                "span_token_budget": span_token_budget,
                "avg_seeds": bucket["avg_seeds"] / count,
                "avg_seed_scores": bucket["avg_seed_scores"] / count,
                "avg_span_tokens": avg_span_tokens,
                "avg_exact_scores": avg_exact_scores,
                "avg_segments": bucket["avg_segments"] / count,
                "read_reduction": context / max(avg_span_tokens, 1e-9),
                "score_reduction": context / max(avg_exact_scores, 1e-9),
                "seed_key_hit": bucket["seed_key_hit"] / count,
                "seed_needle_hit": bucket["seed_needle_hit"] / count,
                "span_key_hit": bucket["span_key_hit"] / count,
                "span_needle_hit": bucket["span_needle_hit"] / count,
                "output_cosine": bucket["output_cosine"] / count,
                "output_cosine_flat": bucket["output_cosine_flat"] / count,
                "output_mse": bucket["output_mse"] / count,
                "relative_error": bucket["relative_error"] / count,
                "teacher_key_mass": bucket["teacher_key_mass"] / count,
                "teacher_topk_key_hit": bucket["teacher_topk_key_hit"] / count,
                "teacher_best_key_rank": bucket["teacher_best_key_rank"] / count,
                "layers": int(count),
            },
        )

    print("span_statement_done", flush=True)


if __name__ == "__main__":
    main()
