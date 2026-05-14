"""Evidence-in-haystack summoner benchmark for SVA.

This benchmark measures whether SVA summons the specific evidence span needed
by a passkey query as the context grows. It isolates the summoner: the verifier
can only use evidence that the summon stage actually brings forward.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sva_block_elevator_benchmark import comma_ints, layer_qkv_from_hidden, project_catalog
from sva_passkey_language_benchmark import FILLER, comma_strings, repeated_to_length

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import load_sva_artifact_bundle
from sva.ops import product_quantized_scores
from sva_pretrained_socket_test import parse_layer_list


@dataclass(frozen=True)
class EvidenceCase:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    context: int
    placement: str
    key: str
    needle_start: int
    needle_end: int
    key_start: int
    key_end: int


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


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def build_evidence_case(
    tokenizer: Any,
    context: int,
    key: str,
    placement: str,
    device: torch.device,
) -> EvidenceCase:
    bos = [int(tokenizer.bos_token_id)] if tokenizer.bos_token_id is not None else []
    needle_prefix = encode_text(tokenizer, "\nMemorandum: the private passkey is")
    key_ids = encode_text(tokenizer, f" {key}")
    needle_suffix = encode_text(tokenizer, ".\n")
    needle = needle_prefix + key_ids + needle_suffix
    query = encode_text(tokenizer, "\nQuestion: What is the private passkey?\nAnswer:")
    filler = encode_text(tokenizer, FILLER)
    fill_len = context - len(bos) - len(needle) - len(query)
    if fill_len < 0:
        raise ValueError(f"Context {context} is too short for the passkey prompt.")

    if placement == "start":
        before = bos
        after = repeated_to_length(filler, fill_len) + query
    elif placement == "middle":
        left = fill_len // 2
        right = fill_len - left
        before = bos + repeated_to_length(filler, left)
        after = repeated_to_length(filler, right) + query
    elif placement == "end":
        before = bos + repeated_to_length(filler, fill_len)
        after = query
    else:
        raise ValueError(f"Unknown placement: {placement}")

    needle_start = len(before)
    key_start = needle_start + len(needle_prefix)
    key_end = key_start + len(key_ids)
    prefix = before + needle + after
    if len(prefix) != context:
        raise AssertionError(f"Prompt length mismatch: expected {context}, got {len(prefix)}.")

    input_ids = torch.tensor([prefix], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return EvidenceCase(
        input_ids=input_ids,
        attention_mask=attention_mask,
        context=context,
        placement=placement,
        key=key,
        needle_start=needle_start,
        needle_end=needle_start + len(needle),
        key_start=key_start,
        key_end=key_end,
    )


def exact_teacher_stats(
    query: torch.Tensor,
    key: torch.Tensor,
    scaling: float,
    key_positions: torch.Tensor,
    topk: int,
) -> dict[str, float]:
    scores = torch.einsum("hd,hkd->hk", query.float(), key.float()) * scaling
    weights = torch.softmax(scores, dim=-1)
    key_mask = torch.zeros(scores.shape[-1], device=scores.device, dtype=torch.bool)
    key_mask[key_positions] = True
    key_mass = weights[:, key_positions].sum(dim=-1)
    top_count = min(topk, scores.shape[-1])
    top_idx = scores.topk(top_count, dim=-1).indices
    top_hit = (top_idx[..., None] == key_positions[None, None, :]).any(dim=-1).float()
    ranks = []
    for head_idx in range(scores.shape[0]):
        order = scores[head_idx].argsort(descending=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        ranks.append(float(inverse[key_positions].min().item() + 1))
    rank_tensor = torch.tensor(ranks, device=scores.device, dtype=torch.float32)
    return {
        "teacher_key_mass": float(key_mass.mean().item()),
        "teacher_topk_key_hit": float(top_hit.mean().item()),
        f"teacher_top{topk}_key_hit": float(top_hit.mean().item()),
        "teacher_best_key_rank": float(rank_tensor.mean().item()),
    }


def union_candidates_for_head(
    *,
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
    head_idx: int,
    anchor_positions: list[int],
    shortlist: int,
    per_anchor_budget: int,
) -> torch.Tensor:
    gathered: list[torch.Tensor] = []
    for anchor_pos in anchor_positions:
        q_item = q_low[head_idx, anchor_pos : anchor_pos + 1]
        scores = product_quantized_scores(q_item[None, :, :], codebooks[head_idx : head_idx + 1], codes[head_idx : head_idx + 1], rank_dim)[0, 0]
        if anchor_pos + 1 < scores.numel():
            scores[anchor_pos + 1 :] = torch.finfo(scores.dtype).min
        actual_shortlist = min(shortlist, anchor_pos + 1)
        actual_budget = min(per_anchor_budget, actual_shortlist)
        coarse_idx = scores.topk(actual_shortlist, dim=-1).indices
        rank_scores = (k_low[head_idx, coarse_idx].float() * q_item[0].float()[None, :]).sum(dim=-1) / math.sqrt(rank_dim)
        keep = rank_scores.topk(actual_budget, dim=-1).indices
        gathered.append(coarse_idx[keep])
    if not gathered:
        return torch.empty(0, device=k_low.device, dtype=torch.long)
    return torch.unique(torch.cat(gathered))


def span_hit(indices: torch.Tensor, start: int, end: int) -> float:
    if indices.numel() == 0:
        return 0.0
    return float(((indices >= start) & (indices < end)).any().item())


def expand_candidates(indices: torch.Tensor, radius: int, context: int) -> torch.Tensor:
    if indices.numel() == 0 or radius <= 0:
        return indices
    offsets = torch.arange(-radius, radius + 1, device=indices.device, dtype=torch.long)
    expanded = indices[:, None] + offsets[None, :]
    expanded = expanded[(expanded >= 0) & (expanded < context)]
    return torch.unique(expanded)


def evaluate_anchor_policy(
    *,
    query_all: torch.Tensor,
    key: torch.Tensor,
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
    scaling: float,
    anchor_positions: list[int],
    shortlist: int,
    per_anchor_budget: int,
    final_budget: int,
    rerank_mode: str,
    expand_radius: int,
    key_start: int,
    key_end: int,
    needle_start: int,
    needle_end: int,
) -> dict[str, float]:
    head_count = key.shape[0]
    key_hits = []
    needle_hits = []
    verified_key_hits = []
    verified_needle_hits = []
    candidate_counts = []
    expanded_counts = []
    verified_counts = []
    rerank_score_counts = []
    candidate_key_hits = []
    candidate_needle_hits = []
    context = key.shape[1]
    for head_idx in range(head_count):
        candidates = union_candidates_for_head(
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
        candidate_counts.append(float(candidates.numel()))
        key_hits.append(span_hit(candidates, key_start, key_end))
        needle_hits.append(span_hit(candidates, needle_start, needle_end))
        expanded = expand_candidates(candidates, expand_radius, context)
        expanded_counts.append(float(expanded.numel()))
        candidate_key_hits.append(span_hit(expanded, key_start, key_end))
        candidate_needle_hits.append(span_hit(expanded, needle_start, needle_end))
        if expanded.numel() == 0:
            verified = expanded
            rerank_score_counts.append(0.0)
        else:
            if rerank_mode == "current":
                scores = (key[head_idx, expanded].float() * query_all[head_idx, -1].float()[None, :]).sum(dim=-1) * scaling
                rerank_score_counts.append(float(expanded.numel()))
            elif rerank_mode == "max_anchor":
                anchor_query = query_all[head_idx, anchor_positions].float()
                scores = torch.einsum("ad,kd->ak", anchor_query, key[head_idx, expanded].float()) * scaling
                scores = scores.max(dim=0).values
                rerank_score_counts.append(float(expanded.numel() * len(anchor_positions)))
            else:
                raise ValueError(f"Unknown rerank mode: {rerank_mode}")
            keep = scores.topk(min(final_budget, scores.numel()), dim=-1).indices
            verified = expanded[keep]
        verified_counts.append(float(verified.numel()))
        verified_key_hits.append(span_hit(verified, key_start, key_end))
        verified_needle_hits.append(span_hit(verified, needle_start, needle_end))

    return {
        "avg_candidates": float(torch.tensor(candidate_counts).mean().item()),
        "avg_expanded_candidates": float(torch.tensor(expanded_counts).mean().item()),
        "avg_verified": float(torch.tensor(verified_counts).mean().item()),
        "avg_rerank_scores": float(torch.tensor(rerank_score_counts).mean().item()),
        "summoned_key_hit": float(torch.tensor(key_hits).mean().item()),
        "summoned_needle_hit": float(torch.tensor(needle_hits).mean().item()),
        "candidate_key_hit": float(torch.tensor(candidate_key_hits).mean().item()),
        "candidate_needle_hit": float(torch.tensor(candidate_needle_hits).mean().item()),
        "verified_key_hit": float(torch.tensor(verified_key_hits).mean().item()),
        "verified_needle_hit": float(torch.tensor(verified_needle_hits).mean().item()),
    }


def add_summary(
    aggregate: dict[tuple[str, int, str, int, str, int], dict[str, float]],
    row: dict[str, float | int | str],
) -> None:
    key = (
        str(row["policy"]),
        int(row["context"]),
        str(row["placement"]),
        int(row["anchor_count"]),
        str(row["rerank_mode"]),
        int(row["expand_radius"]),
    )
    bucket = aggregate.setdefault(
        key,
        {
            "count": 0.0,
            "avg_candidates": 0.0,
            "avg_expanded_candidates": 0.0,
            "avg_verified": 0.0,
            "avg_rerank_scores": 0.0,
            "summoned_key_hit": 0.0,
            "summoned_needle_hit": 0.0,
            "candidate_key_hit": 0.0,
            "candidate_needle_hit": 0.0,
            "verified_key_hit": 0.0,
            "verified_needle_hit": 0.0,
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
    parser = argparse.ArgumentParser(description="SVA evidence-in-haystack summoner benchmark.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--placements", default="start,middle")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--key", default="731942")
    parser.add_argument("--shortlist", type=int, default=8192)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--anchor-counts", default="1,4,8,16")
    parser.add_argument("--rerank-modes", default="current")
    parser.add_argument("--expand-radii", default="0")
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
    rerank_modes = comma_strings(args.rerank_modes)
    expand_radii = comma_ints(args.expand_radii)
    layers = parse_layer_list(args.layers, len(model.model.layers))
    layers = layers if layers is not None else list(range(len(model.model.layers)))

    print("evidence_haystack_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"placements,{args.placements}", flush=True)
    print(f"layers,{args.layers}", flush=True)
    print(f"rerank_modes,{args.rerank_modes}", flush=True)
    print(f"expand_radii,{args.expand_radii}", flush=True)
    print(f"artifact_profile,{bundle.manifest.get('profile_name')}", flush=True)

    aggregate: dict[tuple[str, int, str, int, str, int], dict[str, float]] = {}
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
                query_all, key_all, _, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
                current_pos = context - 1
                query_current = query_all[:, current_pos, :]
                q_low, k_low, codebooks, codes = project_catalog(query_all, key_all, bundle, layer_idx, args.assign_chunk_size)
                teacher = exact_teacher_stats(query_current, key_all, scaling, key_positions, args.topk)
                for anchor_count in anchor_counts:
                    actual_anchor_count = min(anchor_count, context)
                    anchor_positions = list(range(context - actual_anchor_count, context))
                    split_budget = max(1, args.budget // actual_anchor_count)
                    policies = {
                        "split": split_budget,
                        "full": args.budget,
                    }
                    for policy, per_anchor_budget in policies.items():
                        for rerank_mode in rerank_modes:
                            for expand_radius in expand_radii:
                                metrics = evaluate_anchor_policy(
                                    query_all=query_all,
                                    key=key_all,
                                    q_low=q_low,
                                    k_low=k_low,
                                    codebooks=codebooks,
                                    codes=codes,
                                    rank_dim=bundle.rank_dim,
                                    scaling=scaling,
                                    anchor_positions=anchor_positions,
                                    shortlist=args.shortlist,
                                    per_anchor_budget=per_anchor_budget,
                                    final_budget=args.budget,
                                    rerank_mode=rerank_mode,
                                    expand_radius=expand_radius,
                                    key_start=case.key_start,
                                    key_end=case.key_end,
                                    needle_start=case.needle_start,
                                    needle_end=case.needle_end,
                                )
                                row = {
                                    "policy": policy,
                                    "context": context,
                                    "placement": placement,
                                    "layer": layer_idx,
                                    "anchor_count": actual_anchor_count,
                                    "per_anchor_budget": per_anchor_budget,
                                    "rerank_mode": rerank_mode,
                                    "expand_radius": expand_radius,
                                    "key_start": case.key_start,
                                    "key_end": case.key_end,
                                    **teacher,
                                    **metrics,
                                }
                                emit("evidence_haystack_row", row)
                                add_summary(aggregate, row)
                del query_all, key_all, q_low, k_low, codebooks, codes
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for (policy, context, placement, anchor_count, rerank_mode, expand_radius), bucket in sorted(aggregate.items()):
        count = max(bucket["count"], 1.0)
        avg_candidates = bucket["avg_candidates"] / count
        avg_expanded_candidates = bucket["avg_expanded_candidates"] / count
        avg_verified = bucket["avg_verified"] / count
        avg_rerank_scores = bucket["avg_rerank_scores"] / count
        emit(
            "evidence_haystack_summary",
            {
                "policy": policy,
                "context": context,
                "placement": placement,
                "anchor_count": anchor_count,
                "rerank_mode": rerank_mode,
                "expand_radius": expand_radius,
                "avg_candidates": avg_candidates,
                "avg_expanded_candidates": avg_expanded_candidates,
                "avg_verified": avg_verified,
                "avg_rerank_scores": avg_rerank_scores,
                "read_reduction": context / max(avg_verified, 1e-9),
                "score_reduction": context / max(avg_rerank_scores, 1e-9),
                "summoned_key_hit": bucket["summoned_key_hit"] / count,
                "summoned_needle_hit": bucket["summoned_needle_hit"] / count,
                "candidate_key_hit": bucket["candidate_key_hit"] / count,
                "candidate_needle_hit": bucket["candidate_needle_hit"] / count,
                "verified_key_hit": bucket["verified_key_hit"] / count,
                "verified_needle_hit": bucket["verified_needle_hit"] / count,
                "teacher_key_mass": bucket["teacher_key_mass"] / count,
                "teacher_topk_key_hit": bucket["teacher_topk_key_hit"] / count,
                "teacher_best_key_rank": bucket["teacher_best_key_rank"] / count,
                "layers": int(count),
            },
        )

    print("evidence_haystack_done", flush=True)


if __name__ == "__main__":
    main()
