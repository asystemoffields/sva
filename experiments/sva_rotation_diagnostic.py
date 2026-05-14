"""Rotation diagnostic for SVA product-code catalogs.

The learned SVA ranker emits a low-rank Q/K space. Exact low-rank dot products
are rotation-invariant, but product quantization is not: splitting dimensions
into fixed subspaces can make the catalog easier or harder to search. This
diagnostic tests whether a Hadamard-style rotation before PQ improves candidate
survival for the same low-rank ranker.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sva_block_elevator_benchmark import comma_ints, layer_qkv_from_hidden, project_catalog
from sva_evidence_haystack_benchmark import build_evidence_case
from sva_passkey_language_benchmark import comma_strings
from sva_pq_lookup_test import encode_product_keys, fit_product_codebooks, product_quantized_scores
from sva_pretrained_socket_test import parse_layer_list

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import load_sva_artifact_bundle


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


def normalized_hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"Hadamard rotation requires a power-of-two dimension, got {dim}.")
    matrix = torch.ones(1, 1, device=device, dtype=torch.float32)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix / math.sqrt(dim)


def rotation_matrix(label: str, dim: int, seed: int, device: torch.device) -> torch.Tensor | None:
    if label == "identity":
        return None
    if label == "hadamard":
        return normalized_hadamard_matrix(dim, device)
    if label == "signed_hadamard":
        matrix = normalized_hadamard_matrix(dim, device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), device=device, generator=generator, dtype=torch.int64).float()
        signs = signs.mul_(2.0).sub_(1.0)
        perm = torch.randperm(dim, device=device, generator=generator)
        return matrix[:, perm] * signs[None, :]
    raise ValueError(f"Unknown rotation variant: {label}")


def apply_rotation(x: torch.Tensor, rotation: torch.Tensor | None) -> torch.Tensor:
    if rotation is None:
        return x
    return torch.einsum("...r,rs->...s", x.float(), rotation).to(x.dtype)


def causal_mask(scores: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    key_positions = torch.arange(scores.shape[-1], device=scores.device)
    allowed = key_positions[None, None, :] <= query_positions[None, :, None]
    return scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)


def candidate_recall(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
    target_valid: torch.Tensor,
    query_positions: torch.Tensor,
    budget: int,
) -> tuple[float, int, int]:
    masked = causal_mask(scores, query_positions)
    actual_budget = min(budget, scores.shape[-1])
    candidate_idx = masked.topk(actual_budget, dim=-1).indices
    candidate_mask = torch.zeros_like(masked, dtype=torch.bool)
    candidate_mask.scatter_(dim=-1, index=candidate_idx, value=True)
    hits_t = candidate_mask.gather(dim=-1, index=target_idx.clamp(0, scores.shape[-1] - 1)) & target_valid
    hits = int(hits_t.sum().item())
    total = int(target_valid.sum().item())
    return hits / total if total else float("nan"), hits, total


def topk_targets(scores: torch.Tensor, query_positions: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    masked = causal_mask(scores, query_positions)
    actual_topk = min(topk, scores.shape[-1])
    idx = masked.topk(actual_topk, dim=-1).indices
    rank = torch.arange(actual_topk, device=scores.device)
    valid = rank[None, None, :] <= query_positions[None, :, None]
    return idx, valid.expand_as(idx)


def score_alignment(approx: torch.Tensor, exact: torch.Tensor, query_positions: torch.Tensor) -> tuple[float, float]:
    key_positions = torch.arange(exact.shape[-1], device=exact.device)
    allowed = key_positions[None, None, :] <= query_positions[None, :, None]
    approx_allowed = approx[allowed.expand_as(approx)].float()
    exact_allowed = exact[allowed.expand_as(exact)].float()
    approx_centered = approx_allowed - approx_allowed.mean()
    exact_centered = exact_allowed - exact_allowed.mean()
    denom = approx_centered.norm() * exact_centered.norm()
    cosine = float((approx_centered @ exact_centered / denom.clamp_min(1e-12)).item())
    mse = float(torch.mean((approx_allowed - exact_allowed) ** 2).item())
    return cosine, mse


def normalized_code_entropy(codes: torch.Tensor, codewords: int) -> float:
    entropies: list[float] = []
    for head_idx in range(codes.shape[0]):
        for subspace_idx in range(codes.shape[-1]):
            counts = torch.bincount(codes[head_idx, :, subspace_idx].long(), minlength=codewords).float()
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum() / math.log(max(codewords, 2))
            entropies.append(float(entropy.item()))
    return float(sum(entropies) / max(len(entropies), 1))


def mean_code_max_fraction(codes: torch.Tensor, codewords: int) -> float:
    fractions: list[float] = []
    for head_idx in range(codes.shape[0]):
        for subspace_idx in range(codes.shape[-1]):
            counts = torch.bincount(codes[head_idx, :, subspace_idx].long(), minlength=codewords).float()
            fractions.append(float((counts.max() / counts.sum().clamp_min(1.0)).item()))
    return float(sum(fractions) / max(len(fractions), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether low-rank rotations improve SVA PQ catalogs.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--contexts", default="2048,8192")
    parser.add_argument("--placements", default="middle,end")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--variants", default="artifact_identity,refit_identity,hadamard,signed_hadamard")
    parser.add_argument("--budgets", default="512,1024,2048")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--query-samples", type=int, default=32)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--key", default="731942")
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
    budgets = comma_ints(args.budgets)
    variants = comma_strings(args.variants)
    layers = parse_layer_list(args.layers, len(model.model.layers))
    layers = layers if layers is not None else list(range(len(model.model.layers)))
    subspaces = bundle.coarse_subspaces
    codewords = bundle.coarse_codewords

    print("rotation_diagnostic_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"artifact_profile,{bundle.manifest.get('profile_name')}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"placements,{args.placements}", flush=True)
    print(f"layers,{args.layers}", flush=True)
    print(f"variants,{args.variants}", flush=True)

    aggregate: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))

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

            query_positions = torch.linspace(
                max(0, min(args.min_query_pos, context - 1)),
                context - 1,
                steps=min(args.query_samples, context),
                device=device,
            ).long().unique()
            if query_positions[-1].item() != context - 1:
                query_positions = torch.unique(torch.cat([query_positions, torch.tensor([context - 1], device=device)]))
            position_ids = torch.arange(context, device=device).unsqueeze(0)

            for layer_idx in layers:
                query_all, key_all, _, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
                q_low, k_low, artifact_codebooks, artifact_codes = project_catalog(query_all, key_all, bundle, layer_idx, args.assign_chunk_size)
                full_scores = torch.einsum("hqd,hkd->hqk", query_all[:, query_positions].float(), key_all.float()) * scaling
                low_scores = torch.einsum("hqr,hkr->hqk", q_low[:, query_positions].float(), k_low.float()) / math.sqrt(bundle.rank_dim)
                teacher_idx, teacher_valid = topk_targets(full_scores, query_positions, args.topk)
                low_idx, low_valid = topk_targets(low_scores, query_positions, args.topk)

                for variant in variants:
                    if variant == "artifact_identity":
                        rotated_q = q_low[:, query_positions]
                        codebooks = artifact_codebooks
                        codes = artifact_codes
                    else:
                        rotation_label = "identity" if variant == "refit_identity" else variant
                        rotation = rotation_matrix(
                            rotation_label,
                            bundle.rank_dim,
                            args.seed + context * 17 + layer_idx * 101,
                            device,
                        )
                        rotated_q = apply_rotation(q_low[:, query_positions], rotation)
                        rotated_k = apply_rotation(k_low, rotation)
                        codebooks = fit_product_codebooks(
                            rotated_k,
                            subspaces,
                            codewords,
                            args.kmeans_iters,
                            args.seed + context * 31 + layer_idx * 997,
                            args.assign_chunk_size,
                        )
                        codes = encode_product_keys(rotated_k, codebooks, args.assign_chunk_size)

                    pq_scores = product_quantized_scores(rotated_q, codebooks, codes, bundle.rank_dim)
                    align_cosine, align_mse = score_alignment(pq_scores, low_scores, query_positions)
                    entropy = normalized_code_entropy(codes, codewords)
                    max_fraction = mean_code_max_fraction(codes, codewords)
                    for budget in budgets:
                        teacher_recall, teacher_hits, teacher_total = candidate_recall(
                            pq_scores,
                            teacher_idx,
                            teacher_valid,
                            query_positions,
                            budget,
                        )
                        low_recall, low_hits, low_total = candidate_recall(
                            pq_scores,
                            low_idx,
                            low_valid,
                            query_positions,
                            budget,
                        )
                        row = {
                            "context": context,
                            "placement": placement,
                            "layer": layer_idx,
                            "variant": variant,
                            "budget": budget,
                            "topk": args.topk,
                            "query_samples": int(query_positions.numel()),
                            "teacher_topk_recall": teacher_recall,
                            "teacher_hits": teacher_hits,
                            "teacher_total": teacher_total,
                            "low_topk_recall": low_recall,
                            "low_hits": low_hits,
                            "low_total": low_total,
                            "score_cosine": align_cosine,
                            "score_mse": align_mse,
                            "code_entropy": entropy,
                            "code_max_fraction": max_fraction,
                        }
                        emit("rotation_result", row)
                        bucket = aggregate[(variant, budget)]
                        bucket["rows"] += 1.0
                        for metric in (
                            "teacher_topk_recall",
                            "low_topk_recall",
                            "score_cosine",
                            "score_mse",
                            "code_entropy",
                            "code_max_fraction",
                        ):
                            bucket[metric] += float(row[metric])

                del query_all, key_all, q_low, k_low, artifact_codebooks, artifact_codes
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for (variant, budget), bucket in sorted(aggregate.items()):
        rows = max(bucket["rows"], 1.0)
        emit(
            "rotation_summary",
            {
                "variant": variant,
                "budget": budget,
                "teacher_topk_recall": bucket["teacher_topk_recall"] / rows,
                "low_topk_recall": bucket["low_topk_recall"] / rows,
                "score_cosine": bucket["score_cosine"] / rows,
                "score_mse": bucket["score_mse"] / rows,
                "code_entropy": bucket["code_entropy"] / rows,
                "code_max_fraction": bucket["code_max_fraction"] / rows,
                "rows": int(rows),
            },
        )

    print("rotation_diagnostic_done", flush=True)


if __name__ == "__main__":
    main()
