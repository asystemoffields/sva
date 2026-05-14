"""Supervised coarse-stage PQ lookup test for SVA.

This keeps the strong fine `16x256` PQ scorer from the coarse-to-fine test,
then trains a separate cheap coarse scorer to summon the fine-PQ winners.
It can also fit attention-weighted coarse codebooks directly in the fine
ranker space, testing whether the coarse catalog should be optimized for
candidate survival instead of learned as a separate scorer.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sva_coarse_to_fine_pq_test import parse_pq_configs
from sva_learned_ivf_lookup_test import assign_to_centroids, learned_scores, project_keys, project_queries
from sva_learned_ranker_test import (
    LowRankRanker,
    layer_qk,
    make_variant_text,
    set_seed,
    train_ranker,
)
from sva_pq_lookup_test import (
    causal_mask_scores,
    ceil_log2,
    encode_product_keys,
    fit_product_codebooks,
    product_quantized_scores,
)
from sva_real_qk_address_sweep import (
    comma_ints,
    make_long_text,
    parse_layers,
    percentile,
    sample_query_positions,
    topk_indices_for_queries,
)


AggregateKey = tuple[str, int, int, int, int, int, int, int, int]


def comma_floats(value: str) -> list[float]:
    if not value:
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def boost_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def candidate_counts(eval_positions: np.ndarray, n_heads: int, budget: int) -> list[int]:
    counts = np.minimum(eval_positions + 1, budget).astype(np.int64)
    return np.tile(counts[None, :], (n_heads, 1)).reshape(-1).tolist()


@torch.no_grad()
def key_label_weights(
    label_idx: np.ndarray,
    label_valid: np.ndarray,
    seq_len: int,
    boost: float,
    device: torch.device,
) -> torch.Tensor:
    n_heads = label_idx.shape[0]
    weights = torch.ones(n_heads, seq_len, device=device, dtype=torch.float32)
    for head_idx in range(n_heads):
        valid_idx = label_idx[head_idx][label_valid[head_idx]]
        if valid_idx.size == 0:
            continue
        idx = torch.tensor(valid_idx.reshape(-1), device=device, dtype=torch.long).clamp(0, seq_len - 1)
        weights[head_idx].index_add_(0, idx, torch.full_like(idx, float(boost), dtype=torch.float32))
    return weights


@torch.no_grad()
def fit_weighted_product_codebooks(
    k_low: torch.Tensor,
    weights: torch.Tensor,
    subspaces: int,
    codewords: int,
    iterations: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    n_heads, seq_len, rank_dim = k_low.shape
    if rank_dim % subspaces != 0:
        raise ValueError(f"rank_dim={rank_dim} must be divisible by subspaces={subspaces}")
    if weights.shape != (n_heads, seq_len):
        raise ValueError("Weight shape must match key heads and sequence length.")

    sub_dim = rank_dim // subspaces
    actual_codewords = min(codewords, seq_len)
    codebooks: list[torch.Tensor] = []

    for head_idx in range(n_heads):
        head_codebooks: list[torch.Tensor] = []
        head_weights = weights[head_idx].float().clamp_min(1e-6)
        probs = head_weights / head_weights.sum().clamp_min(1e-6)
        for subspace_idx in range(subspaces):
            start = subspace_idx * sub_dim
            end = start + sub_dim
            x = k_low[head_idx, :, start:end].float()
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed + head_idx * 9973 + subspace_idx * 131 + actual_codewords)
            initial = torch.multinomial(probs, actual_codewords, replacement=False, generator=generator)
            centroids = x[initial].clone()

            for _ in range(iterations):
                labels = assign_to_centroids(x, centroids, chunk_size)
                sums = torch.zeros(actual_codewords, sub_dim, device=x.device, dtype=torch.float32)
                weighted_x = x * head_weights[:, None]
                sums.index_add_(0, labels, weighted_x)
                counts = torch.zeros(actual_codewords, device=x.device, dtype=torch.float32)
                counts.index_add_(0, labels, head_weights)
                nonempty = counts > 0
                next_centroids = centroids.clone()
                next_centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
                centroids = next_centroids

            head_codebooks.append(centroids.to(k_low.dtype))
        codebooks.append(torch.stack(head_codebooks, dim=0))

    return torch.stack(codebooks, dim=0)


@torch.no_grad()
def topk_labels_from_scores(
    scores: torch.Tensor,
    query_positions: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    masked = causal_mask_scores(scores, query_positions)
    actual_topk = min(topk, scores.shape[-1])
    idx = masked.topk(actual_topk, dim=-1).indices
    positions = torch.tensor(query_positions, device=scores.device, dtype=torch.long)
    valid = idx <= positions[None, :, None]
    return idx.cpu().numpy(), valid.cpu().numpy()


@torch.no_grad()
def evaluate_scores(
    label: str,
    layer_idx: int | str,
    seq_len: int,
    fine_rank_dim: int,
    coarse_rank_dim: int,
    coarse_label_topk: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    fine_subspaces: int,
    fine_codewords: int,
    shortlist: int,
    budget: int,
    kmeans_iters: int,
    fine_train_steps: int,
    coarse_train_steps: int,
    fine_loss: float,
    coarse_loss: float,
    scores: torch.Tensor,
    eval_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    aggregate: dict[AggregateKey, dict[str, object]],
) -> None:
    masked_scores = causal_mask_scores(scores, eval_positions)
    actual_budget = min(budget, seq_len)
    candidate_idx = masked_scores.topk(actual_budget, dim=-1).indices
    candidate_mask = torch.zeros_like(masked_scores, dtype=torch.bool)
    candidate_mask.scatter_(dim=-1, index=candidate_idx, value=True)

    top_idx_t = torch.tensor(top_idx, device=scores.device, dtype=torch.long)
    top_valid_t = torch.tensor(top_valid, device=scores.device, dtype=torch.bool)
    hits_t = candidate_mask.gather(dim=-1, index=top_idx_t.clamp(0, seq_len - 1)) & top_valid_t
    hits = int(hits_t.sum().item())
    total = int(top_valid_t.sum().item())
    exact_counts = candidate_counts(eval_positions, scores.shape[0], budget)
    fine_counts = candidate_counts(eval_positions, scores.shape[0], budget)
    print_result(
        label,
        layer_idx,
        seq_len,
        fine_rank_dim,
        coarse_rank_dim,
        coarse_label_topk,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
        kmeans_iters,
        fine_train_steps,
        coarse_train_steps,
        fine_loss,
        coarse_loss,
        exact_counts,
        fine_counts,
        hits,
        total,
    )

    key = (
        label,
        coarse_rank_dim,
        coarse_label_topk,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
    )
    bucket = aggregate[key]
    bucket["exact_counts"].extend(exact_counts)
    bucket["fine_counts"].extend(fine_counts)
    bucket["hits"] = int(bucket["hits"]) + hits
    bucket["total"] = int(bucket["total"]) + total


@torch.no_grad()
def evaluate_stage(
    label: str,
    layer_idx: int | str,
    seq_len: int,
    fine_rank_dim: int,
    coarse_rank_dim: int,
    coarse_label_topk: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    fine_subspaces: int,
    fine_codewords: int,
    shortlist: int,
    budget: int,
    kmeans_iters: int,
    fine_train_steps: int,
    coarse_train_steps: int,
    fine_loss: float,
    coarse_loss: float,
    coarse_scores: torch.Tensor,
    fine_scores: torch.Tensor,
    eval_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    aggregate: dict[AggregateKey, dict[str, object]],
) -> None:
    n_heads, _, seq_len_from_scores = coarse_scores.shape
    if seq_len_from_scores != seq_len:
        raise ValueError("Score sequence length mismatch.")

    masked_coarse = causal_mask_scores(coarse_scores, eval_positions)
    actual_shortlist = min(shortlist, seq_len)
    coarse_idx = masked_coarse.topk(actual_shortlist, dim=-1).indices
    gathered_fine = fine_scores.gather(dim=-1, index=coarse_idx)
    positions = torch.tensor(eval_positions, device=fine_scores.device, dtype=torch.long)
    valid_shortlist = coarse_idx <= positions[None, :, None]
    gathered_fine = gathered_fine.masked_fill(~valid_shortlist, torch.finfo(gathered_fine.dtype).min)
    actual_budget = min(budget, actual_shortlist)
    keep_in_shortlist = gathered_fine.topk(actual_budget, dim=-1).indices
    final_idx = coarse_idx.gather(dim=-1, index=keep_in_shortlist)

    candidate_mask = torch.zeros_like(fine_scores, dtype=torch.bool)
    candidate_mask.scatter_(dim=-1, index=final_idx, value=True)
    top_idx_t = torch.tensor(top_idx, device=fine_scores.device, dtype=torch.long)
    top_valid_t = torch.tensor(top_valid, device=fine_scores.device, dtype=torch.bool)
    hits_t = candidate_mask.gather(dim=-1, index=top_idx_t.clamp(0, seq_len - 1)) & top_valid_t
    hits = int(hits_t.sum().item())
    total = int(top_valid_t.sum().item())
    exact_counts = candidate_counts(eval_positions, n_heads, budget)
    fine_counts = candidate_counts(eval_positions, n_heads, shortlist)
    print_result(
        label,
        layer_idx,
        seq_len,
        fine_rank_dim,
        coarse_rank_dim,
        coarse_label_topk,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
        kmeans_iters,
        fine_train_steps,
        coarse_train_steps,
        fine_loss,
        coarse_loss,
        exact_counts,
        fine_counts,
        hits,
        total,
    )

    key = (
        label,
        coarse_rank_dim,
        coarse_label_topk,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
    )
    bucket = aggregate[key]
    bucket["exact_counts"].extend(exact_counts)
    bucket["fine_counts"].extend(fine_counts)
    bucket["hits"] = int(bucket["hits"]) + hits
    bucket["total"] = int(bucket["total"]) + total


def print_result(
    label: str,
    layer_idx: int | str,
    seq_len: int,
    fine_rank_dim: int,
    coarse_rank_dim: int,
    coarse_label_topk: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    fine_subspaces: int,
    fine_codewords: int,
    shortlist: int,
    budget: int,
    kmeans_iters: int,
    fine_train_steps: int,
    coarse_train_steps: int,
    fine_loss: float,
    coarse_loss: float,
    exact_counts: list[int],
    fine_counts: list[int],
    hits: int,
    total: int,
) -> None:
    recall = hits / total if total else float("nan")
    exact_avg = sum(exact_counts) / max(len(exact_counts), 1)
    fine_avg = sum(fine_counts) / max(len(fine_counts), 1)
    coarse_bits = coarse_subspaces * ceil_log2(max(coarse_codewords, 2)) if coarse_codewords > 0 else 0
    fine_bits = fine_subspaces * ceil_log2(max(fine_codewords, 2)) if fine_codewords > 0 else 0
    print(
        "supervised_coarse_pq_result,"
        f"{label},{layer_idx},{seq_len},{fine_rank_dim},{coarse_rank_dim},{coarse_label_topk},"
        f"{coarse_subspaces},{coarse_codewords},{coarse_bits},"
        f"{fine_subspaces},{fine_codewords},{fine_bits},"
        f"{shortlist},{budget},{kmeans_iters},{fine_train_steps},{coarse_train_steps},"
        f"{fine_loss:.6f},{coarse_loss:.6f},"
        f"{exact_avg:.1f},{percentile(exact_counts, 50):.1f},{percentile(exact_counts, 95):.1f},"
        f"{fine_avg:.1f},{percentile(fine_counts, 50):.1f},{percentile(fine_counts, 95):.1f},"
        f"{recall:.6f},{hits},{total}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised coarse PQ stage for SVA.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="reverse")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--fine-rank-dim", type=int, default=64)
    parser.add_argument("--coarse-rank-dims", default="16,32,64")
    parser.add_argument("--coarse-label-topk", type=int, default=512)
    parser.add_argument("--coarse-label-source", choices=["fine_pq", "attention"], default="fine_pq")
    parser.add_argument("--weighted-coarse-boosts", default="")
    parser.add_argument("--coarse-configs", default="4x16,4x64,8x16")
    parser.add_argument("--fine-configs", default="16x256")
    parser.add_argument("--shortlists", default="1024,2048,4096")
    parser.add_argument("--budgets", default="512")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--eval-query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--fine-train-steps", type=int, default=160)
    parser.add_argument("--coarse-train-steps", type=int, default=160)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(getattr(config, "max_position_embeddings", 0) or 0)
    requested = args.max_length if args.max_length > 0 else model_window
    if requested <= 0:
        requested = 8192
    effective_max_length = min(requested, model_window) if model_window > 0 else requested

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_text = make_long_text(args.text_repeats)
    eval_repeats = args.eval_text_repeats if args.eval_text_repeats > 0 else args.text_repeats
    eval_text = make_variant_text(eval_repeats, args.eval_text_mode)
    train_batch = tokenizer(
        train_text,
        return_tensors="pt",
        truncation=True,
        max_length=effective_max_length,
    ).to(device)
    eval_batch = tokenizer(
        eval_text,
        return_tensors="pt",
        truncation=True,
        max_length=effective_max_length,
    ).to(device)
    train_seq_len = int(train_batch["input_ids"].shape[1])
    eval_seq_len = int(eval_batch["input_ids"].shape[1])
    layers = parse_layers(args.layers, len(config.layers) if hasattr(config, "layers") else config.num_hidden_layers)
    coarse_rank_dims = comma_ints(args.coarse_rank_dims)
    coarse_configs = parse_pq_configs(args.coarse_configs)
    fine_configs = parse_pq_configs(args.fine_configs)
    weighted_coarse_boosts = comma_floats(args.weighted_coarse_boosts)
    shortlists = comma_ints(args.shortlists)
    budgets = comma_ints(args.budgets)
    train_positions = sample_query_positions(
        train_seq_len,
        max(args.topk, args.coarse_label_topk),
        args.train_query_samples,
        args.min_query_pos,
    )
    eval_positions = sample_query_positions(
        eval_seq_len,
        args.topk,
        args.eval_query_samples,
        args.min_query_pos,
    )

    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"model_max_position_embeddings,{model_window}")
    print(f"requested_max_length,{requested}")
    print(f"effective_max_length,{effective_max_length}")
    print(f"train_seq_len,{train_seq_len}")
    print(f"eval_seq_len,{eval_seq_len}")
    print(f"eval_text_mode,{args.eval_text_mode}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"layers,{';'.join(str(layer) for layer in layers)}")
    print(f"fine_rank_dim,{args.fine_rank_dim}")
    print(f"coarse_rank_dims,{';'.join(str(value) for value in coarse_rank_dims)}")
    print(f"coarse_label_topk,{args.coarse_label_topk}")
    print(f"coarse_label_source,{args.coarse_label_source}")
    print(f"weighted_coarse_boosts,{';'.join(f'{value:g}' for value in weighted_coarse_boosts)}")
    print(f"coarse_configs,{';'.join(f'{a}x{b}' for a, b in coarse_configs)}")
    print(f"fine_configs,{';'.join(f'{a}x{b}' for a, b in fine_configs)}")
    print(f"shortlists,{';'.join(str(value) for value in shortlists)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(
        "supervised_coarse_pq_header,"
        "label,layer,seq_len,fine_rank_dim,coarse_rank_dim,coarse_label_topk,"
        "coarse_subspaces,coarse_codewords,coarse_bits,"
        "fine_subspaces,fine_codewords,fine_bits,"
        "shortlist,budget,kmeans_iters,fine_train_steps,coarse_train_steps,"
        "fine_loss,coarse_loss,"
        "avg_exact_candidates,p50_exact_candidates,p95_exact_candidates,"
        "avg_fine_candidates,p50_fine_candidates,p95_fine_candidates,"
        "topk_recall,hits,total"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    model.eval()
    print("progress,model_loaded", flush=True)
    with torch.no_grad():
        train_outputs = model(**train_batch, output_hidden_states=True, use_cache=False)
        eval_outputs = model(**eval_batch, output_hidden_states=True, use_cache=False)
    print("progress,hidden_states_ready", flush=True)

    train_hidden_states = train_outputs.hidden_states
    eval_hidden_states = eval_outputs.hidden_states
    train_position_ids = torch.arange(train_seq_len, device=device).unsqueeze(0)
    eval_position_ids = torch.arange(eval_seq_len, device=device).unsqueeze(0)
    train_position_tensor = torch.tensor(train_positions, device=device, dtype=torch.long)

    aggregate: dict[AggregateKey, dict[str, object]] = defaultdict(
        lambda: {"exact_counts": [], "fine_counts": [], "hits": 0, "total": 0}
    )

    for layer_idx in layers:
        print(f"progress,layer_start,{layer_idx}", flush=True)
        train_query_all, train_key, train_scaling = layer_qk(
            model,
            train_hidden_states,
            layer_idx,
            train_position_ids,
        )
        eval_query_all, eval_key, eval_scaling = layer_qk(
            model,
            eval_hidden_states,
            layer_idx,
            eval_position_ids,
        )
        train_top_idx, train_top_valid = topk_indices_for_queries(
            train_query_all,
            train_key,
            train_positions,
            max(args.topk, args.coarse_label_topk),
            train_scaling,
        )
        eval_top_idx, eval_top_valid = topk_indices_for_queries(
            eval_query_all,
            eval_key,
            eval_positions,
            args.topk,
            eval_scaling,
        )
        train_query = train_query_all[:, train_position_tensor, :].contiguous()

        torch.manual_seed(args.seed + layer_idx * 1000 + args.fine_rank_dim)
        fine_ranker = LowRankRanker(train_query_all.shape[0], train_query_all.shape[-1], args.fine_rank_dim).to(device)
        fine_loss = train_ranker(
            fine_ranker,
            train_key,
            train_query,
            train_positions,
            train_top_idx[:, :, : args.topk],
            train_top_valid[:, :, : args.topk],
            args.fine_train_steps,
            args.batch_queries,
            args.lr,
            args.weight_decay,
            args.seed + layer_idx * 1000 + args.fine_rank_dim,
        )
        print(f"progress,fine_ranker_trained,{layer_idx},{fine_loss:.6f}", flush=True)

        train_fine_k_low = project_keys(fine_ranker, train_key)
        eval_fine_k_low = project_keys(fine_ranker, eval_key)
        train_fine_q_low = project_queries(fine_ranker, train_query_all, train_positions)
        eval_fine_q_low = project_queries(fine_ranker, eval_query_all, eval_positions)
        exact_scores = learned_scores(eval_fine_q_low, eval_fine_k_low, args.fine_rank_dim).float()

        fine_score_cache: dict[tuple[int, int], torch.Tensor] = {}
        train_fine_label_idx: np.ndarray | None = None
        train_fine_label_valid: np.ndarray | None = None
        for fine_subspaces, fine_codewords in fine_configs:
            fine_codebooks = fit_product_codebooks(
                train_fine_k_low,
                fine_subspaces,
                fine_codewords,
                args.kmeans_iters,
                args.seed + layer_idx * 1000 + fine_subspaces * 101 + fine_codewords,
                args.assign_chunk_size,
            )
            train_fine_codes = encode_product_keys(train_fine_k_low, fine_codebooks, args.assign_chunk_size)
            eval_fine_codes = encode_product_keys(eval_fine_k_low, fine_codebooks, args.assign_chunk_size)
            train_fine_scores = product_quantized_scores(
                train_fine_q_low,
                fine_codebooks,
                train_fine_codes,
                args.fine_rank_dim,
            )
            eval_fine_scores = product_quantized_scores(
                eval_fine_q_low,
                fine_codebooks,
                eval_fine_codes,
                args.fine_rank_dim,
            )
            actual_fine_codewords = int(fine_codebooks.shape[2])
            fine_score_cache[(fine_subspaces, actual_fine_codewords)] = eval_fine_scores
            print(
                f"progress,fine_pq_ready,{layer_idx},{fine_subspaces},{actual_fine_codewords}",
                flush=True,
            )
            if train_fine_label_idx is None and args.coarse_label_source == "fine_pq":
                train_fine_label_idx, train_fine_label_valid = topk_labels_from_scores(
                    train_fine_scores,
                    train_positions,
                    args.coarse_label_topk,
                )
                print(f"progress,fine_pq_labels_ready,{layer_idx},{args.coarse_label_topk}", flush=True)

            for budget in budgets:
                evaluate_scores(
                    "fine_pq",
                    layer_idx,
                    eval_seq_len,
                    args.fine_rank_dim,
                    0,
                    args.coarse_label_topk,
                    0,
                    0,
                    fine_subspaces,
                    actual_fine_codewords,
                    budget,
                    budget,
                    args.kmeans_iters,
                    args.fine_train_steps,
                    0,
                    fine_loss,
                    float("nan"),
                    eval_fine_scores,
                    eval_positions,
                    eval_top_idx,
                    eval_top_valid,
                    aggregate,
                )

            del fine_codebooks, train_fine_codes, eval_fine_codes, train_fine_scores
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for budget in budgets:
            evaluate_scores(
                "exact_ranker",
                layer_idx,
                eval_seq_len,
                args.fine_rank_dim,
                0,
                args.coarse_label_topk,
                0,
                0,
                0,
                0,
                budget,
                budget,
                0,
                args.fine_train_steps,
                0,
                fine_loss,
                float("nan"),
                exact_scores,
                eval_positions,
                eval_top_idx,
                eval_top_valid,
                aggregate,
            )

        for coarse_subspaces, coarse_codewords in coarse_configs:
            if args.fine_rank_dim % coarse_subspaces != 0:
                continue
            baseline_codebooks = fit_product_codebooks(
                train_fine_k_low,
                coarse_subspaces,
                coarse_codewords,
                args.kmeans_iters,
                args.seed + layer_idx * 1000 + coarse_subspaces * 101 + coarse_codewords,
                args.assign_chunk_size,
            )
            baseline_codes = encode_product_keys(eval_fine_k_low, baseline_codebooks, args.assign_chunk_size)
            baseline_scores = product_quantized_scores(
                eval_fine_q_low,
                baseline_codebooks,
                baseline_codes,
                args.fine_rank_dim,
            )
            actual_coarse_codewords = int(baseline_codebooks.shape[2])
            print(
                f"progress,unsupervised_coarse_pq_ready,{layer_idx},{coarse_subspaces},{actual_coarse_codewords}",
                flush=True,
            )
            for (fine_subspaces, fine_codewords), fine_scores in fine_score_cache.items():
                for shortlist in shortlists:
                    for budget in budgets:
                        if shortlist < budget:
                            continue
                        evaluate_stage(
                            "unsupervised_coarse_to_fine",
                            layer_idx,
                            eval_seq_len,
                            args.fine_rank_dim,
                            args.fine_rank_dim,
                            args.coarse_label_topk,
                            coarse_subspaces,
                            actual_coarse_codewords,
                            fine_subspaces,
                            fine_codewords,
                            shortlist,
                            budget,
                            args.kmeans_iters,
                            args.fine_train_steps,
                            0,
                            fine_loss,
                            float("nan"),
                            baseline_scores,
                            fine_scores,
                            eval_positions,
                            eval_top_idx,
                            eval_top_valid,
                            aggregate,
                        )
            del baseline_codebooks, baseline_codes, baseline_scores
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if args.coarse_label_source == "attention":
            coarse_label_idx = train_top_idx[:, :, : args.coarse_label_topk]
            coarse_label_valid = train_top_valid[:, :, : args.coarse_label_topk]
        else:
            if train_fine_label_idx is None or train_fine_label_valid is None:
                raise RuntimeError("Fine-PQ labels were not built.")
            coarse_label_idx = train_fine_label_idx
            coarse_label_valid = train_fine_label_valid

        if weighted_coarse_boosts:
            for boost in weighted_coarse_boosts:
                weights = key_label_weights(
                    coarse_label_idx,
                    coarse_label_valid,
                    train_seq_len,
                    boost,
                    device,
                )
                for coarse_subspaces, coarse_codewords in coarse_configs:
                    if args.fine_rank_dim % coarse_subspaces != 0:
                        continue
                    weighted_codebooks = fit_weighted_product_codebooks(
                        train_fine_k_low,
                        weights,
                        coarse_subspaces,
                        coarse_codewords,
                        args.kmeans_iters,
                        args.seed
                        + layer_idx * 1000
                        + int(boost * 1000)
                        + coarse_subspaces * 101
                        + coarse_codewords,
                        args.assign_chunk_size,
                    )
                    weighted_codes = encode_product_keys(eval_fine_k_low, weighted_codebooks, args.assign_chunk_size)
                    weighted_scores = product_quantized_scores(
                        eval_fine_q_low,
                        weighted_codebooks,
                        weighted_codes,
                        args.fine_rank_dim,
                    )
                    actual_coarse_codewords = int(weighted_codebooks.shape[2])
                    print(
                        "progress,weighted_coarse_pq_ready,"
                        f"{layer_idx},{boost:g},{coarse_subspaces},{actual_coarse_codewords}",
                        flush=True,
                    )
                    for (fine_subspaces, fine_codewords), fine_scores in fine_score_cache.items():
                        for shortlist in shortlists:
                            for budget in budgets:
                                if shortlist < budget:
                                    continue
                                evaluate_stage(
                                    f"weighted_coarse_to_fine_b{boost_label(boost)}",
                                    layer_idx,
                                    eval_seq_len,
                                    args.fine_rank_dim,
                                    args.fine_rank_dim,
                                    args.coarse_label_topk,
                                    coarse_subspaces,
                                    actual_coarse_codewords,
                                    fine_subspaces,
                                    fine_codewords,
                                    shortlist,
                                    budget,
                                    args.kmeans_iters,
                                    args.fine_train_steps,
                                    0,
                                    fine_loss,
                                    float("nan"),
                                    weighted_scores,
                                    fine_scores,
                                    eval_positions,
                                    eval_top_idx,
                                    eval_top_valid,
                                    aggregate,
                                )
                    del weighted_codebooks, weighted_codes, weighted_scores
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                del weights
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        for coarse_rank_dim in coarse_rank_dims:
            torch.manual_seed(args.seed + layer_idx * 2000 + coarse_rank_dim)
            coarse_ranker = LowRankRanker(train_query_all.shape[0], train_query_all.shape[-1], coarse_rank_dim).to(device)
            coarse_loss = train_ranker(
                coarse_ranker,
                train_key,
                train_query,
                train_positions,
                coarse_label_idx,
                coarse_label_valid,
                args.coarse_train_steps,
                args.batch_queries,
                args.lr,
                args.weight_decay,
                args.seed + layer_idx * 2000 + coarse_rank_dim,
            )
            print(
                f"progress,supervised_coarse_ranker_trained,{layer_idx},{coarse_rank_dim},{coarse_loss:.6f}",
                flush=True,
            )
            train_coarse_k_low = project_keys(coarse_ranker, train_key)
            eval_coarse_k_low = project_keys(coarse_ranker, eval_key)
            eval_coarse_q_low = project_queries(coarse_ranker, eval_query_all, eval_positions)

            for coarse_subspaces, coarse_codewords in coarse_configs:
                if coarse_rank_dim % coarse_subspaces != 0:
                    continue
                coarse_codebooks = fit_product_codebooks(
                    train_coarse_k_low,
                    coarse_subspaces,
                    coarse_codewords,
                    args.kmeans_iters,
                    args.seed + layer_idx * 1000 + coarse_rank_dim * 17 + coarse_subspaces * 101 + coarse_codewords,
                    args.assign_chunk_size,
                )
                coarse_codes = encode_product_keys(eval_coarse_k_low, coarse_codebooks, args.assign_chunk_size)
                coarse_scores = product_quantized_scores(
                    eval_coarse_q_low,
                    coarse_codebooks,
                    coarse_codes,
                    coarse_rank_dim,
                )
                actual_coarse_codewords = int(coarse_codebooks.shape[2])
                print(
                    f"progress,supervised_coarse_pq_ready,{layer_idx},{coarse_rank_dim},{coarse_subspaces},{actual_coarse_codewords}",
                    flush=True,
                )
                for (fine_subspaces, fine_codewords), fine_scores in fine_score_cache.items():
                    for shortlist in shortlists:
                        for budget in budgets:
                            if shortlist < budget:
                                continue
                            evaluate_stage(
                                "supervised_coarse_to_fine",
                                layer_idx,
                                eval_seq_len,
                                args.fine_rank_dim,
                                coarse_rank_dim,
                                args.coarse_label_topk,
                                coarse_subspaces,
                                actual_coarse_codewords,
                                fine_subspaces,
                                fine_codewords,
                                shortlist,
                                budget,
                                args.kmeans_iters,
                                args.fine_train_steps,
                                args.coarse_train_steps,
                                fine_loss,
                                coarse_loss,
                                coarse_scores,
                                fine_scores,
                                eval_positions,
                                eval_top_idx,
                                eval_top_valid,
                                aggregate,
                            )
                del coarse_codebooks, coarse_codes, coarse_scores
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del coarse_ranker, train_coarse_k_low, eval_coarse_k_low, eval_coarse_q_low
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del fine_score_cache
        del train_query_all, train_key, eval_query_all, eval_key, train_query
        del fine_ranker, train_fine_k_low, eval_fine_k_low, train_fine_q_low, eval_fine_q_low, exact_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (
        label,
        coarse_rank_dim,
        coarse_label_topk,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
    ), bucket in sorted(aggregate.items()):
        print_result(
            label,
            "all",
            eval_seq_len,
            args.fine_rank_dim,
            coarse_rank_dim,
            coarse_label_topk,
            coarse_subspaces,
            coarse_codewords,
            fine_subspaces,
            fine_codewords,
            shortlist,
            budget,
            args.kmeans_iters if label != "exact_ranker" else 0,
            args.fine_train_steps,
            args.coarse_train_steps if label == "supervised_coarse_to_fine" else 0,
            float("nan"),
            float("nan"),
            bucket["exact_counts"],  # type: ignore[arg-type]
            bucket["fine_counts"],  # type: ignore[arg-type]
            int(bucket["hits"]),
            int(bucket["total"]),
        )


if __name__ == "__main__":
    main()
