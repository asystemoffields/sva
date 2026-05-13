"""Product-quantized learned-score lookup test for SVA."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sva_learned_ivf_lookup_test import (
    assign_to_centroids,
    learned_scores,
    project_keys,
    project_queries,
)
from sva_learned_ranker_test import (
    LowRankRanker,
    layer_qk,
    make_variant_text,
    set_seed,
    train_ranker,
)
from sva_real_qk_address_sweep import (
    comma_ints,
    make_long_text,
    parse_layers,
    percentile,
    sample_query_positions,
    topk_indices_for_queries,
)


def ceil_log2(value: int) -> int:
    return max(1, int(math.ceil(math.log2(value))))


def candidate_counts(eval_positions: np.ndarray, n_heads: int, budget: int) -> list[int]:
    counts = np.minimum(eval_positions + 1, budget).astype(np.int64)
    return np.tile(counts[None, :], (n_heads, 1)).reshape(-1).tolist()


def causal_mask_scores(scores: torch.Tensor, query_positions: np.ndarray) -> torch.Tensor:
    positions = torch.tensor(query_positions, device=scores.device, dtype=torch.long)
    key_positions = torch.arange(scores.shape[-1], device=scores.device)
    allowed = key_positions[None, None, :] <= positions[None, :, None]
    return scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)


def score_alignment(approx: torch.Tensor, exact: torch.Tensor, query_positions: np.ndarray) -> tuple[float, float]:
    positions = torch.tensor(query_positions, device=exact.device, dtype=torch.long)
    key_positions = torch.arange(exact.shape[-1], device=exact.device)
    allowed = key_positions[None, None, :] <= positions[None, :, None]
    approx_allowed = approx[allowed.expand_as(approx)].float()
    exact_allowed = exact[allowed.expand_as(exact)].float()
    approx_centered = approx_allowed - approx_allowed.mean()
    exact_centered = exact_allowed - exact_allowed.mean()
    denom = approx_centered.norm() * exact_centered.norm()
    cos = float((approx_centered @ exact_centered / denom.clamp_min(1e-12)).item())
    mse = float(torch.mean((approx_allowed - exact_allowed) ** 2).item())
    return cos, mse


@torch.no_grad()
def fit_product_codebooks(
    k_low: torch.Tensor,
    subspaces: int,
    codewords: int,
    iterations: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    n_heads, seq_len, rank_dim = k_low.shape
    if rank_dim % subspaces != 0:
        raise ValueError(f"rank_dim={rank_dim} must be divisible by subspaces={subspaces}")

    sub_dim = rank_dim // subspaces
    actual_codewords = min(codewords, seq_len)
    codebooks: list[torch.Tensor] = []

    for head_idx in range(n_heads):
        head_codebooks: list[torch.Tensor] = []
        for subspace_idx in range(subspaces):
            start = subspace_idx * sub_dim
            end = start + sub_dim
            x = k_low[head_idx, :, start:end].float()
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed + head_idx * 9973 + subspace_idx * 131 + actual_codewords)
            initial = torch.randperm(seq_len, generator=generator, device=x.device)[:actual_codewords]
            centroids = x[initial].clone()

            for _ in range(iterations):
                labels = assign_to_centroids(x, centroids, chunk_size)
                sums = torch.zeros(actual_codewords, sub_dim, device=x.device, dtype=torch.float32)
                sums.index_add_(0, labels, x)
                counts = torch.bincount(labels, minlength=actual_codewords)
                nonempty = counts > 0
                next_centroids = centroids.clone()
                next_centroids[nonempty] = sums[nonempty] / counts[nonempty].float()[:, None]
                centroids = next_centroids

            head_codebooks.append(centroids.to(k_low.dtype))
        codebooks.append(torch.stack(head_codebooks, dim=0))

    return torch.stack(codebooks, dim=0)


@torch.no_grad()
def encode_product_keys(k_low: torch.Tensor, codebooks: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n_heads, _, rank_dim = k_low.shape
    subspaces = codebooks.shape[1]
    sub_dim = rank_dim // subspaces
    codes: list[torch.Tensor] = []

    for head_idx in range(n_heads):
        head_codes: list[torch.Tensor] = []
        for subspace_idx in range(subspaces):
            start = subspace_idx * sub_dim
            end = start + sub_dim
            labels = assign_to_centroids(
                k_low[head_idx, :, start:end],
                codebooks[head_idx, subspace_idx],
                chunk_size,
            )
            head_codes.append(labels)
        codes.append(torch.stack(head_codes, dim=-1))

    return torch.stack(codes, dim=0)


@torch.no_grad()
def product_quantized_scores(
    q_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
) -> torch.Tensor:
    n_heads, n_queries, _ = q_low.shape
    seq_len = codes.shape[1]
    subspaces = codebooks.shape[1]
    sub_dim = rank_dim // subspaces
    q_parts = q_low.float().reshape(n_heads, n_queries, subspaces, sub_dim)
    scores = torch.zeros(n_heads, n_queries, seq_len, device=q_low.device, dtype=torch.float32)

    for head_idx in range(n_heads):
        for subspace_idx in range(subspaces):
            table = q_parts[head_idx, :, subspace_idx] @ codebooks[head_idx, subspace_idx].float().T
            scores[head_idx] += table[:, codes[head_idx, :, subspace_idx].long()]

    return scores / math.sqrt(rank_dim)


@torch.no_grad()
def evaluate_scores(
    label: str,
    layer_idx: int | str,
    seq_len: int,
    rank_dim: int,
    subspaces: int,
    codewords: int,
    budget: int,
    kmeans_iters: int,
    train_steps: int,
    final_loss: float,
    scores: torch.Tensor,
    eval_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    score_cos: float,
    score_mse: float,
    aggregate: dict[tuple[str, int, int, int, int], dict[str, object]],
) -> None:
    n_heads = scores.shape[0]
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
    counts = candidate_counts(eval_positions, n_heads, budget)
    print_pq_result(
        label,
        layer_idx,
        seq_len,
        rank_dim,
        subspaces,
        codewords,
        budget,
        kmeans_iters,
        train_steps,
        final_loss,
        counts,
        hits,
        total,
        score_cos,
        score_mse,
    )

    key = (label, subspaces, codewords, budget, rank_dim)
    bucket = aggregate[key]
    bucket["counts"].extend(counts)
    bucket["hits"] = int(bucket["hits"]) + hits
    bucket["total"] = int(bucket["total"]) + total
    bucket["score_cos_sum"] = float(bucket["score_cos_sum"]) + score_cos
    bucket["score_mse_sum"] = float(bucket["score_mse_sum"]) + score_mse
    bucket["layers"] = int(bucket["layers"]) + (0 if math.isnan(score_cos) else 1)


def print_pq_result(
    label: str,
    layer_idx: int | str,
    seq_len: int,
    rank_dim: int,
    subspaces: int,
    codewords: int,
    budget: int,
    kmeans_iters: int,
    train_steps: int,
    final_loss: float,
    counts: list[int],
    hits: int,
    total: int,
    score_cos: float,
    score_mse: float,
) -> None:
    recall = hits / total if total else float("nan")
    avg_candidates = sum(counts) / max(len(counts), 1)
    bits_per_key = subspaces * ceil_log2(max(codewords, 2)) if codewords > 0 else 0
    print(
        "pq_lookup_result,"
        f"{label},{layer_idx},{seq_len},{rank_dim},{subspaces},{codewords},{bits_per_key},{budget},"
        f"{kmeans_iters},{train_steps},{final_loss:.6f},"
        f"{avg_candidates:.1f},{percentile(counts, 50):.1f},{percentile(counts, 95):.1f},"
        f"{recall:.6f},{hits},{total},{score_cos:.6f},{score_mse:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve learned SVA ranker with product-quantized scores.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="reverse")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--subspaces", default="4,8,16")
    parser.add_argument("--codewords", default="16,64,256")
    parser.add_argument("--budgets", default="128,256,512")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--eval-query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))
    subspace_values = comma_ints(args.subspaces)
    codeword_values = comma_ints(args.codewords)
    budgets = comma_ints(args.budgets)

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_repeats = args.eval_text_repeats if args.eval_text_repeats > 0 else args.text_repeats
    train_batch = tokenizer(
        [make_long_text(args.text_repeats)],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=effective_max_length,
    )
    eval_batch = tokenizer(
        [make_variant_text(eval_repeats, args.eval_text_mode)],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=effective_max_length,
    )
    train_batch = {key: value.to(device) for key, value in train_batch.items()}
    eval_batch = {key: value.to(device) for key, value in eval_batch.items()}
    train_seq_len = int(train_batch["input_ids"].shape[1])
    eval_seq_len = int(eval_batch["input_ids"].shape[1])
    train_positions = sample_query_positions(
        train_seq_len,
        args.topk,
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
    print(f"rank_dim,{args.rank_dim}")
    print(f"subspaces,{';'.join(str(value) for value in subspace_values)}")
    print(f"codewords,{';'.join(str(value) for value in codeword_values)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(
        "pq_lookup_header,"
        "label,layer,seq_len,rank_dim,subspaces,codewords,bits_per_key,budget,"
        "kmeans_iters,train_steps,final_loss,"
        "avg_candidates,p50_candidates,p95_candidates,"
        "topk_recall,hits,total,score_cos_to_exact_ranker,score_mse_to_exact_ranker"
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

    aggregate: dict[tuple[str, int, int, int, int], dict[str, object]] = defaultdict(
        lambda: {
            "counts": [],
            "hits": 0,
            "total": 0,
            "score_cos_sum": 0.0,
            "score_mse_sum": 0.0,
            "layers": 0,
        }
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
            args.topk,
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

        torch.manual_seed(args.seed + layer_idx * 1000 + args.rank_dim)
        ranker = LowRankRanker(train_query_all.shape[0], train_query_all.shape[-1], args.rank_dim).to(device)
        final_loss = train_ranker(
            ranker,
            train_key,
            train_query,
            train_positions,
            train_top_idx,
            train_top_valid,
            args.train_steps,
            args.batch_queries,
            args.lr,
            args.weight_decay,
            args.seed + layer_idx * 1000 + args.rank_dim,
        )
        print(f"progress,layer_ranker_trained,{layer_idx},{final_loss:.6f}", flush=True)

        train_k_low = project_keys(ranker, train_key)
        eval_k_low = project_keys(ranker, eval_key)
        eval_q_low = project_queries(ranker, eval_query_all, eval_positions)
        exact_scores = learned_scores(eval_q_low, eval_k_low, args.rank_dim).float()
        print(f"progress,layer_low_rank_ready,{layer_idx}", flush=True)

        for budget in budgets:
            evaluate_scores(
                "exact_ranker",
                layer_idx,
                eval_seq_len,
                args.rank_dim,
                0,
                0,
                budget,
                0,
                args.train_steps,
                final_loss,
                exact_scores,
                eval_positions,
                eval_top_idx,
                eval_top_valid,
                float("nan"),
                float("nan"),
                aggregate,
            )

        for subspaces in subspace_values:
            if args.rank_dim % subspaces != 0:
                print(f"progress,skip_subspaces,{layer_idx},{subspaces}", flush=True)
                continue
            for codewords in codeword_values:
                codebooks = fit_product_codebooks(
                    train_k_low,
                    subspaces,
                    codewords,
                    args.kmeans_iters,
                    args.seed + layer_idx * 1000 + subspaces * 101 + codewords,
                    args.assign_chunk_size,
                )
                codes = encode_product_keys(eval_k_low, codebooks, args.assign_chunk_size)
                pq_scores = product_quantized_scores(eval_q_low, codebooks, codes, args.rank_dim)
                cos, mse = score_alignment(pq_scores, exact_scores, eval_positions)
                actual_codewords = int(codebooks.shape[2])
                print(
                    f"progress,pq_ready,{layer_idx},{subspaces},{actual_codewords},{cos:.6f},{mse:.6f}",
                    flush=True,
                )
                for budget in budgets:
                    evaluate_scores(
                        "pq",
                        layer_idx,
                        eval_seq_len,
                        args.rank_dim,
                        subspaces,
                        actual_codewords,
                        budget,
                        args.kmeans_iters,
                        args.train_steps,
                        final_loss,
                        pq_scores,
                        eval_positions,
                        eval_top_idx,
                        eval_top_valid,
                        cos,
                        mse,
                        aggregate,
                    )

                del codebooks, codes, pq_scores
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del train_query_all, train_key, eval_query_all, eval_key, train_query, ranker
        del train_k_low, eval_k_low, eval_q_low, exact_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (label, subspaces, codewords, budget, rank_dim), bucket in sorted(aggregate.items()):
        layer_count = int(bucket["layers"])
        score_cos = float("nan") if layer_count == 0 else float(bucket["score_cos_sum"]) / layer_count
        score_mse = float("nan") if layer_count == 0 else float(bucket["score_mse_sum"]) / layer_count
        print_pq_result(
            label,
            "all",
            eval_seq_len,
            rank_dim,
            subspaces,
            codewords,
            budget,
            args.kmeans_iters if label == "pq" else 0,
            args.train_steps,
            float("nan"),
            bucket["counts"],  # type: ignore[arg-type]
            int(bucket["hits"]),
            int(bucket["total"]),
            score_cos,
            score_mse,
        )


if __name__ == "__main__":
    main()

