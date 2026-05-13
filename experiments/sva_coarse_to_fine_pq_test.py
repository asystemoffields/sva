"""Coarse-to-fine product-quantized lookup test for SVA."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sva_learned_ivf_lookup_test import learned_scores, project_keys, project_queries
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


def parse_pq_configs(value: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for item in value.split(","):
        clean = item.strip().lower().replace(" ", "")
        if not clean:
            continue
        if "x" not in clean:
            raise ValueError(f"Expected PQ config like 8x256, got {item!r}")
        left, right = clean.split("x", 1)
        configs.append((int(left), int(right)))
    return configs


def candidate_counts(eval_positions: np.ndarray, n_heads: int, budget: int) -> list[int]:
    counts = np.minimum(eval_positions + 1, budget).astype(np.int64)
    return np.tile(counts[None, :], (n_heads, 1)).reshape(-1).tolist()


@torch.no_grad()
def evaluate_topk_scores(
    label: str,
    layer_idx: int | str,
    seq_len: int,
    rank_dim: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    fine_subspaces: int,
    fine_codewords: int,
    shortlist: int,
    budget: int,
    kmeans_iters: int,
    train_steps: int,
    final_loss: float,
    scores: torch.Tensor,
    eval_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    aggregate: dict[tuple[str, int, int, int, int, int, int], dict[str, object]],
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
    print_ctf_result(
        label,
        layer_idx,
        seq_len,
        rank_dim,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
        kmeans_iters,
        train_steps,
        final_loss,
        exact_counts,
        fine_counts,
        hits,
        total,
    )

    key = (label, coarse_subspaces, coarse_codewords, fine_subspaces, fine_codewords, shortlist, budget)
    bucket = aggregate[key]
    bucket["exact_counts"].extend(exact_counts)
    bucket["fine_counts"].extend(fine_counts)
    bucket["hits"] = int(bucket["hits"]) + hits
    bucket["total"] = int(bucket["total"]) + total


@torch.no_grad()
def evaluate_coarse_to_fine(
    layer_idx: int | str,
    seq_len: int,
    rank_dim: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    fine_subspaces: int,
    fine_codewords: int,
    shortlist: int,
    budget: int,
    kmeans_iters: int,
    train_steps: int,
    final_loss: float,
    coarse_scores: torch.Tensor,
    fine_scores: torch.Tensor,
    eval_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    aggregate: dict[tuple[str, int, int, int, int, int, int], dict[str, object]],
) -> None:
    n_heads, n_queries, seq_len_from_scores = coarse_scores.shape
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
    print_ctf_result(
        "coarse_to_fine",
        layer_idx,
        seq_len,
        rank_dim,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
        kmeans_iters,
        train_steps,
        final_loss,
        exact_counts,
        fine_counts,
        hits,
        total,
    )

    key = (
        "coarse_to_fine",
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


def print_ctf_result(
    label: str,
    layer_idx: int | str,
    seq_len: int,
    rank_dim: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    fine_subspaces: int,
    fine_codewords: int,
    shortlist: int,
    budget: int,
    kmeans_iters: int,
    train_steps: int,
    final_loss: float,
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
        "coarse_to_fine_pq_result,"
        f"{label},{layer_idx},{seq_len},{rank_dim},"
        f"{coarse_subspaces},{coarse_codewords},{coarse_bits},"
        f"{fine_subspaces},{fine_codewords},{fine_bits},"
        f"{shortlist},{budget},{kmeans_iters},{train_steps},{final_loss:.6f},"
        f"{exact_avg:.1f},{percentile(exact_counts, 50):.1f},{percentile(exact_counts, 95):.1f},"
        f"{fine_avg:.1f},{percentile(fine_counts, 50):.1f},{percentile(fine_counts, 95):.1f},"
        f"{recall:.6f},{hits},{total}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve learned SVA ranker with coarse-to-fine PQ.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="reverse")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--coarse-configs", default="4x16,8x16,4x64")
    parser.add_argument("--fine-configs", default="8x256,16x256")
    parser.add_argument("--shortlists", default="1024,2048,4096")
    parser.add_argument("--budgets", default="256,512")
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
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))
    coarse_configs = parse_pq_configs(args.coarse_configs)
    fine_configs = parse_pq_configs(args.fine_configs)
    shortlists = comma_ints(args.shortlists)
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
    print(f"coarse_configs,{';'.join(f'{a}x{b}' for a, b in coarse_configs)}")
    print(f"fine_configs,{';'.join(f'{a}x{b}' for a, b in fine_configs)}")
    print(f"shortlists,{';'.join(str(value) for value in shortlists)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(
        "coarse_to_fine_pq_header,"
        "label,layer,seq_len,rank_dim,"
        "coarse_subspaces,coarse_codewords,coarse_bits,"
        "fine_subspaces,fine_codewords,fine_bits,"
        "shortlist,budget,kmeans_iters,train_steps,final_loss,"
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

    aggregate: dict[tuple[str, int, int, int, int, int, int], dict[str, object]] = defaultdict(
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

        fine_score_cache: dict[tuple[int, int], torch.Tensor] = {}
        for fine_subspaces, fine_codewords in fine_configs:
            fine_codebooks = fit_product_codebooks(
                train_k_low,
                fine_subspaces,
                fine_codewords,
                args.kmeans_iters,
                args.seed + layer_idx * 1000 + fine_subspaces * 101 + fine_codewords,
                args.assign_chunk_size,
            )
            fine_codes = encode_product_keys(eval_k_low, fine_codebooks, args.assign_chunk_size)
            fine_scores = product_quantized_scores(eval_q_low, fine_codebooks, fine_codes, args.rank_dim)
            fine_score_cache[(fine_subspaces, int(fine_codebooks.shape[2]))] = fine_scores
            print(
                f"progress,fine_pq_ready,{layer_idx},{fine_subspaces},{int(fine_codebooks.shape[2])}",
                flush=True,
            )
            for budget in budgets:
                evaluate_topk_scores(
                    "fine_pq",
                    layer_idx,
                    eval_seq_len,
                    args.rank_dim,
                    0,
                    0,
                    fine_subspaces,
                    int(fine_codebooks.shape[2]),
                    budget,
                    budget,
                    args.kmeans_iters,
                    args.train_steps,
                    final_loss,
                    fine_scores,
                    eval_positions,
                    eval_top_idx,
                    eval_top_valid,
                    aggregate,
                )
            del fine_codebooks, fine_codes
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for budget in budgets:
            evaluate_topk_scores(
                "exact_ranker",
                layer_idx,
                eval_seq_len,
                args.rank_dim,
                0,
                0,
                0,
                0,
                budget,
                budget,
                0,
                args.train_steps,
                final_loss,
                exact_scores,
                eval_positions,
                eval_top_idx,
                eval_top_valid,
                aggregate,
            )

        for coarse_subspaces, coarse_codewords in coarse_configs:
            coarse_codebooks = fit_product_codebooks(
                train_k_low,
                coarse_subspaces,
                coarse_codewords,
                args.kmeans_iters,
                args.seed + layer_idx * 1000 + coarse_subspaces * 101 + coarse_codewords,
                args.assign_chunk_size,
            )
            coarse_codes = encode_product_keys(eval_k_low, coarse_codebooks, args.assign_chunk_size)
            coarse_scores = product_quantized_scores(eval_q_low, coarse_codebooks, coarse_codes, args.rank_dim)
            actual_coarse_codewords = int(coarse_codebooks.shape[2])
            print(
                f"progress,coarse_pq_ready,{layer_idx},{coarse_subspaces},{actual_coarse_codewords}",
                flush=True,
            )

            for (fine_subspaces, fine_codewords), fine_scores in fine_score_cache.items():
                for shortlist in shortlists:
                    for budget in budgets:
                        if shortlist < budget:
                            continue
                        evaluate_coarse_to_fine(
                            layer_idx,
                            eval_seq_len,
                            args.rank_dim,
                            coarse_subspaces,
                            actual_coarse_codewords,
                            fine_subspaces,
                            fine_codewords,
                            shortlist,
                            budget,
                            args.kmeans_iters,
                            args.train_steps,
                            final_loss,
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

        del fine_score_cache
        del train_query_all, train_key, eval_query_all, eval_key, train_query, ranker
        del train_k_low, eval_k_low, eval_q_low, exact_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (
        label,
        coarse_subspaces,
        coarse_codewords,
        fine_subspaces,
        fine_codewords,
        shortlist,
        budget,
    ), bucket in sorted(aggregate.items()):
        print_ctf_result(
            label,
            "all",
            eval_seq_len,
            args.rank_dim,
            coarse_subspaces,
            coarse_codewords,
            fine_subspaces,
            fine_codewords,
            shortlist,
            budget,
            args.kmeans_iters if label != "exact_ranker" else 0,
            args.train_steps,
            float("nan"),
            bucket["exact_counts"],  # type: ignore[arg-type]
            bucket["fine_counts"],  # type: ignore[arg-type]
            int(bucket["hits"]),
            int(bucket["total"]),
        )


if __name__ == "__main__":
    main()
