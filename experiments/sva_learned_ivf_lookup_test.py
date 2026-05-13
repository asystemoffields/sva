"""Score-aware IVF serving test for the learned SVA compressed ranker."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sva_learned_lsh_lookup_test import bitset_indices, target_prefix_for_position
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


@torch.no_grad()
def project_queries(
    ranker: LowRankRanker,
    query: torch.Tensor,
    query_positions: np.ndarray,
) -> torch.Tensor:
    q_pos = torch.tensor(query_positions, device=query.device, dtype=torch.long)
    sampled_query = query[:, q_pos, :]
    q_low = torch.einsum("hqd,hdr->hqr", sampled_query, ranker.q_proj)
    scale = ranker.logit_scale.exp().clamp(0.01, 100.0)
    return q_low * scale[:, None, None]


@torch.no_grad()
def project_keys(ranker: LowRankRanker, key: torch.Tensor) -> torch.Tensor:
    return torch.einsum("hkd,hdr->hkr", key, ranker.k_proj)


@torch.no_grad()
def learned_scores(q_low: torch.Tensor, k_low: torch.Tensor, rank_dim: int) -> torch.Tensor:
    return torch.einsum("hqr,hkr->hqk", q_low, k_low) / math.sqrt(rank_dim)


@torch.no_grad()
def assign_to_centroids(x: torch.Tensor, centroids: torch.Tensor, chunk_size: int) -> torch.Tensor:
    x_float = x.float()
    centroids_float = centroids.float()
    centroid_norm = (centroids_float * centroids_float).sum(dim=-1)
    labels: list[torch.Tensor] = []
    for start in range(0, x_float.shape[0], chunk_size):
        chunk = x_float[start : start + chunk_size]
        chunk_norm = (chunk * chunk).sum(dim=-1, keepdim=True)
        distance = chunk_norm + centroid_norm[None, :] - 2.0 * (chunk @ centroids_float.T)
        labels.append(distance.argmin(dim=-1))
    return torch.cat(labels, dim=0)


@torch.no_grad()
def fit_kmeans_centroids(
    k_low: torch.Tensor,
    n_centroids: int,
    iterations: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    n_heads, seq_len, rank_dim = k_low.shape
    actual_centroids = min(n_centroids, seq_len)
    all_centroids: list[torch.Tensor] = []

    for head_idx in range(n_heads):
        x = k_low[head_idx].float()
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed + head_idx * 9973 + actual_centroids)
        initial = torch.randperm(seq_len, generator=generator, device=x.device)[:actual_centroids]
        centroids = x[initial].clone()

        for _ in range(iterations):
            labels = assign_to_centroids(x, centroids, chunk_size)
            sums = torch.zeros(actual_centroids, rank_dim, device=x.device, dtype=torch.float32)
            sums.index_add_(0, labels, x)
            counts = torch.bincount(labels, minlength=actual_centroids)
            nonempty = counts > 0
            next_centroids = centroids.clone()
            next_centroids[nonempty] = sums[nonempty] / counts[nonempty].float()[:, None]
            centroids = next_centroids

        all_centroids.append(centroids.to(k_low.dtype))

    return torch.stack(all_centroids, dim=0)


def build_centroid_buckets(labels: np.ndarray, n_centroids: int) -> list[int]:
    buckets = [0 for _ in range(n_centroids)]
    for index, label in enumerate(labels):
        buckets[int(label)] |= 1 << index
    return buckets


@torch.no_grad()
def lookup_ivf_candidates(
    q_low: torch.Tensor,
    centroids: torch.Tensor,
    eval_labels: torch.Tensor,
    query_positions: np.ndarray,
    probes: int,
    target_context: int,
) -> tuple[list[tuple[int, int, int]], list[int], list[float]]:
    n_heads, n_queries, _ = q_low.shape
    seq_len = eval_labels.shape[1]
    n_centroids = centroids.shape[1]
    actual_probes = min(probes, n_centroids)
    candidate_records: list[tuple[int, int, int]] = []
    actual_counts: list[int] = []
    million_counts: list[float] = []

    bucket_sets = [
        build_centroid_buckets(eval_labels[head_idx].cpu().numpy(), n_centroids)
        for head_idx in range(n_heads)
    ]

    for head_idx in range(n_heads):
        route_scores = q_low[head_idx].float() @ centroids[head_idx].float().T
        route_ids = route_scores.topk(actual_probes, dim=-1).indices.cpu().numpy()
        buckets = bucket_sets[head_idx]

        for query_idx in range(n_queries):
            candidates = 0
            for centroid_idx in route_ids[query_idx]:
                candidates |= buckets[int(centroid_idx)]

            query_pos = int(query_positions[query_idx])
            prefix_candidates = candidates & ((1 << (query_pos + 1)) - 1)
            actual_count = prefix_candidates.bit_count()
            actual_counts.append(actual_count)

            prefix_density = actual_count / max(query_pos + 1, 1)
            target_prefix = target_prefix_for_position(query_pos, seq_len, target_context)
            million_counts.append(prefix_density * target_prefix)
            candidate_records.append((head_idx, query_idx, prefix_candidates))

    return candidate_records, actual_counts, million_counts


def evaluate_candidate_budget(
    candidate_records: list[tuple[int, int, int]],
    rank_scores: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    budget: int,
) -> tuple[int, int, int]:
    raw_hits = 0
    verified_hits = 0
    total = 0

    for head_idx, query_idx, prefix_candidates in candidate_records:
        if prefix_candidates.bit_count() <= budget:
            verified_candidates = prefix_candidates
        else:
            indices = bitset_indices(prefix_candidates)
            scores = rank_scores[head_idx, query_idx, indices]
            keep = np.argpartition(scores, -budget)[-budget:]
            verified_candidates = 0
            for index in keep:
                verified_candidates |= 1 << indices[int(index)]

        for rank_idx, key_idx in enumerate(top_idx[head_idx, query_idx]):
            if top_valid[head_idx, query_idx, rank_idx]:
                total += 1
                key_bit = 1 << int(key_idx)
                raw_hits += int(bool(prefix_candidates & key_bit))
                verified_hits += int(bool(verified_candidates & key_bit))

    return raw_hits, verified_hits, total


def print_ivf_result(
    layer_idx: int | str,
    seq_len: int,
    target_context: int,
    rank_dim: int,
    centroids: int,
    probes: int,
    budget: int,
    kmeans_iters: int,
    train_steps: int,
    final_loss: float,
    actual_counts: list[int],
    million_counts: list[float],
    raw_hits: int,
    verified_hits: int,
    total: int,
) -> None:
    raw_recall = raw_hits / total if total else float("nan")
    verified_recall = verified_hits / total if total else float("nan")
    actual_avg = sum(actual_counts) / max(len(actual_counts), 1)
    million_avg = sum(million_counts) / max(len(million_counts), 1)
    print(
        "learned_ivf_result,"
        f"{layer_idx},{seq_len},{target_context},{rank_dim},{centroids},{probes},{budget},"
        f"{kmeans_iters},{train_steps},{final_loss:.6f},"
        f"{actual_avg:.1f},{percentile(actual_counts, 50):.1f},{percentile(actual_counts, 95):.1f},"
        f"{million_avg:.1f},{percentile(million_counts, 50):.1f},{percentile(million_counts, 95):.1f},"
        f"{raw_recall:.6f},{verified_recall:.6f},{raw_hits},{verified_hits},{total}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve learned SVA ranker with IVF routing.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="reverse")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--centroids", default="512,1024,2048")
    parser.add_argument("--probes", default="1,2,4,8")
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
    parser.add_argument("--target-context", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))
    centroid_values = comma_ints(args.centroids)
    probe_values = comma_ints(args.probes)
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
    print(f"target_context,{args.target_context}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"layers,{';'.join(str(layer) for layer in layers)}")
    print(f"rank_dim,{args.rank_dim}")
    print(f"centroids,{';'.join(str(value) for value in centroid_values)}")
    print(f"probes,{';'.join(str(value) for value in probe_values)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(
        "learned_ivf_header,"
        "layer,seq_len,target_context,rank_dim,centroids,probes,budget,kmeans_iters,train_steps,final_loss,"
        "avg_candidates,p50_candidates,p95_candidates,"
        "avg_empirical_million_candidates,p50_empirical_million_candidates,p95_empirical_million_candidates,"
        "raw_topk_recall,verified_topk_recall,raw_hits,verified_hits,total"
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
    eval_position_tensor = torch.tensor(eval_positions, device=device, dtype=torch.long)

    aggregate: dict[tuple[int, int, int], dict[str, object]] = defaultdict(
        lambda: {"actual": [], "million": [], "raw_hits": 0, "verified_hits": 0, "total": 0}
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
        rank_scores = learned_scores(eval_q_low, eval_k_low, args.rank_dim).float().cpu().numpy()
        print(f"progress,layer_low_rank_ready,{layer_idx}", flush=True)

        for centroids_count in centroid_values:
            centroids = fit_kmeans_centroids(
                train_k_low,
                centroids_count,
                args.kmeans_iters,
                args.seed + layer_idx * 1000,
                args.assign_chunk_size,
            )
            eval_labels = torch.stack(
                [
                    assign_to_centroids(eval_k_low[head_idx], centroids[head_idx], args.assign_chunk_size)
                    for head_idx in range(eval_k_low.shape[0])
                ],
                dim=0,
            )
            actual_centroids = int(centroids.shape[1])
            print(f"progress,centroids_ready,{layer_idx},{actual_centroids}", flush=True)

            for probes in probe_values:
                candidate_records, actual, million = lookup_ivf_candidates(
                    eval_q_low,
                    centroids,
                    eval_labels,
                    eval_positions,
                    probes,
                    args.target_context,
                )
                print(
                    f"progress,lookup_ready,{layer_idx},{actual_centroids},{probes},"
                    f"{sum(actual) / max(len(actual), 1):.1f},"
                    f"{sum(million) / max(len(million), 1):.1f}",
                    flush=True,
                )
                for budget in budgets:
                    raw_hits, verified_hits, total = evaluate_candidate_budget(
                        candidate_records,
                        rank_scores,
                        eval_top_idx,
                        eval_top_valid,
                        budget,
                    )
                    print_ivf_result(
                        layer_idx,
                        eval_seq_len,
                        args.target_context,
                        args.rank_dim,
                        actual_centroids,
                        probes,
                        budget,
                        args.kmeans_iters,
                        args.train_steps,
                        final_loss,
                        actual,
                        million,
                        raw_hits,
                        verified_hits,
                        total,
                    )
                    key_tuple = (actual_centroids, probes, budget)
                    bucket = aggregate[key_tuple]
                    bucket["actual"].extend(actual)
                    bucket["million"].extend(million)
                    bucket["raw_hits"] = int(bucket["raw_hits"]) + raw_hits
                    bucket["verified_hits"] = int(bucket["verified_hits"]) + verified_hits
                    bucket["total"] = int(bucket["total"]) + total

            del centroids, eval_labels
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del train_query_all, train_key, eval_query_all, eval_key, train_query, ranker
        del train_k_low, eval_k_low, eval_q_low, rank_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (centroids_count, probes, budget), bucket in sorted(aggregate.items()):
        print_ivf_result(
            "all",
            eval_seq_len,
            args.target_context,
            args.rank_dim,
            centroids_count,
            probes,
            budget,
            args.kmeans_iters,
            args.train_steps,
            float("nan"),
            bucket["actual"],  # type: ignore[arg-type]
            bucket["million"],  # type: ignore[arg-type]
            int(bucket["raw_hits"]),
            int(bucket["verified_hits"]),
            int(bucket["total"]),
        )


if __name__ == "__main__":
    main()
