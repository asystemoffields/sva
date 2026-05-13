"""Supervised query-cell router for learned SVA ranker serving."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sva_learned_ivf_lookup_test import (
    evaluate_candidate_budget,
    fit_kmeans_centroids,
    learned_scores,
    project_keys,
    project_queries,
)
from sva_learned_multiwrite_ivf_lookup_test import (
    assign_to_multiple_centroids,
    build_multiwrite_buckets,
)
from sva_learned_ranker_test import (
    LowRankRanker,
    layer_qk,
    make_variant_text,
    set_seed,
    train_ranker,
)
from sva_learned_lsh_lookup_test import target_prefix_for_position
from sva_real_qk_address_sweep import (
    comma_ints,
    make_long_text,
    parse_layers,
    percentile,
    sample_query_positions,
    topk_indices_for_queries,
)


def build_supervised_pairs(
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    query_labels: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n_heads, n_queries, topk = top_idx.shape
    pairs_by_head: list[tuple[torch.Tensor, torch.Tensor]] = []

    for head_idx in range(n_heads):
        labels_np = query_labels[head_idx].cpu().numpy()
        pairs: set[tuple[int, int]] = set()
        for query_idx in range(n_queries):
            for rank_idx in range(topk):
                if not top_valid[head_idx, query_idx, rank_idx]:
                    continue
                key_idx = int(top_idx[head_idx, query_idx, rank_idx])
                for cell_idx in labels_np[query_idx]:
                    pairs.add((key_idx, int(cell_idx)))

        if pairs:
            key_indices, cell_indices = zip(*sorted(pairs))
            pairs_by_head.append(
                (
                    torch.tensor(key_indices, dtype=torch.long),
                    torch.tensor(cell_indices, dtype=torch.long),
                )
            )
        else:
            pairs_by_head.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))

    return pairs_by_head


def train_key_cell_writers(
    k_low: torch.Tensor,
    pairs_by_head: list[tuple[torch.Tensor, torch.Tensor]],
    n_cells: int,
    steps: int,
    batch_pairs: int,
    negatives: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> torch.Tensor:
    n_heads, _, rank_dim = k_low.shape
    writers: list[torch.Tensor] = []
    scale = 1.0 / math.sqrt(rank_dim)

    for head_idx in range(n_heads):
        generator = torch.Generator(device=k_low.device)
        generator.manual_seed(seed + head_idx * 1009 + n_cells)
        writer = torch.nn.Parameter(
            torch.randn(n_cells, rank_dim, device=k_low.device, generator=generator) * scale
        )
        optimizer = torch.optim.AdamW([writer], lr=lr, weight_decay=weight_decay)
        key_indices, cell_indices = pairs_by_head[head_idx]
        key_indices = key_indices.to(k_low.device)
        cell_indices = cell_indices.to(k_low.device)

        if key_indices.numel() == 0:
            writers.append(writer.detach().to(k_low.dtype))
            continue

        for _ in range(steps):
            pair_ids = torch.randint(key_indices.numel(), (batch_pairs,), device=k_low.device, generator=generator)
            keys = k_low[head_idx, key_indices[pair_ids]].float()
            pos_cells = cell_indices[pair_ids]
            pos_scores = (keys * writer[pos_cells]).sum(dim=-1) * scale

            if n_cells > 1:
                neg_offsets = torch.randint(
                    1,
                    n_cells,
                    (batch_pairs, negatives),
                    device=k_low.device,
                    generator=generator,
                )
                neg_cells = (pos_cells[:, None] + neg_offsets) % n_cells
            else:
                neg_cells = torch.zeros(batch_pairs, negatives, device=k_low.device, dtype=torch.long)
            neg_scores = torch.einsum("br,bnr->bn", keys, writer[neg_cells]) * scale

            loss = F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        writers.append(writer.detach().to(k_low.dtype))

    return torch.stack(writers, dim=0)


@torch.no_grad()
def assign_keys_to_writer_cells(
    k_low: torch.Tensor,
    writer_cells: torch.Tensor,
    writes: int,
    chunk_size: int,
) -> torch.Tensor:
    n_heads, seq_len, _ = k_low.shape
    n_cells = writer_cells.shape[1]
    actual_writes = min(writes, n_cells)
    labels: list[torch.Tensor] = []

    for head_idx in range(n_heads):
        head_labels: list[torch.Tensor] = []
        writers = writer_cells[head_idx].float()
        for start in range(0, seq_len, chunk_size):
            chunk = k_low[head_idx, start : start + chunk_size].float()
            scores = chunk @ writers.T
            head_labels.append(scores.topk(actual_writes, dim=-1).indices)
        labels.append(torch.cat(head_labels, dim=0))

    return torch.stack(labels, dim=0)


@torch.no_grad()
def lookup_query_cell_candidates(
    q_low: torch.Tensor,
    query_centroids: torch.Tensor,
    eval_key_labels: torch.Tensor,
    query_positions: np.ndarray,
    probes: int,
    target_context: int,
    chunk_size: int,
) -> tuple[list[tuple[int, int, int]], list[int], list[float]]:
    n_heads, n_queries, _ = q_low.shape
    seq_len = eval_key_labels.shape[1]
    n_cells = query_centroids.shape[1]
    actual_probes = min(probes, n_cells)
    candidate_records: list[tuple[int, int, int]] = []
    actual_counts: list[int] = []
    million_counts: list[float] = []

    bucket_sets = [
        build_multiwrite_buckets(eval_key_labels[head_idx].cpu().numpy(), n_cells)
        for head_idx in range(n_heads)
    ]

    query_labels = torch.stack(
        [
            assign_to_multiple_centroids(q_low[head_idx], query_centroids[head_idx], actual_probes, chunk_size)
            for head_idx in range(n_heads)
        ],
        dim=0,
    ).cpu().numpy()

    for head_idx in range(n_heads):
        buckets = bucket_sets[head_idx]
        for query_idx in range(n_queries):
            candidates = 0
            for cell_idx in query_labels[head_idx, query_idx]:
                candidates |= buckets[int(cell_idx)]

            query_pos = int(query_positions[query_idx])
            prefix_candidates = candidates & ((1 << (query_pos + 1)) - 1)
            actual_count = prefix_candidates.bit_count()
            actual_counts.append(actual_count)

            prefix_density = actual_count / max(query_pos + 1, 1)
            target_prefix = target_prefix_for_position(query_pos, seq_len, target_context)
            million_counts.append(prefix_density * target_prefix)
            candidate_records.append((head_idx, query_idx, prefix_candidates))

    return candidate_records, actual_counts, million_counts


def print_router_result(
    layer_idx: int | str,
    seq_len: int,
    target_context: int,
    rank_dim: int,
    cells: int,
    query_writes: int,
    key_writes: int,
    probes: int,
    budget: int,
    writer_steps: int,
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
        "supervised_query_router_result,"
        f"{layer_idx},{seq_len},{target_context},{rank_dim},{cells},{query_writes},{key_writes},{probes},{budget},"
        f"{writer_steps},{train_steps},{final_loss:.6f},"
        f"{actual_avg:.1f},{percentile(actual_counts, 50):.1f},{percentile(actual_counts, 95):.1f},"
        f"{million_avg:.1f},{percentile(million_counts, 50):.1f},{percentile(million_counts, 95):.1f},"
        f"{raw_recall:.6f},{verified_recall:.6f},{raw_hits},{verified_hits},{total}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve learned SVA ranker with supervised query-cell routing.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="reverse")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--cells", default="256,512")
    parser.add_argument("--query-writes", type=int, default=1)
    parser.add_argument("--key-writes", default="4,8,16")
    parser.add_argument("--probes", default="1,2,4")
    parser.add_argument("--budgets", default="256,512")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=512)
    parser.add_argument("--eval-query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--ranker-steps", type=int, default=160)
    parser.add_argument("--ranker-batch-queries", type=int, default=16)
    parser.add_argument("--ranker-lr", type=float, default=3e-3)
    parser.add_argument("--ranker-weight-decay", type=float, default=1e-4)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--writer-steps", type=int, default=160)
    parser.add_argument("--writer-batch-pairs", type=int, default=512)
    parser.add_argument("--writer-negatives", type=int, default=16)
    parser.add_argument("--writer-lr", type=float, default=2e-3)
    parser.add_argument("--writer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--target-context", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))
    cell_values = comma_ints(args.cells)
    key_write_values = comma_ints(args.key_writes)
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
    print(f"cells,{';'.join(str(value) for value in cell_values)}")
    print(f"query_writes,{args.query_writes}")
    print(f"key_writes,{';'.join(str(value) for value in key_write_values)}")
    print(f"probes,{';'.join(str(value) for value in probe_values)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(
        "supervised_query_router_header,"
        "layer,seq_len,target_context,rank_dim,cells,query_writes,key_writes,probes,budget,"
        "writer_steps,ranker_steps,ranker_final_loss,"
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

    aggregate: dict[tuple[int, int, int, int], dict[str, object]] = defaultdict(
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
            args.ranker_steps,
            args.ranker_batch_queries,
            args.ranker_lr,
            args.ranker_weight_decay,
            args.seed + layer_idx * 1000 + args.rank_dim,
        )
        print(f"progress,layer_ranker_trained,{layer_idx},{final_loss:.6f}", flush=True)

        train_k_low = project_keys(ranker, train_key)
        train_q_low = project_queries(ranker, train_query_all, train_positions)
        eval_k_low = project_keys(ranker, eval_key)
        eval_q_low = project_queries(ranker, eval_query_all, eval_positions)
        rank_scores = learned_scores(eval_q_low, eval_k_low, args.rank_dim).float().cpu().numpy()
        print(f"progress,layer_low_rank_ready,{layer_idx}", flush=True)

        seen_actual_cells: set[int] = set()
        for requested_cells in cell_values:
            expected_cells = min(requested_cells, train_q_low.shape[1])
            if expected_cells in seen_actual_cells:
                print(f"progress,query_cells_skipped_duplicate,{layer_idx},{requested_cells},{expected_cells}", flush=True)
                continue
            seen_actual_cells.add(expected_cells)
            query_centroids = fit_kmeans_centroids(
                train_q_low,
                requested_cells,
                args.kmeans_iters,
                args.seed + layer_idx * 1000 + 17,
                args.assign_chunk_size,
            )
            actual_cells = int(query_centroids.shape[1])
            train_query_labels = torch.stack(
                [
                    assign_to_multiple_centroids(
                        train_q_low[head_idx],
                        query_centroids[head_idx],
                        args.query_writes,
                        args.assign_chunk_size,
                    )
                    for head_idx in range(train_q_low.shape[0])
                ],
                dim=0,
            )
            pairs_by_head = build_supervised_pairs(train_top_idx, train_top_valid, train_query_labels)
            pair_counts = [int(keys.numel()) for keys, _ in pairs_by_head]
            print(
                f"progress,query_cells_ready,{layer_idx},{actual_cells},"
                f"{sum(pair_counts)},{min(pair_counts) if pair_counts else 0},{max(pair_counts) if pair_counts else 0}",
                flush=True,
            )

            writer_cells = train_key_cell_writers(
                train_k_low,
                pairs_by_head,
                actual_cells,
                args.writer_steps,
                args.writer_batch_pairs,
                args.writer_negatives,
                args.writer_lr,
                args.writer_weight_decay,
                args.seed + layer_idx * 1000 + actual_cells,
            )
            print(f"progress,key_writers_trained,{layer_idx},{actual_cells}", flush=True)

            for key_writes in key_write_values:
                eval_key_labels = assign_keys_to_writer_cells(
                    eval_k_low,
                    writer_cells,
                    key_writes,
                    args.assign_chunk_size,
                )
                print(f"progress,key_writes_ready,{layer_idx},{actual_cells},{key_writes}", flush=True)

                for probes in probe_values:
                    candidate_records, actual, million = lookup_query_cell_candidates(
                        eval_q_low,
                        query_centroids,
                        eval_key_labels,
                        eval_positions,
                        probes,
                        args.target_context,
                        args.assign_chunk_size,
                    )
                    print(
                        f"progress,lookup_ready,{layer_idx},{actual_cells},{key_writes},{probes},"
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
                        print_router_result(
                            layer_idx,
                            eval_seq_len,
                            args.target_context,
                            args.rank_dim,
                            actual_cells,
                            args.query_writes,
                            key_writes,
                            probes,
                            budget,
                            args.writer_steps,
                            args.ranker_steps,
                            final_loss,
                            actual,
                            million,
                            raw_hits,
                            verified_hits,
                            total,
                        )
                        key_tuple = (actual_cells, key_writes, probes, budget)
                        bucket = aggregate[key_tuple]
                        bucket["actual"].extend(actual)
                        bucket["million"].extend(million)
                        bucket["raw_hits"] = int(bucket["raw_hits"]) + raw_hits
                        bucket["verified_hits"] = int(bucket["verified_hits"]) + verified_hits
                        bucket["total"] = int(bucket["total"]) + total

                del eval_key_labels
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del query_centroids, train_query_labels, writer_cells
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del train_query_all, train_key, eval_query_all, eval_key, train_query, ranker
        del train_k_low, train_q_low, eval_k_low, eval_q_low, rank_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (cells, key_writes, probes, budget), bucket in sorted(aggregate.items()):
        print_router_result(
            "all",
            eval_seq_len,
            args.target_context,
            args.rank_dim,
            cells,
            args.query_writes,
            key_writes,
            probes,
            budget,
            args.writer_steps,
            args.ranker_steps,
            float("nan"),
            bucket["actual"],  # type: ignore[arg-type]
            bucket["million"],  # type: ignore[arg-type]
            int(bucket["raw_hits"]),
            int(bucket["verified_hits"]),
            int(bucket["total"]),
        )


if __name__ == "__main__":
    main()
