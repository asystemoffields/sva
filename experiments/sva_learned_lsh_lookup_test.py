"""LSH serving test for the learned SVA compressed ranker.

The learned ranker shows that a compact Q/K score can preserve full-attention
top keys. This test asks whether random-hyperplane lookup over that learned
space can summon useful candidates before exact verification.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sva_learned_ranker_test import (
    LowRankRanker,
    layer_qk,
    make_variant_text,
    set_seed,
    train_ranker,
)
from sva_real_qk_address_sweep import (
    comma_ints,
    expected_candidates,
    make_long_text,
    neighbor_masks,
    pack_codes,
    parse_layers,
    percentile,
    sample_query_positions,
    topk_indices_for_queries,
)


def build_buckets(codes: np.ndarray, bits: int, tables: int) -> list[dict[int, int]]:
    code_mask = (1 << bits) - 1
    buckets: list[dict[int, int]] = []
    for table_idx in range(tables):
        table: dict[int, int] = {}
        for index, code in enumerate(codes[:, table_idx] & code_mask):
            key = int(code)
            table[key] = table.get(key, 0) | (1 << index)
        buckets.append(table)
    return buckets


def build_head_buckets(k_codes: np.ndarray, bits: int, tables: int) -> list[list[dict[int, int]]]:
    return [build_buckets(k_codes[head_idx], bits, tables) for head_idx in range(k_codes.shape[0])]


def bitset_indices(mask: int) -> list[int]:
    indices: list[int] = []
    while mask:
        lowest = mask & -mask
        indices.append(lowest.bit_length() - 1)
        mask ^= lowest
    return indices


def target_prefix_for_position(position: int, seq_len: int, target_context: int) -> int:
    fraction = (position + 1) / max(seq_len, 1)
    return max(1, min(target_context, int(round(fraction * target_context))))


@torch.no_grad()
def low_rank_vectors(
    ranker: LowRankRanker,
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_pos = torch.tensor(query_positions, device=query.device, dtype=torch.long)
    sampled_query = query[:, q_pos, :]
    q_low = torch.einsum("hqd,hdr->hqr", sampled_query, ranker.q_proj)
    k_low = torch.einsum("hkd,hdr->hkr", key, ranker.k_proj)
    scores = torch.einsum("hqr,hkr->hqk", q_low, k_low) / math.sqrt(ranker.rank_dim)
    scale = ranker.logit_scale.exp().clamp(0.01, 100.0)
    scores = scores * scale[:, None, None]
    return q_low, k_low, scores


@torch.no_grad()
def make_lsh_codes(
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    max_tables: int,
    max_bits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = q_low.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    projections = torch.randn(
        q_low.shape[0],
        max_tables,
        max_bits,
        q_low.shape[-1],
        generator=generator,
        device=device,
        dtype=q_low.dtype,
    ) / math.sqrt(q_low.shape[-1])
    q_signs = torch.einsum("hqr,htbr->hqtb", q_low, projections) > 0
    k_signs = torch.einsum("hkr,htbr->hktb", k_low, projections) > 0
    q_codes = pack_codes(q_signs)
    k_codes = pack_codes(k_signs)
    del q_signs, k_signs, projections
    return q_codes, k_codes


def lookup_lsh_candidates(
    q_codes: np.ndarray,
    head_buckets: list[list[dict[int, int]]],
    seq_len: int,
    query_positions: np.ndarray,
    bits: int,
    radius: int,
    target_context: int,
) -> tuple[list[tuple[int, int, int]], list[int], list[float]]:
    code_mask = (1 << bits) - 1
    masks = neighbor_masks(bits, radius)
    n_heads, n_queries, _ = q_codes.shape
    actual_counts: list[int] = []
    million_counts: list[float] = []
    candidate_records: list[tuple[int, int, int]] = []

    for head_idx in range(n_heads):
        table_buckets = head_buckets[head_idx]
        for query_idx in range(n_queries):
            candidates = 0
            for table_idx, buckets in enumerate(table_buckets):
                q_code = int(q_codes[head_idx, query_idx, table_idx] & code_mask)
                for mask in masks:
                    candidates |= buckets.get(q_code ^ mask, 0)

            query_pos = int(query_positions[query_idx])
            prefix_mask = (1 << (query_pos + 1)) - 1
            prefix_candidates = candidates & prefix_mask
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


def print_lsh_result(
    layer_idx: int | str,
    seq_len: int,
    target_context: int,
    rank_dim: int,
    bits: int,
    tables: int,
    radius: int,
    budget: int,
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
    random_avg = expected_candidates(target_context, bits, radius, tables)
    print(
        "learned_lsh_result,"
        f"{layer_idx},{seq_len},{target_context},{rank_dim},{bits},{tables},{radius},{budget},"
        f"{train_steps},{final_loss:.6f},"
        f"{actual_avg:.1f},{percentile(actual_counts, 50):.1f},{percentile(actual_counts, 95):.1f},"
        f"{million_avg:.1f},{percentile(million_counts, 50):.1f},{percentile(million_counts, 95):.1f},"
        f"{raw_recall:.6f},{verified_recall:.6f},{raw_hits},{verified_hits},{total},{random_avg:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve learned SVA ranker with LSH lookup.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="reverse")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--bits", default="18,20,22,24")
    parser.add_argument("--tables", default="64,128")
    parser.add_argument("--radii", default="1,2")
    parser.add_argument("--budgets", default="256,512")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--eval-query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-context", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))
    bits_values = comma_ints(args.bits)
    table_values = comma_ints(args.tables)
    radius_values = comma_ints(args.radii)
    budgets = comma_ints(args.budgets)
    max_bits = max(bits_values)
    max_tables = max(table_values)

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
    print(f"bits,{';'.join(str(value) for value in bits_values)}")
    print(f"tables,{';'.join(str(value) for value in table_values)}")
    print(f"radii,{';'.join(str(value) for value in radius_values)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(
        "learned_lsh_header,"
        "layer,seq_len,target_context,rank_dim,bits,tables,radius,budget,train_steps,final_loss,"
        "avg_lsh_candidates,p50_lsh_candidates,p95_lsh_candidates,"
        "avg_empirical_million_candidates,p50_empirical_million_candidates,p95_empirical_million_candidates,"
        "raw_topk_recall,verified_topk_recall,raw_hits,verified_hits,total,random_expected_million_candidates"
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
        eval_query = eval_query_all[:, eval_position_tensor, :].contiguous()

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
        q_low, k_low, rank_scores_t = low_rank_vectors(ranker, eval_query_all, eval_key, eval_positions)
        q_codes, k_codes = make_lsh_codes(
            q_low,
            k_low,
            max_tables,
            max_bits,
            args.seed + 100_000 + layer_idx * 1000 + args.rank_dim,
        )
        rank_scores = rank_scores_t.float().cpu().numpy()
        del q_low, k_low, rank_scores_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"progress,layer_lsh_codes_ready,{layer_idx}", flush=True)

        for bits in bits_values:
            for tables in table_values:
                head_buckets = build_head_buckets(k_codes, bits, tables)
                print(f"progress,buckets_ready,{layer_idx},{bits},{tables}", flush=True)
                for radius in radius_values:
                    candidate_records, actual, million = lookup_lsh_candidates(
                        q_codes,
                        head_buckets,
                        eval_seq_len,
                        eval_positions,
                        bits,
                        radius,
                        args.target_context,
                    )
                    print(
                        f"progress,lookup_ready,{layer_idx},{bits},{tables},{radius},"
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
                        print_lsh_result(
                            layer_idx,
                            eval_seq_len,
                            args.target_context,
                            args.rank_dim,
                            bits,
                            tables,
                            radius,
                            budget,
                            args.train_steps,
                            final_loss,
                            actual,
                            million,
                            raw_hits,
                            verified_hits,
                            total,
                        )
                        key_tuple = (bits, tables, radius, budget)
                        bucket = aggregate[key_tuple]
                        bucket["actual"].extend(actual)
                        bucket["million"].extend(million)
                        bucket["raw_hits"] = int(bucket["raw_hits"]) + raw_hits
                        bucket["verified_hits"] = int(bucket["verified_hits"]) + verified_hits
                        bucket["total"] = int(bucket["total"]) + total

        del train_query_all, train_key, eval_query_all, eval_key, train_query, eval_query, ranker
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (bits, tables, radius, budget), bucket in sorted(aggregate.items()):
        print_lsh_result(
            "all",
            eval_seq_len,
            args.target_context,
            args.rank_dim,
            bits,
            tables,
            radius,
            budget,
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
