"""Million-token SVA address-pressure simulation from real SmolLM2 Q/K.

The model forward pass stays inside SmolLM2's configured context window. The
million-token estimate is produced by scaling the empirical address hit rate
from real keys, so this isolates lookup selectivity without changing RoPE.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

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


def target_prefix_for_position(position: int, seq_len: int, target_context: int) -> int:
    fraction = (position + 1) / max(seq_len, 1)
    return max(1, min(target_context, int(round(fraction * target_context))))


def evaluate_million_config(
    q_codes: np.ndarray,
    k_codes: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    query_positions: np.ndarray,
    bits: int,
    tables: int,
    radius: int,
    target_context: int,
) -> tuple[list[int], list[float], int, int]:
    code_mask = (1 << bits) - 1
    masks = neighbor_masks(bits, radius)
    n_heads, n_queries, _ = q_codes.shape
    seq_len = k_codes.shape[1]
    actual_counts: list[int] = []
    million_counts: list[float] = []
    hits = 0
    total = 0

    for head_idx in range(n_heads):
        table_buckets = build_buckets(k_codes[head_idx], bits, tables)
        for query_idx in range(n_queries):
            candidates = 0
            for table_idx, buckets in enumerate(table_buckets):
                q_code = int(q_codes[head_idx, query_idx, table_idx] & code_mask)
                for mask in masks:
                    candidates |= buckets.get(q_code ^ mask, 0)

            query_pos = int(query_positions[query_idx])
            prefix_mask = (1 << (query_pos + 1)) - 1
            prefix_candidates = candidates & prefix_mask
            actual_counts.append(prefix_candidates.bit_count())

            empirical_density = candidates.bit_count() / max(seq_len, 1)
            target_prefix = target_prefix_for_position(query_pos, seq_len, target_context)
            million_counts.append(empirical_density * target_prefix)

            for rank_idx, key_idx in enumerate(top_idx[head_idx, query_idx]):
                if top_valid[head_idx, query_idx, rank_idx]:
                    total += 1
                    hits += int((prefix_candidates >> int(key_idx)) & 1)

    return actual_counts, million_counts, hits, total


def print_result(
    layer_idx: int | str,
    seq_len: int,
    target_context: int,
    bits: int,
    tables: int,
    radius: int,
    actual_counts: list[int],
    million_counts: list[float],
    hits: int,
    total: int,
) -> None:
    recall = hits / total if total else float("nan")
    actual_avg = sum(actual_counts) / max(len(actual_counts), 1)
    million_avg = sum(million_counts) / max(len(million_counts), 1)
    random_avg = expected_candidates(target_context, bits, radius, tables)
    print(
        "million_result,"
        f"{layer_idx},{seq_len},{target_context},{bits},{tables},{radius},"
        f"{actual_avg:.3f},{percentile(actual_counts, 50):.1f},{percentile(actual_counts, 95):.1f},"
        f"{million_avg:.1f},{percentile(million_counts, 50):.1f},{percentile(million_counts, 95):.1f},"
        f"{recall:.6f},{hits},{total},{random_avg:.1f}",
        flush=True,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Million-token SVA pressure simulation.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--bits", default="20,22,24,26")
    parser.add_argument("--tables", default="64,128,256")
    parser.add_argument("--radii", default="1,2")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--target-context", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)

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

    bits_values = comma_ints(args.bits)
    table_values = comma_ints(args.tables)
    radius_values = comma_ints(args.radii)
    max_bits = max(bits_values)
    max_tables = max(table_values)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        [make_long_text(args.text_repeats)],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=effective_max_length,
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    seq_len = int(batch["input_ids"].shape[1])
    query_positions = sample_query_positions(seq_len, args.topk, args.query_samples, args.min_query_pos)

    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"model_max_position_embeddings,{model_window}")
    print(f"requested_max_length,{requested}")
    print(f"effective_max_length,{effective_max_length}")
    print(f"seq_len,{seq_len}")
    print(f"target_context,{args.target_context}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"layers,{';'.join(str(layer) for layer in layers)}")
    print(f"bits,{';'.join(str(value) for value in bits_values)}")
    print(f"tables,{';'.join(str(value) for value in table_values)}")
    print(f"radii,{';'.join(str(value) for value in radius_values)}")
    print(f"topk,{args.topk}")
    print(f"query_samples,{len(query_positions)}")
    print(f"first_query_pos,{int(query_positions[0])}")
    print(f"last_query_pos,{int(query_positions[-1])}")
    print(
        "million_header,"
        "layer,seq_len,target_context,bits,tables,radius,"
        "avg_actual_candidates,p50_actual_candidates,p95_actual_candidates,"
        "avg_empirical_million_candidates,p50_empirical_million_candidates,p95_empirical_million_candidates,"
        "topk_recall,hits,total,random_expected_million_candidates"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    model.eval()
    outputs = model(**batch, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    query_position_tensor = torch.tensor(query_positions, device=device, dtype=torch.long)

    aggregate: dict[tuple[int, int, int], dict[str, object]] = {}
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        # HF Llama hidden_states are layer-boundary states; attention sees the input-layernormed state.
        hidden = layer.input_layernorm(hidden_states[layer_idx])
        hidden_shape = (hidden.shape[0], hidden.shape[1], -1, layer.self_attn.head_dim)
        query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        cos, sin = model.model.rotary_emb(hidden, position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, layer.self_attn.num_key_value_groups)
        query = query[0]
        key = key[0]

        top_idx, top_valid = topk_indices_for_queries(
            query,
            key,
            query_positions,
            args.topk,
            float(layer.self_attn.scaling),
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + 10_000 * layer_idx)
        projections = torch.randn(
            query.shape[0],
            max_tables,
            max_bits,
            query.shape[-1],
            generator=generator,
            device=device,
            dtype=query.dtype,
        ) / math.sqrt(query.shape[-1])
        sampled_query = query[:, query_position_tensor, :]
        q_signs = torch.einsum("hqd,hrmd->hqrm", sampled_query, projections) > 0
        k_signs = torch.einsum("hkd,hrmd->hkrm", key, projections) > 0
        q_codes = pack_codes(q_signs)
        k_codes = pack_codes(k_signs)
        del q_signs, k_signs, projections
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for bits in bits_values:
            for tables in table_values:
                for radius in radius_values:
                    actual_counts, million_counts, hits, total = evaluate_million_config(
                        q_codes,
                        k_codes,
                        top_idx,
                        top_valid,
                        query_positions,
                        bits,
                        tables,
                        radius,
                        args.target_context,
                    )
                    print_result(
                        layer_idx,
                        seq_len,
                        args.target_context,
                        bits,
                        tables,
                        radius,
                        actual_counts,
                        million_counts,
                        hits,
                        total,
                    )
                    key_tuple = (bits, tables, radius)
                    bucket = aggregate.setdefault(
                        key_tuple,
                        {"actual": [], "million": [], "hits": 0, "total": 0},
                    )
                    bucket["actual"].extend(actual_counts)
                    bucket["million"].extend(million_counts)
                    bucket["hits"] = int(bucket["hits"]) + hits
                    bucket["total"] = int(bucket["total"]) + total

    for (bits, tables, radius), bucket in sorted(aggregate.items()):
        print_result(
            "all",
            seq_len,
            args.target_context,
            bits,
            tables,
            radius,
            bucket["actual"],  # type: ignore[arg-type]
            bucket["million"],  # type: ignore[arg-type]
            int(bucket["hits"]),
            int(bucket["total"]),
        )


if __name__ == "__main__":
    main()
