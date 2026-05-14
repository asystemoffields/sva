"""High-bit SVA address sweep on real SmolLM2 Q/K activations.

This test respects the model's configured context window, then asks whether
high-resolution SVA addresses still summon the full-attention top keys.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


TEXTS = [
    (
        "Summon-Verify Attention asks a query to summon a small set of candidate memories, "
        "then verifies them exactly. The important question is whether a pretrained model's "
        "own keys and queries already contain enough address structure for that lookup to work."
    ),
    (
        "A tiny language model is useful here because its learned keys and queries are real, "
        "but the model is still inspectable. If the socket test works on a small Llama-style "
        "decoder, the next experiment can move to a larger model without changing the premise."
    ),
    (
        "The river moved quietly through the city while the old library kept its lights on late "
        "into the evening. A courier crossed the bridge, checked the address twice, and found "
        "the right door by matching small details that nobody else seemed to notice."
    ),
    (
        "Sparse attention methods must preserve the important behavior of full attention while "
        "reading far fewer cached tokens. The failure mode is simple: if the retrieval stage misses "
        "one crucial earlier token, the exact verifier can only be precisely wrong."
    ),
    (
        "Modern decoder blocks usually combine grouped-query attention, rotary position embeddings, "
        "RMS normalization, and a gated feedforward network. A socketed replacement should leave those "
        "learned components intact and change only the way candidate keys are selected."
    ),
    (
        "In an engineering notebook, the first result is rarely the final answer. A useful experiment "
        "separates the scientific risk from the systems risk, records what actually happened, and then "
        "makes the next measurement sharper."
    ),
]


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_layers(value: str, n_layers: int) -> list[int]:
    if value == "all":
        return list(range(n_layers))
    layers = comma_ints(value)
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} is outside 0..{n_layers - 1}")
    return layers


def make_long_text(repeats: int) -> str:
    paragraphs = TEXTS * max(repeats, 1)
    return " ".join(paragraphs)


def hamming_ball(bits: int, radius: int) -> int:
    return sum(math.comb(bits, distance) for distance in range(radius + 1))


def expected_candidates(context: int, bits: int, radius: int, tables: int) -> float:
    average_prefix = context / 2.0
    per_table = hamming_ball(bits, radius) / (2**bits)
    probability = 1.0 - (1.0 - per_table) ** tables
    return average_prefix * probability


@lru_cache(maxsize=None)
def neighbor_masks(bits: int, radius: int) -> tuple[int, ...]:
    masks = [0]
    if radius >= 1:
        masks.extend(1 << i for i in range(bits))
    if radius >= 2:
        masks.extend((1 << i) ^ (1 << j) for i in range(bits) for j in range(i + 1, bits))
    if radius > 2:
        raise ValueError("This sweep currently supports probe radii up to 2.")
    return tuple(masks)


def pack_codes(signs: torch.Tensor) -> np.ndarray:
    weights = (1 << torch.arange(signs.shape[-1], device=signs.device, dtype=torch.int64))
    codes = (signs.to(torch.int64) * weights).sum(dim=-1)
    return codes.cpu().numpy().astype(np.int64, copy=False)


def build_buckets(codes: np.ndarray) -> dict[int, int]:
    buckets: dict[int, int] = {}
    for index, code in enumerate(codes):
        key = int(code)
        buckets[key] = buckets.get(key, 0) | (1 << index)
    return buckets


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def sample_query_positions(seq_len: int, topk: int, samples: int, min_query_pos: int) -> np.ndarray:
    start = max(topk - 1, min_query_pos)
    if start >= seq_len:
        start = max(0, seq_len - 1)
    positions = np.linspace(start, seq_len - 1, num=min(samples, seq_len - start), dtype=np.int64)
    return np.unique(positions)


def topk_indices_for_queries(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: np.ndarray,
    topk: int,
    scaling: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_pos = torch.tensor(query_positions, device=query.device, dtype=torch.long)
    sampled_query = query[:, q_pos, :].float()
    scores = torch.einsum("hqd,hkd->hqk", sampled_query, key.float()) * scaling
    key_positions = torch.arange(key.shape[1], device=query.device)
    allowed = key_positions[None, None, :] <= q_pos[None, :, None]
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    actual_topk = min(topk, key.shape[1])
    top_idx = scores.topk(actual_topk, dim=-1).indices.cpu().numpy()
    rank = np.arange(actual_topk, dtype=np.int64)[None, None, :]
    valid = rank < (query_positions[None, :, None] + 1)
    valid = np.broadcast_to(valid, top_idx.shape)
    return top_idx, valid


def evaluate_config(
    q_codes: np.ndarray,
    k_codes: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    query_positions: np.ndarray,
    bits: int,
    tables: int,
    radius: int,
) -> tuple[list[int], int, int]:
    code_mask = (1 << bits) - 1
    masks = neighbor_masks(bits, radius)
    n_heads, n_queries, max_tables = q_codes.shape
    counts: list[int] = []
    hits = 0
    total = 0

    for head_idx in range(n_heads):
        table_buckets = [
            build_buckets(k_codes[head_idx, :, table_idx] & code_mask)
            for table_idx in range(tables)
        ]
        for query_idx in range(n_queries):
            candidates = 0
            for table_idx, buckets in enumerate(table_buckets):
                q_code = int(q_codes[head_idx, query_idx, table_idx] & code_mask)
                for mask in masks:
                    candidates |= buckets.get(q_code ^ mask, 0)
            prefix_mask = (1 << (int(query_positions[query_idx]) + 1)) - 1
            candidates &= prefix_mask
            counts.append(candidates.bit_count())

            for rank_idx, key_idx in enumerate(top_idx[head_idx, query_idx]):
                if top_valid[head_idx, query_idx, rank_idx]:
                    total += 1
                    hits += int((candidates >> int(key_idx)) & 1)
    return counts, hits, total


def print_result(
    layer_idx: int | str,
    seq_len: int,
    bits: int,
    tables: int,
    radius: int,
    counts: list[int],
    hits: int,
    total: int,
    million_context: int,
) -> None:
    recall = hits / total if total else float("nan")
    avg = sum(counts) / max(len(counts), 1)
    print(
        "address_result,"
        f"{layer_idx},{seq_len},{bits},{tables},{radius},"
        f"{avg:.3f},{percentile(counts, 50):.1f},{percentile(counts, 95):.1f},"
        f"{recall:.6f},{hits},{total},"
        f"{expected_candidates(seq_len, bits, radius, tables):.1f},"
        f"{expected_candidates(million_context, bits, radius, tables):.1f}",
        flush=True,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Real-QK high-bit SVA address sweep.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=256)
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--bits", default="14,16,18,20,22,24")
    parser.add_argument("--tables", default="64,128")
    parser.add_argument("--radii", default="1,2")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=32)
    parser.add_argument("--million-context", type=int, default=1_000_000)
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
    text = make_long_text(args.text_repeats)
    batch = tokenizer(
        [text],
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
    print("address_header,layer,seq_len,bits,tables,radius,avg_candidates,p50_candidates,p95_candidates,topk_recall,hits,total,expected_at_seq,expected_at_million")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    model.eval()
    outputs = model(**batch, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

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
        sampled_query = query[:, torch.tensor(query_positions, device=device, dtype=torch.long), :]
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
                    counts, hits, total = evaluate_config(
                        q_codes,
                        k_codes,
                        top_idx,
                        top_valid,
                        query_positions,
                        bits,
                        tables,
                        radius,
                    )
                    print_result(layer_idx, seq_len, bits, tables, radius, counts, hits, total, args.million_context)
                    key_tuple = (bits, tables, radius)
                    bucket = aggregate.setdefault(key_tuple, {"counts": [], "hits": 0, "total": 0})
                    bucket["counts"].extend(counts)
                    bucket["hits"] = int(bucket["hits"]) + hits
                    bucket["total"] = int(bucket["total"]) + total

    for (bits, tables, radius), bucket in sorted(aggregate.items()):
        print_result(
            "all",
            seq_len,
            bits,
            tables,
            radius,
            bucket["counts"],  # type: ignore[arg-type]
            int(bucket["hits"]),
            int(bucket["total"]),
            args.million_context,
        )


if __name__ == "__main__":
    main()
