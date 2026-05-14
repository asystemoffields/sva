"""Cached-key decode benchmark for deployment-shaped SVA lookup.

This benchmark freezes artifacts from calibration text, computes held-out Q/K/V
states, precomputes the key-side SVA catalog once, and then measures per-query
decode lookup against full attention. It isolates the serving shape from the
reference socket's current habit of rebuilding product codes inside every
forward pass.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from sva_full_deployment_benchmark import (
    CALIBRATION_DOCS,
    EVAL_DOCS,
    calibration_stream,
    comma_ints,
    load_documents,
    repeated_document,
)
from sva_pq_lookup_test import encode_product_keys, product_quantized_scores
from sva_pretrained_socket_test import (
    build_artifacts_for_hidden_states,
    build_progressive_three_stage_artifacts,
    encode_batch,
    format_layer_list,
    parse_layer_list,
)
from sva_real_qk_address_sweep import sample_query_positions, topk_indices_for_queries


@dataclass
class QualityBucket:
    candidate_hits: int = 0
    verified_hits: int = 0
    total: int = 0
    candidate_count_sum: float = 0.0
    verified_count_sum: float = 0.0
    count_items: int = 0


@dataclass
class TimingBucket:
    full_ms: list[float] = field(default_factory=list)
    sva_ms: list[float] = field(default_factory=list)
    catalog_ms: list[float] = field(default_factory=list)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark(fn, device: torch.device, warmup: int, repeats: int) -> tuple[object, float, float, float]:
    result = None
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            result = fn()
        synchronize(device)

        times: list[float] = []
        for _ in range(max(repeats, 1)):
            synchronize(device)
            start = time.perf_counter()
            result = fn()
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000.0)
    return result, average(times), percentile(times, 50), percentile(times, 95)


@torch.no_grad()
def layer_qkv_from_hidden(
    model,
    hidden_states: tuple[torch.Tensor, ...],
    layer_idx: int,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    layer = model.model.layers[layer_idx]
    hidden = layer.input_layernorm(hidden_states[layer_idx])
    hidden_shape = (hidden.shape[0], hidden.shape[1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = repeat_kv(key, layer.self_attn.num_key_value_groups)
    value = repeat_kv(value, layer.self_attn.num_key_value_groups)
    return query[0].float(), key[0].float(), value[0].float(), float(layer.self_attn.scaling)


@torch.no_grad()
def project_cached_catalog(
    query: torch.Tensor,
    key: torch.Tensor,
    artifact,
    rank_dim: int,
    assign_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_proj = artifact.q_proj.to(device=query.device, dtype=torch.float32)
    k_proj = artifact.k_proj.to(device=query.device, dtype=torch.float32)
    scale = artifact.logit_scale.to(device=query.device, dtype=torch.float32).exp().clamp(0.01, 100.0)
    q_low = torch.einsum("htd,hdr->htr", query.float(), q_proj) * scale[:, None, None]
    k_low = torch.einsum("hsd,hdr->hsr", key.float(), k_proj)
    codebooks = artifact.coarse_codebooks.to(device=query.device, dtype=torch.float32)
    codes = encode_product_keys(k_low, codebooks, assign_chunk_size)
    if q_low.shape[-1] != rank_dim or k_low.shape[-1] != rank_dim:
        raise ValueError("Projected catalog rank does not match --rank-dim.")
    return q_low, k_low, codebooks, codes


@torch.no_grad()
def full_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    selected_query = query[:, positions, :]
    scores = torch.einsum("hqd,hkd->hqk", selected_query, key) * scaling
    key_positions = torch.arange(key.shape[1], device=query.device)
    allowed = key_positions[None, None, :] <= positions[None, :, None]
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
    return torch.einsum("hqk,hkd->hqd", weights, value)


@torch.no_grad()
def sva_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_low_all: torch.Tensor,
    k_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    positions: torch.Tensor,
    shortlist: int,
    budget: int,
    rank_dim: int,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_query = query[:, positions, :]
    q_low = q_low_all[:, positions, :]
    seq_len = key.shape[1]
    actual_shortlist = min(shortlist, seq_len)
    actual_budget = min(budget, actual_shortlist)

    coarse_scores = product_quantized_scores(q_low, codebooks, codes, rank_dim)
    key_positions = torch.arange(seq_len, device=query.device)
    allowed = key_positions[None, None, :] <= positions[None, :, None]
    coarse_scores = coarse_scores.masked_fill(~allowed, torch.finfo(coarse_scores.dtype).min)
    coarse_idx = coarse_scores.topk(actual_shortlist, dim=-1).indices

    source_low = k_low[:, None, :, :].expand(k_low.shape[0], positions.numel(), seq_len, rank_dim)
    shortlist_low = source_low.gather(
        dim=2,
        index=coarse_idx[..., None].expand(k_low.shape[0], positions.numel(), actual_shortlist, rank_dim),
    )
    rank_scores = (shortlist_low * q_low[:, :, None, :]).sum(dim=-1) / math.sqrt(rank_dim)
    rank_keep = rank_scores.topk(actual_budget, dim=-1).indices
    final_idx = coarse_idx.gather(dim=-1, index=rank_keep)

    selected_keys = key[:, None, :, :].expand(key.shape[0], positions.numel(), seq_len, key.shape[-1]).gather(
        dim=2,
        index=final_idx[..., None].expand(key.shape[0], positions.numel(), actual_budget, key.shape[-1]),
    )
    selected_values = value[:, None, :, :].expand(value.shape[0], positions.numel(), seq_len, value.shape[-1]).gather(
        dim=2,
        index=final_idx[..., None].expand(value.shape[0], positions.numel(), actual_budget, value.shape[-1]),
    )
    selected_scores = (selected_keys * selected_query[:, :, None, :]).sum(dim=-1) * scaling
    weights = torch.softmax(selected_scores, dim=-1, dtype=torch.float32).to(value.dtype)
    output = (weights[..., None] * selected_values).sum(dim=-2)
    return output, coarse_idx, final_idx


def update_quality(
    bucket: QualityBucket,
    coarse_idx: torch.Tensor,
    final_idx: torch.Tensor,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
) -> None:
    heads, queries, shortlist = coarse_idx.shape
    budget = final_idx.shape[-1]
    seq_len = int(max(coarse_idx.max().item(), final_idx.max().item(), np.max(top_idx))) + 1
    candidate_mask = torch.zeros(heads, queries, seq_len, device=coarse_idx.device, dtype=torch.bool)
    candidate_mask.scatter_(dim=-1, index=coarse_idx.clamp_max(seq_len - 1), value=True)
    verified_mask = torch.zeros(heads, queries, seq_len, device=final_idx.device, dtype=torch.bool)
    verified_mask.scatter_(dim=-1, index=final_idx.clamp_max(seq_len - 1), value=True)
    top_idx_t = torch.tensor(top_idx, device=coarse_idx.device, dtype=torch.long).clamp_max(seq_len - 1)
    top_valid_t = torch.tensor(top_valid, device=coarse_idx.device, dtype=torch.bool)
    candidate_hits = candidate_mask.gather(dim=-1, index=top_idx_t) & top_valid_t
    verified_hits = verified_mask.gather(dim=-1, index=top_idx_t) & top_valid_t
    bucket.candidate_hits += int(candidate_hits.sum().item())
    bucket.verified_hits += int(verified_hits.sum().item())
    bucket.total += int(top_valid_t.sum().item())
    bucket.candidate_count_sum += float(shortlist * heads * queries)
    bucket.verified_count_sum += float(budget * heads * queries)
    bucket.count_items += int(heads * queries)


def print_quality_summary(key: tuple[int, int], bucket: QualityBucket) -> None:
    shortlist, budget = key
    candidate_recall = bucket.candidate_hits / max(bucket.total, 1)
    verified_recall = bucket.verified_hits / max(bucket.total, 1)
    avg_candidate = bucket.candidate_count_sum / max(bucket.count_items, 1)
    avg_verified = bucket.verified_count_sum / max(bucket.count_items, 1)
    print(
        "cached_decode_quality_summary,"
        f"{shortlist},{budget},{bucket.total},"
        f"{candidate_recall:.6f},{verified_recall:.6f},"
        f"{avg_candidate:.3f},{avg_verified:.3f}",
        flush=True,
    )


def print_timing_summary(key: tuple[int, int, int], bucket: TimingBucket) -> None:
    shortlist, budget, query_count = key
    print(
        "cached_decode_timing_summary,"
        f"{shortlist},{budget},{query_count},"
        f"{average(bucket.catalog_ms):.3f},{percentile(bucket.catalog_ms, 50):.3f},{percentile(bucket.catalog_ms, 95):.3f},"
        f"{average(bucket.full_ms):.3f},{percentile(bucket.full_ms, 50):.3f},{percentile(bucket.full_ms, 95):.3f},"
        f"{average(bucket.sva_ms):.3f},{percentile(bucket.sva_ms, 50):.3f},{percentile(bucket.sva_ms, 95):.3f},"
        f"{average(bucket.full_ms) / average(bucket.sva_ms):.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached-key deployment decode benchmark for SVA.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--calibration-doc-limit", type=int, default=None)
    parser.add_argument("--eval-doc-limit", type=int, default=4)
    parser.add_argument("--calibration-repeats", type=int, default=320)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--allow-beyond-model-context", action="store_true")
    parser.add_argument("--socket-layers", default="")
    parser.add_argument("--route-source", choices=["qk"], default="qk")
    parser.add_argument("--artifact-training", choices=["teacher", "progressive"], default="teacher")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--coarse-subspaces", type=int, default=4)
    parser.add_argument("--coarse-codewords", type=int, default=64)
    parser.add_argument("--coarse-shortlists", default="512,1024,2048")
    parser.add_argument("--budgets", default="128,256,512")
    parser.add_argument("--coarse-label-topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=256)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--ranker-train-steps", type=int, default=240)
    parser.add_argument("--coarse-hard-steps", type=int, default=120)
    parser.add_argument("--coarse-hard-pool", type=int, default=512)
    parser.add_argument("--coarse-hard-negatives", type=int, default=64)
    parser.add_argument("--coarse-hard-margin", type=float, default=1.0)
    parser.add_argument("--coarse-hard-lr-scale", type=float, default=0.5)
    parser.add_argument("--weighted-boost", type=float, default=4.0)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--ranker-lr", type=float, default=0.003)
    parser.add_argument("--ranker-weight-decay", type=float, default=0.0001)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--quality-query-samples", type=int, default=128)
    parser.add_argument("--timing-query-counts", default="1,4,16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()
    args.mode = "three_stage"
    args.tables = 16
    args.bits = 10
    args.budget = 512
    args.probe_radius = 1
    args.seed = 17
    args.prefilter_dim = 0
    args.prefilter_budget = 0
    args.coarse_shortlist = 1024
    args.diagnose_topk = 0
    args.head_report_limit = 0

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

    shortlists = comma_ints(args.coarse_shortlists)
    budgets = comma_ints(args.budgets)
    timing_query_counts = comma_ints(args.timing_query_counts) if args.timing_query_counts.strip() else []

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    model_context = getattr(model.config, "max_position_embeddings", None)
    if model_context is not None and args.context_length > int(model_context) and not args.allow_beyond_model_context:
        raise ValueError(
            f"Requested context {args.context_length} exceeds model max_position_embeddings {model_context}."
        )

    calibration_docs = load_documents(
        args.calibration_file,
        CALIBRATION_DOCS,
        args.calibration_doc_limit,
        "calibration",
    )
    eval_docs = load_documents(args.eval_file, EVAL_DOCS, args.eval_doc_limit, "eval")
    calibration_batch = encode_batch(
        tokenizer,
        [calibration_stream(calibration_docs, args.calibration_repeats)],
        args.context_length,
        device,
    )
    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    train_layers = socket_layers if socket_layers is not None else list(range(len(model.model.layers)))

    print("cached_decode_benchmark_start")
    print(f"model_id,{args.model_id}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"context_length,{args.context_length}")
    print(f"calibration_docs,{len(calibration_docs)}")
    print(f"eval_docs,{len(eval_docs)}")
    print(f"socket_layers,{format_layer_list(socket_layers)}")
    print(f"artifact_training,{args.artifact_training}")
    print(f"shortlists,{args.coarse_shortlists}")
    print(f"budgets,{args.budgets}")
    print(
        "cached_decode_quality_header,"
        "shortlist,budget,top_items,candidate_top16_recall,verified_top16_recall,avg_candidate,avg_verified"
    )
    print(
        "cached_decode_timing_header,"
        "shortlist,budget,query_count,catalog_build_ms_avg,catalog_build_ms_p50,catalog_build_ms_p95,"
        "full_decode_ms_avg,full_decode_ms_p50,full_decode_ms_p95,"
        "sva_decode_ms_avg,sva_decode_ms_p50,sva_decode_ms_p95,full_over_sva"
    )

    if args.artifact_training == "progressive":
        artifacts = build_progressive_three_stage_artifacts(model, calibration_batch, socket_layers, args, device)
    else:
        with torch.no_grad():
            calibration_output = model(**calibration_batch, use_cache=False, output_hidden_states=True)
        if calibration_output.hidden_states is None:
            raise ValueError("Artifact training requires hidden states.")
        artifacts = build_artifacts_for_hidden_states(model, calibration_output.hidden_states, socket_layers, args, device)
        del calibration_output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    quality: dict[tuple[int, int], QualityBucket] = defaultdict(QualityBucket)
    timing: dict[tuple[int, int, int], TimingBucket] = defaultdict(TimingBucket)

    min_quality_pos = max(args.min_query_pos, max(shortlists), max(budgets), args.coarse_label_topk) - 1
    for doc in eval_docs:
        eval_batch = encode_batch(tokenizer, [repeated_document(doc, args.eval_repeats)], args.context_length, device)
        seq_len = int(eval_batch["input_ids"].shape[1])
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        quality_positions_np = sample_query_positions(
            seq_len,
            args.coarse_label_topk,
            args.quality_query_samples,
            min_quality_pos,
        )
        quality_positions_t = torch.tensor(quality_positions_np, device=device, dtype=torch.long)

        with torch.no_grad():
            output = model(**eval_batch, use_cache=False, output_hidden_states=True)
        if output.hidden_states is None:
            raise ValueError("Eval pass requires hidden states.")

        for layer_idx in train_layers:
            artifact = artifacts[layer_idx]
            query, key, value, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
            _, catalog_ms, _, _ = benchmark(
                lambda: project_cached_catalog(query, key, artifact, args.rank_dim, args.assign_chunk_size),
                device,
                0,
                1,
            )
            q_low, k_low, codebooks, codes = project_cached_catalog(
                query,
                key,
                artifact,
                args.rank_dim,
                args.assign_chunk_size,
            )

            top_idx, top_valid = topk_indices_for_queries(
                query,
                key,
                quality_positions_np,
                args.coarse_label_topk,
                scaling,
            )
            for shortlist in shortlists:
                for budget in budgets:
                    if budget > shortlist:
                        continue
                    _, coarse_idx, final_idx = sva_decode_attention(
                        query,
                        key,
                        value,
                        q_low,
                        k_low,
                        codebooks,
                        codes,
                        quality_positions_t,
                        shortlist,
                        budget,
                        args.rank_dim,
                        scaling,
                    )
                    update_quality(quality[(shortlist, budget)], coarse_idx, final_idx, top_idx, top_valid)

                    for query_count in timing_query_counts:
                        timing_positions = quality_positions_t[-query_count:]
                        _, full_ms, _, _ = benchmark(
                            lambda: full_decode_attention(query, key, value, timing_positions, scaling),
                            device,
                            args.warmup,
                            args.repeats,
                        )
                        _, sva_ms, _, _ = benchmark(
                            lambda: sva_decode_attention(
                                query,
                                key,
                                value,
                                q_low,
                                k_low,
                                codebooks,
                                codes,
                                timing_positions,
                                shortlist,
                                budget,
                                args.rank_dim,
                                scaling,
                            ),
                            device,
                            args.warmup,
                            args.repeats,
                        )
                        bucket = timing[(shortlist, budget, query_count)]
                        bucket.catalog_ms.append(catalog_ms)
                        bucket.full_ms.append(full_ms)
                        bucket.sva_ms.append(sva_ms)

            del query, key, value, q_low, k_low, codebooks, codes
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for key in sorted(quality):
        print_quality_summary(key, quality[key])
    for key in sorted(timing):
        print_timing_summary(key, timing[key])
    print("cached_decode_benchmark_done")


if __name__ == "__main__":
    main()
