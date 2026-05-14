"""Long-context SVA recall simulation from real SmolLM2 Q/K activations.

This is a retrieval-accuracy proxy, not a full long-context language benchmark.
It freezes the exported 8k SVA artifact, extracts real Q/K activations from a
held-out 8k sequence, synthesizes longer cached key banks from that activation
distribution, and compares SVA candidates against exact full-attention top keys.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import SVAArtifactBundle, load_sva_artifact_bundle
from sva.ops import encode_product_keys, product_quantized_scores
from sva_full_deployment_benchmark import EVAL_DOCS, repeated_document
from sva_pretrained_socket_test import encode_batch, parse_layer_list


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def emit(prefix: str, row: dict[str, float | int | str]) -> None:
    print(prefix + "," + ",".join(f"{key}={row_value(value)}" for key, value in row.items()), flush=True)


@torch.no_grad()
def layer_qk_from_hidden(
    model: Any,
    hidden_states: tuple[torch.Tensor, ...],
    layer_idx: int,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    layer = model.model.layers[layer_idx]
    hidden = layer.input_layernorm(hidden_states[layer_idx])
    hidden_shape = (hidden.shape[0], hidden.shape[1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = repeat_kv(key, layer.self_attn.num_key_value_groups)
    return query[0].float(), key[0].float(), float(layer.self_attn.scaling)


def sample_query_positions(seq_len: int, samples: int, min_pos: int) -> torch.Tensor:
    start = min(max(min_pos, 0), seq_len - 1)
    positions = np.linspace(start, seq_len - 1, num=min(samples, seq_len - start), dtype=np.int64)
    return torch.tensor(np.unique(positions), dtype=torch.long)


@torch.no_grad()
def make_synthetic_key_bank(
    base_key: torch.Tensor,
    context: int,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    heads, base_len, head_dim = base_key.shape
    generator = torch.Generator(device=base_key.device)
    generator.manual_seed(seed + context * 17 + heads * 101)
    source_idx = torch.randint(base_len, (context,), device=base_key.device, generator=generator)
    bank = base_key[:, source_idx, :].clone()
    if noise_std > 0:
        noise = torch.randn(bank.shape, device=bank.device, dtype=bank.dtype, generator=generator) * noise_std
        bank = bank + noise
    return bank


@torch.no_grad()
def exact_topk_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    topk: int,
    scaling: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, queries, _ = query.shape
    top_scores = torch.full((heads, queries, 0), -float("inf"), device=query.device)
    top_indices = torch.empty((heads, queries, 0), device=query.device, dtype=torch.long)
    for start in range(0, key.shape[1], chunk_size):
        end = min(start + chunk_size, key.shape[1])
        scores = torch.einsum("hqd,hkd->hqk", query.float(), key[:, start:end, :].float()) * scaling
        chunk_k = min(topk, scores.shape[-1])
        chunk_scores, chunk_idx = scores.topk(chunk_k, dim=-1)
        chunk_idx = chunk_idx + start
        merged_scores = torch.cat([top_scores, chunk_scores], dim=-1)
        merged_idx = torch.cat([top_indices, chunk_idx], dim=-1)
        keep_k = min(topk, merged_scores.shape[-1])
        top_scores, order = merged_scores.topk(keep_k, dim=-1)
        top_indices = merged_idx.gather(dim=-1, index=order)
    return top_scores, top_indices


@torch.no_grad()
def project_catalog(
    query: torch.Tensor,
    key: torch.Tensor,
    bundle: SVAArtifactBundle,
    layer_idx: int,
    assign_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    artifact = bundle.layers[layer_idx]
    q_proj = artifact.q_proj.to(device=query.device, dtype=torch.float32)
    k_proj = artifact.k_proj.to(device=query.device, dtype=torch.float32)
    scale = artifact.logit_scale.to(device=query.device, dtype=torch.float32).exp().clamp(0.01, 100.0)
    q_low = torch.einsum("hqd,hdr->hqr", query.float(), q_proj) * scale[:, None, None]
    k_low = torch.einsum("hkd,hdr->hkr", key.float(), k_proj)
    codebooks = artifact.coarse_codebooks.to(device=query.device, dtype=torch.float32)
    codes = encode_product_keys(k_low, codebooks, assign_chunk_size)
    return q_low, k_low, codebooks, codes


@torch.no_grad()
def sva_candidates(
    query: torch.Tensor,
    key: torch.Tensor,
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    shortlist: int,
    budget: int,
    rank_dim: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = key.shape[1]
    actual_shortlist = min(shortlist, context)
    actual_budget = min(budget, actual_shortlist)
    coarse_scores = product_quantized_scores(q_low, codebooks, codes, rank_dim)
    coarse_idx = coarse_scores.topk(actual_shortlist, dim=-1).indices
    heads, queries, _ = coarse_idx.shape
    selected_low = k_low[:, None, :, :].expand(heads, queries, context, rank_dim).gather(
        dim=2,
        index=coarse_idx[..., None].expand(heads, queries, actual_shortlist, rank_dim),
    )
    rank_scores = (selected_low.float() * q_low[:, :, None, :].float()).sum(dim=-1) / math.sqrt(rank_dim)
    rank_keep = rank_scores.topk(actual_budget, dim=-1).indices
    final_idx = coarse_idx.gather(dim=-1, index=rank_keep)
    return coarse_idx, final_idx


def recall_at_topk(candidate_idx: torch.Tensor, teacher_idx: torch.Tensor) -> float:
    hits = 0
    total = teacher_idx.numel()
    for head_idx in range(teacher_idx.shape[0]):
        for query_idx in range(teacher_idx.shape[1]):
            candidate = set(int(item) for item in candidate_idx[head_idx, query_idx].detach().cpu().tolist())
            for target in teacher_idx[head_idx, query_idx].detach().cpu().tolist():
                hits += int(int(target) in candidate)
    return hits / max(total, 1)


def cache_bytes(config: Any, context: int, dtype_bytes: int) -> dict[str, float]:
    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(config.hidden_size // config.num_attention_heads)
    full_kv = layers * context * kv_heads * head_dim * 2 * dtype_bytes
    return {
        "full_kv_cache_gb": full_kv / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-context SVA retrieval recall simulation.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--base-context", type=int, default=8192)
    parser.add_argument("--target-contexts", default="8192,32768,131072,1000000")
    parser.add_argument("--layers", default="0,7,15,23,29")
    parser.add_argument("--eval-doc-index", type=int, default=0)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--query-samples", type=int, default=16)
    parser.add_argument("--min-query-pos", type=int, default=512)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--shortlists", default="1024,2048")
    parser.add_argument("--budgets", default="256,512")
    parser.add_argument("--synthetic-noise-std", type=float, default=0.01)
    parser.add_argument("--teacher-chunk-size", type=int, default=65536)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=17)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    bundle = load_sva_artifact_bundle(args.artifact_dir, map_location=device)
    target_contexts = comma_ints(args.target_contexts)
    shortlists = comma_ints(args.shortlists)
    budgets = comma_ints(args.budgets)
    layers = parse_layer_list(args.layers, len(model.model.layers))
    layers = layers if layers is not None else list(range(len(model.model.layers)))

    doc = EVAL_DOCS[args.eval_doc_index]
    batch = encode_batch(tokenizer, [repeated_document(doc, args.eval_repeats)], args.base_context, device)
    seq_len = int(batch["input_ids"].shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    query_positions = sample_query_positions(seq_len, args.query_samples, args.min_query_pos).to(device)

    print("long_context_recall_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"base_context,{seq_len}", flush=True)
    print(f"target_contexts,{args.target_contexts}", flush=True)
    print(f"layers,{args.layers}", flush=True)
    print(f"artifact_profile,{bundle.manifest.get('profile_name')}", flush=True)

    with torch.no_grad():
        output = model(**batch, use_cache=False, output_hidden_states=True)
    if output.hidden_states is None:
        raise ValueError("Expected hidden states from model forward.")

    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    aggregate: dict[tuple[int, int, int], dict[str, float]] = {}
    for layer_idx in layers:
        query_all, key_base, scaling = layer_qk_from_hidden(model, output.hidden_states, layer_idx, position_ids)
        query = query_all[:, query_positions, :]
        for context in target_contexts:
            synchronize(device)
            start = time.perf_counter()
            key_bank = make_synthetic_key_bank(
                key_base,
                context,
                args.synthetic_noise_std,
                args.seed + layer_idx * 1000,
            )
            _, teacher_idx = exact_topk_chunked(
                query,
                key_bank,
                args.topk,
                scaling,
                args.teacher_chunk_size,
            )
            q_low, k_low, codebooks, codes = project_catalog(
                query,
                key_bank,
                bundle,
                layer_idx,
                args.assign_chunk_size,
            )
            catalog_seconds = time.perf_counter() - start
            for shortlist in shortlists:
                for budget in budgets:
                    if budget > shortlist:
                        continue
                    candidate_idx, final_idx = sva_candidates(
                        query,
                        key_bank,
                        q_low,
                        k_low,
                        codebooks,
                        codes,
                        shortlist,
                        budget,
                        bundle.rank_dim,
                        args.teacher_chunk_size,
                    )
                    candidate_recall = recall_at_topk(candidate_idx, teacher_idx)
                    verified_recall = recall_at_topk(final_idx, teacher_idx)
                    storage = cache_bytes(model.config, context, dtype_bytes)
                    row = {
                        "layer": layer_idx,
                        "context": context,
                        "extension_x": context / max(seq_len, 1),
                        "shortlist": shortlist,
                        "budget": budget,
                        "topk": args.topk,
                        "queries": int(query.shape[0] * query.shape[1]),
                        "candidate_topk_recall": candidate_recall,
                        "verified_topk_recall": verified_recall,
                        "exact_score_reduction": context / max(budget, 1),
                        "value_read_reduction": context / max(budget, 1),
                        "catalog_plus_teacher_s": catalog_seconds,
                        **storage,
                    }
                    emit("long_context_recall_row", row)
                    key = (context, shortlist, budget)
                    bucket = aggregate.setdefault(
                        key,
                        {
                            "candidate": 0.0,
                            "verified": 0.0,
                            "count": 0.0,
                            "storage": storage["full_kv_cache_gb"],
                        },
                    )
                    bucket["candidate"] += candidate_recall
                    bucket["verified"] += verified_recall
                    bucket["count"] += 1.0
            del key_bank, q_low, k_low, codebooks, codes, teacher_idx
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del query_all, key_base, query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (context, shortlist, budget), bucket in sorted(aggregate.items()):
        count = max(bucket["count"], 1.0)
        emit(
            "long_context_recall_summary",
            {
                "context": context,
                "extension_x": context / max(seq_len, 1),
                "shortlist": shortlist,
                "budget": budget,
                "layers": int(count),
                "candidate_topk_recall": bucket["candidate"] / count,
                "verified_topk_recall": bucket["verified"] / count,
                "exact_score_reduction": context / max(budget, 1),
                "value_read_reduction": context / max(budget, 1),
                "full_kv_cache_gb": bucket["storage"],
            },
        )
    print("long_context_recall_done", flush=True)


if __name__ == "__main__":
    main()
