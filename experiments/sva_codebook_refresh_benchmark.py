"""Held-out codebook refresh benchmark for SVA catalogs.

The rotation diagnostic showed that refitting product-code codebooks on the
evaluated key bank sharply improves PQ score alignment. This benchmark removes
that transductive advantage: it fits refreshed codebooks on calibration text,
then evaluates recall on held-out documents.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sva_block_elevator_benchmark import layer_qkv_from_hidden, project_catalog
from sva_full_deployment_benchmark import (
    CALIBRATION_DOCS,
    EVAL_DOCS,
    calibration_stream,
    comma_ints,
    load_documents,
    repeated_document,
)
from sva_passkey_language_benchmark import comma_strings
from sva_pq_lookup_test import encode_product_keys, fit_product_codebooks, product_quantized_scores
from sva_pretrained_socket_test import encode_batch, parse_layer_list
from sva_rotation_diagnostic import (
    apply_rotation,
    candidate_recall,
    mean_code_max_fraction,
    normalized_code_entropy,
    rotation_matrix,
    score_alignment,
    topk_targets,
)
from sva_supervised_coarse_pq_test import fit_weighted_product_codebooks

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import load_sva_artifact_bundle


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value.replace("\n", " ").replace(",", ";")
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def emit(prefix: str, row: dict[str, float | int | str]) -> None:
    print(prefix + "," + ",".join(f"{key}={row_value(value)}" for key, value in row.items()), flush=True)


def query_positions_for(seq_len: int, samples: int, min_pos: int, device: torch.device) -> torch.Tensor:
    start = max(0, min(min_pos, seq_len - 1))
    positions = torch.linspace(start, seq_len - 1, steps=min(samples, seq_len), device=device).long().unique()
    if positions[-1].item() != seq_len - 1:
        positions = torch.unique(torch.cat([positions, torch.tensor([seq_len - 1], device=device)]))
    return positions


def repeated_document_to_length(tokenizer, doc, min_tokens: int, initial_repeats: int) -> str:
    repeats = max(1, initial_repeats)
    while True:
        text = repeated_document(doc, repeats)
        token_count = len(tokenizer(text, add_special_tokens=True).input_ids)
        if token_count >= min_tokens:
            return text
        scale = max(2, math.ceil(min_tokens / max(token_count, 1)))
        repeats *= scale


def calibration_stream_to_length(tokenizer, docs, min_tokens: int, initial_repeats: int) -> str:
    repeats = max(1, initial_repeats)
    while True:
        text = calibration_stream(docs, repeats)
        token_count = len(tokenizer(text, add_special_tokens=True).input_ids)
        if token_count >= min_tokens:
            return text
        scale = max(2, math.ceil(min_tokens / max(token_count, 1)))
        repeats *= scale


def variant_rotation_label(variant: str) -> str:
    if variant in {
        "artifact_identity",
        "calib_identity",
        "calib_attention_weighted",
        "calib_attention_weighted_strong",
        "eval_refit_identity",
    }:
        return "identity"
    if variant == "calib_hadamard":
        return "hadamard"
    if variant == "calib_signed_hadamard":
        return "signed_hadamard"
    raise ValueError(f"Unknown variant: {variant}")


def is_calibration_variant(variant: str) -> bool:
    return variant in {
        "calib_identity",
        "calib_hadamard",
        "calib_signed_hadamard",
        "calib_attention_weighted",
        "calib_attention_weighted_strong",
    }


def is_attention_weighted_variant(variant: str) -> bool:
    return variant in {
        "calib_attention_weighted",
        "calib_attention_weighted_strong",
        "attention_weighted",
        "attention_weighted_strong",
    }


def attention_boost_for_variant(variant: str, base_boost: float) -> float:
    if variant in {"calib_attention_weighted_strong", "attention_weighted_strong"}:
        return base_boost * 4.0
    return base_boost


@torch.no_grad()
def topk_key_weights(
    top_idx: torch.Tensor,
    top_valid: torch.Tensor,
    seq_len: int,
    boost: float,
) -> torch.Tensor:
    weights = torch.ones(top_idx.shape[0], seq_len, device=top_idx.device, dtype=torch.float32)
    for head_idx in range(top_idx.shape[0]):
        valid_idx = top_idx[head_idx][top_valid[head_idx]]
        if valid_idx.numel() == 0:
            continue
        weights[head_idx].index_add_(
            0,
            valid_idx.long().clamp(0, seq_len - 1),
            torch.full((valid_idx.numel(),), float(boost), device=top_idx.device, dtype=torch.float32),
        )
    return weights


@torch.no_grad()
def fit_calibration_codebooks(
    *,
    model,
    bundle,
    calibration_batch: dict[str, torch.Tensor],
    layers: list[int],
    variants: list[str],
    topk: int,
    calibration_query_samples: int,
    min_query_pos: int,
    attention_boost: float,
    kmeans_iters: int,
    assign_chunk_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], torch.Tensor | None]]:
    if not any(is_calibration_variant(variant) for variant in variants):
        return {}, {}

    seq_len = int(calibration_batch["input_ids"].shape[1])
    output = model(**calibration_batch, use_cache=False, output_hidden_states=True)
    if output.hidden_states is None:
        raise ValueError("Expected calibration hidden states.")
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    codebooks_by_variant: dict[tuple[str, int], torch.Tensor] = {}
    rotations_by_variant: dict[tuple[str, int], torch.Tensor | None] = {}

    for layer_idx in layers:
        query_all, key_all, _, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
        _, k_low, _, _ = project_catalog(query_all, key_all, bundle, layer_idx, assign_chunk_size)
        weighted_targets: tuple[torch.Tensor, torch.Tensor] | None = None
        for variant in variants:
            if not is_calibration_variant(variant):
                continue
            rotation_label = variant_rotation_label(variant)
            rotation = rotation_matrix(rotation_label, bundle.rank_dim, seed + layer_idx * 1009, device)
            rotated_k = apply_rotation(k_low, rotation)
            if is_attention_weighted_variant(variant):
                if weighted_targets is None:
                    query_positions = query_positions_for(seq_len, calibration_query_samples, min_query_pos, device)
                    full_scores = (
                        torch.einsum("hqd,hkd->hqk", query_all[:, query_positions].float(), key_all.float())
                        * scaling
                    )
                    weighted_targets = topk_targets(full_scores, query_positions, topk)
                top_idx, top_valid = weighted_targets
                weights = topk_key_weights(
                    top_idx,
                    top_valid,
                    seq_len,
                    attention_boost_for_variant(variant, attention_boost),
                )
                codebooks = fit_weighted_product_codebooks(
                    rotated_k,
                    weights,
                    bundle.coarse_subspaces,
                    bundle.coarse_codewords,
                    kmeans_iters,
                    seed + layer_idx * 9973 + len(variant) * 37,
                    assign_chunk_size,
                )
            else:
                codebooks = fit_product_codebooks(
                    rotated_k,
                    bundle.coarse_subspaces,
                    bundle.coarse_codewords,
                    kmeans_iters,
                    seed + layer_idx * 9973 + len(variant) * 37,
                    assign_chunk_size,
                )
            codebooks_by_variant[(variant, layer_idx)] = codebooks.detach()
            rotations_by_variant[(variant, layer_idx)] = rotation.detach() if rotation is not None else None
            codes = encode_product_keys(rotated_k, codebooks, assign_chunk_size)
            emit(
                "codebook_refresh_fit",
                {
                    "variant": variant,
                    "layer": layer_idx,
                    "calibration_seq_len": seq_len,
                    "attention_boost": attention_boost_for_variant(variant, attention_boost)
                    if is_attention_weighted_variant(variant)
                    else 0.0,
                    "code_entropy": normalized_code_entropy(codes, bundle.coarse_codewords),
                    "code_max_fraction": mean_code_max_fraction(codes, bundle.coarse_codewords),
                },
            )
        del query_all, key_all, k_low
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del output
    return codebooks_by_variant, rotations_by_variant


def add_aggregate(
    aggregate: dict[tuple[str, int, int], dict[str, float]],
    row: dict[str, float | int | str],
    teacher_hits: int,
    teacher_total: int,
    low_hits: int,
    low_total: int,
) -> None:
    key = (str(row["variant"]), int(row["context"]), int(row["budget"]))
    bucket = aggregate.setdefault(
        key,
        {
            "rows": 0.0,
            "teacher_hits": 0.0,
            "teacher_total": 0.0,
            "low_hits": 0.0,
            "low_total": 0.0,
            "score_cosine": 0.0,
            "score_mse": 0.0,
            "code_entropy": 0.0,
            "code_max_fraction": 0.0,
        },
    )
    bucket["rows"] += 1.0
    bucket["teacher_hits"] += float(teacher_hits)
    bucket["teacher_total"] += float(teacher_total)
    bucket["low_hits"] += float(low_hits)
    bucket["low_total"] += float(low_total)
    for metric in ("score_cosine", "score_mse", "code_entropy", "code_max_fraction"):
        bucket[metric] += float(row[metric])


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out SVA codebook refresh benchmark.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--calibration-doc-limit", type=int, default=None)
    parser.add_argument("--eval-doc-limit", type=int, default=4)
    parser.add_argument("--calibration-repeats", type=int, default=320)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--calibration-length", type=int, default=0)
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--allow-beyond-model-context", action="store_true")
    parser.add_argument(
        "--variants",
        default="artifact_identity,calib_identity,calib_signed_hadamard,eval_refit_identity",
    )
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--budgets", default="512,1024,2048")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--query-samples", type=int, default=64)
    parser.add_argument("--calibration-query-samples", type=int, default=128)
    parser.add_argument("--attention-boost", type=float, default=4.0)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--kmeans-iters", type=int, default=8)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--attn-implementation", default="eager")
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

    contexts = comma_ints(args.contexts)
    budgets = comma_ints(args.budgets)
    variants = comma_strings(args.variants)
    calibration_length = args.calibration_length or max(contexts)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    bundle = load_sva_artifact_bundle(args.artifact_dir, map_location=device)
    layers = parse_layer_list(args.layers, len(model.model.layers))
    layers = layers if layers is not None else list(range(len(model.model.layers)))

    model_context = getattr(model.config, "max_position_embeddings", None)
    if (
        model_context is not None
        and max(max(contexts), calibration_length) > int(model_context)
        and not args.allow_beyond_model_context
    ):
        raise ValueError(
            f"Requested context {max(max(contexts), calibration_length)} exceeds model max_position_embeddings "
            f"{model_context}. Pass --allow-beyond-model-context to override."
        )

    calibration_docs = load_documents(args.calibration_file, CALIBRATION_DOCS, args.calibration_doc_limit, "calibration")
    eval_docs = load_documents(args.eval_file, EVAL_DOCS, args.eval_doc_limit, "eval")
    calibration_text = calibration_stream_to_length(tokenizer, calibration_docs, calibration_length, args.calibration_repeats)
    calibration_batch = encode_batch(tokenizer, [calibration_text], calibration_length, device)

    print("codebook_refresh_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"artifact_profile,{bundle.manifest.get('profile_name')}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"calibration_docs,{len(calibration_docs)}", flush=True)
    print(f"eval_docs,{len(eval_docs)}", flush=True)
    print(f"calibration_seq_len,{calibration_batch['input_ids'].shape[1]}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"layers,{args.layers}", flush=True)
    print(f"variants,{args.variants}", flush=True)

    calibration_codebooks, calibration_rotations = fit_calibration_codebooks(
        model=model,
        bundle=bundle,
        calibration_batch=calibration_batch,
        layers=layers,
        variants=variants,
        topk=args.topk,
        calibration_query_samples=args.calibration_query_samples,
        min_query_pos=args.min_query_pos,
        attention_boost=args.attention_boost,
        kmeans_iters=args.kmeans_iters,
        assign_chunk_size=args.assign_chunk_size,
        seed=args.seed,
        device=device,
    )
    del calibration_batch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    aggregate: dict[tuple[str, int, int], dict[str, float]] = {}
    for context in contexts:
        for doc_index, doc in enumerate(eval_docs):
            eval_text = repeated_document_to_length(tokenizer, doc, context, args.eval_repeats)
            batch = encode_batch(tokenizer, [eval_text], context, device)
            seq_len = int(batch["input_ids"].shape[1])
            with torch.no_grad():
                output = model(**batch, use_cache=False, output_hidden_states=True)
            if output.hidden_states is None:
                raise ValueError("Expected eval hidden states.")
            query_positions = query_positions_for(seq_len, args.query_samples, args.min_query_pos, device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            for layer_idx in layers:
                query_all, key_all, _, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
                q_low, k_low, artifact_codebooks, artifact_codes = project_catalog(query_all, key_all, bundle, layer_idx, args.assign_chunk_size)
                full_scores = torch.einsum("hqd,hkd->hqk", query_all[:, query_positions].float(), key_all.float()) * scaling
                low_scores = torch.einsum("hqr,hkr->hqk", q_low[:, query_positions].float(), k_low.float()) / math.sqrt(bundle.rank_dim)
                teacher_idx, teacher_valid = topk_targets(full_scores, query_positions, args.topk)
                low_idx, low_valid = topk_targets(low_scores, query_positions, args.topk)

                for variant in variants:
                    if variant == "artifact_identity":
                        rotated_q = q_low[:, query_positions]
                        codebooks = artifact_codebooks
                        codes = artifact_codes
                    elif is_calibration_variant(variant):
                        rotation = calibration_rotations[(variant, layer_idx)]
                        rotated_q = apply_rotation(q_low[:, query_positions], rotation)
                        rotated_k = apply_rotation(k_low, rotation)
                        codebooks = calibration_codebooks[(variant, layer_idx)].to(device=device, dtype=rotated_k.dtype)
                        codes = encode_product_keys(rotated_k, codebooks, args.assign_chunk_size)
                    elif variant == "eval_refit_identity":
                        rotated_q = q_low[:, query_positions]
                        codebooks = fit_product_codebooks(
                            k_low,
                            bundle.coarse_subspaces,
                            bundle.coarse_codewords,
                            args.kmeans_iters,
                            args.seed + context * 31 + doc_index * 313 + layer_idx * 997,
                            args.assign_chunk_size,
                        )
                        codes = encode_product_keys(k_low, codebooks, args.assign_chunk_size)
                    else:
                        raise ValueError(f"Unknown variant: {variant}")

                    pq_scores = product_quantized_scores(rotated_q, codebooks, codes, bundle.rank_dim)
                    align_cosine, align_mse = score_alignment(pq_scores, low_scores, query_positions)
                    entropy = normalized_code_entropy(codes, bundle.coarse_codewords)
                    max_fraction = mean_code_max_fraction(codes, bundle.coarse_codewords)
                    for budget in budgets:
                        teacher_recall, teacher_hits, teacher_total = candidate_recall(
                            pq_scores,
                            teacher_idx,
                            teacher_valid,
                            query_positions,
                            budget,
                        )
                        low_recall, low_hits, low_total = candidate_recall(
                            pq_scores,
                            low_idx,
                            low_valid,
                            query_positions,
                            budget,
                        )
                        row = {
                            "context": context,
                            "seq_len": seq_len,
                            "doc": doc.doc_id,
                            "layer": layer_idx,
                            "variant": variant,
                            "budget": budget,
                            "topk": args.topk,
                            "query_samples": int(query_positions.numel()),
                            "teacher_topk_recall": teacher_recall,
                            "teacher_hits": teacher_hits,
                            "teacher_total": teacher_total,
                            "low_topk_recall": low_recall,
                            "low_hits": low_hits,
                            "low_total": low_total,
                            "score_cosine": align_cosine,
                            "score_mse": align_mse,
                            "code_entropy": entropy,
                            "code_max_fraction": max_fraction,
                        }
                        emit("codebook_refresh_result", row)
                        add_aggregate(aggregate, row, teacher_hits, teacher_total, low_hits, low_total)

                del query_all, key_all, q_low, k_low, artifact_codebooks, artifact_codes
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del output, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for (variant, context, budget), bucket in sorted(aggregate.items()):
        rows = max(bucket["rows"], 1.0)
        emit(
            "codebook_refresh_summary",
            {
                "variant": variant,
                "context": context,
                "budget": budget,
                "teacher_topk_recall": bucket["teacher_hits"] / max(bucket["teacher_total"], 1.0),
                "low_topk_recall": bucket["low_hits"] / max(bucket["low_total"], 1.0),
                "score_cosine": bucket["score_cosine"] / rows,
                "score_mse": bucket["score_mse"] / rows,
                "code_entropy": bucket["code_entropy"] / rows,
                "code_max_fraction": bucket["code_max_fraction"] / rows,
                "rows": int(rows),
            },
        )

    print("codebook_refresh_done", flush=True)


if __name__ == "__main__":
    main()
