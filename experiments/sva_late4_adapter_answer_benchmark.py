"""Full answer-decode validation for saved late4 SVA adapter bundles."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import patch_llama_attention
from sva_late4_logit_distill import (
    load_adapter_bundle,
    load_adapter_weights,
    wrap_socket_layers_with_adapters,
)
from sva_passkey_language_benchmark import (
    build_prompt_case,
    comma_ints,
    comma_strings,
    compare_answer_logits,
    emit,
    emit_score,
    score_answer_decode,
)
from sva_pretrained_socket_test import format_layer_list, parse_layer_list


def mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if not math.isnan(row[key])]
    if not values:
        return math.nan
    return sum(values) / len(values)


def emit_comparison(
    variant: str,
    case,
    full_result,
    candidate_result,
) -> dict[str, float]:
    comparison = {
        "answer_nll_delta": candidate_result.answer_nll - full_result.answer_nll,
        "prefill_slowdown": candidate_result.prefill_ms / max(full_result.prefill_ms, 1e-9),
        "decode_slowdown": candidate_result.decode_ms / max(full_result.decode_ms, 1e-9),
        **compare_answer_logits(full_result, candidate_result),
    }
    emit(
        "adapter_answer_compare",
        {
            "variant": variant,
            "context": case.context,
            "placement": case.placement,
            "key": case.key,
            **comparison,
        },
    )
    return comparison


def emit_mean(variant: str, comparisons: list[dict[str, float]]) -> None:
    if not comparisons:
        return
    emit(
        "adapter_answer_mean",
        {
            "variant": variant,
            "cases": len(comparisons),
            "answer_nll_delta": mean(comparisons, "answer_nll_delta"),
            "prefill_slowdown": mean(comparisons, "prefill_slowdown"),
            "decode_slowdown": mean(comparisons, "decode_slowdown"),
            "answer_kl_to_full": mean(comparisons, "answer_kl_to_full"),
            "answer_top1_agreement": mean(comparisons, "answer_top1_agreement"),
            "answer_logit_cosine": mean(comparisons, "answer_logit_cosine"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved late4 SVA adapter on full answer decode.")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--contexts", default="32768")
    parser.add_argument("--keys", default="731942,184207,905613")
    parser.add_argument("--placements", default="start,middle,end")
    parser.add_argument("--socket-layers", default="")
    parser.add_argument("--shortlist", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--query-chunk-size", type=int, default=None)
    parser.add_argument("--summon-mode", choices=["scan", "inverted"], default="")
    parser.add_argument("--inverted-cells-per-subspace", type=int, default=32)
    parser.add_argument("--adapter-rank", type=int, default=None)
    parser.add_argument("--adapter-scale", type=float, default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    manifest, adapter_state = load_adapter_bundle(args.adapter_dir)
    model_id = args.model_id or str(manifest["model_id"])
    artifact_dir = args.artifact_dir if args.artifact_dir is not None else Path(str(manifest["artifact_dir"]))
    shortlist = args.shortlist if args.shortlist is not None else int(manifest["shortlist"])
    budget = args.budget if args.budget is not None else int(manifest["budget"])
    query_chunk_size = args.query_chunk_size if args.query_chunk_size is not None else int(manifest["query_chunk_size"])
    summon_mode = args.summon_mode or str(manifest["summon_mode"])
    adapter_rank = args.adapter_rank if args.adapter_rank is not None else int(manifest["adapter_rank"])
    adapter_scale = args.adapter_scale if args.adapter_scale is not None else float(manifest["adapter_scale"])

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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    contexts = comma_ints(args.contexts)
    keys = comma_strings(args.keys)
    placements = comma_strings(args.placements)
    if args.socket_layers:
        socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    else:
        socket_layers = [int(layer) for layer in manifest["socket_layers"]]
    if socket_layers is None:
        raise ValueError("Adapter answer benchmark expects explicit socket layers.")

    print("late4_adapter_answer_start", flush=True)
    print(f"model_id,{model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"artifact_dir,{artifact_dir}", flush=True)
    print(f"adapter_dir,{args.adapter_dir}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"keys,{','.join(keys)}", flush=True)
    print(f"placements,{args.placements}", flush=True)
    print(f"socket_layers,{format_layer_list(socket_layers)}", flush=True)
    print(f"shortlist,{shortlist}", flush=True)
    print(f"budget,{budget}", flush=True)
    print(f"adapter_rank,{adapter_rank}", flush=True)

    cases = [build_prompt_case(tokenizer, context, key, placement, device) for context in contexts for key in keys for placement in placements]
    full_results = []
    for case in cases:
        result = score_answer_decode(model, case, device, patcher=None)
        emit_score("adapter_answer_row", "full", case, result)
        full_results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    patcher = patch_llama_attention(
        model,
        artifact_dir,
        shortlist=shortlist,
        budget=budget,
        assign_chunk_size=args.assign_chunk_size,
        query_chunk_size=query_chunk_size,
        summon_mode=summon_mode,
        inverted_cells_per_subspace=args.inverted_cells_per_subspace,
        layers=socket_layers,
    )

    unadapted_comparisons: list[dict[str, float]] = []
    for case, full_result in zip(cases, full_results, strict=True):
        unadapted_result = score_answer_decode(model, case, device, patcher=patcher)
        emit_score("adapter_answer_row", "sva_unadapted", case, unadapted_result)
        if full_result.status == "ok" and unadapted_result.status == "ok":
            unadapted_comparisons.append(emit_comparison("sva_unadapted", case, full_result, unadapted_result))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    wrap_socket_layers_with_adapters(model, socket_layers, adapter_rank, adapter_scale)
    load_adapter_weights(model, socket_layers, adapter_state)
    model.eval()

    adapted_comparisons: list[dict[str, float]] = []
    try:
        for case, full_result in zip(cases, full_results, strict=True):
            adapted_result = score_answer_decode(model, case, device, patcher=patcher)
            emit_score("adapter_answer_row", "sva_adapted", case, adapted_result)
            if full_result.status == "ok" and adapted_result.status == "ok":
                adapted_comparisons.append(emit_comparison("sva_adapted", case, full_result, adapted_result))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        patcher.unpatch()

    emit_mean("sva_unadapted", unadapted_comparisons)
    emit_mean("sva_adapted", adapted_comparisons)
    print("late4_adapter_answer_done", flush=True)


if __name__ == "__main__":
    main()
