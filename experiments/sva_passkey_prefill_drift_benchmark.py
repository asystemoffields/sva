"""Passkey prefill drift benchmark for SVA profiles.

This isolates the final prompt-position logits before answer decoding. It is
meant to separate accumulated prefill drift from decode-time summon behavior.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import SVALlamaPatcher, patch_llama_attention
from sva_passkey_language_benchmark import (
    PromptCase,
    build_prompt_case,
    comma_ints,
    comma_strings,
    emit,
    memory_gb,
    sync_if_needed,
)
from sva_pretrained_socket_test import format_layer_list, parse_layer_list


@dataclass
class PrefillResult:
    status: str
    prefill_ms: float = math.nan
    first_token_nll: float = math.nan
    first_token_ppl: float = math.nan
    first_token_correct: int = 0
    top1_id: int = -1
    peak_memory_gb: float = 0.0
    logits: torch.Tensor | None = None
    stats: dict[str, float] | None = None
    error: str = ""


def parse_profiles(value: str) -> list[tuple[str, Path]]:
    profiles: list[tuple[str, Path]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Profiles must be formatted as name=path separated by semicolons.")
        name, path = item.split("=", 1)
        profiles.append((name.strip(), Path(path.strip())))
    if not profiles:
        raise ValueError("Expected at least one profile.")
    return profiles


@torch.no_grad()
def score_prefill(
    model: Any,
    case: PromptCase,
    device: torch.device,
    patcher: SVALlamaPatcher | None = None,
) -> PrefillResult:
    if patcher is not None:
        patcher.reset_catalogs()
        patcher.reset_stats()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        sync_if_needed(device)
        start = time.perf_counter()
        output = model(input_ids=case.input_ids, attention_mask=case.attention_mask, use_cache=True)
        sync_if_needed(device)
        prefill_ms = (time.perf_counter() - start) * 1000.0
        logits = output.logits[:, -1, :].float()
        target = int(case.answer_ids[0].item())
        log_probs = F.log_softmax(logits, dim=-1)
        nll = float((-log_probs[0, target]).item())
        top1 = int(logits.argmax(dim=-1).item())
        stats = patcher.stats.summary() if patcher is not None else {}
        peak_gb = memory_gb(device)
        del output
        return PrefillResult(
            status="ok",
            prefill_ms=prefill_ms,
            first_token_nll=nll,
            first_token_ppl=float(math.exp(min(nll, 20.0))),
            first_token_correct=int(top1 == target),
            top1_id=top1,
            peak_memory_gb=peak_gb,
            logits=logits.detach().cpu(),
            stats=stats,
        )
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return PrefillResult(status="runtime_error", error=str(exc)[:240])


def compare_logits(full: PrefillResult, sva: PrefillResult) -> dict[str, float]:
    if full.logits is None or sva.logits is None:
        return {
            "prefill_kl_to_full": math.nan,
            "prefill_top1_agreement": math.nan,
            "prefill_logit_cosine": math.nan,
        }
    full_logits = full.logits.float()
    sva_logits = sva.logits.float()
    full_log_probs = F.log_softmax(full_logits, dim=-1)
    sva_log_probs = F.log_softmax(sva_logits, dim=-1)
    full_probs = full_log_probs.exp()
    kl = (full_probs * (full_log_probs - sva_log_probs)).sum(dim=-1).mean()
    cosine = F.cosine_similarity(full_logits, sva_logits, dim=-1).mean()
    top1 = (full_logits.argmax(dim=-1) == sva_logits.argmax(dim=-1)).float().mean()
    return {
        "prefill_kl_to_full": float(kl.item()),
        "prefill_top1_agreement": float(top1.item()),
        "prefill_logit_cosine": float(cosine.item()),
    }


def stats_value(stats: dict[str, float] | None, key: str) -> float:
    if not stats:
        return math.nan
    return float(stats.get(key, math.nan))


def emit_prefill(prefix: str, profile: str, case: PromptCase, result: PrefillResult) -> None:
    emit(
        prefix,
        {
            "profile": profile,
            "status": result.status,
            "context": case.context,
            "placement": case.placement,
            "key": case.key,
            "prefill_ms": result.prefill_ms,
            "first_token_nll": result.first_token_nll,
            "first_token_ppl": result.first_token_ppl,
            "first_token_correct": result.first_token_correct,
            "top1_id": result.top1_id,
            "peak_memory_gb": result.peak_memory_gb,
            "avg_summoned": stats_value(result.stats, "avg_summoned"),
            "avg_verified": stats_value(result.stats, "avg_verified"),
            "avg_cell_visits": stats_value(result.stats, "avg_cell_visits"),
            "error": result.error,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Passkey prefill drift benchmark for SVA profiles.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--contexts", default="16384,32768")
    parser.add_argument("--placements", default="start")
    parser.add_argument("--key", default="731942")
    parser.add_argument("--shortlist", type=int, default=8192)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--summon-mode", choices=["scan", "inverted"], default="scan")
    parser.add_argument("--socket-layers", default="")
    parser.add_argument("--attn-implementation", default="sdpa")
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    profiles = parse_profiles(args.profiles)
    contexts = comma_ints(args.contexts)
    placements = comma_strings(args.placements)
    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))

    print("passkey_prefill_drift_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"placements,{args.placements}", flush=True)
    print(f"summon_mode,{args.summon_mode}", flush=True)
    print(f"socket_layers,{format_layer_list(socket_layers)}", flush=True)
    print(f"profiles,{args.profiles}", flush=True)

    for context in contexts:
        for placement in placements:
            case = build_prompt_case(tokenizer, context, args.key, placement, device)
            full = score_prefill(model, case, device)
            emit_prefill("passkey_prefill_row", "full", case, full)
            for profile_name, artifact_dir in profiles:
                patcher = patch_llama_attention(
                    model,
                    artifact_dir,
                    shortlist=args.shortlist,
                    budget=args.budget,
                    assign_chunk_size=args.assign_chunk_size,
                    query_chunk_size=args.query_chunk_size,
                    summon_mode=args.summon_mode,
                    layers=socket_layers,
                )
                try:
                    sva = score_prefill(model, case, device, patcher)
                    emit_prefill("passkey_prefill_row", profile_name, case, sva)
                finally:
                    patcher.unpatch()
                if full.status == "ok" and sva.status == "ok":
                    emit(
                        "passkey_prefill_compare",
                        {
                            "profile": profile_name,
                            "context": context,
                            "placement": placement,
                            "first_token_nll_delta": sva.first_token_nll - full.first_token_nll,
                            "prefill_slowdown": sva.prefill_ms / max(full.prefill_ms, 1e-9),
                            **compare_logits(full, sva),
                        },
                    )
                del sva
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del full
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("passkey_prefill_drift_done", flush=True)


if __name__ == "__main__":
    main()
