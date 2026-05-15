"""Passkey-style long-context language benchmark for SVA.

The benchmark builds a long prompt containing a single passkey, asks for that
passkey at the end, and scores the correct answer tokens. It compares full
attention with the production SVA adapter where full attention is still used as
teacher, and can continue with SVA-only rows at longer contexts.
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
from sva_pretrained_socket_test import format_layer_list, parse_layer_list


FILLER = (
    "\nThe archive paragraph describes routine operations, release checks, "
    "support notes, billing policies, telescope alignment, database migration, "
    "and scheduler behavior. It contains many ordinary nouns and numbers, but "
    "none of them are the private passkey requested later."
)


@dataclass(frozen=True)
class PromptCase:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    answer_ids: torch.Tensor
    key: str
    placement: str
    context: int


@dataclass
class ScoreResult:
    status: str
    prefill_ms: float = math.nan
    decode_ms: float = math.nan
    answer_nll: float = math.nan
    answer_ppl: float = math.nan
    first_token_correct: int = 0
    all_answer_top1: int = 0
    peak_memory_gb: float = 0.0
    top1_ids: list[int] | None = None
    answer_logits: torch.Tensor | None = None
    prefill_stats: dict[str, float] | None = None
    decode_stats: dict[str, float] | None = None
    error: str = ""


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def comma_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


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


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def repeated_to_length(tokens: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not tokens:
        raise ValueError("Cannot repeat an empty token list.")
    repeats = (length + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:length]


def build_prompt_case(
    tokenizer: Any,
    context: int,
    key: str,
    placement: str,
    device: torch.device,
) -> PromptCase:
    if tokenizer.bos_token_id is not None:
        bos = [int(tokenizer.bos_token_id)]
    else:
        bos = []
    needle = encode_text(tokenizer, f"\nMemorandum: the private passkey is {key}.\n")
    query = encode_text(tokenizer, "\nQuestion: What is the private passkey?\nAnswer:")
    filler = encode_text(tokenizer, FILLER)
    answer = encode_text(tokenizer, f" {key}")
    fixed = len(bos) + len(needle) + len(query)
    fill_len = context - fixed
    if fill_len < 0:
        raise ValueError(f"Context {context} is too short for the passkey prompt; need at least {fixed} tokens.")

    if placement == "start":
        prefix = bos + needle + repeated_to_length(filler, fill_len) + query
    elif placement == "middle":
        left = fill_len // 2
        right = fill_len - left
        prefix = bos + repeated_to_length(filler, left) + needle + repeated_to_length(filler, right) + query
    elif placement == "end":
        prefix = bos + repeated_to_length(filler, fill_len) + needle + query
    else:
        raise ValueError(f"Unknown placement: {placement}")

    if len(prefix) != context:
        raise AssertionError(f"Prompt length mismatch: expected {context}, got {len(prefix)}.")
    input_ids = torch.tensor([prefix], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    answer_ids = torch.tensor(answer, device=device, dtype=torch.long)
    return PromptCase(
        input_ids=input_ids,
        attention_mask=attention_mask,
        answer_ids=answer_ids,
        key=key,
        placement=placement,
        context=context,
    )


@torch.no_grad()
def score_answer_decode(
    model: Any,
    case: PromptCase,
    device: torch.device,
    patcher: SVALlamaPatcher | None = None,
) -> ScoreResult:
    if patcher is not None:
        patcher.reset_catalogs()
        patcher.reset_stats()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        sync_if_needed(device)
        start = time.perf_counter()
        output = model(**{"input_ids": case.input_ids, "attention_mask": case.attention_mask}, use_cache=True)
        sync_if_needed(device)
        prefill_ms = (time.perf_counter() - start) * 1000
        prefill_stats = patcher.stats.summary() if patcher is not None else {}
        if patcher is not None:
            patcher.reset_stats()

        logits: list[torch.Tensor] = []
        top1_ids: list[int] = []
        losses: list[torch.Tensor] = []

        first_logits = output.logits[:, -1, :].float()
        first_target = case.answer_ids[0:1]
        first_log_probs = F.log_softmax(first_logits, dim=-1)
        losses.append(-first_log_probs[0, first_target[0]])
        logits.append(first_logits.detach().cpu())
        top1_ids.append(int(first_logits.argmax(dim=-1).item()))

        past = output.past_key_values
        decode_elapsed = 0.0
        current_mask = case.attention_mask
        for answer_index in range(len(case.answer_ids) - 1):
            next_input = case.answer_ids[answer_index : answer_index + 1].view(1, 1)
            current_mask = torch.cat([current_mask, torch.ones((1, 1), device=device, dtype=current_mask.dtype)], dim=1)
            sync_if_needed(device)
            step_start = time.perf_counter()
            step = model(
                input_ids=next_input,
                attention_mask=current_mask,
                past_key_values=past,
                use_cache=True,
            )
            sync_if_needed(device)
            decode_elapsed += time.perf_counter() - step_start
            past = step.past_key_values
            step_logits = step.logits[:, -1, :].float()
            target = case.answer_ids[answer_index + 1]
            step_log_probs = F.log_softmax(step_logits, dim=-1)
            losses.append(-step_log_probs[0, target])
            logits.append(step_logits.detach().cpu())
            top1_ids.append(int(step_logits.argmax(dim=-1).item()))

        loss_tensor = torch.stack(losses)
        answer_nll = float(loss_tensor.mean().item())
        answer_ppl = float(math.exp(min(answer_nll, 20.0)))
        answer_targets = [int(item) for item in case.answer_ids.detach().cpu().tolist()]
        top1_match = [int(pred == target) for pred, target in zip(top1_ids, answer_targets, strict=False)]
        decode_stats = patcher.stats.summary() if patcher is not None else {}
        peak_gb = memory_gb(device)
        del output, past
        return ScoreResult(
            status="ok",
            prefill_ms=prefill_ms,
            decode_ms=decode_elapsed * 1000,
            answer_nll=answer_nll,
            answer_ppl=answer_ppl,
            first_token_correct=top1_match[0] if top1_match else 0,
            all_answer_top1=int(all(top1_match)),
            peak_memory_gb=peak_gb,
            top1_ids=top1_ids,
            answer_logits=torch.cat(logits, dim=0),
            prefill_stats=prefill_stats,
            decode_stats=decode_stats,
        )
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return ScoreResult(status="runtime_error", error=str(exc)[:240])


def compare_answer_logits(full: ScoreResult, sva: ScoreResult) -> dict[str, float]:
    if full.answer_logits is None or sva.answer_logits is None:
        return {
            "answer_kl_to_full": math.nan,
            "answer_top1_agreement": math.nan,
            "answer_logit_cosine": math.nan,
        }
    full_logits = full.answer_logits.float()
    sva_logits = sva.answer_logits.float()
    full_log_probs = F.log_softmax(full_logits, dim=-1)
    sva_log_probs = F.log_softmax(sva_logits, dim=-1)
    full_probs = full_log_probs.exp()
    kl = (full_probs * (full_log_probs - sva_log_probs)).sum(dim=-1)
    agreement = (full_logits.argmax(dim=-1) == sva_logits.argmax(dim=-1)).float()
    cosine = F.cosine_similarity(full_logits, sva_logits, dim=-1)
    return {
        "answer_kl_to_full": float(kl.mean().item()),
        "answer_top1_agreement": float(agreement.mean().item()),
        "answer_logit_cosine": float(cosine.mean().item()),
    }


def stats_value(stats: dict[str, float] | None, key: str) -> float:
    if not stats:
        return math.nan
    return float(stats.get(key, math.nan))


def emit_score(prefix: str, variant: str, case: PromptCase, result: ScoreResult) -> None:
    decode_verified = stats_value(result.decode_stats, "avg_verified")
    reduction = case.context / decode_verified if decode_verified and not math.isnan(decode_verified) else math.nan
    emit(
        prefix,
        {
            "variant": variant,
            "status": result.status,
            "context": case.context,
            "placement": case.placement,
            "key": case.key,
            "answer_tokens": int(case.answer_ids.numel()),
            "prefill_ms": result.prefill_ms,
            "decode_ms": result.decode_ms,
            "answer_nll": result.answer_nll,
            "answer_ppl": result.answer_ppl,
            "first_token_correct": result.first_token_correct,
            "all_answer_top1": result.all_answer_top1,
            "peak_memory_gb": result.peak_memory_gb,
            "prefill_avg_summoned": stats_value(result.prefill_stats, "avg_summoned"),
            "prefill_avg_refill_pool": stats_value(result.prefill_stats, "avg_refill_pool"),
            "prefill_avg_verified": stats_value(result.prefill_stats, "avg_verified"),
            "decode_avg_summoned": stats_value(result.decode_stats, "avg_summoned"),
            "decode_avg_refill_pool": stats_value(result.decode_stats, "avg_refill_pool"),
            "decode_avg_verified": decode_verified,
            "decode_avg_cell_visits": stats_value(result.decode_stats, "avg_cell_visits"),
            "decode_exact_read_reduction": reduction,
            "error": result.error,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Passkey long-context language benchmark for SVA.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--long-artifact-dir", type=Path, default=None)
    parser.add_argument("--long-artifact-min-context", type=int, default=0)
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--teacher-context-max", type=int, default=32768)
    parser.add_argument("--placements", default="start")
    parser.add_argument("--key", default="731942")
    parser.add_argument("--keys", default="")
    parser.add_argument("--shortlist", type=int, default=2048)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--summon-mode", choices=["scan", "inverted", "inverted_static"], default="inverted")
    parser.add_argument("--socket-layers", default="")
    parser.add_argument("--inverted-cells-per-subspace", type=int, default=32)
    parser.add_argument("--adaptive-min-budget", type=int, default=128)
    parser.add_argument("--adaptive-mid-budget", type=int, default=256)
    parser.add_argument("--adaptive-low-margin", type=float, default=0.35)
    parser.add_argument("--adaptive-high-margin", type=float, default=0.70)
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

    contexts = comma_ints(args.contexts)
    placements = comma_strings(args.placements)
    keys = comma_strings(args.keys) if args.keys else [args.key]
    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))

    print("passkey_language_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"artifact_dir,{args.artifact_dir}", flush=True)
    if args.long_artifact_dir is not None:
        print(f"long_artifact_dir,{args.long_artifact_dir}", flush=True)
        print(f"long_artifact_min_context,{args.long_artifact_min_context}", flush=True)
    print(f"contexts,{args.contexts}", flush=True)
    print(f"placements,{args.placements}", flush=True)
    print(f"keys,{','.join(keys)}", flush=True)
    print(f"summon_mode,{args.summon_mode}", flush=True)
    print(f"socket_layers,{format_layer_list(socket_layers)}", flush=True)

    for key in keys:
        for context in contexts:
            for placement in placements:
                case = build_prompt_case(tokenizer, context, key, placement, device)
                full_result: ScoreResult | None = None
                if context <= args.teacher_context_max:
                    full_result = score_answer_decode(model, case, device, patcher=None)
                    emit_score("passkey_language_row", "full", case, full_result)

                selected_artifact_dir = args.artifact_dir
                sva_variant = "sva"
                if args.long_artifact_dir is not None and context >= args.long_artifact_min_context:
                    selected_artifact_dir = args.long_artifact_dir
                    sva_variant = "sva_long"

                patcher = patch_llama_attention(
                    model,
                    selected_artifact_dir,
                    shortlist=args.shortlist,
                    budget=args.budget,
                    assign_chunk_size=args.assign_chunk_size,
                    query_chunk_size=args.query_chunk_size,
                    summon_mode=args.summon_mode,
                    inverted_cells_per_subspace=args.inverted_cells_per_subspace,
                    adaptive_min_budget=args.adaptive_min_budget,
                    adaptive_mid_budget=args.adaptive_mid_budget,
                    adaptive_low_margin=args.adaptive_low_margin,
                    adaptive_high_margin=args.adaptive_high_margin,
                    layers=socket_layers,
                )
                try:
                    sva_result = score_answer_decode(model, case, device, patcher=patcher)
                    emit_score("passkey_language_row", sva_variant, case, sva_result)
                finally:
                    patcher.unpatch()

                if full_result is not None and full_result.status == "ok" and sva_result.status == "ok":
                    comparison = compare_answer_logits(full_result, sva_result)
                    emit(
                        "passkey_language_compare",
                        {
                            "context": context,
                            "placement": placement,
                            "variant": sva_variant,
                            "answer_nll_delta": sva_result.answer_nll - full_result.answer_nll,
                            "prefill_slowdown": sva_result.prefill_ms / max(full_result.prefill_ms, 1e-9),
                            "decode_slowdown": sva_result.decode_ms / max(full_result.decode_ms, 1e-9),
                            **comparison,
                        },
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print("passkey_language_done", flush=True)


if __name__ == "__main__":
    main()
