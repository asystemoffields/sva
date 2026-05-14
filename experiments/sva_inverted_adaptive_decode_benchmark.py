"""8k cached-decode benchmark for inverted-code adaptive SVA."""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class Document:
    doc_id: str
    domain: str
    text: str


EVAL_DOCS = [
    Document(
        "eval_architecture_note",
        "systems",
        (
            "The database migration splits ownership records into a primary table and an audit stream. "
            "The primary table answers current-state queries, while the audit stream preserves every "
            "change with a timestamp, actor id, and reason code. Backfill workers read the old table in "
            "range order, write idempotent events, and pause whenever replica lag exceeds the threshold. "
            "A verification pass then samples accounts across each range and checks that the latest "
            "event reconstructs the visible owner."
        ),
    ),
    Document(
        "eval_algorithm_note",
        "math",
        (
            "The proof sketches a bound for a streaming estimator. Each update modifies a single row "
            "of the sketch, and the estimator returns the median of several independent projections. "
            "The failure probability decreases when the projections are independent, but the memory "
            "cost grows linearly with the number of rows. The useful observation is that heavy entries "
            "need only survive enough projections to dominate the median, not every projection."
        ),
    ),
    Document(
        "eval_patch_plan",
        "code",
        (
            "The patch introduces a small scheduler around the sync worker. The scheduler keeps a heap "
            "of pending tasks, promotes overdue tasks into an immediate queue, and records the last "
            "successful attempt for each account. The migration must preserve existing retry counters. "
            "Tests should cover clock skew, repeated failures, cancellation during shutdown, and a task "
            "that is rescheduled while another worker is already processing the same account."
        ),
    ),
    Document(
        "eval_optics_note",
        "science",
        (
            "The telescope alignment report lists two sources of error. The primary mirror holds its "
            "shape under the expected load, but the detector frame shifts when the enclosure cools. "
            "A short exposure hides the shift because guiding corrections absorb it; a long exposure "
            "shows faint streaking near the edge of the frame. The proposed fix is a temperature-indexed "
            "offset table applied before final image stacking."
        ),
    ),
]


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def repeated_document(doc: Document, repeats: int) -> str:
    return "\n\n".join([f"[doc={doc.doc_id} domain={doc.domain}]", *([doc.text] * max(repeats, 1))])


def encode(tokenizer: Any, text: str, context_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_length)
    return {key: value.to(device) for key, value in batch.items()}


def sync_if_needed(device: torch.device) -> None:
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


def compare_next_logits(full: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    full = full[:, -1, :].float()
    candidate = candidate[:, -1, :].float()
    full_log_probs = F.log_softmax(full, dim=-1)
    candidate_log_probs = F.log_softmax(candidate, dim=-1)
    full_probs = full_log_probs.exp()
    kl = (full_probs * (full_log_probs - candidate_log_probs)).sum(dim=-1).mean()
    return {
        "decode_kl_to_full": float(kl.item()),
        "decode_top1_agreement": float((full.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean().item()),
        "decode_logit_cosine": float(F.cosine_similarity(full, candidate, dim=-1).mean().item()),
        "decode_max_abs_delta": float((full - candidate).abs().max().item()),
    }


@torch.no_grad()
def decode_logits_once(
    model: Any,
    prefix: dict[str, torch.Tensor],
    next_input_id: torch.Tensor,
    decode_attention_mask: torch.Tensor,
    patcher: SVALlamaPatcher | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if patcher is not None:
        patcher.reset_stats()
        patcher.reset_catalogs()
    prefill = model(**prefix, use_cache=True)
    if patcher is not None:
        patcher.reset_stats()
    output = model(
        input_ids=next_input_id,
        attention_mask=decode_attention_mask,
        past_key_values=prefill.past_key_values,
        use_cache=True,
    )
    stats = patcher.stats.summary() if patcher is not None else {}
    return output.logits, stats


@torch.no_grad()
def time_decode(
    model: Any,
    prefix: dict[str, torch.Tensor],
    next_input_id: torch.Tensor,
    decode_attention_mask: torch.Tensor,
    device: torch.device,
    repeats: int,
    warmup: int,
    patcher: SVALlamaPatcher | None = None,
) -> tuple[float, dict[str, float]]:
    def prefill_once() -> Any:
        if patcher is not None:
            patcher.reset_stats()
            patcher.reset_catalogs()
        prefill_output = model(**prefix, use_cache=True)
        if patcher is not None:
            patcher.reset_stats()
        return prefill_output

    def decode_once(prefill_output: Any) -> dict[str, float]:
        _ = model(
            input_ids=next_input_id,
            attention_mask=decode_attention_mask,
            past_key_values=prefill_output.past_key_values,
            use_cache=True,
        )
        return patcher.stats.summary() if patcher is not None else {}

    for _ in range(warmup):
        _ = decode_once(prefill_once())
    sync_if_needed(device)
    elapsed = 0.0
    stats: dict[str, float] = {}
    for _ in range(repeats):
        prefill_output = prefill_once()
        sync_if_needed(device)
        start = time.perf_counter()
        stats = decode_once(prefill_output)
        sync_if_needed(device)
        elapsed += time.perf_counter() - start
    return elapsed / max(repeats, 1), stats


def split_decode_batch(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    prefix = {
        "input_ids": batch["input_ids"][:, :-1],
        "attention_mask": batch["attention_mask"][:, :-1],
    }
    next_input_id = batch["input_ids"][:, -1:]
    decode_attention_mask = batch["attention_mask"]
    return prefix, next_input_id, decode_attention_mask


def work_proxy(
    model: Any,
    key_len: int,
    stats: dict[str, float],
    rank_dim: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    mode: str,
    shortlist: int,
) -> dict[str, float]:
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    avg_verified = float(stats.get("avg_verified", key_len))
    avg_summoned = float(stats.get("avg_summoned", key_len))
    avg_cell_visits = float(stats.get("avg_cell_visits", 0.0))
    full_proxy = float(key_len * head_dim)
    if mode == "full":
        method_proxy = full_proxy
    elif mode == "scan":
        method_proxy = float(key_len * coarse_subspaces + min(shortlist, key_len) * rank_dim + avg_verified * head_dim)
    else:
        method_proxy = float(coarse_codewords * rank_dim + avg_summoned * rank_dim + avg_verified * head_dim)
    return {
        "avg_summoned": avg_summoned,
        "avg_verified": avg_verified,
        "avg_cell_visits": avg_cell_visits,
        "exact_score_reduction": key_len / max(avg_verified, 1e-9),
        "value_read_reduction": key_len / max(avg_verified, 1e-9),
        "method_compute_proxy": method_proxy,
        "method_compute_proxy_ratio": full_proxy / max(method_proxy, 1e-9),
    }


def weighted_average(rows: list[dict[str, float | str]], key: str) -> float:
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        if key not in row:
            continue
        weight = float(row["tokens"])
        total += float(row[key]) * weight
        weight_sum += weight
    return total / max(weight_sum, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8k cached-decode SVA speed variants.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--eval-doc-limit", type=int, default=2)
    parser.add_argument("--shortlist", type=int, default=2048)
    parser.add_argument("--max-budget", type=int, default=512)
    parser.add_argument("--scan-budget", type=int, default=512)
    parser.add_argument("--adaptive-min-budgets", default="64,128")
    parser.add_argument("--adaptive-mid-budget", type=int, default=256)
    parser.add_argument("--cells-per-subspace", default="4,8,16")
    parser.add_argument("--adaptive-low-margin", type=float, default=0.35)
    parser.add_argument("--adaptive-high-margin", type=float, default=0.70)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    adaptive_min_budgets = comma_ints(args.adaptive_min_budgets)
    cells_values = comma_ints(args.cells_per_subspace)
    print("inverted_adaptive_decode_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"attn_implementation,{args.attn_implementation}", flush=True)

    rows: list[dict[str, float | str]] = []
    for doc in EVAL_DOCS[: args.eval_doc_limit]:
        batch = encode(tokenizer, repeated_document(doc, args.eval_repeats), args.context_length, device)
        prefix, next_input_id, decode_attention_mask = split_decode_batch(batch)
        key_len = int(batch["input_ids"].shape[1])
        tokens = int(batch["attention_mask"].sum().item())

        full_logits, _ = decode_logits_once(model, prefix, next_input_id, decode_attention_mask)
        full_seconds, _ = time_decode(
            model,
            prefix,
            next_input_id,
            decode_attention_mask,
            device,
            args.repeats,
            args.warmup,
        )
        full_row: dict[str, float | str] = {
            "variant": "full",
            "doc_id": doc.doc_id,
            "domain": doc.domain,
            "context_len": key_len,
            "tokens": tokens,
            "decode_ms": full_seconds * 1000.0,
            "decode_speedup_vs_full": 1.0,
            "decode_kl_to_full": 0.0,
            "decode_top1_agreement": 1.0,
            "decode_logit_cosine": 1.0,
            "decode_max_abs_delta": 0.0,
            **work_proxy(model, key_len, {}, 64, 2, 256, "full", args.shortlist),
        }
        rows.append(full_row)
        emit("inverted_adaptive_row", full_row)

        scan = patch_llama_attention(
            model,
            args.artifact_dir,
            shortlist=args.shortlist,
            budget=args.scan_budget,
            summon_mode="scan",
        )
        scan_logits, scan_stats = decode_logits_once(model, prefix, next_input_id, decode_attention_mask, scan)
        scan_seconds, scan_stats = time_decode(
            model,
            prefix,
            next_input_id,
            decode_attention_mask,
            device,
            args.repeats,
            args.warmup,
            scan,
        )
        manifest = scan.bundle.manifest
        scan_row = {
            "variant": "scan",
            "doc_id": doc.doc_id,
            "domain": doc.domain,
            "context_len": key_len,
            "tokens": tokens,
            "shortlist": args.shortlist,
            "budget": args.scan_budget,
            "decode_ms": scan_seconds * 1000.0,
            "decode_speedup_vs_full": full_seconds / max(scan_seconds, 1e-9),
            **compare_next_logits(full_logits, scan_logits),
            **work_proxy(
                model,
                key_len,
                scan_stats,
                int(manifest["rank_dim"]),
                int(manifest["coarse_subspaces"]),
                int(manifest["coarse_codewords"]),
                "scan",
                args.shortlist,
            ),
        }
        rows.append(scan_row)
        emit("inverted_adaptive_row", scan_row)
        scan.unpatch()

        for cells in cells_values:
            for min_budget in adaptive_min_budgets:
                inverted = patch_llama_attention(
                    model,
                    args.artifact_dir,
                    shortlist=args.shortlist,
                    budget=args.max_budget,
                    summon_mode="inverted",
                    inverted_cells_per_subspace=cells,
                    adaptive_min_budget=min_budget,
                    adaptive_mid_budget=args.adaptive_mid_budget,
                    adaptive_low_margin=args.adaptive_low_margin,
                    adaptive_high_margin=args.adaptive_high_margin,
                )
                inv_logits, inv_stats = decode_logits_once(model, prefix, next_input_id, decode_attention_mask, inverted)
                inv_seconds, inv_stats = time_decode(
                    model,
                    prefix,
                    next_input_id,
                    decode_attention_mask,
                    device,
                    args.repeats,
                    args.warmup,
                    inverted,
                )
                inv_row = {
                    "variant": "inverted_adaptive",
                    "doc_id": doc.doc_id,
                    "domain": doc.domain,
                    "context_len": key_len,
                    "tokens": tokens,
                    "cells_per_subspace": cells,
                    "min_budget": min_budget,
                    "mid_budget": args.adaptive_mid_budget,
                    "max_budget": args.max_budget,
                    "decode_ms": inv_seconds * 1000.0,
                    "decode_speedup_vs_full": full_seconds / max(inv_seconds, 1e-9),
                    **compare_next_logits(full_logits, inv_logits),
                    **work_proxy(
                        model,
                        key_len,
                        inv_stats,
                        int(manifest["rank_dim"]),
                        int(manifest["coarse_subspaces"]),
                        int(manifest["coarse_codewords"]),
                        "inverted",
                        args.shortlist,
                    ),
                }
                rows.append(inv_row)
                emit("inverted_adaptive_row", inv_row)
                inverted.unpatch()

    for variant in sorted(set(str(row["variant"]) for row in rows)):
        group = [row for row in rows if row["variant"] == variant]
        extra_keys = ["cells_per_subspace", "min_budget"]
        groups: dict[tuple[str, str], list[dict[str, float | str]]] = {}
        for row in group:
            key = tuple(str(row.get(item, "")) for item in extra_keys)
            groups.setdefault(key, []).append(row)
        for key, subgroup in groups.items():
            summary: dict[str, float | str] = {
                "variant": variant,
                "cells_per_subspace": key[0],
                "min_budget": key[1],
                "docs": len(subgroup),
                "tokens": sum(int(row["tokens"]) for row in subgroup),
                "decode_ms": weighted_average(subgroup, "decode_ms"),
                "decode_speedup_vs_full": weighted_average(subgroup, "decode_speedup_vs_full"),
                "decode_kl_to_full": weighted_average(subgroup, "decode_kl_to_full"),
                "decode_top1_agreement": weighted_average(subgroup, "decode_top1_agreement"),
                "decode_logit_cosine": weighted_average(subgroup, "decode_logit_cosine"),
                "decode_max_abs_delta": weighted_average(subgroup, "decode_max_abs_delta"),
                "avg_summoned": weighted_average(subgroup, "avg_summoned"),
                "avg_verified": weighted_average(subgroup, "avg_verified"),
                "avg_cell_visits": weighted_average(subgroup, "avg_cell_visits"),
                "exact_score_reduction": weighted_average(subgroup, "exact_score_reduction"),
                "value_read_reduction": weighted_average(subgroup, "value_read_reduction"),
                "method_compute_proxy_ratio": weighted_average(subgroup, "method_compute_proxy_ratio"),
            }
            emit("inverted_adaptive_summary", summary)
    print("inverted_adaptive_decode_done", flush=True)


if __name__ == "__main__":
    main()
