"""8k head-to-head benchmark for full attention vs production SVA.

This benchmark uses the exported SVA artifact and the production adapter in
`sva/`. It reports behavior agreement, prefill timing, cached-decode timing,
and method-level work proxies.
"""

from __future__ import annotations

import argparse
import json
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


def repeated_document(doc: Document, repeats: int) -> str:
    header = f"[doc={doc.doc_id} domain={doc.domain}]"
    return "\n\n".join([header, *([doc.text] * max(repeats, 1))])


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


def encode(tokenizer: Any, text: str, context_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_length)
    return {key: value.to(device) for key, value in batch.items()}


def shifted_loss_chunked(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    total_loss = torch.zeros((), device=logits.device, dtype=torch.float64)
    total_items = 0
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()
    seq_len = shift_logits.shape[1]
    for start in range(0, seq_len, chunk_tokens):
        end = min(start + chunk_tokens, seq_len)
        valid = shift_mask[:, start:end].reshape(-1)
        if not valid.any():
            continue
        chunk_logits = shift_logits[:, start:end, :].reshape(-1, shift_logits.shape[-1])[valid]
        chunk_labels = shift_labels[:, start:end].reshape(-1)[valid]
        loss_sum = F.cross_entropy(chunk_logits.float(), chunk_labels, reduction="sum")
        total_loss += loss_sum.double()
        total_items += int(valid.sum().item())
    return (total_loss / max(total_items, 1)).float()


def compare_logits_chunked(
    full_logits: torch.Tensor,
    sva_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_tokens: int,
) -> dict[str, float]:
    valid_mask = attention_mask[:, 1:].bool()
    full_shift = full_logits[:, :-1, :]
    sva_shift = sva_logits[:, :-1, :]
    seq_len = full_shift.shape[1]
    kl_sum = 0.0
    agree_sum = 0.0
    cos_sum = 0.0
    total = 0
    for start in range(0, seq_len, chunk_tokens):
        end = min(start + chunk_tokens, seq_len)
        valid = valid_mask[:, start:end].reshape(-1)
        if not valid.any():
            continue
        full = full_shift[:, start:end, :].reshape(-1, full_shift.shape[-1])[valid].float()
        sva = sva_shift[:, start:end, :].reshape(-1, sva_shift.shape[-1])[valid].float()
        full_log_probs = F.log_softmax(full, dim=-1)
        sva_log_probs = F.log_softmax(sva, dim=-1)
        full_probs = full_log_probs.exp()
        kl = (full_probs * (full_log_probs - sva_log_probs)).sum(dim=-1)
        cosine = F.cosine_similarity(full, sva, dim=-1)
        agree = full.argmax(dim=-1) == sva.argmax(dim=-1)
        count = int(full.shape[0])
        kl_sum += float(kl.sum().item())
        cos_sum += float(cosine.sum().item())
        agree_sum += float(agree.float().sum().item())
        total += count
    denom = max(total, 1)
    return {
        "kl_to_full": kl_sum / denom,
        "top1_agreement": agree_sum / denom,
        "logit_cosine": cos_sum / denom,
    }


@torch.no_grad()
def full_logits(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    return model(**batch, use_cache=False).logits


@torch.no_grad()
def sva_logits(model: Any, patcher: SVALlamaPatcher, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    patcher.reset_stats()
    patcher.reset_catalogs()
    model.eval()
    return model(**batch, use_cache=False).logits


@torch.no_grad()
def time_prefill(
    model: Any,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    repeats: int,
    warmup: int,
    patcher: SVALlamaPatcher | None = None,
) -> float:
    for _ in range(warmup):
        if patcher is not None:
            patcher.reset_catalogs()
        _ = model(**batch, use_cache=True)
    sync_if_needed(device)
    elapsed = 0.0
    for _ in range(repeats):
        if patcher is not None:
            patcher.reset_catalogs()
        sync_if_needed(device)
        start = time.perf_counter()
        _ = model(**batch, use_cache=True)
        sync_if_needed(device)
        elapsed += time.perf_counter() - start
    return elapsed / max(repeats, 1)


@torch.no_grad()
def time_decode_step(
    model: Any,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    repeats: int,
    warmup: int,
    patcher: SVALlamaPatcher | None = None,
) -> tuple[float, dict[str, float]]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    next_id = input_ids[:, -1:]
    decode_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)

    def prefill_cache() -> Any:
        if patcher is not None:
            patcher.reset_stats()
            patcher.reset_catalogs()
        prefill = model(**batch, use_cache=True)
        if patcher is not None:
            patcher.reset_stats()
        return prefill

    def decode_once(prefill: Any) -> dict[str, float]:
        _ = model(
            input_ids=next_id,
            attention_mask=decode_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        return patcher.stats.summary() if patcher is not None else {}

    for _ in range(warmup):
        _ = decode_once(prefill_cache())
    sync_if_needed(device)
    elapsed = 0.0
    stats: dict[str, float] = {}
    for _ in range(repeats):
        prefill = prefill_cache()
        sync_if_needed(device)
        start = time.perf_counter()
        stats = decode_once(prefill)
        sync_if_needed(device)
        elapsed += time.perf_counter() - start
    return elapsed / max(repeats, 1), stats


def work_proxies(
    model: Any,
    context_len: int,
    prefill_stats: dict[str, float],
    decode_stats: dict[str, float],
    shortlist: int,
    budget: int,
    coarse_subspaces: int,
    rank_dim: int,
) -> dict[str, float]:
    config = model.config
    head_dim = int(config.hidden_size // config.num_attention_heads)
    full_prefill_avg_keys = (context_len + 1) / 2.0
    full_decode_keys = float(context_len)
    sva_prefill_verified = float(prefill_stats.get("avg_verified", float("nan")))
    sva_decode_verified = float(decode_stats.get("avg_verified", float("nan")))
    actual_shortlist = min(shortlist, context_len)
    actual_budget = min(budget, actual_shortlist)
    full_decode_compute_proxy = full_decode_keys * head_dim
    sva_decode_compute_proxy = (
        full_decode_keys * coarse_subspaces
        + actual_shortlist * rank_dim
        + min(sva_decode_verified, actual_budget) * head_dim
    )
    return {
        "full_prefill_avg_exact_scores": full_prefill_avg_keys,
        "sva_prefill_avg_exact_scores": sva_prefill_verified,
        "prefill_exact_score_reduction": full_prefill_avg_keys / max(sva_prefill_verified, 1e-9),
        "full_decode_exact_scores": full_decode_keys,
        "sva_decode_exact_scores": sva_decode_verified,
        "decode_exact_score_reduction": full_decode_keys / max(sva_decode_verified, 1e-9),
        "full_decode_value_reads": full_decode_keys,
        "sva_decode_value_reads": sva_decode_verified,
        "decode_value_read_reduction": full_decode_keys / max(sva_decode_verified, 1e-9),
        "full_decode_compute_proxy": full_decode_compute_proxy,
        "sva_decode_compute_proxy": sva_decode_compute_proxy,
        "decode_compute_proxy_ratio": full_decode_compute_proxy / max(sva_decode_compute_proxy, 1e-9),
    }


def weighted_average(rows: list[dict[str, float | str]], key: str) -> float:
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        weight = float(row["tokens"])
        total += float(value) * weight
        weight_sum += weight
    return total / max(weight_sum, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="8k full-attention vs production-SVA head-to-head.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--eval-doc-limit", type=int, default=2)
    parser.add_argument("--shortlist", type=int, default=2048)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--decode-repeats", type=int, default=5)
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
    patcher = patch_llama_attention(model, args.artifact_dir, shortlist=args.shortlist, budget=args.budget)

    manifest = patcher.bundle.manifest
    print("head_to_head_8k_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"attn_implementation,{args.attn_implementation}", flush=True)
    print(f"artifact,{json.dumps(manifest, sort_keys=True)}", flush=True)

    rows: list[dict[str, float | str]] = []
    docs = EVAL_DOCS[: args.eval_doc_limit]
    for doc in docs:
        text = repeated_document(doc, args.eval_repeats)
        batch = encode(tokenizer, text, args.context_length, device)
        tokens = int(batch["attention_mask"][:, 1:].sum().item())
        context_len = int(batch["input_ids"].shape[1])

        patcher.unpatch()
        full = full_logits(model, batch)
        full_loss = shifted_loss_chunked(full, batch["input_ids"], batch["attention_mask"], args.chunk_tokens)
        full_prefill_seconds = time_prefill(model, batch, device, args.timing_repeats, args.warmup)
        full_decode_seconds, _ = time_decode_step(model, batch, device, args.decode_repeats, args.warmup)

        patcher.patch()
        sva = sva_logits(model, patcher, batch)
        sva_prefill_stats = patcher.stats.summary()
        sva_loss = shifted_loss_chunked(sva, batch["input_ids"], batch["attention_mask"], args.chunk_tokens)
        logit_metrics = compare_logits_chunked(full, sva, batch["attention_mask"], args.chunk_tokens)
        sva_prefill_seconds = time_prefill(model, batch, device, args.timing_repeats, args.warmup, patcher)
        sva_decode_seconds, sva_decode_stats = time_decode_step(
            model,
            batch,
            device,
            args.decode_repeats,
            args.warmup,
            patcher,
        )

        proxies = work_proxies(
            model,
            context_len,
            sva_prefill_stats,
            sva_decode_stats,
            args.shortlist,
            args.budget,
            int(manifest["coarse_subspaces"]),
            int(manifest["rank_dim"]),
        )
        row: dict[str, float | str] = {
            "doc_id": doc.doc_id,
            "domain": doc.domain,
            "context_len": context_len,
            "tokens": tokens,
            "shortlist": args.shortlist,
            "budget": args.budget,
            "full_loss": float(full_loss.item()),
            "sva_loss": float(sva_loss.item()),
            "loss_delta": float((sva_loss - full_loss).item()),
            **logit_metrics,
            "full_prefill_ms": full_prefill_seconds * 1000.0,
            "sva_prefill_ms": sva_prefill_seconds * 1000.0,
            "prefill_speedup": full_prefill_seconds / max(sva_prefill_seconds, 1e-9),
            "full_decode_ms": full_decode_seconds * 1000.0,
            "sva_decode_ms": sva_decode_seconds * 1000.0,
            "decode_speedup": full_decode_seconds / max(sva_decode_seconds, 1e-9),
            "sva_prefill_avg_summoned": float(sva_prefill_stats.get("avg_summoned", float("nan"))),
            "sva_prefill_avg_verified": float(sva_prefill_stats.get("avg_verified", float("nan"))),
            "sva_decode_avg_verified": float(sva_decode_stats.get("avg_verified", float("nan"))),
            **proxies,
        }
        rows.append(row)
        emit("head_to_head_row", row)

        del full, sva
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_keys = [
        "loss_delta",
        "kl_to_full",
        "top1_agreement",
        "logit_cosine",
        "full_prefill_ms",
        "sva_prefill_ms",
        "prefill_speedup",
        "full_decode_ms",
        "sva_decode_ms",
        "decode_speedup",
        "sva_prefill_avg_verified",
        "sva_decode_avg_verified",
        "decode_exact_score_reduction",
        "decode_value_read_reduction",
        "decode_compute_proxy_ratio",
    ]
    summary: dict[str, float | int] = {
        "docs": len(rows),
        "tokens": sum(int(row["tokens"]) for row in rows),
        "context_len": args.context_length,
        "shortlist": args.shortlist,
        "budget": args.budget,
    }
    for key in summary_keys:
        summary[key] = weighted_average(rows, key)
    emit("head_to_head_summary", summary)
    print("head_to_head_8k_done", flush=True)


if __name__ == "__main__":
    main()
