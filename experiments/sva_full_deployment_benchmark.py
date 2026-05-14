"""Full deployment benchmark for socketed Summon-Verify Attention.

The benchmark builds SVA artifacts on calibration text, freezes them, and then
compares a socketed model against full attention on held-out documents. It is
meant to answer the deployment question rather than the transductive harness
question: can one calibrated lookup system serve new text without fitting on
the eval sequence?
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from sva_pretrained_socket_test import (
    SVAConfig,
    SVALlamaAttention,
    build_artifacts_for_hidden_states,
    build_progressive_three_stage_artifacts,
    compare_logits,
    config_from_args,
    encode_batch,
    format_layer_list,
    make_nested_stats,
    make_stats,
    parse_layer_list,
    run_model,
    shifted_loss,
)


@dataclass(frozen=True)
class Document:
    doc_id: str
    domain: str
    text: str


CALIBRATION_DOCS = [
    Document(
        "calib_systems_memo",
        "systems",
        (
            "The service keeps a rolling index of session records, each record carrying a user id, "
            "a monotonic event number, and a compact summary of the last committed action. During "
            "normal traffic the writer batches small updates, but the reader must answer point "
            "queries immediately. The design note proposes a two-level cache: a tiny resident table "
            "for hot sessions and a larger append-only segment for recent history. Every recovery "
            "step must preserve the order of writes because later reconciliation depends on the "
            "exact sequence of state transitions."
        ),
    ),
    Document(
        "calib_policy_memo",
        "policy",
        (
            "The procurement policy separates selection criteria from approval authority. A team may "
            "rank vendors by latency, retention guarantees, audit access, and total cost, but the "
            "final approval requires a signed exception when a higher-cost vendor is chosen. The "
            "memo also requires each contract to name a data deletion window, an incident contact, "
            "and a renewal trigger. Missing renewal triggers are treated as operational risk rather "
            "than legal risk because the failure usually appears first as an unplanned service gap."
        ),
    ),
    Document(
        "calib_research_note",
        "science",
        (
            "The experiment compares two electrolyte additives under identical charge cycles. The "
            "first additive lowers impedance during the initial formation step, while the second "
            "keeps impedance stable after repeated thermal exposure. The lab notebook marks three "
            "readings as suspect because the chamber temperature drifted during the measurement. "
            "The conclusion is deliberately narrow: the second additive is more stable in this cell "
            "geometry, but the result does not yet separate interface chemistry from separator wear."
        ),
    ),
    Document(
        "calib_code_review",
        "code",
        (
            "The parser patch moves token normalization before scope construction. That makes every "
            "identifier lookup use the same canonical spelling, but it also changes error recovery "
            "because malformed imports now fail before the scope table exists. The review asks for "
            "one extra fixture with mixed case imports, one fixture with a missing closing brace, "
            "and a benchmark that counts allocations inside the scanner loop. The expected behavior "
            "is stable names, unchanged diagnostics, and lower temporary string churn."
        ),
    ),
    Document(
        "calib_incident_report",
        "operations",
        (
            "At 09:41 the queue consumer began replaying acknowledged messages after a leader change. "
            "The immediate symptom was duplicate invoice emails, but the root cause was a stale lease "
            "record that survived the failover. The mitigation disabled automatic replay for the "
            "affected shard, drained the queue through a single worker, and compared every emitted "
            "invoice id against the payment ledger. The postmortem assigns follow-up work to lease "
            "expiry tests, replay idempotence checks, and dashboard alerts for duplicate output."
        ),
    ),
    Document(
        "calib_design_dialogue",
        "dialogue",
        (
            "Mira asked whether the dashboard should show every diagnostic or only the ones that "
            "changed since the last deployment. Rowan argued for a compact default view with a drill "
            "down for each subsystem. They agreed that the first screen should answer three questions: "
            "is the release healthy, which service changed, and what evidence supports that judgment. "
            "The follow-up task is to move low-priority counters behind an expandable section while "
            "keeping the rollback button visible."
        ),
    ),
]


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
        "eval_editorial_scene",
        "narrative",
        (
            "The station archive closed at midnight, but the copy desk still glowed behind frosted "
            "glass. Lena carried a folder of corrected captions to the night editor and waited while "
            "he compared each caption against the printed plates. One line described a public square "
            "before the renovation; another named a minister by an outdated title. The editor marked "
            "both changes in blue pencil, then placed the folder beside the morning edition."
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
        "eval_support_ticket",
        "support",
        (
            "The customer reports that exported invoices are missing regional tax labels for two "
            "subsidiaries. The raw totals are correct, and the labels appear inside the web view, so "
            "the likely fault is in the PDF template rather than the tax calculation service. The "
            "support note asks engineering to inspect the locale mapping for the export path and to "
            "confirm whether the affected subsidiaries share the same billing profile."
        ),
    ),
    Document(
        "eval_compliance_brief",
        "policy",
        (
            "The compliance brief requires every analytics job to declare its input tables, retention "
            "period, and deletion path. Jobs that join customer activity with account metadata must "
            "also record the business purpose and the approving team. The review board accepts cached "
            "aggregates when the source table expires, but only if the aggregate cannot be reversed "
            "into individual activity records. The next audit will sample jobs created after April."
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
    Document(
        "eval_planning_note",
        "operations",
        (
            "The launch plan moves documentation review before the release branch is cut. Product will "
            "sign off on screenshots by Tuesday, support will prepare the customer note by Wednesday, "
            "and engineering will hold the final deploy window on Thursday afternoon. The checklist "
            "keeps two contingency slots open: one for translation changes and one for a rollback drill "
            "if the staging environment reports a new warning."
        ),
    ),
]


def comma_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def load_documents(path: str | None, defaults: list[Document], limit: int | None, prefix: str) -> list[Document]:
    if path is None:
        docs = defaults
    else:
        file_path = Path(path)
        raw = file_path.read_text(encoding="utf-8")
        docs = []
        if file_path.suffix.lower() == ".jsonl":
            for index, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    docs.append(Document(f"{prefix}_{index}", "external", item))
                else:
                    docs.append(
                        Document(
                            str(item.get("id", f"{prefix}_{index}")),
                            str(item.get("domain", "external")),
                            str(item["text"]),
                        )
                    )
        else:
            parts = [part.strip() for part in raw.split("\n---\n") if part.strip()]
            if not parts and raw.strip():
                parts = [raw.strip()]
            docs = [Document(f"{prefix}_{index}", "external", text) for index, text in enumerate(parts)]

    if limit is not None:
        docs = docs[:limit]
    if not docs:
        raise ValueError(f"No documents available for {prefix}.")
    return docs


def repeated_document(doc: Document, repeats: int) -> str:
    header = f"[doc={doc.doc_id} domain={doc.domain}]"
    return "\n\n".join([header, *([doc.text] * max(repeats, 1))])


def calibration_stream(docs: list[Document], repeats: int) -> str:
    sections = []
    for _ in range(max(repeats, 1)):
        sections.extend(f"[doc={doc.doc_id} domain={doc.domain}]\n{doc.text}" for doc in docs)
    return "\n\n".join(sections)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def timed_logits(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device, repeats: int) -> tuple[torch.Tensor, float]:
    elapsed = 0.0
    logits = None
    for _ in range(max(repeats, 1)):
        sync_if_needed(device)
        start = time.perf_counter()
        logits = run_model(model, batch)
        sync_if_needed(device)
        elapsed += time.perf_counter() - start
    assert logits is not None
    return logits, elapsed / max(repeats, 1)


def patch_model_reversible(model: nn.Module, cfg: SVAConfig, layer_indices: list[int] | None) -> dict[int, nn.Module]:
    selected = None if layer_indices is None else set(layer_indices)
    originals = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if selected is None or layer_idx in selected:
            originals[layer_idx] = layer.self_attn
            layer.self_attn = SVALlamaAttention(layer.self_attn, cfg)
    return originals


def restore_model(model: nn.Module, originals: dict[int, nn.Module]) -> None:
    for layer_idx, original in originals.items():
        model.model.layers[layer_idx].self_attn = original


def ratio(stats: dict[str, float], numerator: str, denominator: str) -> float:
    total = stats.get(denominator, 0.0)
    if total <= 0:
        return float("nan")
    return stats.get(numerator, 0.0) / total


def stat_metrics(cfg: SVAConfig, diagnose_topk: int) -> dict[str, float]:
    metrics = {
        "avg_summoned": ratio(cfg.stats, "summoned", "queries"),
        "avg_exact_scored": ratio(cfg.stats, "exact_scored", "queries"),
        "avg_verified": ratio(cfg.stats, "verified", "queries"),
    }
    if diagnose_topk > 0:
        metrics[f"candidate_top{diagnose_topk}_recall"] = ratio(cfg.stats, "candidate_topk_hits", "topk_items")
        metrics[f"exact_top{diagnose_topk}_recall"] = ratio(cfg.stats, "exact_topk_hits", "topk_items")
        metrics[f"verified_top{diagnose_topk}_recall"] = ratio(cfg.stats, "verified_topk_hits", "topk_items")
    return metrics


def make_eval_batches(
    tokenizer,
    docs: list[Document],
    repeats: int,
    max_length: int,
    device: torch.device,
) -> Iterable[tuple[Document, dict[str, torch.Tensor]]]:
    for doc in docs:
        yield doc, encode_batch(tokenizer, [repeated_document(doc, repeats)], max_length, device)


def weighted_average(rows: list[dict[str, str]], key: str) -> float:
    total_weight = 0.0
    total = 0.0
    for row in rows:
        value = row.get(key)
        if value is None or value == "nan":
            continue
        weight = float(row["tokens"])
        total += float(value) * weight
        total_weight += weight
    return total / total_weight if total_weight else float("nan")


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def emit_row(prefix: str, row: dict[str, str]) -> None:
    ordered = ",".join(f"{key}={value}" for key, value in row.items())
    print(f"{prefix},{ordered}", flush=True)


def write_csv(path: str | None, rows: list[dict[str, str]]) -> None:
    if path is None or not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out deployment benchmark for pretrained SVA sockets.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--calibration-doc-limit", type=int, default=None)
    parser.add_argument("--eval-doc-limit", type=int, default=None)
    parser.add_argument("--calibration-repeats", type=int, default=80)
    parser.add_argument("--eval-repeats", type=int, default=80)
    parser.add_argument("--calibration-length", type=int, default=0)
    parser.add_argument("--context-lengths", default="2048")
    parser.add_argument("--allow-beyond-model-context", action="store_true")
    parser.add_argument(
        "--socket-layers",
        default="",
        help="Comma-separated layers or ranges to replace. Empty means all layers.",
    )
    parser.add_argument("--route-source", choices=["qk", "hidden"], default="qk")
    parser.add_argument("--artifact-training", choices=["teacher", "progressive"], default="teacher")
    parser.add_argument("--tables", type=int, default=16)
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--budgets", default="256,512")
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prefilter-dim", type=int, default=0)
    parser.add_argument("--prefilter-budget", type=int, default=0)
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--coarse-subspaces", type=int, default=4)
    parser.add_argument("--coarse-codewords", type=int, default=64)
    parser.add_argument("--coarse-shortlists", default="512,1024")
    parser.add_argument("--coarse-label-topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--ranker-train-steps", type=int, default=160)
    parser.add_argument("--coarse-hard-steps", type=int, default=80)
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
    parser.add_argument("--diagnose-topk", type=int, default=16)
    parser.add_argument("--head-report-limit", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    args.mode = "three_stage"

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

    context_lengths = comma_ints(args.context_lengths)
    coarse_shortlists = comma_ints(args.coarse_shortlists)
    budgets = comma_ints(args.budgets)
    max_context = max(context_lengths)
    calibration_length = args.calibration_length or max_context

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    model_context = getattr(model.config, "max_position_embeddings", None)
    if (
        model_context is not None
        and max(max_context, calibration_length) > int(model_context)
        and not args.allow_beyond_model_context
    ):
        raise ValueError(
            f"Requested context {max(max_context, calibration_length)} exceeds model max_position_embeddings "
            f"{model_context}. Pass --allow-beyond-model-context to override."
        )

    calibration_docs = load_documents(args.calibration_file, CALIBRATION_DOCS, args.calibration_doc_limit, "calibration")
    eval_docs = load_documents(args.eval_file, EVAL_DOCS, args.eval_doc_limit, "eval")
    calibration_text = calibration_stream(calibration_docs, args.calibration_repeats)
    calibration_batch = encode_batch(tokenizer, [calibration_text], calibration_length, device)

    socket_layers = parse_layer_list(args.socket_layers, len(model.model.layers))
    socket_layer_count = len(socket_layers) if socket_layers is not None else len(model.model.layers)

    print("deployment_benchmark_start")
    print(f"model_id,{args.model_id}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"calibration_docs,{len(calibration_docs)}")
    print(f"eval_docs,{len(eval_docs)}")
    print(f"calibration_seq_len,{calibration_batch['input_ids'].shape[1]}")
    print(f"context_lengths,{';'.join(str(value) for value in context_lengths)}")
    print(f"socket_layers,{format_layer_list(socket_layers)}")
    print(f"socket_layer_count,{socket_layer_count}")
    print(f"artifact_training,{args.artifact_training}")
    print(f"route_source,{args.route_source}")

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

    rows: list[dict[str, str]] = []
    for context_len in context_lengths:
        for doc, batch in make_eval_batches(tokenizer, eval_docs, args.eval_repeats, context_len, device):
            tokens = int(batch["attention_mask"][:, 1:].sum().item())
            full_logits, full_seconds = timed_logits(model, batch, device, args.timing_repeats)
            full_loss = shifted_loss(full_logits, batch["input_ids"], batch["attention_mask"])

            for shortlist in coarse_shortlists:
                for budget in budgets:
                    if budget > shortlist:
                        continue
                    args.coarse_shortlist = shortlist
                    args.budget = budget
                    cfg = config_from_args(args, artifacts)
                    cfg.stats = make_stats()
                    cfg.layer_stats = make_nested_stats()
                    cfg.head_stats = make_nested_stats()

                    originals = patch_model_reversible(model, cfg, socket_layers)
                    try:
                        sva_logits, sva_seconds = timed_logits(model, batch, device, args.timing_repeats)
                    finally:
                        restore_model(model, originals)

                    sva_loss = shifted_loss(sva_logits, batch["input_ids"], batch["attention_mask"])
                    logit_metrics = compare_logits(full_logits, sva_logits, batch["attention_mask"])
                    stats = stat_metrics(cfg, args.diagnose_topk)
                    row = {
                        "model_id": args.model_id,
                        "artifact_training": args.artifact_training,
                        "route_source": args.route_source,
                        "socket_layers": format_layer_list(socket_layers).replace(",", ";"),
                        "socket_layer_count": str(socket_layer_count),
                        "context_len": str(context_len),
                        "calibration_len": str(calibration_batch["input_ids"].shape[1]),
                        "doc_id": doc.doc_id,
                        "domain": doc.domain,
                        "tokens": str(tokens),
                        "coarse_shortlist": str(shortlist),
                        "budget": str(budget),
                        "rank_dim": str(args.rank_dim),
                        "coarse_subspaces": str(args.coarse_subspaces),
                        "coarse_codewords": str(args.coarse_codewords),
                        "full_loss": row_value(float(full_loss.item())),
                        "sva_loss": row_value(float(sva_loss.item())),
                        "loss_delta": row_value(float((sva_loss - full_loss).item())),
                        "kl_to_full": row_value(logit_metrics["kl_to_full"]),
                        "top1_agreement": row_value(logit_metrics["top1_agreement"]),
                        "logit_cosine": row_value(logit_metrics["logit_cosine"]),
                        "avg_summoned": row_value(stats["avg_summoned"]),
                        "avg_exact_scored": row_value(stats["avg_exact_scored"]),
                        "avg_verified": row_value(stats["avg_verified"]),
                        "full_ms": row_value(full_seconds * 1000.0),
                        "sva_ms": row_value(sva_seconds * 1000.0),
                        "speedup": row_value(full_seconds / sva_seconds if sva_seconds > 0 else float("nan")),
                    }
                    if args.diagnose_topk > 0:
                        row[f"candidate_top{args.diagnose_topk}_recall"] = row_value(
                            stats[f"candidate_top{args.diagnose_topk}_recall"]
                        )
                        row[f"exact_top{args.diagnose_topk}_recall"] = row_value(
                            stats[f"exact_top{args.diagnose_topk}_recall"]
                        )
                        row[f"verified_top{args.diagnose_topk}_recall"] = row_value(
                            stats[f"verified_top{args.diagnose_topk}_recall"]
                        )
                    rows.append(row)
                    emit_row("deployment_row", row)

                    del sva_logits
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            del full_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(args.output_csv, rows)

    for context_len in context_lengths:
        for shortlist in coarse_shortlists:
            for budget in budgets:
                group = [
                    row
                    for row in rows
                    if int(row["context_len"]) == context_len
                    and int(row["coarse_shortlist"]) == shortlist
                    and int(row["budget"]) == budget
                ]
                if not group:
                    continue
                summary = {
                    "context_len": str(context_len),
                    "coarse_shortlist": str(shortlist),
                    "budget": str(budget),
                    "docs": str(len(group)),
                    "tokens": str(sum(int(row["tokens"]) for row in group)),
                    "loss_delta": row_value(weighted_average(group, "loss_delta")),
                    "kl_to_full": row_value(weighted_average(group, "kl_to_full")),
                    "top1_agreement": row_value(weighted_average(group, "top1_agreement")),
                    "logit_cosine": row_value(weighted_average(group, "logit_cosine")),
                    "avg_verified": row_value(weighted_average(group, "avg_verified")),
                    "full_ms_total": row_value(sum(float(row["full_ms"]) for row in group)),
                    "sva_ms_total": row_value(sum(float(row["sva_ms"]) for row in group)),
                }
                full_total = float(summary["full_ms_total"])
                sva_total = float(summary["sva_ms_total"])
                summary["speedup"] = row_value(full_total / sva_total if sva_total > 0 else float("nan"))
                if args.diagnose_topk > 0:
                    key = f"verified_top{args.diagnose_topk}_recall"
                    summary[key] = row_value(weighted_average(group, key))
                emit_row("deployment_summary", summary)

    if args.output_csv is not None:
        print(f"deployment_csv,{args.output_csv}")
    print("deployment_benchmark_done")


if __name__ == "__main__":
    main()
