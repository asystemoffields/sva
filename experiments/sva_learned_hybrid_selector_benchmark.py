"""Learned token/block hybrid SVA selector benchmark.

The selector sees cheap pre-verifier features and decides whether a head/query
should use scattered token SVA or contiguous block SVA. Labels come from the
lower-output-error path against exact full attention.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sva_block_elevator_benchmark import (
    block_scores,
    block_statement_output,
    comma_ints,
    exact_attention_streaming,
    layer_qkv_from_hidden,
    make_synthetic_kv_bank,
    output_metrics,
    project_catalog,
    sample_query_positions,
    token_sva_output,
)
from sva_block_hybrid_benchmark import (
    coarse_entropy_and_margin,
    hybrid_output,
    per_item_relative_error,
    recall_matrix_block,
    recall_matrix_token,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sva import load_sva_artifact_bundle
from sva_full_deployment_benchmark import EVAL_DOCS, repeated_document
from sva_pretrained_socket_test import encode_batch, parse_layer_list


@dataclass
class HybridCase:
    split: str
    context: int
    layer_idx: int
    block_size: int
    block_budget: int
    features: torch.Tensor
    labels: torch.Tensor
    entropy: torch.Tensor
    token_out: torch.Tensor
    block_out: torch.Tensor
    teacher_out: torch.Tensor
    token_recall: torch.Tensor
    block_recall: torch.Tensor
    token_error: torch.Tensor
    block_error: torch.Tensor


def comma_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float.")
    return values


def row_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def emit(prefix: str, row: dict[str, float | int | str]) -> None:
    print(prefix + "," + ",".join(f"{key}={row_value(value)}" for key, value in row.items()), flush=True)


def flat_metrics(candidate: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
    cosine = torch.nn.functional.cosine_similarity(candidate.float(), teacher.float(), dim=-1)
    mse = (candidate.float() - teacher.float()).square().mean(dim=-1)
    rel = (candidate.float() - teacher.float()).norm(dim=-1) / teacher.float().norm(dim=-1).clamp_min(1e-8)
    return {
        "output_cosine": float(cosine.mean().item()),
        "output_mse": float(mse.mean().item()),
        "relative_error": float(rel.mean().item()),
    }


def flatten_hw(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1).cpu()


def flatten_hwd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1, tensor.shape[-1]).cpu()


def normalized_top_entropy(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = min(topk, scores.shape[-1])
    top_scores = scores.topk(k, dim=-1).values.float()
    probs = torch.softmax(top_scores - top_scores[..., :1], dim=-1)
    entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1) / math.log(max(k, 2))
    if k > 1:
        margin = top_scores[..., 0] - top_scores[..., 1]
    else:
        margin = torch.zeros_like(entropy)
    spread = top_scores[..., 0] - top_scores[..., -1]
    return entropy, margin, spread


def selector_features(
    *,
    entropy: torch.Tensor,
    margin: torch.Tensor,
    q_low: torch.Tensor,
    block_score: torch.Tensor,
    context: int,
    base_context: int,
    layer_idx: int,
    layer_count: int,
    block_size: int,
    block_budget: int,
    token_budget: int,
) -> torch.Tensor:
    heads, queries = entropy.shape
    block_entropy, block_margin, block_spread = normalized_top_entropy(block_score, min(64, block_score.shape[-1]))
    q_norm = q_low.float().norm(dim=-1)
    head_frac = torch.arange(heads, device=entropy.device, dtype=torch.float32)[:, None].expand(heads, queries)
    head_frac = head_frac / max(heads - 1, 1)
    query_frac = torch.arange(queries, device=entropy.device, dtype=torch.float32)[None, :].expand(heads, queries)
    query_frac = query_frac / max(queries - 1, 1)
    constants = [
        math.log2(max(context, 1) / max(base_context, 1)),
        layer_idx / max(layer_count - 1, 1),
        block_size / 128.0,
        block_budget / 64.0,
        min(context, block_size * block_budget) / max(min(token_budget, context), 1),
        context / max(min(token_budget, context), 1),
    ]
    constant_features = [
        torch.full_like(entropy, float(value), dtype=torch.float32)
        for value in constants
    ]
    stacked = torch.stack(
        [
            entropy.float(),
            margin.float(),
            q_norm.float(),
            block_entropy.float(),
            block_margin.float(),
            block_spread.float(),
            head_frac,
            query_frac,
            *constant_features,
        ],
        dim=-1,
    )
    return stacked.reshape(-1, stacked.shape[-1]).detach().cpu()


def summarize_case(
    selector: str,
    case: HybridCase,
    choose_token_flat: torch.Tensor,
    token_budget: int,
) -> dict[str, float | int | str]:
    choose_token = choose_token_flat.bool()
    mixed = torch.where(choose_token[:, None], case.token_out, case.block_out)
    token_fraction = float(choose_token.float().mean().item())
    block_tokens = min(case.context, case.block_size * case.block_budget)
    avg_tokens_read = token_fraction * min(token_budget, case.context) + (1.0 - token_fraction) * block_tokens
    avg_segments = token_fraction * min(token_budget, case.context) + (1.0 - token_fraction) * case.block_budget
    recall = torch.where(choose_token, case.token_recall, case.block_recall).mean().item()
    return {
        "selector": selector,
        "split": case.split,
        "layer": case.layer_idx,
        "context": case.context,
        "block_size": case.block_size,
        "block_budget": case.block_budget,
        "token_fraction": token_fraction,
        "avg_tokens_read": avg_tokens_read,
        "read_reduction": case.context / max(avg_tokens_read, 1e-9),
        "avg_segments": avg_segments,
        "segment_reduction": min(token_budget, case.context) / max(avg_segments, 1e-9),
        "top16_recall": float(recall),
        **flat_metrics(mixed, case.teacher_out),
    }


def add_summary(
    aggregate: dict[tuple[str, str, int, int, int], dict[str, float]],
    row: dict[str, float | int | str],
) -> None:
    key = (
        str(row["selector"]),
        str(row["split"]),
        int(row["context"]),
        int(row["block_size"]),
        int(row["block_budget"]),
    )
    bucket = aggregate.setdefault(
        key,
        {
            "count": 0.0,
            "token_fraction": 0.0,
            "avg_tokens_read": 0.0,
            "read_reduction": 0.0,
            "avg_segments": 0.0,
            "segment_reduction": 0.0,
            "top16_recall": 0.0,
            "output_cosine": 0.0,
            "relative_error": 0.0,
        },
    )
    bucket["count"] += 1.0
    for metric in bucket:
        if metric != "count":
            bucket[metric] += float(row[metric])


def train_selector(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-5)
    x = (features - mean) / std
    y = labels.float()
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 16),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 1),
    )
    pos = y.sum().clamp_min(1.0)
    neg = (y.numel() - y.sum()).clamp_min(1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=neg / pos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        pred = probs >= 0.5
        accuracy = (pred == y.bool()).float().mean().item()
        positive_rate = y.mean().item()
    return model, mean, std, {
        "train_accuracy": float(accuracy),
        "train_positive_rate": float(positive_rate),
        "feature_count": int(features.shape[1]),
        "example_count": int(features.shape[0]),
    }


@torch.no_grad()
def collect_cases(
    *,
    split: str,
    model: Any,
    tokenizer: Any,
    bundle: Any,
    args: argparse.Namespace,
    doc_index: int,
    seed: int,
    device: torch.device,
) -> list[HybridCase]:
    doc = EVAL_DOCS[doc_index % len(EVAL_DOCS)]
    batch = encode_batch(tokenizer, [repeated_document(doc, args.eval_repeats)], args.base_context, device)
    seq_len = int(batch["input_ids"].shape[1])
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    query_positions = sample_query_positions(seq_len, args.query_samples, args.min_query_pos).to(device)
    with torch.no_grad():
        output = model(**batch, use_cache=False, output_hidden_states=True)
    if output.hidden_states is None:
        raise ValueError("Expected hidden states from model forward.")

    cases: list[HybridCase] = []
    for layer_idx in args.layers:
        query_all, key_base, value_base, scaling = layer_qkv_from_hidden(model, output.hidden_states, layer_idx, position_ids)
        query = query_all[:, query_positions, :]
        for context in args.target_contexts:
            key_bank, value_bank = make_synthetic_kv_bank(
                key_base,
                value_base,
                context,
                args.synthetic_noise_std,
                seed + layer_idx * 1000,
            )
            teacher_out, teacher_topk = exact_attention_streaming(
                query,
                key_bank,
                value_bank,
                scaling,
                args.teacher_chunk_size,
            )
            q_low, k_low, codebooks, codes = project_catalog(query, key_bank, bundle, layer_idx, args.assign_chunk_size)
            entropy, margin = coarse_entropy_and_margin(q_low, codebooks, codes, bundle.rank_dim, args.token_shortlist)
            token_out, token_idx = token_sva_output(
                query,
                key_bank,
                value_bank,
                q_low,
                k_low,
                codebooks,
                codes,
                bundle.rank_dim,
                scaling,
                args.token_shortlist,
                args.token_budget,
            )
            token_error = per_item_relative_error(token_out, teacher_out)
            token_recall = recall_matrix_token(token_idx, teacher_topk)
            emit(
                "learned_hybrid_baseline",
                {
                    "split": split,
                    "variant": "token",
                    "layer": layer_idx,
                    "context": context,
                    "tokens_read": min(args.token_budget, context),
                    "segments": min(args.token_budget, context),
                    **output_metrics(token_out, teacher_out),
                },
            )
            for block_size in args.block_sizes:
                block_score = block_scores(q_low, k_low, codebooks, codes, bundle.rank_dim, block_size, "centroid")
                blocks = block_score.shape[-1]
                for block_budget in args.block_budgets:
                    actual_blocks = min(block_budget, blocks)
                    selected_blocks = block_score.topk(actual_blocks, dim=-1).indices
                    block_out = block_statement_output(query, key_bank, value_bank, selected_blocks, block_size, scaling)
                    block_error = per_item_relative_error(block_out, teacher_out)
                    block_recall_values = recall_matrix_block(selected_blocks, teacher_topk, block_size)
                    features = selector_features(
                        entropy=entropy,
                        margin=margin,
                        q_low=q_low,
                        block_score=block_score,
                        context=context,
                        base_context=args.base_context,
                        layer_idx=layer_idx,
                        layer_count=args.layer_count,
                        block_size=block_size,
                        block_budget=actual_blocks,
                        token_budget=args.token_budget,
                    )
                    labels = flatten_hw(token_error <= block_error)
                    cases.append(
                        HybridCase(
                            split=split,
                            context=context,
                            layer_idx=layer_idx,
                            block_size=block_size,
                            block_budget=actual_blocks,
                            features=features,
                            labels=labels,
                            entropy=flatten_hw(entropy),
                            token_out=flatten_hwd(token_out),
                            block_out=flatten_hwd(block_out),
                            teacher_out=flatten_hwd(teacher_out),
                            token_recall=flatten_hw(token_recall),
                            block_recall=flatten_hw(block_recall_values),
                            token_error=flatten_hw(token_error),
                            block_error=flatten_hw(block_error),
                        )
                    )
                    emit(
                        "learned_hybrid_baseline",
                        {
                            "split": split,
                            "variant": "block",
                            "layer": layer_idx,
                            "context": context,
                            "block_size": block_size,
                            "block_budget": actual_blocks,
                            "tokens_read": min(context, block_size * actual_blocks),
                            "segments": actual_blocks,
                            **output_metrics(block_out, teacher_out),
                        },
                    )
                    del block_out, block_error, block_recall_values, selected_blocks
            del key_bank, value_bank, teacher_out, teacher_topk, q_low, k_low, codebooks, codes
            del token_out, token_idx, token_error, token_recall, entropy, margin
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del query_all, key_base, value_base, query
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del output
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned token/block SVA selector benchmark.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/hf_artifacts/sva-smollm2-135m-2x256-v1"))
    parser.add_argument("--base-context", type=int, default=8192)
    parser.add_argument("--target-contexts", default="8192,32768,131072")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--train-doc-index", type=int, default=0)
    parser.add_argument("--test-doc-index", type=int, default=1)
    parser.add_argument("--eval-repeats", type=int, default=320)
    parser.add_argument("--query-samples", type=int, default=8)
    parser.add_argument("--min-query-pos", type=int, default=512)
    parser.add_argument("--token-shortlist", type=int, default=8192)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--block-sizes", default="64,128")
    parser.add_argument("--block-budgets", default="16,32")
    parser.add_argument("--selector-thresholds", default="0.35,0.50,0.65")
    parser.add_argument("--entropy-threshold", type=float, default=0.55)
    parser.add_argument("--synthetic-noise-std", type=float, default=0.01)
    parser.add_argument("--teacher-chunk-size", type=int, default=65536)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--train-seed", type=int, default=29)
    parser.add_argument("--test-seed", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.03)
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
    layer_list = parse_layer_list(args.layers, len(model.model.layers))
    args.layers = layer_list if layer_list is not None else list(range(len(model.model.layers)))
    args.layer_count = len(model.model.layers)
    args.target_contexts = comma_ints(args.target_contexts)
    args.block_sizes = comma_ints(args.block_sizes)
    args.block_budgets = comma_ints(args.block_budgets)
    thresholds = comma_floats(args.selector_thresholds)

    print("learned_hybrid_start", flush=True)
    print(f"model_id,{args.model_id}", flush=True)
    print(f"device,{device}", flush=True)
    print(f"dtype,{dtype}", flush=True)
    print(f"target_contexts,{','.join(str(item) for item in args.target_contexts)}", flush=True)
    print(f"layers,{','.join(str(item) for item in args.layers)}", flush=True)
    print(f"artifact_profile,{bundle.manifest.get('profile_name')}", flush=True)

    train_cases = collect_cases(
        split="train",
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        args=args,
        doc_index=args.train_doc_index,
        seed=args.train_seed,
        device=device,
    )
    test_cases = collect_cases(
        split="test",
        model=model,
        tokenizer=tokenizer,
        bundle=bundle,
        args=args,
        doc_index=args.test_doc_index,
        seed=args.test_seed,
        device=device,
    )

    train_features = torch.cat([case.features for case in train_cases], dim=0)
    train_labels = torch.cat([case.labels for case in train_cases], dim=0)
    selector, mean, std, train_stats = train_selector(train_features, train_labels, args.epochs, args.lr, args.train_seed)
    emit("learned_hybrid_train", train_stats)

    aggregate: dict[tuple[str, str, int, int, int], dict[str, float]] = {}
    for case in train_cases + test_cases:
        oracle = case.token_error <= case.block_error
        row = summarize_case("oracle_error", case, oracle, args.token_budget)
        emit("learned_hybrid_row", row)
        add_summary(aggregate, row)

        entropy_choice = case.entropy <= args.entropy_threshold
        row = summarize_case(f"entropy_le_{args.entropy_threshold:.2f}", case, entropy_choice, args.token_budget)
        emit("learned_hybrid_row", row)
        add_summary(aggregate, row)

        with torch.no_grad():
            probs = torch.sigmoid(selector((case.features - mean) / std).squeeze(-1))
        for threshold in thresholds:
            row = summarize_case(f"learned_ge_{threshold:.2f}", case, probs >= threshold, args.token_budget)
            emit("learned_hybrid_row", row)
            add_summary(aggregate, row)

    for (selector_name, split, context, block_size, block_budget), bucket in sorted(aggregate.items()):
        count = max(bucket["count"], 1.0)
        emit(
            "learned_hybrid_summary",
            {
                "selector": selector_name,
                "split": split,
                "context": context,
                "block_size": block_size,
                "block_budget": block_budget,
                "token_fraction": bucket["token_fraction"] / count,
                "avg_tokens_read": bucket["avg_tokens_read"] / count,
                "read_reduction": bucket["read_reduction"] / count,
                "avg_segments": bucket["avg_segments"] / count,
                "segment_reduction": bucket["segment_reduction"] / count,
                "top16_recall": bucket["top16_recall"] / count,
                "output_cosine": bucket["output_cosine"] / count,
                "relative_error": bucket["relative_error"] / count,
                "layers": int(count),
            },
        )

    print("learned_hybrid_done", flush=True)


if __name__ == "__main__":
    main()
