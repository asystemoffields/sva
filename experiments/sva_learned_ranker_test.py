"""Learned compressed Q/K ranker test for SVA.

This isolates the next SVA risk after random address projections: whether a
small trained Q/K score can bring the full-attention top keys to the exact
verifier at useful candidate budgets.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from sva_real_qk_address_sweep import (
    TEXTS,
    comma_ints,
    make_long_text,
    parse_layers,
    percentile,
    sample_query_positions,
    topk_indices_for_queries,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_query_positions(
    seq_len: int,
    topk: int,
    train_samples: int,
    eval_samples: int,
    min_query_pos: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = sample_query_positions(seq_len, topk, train_samples + eval_samples, min_query_pos)
    rng = np.random.default_rng(seed)
    shuffled = positions.copy()
    rng.shuffle(shuffled)
    train = np.sort(shuffled[: min(train_samples, len(shuffled))])
    eval_positions = np.sort(shuffled[min(train_samples, len(shuffled)) :])
    eval_positions = eval_positions[: min(eval_samples, len(eval_positions))]
    if len(eval_positions) == 0:
        raise ValueError("Need at least one eval query position.")
    return train, eval_positions


def make_variant_text(repeats: int, mode: str) -> str:
    if mode == "same":
        return make_long_text(repeats)
    if mode == "reverse":
        paragraphs = list(reversed(TEXTS))
    elif mode == "rotate":
        paragraphs = TEXTS[1:] + TEXTS[:1]
    else:
        raise ValueError(f"Unknown eval text mode: {mode}")
    return " ".join(paragraphs * max(repeats, 1))


def target_distribution(top_idx: torch.Tensor, top_valid: torch.Tensor, seq_len: int) -> torch.Tensor:
    weights = top_valid.float()
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    weights = weights / denom
    target = torch.zeros((*top_idx.shape[:2], seq_len), device=top_idx.device, dtype=torch.float32)
    target.scatter_add_(dim=-1, index=top_idx.clamp(0, seq_len - 1), src=weights)
    return target


def causal_mask_scores(scores: torch.Tensor, query_positions: torch.Tensor) -> torch.Tensor:
    key_positions = torch.arange(scores.shape[-1], device=scores.device)
    allowed = key_positions[None, None, :] <= query_positions[None, :, None]
    return scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)


class LowRankRanker(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, rank_dim: int) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(head_dim)
        self.rank_dim = rank_dim
        self.q_proj = nn.Parameter(torch.randn(n_heads, head_dim, rank_dim) * scale)
        self.k_proj = nn.Parameter(torch.randn(n_heads, head_dim, rank_dim) * scale)
        self.logit_scale = nn.Parameter(torch.zeros(n_heads))

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        q_low = torch.einsum("hqd,hdr->hqr", query, self.q_proj)
        k_low = torch.einsum("hkd,hdr->hkr", key, self.k_proj)
        scores = torch.einsum("hqr,hkr->hqk", q_low, k_low) / math.sqrt(self.rank_dim)
        scale = self.logit_scale.exp().clamp(0.01, 100.0)
        return scores * scale[:, None, None]


def train_ranker(
    ranker: LowRankRanker,
    key: torch.Tensor,
    train_query: torch.Tensor,
    train_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    steps: int,
    batch_queries: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> float:
    device = key.device
    seq_len = key.shape[1]
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=lr, weight_decay=weight_decay)
    positions_t = torch.tensor(train_positions, device=device, dtype=torch.long)
    top_idx_t = torch.tensor(top_idx, device=device, dtype=torch.long)
    top_valid_t = torch.tensor(top_valid, device=device, dtype=torch.bool)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    n_queries = train_query.shape[1]
    final_loss = float("nan")

    for _ in range(steps):
        if batch_queries >= n_queries:
            batch_idx = torch.arange(n_queries, device=device)
        else:
            batch_idx = torch.randint(n_queries, (batch_queries,), device=device, generator=generator)

        query_batch = train_query[:, batch_idx, :]
        position_batch = positions_t[batch_idx]
        scores = causal_mask_scores(ranker(query_batch, key), position_batch)
        target = target_distribution(top_idx_t[:, batch_idx, :], top_valid_t[:, batch_idx, :], seq_len)
        log_probs = F.log_softmax(scores, dim=-1)
        loss = -(target * log_probs).sum(dim=-1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().item())

    return final_loss


@torch.no_grad()
def evaluate_ranker(
    phase: str,
    layer_idx: int | str,
    seq_len: int,
    rank_dim: int,
    budgets: list[int],
    train_steps: int,
    final_loss: float,
    ranker: LowRankRanker,
    key: torch.Tensor,
    eval_query: torch.Tensor,
    eval_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    aggregate: dict[tuple[str, int, int], dict[str, object]],
) -> None:
    device = key.device
    positions_t = torch.tensor(eval_positions, device=device, dtype=torch.long)
    scores = causal_mask_scores(ranker(eval_query, key), positions_t)
    top_idx_t = torch.tensor(top_idx, device=device, dtype=torch.long)
    top_valid_t = torch.tensor(top_valid, device=device, dtype=torch.bool)

    for budget in budgets:
        actual_budget = min(budget, seq_len)
        candidate_idx = scores.topk(actual_budget, dim=-1).indices
        candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
        candidate_mask.scatter_(dim=-1, index=candidate_idx, value=True)
        hits_t = candidate_mask.gather(dim=-1, index=top_idx_t.clamp(0, seq_len - 1)) & top_valid_t
        hits = int(hits_t.sum().item())
        total = int(top_valid_t.sum().item())
        counts = np.minimum(eval_positions + 1, budget).astype(np.int64)
        counts = np.tile(counts[None, :], (eval_query.shape[0], 1)).reshape(-1)
        print_ranker_result(
            phase,
            layer_idx,
            seq_len,
            rank_dim,
            budget,
            train_steps,
            final_loss,
            counts.tolist(),
            hits,
            total,
        )

        bucket = aggregate.setdefault(
            (phase, rank_dim, budget),
            {"counts": [], "hits": 0, "total": 0},
        )
        bucket["counts"].extend(counts.tolist())
        bucket["hits"] = int(bucket["hits"]) + hits
        bucket["total"] = int(bucket["total"]) + total


def print_ranker_result(
    phase: str,
    layer_idx: int | str,
    seq_len: int,
    rank_dim: int,
    budget: int,
    train_steps: int,
    final_loss: float,
    counts: list[int],
    hits: int,
    total: int,
) -> None:
    recall = hits / total if total else float("nan")
    avg_candidates = sum(counts) / max(len(counts), 1)
    print(
        "ranker_result,"
        f"{phase},{layer_idx},{seq_len},{rank_dim},{budget},{train_steps},{final_loss:.6f},"
        f"{avg_candidates:.1f},{percentile(counts, 50):.1f},{percentile(counts, 95):.1f},"
        f"{recall:.6f},{hits},{total}",
        flush=True,
    )


@torch.no_grad()
def layer_qk(
    model: AutoModelForCausalLM,
    hidden_states: tuple[torch.Tensor, ...],
    layer_idx: int,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    layer = model.model.layers[layer_idx]
    hidden = hidden_states[layer_idx]
    hidden_shape = (hidden.shape[0], hidden.shape[1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key = repeat_kv(key, layer.self_attn.num_key_value_groups)
    return query[0].float(), key[0].float(), float(layer.self_attn.scaling)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train low-rank SVA Q/K rankers.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="same")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--rank-dims", default="16,32,64")
    parser.add_argument("--budgets", default="64,128,256,512,1024")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--eval-query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))
    rank_dims = comma_ints(args.rank_dims)
    budgets = comma_ints(args.budgets)

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eval_repeats = args.eval_text_repeats if args.eval_text_repeats > 0 else args.text_repeats
    train_batch = tokenizer(
        [make_long_text(args.text_repeats)],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=effective_max_length,
    )
    train_batch = {key: value.to(device) for key, value in train_batch.items()}
    train_seq_len = int(train_batch["input_ids"].shape[1])
    if args.eval_text_mode == "same" and eval_repeats == args.text_repeats:
        eval_batch = train_batch
        eval_seq_len = train_seq_len
        train_positions, eval_positions = split_query_positions(
            train_seq_len,
            args.topk,
            args.train_query_samples,
            args.eval_query_samples,
            args.min_query_pos,
            args.seed,
        )
    else:
        eval_batch = tokenizer(
            [make_variant_text(eval_repeats, args.eval_text_mode)],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=effective_max_length,
        )
        eval_batch = {key: value.to(device) for key, value in eval_batch.items()}
        eval_seq_len = int(eval_batch["input_ids"].shape[1])
        train_positions = sample_query_positions(
            train_seq_len,
            args.topk,
            args.train_query_samples,
            args.min_query_pos,
        )
        eval_positions = sample_query_positions(
            eval_seq_len,
            args.topk,
            args.eval_query_samples,
            args.min_query_pos,
        )

    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"model_max_position_embeddings,{model_window}")
    print(f"requested_max_length,{requested}")
    print(f"effective_max_length,{effective_max_length}")
    print(f"seq_len,{train_seq_len}")
    print(f"train_seq_len,{train_seq_len}")
    print(f"eval_seq_len,{eval_seq_len}")
    print(f"eval_text_mode,{args.eval_text_mode}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"layers,{';'.join(str(layer) for layer in layers)}")
    print(f"rank_dims,{';'.join(str(value) for value in rank_dims)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(f"first_train_query_pos,{int(train_positions[0]) if len(train_positions) else -1}")
    print(f"last_train_query_pos,{int(train_positions[-1]) if len(train_positions) else -1}")
    print(f"first_eval_query_pos,{int(eval_positions[0])}")
    print(f"last_eval_query_pos,{int(eval_positions[-1])}")
    print(
        "ranker_header,"
        "phase,layer,seq_len,rank_dim,budget,train_steps,final_loss,"
        "avg_candidates,p50_candidates,p95_candidates,topk_recall,hits,total"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    model.eval()
    with torch.no_grad():
        train_outputs = model(**train_batch, output_hidden_states=True, use_cache=False)
        if eval_batch is train_batch:
            eval_outputs = train_outputs
        else:
            eval_outputs = model(**eval_batch, output_hidden_states=True, use_cache=False)
    train_hidden_states = train_outputs.hidden_states
    eval_hidden_states = eval_outputs.hidden_states
    train_position_ids = torch.arange(train_seq_len, device=device).unsqueeze(0)
    eval_position_ids = torch.arange(eval_seq_len, device=device).unsqueeze(0)
    train_position_tensor = torch.tensor(train_positions, device=device, dtype=torch.long)
    eval_position_tensor = torch.tensor(eval_positions, device=device, dtype=torch.long)
    aggregate: dict[tuple[str, int, int], dict[str, object]] = defaultdict(
        lambda: {"counts": [], "hits": 0, "total": 0}
    )

    for layer_idx in layers:
        train_query_all, train_key, train_scaling = layer_qk(
            model,
            train_hidden_states,
            layer_idx,
            train_position_ids,
        )
        if eval_hidden_states is train_hidden_states:
            eval_query_all = train_query_all
            eval_key = train_key
            eval_scaling = train_scaling
        else:
            eval_query_all, eval_key, eval_scaling = layer_qk(
                model,
                eval_hidden_states,
                layer_idx,
                eval_position_ids,
            )
        train_top_idx, train_top_valid = topk_indices_for_queries(
            train_query_all,
            train_key,
            train_positions,
            args.topk,
            train_scaling,
        )
        eval_top_idx, eval_top_valid = topk_indices_for_queries(
            eval_query_all,
            eval_key,
            eval_positions,
            args.topk,
            eval_scaling,
        )
        train_query = train_query_all[:, train_position_tensor, :].contiguous()
        eval_query = eval_query_all[:, eval_position_tensor, :].contiguous()

        for rank_dim in rank_dims:
            torch.manual_seed(args.seed + layer_idx * 1000 + rank_dim)
            ranker = LowRankRanker(train_query_all.shape[0], train_query_all.shape[-1], rank_dim).to(device)
            evaluate_ranker(
                "random",
                layer_idx,
                eval_seq_len,
                rank_dim,
                budgets,
                0,
                float("nan"),
                ranker,
                eval_key,
                eval_query,
                eval_positions,
                eval_top_idx,
                eval_top_valid,
                aggregate,
            )
            final_loss = train_ranker(
                ranker,
                train_key,
                train_query,
                train_positions,
                train_top_idx,
                train_top_valid,
                args.train_steps,
                args.batch_queries,
                args.lr,
                args.weight_decay,
                args.seed + layer_idx * 1000 + rank_dim,
            )
            evaluate_ranker(
                "trained",
                layer_idx,
                eval_seq_len,
                rank_dim,
                budgets,
                args.train_steps,
                final_loss,
                ranker,
                eval_key,
                eval_query,
                eval_positions,
                eval_top_idx,
                eval_top_valid,
                aggregate,
            )

        del train_query_all, train_key, eval_query_all, eval_key, train_query, eval_query
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (phase, rank_dim, budget), bucket in sorted(aggregate.items()):
        print_ranker_result(
            phase,
            "all",
            eval_seq_len,
            rank_dim,
            budget,
            args.train_steps if phase == "trained" else 0,
            float("nan"),
            bucket["counts"],  # type: ignore[arg-type]
            int(bucket["hits"]),
            int(bucket["total"]),
        )


if __name__ == "__main__":
    main()
