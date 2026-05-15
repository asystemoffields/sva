"""Supervised page-side routing test for SVA.

This is the "lantern" lookup: each key writes itself to learned route cells
during prefill, and each query probes a few route cells during decode. The
router is trained against full-attention top-key labels, then evaluated as a
sublinear lookup before exact verification.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sva_learned_ivf_lookup_test import evaluate_candidate_budget, learned_scores, project_keys, project_queries
from sva_learned_lsh_lookup_test import target_prefix_for_position
from sva_learned_ranker_test import LowRankRanker, layer_qk, make_variant_text, set_seed, train_ranker
from sva_real_qk_address_sweep import (
    comma_ints,
    make_long_text,
    parse_layers,
    percentile,
    sample_query_positions,
    topk_indices_for_queries,
)


class LanternRouter(nn.Module):
    def __init__(self, n_heads: int, rank_dim: int, cells: int, temperature: float) -> None:
        super().__init__()
        scale = 1.0 / math.sqrt(rank_dim)
        self.q_routes = nn.Parameter(torch.randn(n_heads, cells, rank_dim) * scale)
        self.k_routes = nn.Parameter(torch.randn(n_heads, cells, rank_dim) * scale)
        self.temperature = temperature

    def q_logits(self, q_low: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q_low.float(), dim=-1)
        routes = F.normalize(self.q_routes.float(), dim=-1)
        return torch.einsum("hqr,hcr->hqc", q, routes) / self.temperature

    def k_logits(self, k_low: torch.Tensor) -> torch.Tensor:
        k = F.normalize(k_low.float(), dim=-1)
        routes = F.normalize(self.k_routes.float(), dim=-1)
        return torch.einsum("hkr,hcr->hkc", k, routes) / self.temperature


def gather_head_keys(k_low: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    n_heads, _, rank_dim = k_low.shape
    expanded = indices[..., None].expand(n_heads, *indices.shape[1:], rank_dim)
    gather_source = k_low
    for _ in range(indices.ndim - 2):
        gather_source = gather_source[:, None, :, :]
    gather_source = gather_source.expand(n_heads, *indices.shape[1:-1], k_low.shape[1], rank_dim)
    return gather_source.gather(dim=-2, index=expanded)


def train_lantern_router(
    router: LanternRouter,
    q_low: torch.Tensor,
    k_low: torch.Tensor,
    query_positions: np.ndarray,
    top_idx: np.ndarray,
    top_valid: np.ndarray,
    steps: int,
    batch_queries: int,
    negative_samples: int,
    lr: float,
    weight_decay: float,
    seed: int,
    negative_weight: float,
    balance_samples: int,
    balance_weight: float,
    key_alignment_weight: float,
    query_alignment_weight: float,
) -> float:
    device = q_low.device
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=weight_decay)
    positions_t = torch.tensor(query_positions, device=device, dtype=torch.long)
    top_idx_t = torch.tensor(top_idx, device=device, dtype=torch.long)
    top_valid_t = torch.tensor(top_valid, device=device, dtype=torch.bool)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    n_queries = q_low.shape[1]
    eps = 1e-6
    final_loss = float("nan")

    for _ in range(steps):
        if batch_queries >= n_queries:
            batch_idx = torch.arange(n_queries, device=device)
        else:
            batch_idx = torch.randint(n_queries, (batch_queries,), device=device, generator=generator)

        q_batch = q_low[:, batch_idx, :]
        pos_idx = top_idx_t[:, batch_idx, :]
        pos_valid = top_valid_t[:, batch_idx, :]
        pos_keys = gather_head_keys(k_low, pos_idx.clamp(0, k_low.shape[1] - 1))

        max_prefix = int(positions_t[batch_idx].max().item()) + 1
        neg_idx = torch.randint(
            max(max_prefix, 1),
            (len(batch_idx), negative_samples),
            device=device,
            generator=generator,
        )
        neg_idx = torch.minimum(neg_idx, positions_t[batch_idx, None])
        neg_idx = neg_idx[None, :, :].expand(k_low.shape[0], -1, -1)
        neg_keys = gather_head_keys(k_low, neg_idx)
        balance_idx = torch.randint(
            k_low.shape[1],
            (min(balance_samples, k_low.shape[1]),),
            device=device,
            generator=generator,
        )
        balance_keys = k_low[:, balance_idx, :]

        q_prob = F.softmax(router.q_logits(q_batch), dim=-1)
        pos_prob = F.softmax(
            torch.einsum(
                "hbtr,hcr->hbtc",
                F.normalize(pos_keys.float(), dim=-1),
                F.normalize(router.k_routes.float(), dim=-1),
            )
            / router.temperature,
            dim=-1,
        )
        neg_prob = F.softmax(
            torch.einsum(
                "hbnr,hcr->hbnc",
                F.normalize(neg_keys.float(), dim=-1),
                F.normalize(router.k_routes.float(), dim=-1),
            )
            / router.temperature,
            dim=-1,
        )

        pos_overlap = (q_prob[:, :, None, :] * pos_prob).sum(dim=-1).clamp(eps, 1.0)
        neg_overlap = (q_prob[:, :, None, :] * neg_prob).sum(dim=-1).clamp(eps, 1.0 - eps)
        balance_prob = F.softmax(router.k_logits(balance_keys), dim=-1)
        mean_usage = balance_prob.mean(dim=1).clamp_min(eps)
        balance_loss = (mean_usage * (mean_usage * mean_usage.shape[-1]).log()).sum(dim=-1).mean()
        pos_loss = (-(pos_overlap.log()) * pos_valid.float()).sum() / pos_valid.float().sum().clamp_min(1.0)
        neg_loss = -(1.0 - neg_overlap).log().mean()
        entropy = -(q_prob * q_prob.clamp_min(eps).log()).sum(dim=-1).mean() / math.log(q_prob.shape[-1])
        key_alignment = (
            (-(q_prob[:, :, None, :].detach() * pos_prob.clamp_min(eps).log()).sum(dim=-1)) * pos_valid.float()
        ).sum() / pos_valid.float().sum().clamp_min(1.0)
        query_target = (pos_prob.detach() * pos_valid[..., None].float()).sum(dim=2)
        query_target = query_target / query_target.sum(dim=-1, keepdim=True).clamp_min(eps)
        query_alignment = -(query_target * q_prob.clamp_min(eps).log()).sum(dim=-1).mean()
        loss = (
            pos_loss
            + negative_weight * neg_loss
            + balance_weight * balance_loss
            + key_alignment_weight * key_alignment
            + query_alignment_weight * query_alignment
            + 0.01 * entropy
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().item())

    return final_loss


@torch.no_grad()
def assign_lantern_writes(
    router: LanternRouter,
    k_low: torch.Tensor,
    writes: int,
    write_candidates: int,
    max_load_factor: float,
    chunk_size: int,
) -> torch.Tensor:
    n_heads, seq_len, _ = k_low.shape
    actual_writes = min(writes, router.k_routes.shape[1])
    candidate_cells = min(max(write_candidates, actual_writes), router.k_routes.shape[1])
    average_load = math.ceil((seq_len * actual_writes) / router.k_routes.shape[1])
    capacity = max(actual_writes, int(math.ceil(average_load * max_load_factor)))
    labels: list[torch.Tensor] = []
    for head_idx in range(n_heads):
        routes = F.normalize(router.k_routes[head_idx].float(), dim=-1)
        head_scores: list[torch.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            chunk = k_low[head_idx, start : start + chunk_size]
            scores = (F.normalize(chunk.float(), dim=-1) @ routes.T) / router.temperature
            head_scores.append(scores.topk(candidate_cells, dim=-1).indices.cpu())
        top_cells = torch.cat(head_scores, dim=0)
        head_labels = torch.empty(seq_len, actual_writes, dtype=torch.long)
        loads = [0 for _ in range(router.k_routes.shape[1])]
        for key_idx in range(seq_len):
            chosen: list[int] = []
            for cell in top_cells[key_idx].tolist():
                if loads[cell] < capacity and cell not in chosen:
                    loads[cell] += 1
                    chosen.append(cell)
                    if len(chosen) == actual_writes:
                        break
            if len(chosen) < actual_writes:
                for cell in top_cells[key_idx].tolist():
                    if cell not in chosen:
                        loads[cell] += 1
                        chosen.append(cell)
                        if len(chosen) == actual_writes:
                            break
            head_labels[key_idx] = torch.tensor(chosen, dtype=torch.long)
        labels.append(head_labels.to(k_low.device))
    return torch.stack(labels, dim=0)


def build_multiwrite_buckets(labels: np.ndarray, cells: int) -> list[int]:
    buckets = [0 for _ in range(cells)]
    for index, row in enumerate(labels):
        key_bit = 1 << index
        for label in row:
            buckets[int(label)] |= key_bit
    return buckets


@torch.no_grad()
def lookup_lantern_candidates(
    router: LanternRouter,
    q_low: torch.Tensor,
    labels: torch.Tensor,
    query_positions: np.ndarray,
    probes: int,
    target_context: int,
) -> tuple[list[tuple[int, int, int]], list[int], list[float]]:
    n_heads, n_queries, _ = q_low.shape
    cells = labels.shape[-1] if labels.ndim == 2 else router.q_routes.shape[1]
    actual_probes = min(probes, router.q_routes.shape[1])
    q_route_ids = router.q_logits(q_low).topk(actual_probes, dim=-1).indices.cpu().numpy()
    label_np = labels.cpu().numpy()
    bucket_sets = [build_multiwrite_buckets(label_np[head_idx], router.q_routes.shape[1]) for head_idx in range(n_heads)]
    seq_len = labels.shape[1]
    candidate_records: list[tuple[int, int, int]] = []
    actual_counts: list[int] = []
    million_counts: list[float] = []

    for head_idx in range(n_heads):
        buckets = bucket_sets[head_idx]
        for query_idx in range(n_queries):
            candidates = 0
            for cell_idx in q_route_ids[head_idx, query_idx]:
                candidates |= buckets[int(cell_idx)]
            query_pos = int(query_positions[query_idx])
            prefix_candidates = candidates & ((1 << (query_pos + 1)) - 1)
            actual_count = prefix_candidates.bit_count()
            actual_counts.append(actual_count)
            prefix_density = actual_count / max(query_pos + 1, 1)
            target_prefix = target_prefix_for_position(query_pos, seq_len, target_context)
            million_counts.append(prefix_density * target_prefix)
            candidate_records.append((head_idx, query_idx, prefix_candidates))

    _ = cells
    return candidate_records, actual_counts, million_counts


def print_lantern_result(
    layer_idx: int | str,
    seq_len: int,
    target_context: int,
    rank_dim: int,
    cells: int,
    writes: int,
    probes: int,
    budget: int,
    ranker_steps: int,
    router_steps: int,
    ranker_loss: float,
    router_loss: float,
    actual_counts: list[int],
    million_counts: list[float],
    raw_hits: int,
    verified_hits: int,
    total: int,
) -> None:
    raw_recall = raw_hits / total if total else float("nan")
    verified_recall = verified_hits / total if total else float("nan")
    actual_avg = sum(actual_counts) / max(len(actual_counts), 1)
    million_avg = sum(million_counts) / max(len(million_counts), 1)
    print(
        "lantern_result,"
        f"{layer_idx},{seq_len},{target_context},{rank_dim},{cells},{writes},{probes},{budget},"
        f"{ranker_steps},{router_steps},{ranker_loss:.6f},{router_loss:.6f},"
        f"{actual_avg:.1f},{percentile(actual_counts, 50):.1f},{percentile(actual_counts, 95):.1f},"
        f"{million_avg:.1f},{percentile(million_counts, 50):.1f},{percentile(million_counts, 95):.1f},"
        f"{raw_recall:.6f},{verified_recall:.6f},{raw_hits},{verified_hits},{total}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate supervised Lantern SVA routing.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--text-repeats", type=int, default=320)
    parser.add_argument("--eval-text-repeats", type=int, default=0)
    parser.add_argument("--eval-text-mode", choices=["same", "reverse", "rotate"], default="reverse")
    parser.add_argument("--layers", default="0,1,5,10,18,24,29")
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--cells", default="512,1024,2048")
    parser.add_argument("--writes", default="2,4,8")
    parser.add_argument("--probes", default="1,2,4")
    parser.add_argument("--budgets", default="256,512")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--train-query-samples", type=int, default=128)
    parser.add_argument("--eval-query-samples", type=int, default=64)
    parser.add_argument("--min-query-pos", type=int, default=128)
    parser.add_argument("--ranker-steps", type=int, default=160)
    parser.add_argument("--router-steps", type=int, default=240)
    parser.add_argument("--batch-queries", type=int, default=16)
    parser.add_argument("--negative-samples", type=int, default=32)
    parser.add_argument("--ranker-lr", type=float, default=3e-3)
    parser.add_argument("--router-lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--router-temperature", type=float, default=0.07)
    parser.add_argument("--negative-weight", type=float, default=0.5)
    parser.add_argument("--balance-samples", type=int, default=1024)
    parser.add_argument("--balance-weight", type=float, default=0.5)
    parser.add_argument("--key-alignment-weight", type=float, default=0.0)
    parser.add_argument("--query-alignment-weight", type=float, default=0.0)
    parser.add_argument("--write-candidates", type=int, default=64)
    parser.add_argument("--max-load-factor", type=float, default=2.0)
    parser.add_argument("--assign-chunk-size", type=int, default=8192)
    parser.add_argument("--target-context", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    model_window = int(config.max_position_embeddings)
    requested = args.max_length if args.max_length > 0 else model_window
    effective_max_length = min(requested, model_window)
    layers = parse_layers(args.layers, int(config.num_hidden_layers))
    cell_values = comma_ints(args.cells)
    write_values = comma_ints(args.writes)
    probe_values = comma_ints(args.probes)
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
    eval_batch = tokenizer(
        [make_variant_text(eval_repeats, args.eval_text_mode)],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=effective_max_length,
    )
    train_batch = {key: value.to(device) for key, value in train_batch.items()}
    eval_batch = {key: value.to(device) for key, value in eval_batch.items()}
    train_seq_len = int(train_batch["input_ids"].shape[1])
    eval_seq_len = int(eval_batch["input_ids"].shape[1])
    train_positions = sample_query_positions(train_seq_len, args.topk, args.train_query_samples, args.min_query_pos)
    eval_positions = sample_query_positions(eval_seq_len, args.topk, args.eval_query_samples, args.min_query_pos)

    print("metric,value")
    print(f"model_id,{args.model_id}")
    print(f"model_max_position_embeddings,{model_window}")
    print(f"requested_max_length,{requested}")
    print(f"effective_max_length,{effective_max_length}")
    print(f"train_seq_len,{train_seq_len}")
    print(f"eval_seq_len,{eval_seq_len}")
    print(f"eval_text_mode,{args.eval_text_mode}")
    print(f"target_context,{args.target_context}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(f"layers,{';'.join(str(layer) for layer in layers)}")
    print(f"rank_dim,{args.rank_dim}")
    print(f"cells,{';'.join(str(value) for value in cell_values)}")
    print(f"writes,{';'.join(str(value) for value in write_values)}")
    print(f"probes,{';'.join(str(value) for value in probe_values)}")
    print(f"budgets,{';'.join(str(value) for value in budgets)}")
    print(f"topk,{args.topk}")
    print(f"train_query_samples,{len(train_positions)}")
    print(f"eval_query_samples,{len(eval_positions)}")
    print(f"write_candidates,{args.write_candidates}")
    print(f"max_load_factor,{args.max_load_factor}")
    print(f"key_alignment_weight,{args.key_alignment_weight}")
    print(f"query_alignment_weight,{args.query_alignment_weight}")
    print(
        "lantern_header,"
        "layer,seq_len,target_context,rank_dim,cells,writes,probes,budget,ranker_steps,router_steps,"
        "ranker_loss,router_loss,avg_candidates,p50_candidates,p95_candidates,"
        "avg_empirical_million_candidates,p50_empirical_million_candidates,p95_empirical_million_candidates,"
        "raw_topk_recall,verified_topk_recall,raw_hits,verified_hits,total"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    model.eval()
    print("progress,model_loaded", flush=True)
    with torch.no_grad():
        train_outputs = model(**train_batch, output_hidden_states=True, use_cache=False)
        eval_outputs = model(**eval_batch, output_hidden_states=True, use_cache=False)
    print("progress,hidden_states_ready", flush=True)

    train_hidden_states = train_outputs.hidden_states
    eval_hidden_states = eval_outputs.hidden_states
    train_position_ids = torch.arange(train_seq_len, device=device).unsqueeze(0)
    eval_position_ids = torch.arange(eval_seq_len, device=device).unsqueeze(0)
    train_position_tensor = torch.tensor(train_positions, device=device, dtype=torch.long)

    aggregate: dict[tuple[int, int, int, int], dict[str, object]] = defaultdict(
        lambda: {"actual": [], "million": [], "raw_hits": 0, "verified_hits": 0, "total": 0}
    )

    for layer_idx in layers:
        print(f"progress,layer_start,{layer_idx}", flush=True)
        train_query_all, train_key, train_scaling = layer_qk(model, train_hidden_states, layer_idx, train_position_ids)
        eval_query_all, eval_key, eval_scaling = layer_qk(model, eval_hidden_states, layer_idx, eval_position_ids)
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

        torch.manual_seed(args.seed + layer_idx * 1000 + args.rank_dim)
        ranker = LowRankRanker(train_query_all.shape[0], train_query_all.shape[-1], args.rank_dim).to(device)
        ranker_loss = train_ranker(
            ranker,
            train_key,
            train_query,
            train_positions,
            train_top_idx,
            train_top_valid,
            args.ranker_steps,
            args.batch_queries,
            args.ranker_lr,
            args.weight_decay,
            args.seed + layer_idx * 1000 + args.rank_dim,
        )
        print(f"progress,layer_ranker_trained,{layer_idx},{ranker_loss:.6f}", flush=True)

        train_q_low = project_queries(ranker, train_query_all, train_positions)
        train_k_low = project_keys(ranker, train_key)
        eval_q_low = project_queries(ranker, eval_query_all, eval_positions)
        eval_k_low = project_keys(ranker, eval_key)
        rank_scores = learned_scores(eval_q_low, eval_k_low, args.rank_dim).float().cpu().numpy()
        print(f"progress,layer_low_rank_ready,{layer_idx}", flush=True)

        for cells in cell_values:
            torch.manual_seed(args.seed + layer_idx * 1000 + cells)
            router = LanternRouter(train_q_low.shape[0], args.rank_dim, cells, args.router_temperature).to(device)
            router_loss = train_lantern_router(
                router,
                train_q_low,
                train_k_low,
                train_positions,
                train_top_idx,
                train_top_valid,
                args.router_steps,
                args.batch_queries,
                args.negative_samples,
                args.router_lr,
                args.weight_decay,
                args.seed + layer_idx * 1000 + cells,
                args.negative_weight,
                args.balance_samples,
                args.balance_weight,
                args.key_alignment_weight,
                args.query_alignment_weight,
            )
            print(f"progress,router_trained,{layer_idx},{cells},{router_loss:.6f}", flush=True)

            for writes in write_values:
                eval_labels = assign_lantern_writes(
                    router,
                    eval_k_low,
                    writes,
                    args.write_candidates,
                    args.max_load_factor,
                    args.assign_chunk_size,
                )
                print(f"progress,writes_ready,{layer_idx},{cells},{writes}", flush=True)
                for probes in probe_values:
                    candidate_records, actual, million = lookup_lantern_candidates(
                        router,
                        eval_q_low,
                        eval_labels,
                        eval_positions,
                        probes,
                        args.target_context,
                    )
                    print(
                        f"progress,lookup_ready,{layer_idx},{cells},{writes},{probes},"
                        f"{sum(actual) / max(len(actual), 1):.1f},"
                        f"{sum(million) / max(len(million), 1):.1f}",
                        flush=True,
                    )
                    for budget in budgets:
                        raw_hits, verified_hits, total = evaluate_candidate_budget(
                            candidate_records,
                            rank_scores,
                            eval_top_idx,
                            eval_top_valid,
                            budget,
                        )
                        print_lantern_result(
                            layer_idx,
                            eval_seq_len,
                            args.target_context,
                            args.rank_dim,
                            cells,
                            writes,
                            probes,
                            budget,
                            args.ranker_steps,
                            args.router_steps,
                            ranker_loss,
                            router_loss,
                            actual,
                            million,
                            raw_hits,
                            verified_hits,
                            total,
                        )
                        key_tuple = (cells, writes, probes, budget)
                        bucket = aggregate[key_tuple]
                        bucket["actual"].extend(actual)
                        bucket["million"].extend(million)
                        bucket["raw_hits"] = int(bucket["raw_hits"]) + raw_hits
                        bucket["verified_hits"] = int(bucket["verified_hits"]) + verified_hits
                        bucket["total"] = int(bucket["total"]) + total
                del eval_labels
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del router
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del train_query_all, train_key, eval_query_all, eval_key, train_query, ranker
        del train_q_low, train_k_low, eval_q_low, eval_k_low, rank_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for (cells, writes, probes, budget), bucket in sorted(aggregate.items()):
        print_lantern_result(
            "all",
            eval_seq_len,
            args.target_context,
            args.rank_dim,
            cells,
            writes,
            probes,
            budget,
            args.ranker_steps,
            args.router_steps,
            float("nan"),
            float("nan"),
            bucket["actual"],  # type: ignore[arg-type]
            bucket["million"],  # type: ignore[arg-type]
            int(bucket["raw_hits"]),
            int(bucket["verified_hits"]),
            int(bucket["total"]),
        )


if __name__ == "__main__":
    main()
