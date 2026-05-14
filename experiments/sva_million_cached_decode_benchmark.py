"""Synthetic million-token cached-decode throughput benchmark for SVA.

This isolates the serving-shaped lookup after key-side SVA catalogs have
already been cached. It compares full decode attention with SVA decode lookup
over large cached contexts, including the final exact-attention verifier.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def code_dtype(codewords: int) -> torch.dtype:
    if codewords <= 256:
        return torch.uint8
    if codewords <= 32768:
        return torch.int16
    return torch.int32


@dataclass
class CacheTensors:
    query: torch.Tensor
    query_low: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_low: torch.Tensor
    codebooks: torch.Tensor
    codes: torch.Tensor


@dataclass
class TimedResult:
    avg: float
    p50: float
    p95: float


@dataclass
class ComponentResult:
    coarse_scan_ms: float
    shortlist_topk_ms: float
    exact_rescore_ms: float
    verifier_ms: float
    total_ms: float


def timed_call(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> TimedResult:
    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            _ = fn()
        synchronize(device)

        times: list[float] = []
        for _ in range(max(repeats, 1)):
            synchronize(device)
            start = time.perf_counter()
            result = fn()
            synchronize(device)
            times.append((time.perf_counter() - start) * 1000.0)
            if isinstance(result, torch.Tensor):
                _ = float(result.reshape(-1)[0].item())
    return TimedResult(average(times), percentile(times, 50), percentile(times, 95))


def make_cache(
    heads: int,
    queries: int,
    context: int,
    head_dim: int,
    rank_dim: int,
    subspaces: int,
    codewords: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> CacheTensors:
    if rank_dim % subspaces != 0:
        raise ValueError("--rank-dim must be divisible by --coarse-subspaces.")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + context * 13 + queries * 101)
    sub_dim = rank_dim // subspaces
    query = torch.randn(heads, queries, head_dim, device=device, dtype=dtype, generator=generator)
    query_low = torch.randn(heads, queries, rank_dim, device=device, dtype=dtype, generator=generator)
    key = torch.randn(heads, context, head_dim, device=device, dtype=dtype, generator=generator)
    value = torch.randn(heads, context, head_dim, device=device, dtype=dtype, generator=generator)
    key_low = torch.randn(heads, context, rank_dim, device=device, dtype=dtype, generator=generator)
    codebooks = torch.randn(
        heads,
        subspaces,
        codewords,
        sub_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    codes = torch.randint(
        codewords,
        (heads, context, subspaces),
        device=device,
        dtype=code_dtype(codewords),
        generator=generator,
    )
    return CacheTensors(query, query_low, key, value, key_low, codebooks, codes)


def full_decode_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("hqd,hkd->hqk", query, key) * scale
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
    return torch.einsum("hqk,hkd->hqd", weights, value)


def pq_scores_loop(
    query_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
) -> torch.Tensor:
    heads, queries, _ = query_low.shape
    context = codes.shape[1]
    subspaces = codebooks.shape[1]
    sub_dim = rank_dim // subspaces
    q_parts = query_low.float().reshape(heads, queries, subspaces, sub_dim)
    scores = torch.zeros(heads, queries, context, device=query_low.device, dtype=torch.float32)

    for head_idx in range(heads):
        for subspace_idx in range(subspaces):
            table = q_parts[head_idx, :, subspace_idx] @ codebooks[head_idx, subspace_idx].float().T
            scores[head_idx] += table[:, codes[head_idx, :, subspace_idx].long()]

    return scores / math.sqrt(rank_dim)


def pq_scores_vectorized(
    query_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    rank_dim: int,
) -> torch.Tensor:
    heads, queries, _ = query_low.shape
    context = codes.shape[1]
    subspaces = codebooks.shape[1]
    sub_dim = rank_dim // subspaces
    q_parts = query_low.float().reshape(heads, queries, subspaces, sub_dim)
    tables = torch.einsum("hqmd,hmcd->hqmc", q_parts, codebooks.float())
    scores = torch.zeros(heads, queries, context, device=query_low.device, dtype=torch.float32)

    for subspace_idx in range(subspaces):
        table = tables[:, :, subspace_idx, :]
        sub_codes = codes[:, :, subspace_idx].long()
        scores += table.gather(dim=-1, index=sub_codes[:, None, :].expand(heads, queries, context))

    return scores / math.sqrt(rank_dim)


def exact_shortlist_scores(
    query_low: torch.Tensor,
    key_low: torch.Tensor,
    shortlist_idx: torch.Tensor,
) -> torch.Tensor:
    heads, queries, shortlist = shortlist_idx.shape
    rank_dim = key_low.shape[-1]
    selected = key_low[:, None, :, :].expand(heads, queries, key_low.shape[1], rank_dim).gather(
        dim=2,
        index=shortlist_idx[..., None].expand(heads, queries, shortlist, rank_dim),
    )
    return (selected.float() * query_low[:, :, None, :].float()).sum(dim=-1) / math.sqrt(rank_dim)


def verifier_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    final_idx: torch.Tensor,
) -> torch.Tensor:
    heads, queries, budget = final_idx.shape
    head_dim = key.shape[-1]
    selected_keys = key[:, None, :, :].expand(heads, queries, key.shape[1], head_dim).gather(
        dim=2,
        index=final_idx[..., None].expand(heads, queries, budget, head_dim),
    )
    selected_values = value[:, None, :, :].expand(heads, queries, value.shape[1], head_dim).gather(
        dim=2,
        index=final_idx[..., None].expand(heads, queries, budget, head_dim),
    )
    scores = (selected_keys.float() * query[:, :, None, :].float()).sum(dim=-1) / math.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
    return (weights[..., None] * selected_values).sum(dim=-2)


def sva_decode_attention(
    query: torch.Tensor,
    query_low: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_low: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    shortlist: int,
    budget: int,
    rank_dim: int,
    score_impl: str,
) -> torch.Tensor:
    shortlist = min(shortlist, key.shape[1])
    budget = min(budget, shortlist)
    if score_impl == "loop":
        coarse_scores = pq_scores_loop(query_low, codebooks, codes, rank_dim)
    elif score_impl == "vectorized":
        coarse_scores = pq_scores_vectorized(query_low, codebooks, codes, rank_dim)
    else:
        raise ValueError(f"Unknown score implementation: {score_impl}")

    coarse_idx = coarse_scores.topk(shortlist, dim=-1).indices
    rank_scores = exact_shortlist_scores(query_low, key_low, coarse_idx)
    final_idx = coarse_idx.gather(dim=-1, index=rank_scores.topk(budget, dim=-1).indices)
    return verifier_output(query, key, value, final_idx)


def component_timing(
    cache: CacheTensors,
    shortlist: int,
    budget: int,
    rank_dim: int,
    score_impl: str,
    device: torch.device,
) -> ComponentResult:
    shortlist = min(shortlist, cache.key.shape[1])
    budget = min(budget, shortlist)

    synchronize(device)
    start = time.perf_counter()
    if score_impl == "loop":
        coarse_scores = pq_scores_loop(cache.query_low, cache.codebooks, cache.codes, rank_dim)
    else:
        coarse_scores = pq_scores_vectorized(cache.query_low, cache.codebooks, cache.codes, rank_dim)
    synchronize(device)
    after_coarse = time.perf_counter()
    coarse_idx = coarse_scores.topk(shortlist, dim=-1).indices
    synchronize(device)
    after_shortlist = time.perf_counter()
    rank_scores = exact_shortlist_scores(cache.query_low, cache.key_low, coarse_idx)
    final_idx = coarse_idx.gather(dim=-1, index=rank_scores.topk(budget, dim=-1).indices)
    synchronize(device)
    after_rescore = time.perf_counter()
    _ = verifier_output(cache.query, cache.key, cache.value, final_idx)
    synchronize(device)
    after_verify = time.perf_counter()

    return ComponentResult(
        coarse_scan_ms=(after_coarse - start) * 1000.0,
        shortlist_topk_ms=(after_shortlist - after_coarse) * 1000.0,
        exact_rescore_ms=(after_rescore - after_shortlist) * 1000.0,
        verifier_ms=(after_verify - after_rescore) * 1000.0,
        total_ms=(after_verify - start) * 1000.0,
    )


def bytes_for(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def format_float(value: float) -> str:
    return f"{value:.3f}" if math.isfinite(value) else "nan"


def run_one(
    cache: CacheTensors,
    context: int,
    heads: int,
    queries: int,
    head_dim: int,
    rank_dim: int,
    subspaces: int,
    codewords: int,
    shortlist: int,
    budget: int,
    variant: str,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> None:
    if variant == "full":
        full = timed_call(lambda: full_decode_attention(cache.query, cache.key, cache.value), device, warmup, repeats)
        print(
            "million_cached_decode_result,"
            f"{context},{heads},{queries},{head_dim},{rank_dim},{subspaces},{codewords},"
            f"{shortlist},{budget},{variant},"
            f"{bytes_for(cache.key, cache.value)},{bytes_for(cache.key_low, cache.codes)},"
            f"{heads * queries * context * 4},"
            f"{format_float(full.avg)},{format_float(full.p50)},{format_float(full.p95)},"
            "nan,nan,nan,nan,nan,nan,nan,nan",
            flush=True,
        )
        return

    score_impl = "loop" if variant == "sva_loop" else "vectorized"
    fn = lambda: sva_decode_attention(
        cache.query,
        cache.query_low,
        cache.key,
        cache.value,
        cache.key_low,
        cache.codebooks,
        cache.codes,
        shortlist,
        budget,
        rank_dim,
        score_impl,
    )
    if variant == "sva_compile":
        try:
            compiled_fn = torch.compile(fn, mode="reduce-overhead", fullgraph=False)
            fn = compiled_fn
        except Exception as exc:  # pragma: no cover - compile availability is environment-specific.
            print(f"million_cached_decode_compile_unavailable,{type(exc).__name__},{exc}", flush=True)
            return

    try:
        total = timed_call(fn, device, warmup, repeats)
    except Exception as exc:
        print(f"million_cached_decode_variant_failed,{variant},{type(exc).__name__},{exc}", flush=True)
        return

    component_repeats: list[ComponentResult] = []
    with torch.inference_mode():
        for _ in range(max(1, min(repeats, 3))):
            component_repeats.append(component_timing(cache, shortlist, budget, rank_dim, score_impl, device))
    coarse = average([item.coarse_scan_ms for item in component_repeats])
    shortlist_topk = average([item.shortlist_topk_ms for item in component_repeats])
    exact = average([item.exact_rescore_ms for item in component_repeats])
    verifier = average([item.verifier_ms for item in component_repeats])
    component_total = average([item.total_ms for item in component_repeats])

    print(
        "million_cached_decode_result,"
        f"{context},{heads},{queries},{head_dim},{rank_dim},{subspaces},{codewords},"
        f"{shortlist},{budget},{variant},"
        f"{bytes_for(cache.key, cache.value)},{bytes_for(cache.key_low, cache.codes)},"
        f"{heads * queries * context * 4},"
        "nan,nan,nan,"
        f"{format_float(total.avg)},{format_float(total.p50)},{format_float(total.p95)},"
        f"{format_float(coarse)},{format_float(shortlist_topk)},{format_float(exact)},"
        f"{format_float(verifier)},{format_float(component_total)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic million-token cached SVA decode benchmark.")
    parser.add_argument("--contexts", default="8192,65536,262144,1000000")
    parser.add_argument("--heads", type=int, default=9)
    parser.add_argument("--queries", default="1,4,16")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--coarse-subspaces", type=int, default=4)
    parser.add_argument("--coarse-codewords", type=int, default=64)
    parser.add_argument("--shortlists", default="1024,2048")
    parser.add_argument("--budgets", default="256,512")
    parser.add_argument("--variants", default="full,sva_vectorized,sva_compile")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    contexts = comma_ints(args.contexts)
    queries_values = comma_ints(args.queries)
    shortlists = comma_ints(args.shortlists)
    budgets = comma_ints(args.budgets)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    print("metric,value")
    print(f"contexts,{args.contexts}")
    print(f"heads,{args.heads}")
    print(f"queries,{args.queries}")
    print(f"head_dim,{args.head_dim}")
    print(f"rank_dim,{args.rank_dim}")
    print(f"coarse_subspaces,{args.coarse_subspaces}")
    print(f"coarse_codewords,{args.coarse_codewords}")
    print(f"shortlists,{args.shortlists}")
    print(f"budgets,{args.budgets}")
    print(f"variants,{args.variants}")
    print(f"warmup,{args.warmup}")
    print(f"repeats,{args.repeats}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(
        "million_cached_decode_header,"
        "context,heads,queries,head_dim,rank_dim,coarse_subspaces,coarse_codewords,"
        "shortlist,budget,variant,key_value_bytes,catalog_bytes,score_bytes,"
        "full_ms_avg,full_ms_p50,full_ms_p95,"
        "sva_ms_avg,sva_ms_p50,sva_ms_p95,"
        "coarse_scan_ms,shortlist_topk_ms,exact_rescore_ms,verifier_ms,component_total_ms",
        flush=True,
    )

    for context in contexts:
        for queries in queries_values:
            cache = make_cache(
                args.heads,
                queries,
                context,
                args.head_dim,
                args.rank_dim,
                args.coarse_subspaces,
                args.coarse_codewords,
                device,
                dtype,
                args.seed,
            )
            for shortlist in shortlists:
                for budget in budgets:
                    if budget > shortlist:
                        continue
                    for variant in variants:
                        run_one(
                            cache,
                            context,
                            args.heads,
                            queries,
                            args.head_dim,
                            args.rank_dim,
                            args.coarse_subspaces,
                            args.coarse_codewords,
                            shortlist,
                            budget,
                            variant,
                            args.warmup,
                            args.repeats,
                            device,
                        )
            del cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("million_cached_decode_done", flush=True)


if __name__ == "__main__":
    main()
