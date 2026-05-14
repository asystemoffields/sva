"""Synthetic throughput benchmark for coarse PQ plus exact low-rank rescoring."""

from __future__ import annotations

import argparse
import time

import torch


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def pq_configs(value: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        left, right = item.split("x", 1)
        configs.append((int(left), int(right)))
    return configs


def code_dtype(codewords: int) -> torch.dtype:
    if codewords <= 256:
        return torch.uint8
    if codewords <= 32768:
        return torch.int16
    return torch.int32


def make_tables(
    heads: int,
    queries: int,
    subspaces: int,
    codewords: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(heads, queries, subspaces, codewords, device=device, dtype=dtype)


def make_codes(
    heads: int,
    context: int,
    subspaces: int,
    codewords: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(codewords, (heads, context, subspaces), device=device, dtype=code_dtype(codewords))


def pq_scan_scores(tables: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    heads, queries, subspaces, _ = tables.shape
    context = codes.shape[1]
    scores = torch.zeros(heads, queries, context, device=tables.device, dtype=torch.float32)

    for subspace_idx in range(subspaces):
        sub_codes = codes[:, :, subspace_idx].long()
        scores += tables[:, :, subspace_idx, :].float().gather(
            dim=-1,
            index=sub_codes[:, None, :].expand(heads, queries, context),
        )

    return scores


def exact_shortlist_scores(
    query_low: torch.Tensor,
    key_low: torch.Tensor,
    shortlist_idx: torch.Tensor,
) -> torch.Tensor:
    heads, queries, shortlist = shortlist_idx.shape
    rank_dim = key_low.shape[-1]
    expanded_keys = key_low[:, None, :, :].expand(heads, queries, key_low.shape[1], rank_dim)
    selected_keys = expanded_keys.gather(
        dim=2,
        index=shortlist_idx[..., None].expand(heads, queries, shortlist, rank_dim),
    )
    return (selected_keys.float() * query_low[:, :, None, :].float()).sum(dim=-1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q / 100.0).item())


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def benchmark_config(
    context: int,
    heads: int,
    queries: int,
    rank_dim: int,
    coarse_subspaces: int,
    coarse_codewords: int,
    shortlist: int,
    budget: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    coarse_tables = make_tables(heads, queries, coarse_subspaces, coarse_codewords, device, dtype)
    coarse_codes = make_codes(heads, context, coarse_subspaces, coarse_codewords, device)
    key_low = torch.randn(heads, context, rank_dim, device=device, dtype=dtype)
    query_low = torch.randn(heads, queries, rank_dim, device=device, dtype=dtype)
    shortlist = min(shortlist, context)
    budget = min(budget, shortlist)

    for _ in range(warmup):
        coarse_scores = pq_scan_scores(coarse_tables, coarse_codes)
        coarse_idx = coarse_scores.topk(shortlist, dim=-1).indices
        exact_scores = exact_shortlist_scores(query_low, key_low, coarse_idx)
        _ = exact_scores.topk(budget, dim=-1).indices
    synchronize(device)

    coarse_scan_times: list[float] = []
    shortlist_topk_times: list[float] = []
    exact_rescore_times: list[float] = []
    verify_topk_times: list[float] = []
    total_times: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        coarse_scores = pq_scan_scores(coarse_tables, coarse_codes)
        synchronize(device)
        after_coarse = time.perf_counter()
        coarse_idx = coarse_scores.topk(shortlist, dim=-1).indices
        synchronize(device)
        after_shortlist = time.perf_counter()
        exact_scores = exact_shortlist_scores(query_low, key_low, coarse_idx)
        synchronize(device)
        after_exact = time.perf_counter()
        _ = exact_scores.topk(budget, dim=-1).indices
        synchronize(device)
        after_verify = time.perf_counter()

        coarse_scan_times.append((after_coarse - start) * 1000.0)
        shortlist_topk_times.append((after_shortlist - after_coarse) * 1000.0)
        exact_rescore_times.append((after_exact - after_shortlist) * 1000.0)
        verify_topk_times.append((after_verify - after_exact) * 1000.0)
        total_times.append((after_verify - start) * 1000.0)

    coarse_bits_per_key = coarse_subspaces * max(1, (coarse_codewords - 1).bit_length())
    coarse_code_lookups = heads * queries * context * coarse_subspaces
    exact_dot_products = heads * queries * shortlist
    exact_multiply_adds = exact_dot_products * rank_dim
    key_bytes = heads * context * rank_dim * key_low.element_size()
    print(
        "coarse_exact_rescore_result,"
        f"{context},{heads},{queries},{rank_dim},"
        f"{coarse_subspaces},{coarse_codewords},{coarse_bits_per_key},"
        f"{shortlist},{budget},"
        f"{coarse_code_lookups},{exact_dot_products},{exact_multiply_adds},{key_bytes},"
        f"{average(coarse_scan_times):.3f},{percentile(coarse_scan_times, 50):.3f},{percentile(coarse_scan_times, 95):.3f},"
        f"{average(shortlist_topk_times):.3f},{percentile(shortlist_topk_times, 50):.3f},{percentile(shortlist_topk_times, 95):.3f},"
        f"{average(exact_rescore_times):.3f},{percentile(exact_rescore_times, 50):.3f},{percentile(exact_rescore_times, 95):.3f},"
        f"{average(verify_topk_times):.3f},{percentile(verify_topk_times, 50):.3f},{percentile(verify_topk_times, 95):.3f},"
        f"{average(total_times):.3f},{percentile(total_times, 50):.3f},{percentile(total_times, 95):.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark coarse PQ plus exact low-rank rescoring.")
    parser.add_argument("--context", type=int, default=1_000_000)
    parser.add_argument("--heads", type=int, default=9)
    parser.add_argument("--queries", default="1")
    parser.add_argument("--rank-dims", default="64")
    parser.add_argument("--coarse-configs", default="4x64")
    parser.add_argument("--shortlists", default="1024,1536,2048")
    parser.add_argument("--budgets", default="512")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    print("metric,value")
    print(f"context,{args.context}")
    print(f"heads,{args.heads}")
    print(f"queries,{args.queries}")
    print(f"rank_dims,{args.rank_dims}")
    print(f"coarse_configs,{args.coarse_configs}")
    print(f"shortlists,{args.shortlists}")
    print(f"budgets,{args.budgets}")
    print(f"warmup,{args.warmup}")
    print(f"repeats,{args.repeats}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(
        "coarse_exact_rescore_header,"
        "context,heads,queries,rank_dim,"
        "coarse_subspaces,coarse_codewords,coarse_bits_per_key,"
        "shortlist,budget,"
        "coarse_code_lookups,exact_dot_products,exact_multiply_adds,key_bytes,"
        "coarse_scan_ms_avg,coarse_scan_ms_p50,coarse_scan_ms_p95,"
        "shortlist_topk_ms_avg,shortlist_topk_ms_p50,shortlist_topk_ms_p95,"
        "exact_rescore_ms_avg,exact_rescore_ms_p50,exact_rescore_ms_p95,"
        "verify_topk_ms_avg,verify_topk_ms_p50,verify_topk_ms_p95,"
        "total_ms_avg,total_ms_p50,total_ms_p95"
    )

    for queries in comma_ints(args.queries):
        for rank_dim in comma_ints(args.rank_dims):
            for coarse_subspaces, coarse_codewords in pq_configs(args.coarse_configs):
                for shortlist in comma_ints(args.shortlists):
                    for budget in comma_ints(args.budgets):
                        benchmark_config(
                            args.context,
                            args.heads,
                            queries,
                            rank_dim,
                            coarse_subspaces,
                            coarse_codewords,
                            shortlist,
                            budget,
                            args.warmup,
                            args.repeats,
                            device,
                            dtype,
                        )


if __name__ == "__main__":
    main()
