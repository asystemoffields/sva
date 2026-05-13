"""Synthetic throughput benchmark for product-quantized SVA score scans."""

from __future__ import annotations

import argparse
import time

import torch


def comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    return torch.randint(codewords, (heads, context, subspaces), device=device, dtype=torch.long)


def pq_scan_scores(tables: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    heads, queries, subspaces, _ = tables.shape
    context = codes.shape[1]
    scores = torch.zeros(heads, queries, context, device=tables.device, dtype=torch.float32)

    for subspace_idx in range(subspaces):
        scores += tables[:, :, subspace_idx, :].float().gather(
            dim=-1,
            index=codes[:, None, :, subspace_idx].expand(heads, queries, context),
        )

    return scores


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q / 100.0).item())


def benchmark_config(
    context: int,
    heads: int,
    queries: int,
    subspaces: int,
    codewords: int,
    budget: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    tables = make_tables(heads, queries, subspaces, codewords, device, dtype)
    codes = make_codes(heads, context, subspaces, codewords, device)

    for _ in range(warmup):
        scores = pq_scan_scores(tables, codes)
        _ = scores.topk(min(budget, context), dim=-1).indices
    synchronize(device)

    scan_times: list[float] = []
    topk_times: list[float] = []
    total_times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        scores = pq_scan_scores(tables, codes)
        synchronize(device)
        after_scan = time.perf_counter()
        _ = scores.topk(min(budget, context), dim=-1).indices
        synchronize(device)
        after_topk = time.perf_counter()
        scan_times.append((after_scan - start) * 1000.0)
        topk_times.append((after_topk - after_scan) * 1000.0)
        total_times.append((after_topk - start) * 1000.0)

    ideal_bits_per_key = subspaces * max(1, (codewords - 1).bit_length())
    scanned_codes = heads * queries * context * subspaces
    print(
        "pq_scan_result,"
        f"{context},{heads},{queries},{subspaces},{codewords},{ideal_bits_per_key},{budget},"
        f"{scanned_codes},"
        f"{sum(scan_times) / len(scan_times):.3f},{percentile(scan_times, 50):.3f},{percentile(scan_times, 95):.3f},"
        f"{sum(topk_times) / len(topk_times):.3f},{percentile(topk_times, 50):.3f},{percentile(topk_times, 95):.3f},"
        f"{sum(total_times) / len(total_times):.3f},{percentile(total_times, 50):.3f},{percentile(total_times, 95):.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark synthetic PQ score scan throughput.")
    parser.add_argument("--context", type=int, default=1_000_000)
    parser.add_argument("--heads", type=int, default=9)
    parser.add_argument("--queries", default="1")
    parser.add_argument("--subspaces", default="8,16")
    parser.add_argument("--codewords", default="256")
    parser.add_argument("--budgets", default="512")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
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
    print(f"subspaces,{args.subspaces}")
    print(f"codewords,{args.codewords}")
    print(f"budgets,{args.budgets}")
    print(f"warmup,{args.warmup}")
    print(f"repeats,{args.repeats}")
    print(f"device,{device}")
    print(f"dtype,{dtype}")
    print(
        "pq_scan_header,"
        "context,heads,queries,subspaces,codewords,bits_per_key,budget,scanned_code_lookups,"
        "scan_ms_avg,scan_ms_p50,scan_ms_p95,"
        "topk_ms_avg,topk_ms_p50,topk_ms_p95,"
        "total_ms_avg,total_ms_p50,total_ms_p95"
    )

    for queries in comma_ints(args.queries):
        for subspaces in comma_ints(args.subspaces):
            for codewords in comma_ints(args.codewords):
                for budget in comma_ints(args.budgets):
                    benchmark_config(
                        args.context,
                        args.heads,
                        queries,
                        subspaces,
                        codewords,
                        budget,
                        args.warmup,
                        args.repeats,
                        device,
                        dtype,
                    )


if __name__ == "__main__":
    main()
