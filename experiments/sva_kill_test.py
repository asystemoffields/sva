"""Kill test for Summon-Verify Attention.

Pages write themselves into several content-addressed LSH tables. A query
activates the same addresses, producing a high-recall candidate set. A verifier
then performs exact dot-product attention over only those candidates.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np


def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def make_random_task(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    centers = normalize(rng.normal(size=(args.n_clusters, args.key_dim)))
    assignments = rng.integers(0, args.n_clusters, size=args.n_items)
    keys = normalize(centers[assignments] + args.key_noise * rng.normal(size=(args.n_items, args.key_dim)))
    values = normalize(rng.normal(size=(args.n_items, args.value_dim)))
    targets = rng.integers(0, args.n_items, size=args.n_queries)
    queries = normalize(keys[targets] + args.query_noise * rng.normal(size=(args.n_queries, args.key_dim)))
    return keys, values, queries, targets


def make_binding_task(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    centers = normalize(rng.normal(size=(args.n_clusters, args.entity_dim)))
    assignments = rng.integers(0, args.n_clusters, size=args.n_entities)
    entities = normalize(
        centers[assignments]
        + args.key_noise * rng.normal(size=(args.n_entities, args.entity_dim))
    )
    attrs = normalize(rng.normal(size=(args.n_attrs, args.attr_dim)))
    keys = []
    for entity in entities:
        for attr in attrs:
            keys.append(normalize(np.concatenate([entity, args.attr_scale * attr])))
    keys = np.stack(keys)
    values = normalize(rng.normal(size=(keys.shape[0], args.value_dim)))
    targets = rng.integers(0, keys.shape[0], size=args.n_queries)
    queries = normalize(keys[targets] + args.query_noise * rng.normal(size=(args.n_queries, keys.shape[1])))
    return keys, values, queries, targets


def lsh_codes(vectors: np.ndarray, projections: np.ndarray) -> np.ndarray:
    bits = (vectors @ projections.T) > 0
    powers = (1 << np.arange(projections.shape[0], dtype=np.int64))
    return bits.astype(np.int64) @ powers


def build_lsh_tables(
    keys: np.ndarray,
    n_tables: int,
    n_bits: int,
    rng: np.random.Generator,
) -> tuple[list[dict[int, list[int]]], np.ndarray]:
    projections = rng.normal(size=(n_tables, n_bits, keys.shape[1])) / math.sqrt(keys.shape[1])
    tables: list[dict[int, list[int]]] = []
    for table_idx in range(n_tables):
        codes = lsh_codes(keys, projections[table_idx])
        table: dict[int, list[int]] = defaultdict(list)
        for item_idx, code in enumerate(codes):
            table[int(code)].append(item_idx)
        tables.append(table)
    return tables, projections


def lsh_candidates(
    query: np.ndarray,
    tables: list[dict[int, list[int]]],
    projections: np.ndarray,
) -> np.ndarray:
    candidates: set[int] = set()
    for table, projection in zip(tables, projections):
        code = int(lsh_codes(query[None, :], projection)[0])
        candidates.update(table.get(code, ()))
    if not candidates:
        return np.empty(0, dtype=np.int64)
    return np.fromiter(candidates, dtype=np.int64)


def bank_candidates(
    query: np.ndarray,
    banks: list[list[int]],
    projection: np.ndarray,
) -> np.ndarray:
    bank = int(np.argmax(projection @ query))
    return np.array(banks[bank], dtype=np.int64)


def build_banks(keys: np.ndarray, n_banks: int, rng: np.random.Generator) -> tuple[list[list[int]], np.ndarray]:
    projection = rng.normal(size=(n_banks, keys.shape[1])) / math.sqrt(keys.shape[1])
    assignments = np.argmax(keys @ projection.T, axis=1)
    banks: list[list[int]] = [[] for _ in range(n_banks)]
    for item_idx, bank in enumerate(assignments):
        banks[int(bank)].append(item_idx)
    return banks, projection


def exact_attention(
    keys: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    logit_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = logit_scale * (queries @ keys.T)
    weights = softmax(scores, axis=1)
    return weights @ values, scores.argmax(axis=1)


def candidate_attention(
    keys: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    candidates: np.ndarray,
    budget: int,
    logit_scale: float,
) -> tuple[np.ndarray, int, int]:
    if candidates.size == 0:
        return np.zeros(values.shape[1]), -1, 0
    scores = logit_scale * (query @ keys[candidates].T)
    order = np.argsort(-scores)
    if budget > 0:
        order = order[:budget]
    chosen = candidates[order]
    chosen_scores = scores[order]
    weights = softmax(chosen_scores[None, :], axis=1)[0]
    return weights @ values[chosen], int(chosen[0]), int(chosen.size)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sum(normalize(a) * normalize(b), axis=1)))


def evaluate_candidate_method(
    keys: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    targets: np.ndarray,
    teacher: np.ndarray,
    get_candidates,
    budget: int,
    logit_scale: float,
) -> dict[str, float]:
    outputs = []
    top1 = []
    recall = []
    counts = []
    for query, target in zip(queries, targets):
        candidates = get_candidates(query)
        recall.append(target in set(candidates.tolist()))
        output, winner, count = candidate_attention(
            keys,
            values,
            query,
            candidates,
            budget,
            logit_scale,
        )
        outputs.append(output)
        top1.append(winner == target)
        counts.append(count)
    outputs_arr = np.stack(outputs)
    return {
        "recall": float(np.mean(recall)),
        "top1": float(np.mean(top1)),
        "cos_teacher": cosine(outputs_arr, teacher),
        "mse_teacher": float(np.mean((outputs_arr - teacher) ** 2)),
        "avg_candidates": float(np.mean(counts)),
    }


def evaluate_random_baseline(
    keys: np.ndarray,
    values: np.ndarray,
    queries: np.ndarray,
    targets: np.ndarray,
    teacher: np.ndarray,
    avg_count: int,
    budget: int,
    logit_scale: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    avg_count = max(1, min(avg_count, keys.shape[0]))

    def get_candidates(_query: np.ndarray) -> np.ndarray:
        return rng.choice(keys.shape[0], size=avg_count, replace=False)

    return evaluate_candidate_method(
        keys,
        values,
        queries,
        targets,
        teacher,
        get_candidates,
        budget,
        logit_scale,
    )


def run_once(args: argparse.Namespace, seed: int) -> list[dict[str, str | float]]:
    rng = np.random.default_rng(seed)
    if args.task == "binding":
        keys, values, queries, targets = make_binding_task(args, rng)
    else:
        keys, values, queries, targets = make_random_task(args, rng)

    teacher, teacher_top = exact_attention(keys, values, queries, args.logit_scale)
    rows: list[dict[str, str | float]] = [
        {
            "method": "full_attention",
            "recall": float((teacher_top == targets).mean()),
            "top1": float((teacher_top == targets).mean()),
            "cos_teacher": 1.0,
            "mse_teacher": 0.0,
            "avg_candidates": float(keys.shape[0]),
        }
    ]

    banks, bank_projection = build_banks(keys, args.n_banks, rng)
    bank_result = evaluate_candidate_method(
        keys,
        values,
        queries,
        targets,
        teacher,
        lambda q: bank_candidates(q, banks, bank_projection),
        args.budget,
        args.logit_scale,
    )
    rows.append({"method": "coarse_bank_verify", **bank_result})

    for n_tables in args.tables:
        tables, projections = build_lsh_tables(keys, n_tables, args.bits, rng)
        lsh_result = evaluate_candidate_method(
            keys,
            values,
            queries,
            targets,
            teacher,
            lambda q, tables=tables, projections=projections: lsh_candidates(q, tables, projections),
            args.budget,
            args.logit_scale,
        )
        rows.append({"method": f"sva_{n_tables}x{args.bits}", **lsh_result})
        random_result = evaluate_random_baseline(
            keys,
            values,
            queries,
            targets,
            teacher,
            round(lsh_result["avg_candidates"]),
            args.budget,
            args.logit_scale,
            rng,
        )
        rows.append({"method": f"random_{n_tables}x{args.bits}_budget", **random_result})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Toy kill test for Summon-Verify Attention."
    )
    parser.add_argument("--task", choices=["random", "binding"], default="random")
    parser.add_argument("--n-items", type=int, default=4096)
    parser.add_argument("--n-queries", type=int, default=1024)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--key-noise", type=float, default=0.08)
    parser.add_argument("--query-noise", type=float, default=0.01)
    parser.add_argument("--n-entities", type=int, default=512)
    parser.add_argument("--n-attrs", type=int, default=8)
    parser.add_argument("--entity-dim", type=int, default=32)
    parser.add_argument("--attr-dim", type=int, default=32)
    parser.add_argument("--attr-scale", type=float, default=0.7)
    parser.add_argument("--n-banks", type=int, default=128)
    parser.add_argument("--tables", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--logit-scale", type=float, default=16.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    all_rows = []
    for trial in range(args.trials):
        all_rows.extend(run_once(args, args.seed + 1000 * trial))

    methods = list(dict.fromkeys(row["method"] for row in all_rows))
    print("method,recall,top1,cos_teacher,mse_teacher,avg_candidates")
    for method in methods:
        rows = [row for row in all_rows if row["method"] == method]
        print(
            f"{method},"
            f"{np.mean([float(r['recall']) for r in rows]):.4f},"
            f"{np.mean([float(r['top1']) for r in rows]):.4f},"
            f"{np.mean([float(r['cos_teacher']) for r in rows]):.4f},"
            f"{np.mean([float(r['mse_teacher']) for r in rows]):.6f},"
            f"{np.mean([float(r['avg_candidates']) for r in rows]):.1f}"
        )


if __name__ == "__main__":
    main()
