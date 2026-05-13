"""Kill test for Summon-Verify Attention.

Pages write themselves into several content-addressed LSH tables. A query
activates the same addresses, producing a high-recall candidate set. A verifier
then performs exact dot-product attention over only those candidates.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from itertools import combinations

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


def sample_queries(
    keys: np.ndarray,
    n_queries: int,
    query_noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    targets = rng.integers(0, keys.shape[0], size=n_queries)
    queries = normalize(keys[targets] + query_noise * rng.normal(size=(n_queries, keys.shape[1])))
    return queries, targets


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
    return build_lsh_tables_from_projections(keys, projections)


def build_lsh_tables_from_projections(
    keys: np.ndarray,
    projections: np.ndarray,
) -> tuple[list[dict[int, list[int]]], np.ndarray]:
    tables: list[dict[int, list[int]]] = []
    for table_idx in range(projections.shape[0]):
        codes = lsh_codes(keys, projections[table_idx])
        table: dict[int, list[int]] = defaultdict(list)
        for item_idx, code in enumerate(codes):
            table[int(code)].append(item_idx)
        tables.append(table)
    return tables, projections


def table_train_stats(
    table: dict[int, list[int]],
    projection: np.ndarray,
    train_queries: np.ndarray,
    train_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hits = np.zeros(train_queries.shape[0], dtype=bool)
    counts = np.zeros(train_queries.shape[0], dtype=np.float64)
    for idx, (query, target) in enumerate(zip(train_queries, train_targets)):
        code = int(lsh_codes(query[None, :], projection)[0])
        bucket = table.get(code, ())
        hits[idx] = target in bucket
        counts[idx] = len(bucket)
    return hits, counts


def build_selected_lsh_tables(
    keys: np.ndarray,
    n_tables: int,
    n_bits: int,
    pool_size: int,
    train_queries: np.ndarray,
    train_targets: np.ndarray,
    count_penalty: float,
    rng: np.random.Generator,
) -> tuple[list[dict[int, list[int]]], np.ndarray]:
    pool_size = max(n_tables, pool_size)
    candidate_tables = []
    for _ in range(pool_size):
        projection = rng.normal(size=(n_bits, keys.shape[1])) / math.sqrt(keys.shape[1])
        tables, _ = build_lsh_tables_from_projections(keys, projection[None, :, :])
        hits, counts = table_train_stats(tables[0], projection, train_queries, train_targets)
        candidate_tables.append((tables[0], projection, hits, counts))

    selected = []
    selected_ids: set[int] = set()
    covered = np.zeros(train_queries.shape[0], dtype=bool)
    for _ in range(n_tables):
        best_id = -1
        best_score = -float("inf")
        for candidate_id, (_table, _projection, hits, counts) in enumerate(candidate_tables):
            if candidate_id in selected_ids:
                continue
            new_hits = np.logical_and(hits, np.logical_not(covered))
            score = float(new_hits.sum()) - count_penalty * float(counts.mean())
            if score > best_score:
                best_id = candidate_id
                best_score = score
        if best_id < 0:
            break
        selected_ids.add(best_id)
        table, projection, hits, _counts = candidate_tables[best_id]
        selected.append((table, projection))
        covered = np.logical_or(covered, hits)

    tables = [table for table, _projection in selected]
    projections = np.stack([projection for _table, projection in selected])
    return tables, projections


def lsh_candidates(
    query: np.ndarray,
    tables: list[dict[int, list[int]]],
    projections: np.ndarray,
    probe_radius: int = 0,
) -> np.ndarray:
    candidates: set[int] = set()
    for table, projection in zip(tables, projections):
        code = int(lsh_codes(query[None, :], projection)[0])
        for near_code in nearby_codes(code, projection.shape[0], probe_radius):
            candidates.update(table.get(near_code, ()))
    if not candidates:
        return np.empty(0, dtype=np.int64)
    return np.fromiter(candidates, dtype=np.int64)


def nearby_codes(code: int, n_bits: int, radius: int) -> list[int]:
    codes = [code]
    if radius <= 0:
        return codes
    bit_masks = [1 << bit for bit in range(n_bits)]
    max_radius = min(radius, n_bits)
    for distance in range(1, max_radius + 1):
        for masks in combinations(bit_masks, distance):
            flip = 0
            for mask in masks:
                flip ^= mask
            codes.append(code ^ flip)
    return codes


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
    raw_counts = []
    for query, target in zip(queries, targets):
        candidates = get_candidates(query)
        recall.append(target in set(candidates.tolist()))
        raw_counts.append(candidates.size)
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
        "avg_summoned": float(np.mean(raw_counts)),
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
            "avg_summoned": float(keys.shape[0]),
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
        for probe_radius in args.probe_radii:
            if probe_radius <= 0:
                continue
            probe_result = evaluate_candidate_method(
                keys,
                values,
                queries,
                targets,
                teacher,
                lambda q, tables=tables, projections=projections, probe_radius=probe_radius: lsh_candidates(
                    q,
                    tables,
                    projections,
                    probe_radius,
                ),
                args.budget,
                args.logit_scale,
            )
            rows.append({"method": f"sva_probe{probe_radius}_{n_tables}x{args.bits}", **probe_result})
        if args.learned_pool > 0:
            train_queries, train_targets = sample_queries(
                keys,
                args.learned_train_queries,
                args.query_noise,
                rng,
            )
            selected_tables, selected_projections = build_selected_lsh_tables(
                keys,
                n_tables,
                args.bits,
                args.learned_pool,
                train_queries,
                train_targets,
                args.selection_count_penalty,
                rng,
            )
            selected_result = evaluate_candidate_method(
                keys,
                values,
                queries,
                targets,
                teacher,
                lambda q, tables=selected_tables, projections=selected_projections: lsh_candidates(
                    q,
                    tables,
                    projections,
                ),
                args.budget,
                args.logit_scale,
            )
            rows.append({"method": f"sva_selected_{n_tables}x{args.bits}", **selected_result})
            for probe_radius in args.probe_radii:
                if probe_radius <= 0:
                    continue
                selected_probe_result = evaluate_candidate_method(
                    keys,
                    values,
                    queries,
                    targets,
                    teacher,
                    lambda q, tables=selected_tables, projections=selected_projections, probe_radius=probe_radius: lsh_candidates(
                        q,
                        tables,
                        projections,
                        probe_radius,
                    ),
                    args.budget,
                    args.logit_scale,
                )
                rows.append(
                    {
                        "method": f"sva_selected_probe{probe_radius}_{n_tables}x{args.bits}",
                        **selected_probe_result,
                    }
                )
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
    parser.add_argument("--learned-pool", type=int, default=0)
    parser.add_argument("--learned-train-queries", type=int, default=2048)
    parser.add_argument("--selection-count-penalty", type=float, default=0.0)
    parser.add_argument("--probe-radii", type=int, nargs="*", default=[])
    args = parser.parse_args()

    all_rows = []
    for trial in range(args.trials):
        all_rows.extend(run_once(args, args.seed + 1000 * trial))

    methods = list(dict.fromkeys(row["method"] for row in all_rows))
    print("method,recall,top1,cos_teacher,mse_teacher,avg_summoned,avg_candidates")
    for method in methods:
        rows = [row for row in all_rows if row["method"] == method]
        print(
            f"{method},"
            f"{np.mean([float(r['recall']) for r in rows]):.4f},"
            f"{np.mean([float(r['top1']) for r in rows]):.4f},"
            f"{np.mean([float(r['cos_teacher']) for r in rows]):.4f},"
            f"{np.mean([float(r['mse_teacher']) for r in rows]):.6f},"
            f"{np.mean([float(r['avg_summoned']) for r in rows]):.1f},"
            f"{np.mean([float(r['avg_candidates']) for r in rows]):.1f}"
        )


if __name__ == "__main__":
    main()
