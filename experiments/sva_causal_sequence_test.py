"""Causal sequence kill test for Summon-Verify Attention.

This test moves SVA from static lookup into an autoregressive cache setting.
Each token writes a key/value page into an incremental SVA table. Later queries
must recover one prior page using only pages already written.
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sum(normalize(a) * normalize(b), axis=1)))


def lsh_code(vector: np.ndarray, projection: np.ndarray) -> int:
    bits = (projection @ vector) > 0
    powers = 1 << np.arange(projection.shape[0], dtype=np.int64)
    return int(bits.astype(np.int64) @ powers)


def nearby_codes(code: int, n_bits: int, radius: int) -> list[int]:
    codes = [code]
    if radius <= 0:
        return codes
    bit_masks = [1 << bit for bit in range(n_bits)]
    for distance in range(1, min(radius, n_bits) + 1):
        for masks in combinations(bit_masks, distance):
            flip = 0
            for mask in masks:
                flip ^= mask
            codes.append(code ^ flip)
    return codes


def make_sequence(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    centers = normalize(rng.normal(size=(args.n_clusters, args.key_dim)))
    assignments = rng.integers(0, args.n_clusters, size=args.seq_len)
    keys = normalize(
        centers[assignments]
        + args.key_noise * rng.normal(size=(args.seq_len, args.key_dim))
    )
    values = normalize(rng.normal(size=(args.seq_len, args.value_dim)))
    return keys, values


def choose_target(timestep: int, args: argparse.Namespace, rng: np.random.Generator) -> int:
    if args.target_mode == "uniform":
        return int(rng.integers(0, timestep))
    max_age = min(args.max_age, timestep)
    age = int(rng.integers(1, max_age + 1))
    return timestep - age


def make_dataset(
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    dataset = []
    for _ in range(args.n_sequences):
        keys, values = make_sequence(args, rng)
        queries = []
        targets = []
        for timestep in range(1, args.seq_len):
            target = choose_target(timestep, args, rng)
            query = normalize(keys[target] + args.query_noise * rng.normal(size=args.key_dim))
            queries.append(query)
            targets.append(target)
        dataset.append((keys, values, np.stack(queries), np.array(targets, dtype=np.int64)))
    return dataset


def build_empty_tables(n_tables: int) -> list[dict[int, list[int]]]:
    return [defaultdict(list) for _ in range(n_tables)]


def write_page(
    tables: list[dict[int, list[int]]],
    projections: np.ndarray,
    key: np.ndarray,
    position: int,
) -> None:
    for table, projection in zip(tables, projections):
        table[lsh_code(key, projection)].append(position)


def get_candidates(
    tables: list[dict[int, list[int]]],
    projections: np.ndarray,
    query: np.ndarray,
    probe_radius: int,
) -> np.ndarray:
    candidates: set[int] = set()
    for table, projection in zip(tables, projections):
        code = lsh_code(query, projection)
        for near_code in nearby_codes(code, projection.shape[0], probe_radius):
            candidates.update(table.get(near_code, ()))
    if not candidates:
        return np.empty(0, dtype=np.int64)
    return np.fromiter(candidates, dtype=np.int64)


def exact_attention(
    keys: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    logit_scale: float,
) -> tuple[np.ndarray, int]:
    scores = logit_scale * (query @ keys.T)
    weights = softmax(scores[None, :], axis=1)[0]
    return weights @ values, int(scores.argmax())


def candidate_attention(
    keys: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    candidates: np.ndarray,
    budget: int,
    logit_scale: float,
    prefilter_projection: np.ndarray | None,
    prefilter_key_features: np.ndarray | None,
    prefilter_budget: int,
) -> tuple[np.ndarray, int, int, int]:
    if candidates.size == 0:
        return np.zeros(values.shape[1]), -1, 0, 0
    exact_candidates = candidates
    if (
        prefilter_projection is not None
        and prefilter_key_features is not None
        and prefilter_budget > 0
        and candidates.size > prefilter_budget
    ):
        query_features = query @ prefilter_projection.T
        cheap_scores = query_features @ prefilter_key_features[candidates].T
        exact_candidates = candidates[np.argsort(-cheap_scores)[:prefilter_budget]]

    scores = logit_scale * (query @ keys[exact_candidates].T)
    order = np.argsort(-scores)
    if budget > 0:
        order = order[:budget]
    chosen = exact_candidates[order]
    chosen_scores = scores[order]
    weights = softmax(chosen_scores[None, :], axis=1)[0]
    return weights @ values[chosen], int(chosen[0]), int(chosen.size), int(exact_candidates.size)


def summarize(name: str, rows: list[dict[str, float]]) -> dict[str, str | float]:
    return {
        "method": name,
        "teacher_top_target": float(np.mean([row["teacher_top_target"] for row in rows])),
        "raw_recall": float(np.mean([row["raw_recall"] for row in rows])),
        "top1_target": float(np.mean([row["top1_target"] for row in rows])),
        "top1_teacher": float(np.mean([row["top1_teacher"] for row in rows])),
        "cos_teacher": float(cosine(
            np.stack([row["output"] for row in rows]),
            np.stack([row["teacher"] for row in rows]),
        )),
        "avg_prefix": float(np.mean([row["prefix_len"] for row in rows])),
        "avg_summoned": float(np.mean([row["summoned"] for row in rows])),
        "avg_exact_scored": float(np.mean([row["exact_scored"] for row in rows])),
        "avg_verified": float(np.mean([row["verified"] for row in rows])),
    }


def run_variant(
    args: argparse.Namespace,
    n_tables: int,
    probe_radius: int,
    use_prefilter: bool,
    dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> dict[str, str | float]:
    projections = rng.normal(size=(n_tables, args.bits, args.key_dim)) / math.sqrt(args.key_dim)
    prefilter_projection = None
    if use_prefilter and args.prefilter_dim > 0:
        prefilter_projection = rng.normal(size=(args.prefilter_dim, args.key_dim)) / math.sqrt(args.key_dim)

    rows: list[dict[str, float]] = []
    for keys, values, queries, targets in dataset:
        prefilter_key_features = None
        if prefilter_projection is not None:
            prefilter_key_features = keys @ prefilter_projection.T
        tables = build_empty_tables(n_tables)
        write_page(tables, projections, keys[0], 0)
        for timestep, (query, target) in enumerate(zip(queries, targets), start=1):
            teacher, teacher_top = exact_attention(
                keys[:timestep],
                values[:timestep],
                query,
                args.logit_scale,
            )
            candidates = get_candidates(tables, projections, query, probe_radius)
            output, winner, verified, exact_scored = candidate_attention(
                keys,
                values,
                query,
                candidates,
                args.budget,
                args.logit_scale,
                prefilter_projection,
                prefilter_key_features,
                args.prefilter_budget,
            )
            rows.append(
                {
                    "teacher_top_target": float(teacher_top == target),
                    "raw_recall": float(target in set(candidates.tolist())),
                    "top1_target": float(winner == target),
                    "top1_teacher": float(winner == teacher_top),
                    "teacher": teacher,
                    "output": output,
                    "prefix_len": float(timestep),
                    "summoned": float(candidates.size),
                    "exact_scored": float(exact_scored),
                    "verified": float(verified),
                }
            )
            write_page(tables, projections, keys[timestep], timestep)

    suffix = f"{n_tables}x{args.bits}"
    if probe_radius > 0 and use_prefilter:
        name = f"sva_causal_probe{probe_radius}_prefilter{args.prefilter_dim}d{args.prefilter_budget}_{suffix}"
    elif probe_radius > 0:
        name = f"sva_causal_probe{probe_radius}_{suffix}"
    elif use_prefilter:
        name = f"sva_causal_prefilter{args.prefilter_dim}d{args.prefilter_budget}_{suffix}"
    else:
        name = f"sva_causal_{suffix}"
    return summarize(name, rows)


def full_attention_summary(
    args: argparse.Namespace,
    dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, str | float]:
    rows: list[dict[str, float]] = []
    for keys, values, queries, targets in dataset:
        for timestep, (query, target) in enumerate(zip(queries, targets), start=1):
            teacher, teacher_top = exact_attention(
                keys[:timestep],
                values[:timestep],
                query,
                args.logit_scale,
            )
            rows.append(
                {
                    "teacher_top_target": float(teacher_top == target),
                    "raw_recall": 1.0,
                    "top1_target": float(teacher_top == target),
                    "top1_teacher": 1.0,
                    "teacher": teacher,
                    "output": teacher,
                    "prefix_len": float(timestep),
                    "summoned": float(timestep),
                    "exact_scored": float(timestep),
                    "verified": float(timestep),
                }
            )
    return summarize("full_causal_attention", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal sequence kill test for Summon-Verify Attention.")
    parser.add_argument("--n-sequences", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--key-noise", type=float, default=0.08)
    parser.add_argument("--query-noise", type=float, default=0.10)
    parser.add_argument("--target-mode", choices=["uniform", "recent"], default="uniform")
    parser.add_argument("--max-age", type=int, default=128)
    parser.add_argument("--tables", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--bits", type=int, default=10)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--probe-radius", type=int, default=1)
    parser.add_argument("--prefilter-dim", type=int, default=32)
    parser.add_argument("--prefilter-budget", type=int, default=128)
    parser.add_argument("--logit-scale", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    dataset = make_dataset(args, np.random.default_rng(args.seed))
    rows = [full_attention_summary(args, dataset)]
    for n_tables in args.tables:
        rows.append(run_variant(args, n_tables, 0, False, dataset, np.random.default_rng(args.seed + n_tables)))
        if args.probe_radius > 0:
            rows.append(
                run_variant(
                    args,
                    n_tables,
                    args.probe_radius,
                    False,
                    dataset,
                    np.random.default_rng(args.seed + 10_000 + n_tables),
                )
            )
        if args.prefilter_dim > 0 and args.prefilter_budget > 0 and args.probe_radius > 0:
            rows.append(
                run_variant(
                    args,
                    n_tables,
                    args.probe_radius,
                    True,
                    dataset,
                    np.random.default_rng(args.seed + 20_000 + n_tables),
                )
            )

    headers = [
        "method",
        "teacher_top_target",
        "raw_recall",
        "top1_target",
        "top1_teacher",
        "cos_teacher",
        "avg_prefix",
        "avg_summoned",
        "avg_exact_scored",
        "avg_verified",
    ]
    print(",".join(headers))
    for row in rows:
        print(
            f"{row['method']},"
            f"{float(row['teacher_top_target']):.4f},"
            f"{float(row['raw_recall']):.4f},"
            f"{float(row['top1_target']):.4f},"
            f"{float(row['top1_teacher']):.4f},"
            f"{float(row['cos_teacher']):.4f},"
            f"{float(row['avg_prefix']):.1f},"
            f"{float(row['avg_summoned']):.1f},"
            f"{float(row['avg_exact_scored']):.1f},"
            f"{float(row['avg_verified']):.1f}"
        )


if __name__ == "__main__":
    main()
