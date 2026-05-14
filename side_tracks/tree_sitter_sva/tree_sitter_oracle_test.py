"""Oracle coverage test for syntax-tree routing as a side-track SVA hierarchy."""

from __future__ import annotations

import argparse
import bisect
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IDENTIFIER_TYPES = {"identifier", "property_identifier", "type_identifier"}
ROUTING_NODE_TYPES = {
    "module",
    "class_definition",
    "function_definition",
    "decorated_definition",
    "parameters",
    "argument_list",
    "block",
    "if_statement",
    "elif_clause",
    "else_clause",
    "for_statement",
    "while_statement",
    "with_statement",
    "try_statement",
    "except_clause",
    "assignment",
    "augmented_assignment",
    "return_statement",
    "expression_statement",
    "call",
    "attribute",
    "subscript",
    "list_comprehension",
    "dictionary_comprehension",
    "set_comprehension",
    "generator_expression",
}


@dataclass(frozen=True)
class Token:
    index: int
    start_byte: int
    end_byte: int
    node_type: str
    text: str


@dataclass(frozen=True)
class Unit:
    router: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class Query:
    token_index: int
    targets: tuple[int, ...]


def comma_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def make_parser():
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing tree-sitter dependency. Install with: "
            "python -m pip install -r side_tracks/tree_sitter_sva/requirements.txt"
        ) from exc

    language = Language(tree_sitter_python.language())
    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.set_language(language)
        return parser


def iter_leaves(node) -> Iterable[object]:
    if node.child_count == 0:
        yield node
        return
    for child in node.children:
        yield from iter_leaves(child)


def iter_nodes(node) -> Iterable[object]:
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def token_text(source: bytes, node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def collect_tokens(source: bytes, tree) -> list[Token]:
    tokens: list[Token] = []
    for node in iter_leaves(tree.root_node):
        text = token_text(source, node)
        if not text or text.isspace():
            continue
        tokens.append(
            Token(
                index=len(tokens),
                start_byte=int(node.start_byte),
                end_byte=int(node.end_byte),
                node_type=str(node.type),
                text=text,
            )
        )
    return tokens


def node_token_span(node, token_starts: list[int], token_ends: list[int]) -> tuple[int, int] | None:
    left = bisect.bisect_right(token_ends, int(node.start_byte))
    right = bisect.bisect_left(token_starts, int(node.end_byte))
    if left >= right:
        return None
    return left, right


def collect_tree_units(
    tree,
    tokens: list[Token],
    min_node_tokens: int,
    max_node_tokens: int,
) -> list[Unit]:
    token_starts = [token.start_byte for token in tokens]
    token_ends = [token.end_byte for token in tokens]
    units: list[Unit] = []
    seen: set[tuple[int, int, str]] = set()
    for node in iter_nodes(tree.root_node):
        node_type = str(node.type)
        if node_type not in ROUTING_NODE_TYPES:
            continue
        span = node_token_span(node, token_starts, token_ends)
        if span is None:
            continue
        start, end = span
        width = end - start
        if width < min_node_tokens or width > max_node_tokens:
            continue
        key = (start, end, node_type)
        if key in seen:
            continue
        seen.add(key)
        units.append(Unit("tree", node_type, start, end))
    return units


def collect_fixed_units(tokens: list[Token], chunk_size: int) -> list[Unit]:
    return [
        Unit(f"fixed_{chunk_size}", f"chunk_{start // chunk_size}", start, min(start + chunk_size, len(tokens)))
        for start in range(0, len(tokens), chunk_size)
    ]


def collect_queries(
    tokens: list[Token],
    target_topk: int,
    min_previous: int,
    max_occurrences: int,
) -> list[Query]:
    ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    positions_by_name: dict[str, list[int]] = {}
    for token in tokens:
        if token.node_type not in IDENTIFIER_TYPES or not ident_re.match(token.text):
            continue
        positions_by_name.setdefault(token.text, []).append(token.index)

    queries: list[Query] = []
    for positions in positions_by_name.values():
        if len(positions) < min_previous + 1 or len(positions) > max_occurrences:
            continue
        for occurrence_idx, token_index in enumerate(positions):
            previous = positions[:occurrence_idx]
            if len(previous) < min_previous:
                continue
            targets = tuple(previous[-target_topk:])
            queries.append(Query(token_index, targets))
    return queries


def greedy_oracle(query: Query, units: list[Unit], budget: int) -> tuple[int, int, int]:
    remaining = set(query.targets)
    eligible = [unit for unit in units if unit.start < query.token_index]
    opened_tokens: set[int] = set()
    opened_units = 0

    for _ in range(budget):
        best_unit: Unit | None = None
        best_gain = 0
        best_width = 0
        for unit in eligible:
            clipped_end = min(unit.end, query.token_index)
            if clipped_end <= unit.start:
                continue
            gain = sum(1 for target in remaining if unit.start <= target < clipped_end)
            width = clipped_end - unit.start
            if gain > best_gain or (gain == best_gain and gain > 0 and width < best_width):
                best_unit = unit
                best_gain = gain
                best_width = width
        if best_unit is None or best_gain <= 0:
            break
        clipped_end = min(best_unit.end, query.token_index)
        opened_tokens.update(range(best_unit.start, clipped_end))
        remaining = {target for target in remaining if not (best_unit.start <= target < clipped_end)}
        eligible = [unit for unit in eligible if unit is not best_unit]
        opened_units += 1
        if not remaining:
            break

    hits = len(query.targets) - len(remaining)
    return hits, len(opened_tokens), opened_units


def evaluate_router(router: str, units: list[Unit], queries: list[Query], budgets: list[int]) -> None:
    total_targets = sum(len(query.targets) for query in queries)
    for budget in budgets:
        hits = 0
        opened_tokens: list[float] = []
        opened_units: list[float] = []
        for query in queries:
            query_hits, token_count, unit_count = greedy_oracle(query, units, budget)
            hits += query_hits
            opened_tokens.append(float(token_count))
            opened_units.append(float(unit_count))
        recall = hits / total_targets if total_targets else float("nan")
        avg_opened_tokens = sum(opened_tokens) / max(len(opened_tokens), 1)
        avg_units = sum(opened_units) / max(len(opened_units), 1)
        print(
            "tree_sitter_oracle_result,"
            f"{router},{budget},{len(queries)},{total_targets},{recall:.6f},"
            f"{avg_opened_tokens:.3f},{percentile(opened_tokens, 50):.3f},{percentile(opened_tokens, 95):.3f},"
            f"{avg_units:.3f}",
            flush=True,
        )


def resolve_paths(paths: list[str]) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            alt = (repo_root / raw_path).resolve()
            if alt.exists():
                path = alt
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            resolved.append(path)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-sitter syntax routing oracle for SVA side-track tests.")
    parser.add_argument("--paths", nargs="+", default=["experiments"])
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=4000)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--min-previous", type=int, default=1)
    parser.add_argument("--max-occurrences", type=int, default=96)
    parser.add_argument("--min-node-tokens", type=int, default=2)
    parser.add_argument("--max-node-tokens", type=int, default=256)
    parser.add_argument("--chunk-tokens", default="64,128,256")
    parser.add_argument("--budgets", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    parser_impl = make_parser()
    files = resolve_paths(args.paths)[: args.max_files]
    if not files:
        raise SystemExit("No Python files found for tree-sitter oracle test.")

    all_queries: list[Query] = []
    tree_units: list[Unit] = []
    fixed_units_by_size: dict[int, list[Unit]] = {size: [] for size in comma_ints(args.chunk_tokens)}
    token_offset = 0
    parsed_files = 0

    for path in files:
        source = path.read_bytes()
        tree = parser_impl.parse(source)
        tokens = collect_tokens(source, tree)
        if not tokens:
            continue
        file_queries = collect_queries(tokens, args.target_topk, args.min_previous, args.max_occurrences)
        file_tree_units = collect_tree_units(tree, tokens, args.min_node_tokens, args.max_node_tokens)
        all_queries.extend(
            Query(query.token_index + token_offset, tuple(target + token_offset for target in query.targets))
            for query in file_queries
        )
        tree_units.extend(
            Unit(unit.router, f"{path.name}:{unit.label}", unit.start + token_offset, unit.end + token_offset)
            for unit in file_tree_units
        )
        for size, units in fixed_units_by_size.items():
            units.extend(
                Unit(unit.router, f"{path.name}:{unit.label}", unit.start + token_offset, unit.end + token_offset)
                for unit in collect_fixed_units(tokens, size)
            )
        token_offset += len(tokens)
        parsed_files += 1

    if len(all_queries) > args.max_queries:
        rng = random.Random(args.seed)
        all_queries = rng.sample(all_queries, args.max_queries)

    budgets = comma_ints(args.budgets)
    print("metric,value")
    print(f"files,{parsed_files}")
    print(f"tokens,{token_offset}")
    print(f"queries,{len(all_queries)}")
    print(f"tree_units,{len(tree_units)}")
    print(f"target_topk,{args.target_topk}")
    print(f"max_node_tokens,{args.max_node_tokens}")
    print(
        "tree_sitter_oracle_header,"
        "router,budget,queries,targets,recall,avg_opened_tokens,p50_opened_tokens,p95_opened_tokens,avg_units_opened"
    )
    evaluate_router("tree_sitter_nodes", tree_units, all_queries, budgets)
    for size, units in fixed_units_by_size.items():
        evaluate_router(f"fixed_{size}", units, all_queries, budgets)


if __name__ == "__main__":
    main()
