"""Runtime accounting for SVA adapters."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def _float_stats() -> defaultdict[str, float]:
    return defaultdict(float)


@dataclass
class SVAStats:
    """Mutable aggregate statistics recorded by patched attention layers."""

    totals: defaultdict[str, float] = field(default_factory=_float_stats)
    by_layer: defaultdict[int, defaultdict[str, float]] = field(default_factory=lambda: defaultdict(_float_stats))

    def reset(self) -> None:
        self.totals.clear()
        self.by_layer.clear()

    def add(self, layer_idx: int, values: dict[str, float]) -> None:
        layer_stats = self.by_layer[layer_idx]
        for key, value in values.items():
            self.totals[key] += value
            layer_stats[key] += value

    def ratio(self, numerator: str, denominator: str = "queries") -> float:
        total = self.totals.get(denominator, 0.0)
        if total <= 0:
            return float("nan")
        return self.totals.get(numerator, 0.0) / total

    def summary(self) -> dict[str, float]:
        return {
            "queries": self.totals.get("queries", 0.0),
            "avg_summoned": self.ratio("summoned"),
            "avg_refill_pool": self.ratio("refill_pool"),
            "avg_exact_scored": self.ratio("exact_scored"),
            "avg_verified": self.ratio("verified"),
            "avg_cell_visits": self.ratio("cell_visits"),
            "avg_static_catalog_ms": self.ratio("static_catalog_ms", "profile_calls"),
            "avg_static_refill_ms": self.ratio("static_refill_ms", "profile_calls"),
            "avg_static_budget_ms": self.ratio("static_budget_ms", "profile_calls"),
            "avg_static_gather_ms": self.ratio("static_gather_ms", "profile_calls"),
            "avg_static_exact_score_ms": self.ratio("static_exact_score_ms", "profile_calls"),
            "avg_static_aggregate_ms": self.ratio("static_aggregate_ms", "profile_calls"),
            "avg_static_total_ms": self.ratio("static_total_ms", "profile_calls"),
            "avg_static_projection_ms": self.ratio("static_projection_ms", "profile_outer_calls"),
            "avg_static_key_catalog_ms": self.ratio("static_key_catalog_ms", "profile_outer_calls"),
            "avg_static_outer_total_ms": self.ratio("static_outer_total_ms", "profile_outer_calls"),
        }
