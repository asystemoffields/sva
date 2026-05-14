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
            "avg_exact_scored": self.ratio("exact_scored"),
            "avg_verified": self.ratio("verified"),
        }
