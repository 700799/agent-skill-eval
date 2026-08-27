"""Small aggregation helpers shared by the A/B engine and reports."""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def pct_delta(treatment: float, control: float) -> float:
    """Relative delta of treatment vs control, in percent (negative = savings)."""
    if control == 0:
        return 0.0
    return (treatment - control) / control * 100.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
