"""MetricRegistry — register and call all metric plugins after reconstruction."""

from __future__ import annotations

from typing import Callable

_METRICS: dict[str, Callable] = {}


def register_metric(name: str, fn: Callable) -> None:
    """Register a metric function by name."""
    _METRICS[name] = fn


def run_all_metrics(pred, gt) -> dict[str, float]:
    """
    Call every registered metric function and return results as a dict.
    If gt is None (no ground truth yet), all metrics return 0.0.
    """
    if gt is None:
        return {name: 0.0 for name in _METRICS}
    return {name: fn(pred, gt) for name, fn in _METRICS.items()}


def list_metrics() -> list[str]:
    """Return names of all registered metrics."""
    return sorted(_METRICS.keys())
