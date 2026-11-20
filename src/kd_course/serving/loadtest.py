"""Measure a service: how long requests take, how many it handles, and what each costs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


def _percentile(ordered: list[float], percent: float) -> float:
    """Linear interpolation between the two nearest values."""
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_latencies(latencies: Sequence[float]) -> dict[str, float]:
    """Median, 95th percentile and mean, in milliseconds. Input is in seconds."""
    if not latencies:
        raise ValueError("Cannot summarize an empty list of latencies.")
    ordered = sorted(latencies)
    return {
        "p50_ms": _percentile(ordered, 50) * 1000,
        "p95_ms": _percentile(ordered, 95) * 1000,
        "mean_ms": sum(ordered) / len(ordered) * 1000,
    }


@dataclass
class LoadResult:
    requests: int
    items: int
    errors: int
    seconds: float
    latencies: list[float] = field(default_factory=list)

    @property
    def requests_per_second(self) -> float:
        return self.requests / self.seconds

    @property
    def items_per_second(self) -> float:
        return self.items / self.seconds


def run_load_test(
    call: Callable[[Any], Any], payloads: Sequence[Sequence[Any]], concurrency: int = 1
) -> LoadResult:
    """Send every payload through `call`, `concurrency` at a time, and time each one.

    A payload is one request's worth of items. Failed calls are counted, not raised.
    """

    def timed(payload: Sequence[Any]) -> float | None:
        start = time.perf_counter()
        try:
            call(payload)
        except Exception:
            return None
        return time.perf_counter() - start

    begin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(timed, payloads))
    seconds = time.perf_counter() - begin
    latencies = [outcome for outcome in outcomes if outcome is not None]
    return LoadResult(
        requests=len(payloads),
        items=sum(len(payload) for payload in payloads),
        errors=len(outcomes) - len(latencies),
        seconds=seconds,
        latencies=latencies,
    )


def cost_per_1k(instance_usd_per_hour: float, items_per_second: float) -> float:
    """Dollars per 1,000 items for a machine running flat out at this throughput."""
    if items_per_second <= 0:
        raise ValueError("Throughput must be above zero.")
    return instance_usd_per_hour / (items_per_second * 3600) * 1000
