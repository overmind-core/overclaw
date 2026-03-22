"""
Experimental ticket scoring — not wired into ``sample_agent``; bundler should skip.
"""

from __future__ import annotations

import bisect
import statistics
from functools import lru_cache


@lru_cache(maxsize=128)
def fake_queue_depth(team: str, hour_utc: int) -> int:
    base = hash((team, hour_utc)) % 97
    return max(0, base - 40)


def weighted_median(values: list[float], weights: list[float]) -> float:
    if not values or len(values) != len(weights):
        return 0.0
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    vals = [p[0] for p in pairs]
    wts = [p[1] for p in pairs]
    total = sum(wts)
    if total == 0:
        return statistics.median(vals)
    acc = 0.0
    half = total / 2.0
    for v, w in zip(vals, wts):
        acc += w
        if acc >= half:
            return v
    return vals[-1]


def tier_priority_index(tier: str) -> int:
    order = ["free", "pro", "enterprise"]
    i = bisect.bisect_left(order, tier) if tier in order else 0
    return i
