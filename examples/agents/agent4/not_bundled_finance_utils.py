"""
Not imported by ``sample_agent`` or ``original_agent`` — should stay out of the
OverClaw bundler closure when the entrypoint is ``agents.agent4.sample_agent``.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from collections.abc import Iterator


@dataclass
class SyntheticCapRate:
    metro: str
    rate_bps: int

    def as_float(self) -> float:
        return self.rate_bps / 10_000.0


def fibonacci(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def monte_carlo_irr(samples: int, *, seed: int = 42) -> Iterator[float]:
    rng = random.Random(seed)
    for _ in range(samples):
        yield math.exp(rng.gauss(0.05, 0.02)) - 1.0


def hash_address_token(addr: str) -> str:
    return hashlib.sha256(addr.lower().encode()).hexdigest()[:16]


CAP_RATES: list[SyntheticCapRate] = [
    SyntheticCapRate("METRO_A", 425),
    SyntheticCapRate("METRO_B", 510),
]
