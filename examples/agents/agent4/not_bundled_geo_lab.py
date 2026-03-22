"""
Orphan lab code next to the valuation agent — not reachable from the bundler entry.
"""

import itertools
import re
from collections import Counter


def token_bigrams(text: str) -> list[tuple[str, str]]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return list(zip(tokens, tokens[1:]))


def silly_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def count_zip_prefixes(addresses: list[str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for line in addresses:
        m = re.search(r"\b(\d{5})\b", line)
        if m:
            c[m.group(1)[:3]] += 1
    return c


def powerset_upto(items: list[str], max_size: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for r in range(min(max_size, len(items)) + 1):
        out.extend(itertools.combinations(items, r))
    return out
