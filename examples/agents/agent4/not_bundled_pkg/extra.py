"""Submodule never imported from ``agents.agent4.sample_agent``."""

from enum import Enum, auto


class PhantomRiskTier(Enum):
    ALPHA = auto()
    BETA = auto()
    GAMMA = auto()


def decode_phantom_payload(blob: bytes) -> dict:
    return {"len": len(blob), "xor_sum": sum(blob) % 256}
