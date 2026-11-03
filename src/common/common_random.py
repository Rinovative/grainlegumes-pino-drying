"""Derive independent deterministic random streams from explicit nominal seeds.

Responsibilities:
  - Preserve the stable label-qualified 63-bit subseed algorithm
  - Share identical seed derivation between Dataset partitioning and Training

Design principles:
  - Stream labels separate scientific responsibilities without changing base seeds
  - Derivation is order-independent and uses only standard-library metadata

This module does NOT:
  - Draw samples, mutate global random state or choose scientific seed values
"""

import hashlib


def derive_subseed(seed: int, label: str) -> int:
    """
    Derive a stable label-qualified non-negative Torch-compatible seed.

    Parameters
    ----------
    seed : int
        Base run seed.
    label : str
        Non-empty stream label.

    Returns
    -------
    int
        Deterministic 63-bit sub-seed independent of derivation call order.

    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        msg = f"seed must be a non-negative integer, got {seed!r}."
        raise ValueError(msg)
    if not isinstance(label, str) or not label:
        msg = "seed label must be a non-empty string."
        raise ValueError(msg)
    payload = f"run-subseed-v1\0{seed}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
