"""
generation_cases_seeding.py

Derive version-bound deterministic seeds for Generation scientific workflows.
Responsibilities:
  - Own the Generation algorithm version used in semantic seed identities
  - Validate seed inputs and derive stable unsigned 32-bit substream seeds
Design principles:
  - Semantic labels make random substreams independent of execution ordering
  - One version-bound derivation algorithm serves configuration and sampling
  - Invalid seed domains fail before numerical random-number generation
This module does NOT:
  - Resolve configuration, select sampling designs, or allocate OOD cases
  - Generate numerical samples, fields, schedules, or persisted artifacts
"""

from __future__ import annotations

import hashlib

GENERATOR_VERSION = 1
UINT32_MAX = 2**32 - 1


def derive_seed(seed_base: int, *labels: str) -> int:
    """Derive one stable uint32 seed from an integer base and semantic labels."""
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or not 0 <= seed_base <= UINT32_MAX:
        message = f"seed_base must be an integer in the uint32 range, got {seed_base!r}."
        raise ValueError(message)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        message = "Seed derivation requires one or more non-empty labels."
        raise ValueError(message)
    payload = f"{GENERATOR_VERSION}|{seed_base}|" + "|".join(labels)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], byteorder="big", signed=False)
