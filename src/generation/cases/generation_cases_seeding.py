"""Derive deterministic seeds for Generation scientific workflows.

Responsibilities:
  - Validate seed inputs and derive stable unsigned 32-bit substream seeds
  - Derive the seed of one fixed design from the seed base, material and design kind
  - Derive the scientific campaign seed key that recorded candidate identity seeds depend on

Design principles:
  - Semantic labels make random substreams independent of execution ordering
  - The campaign seed key hashes scientific design values only, never display names or producer identity
  - One derivation algorithm serves configuration and sampling; the code fingerprint is its only identity
  - Invalid seed domains fail before numerical random-number generation

This module does NOT:
  - Resolve configuration, select sampling designs, or allocate OOD cases
  - Generate numerical samples, fields, schedules, or persisted artifacts
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from src import common

if TYPE_CHECKING:
    from collections.abc import Mapping

UINT32_MAX = 2**32 - 1


def derive_seed(seed_base: int, *labels: str) -> int:
    """Derive one stable uint32 seed from an integer base and semantic labels."""
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or not 0 <= seed_base <= UINT32_MAX:
        message = f"seed_base must be an integer in the uint32 range, got {seed_base!r}."
        raise ValueError(message)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        message = "Seed derivation requires one or more non-empty labels."
        raise ValueError(message)
    payload = f"{seed_base}|" + "|".join(labels)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], byteorder="big", signed=False)


def design_seed(seed_base: int, material: str, design_kind: str) -> int:
    """Return the seed of one fixed design, from the task's seed base, the material and the design kind only.

    No plan, role, count or campaign enters it, so point i of a design is the same physical case in every plan.
    """
    return derive_seed(seed_base, "design", material, design_kind)


def campaign_seed_key(
    *,
    task: str,
    designs: Mapping[str, Mapping[str, int]],
    domain_profile: str,
    design_recipe: str,
    seed_base: int,
    purpose: str,
) -> str:
    """Return the stable scientific key that campaign-level identity seeds derive from.

    The key hashes the task, the original design size per material and design kind, the domain profile, the
    design recipe with its seed base, and the campaign purpose. It is independent of the display name, the
    storage layout, and the producing implementation; no per-case random stream derives from it.
    """
    return common.serialization.canonical_json_sha256(
        {
            "task": task,
            "designs": {material: {kind: int(count) for kind, count in kinds.items()} for material, kinds in designs.items()},
            "domain_profile": domain_profile,
            "sampling": {"recipe": design_recipe, "seed_base": seed_base},
            "purpose": purpose,
        }
    )
