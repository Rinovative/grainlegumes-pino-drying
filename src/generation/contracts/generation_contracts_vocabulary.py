"""Define immutable campaign, design, evaluation, and Dataset-membership vocabulary.

Responsibilities:
  - Declare typed evaluation-regime and ID-membership identifiers
  - Own the design kinds of a material and their canonical order
  - Own stable campaign-purpose and material-role collections

Design principles:
  - Vocabulary has no case-planning or runtime dependencies
  - Resolved configuration reuses these values without duplicating them

This module does NOT:
  - Load configuration, assign membership, or plan cases
"""

from __future__ import annotations

from typing import Literal, TypeAlias

IdMembership: TypeAlias = Literal["train", "validation", "id_test"]  # noqa: UP040 -- public runtime aliases must remain their concrete typing or class objects
EvaluationRegime: TypeAlias = Literal[  # noqa: UP040 -- public runtime aliases must remain their concrete typing or class objects
    "id",
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
]

ID_MEMBERSHIPS: tuple[IdMembership, ...] = ("train", "validation", "id_test")
# One fixed Sobol design per material and kind; a Source batch's sampling regime is its design kind.
DESIGN_KINDS = ("train", "validation", "test", "parameter_ood")
EVALUATION_REGIMES: tuple[EvaluationRegime, ...] = (
    "id",
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
)
MATERIAL_ROLES = (
    "seen",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
)
CAMPAIGN_PURPOSES = ("source_generation",)
