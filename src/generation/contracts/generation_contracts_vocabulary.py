"""
generation_contracts_vocabulary.py

Define immutable campaign, evaluation, and Dataset-membership vocabulary.
Responsibilities:
  - Declare typed evaluation-regime and ID-membership identifiers
  - Own stable campaign-purpose, material-role, and split-name collections
Design principles:
  - Vocabulary has no case-planning or runtime dependencies
  - Resolved configuration reuses these values without duplicating them
This module does NOT:
  - Load configuration, assign membership, or plan cases
"""

from __future__ import annotations

from typing import Literal, TypeAlias

IdMembership: TypeAlias = Literal["train", "validation", "id_test"]
EvaluationRegime: TypeAlias = Literal[
    "id",
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
]

PILOT_CAMPAIGN_PURPOSE = "pilot_check"
STEADY_FLOW_ID_DATASET_PURPOSE = "steady_flow_id_dataset"
NO_EVALUATION_REGIME = "not_applicable"
ID_MEMBERSHIPS: tuple[IdMembership, ...] = ("train", "validation", "id_test")
SPLIT_NAMES = (
    *ID_MEMBERSHIPS,
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
    "technical_smoke",
)
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
CAMPAIGN_PURPOSES = (
    "family_generalization",
    STEADY_FLOW_ID_DATASET_PURPOSE,
    "technical_runtime_smoke",
    PILOT_CAMPAIGN_PURPOSE,
)
