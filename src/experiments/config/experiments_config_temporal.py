"""
experiments_config_temporal.py

Resolve explicit transient sampling and temporal-conditioning configuration.
Responsibilities:
  - Require both Dataset sampling and Learning conditioning declarations
  - Delegate semantic validation to their responsible package contracts
  - Return one canonical serializable temporal experiment branch
Design principles:
  - Experiments composes Dataset and Learning choices without duplicating policy
  - Missing modes fail rather than selecting an implicit scientific default
  - Canonical mappings remain suitable for run and resume identity hashing
This module does NOT:
  - Register a transient task or mutate the steady experiment resolver
  - Select packages, read time coordinates, or transform model inputs
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src import datasets, learning


def resolve_transient_sampling(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and serialize one explicit Dataset transient sampling choice."""
    return datasets.contracts.transient.TransientSamplingSpec.from_mapping(value).as_dict()


def resolve_temporal_conditioning(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate and serialize one explicit Learning time-conditioning choice."""
    return learning.temporal.TemporalConditioningSpec.from_mapping(value).as_dict()


def resolve_transient_temporal_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the exact temporal branch required by a transient experiment."""
    if not isinstance(value, Mapping):
        message = "Transient temporal configuration must be a mapping."
        raise TypeError(message)
    required = {"sampling", "temporal_conditioning"}
    if set(value) != required:
        message = f"Transient temporal configuration keys must be exactly {sorted(required)}."
        raise ValueError(message)
    return {
        "sampling": resolve_transient_sampling(value["sampling"]),
        "temporal_conditioning": resolve_temporal_conditioning(value["temporal_conditioning"]),
    }
