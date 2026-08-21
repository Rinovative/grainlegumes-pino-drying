"""
learning_transient_contracts.py

Define immutable transient model-input contracts.

Responsibilities:
  - Bind tensorization to the registered transient task and Dataset contract
  - Declare exact profile channel ordering and temporal conditioning
  - Serialize the model-facing contract for scaling and checkpoint identity

Design principles:
  - Profiles are task-owned and reject aliases
  - The existing temporal-policy owner remains authoritative
  - Optional time is outside Dataset profile membership

This module does NOT:
  - Fit scaling statistics or materialize Dataset samples
  - Define another temporal-conditioning vocabulary
  - Execute model inference or training
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src import datasets, domain
from src.learning.learning_temporal import TemporalConditioningSpec


@dataclass(frozen=True, slots=True)
class TransientTensorizerSpec:
    """
    Describe one exact transient model channel contract.

    Parameters
    ----------
    input_profile : str
        Exact registered transient input-profile identifier.
    temporal_conditioning : TemporalConditioningSpec
        Existing task-supported model-facing time policy.

    Raises
    ------
    TypeError
        If temporal conditioning is not an admitted policy object.
    ValueError
        If the profile identifier is unknown.
    RuntimeError
        If task and Dataset field ordering disagree.

    """

    input_profile: str
    temporal_conditioning: TemporalConditioningSpec

    def __post_init__(self) -> None:
        """Resolve the exact registered profile and Dataset field contract."""
        if not isinstance(self.temporal_conditioning, TemporalConditioningSpec):
            message = "temporal_conditioning must be a TemporalConditioningSpec."
            raise TypeError(message)
        task = domain.tasks.registry.get_task("transient_drying")
        task.input_profile(self.input_profile)
        if self.temporal_conditioning.kind not in task.temporal_conditioning_kinds:
            message = f"Task {task.id!r} does not support temporal conditioning {self.temporal_conditioning.kind!r}."
            raise ValueError(message)
        contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
        groups = (
            contract.dynamic_state,
            contract.static_spatial_conditioning,
            contract.step_boundary_conditioning,
            contract.scalar_conditioning,
        )
        expected = tuple(field.name for group in groups for field in group)
        complete = task.input_profile("canonical_physics_complete_v1")
        if complete.fields != expected:
            message = "Transient task profile disagrees with TRANSIENT_STEP_CONTRACT field ordering."
            raise RuntimeError(message)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TransientTensorizerSpec:
        """Admit one strict serialized tensorizer selection."""
        if not isinstance(value, Mapping):
            message = "Transient tensorizer selection must be a mapping."
            raise TypeError(message)
        required = {"input_profile", "temporal_conditioning"}
        if set(value) != required:
            message = f"Transient tensorizer selection keys must be exactly {sorted(required)}."
            raise ValueError(message)
        temporal = value["temporal_conditioning"]
        if not isinstance(temporal, Mapping):
            message = "Transient temporal conditioning must be a mapping."
            raise TypeError(message)
        return cls(
            input_profile=str(value["input_profile"]),
            temporal_conditioning=TemporalConditioningSpec.from_mapping(temporal),
        )

    @property
    def model_channel_names(self) -> tuple[str, ...]:
        """Return exact profile channels plus optional normalized current time."""
        task = domain.tasks.registry.get_task("transient_drying")
        fields = task.input_profile(self.input_profile).fields
        if self.temporal_conditioning.kind == "normalized_current_time":
            return (*fields, "normalized_current_time")
        return fields

    @property
    def in_channels(self) -> int:
        """Return the derived model input channel count."""
        return len(self.model_channel_names)

    @property
    def positional_embedding(self) -> None:
        """Require explicit coordinate channels instead of a model embedding."""
        return None

    def selection_dict(self) -> dict[str, object]:
        """Return the minimal reconstructable tensorizer selection."""
        return {
            "input_profile": self.input_profile,
            "temporal_conditioning": self.temporal_conditioning.as_dict(),
        }

    def as_dict(self) -> dict[str, object]:
        """Return the complete persisted tensorizer identity."""
        return {
            **self.selection_dict(),
            "model_channel_names": list(self.model_channel_names),
            "positional_embedding": None,
        }
