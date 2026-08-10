"""
===============================================================================
dataset_views.py
===============================================================================
Define the two buildable dataset views without registering another task.
Responsibilities:
  - Register steady-flow and transient-drying dataset-view identities
  - Bind each view to its canonical channel contract and source profiles
  - Declare task-specific parameter-OOD relevance and package regimes
Design principles:
  - Trainable-task registration remains owned by ``src.domain.tasks``
  - Channel order is imported from its existing authoritative owner
  - View and contract digests are deterministic and path independent
This module does NOT:
  - Build packages, load HDF5 data, or register transient training behavior
  - Infer parameter relevance from filenames or broad OOD group names
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, cast

from src import domain
from src.generation import generation_profiles as profiles

from .dataset_transient_contract import TRANSIENT_STEP_CONTRACT

DatasetViewId = Literal["steady_flow", "transient_drying"]
PackageRegime = Literal["id", "parameter_ood", "near_family_ood", "far_family_ood", "extreme_family_ood"]
IdMembership = Literal["train", "validation", "id_test"]
OodGroup = Literal["bed", "operation", "initial_moisture", "material_properties"]

DATASET_VIEW_SCHEMA_VERSION: Final = 1
PACKAGE_REGIMES: Final = ("id", "parameter_ood", "near_family_ood", "far_family_ood", "extreme_family_ood")
ID_MEMBERSHIPS: Final = ("train", "validation", "id_test")
TECHNICAL_SMOKE_MEMBERSHIP: Final = "technical_smoke"
OOD_GROUPS: Final = ("bed", "operation", "initial_moisture", "material_properties")


@dataclass(frozen=True, slots=True)
class DatasetViewSpec:
    """Describe one buildable dataset view and its source relevance contract."""

    id: DatasetViewId
    registered_task_id: str | None
    source_profiles: tuple[str, ...]
    parameter_ood_blocks: tuple[str, ...]
    parameter_ood_groups: tuple[OodGroup, ...]
    contract_digest: str

    @property
    def trainable(self) -> bool:
        """Return whether the view is backed by a registered learning task."""
        return self.registered_task_id is not None


def _digest(payload: object) -> str:
    """Return one canonical SHA-256 contract digest."""
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _steady_contract_digest() -> str:
    """Return the registered task's learned-data contract digest."""
    return domain.tasks.registry.get_task("steady_flow").data_contract_digest


def _transient_contract_digest() -> str:
    """Return the unregistered physical transition contract digest."""
    contract = TRANSIENT_STEP_CONTRACT
    payload = {
        "schema_version": DATASET_VIEW_SCHEMA_VERSION,
        "view": "transient_drying",
        "state": [(field.name, field.unit) for field in contract.dynamic_state],
        "static": [(field.name, field.unit) for field in contract.static_spatial_conditioning],
        "boundary": [(field.name, field.unit) for field in contract.step_boundary_conditioning],
        "scalars": [(field.name, field.unit) for field in contract.scalar_conditioning],
        "target": [(field.name, field.unit) for field in contract.target_increments],
        "dt": {"value": contract.time_step, "unit": contract.time_unit},
        "storage": contract.canonical_storage_representation,
        "target_derivation": contract.target_derivation_stage,
        "material_family_usage": contract.material_family_usage,
    }
    return _digest(payload)


_VIEWS: Final = MappingProxyType(
    {
        "steady_flow": DatasetViewSpec(
            id="steady_flow",
            registered_task_id="steady_flow",
            source_profiles=(profiles.STEADY_FLOW_PROFILE, profiles.TRANSIENT_DRYING_PROFILE),
            parameter_ood_blocks=("airflow",),
            parameter_ood_groups=("bed", "operation"),
            contract_digest=_steady_contract_digest(),
        ),
        "transient_drying": DatasetViewSpec(
            id="transient_drying",
            registered_task_id=None,
            source_profiles=(profiles.TRANSIENT_DRYING_PROFILE,),
            parameter_ood_blocks=("airflow", "initial_moisture", "operation", "material_properties"),
            parameter_ood_groups=OOD_GROUPS,
            contract_digest=_transient_contract_digest(),
        ),
    }
)


def available_views() -> tuple[DatasetViewId, ...]:
    """Return buildable dataset-view identifiers in deterministic order."""
    return tuple(cast("DatasetViewId", name) for name in sorted(_VIEWS))


def get_view(view_id: str) -> DatasetViewSpec:
    """Resolve one exact buildable dataset-view identifier."""
    try:
        return _VIEWS[view_id]
    except KeyError as error:
        available = ", ".join(available_views())
        message = f"Unknown dataset view {view_id!r}. Available views: {available}."
        raise ValueError(message) from error


def validate_view_for_profile(view_id: str, profile_id: str) -> DatasetViewSpec:
    """Resolve a view and require the selected simulation profile to provide it."""
    view = get_view(view_id)
    if profile_id not in view.source_profiles:
        message = f"Dataset view {view.id!r} is unavailable from simulation profile {profile_id!r}."
        raise ValueError(message)
    return view
