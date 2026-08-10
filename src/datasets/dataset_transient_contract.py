"""
===============================================================================
dataset_transient_contract.py
===============================================================================
Define and serialize the unregistered transient drying step-data contract.
Responsibilities:
  - Select Dataset-owned conditioning fields from Generation storage descriptors
  - Declare ordered dynamic, boundary, scalar, target, and ablation channels
  - Serialize and digest the exact persisted transient step contract
Design principles:
  - Generation owns canonical HDF5 source names and physical units
  - Dataset owns model-facing channel selection and step-target derivation
  - Spatial shape is resolved from admitted runtime data rather than declared here
This module does NOT:
  - Register a transient learning task or build transient tensors
  - Validate HDF5 payloads, normalize data, or define rollout behavior
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from src import generation

TRANSIENT_PROFILE_ID: Final = "transient_drying"
TRANSIENT_VIEW_ID: Final = "transient_drying"
TRANSIENT_VIEW_CONTRACT_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class DataField:
    """Describe one ordered logical Dataset field and its physical unit."""

    name: str
    unit: str


@dataclass(frozen=True, slots=True)
class TransientStepContract:
    """Describe the one-hour transient operator data contract."""

    dynamic_state: tuple[DataField, ...]
    static_spatial_conditioning: tuple[DataField, ...]
    step_boundary_conditioning: tuple[DataField, ...]
    scalar_conditioning: tuple[DataField, ...]
    target_increments: tuple[DataField, ...]
    archived_ablation_fields: tuple[DataField, ...]
    tensor_dtype: str
    time_step: float
    time_unit: str
    canonical_storage_representation: str
    target_derivation_stage: str
    material_family_usage: str


_SOURCE_PROFILE: Final = generation.contracts.get_profile_contract(TRANSIENT_PROFILE_ID)


def _source_field(name: str) -> DataField:
    """Return one Dataset descriptor derived from the Generation source schema."""
    source = _SOURCE_PROFILE.field(name)
    return DataField(source.name, source.unit)


def _scheduled_field(source_name: str, endpoint: str) -> DataField:
    """Return one step-endpoint field derived from a Generation schedule field."""
    source = _SOURCE_PROFILE.field(source_name)
    return DataField(f"{source.name}_{endpoint}", source.unit)


TRANSIENT_STEP_CONTRACT: Final = TransientStepContract(
    dynamic_state=tuple(DataField(field.name, field.unit) for field in _SOURCE_PROFILE.transient_fields),
    static_spatial_conditioning=tuple(
        _source_field(name)
        for name in (
            "x",
            "y",
            "u",
            "v",
            "p",
            "eps_bed",
            "rho_bu_dry",
        )
    ),
    step_boundary_conditioning=(
        _scheduled_field("T_in_bc", "t_n"),
        _scheduled_field("T_in_bc", "t_np1"),
        _scheduled_field("phi_in_bc", "t_n"),
        _scheduled_field("phi_in_bc", "t_np1"),
        _source_field("T_amb"),
    ),
    scalar_conditioning=tuple(
        _source_field(name)
        for name in (
            "r_surf_0",
            "r_int_surf",
            "f_surf",
            "A_osw",
            "B_osw",
            "C_osw",
            "k_gr",
            "cp_gr_dry",
        )
    ),
    target_increments=tuple(DataField(f"delta_{field.name}", field.unit) for field in _SOURCE_PROFILE.transient_fields),
    archived_ablation_fields=tuple(
        _source_field(name)
        for name in (
            "Kxx",
            "Kxy",
            "Kyy",
            "p_in_bc",
            "X_0_db_field",
        )
    ),
    tensor_dtype="float32",
    time_step=1.0,
    time_unit=_SOURCE_PROFILE.field("t").unit,
    canonical_storage_representation="absolute_physical_states",
    target_derivation_stage="transient_dataset_builder",
    material_family_usage="metadata_only",
)


def transient_contract_payload() -> dict[str, Any]:
    """Return the exact persisted transient tensor names, units, and step."""
    contract = TRANSIENT_STEP_CONTRACT
    return {
        "state": [{"name": field.name, "unit": field.unit} for field in contract.dynamic_state],
        "static": [{"name": field.name, "unit": field.unit} for field in contract.static_spatial_conditioning],
        "boundary": [{"name": field.name, "unit": field.unit} for field in contract.step_boundary_conditioning],
        "scalars": [{"name": field.name, "unit": field.unit} for field in contract.scalar_conditioning],
        "target": [{"name": field.name, "unit": field.unit} for field in contract.target_increments],
        "dt": {"value": contract.time_step, "unit": contract.time_unit},
        "storage": contract.canonical_storage_representation,
        "target_derivation": contract.target_derivation_stage,
        "material_family_usage": contract.material_family_usage,
    }


def transient_contract_digest() -> str:
    """Return the exact path-independent transient Dataset contract digest."""
    contract = TRANSIENT_STEP_CONTRACT
    payload = {
        "schema_version": TRANSIENT_VIEW_CONTRACT_SCHEMA_VERSION,
        "view": TRANSIENT_VIEW_ID,
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
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
