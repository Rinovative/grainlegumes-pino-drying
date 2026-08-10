"""
===============================================================================
dataset_transient_contract.py
===============================================================================
Define the unregistered transient drying step-data schema.
Responsibilities:
  - Declare ordered dynamic, spatial, boundary, scalar, and target fields
  - Record fixed-step, absolute-state, and metadata semantics
  - Separate baseline conditioning from archived ablation fields
Design principles:
  - Canonical simulation storage retains absolute physical states
  - Step increments belong to later dataset construction
  - Material family remains metadata rather than a model input channel
This module does NOT:
  - Register a transient learning task or build transient tensors
  - Define normalization, models, losses, or rollout behavior
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class DataField:
    """Describe one ordered logical field and its physical unit."""

    name: str
    unit: str


@dataclass(frozen=True, slots=True)
class TransientStepContract:
    """Describe the planned one-hour transient operator data contract."""

    dynamic_state: tuple[DataField, ...]
    static_spatial_conditioning: tuple[DataField, ...]
    step_boundary_conditioning: tuple[DataField, ...]
    scalar_conditioning: tuple[DataField, ...]
    target_increments: tuple[DataField, ...]
    archived_ablation_fields: tuple[DataField, ...]
    tensor_dtype: str
    spatial_shape: tuple[int, int]
    time_step: float
    time_unit: str
    canonical_storage_representation: str
    target_derivation_stage: str
    material_family_usage: str


TRANSIENT_STEP_CONTRACT: Final = TransientStepContract(
    dynamic_state=(
        DataField("T", "K"),
        DataField("phi", "1"),
        DataField("w_surf", "kg/m^3"),
        DataField("w_int", "kg/m^3"),
    ),
    static_spatial_conditioning=(
        DataField("x", "m"),
        DataField("y", "m"),
        DataField("u", "m/s"),
        DataField("v", "m/s"),
        DataField("p", "Pa"),
        DataField("eps_bed", "1"),
        DataField("rho_bu_dry", "kg/m^3"),
    ),
    step_boundary_conditioning=(
        DataField("T_in_bc_t_n", "K"),
        DataField("T_in_bc_t_np1", "K"),
        DataField("phi_in_bc_t_n", "1"),
        DataField("phi_in_bc_t_np1", "1"),
        DataField("T_amb", "K"),
    ),
    scalar_conditioning=(
        DataField("r_surf_0", "1/s"),
        DataField("r_int_surf", "1"),
        DataField("f_surf", "1"),
        DataField("A_osw", "1"),
        DataField("B_osw", "1/K"),
        DataField("C_osw", "1"),
        DataField("k_gr", "W/(m*K)"),
        DataField("cp_gr_dry", "J/(kg*K)"),
    ),
    target_increments=(
        DataField("delta_T", "K"),
        DataField("delta_phi", "1"),
        DataField("delta_w_surf", "kg/m^3"),
        DataField("delta_w_int", "kg/m^3"),
    ),
    archived_ablation_fields=(
        DataField("Kxx", "m^2"),
        DataField("Kxy", "m^2"),
        DataField("Kyy", "m^2"),
        DataField("p_in_bc", "Pa"),
        DataField("X_0_db_field", "kg/kg"),
    ),
    tensor_dtype="float32",
    spatial_shape=(251, 401),
    time_step=1.0,
    time_unit="h",
    canonical_storage_representation="absolute_physical_states",
    target_derivation_stage="transient_dataset_builder",
    material_family_usage="metadata_only",
)
