"""Declare the authoritative two-dimensional transient grain-drying task contract.

Responsibilities:
  - Define ordered transient input profiles and reconstructed-state metrics
  - Bind transient field units and representations to Dataset runtime semantics
  - Declare data-only training defaults without executable dataset selection
  - Own the physical-bound diagnostic of reconstructed states shared by training validation and artifacts

Design principles:
  - Increment outputs remain distinct from reconstructed evaluation fields
  - Input profiles preserve exact channel order and explicit coordinates
  - Physics selection is explicit even when no residual is supported
  - A finite physical-bound violation is a diagnostic: it never stops a rollout, clamps a value or decides eligibility

This module does NOT:
  - Materialize transient Dataset samples or normalize tensors
  - Implement recurrence, rollout, loss, metric, or physics equations
  - Select executable dataset packages or training configurations
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .domain_task_spec import (
    TRAIN_SCALING_KIND,
    FieldSpec,
    InputProfileSpec,
    MetricSpec,
    OutputGroupSpec,
    PhysicsSpec,
    PreprocessingSpec,
    ReferenceFloorSpec,
    TaskSpec,
    field_metric_specs,
)

_TRANSIENT_INPUTS = (
    FieldSpec("T", "state", "K", "identity_before_train_normalization"),
    FieldSpec("phi", "state", "1", "identity_before_train_normalization"),
    FieldSpec("w_surf", "state", "kg/m^3", "identity_before_train_normalization"),
    FieldSpec("w_int", "state", "kg/m^3", "identity_before_train_normalization"),
    FieldSpec("x", "coordinate", "m", "identity_before_train_normalization"),
    FieldSpec("y", "coordinate", "m", "identity_before_train_normalization"),
    FieldSpec("u", "airflow", "m/s", "identity_before_train_normalization"),
    FieldSpec("v", "airflow", "m/s", "identity_before_train_normalization"),
    FieldSpec("p", "airflow", "Pa", "identity_before_train_normalization"),
    FieldSpec("eps_bed", "porosity", "1", "identity_before_train_normalization"),
    FieldSpec("rho_bu_dry", "material", "kg/m^3", "identity_before_train_normalization"),
    FieldSpec("T_in_bc_t_n", "boundary", "K", "identity_before_train_normalization"),
    FieldSpec("T_in_bc_t_n_plus_1", "boundary", "K", "identity_before_train_normalization"),
    FieldSpec("omega_in_bc_t_n", "boundary", "kg/kg", "identity_before_train_normalization"),
    FieldSpec("omega_in_bc_t_n_plus_1", "boundary", "kg/kg", "identity_before_train_normalization"),
    FieldSpec("T_amb", "boundary", "K", "identity_before_train_normalization"),
    FieldSpec("startup_support_time_offset", "boundary", "h", "identity_before_train_normalization"),
    FieldSpec("T_in_bc_startup_support", "boundary", "K", "identity_before_train_normalization"),
    FieldSpec("omega_in_bc_startup_support", "boundary", "kg/kg", "identity_before_train_normalization"),
    FieldSpec("startup_support_present", "boundary", "1", "identity_before_train_normalization"),
    FieldSpec("r_surf_0", "material", "1/s", "identity_before_train_normalization"),
    FieldSpec("r_int_surf", "material", "1", "identity_before_train_normalization"),
    FieldSpec("f_surf", "material", "1", "identity_before_train_normalization"),
    FieldSpec("A_osw", "material", "1", "identity_before_train_normalization"),
    FieldSpec("B_osw", "material", "1/K", "identity_before_train_normalization"),
    FieldSpec("C_osw", "material", "1", "identity_before_train_normalization"),
    FieldSpec("k_gr", "material", "W/(m*K)", "identity_before_train_normalization"),
    FieldSpec("cp_gr_dry", "material", "J/(kg*K)", "identity_before_train_normalization"),
)

_COMPLETE_FIELDS = tuple(field.name for field in _TRANSIENT_INPUTS)

TRANSIENT_DRYING = TaskSpec(
    id="transient_drying",
    inputs=_TRANSIENT_INPUTS,
    outputs=(
        FieldSpec("delta_T", "state", "K", "next_state_minus_current_state"),
        FieldSpec("delta_phi", "state", "1", "next_state_minus_current_state"),
        FieldSpec("delta_w_surf", "state", "kg/m^3", "next_state_minus_current_state"),
        FieldSpec("delta_w_int", "state", "kg/m^3", "next_state_minus_current_state"),
    ),
    metric_fields=(
        FieldSpec("T", "state", "K", "reconstructed_current_plus_predicted_increment"),
        FieldSpec("phi", "state", "1", "reconstructed_current_plus_predicted_increment"),
        FieldSpec("w_surf", "state", "kg/m^3", "reconstructed_current_plus_predicted_increment"),
        FieldSpec("w_int", "state", "kg/m^3", "reconstructed_current_plus_predicted_increment"),
    ),
    output_groups=(
        OutputGroupSpec("temperature", ("T",)),
        OutputGroupSpec("humidity", ("phi",)),
        OutputGroupSpec("grain_moisture", ("w_surf", "w_int")),
    ),
    tensor_layout=("batch", "channel", "y", "x"),
    operator_axes=(2, 3),
    normalization_axes=(0, 2, 3),
    preprocessing=PreprocessingSpec(normalization=TRAIN_SCALING_KIND, fit_split="train"),
    reference_floors=(
        ReferenceFloorSpec("T", 1.0),
        ReferenceFloorSpec("phi", 1.0e-3),
        ReferenceFloorSpec("w_surf", 0.1),
        ReferenceFloorSpec("w_int", 0.1),
    ),
    data_losses=("mse", "h1"),
    default_metrics=(
        MetricSpec(
            id="normalized_drying_group_macro_rmse",
            kind="group_macro_rmse",
            space="physical",
            fields=("T", "phi", "w_surf", "w_int"),
            reduction="material_case_mean",
            direction="minimize",
        ),
        *field_metric_specs(("T", "phi", "w_surf", "w_int")),
    ),
    physics=PhysicsSpec(
        kind="none",
        equation_set="data_only",
        continuity="none",
        boundary="none",
    ),
    input_profiles=(InputProfileSpec("canonical_physics_complete", _COMPLETE_FIELDS, "explicit_x_y"),),
    primary_input_profile="canonical_physics_complete",
    temporal_conditioning_kinds=("none", "normalized_current_time"),
    training_airflow_source="comsol_reference",
)

# Gross divergence screens of the reconstructed state (T, phi, w_surf, w_int), not physical admissibility; see docs/06.
TEMPERATURE_RANGE_K: Final = (0.0, 2_000.0)
PHI_RANGE: Final = (0.0, 1.10)
EXCLUDED_CORNER_BLOCK: Final = 2


def physical_invalid_mask(states: np.ndarray) -> np.ndarray:
    """
    Return the finite reconstructed values outside the maintained field domains.

    ``states`` holds the channels ``(T, phi, w_surf, w_int)`` on axis ``-3`` and the
    grid on the last two axes. Inclusive bounds: T in [0, 2000] K, phi in [0, 1.10],
    non-negative moisture. The 2x2 corner blocks are excluded, and non-finite values
    are never physically invalid (they are counted separately).
    """
    array = np.asarray(states)
    finite = np.isfinite(array)
    invalid = np.zeros(array.shape, dtype=bool)
    lower, upper = TEMPERATURE_RANGE_K
    invalid[..., 0, :, :] = finite[..., 0, :, :] & ((array[..., 0, :, :] < lower) | (array[..., 0, :, :] > upper))
    invalid[..., 1, :, :] = finite[..., 1, :, :] & ((array[..., 1, :, :] < PHI_RANGE[0]) | (array[..., 1, :, :] > PHI_RANGE[1]))
    invalid[..., 2:, :, :] = finite[..., 2:, :, :] & (array[..., 2:, :, :] < 0.0)
    block = EXCLUDED_CORNER_BLOCK
    for rows in (slice(None, block), slice(-block, None)):
        for columns in (slice(None, block), slice(-block, None)):
            invalid[..., rows, columns] = False
    return invalid
