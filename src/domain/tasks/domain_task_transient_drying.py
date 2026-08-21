"""
domain_task_transient_drying.py

Declare the authoritative two-dimensional transient grain-drying task contract.

Responsibilities:
  - Define ordered transient input profiles and reconstructed-state metrics
  - Bind transient field units and representations to Dataset runtime semantics
  - Declare data-only training defaults without executable dataset selection

Design principles:
  - Increment outputs remain distinct from reconstructed evaluation fields
  - Input profiles preserve exact channel order and explicit coordinates
  - Physics selection is explicit even when no residual is supported

This module does NOT:
  - Materialize transient Dataset samples or normalize tensors
  - Implement recurrence, rollout, loss, metric, or physics equations
  - Select executable dataset packages or training configurations
"""

from __future__ import annotations

from .domain_task_spec import (
    TASK_SCHEMA_VERSION,
    DatasetDefaults,
    FieldSpec,
    InputProfileSpec,
    MetricSpec,
    OutputGroupSpec,
    PhysicsSpec,
    PreprocessingSpec,
    TaskSpec,
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
    schema_version=TASK_SCHEMA_VERSION,
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
    default_datasets=DatasetDefaults(
        train="transient_drying__lentil+chickpea__id",
        ood=("transient_drying__kidney_bean__near_family_ood",),
    ),
    preprocessing=PreprocessingSpec(
        input_normalization="train_only_group_specific_scaling_with_unique_state_deduplication",
        output_normalization="train_only_zero_preserving_per_channel_increment_scaling",
        fit_split="train",
    ),
    data_losses=("huber", "mse"),
    default_metrics=(
        MetricSpec(
            id="normalized_drying_group_macro_rmse",
            kind="drying_group_macro_rmse",
            space="normalized",
            fields=("T", "phi", "w_surf", "w_int"),
            reduction="group_macro_element_mean",
            direction="minimize",
        ),
        *(
            MetricSpec(
                id=f"normalized_rmse_{field}",
                kind="rmse",
                space="normalized",
                fields=(field,),
                reduction="element_mean",
                direction="minimize",
            )
            for field in ("T", "phi", "w_surf", "w_int")
        ),
        *(
            MetricSpec(
                id=f"physical_rmse_{field}",
                kind="rmse",
                space="physical",
                fields=(field,),
                reduction="element_mean",
                direction="minimize",
            )
            for field in ("T", "phi", "w_surf", "w_int")
        ),
        *(
            MetricSpec(
                id=f"physical_mae_{field}",
                kind="mae",
                space="physical",
                fields=(field,),
                reduction="element_mean",
                direction="minimize",
            )
            for field in ("T", "phi", "w_surf", "w_int")
        ),
    ),
    physics=PhysicsSpec(
        kind="none",
        equation_set="data_only",
        continuity="none",
        allowed_continuities=("none",),
        boundary="none",
    ),
    input_profiles=(InputProfileSpec("canonical_physics_complete_v1", _COMPLETE_FIELDS, "explicit_x_y"),),
    primary_input_profile="canonical_physics_complete_v1",
    temporal_conditioning_kinds=("none", "normalized_current_time"),
    training_airflow_source="comsol_reference",
)
