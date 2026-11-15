"""Declare the authoritative steady two-dimensional porous-flow task contract.

Responsibilities:
  - Exact ordered steady-flow input and output fields
  - Field units, stored representations, tensor axes, and preprocessing
  - Non-executable Dataset catalog references, semantic metrics, losses, and physics

Design principles:
  - The declaration is immutable and contains only canonical identifiers
  - Learned channel counts derive only from the ordered field declarations
  - The task selects physics semantically without implementing equations

This module does NOT:
  - Load, fingerprint, or validate stored datasets
  - Implement derivatives, residual equations, losses, or metrics
  - Select datasets for explicit executable experiment recipes
  - Define checkpoint, resume, inference, or artifact lifecycle behavior
"""

from __future__ import annotations

from src.domain.physics import domain_physics_contracts as physics_contracts

from .domain_task_spec import (
    TRAIN_SCALING_KIND,
    FieldSpec,
    MetricSpec,
    OutputGroupSpec,
    PhysicsSpec,
    PreprocessingSpec,
    ReferenceFloorSpec,
    TaskSpec,
    field_metric_specs,
)

STEADY_FLOW = TaskSpec(
    id="steady_flow",
    inputs=(
        FieldSpec("x", "coordinate", "m", "identity"),
        FieldSpec("y", "coordinate", "m", "identity"),
        FieldSpec(
            "Kxx",
            "permeability",
            "m^2",
            "dimensionless_log10_ratio_to_1_m2",
        ),
        FieldSpec(
            "Kxy",
            "permeability",
            "m^2",
            "dimensionless_cross_component_ratio_to_geometric_mean",
        ),
        FieldSpec(
            "Kyy",
            "permeability",
            "m^2",
            "dimensionless_log10_ratio_to_1_m2",
        ),
        FieldSpec("eps_bed", "porosity", "1", "identity"),
        FieldSpec("p_in_bc", "boundary", "Pa", "identity"),
    ),
    outputs=(
        FieldSpec("p", "state", "Pa", "identity_before_train_normalization"),
        FieldSpec("u", "state", "m/s", "identity_before_train_normalization"),
        FieldSpec("v", "state", "m/s", "identity_before_train_normalization"),
    ),
    output_groups=(
        OutputGroupSpec("pressure", ("p",)),
        OutputGroupSpec("velocity", ("u", "v"), vector=True),
    ),
    tensor_layout=("batch", "channel", "y", "x"),
    operator_axes=(2, 3),
    normalization_axes=(0, 2, 3),
    preprocessing=PreprocessingSpec(normalization=TRAIN_SCALING_KIND, fit_split="train"),
    reference_floors=(ReferenceFloorSpec("p", 1.0), ReferenceFloorSpec("u", 1.0e-4), ReferenceFloorSpec("v", 1.0e-4)),
    data_losses=("mse", "h1"),
    default_metrics=(
        MetricSpec(
            id="normalized_group_macro_rmse",
            kind="group_macro_rmse",
            space="physical",
            fields=("p", "u", "v"),
            reduction="material_case_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="normalized_velocity_vector_rmse",
            kind="group_rmse",
            space="physical",
            fields=("u", "v"),
            reduction="material_case_mean",
            direction="minimize",
        ),
        MetricSpec(
            id="physical_velocity_vector_rmse",
            kind="vector_rmse",
            space="physical",
            fields=("u", "v"),
            reduction="material_case_mean",
            direction="minimize",
        ),
        *field_metric_specs(("p", "u", "v")),
    ),
    physics=PhysicsSpec(
        kind=physics_contracts.STEADY_BRINKMAN_KIND,
        equation_set=physics_contracts.STEADY_BRINKMAN_EQUATION_SET,
        continuity=physics_contracts.CONTINUITY_KIND,
        boundary=physics_contracts.PRESSURE_BOUNDARY_KIND,
    ),
)
