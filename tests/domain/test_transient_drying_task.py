# ruff: noqa: S101
"""Protect the registered transient-drying task contract."""

from src import domain

_COMPLETE_CHANNEL_COUNT = 28


def test_transient_drying_registers_authoritative_profiles_and_metrics() -> None:
    """Keep transient channels, reconstructions, and objective semantics distinct."""
    task = domain.tasks.registry.get_task("transient_drying")

    complete = task.input_profile("canonical_physics_complete_v1")
    assert complete.fields == task.input_names
    assert len(complete.fields) == _COMPLETE_CHANNEL_COUNT
    assert complete.coordinate_policy == "explicit_x_y"
    assert tuple(name for name in complete.fields if name in {"x", "y"}) == ("x", "y")
    assert task.field("x").representation == ("identity_before_train_normalization")
    assert task.field("y").representation == ("identity_before_train_normalization")
    assert task.preprocessing.input_normalization == ("train_only_group_specific_scaling_with_unique_state_deduplication")
    assert task.preprocessing.output_normalization == ("train_only_zero_preserving_per_channel_increment_scaling")

    assert task.output_names == ("delta_T", "delta_phi", "delta_w_surf", "delta_w_int")
    assert task.metric_names == ("T", "phi", "w_surf", "w_int")
    assert tuple(field for group in task.output_groups for field in group.fields) == task.metric_names
    assert task.temporal_conditioning_kinds == ("none", "normalized_current_time")

    objective = next(metric for metric in task.default_metrics if metric.id == "normalized_drying_group_macro_rmse")
    assert (objective.kind, objective.space, objective.fields, objective.reduction, objective.direction) == (
        "drying_group_macro_rmse",
        "normalized",
        task.metric_names,
        "group_macro_element_mean",
        "minimize",
    )


def test_transient_task_serialization_and_steady_payload_compatibility() -> None:
    """Persist transient extensions without adding absent optional keys to steady payloads."""
    transient = domain.tasks.registry.get_task("transient_drying")
    steady = domain.tasks.registry.get_task("steady_flow")

    assert transient.resolved_contract()["digest"] == transient.contract_digest
    assert transient.resolved_contract()["data_contract_digest"] == transient.data_contract_digest
    assert {"metric_fields", "input_profiles", "primary_input_profile", "temporal_conditioning_kinds", "training_airflow_source"}.issubset(
        transient.contract_payload()
    )
    assert {"metric_fields", "input_profiles", "primary_input_profile", "temporal_conditioning_kinds", "training_airflow_source"}.isdisjoint(
        steady.contract_payload()
    )
    assert {"metric_fields", "input_profiles", "primary_input_profile", "temporal_conditioning_kinds", "training_airflow_source"}.isdisjoint(
        steady.data_contract_payload()
    )
