# ruff: noqa: S101
"""
Protect the immutable registered task and its derived semantic contract.

The tests exercise task-owned physical groups and metric meaning, stable
serialization, registry immutability, ordered declaration rejection, and public
domain exports. Dataset content identity and resolved experiment projection are
covered by their broader pipeline suites.
"""

from dataclasses import replace

import pytest

from src import domain


def test_steady_flow_contract_is_task_owned_and_scientifically_complete() -> None:
    """Protect the task's physical output groups, units, and objective meaning."""
    task = domain.tasks.registry.get_task("steady_flow")

    assert task.in_channels == len(task.input_names)
    assert task.out_channels == len(task.output_names)
    assert task.schema_version == domain.tasks.spec.TASK_SCHEMA_VERSION
    assert task.preprocessing.fit_split == "train"
    assert task.operator_dimensionality == len(task.operator_axes)

    groups = {group.id: group for group in task.output_groups}
    assert {group_id: group.fields for group_id, group in groups.items()} == {
        "pressure": ("p",),
        "velocity": ("u", "v"),
    }
    assert tuple(field for group in task.output_groups for field in group.fields) == task.output_names
    assert task.field("p").unit == "Pa"
    assert {task.field(field).unit for field in groups["velocity"].fields} == {"m/s"}

    objective = next(metric for metric in task.default_metrics if metric.kind == "group_macro_rmse")
    assert objective.id == "normalized_group_macro_rmse"
    assert (objective.space, objective.fields, objective.reduction, objective.direction) == (
        "physical",
        task.output_names,
        "group_macro_element_mean",
        "minimize",
    )
    normalized_vector = next(metric for metric in task.default_metrics if metric.kind == "group_rmse")
    physical_vector = next(metric for metric in task.default_metrics if metric.kind == "vector_rmse")
    assert normalized_vector.fields == physical_vector.fields == groups["velocity"].fields

    resolved = task.resolved_contract()
    assert resolved["digest"] == task.contract_digest
    assert resolved["data_contract_digest"] == task.data_contract_digest
    data_contract = task.data_contract_payload()
    assert "default_metrics" not in data_contract
    assert "output_groups" not in data_contract


def test_task_schema_rejects_an_unsupported_version() -> None:
    """Reject a contract version other than the public current schema."""
    task = domain.tasks.registry.get_task("steady_flow")

    with pytest.raises(ValueError, match="schema_version must be integer"):
        replace(task, schema_version=task.schema_version + 1)


def test_ordered_contract_validator_rejects_drift() -> None:
    """Reject reordered, missing, and duplicated fields derived from TaskSpec."""
    expected = domain.tasks.registry.get_task("steady_flow").input_names
    reordered = list(expected)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    duplicated = [*expected]
    duplicated[1] = duplicated[0]

    for actual in (tuple(reordered), expected[:-1], tuple(duplicated)):
        with pytest.raises(ValueError, match=r"duplicate|does not match|wrong channel order"):
            domain.field_sets.validate_ordered_fields(actual, expected, label="inputs")


def test_public_domain_exports_resolve_and_noncanonical_fields_fail() -> None:
    """Resolve canonical public names and reject unknown task or field names."""
    assert domain.tasks.registry.get_task("steady_flow").id == "steady_flow"
    assert domain.fields.require_known_field("eps_bed") == "eps_bed"
    with pytest.raises(ValueError, match="Unknown task"):
        domain.tasks.registry.get_task("unregistered_task")
    with pytest.raises(ValueError, match="Unknown field"):
        domain.fields.require_known_field("unknown_field")
    with pytest.raises(ValueError, match="Unknown field"):
        domain.fields.require_known_field("pbc")


def test_task_declarations_fail_closed_on_runtime_literals_and_layout(
    synthetic_task: domain.tasks.spec.TaskSpec,
) -> None:
    """
    Construct invalid field roles, metric directions, layouts, and operator axes.

    Each runtime value outside the typed/current 2D contract must fail explicitly,
    because persisted configuration cannot rely on static type checking alone.
    """
    spec = domain.tasks.spec
    with pytest.raises(ValueError, match="unsupported role"):
        spec.FieldSpec("bad", "unsupported", "1", "identity")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported direction"):
        spec.MetricSpec(
            id="bad_direction",
            kind="rmse",
            space="physical",
            fields=("response_b",),
            reduction="element_mean",
            direction="sideways",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="current 2D layout"):
        replace(synthetic_task, tensor_layout=("channel", "batch", "y", "x"))
    with pytest.raises(ValueError, match="current 2D operator/normalizer support"):
        replace(synthetic_task, operator_axes=(1, 2))
