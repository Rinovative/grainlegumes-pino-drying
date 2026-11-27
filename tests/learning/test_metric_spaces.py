"""Protect named field selection, normalized/physical tensor routing, and metric units.

Responsibilities:
  - Check named channels, physical inverse transforms, units, and mixed-unit rejection
  - Check that a prediction that overflows on decoding counts as a non-finite validation case
  - Check that the physical evaluation view decodes with each case's inlet-pressure reference
  - Check that validation's normalized field errors equal the artifact values and leave the objective unchanged

Design principles:
  - Metric-space routing transforms physical tensors exactly once

This module does NOT:
  - Verify macro reduction algebra or training normalization quality
"""

from __future__ import annotations

import copy

import pytest
import torch
from support import configs
from support.scaling import steady_inputs, steady_scaling

from src import domain, experiments, learning

_TASK = domain.tasks.registry.get_task("steady_flow")
_GROUP_OBJECTIVE = next(metric for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_NORMALIZED_FIELD_METRICS = tuple(metric for metric in _TASK.default_metrics if metric.kind == "rmse" and metric.space == "normalized")
_PHYSICAL_FIELD_METRICS = tuple(
    metric for metric in _TASK.default_metrics if metric.kind == "rmse" and metric.space == "physical" and len(metric.fields) == 1
)
# The checkpoint objective of the fixed two-case input below, recorded before the normalized field errors moved to physical statistics.
_OBJECTIVE_ON_FIXED_INPUT = 9.39927782715339


class UnitErrorModel(torch.nn.Module):
    """Return a unit normalized error for every TaskSpec output channel."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return task-derived output channels of normalized ones."""
        return torch.ones(
            (inputs.shape[0], _TASK.out_channels, *inputs.shape[-2:]),
            device=inputs.device,
        )


def _metrics(*metric_ids: str, scales: torch.Tensor | None = None) -> dict[str, learning.metrics.metrics.DatasetMetric]:
    """Build a selected subset of resolved default metrics with the physical Train scales ``scales`` (default two)."""
    config = configs.resolve(configs.direct_config())
    selected = [metric for metric in config["evaluation"]["metrics"] if metric["id"] in metric_ids]
    config["evaluation"]["metrics"] = selected
    return learning.metrics.metrics.build_evaluation_metrics(
        config,
        device=torch.device("cpu"),
        output_scales=torch.full((_TASK.out_channels,), 2.0) if scales is None else scales,
    )


def test_normalized_and_physical_rmse_are_space_correct_once() -> None:
    """
    Evaluate unit normalized error through a Train scaling with physical scale two.

    Normalized metrics must remain one and each physical field RMSE must become
    two exactly once, protecting against missing or duplicate inverse transforms.
    """
    metrics = _metrics(
        _GROUP_OBJECTIVE.id,
        *(metric.id for metric in (*_NORMALIZED_FIELD_METRICS, *_PHYSICAL_FIELD_METRICS)),
    )
    scaling = steady_scaling((3, 4), state_mean=10.0, state_std=2.0)
    raw_batch = {
        "x": steady_inputs(2, (3, 4)),
        "y": torch.full((2, _TASK.out_channels, 3, 4), 10.0),
        "meta": {"simulation_case_id": ["case-a", "case-b"], "material_family": ["lentil", "pea"]},
    }
    values = learning.training.loop.eval_one_epoch(
        UnitErrorModel(),
        [raw_batch],  # type: ignore[arg-type]
        metrics,
        torch.device("cpu"),
        scaling,
    )

    assert values[_GROUP_OBJECTIVE.id] == pytest.approx(1.0)
    for expected, definitions in ((1.0, _NORMALIZED_FIELD_METRICS), (2.0, _PHYSICAL_FIELD_METRICS)):
        ids = tuple(metric.id for metric in definitions)
        assert {metric_id: values[metric_id] for metric_id in ids} == pytest.approx(dict.fromkeys(ids, expected))


class _FixedOutputModel(torch.nn.Module):
    """Return one fixed normalized output whatever the inputs."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the fixed output."""
        del inputs
        return self.output


def test_physical_views_equal_the_physical_targets_through_the_inlet_pressure_reference() -> None:
    """A model that returns the encoded targets of cases with P = 3 and P = 0.5 has zero physical error in Pa and m/s."""
    metrics = _metrics(*(metric.id for metric in _PHYSICAL_FIELD_METRICS))
    scaling = steady_scaling((3, 4), state_mean=10.0, state_std=2.0)
    inputs = torch.cat([steady_inputs(1, (3, 4), inlet_pressure=3.0), steady_inputs(1, (3, 4), inlet_pressure=0.5)])
    targets = torch.linspace(1.0, 50.0, 2 * _TASK.out_channels * 12).reshape(2, _TASK.out_channels, 3, 4)
    encoded = scaling.encode_state(targets, reference=scaling.state_reference(inputs))
    raw_batch = {"x": inputs, "y": targets, "meta": {"simulation_case_id": ["case-a", "case-b"], "material_family": ["lentil", "pea"]}}
    values = learning.training.loop.eval_one_epoch(
        _FixedOutputModel(encoded),
        [raw_batch],  # type: ignore[arg-type]
        metrics,
        torch.device("cpu"),
        scaling,
    )

    for definition in _PHYSICAL_FIELD_METRICS:
        assert values[definition.id] == pytest.approx(0.0, abs=1e-4)


def test_validation_normalized_field_errors_are_the_artifact_values_and_the_objective_is_unchanged() -> None:
    """
    Two cases with inlet pressures 3 and 0.5 and per-field physical Train scales 2, 3 and 4.

    Validation's ``normalized_rmse_<f>`` and ``normalized_mae_<f>`` equal what the artifacts compute with
    ``field_metric_values`` from the physical case statistics (physical error over the physical Train scale),
    not the error of the encoded states; the checkpoint objective keeps its exact value.
    """
    shape, spacing = (6, 6), (0.2, 0.2)
    scales = torch.arange(2.0, 2.0 + _TASK.out_channels)
    normalized_ids = [f"normalized_{kind}_{field}" for kind in ("rmse", "mae") for field in _TASK.output_names]
    metrics = _metrics(_GROUP_OBJECTIVE.id, *normalized_ids, scales=scales)
    scaling = steady_scaling(shape, state_mean=10.0, state_std=2.0)
    inputs = torch.cat([steady_inputs(1, shape, inlet_pressure=3.0), steady_inputs(1, shape, inlet_pressure=0.5)])
    targets = torch.linspace(1.0, 50.0, 2 * _TASK.out_channels * 36).reshape(2, _TASK.out_channels, *shape)
    prediction = torch.linspace(-1.0, 1.5, 2 * _TASK.out_channels * 36).reshape(2, _TASK.out_channels, *shape)
    raw_batch = {"x": inputs, "y": targets, "meta": {"simulation_case_id": ["case-a", "case-b"], "material_family": ["lentil", "pea"]}}
    values = learning.training.loop.eval_one_epoch(
        _FixedOutputModel(prediction),
        [raw_batch],  # type: ignore[arg-type]
        metrics,
        torch.device("cpu"),
        scaling,
    )

    reduction = learning.metrics.metrics
    physical = scaling.decode_state(prediction, reference=scaling.state_reference(inputs))
    rows = reduction.field_statistics(physical, targets, grid_spacing=spacing)
    cases = [
        reduction.CaseStatistics(case, material, _TASK.output_names, rows[row])
        for row, (case, material) in enumerate((("case-a", "lentil"), ("case-b", "pea")))
    ]
    expected, _unavailable = reduction.field_metric_values(
        cases,
        train_scales={field: float(scale) for field, scale in zip(_TASK.output_names, scales, strict=True)},
        reference_floors={field: _TASK.reference_floor(field) for field in _TASK.output_names},
        output_groups=_TASK.output_groups,
    )
    assert {metric_id: values[metric_id] for metric_id in normalized_ids} == pytest.approx(
        {metric_id: expected[metric_id] for metric_id in normalized_ids}, rel=1e-12
    )
    assert values[_GROUP_OBJECTIVE.id] == _OBJECTIVE_ON_FIXED_INPUT


def test_validation_produces_the_curated_relative_l2_above_the_reference_floors() -> None:
    """
    The curated ``physical_relative_l2`` has a producer: references of 10 and physical errors of 2 give 0.2.

    The steady lifecycle test excludes this key, because its tiny references lie below the TaskSpec floors
    and the metric then has no finite value, which keeps it out of the event.
    """
    metrics = _metrics("physical_relative_l2")
    raw_batch = {
        "x": steady_inputs(2, (3, 4)),
        "y": torch.full((2, _TASK.out_channels, 3, 4), 10.0),
        "meta": {"simulation_case_id": ["case-a", "case-b"], "material_family": ["lentil", "pea"]},
    }
    values = learning.training.loop.eval_one_epoch(
        UnitErrorModel(),
        [raw_batch],  # type: ignore[arg-type]
        metrics,
        torch.device("cpu"),
        steady_scaling((3, 4), state_mean=10.0, state_std=2.0),
    )

    assert values["physical_relative_l2"] == pytest.approx(0.2)


class _OverflowingCaseModel(torch.nn.Module):
    """Return a finite normalized output whose second case overflows when decoded to physical units."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized ones for the first case and 1e30 for the second."""
        output = torch.ones((inputs.shape[0], _TASK.out_channels, *inputs.shape[-2:]), device=inputs.device)
        output[1] = 1.0e30
        return output


def test_a_finite_normalized_prediction_that_overflows_on_decoding_is_a_non_finite_case() -> None:
    """A state scale of 1e10 decodes the finite 1e30 to inf: validation counts the case and reports no value instead of raising."""
    raw_batch = {
        "x": steady_inputs(2, (3, 4)),
        "y": torch.full((2, _TASK.out_channels, 3, 4), 10.0),
        "meta": {"simulation_case_id": ["case-a", "case-b"], "material_family": ["lentil", "pea"]},
    }
    values = learning.training.loop.eval_one_epoch(
        _OverflowingCaseModel(),
        [raw_batch],  # type: ignore[arg-type]
        _metrics(_GROUP_OBJECTIVE.id, *(metric.id for metric in _PHYSICAL_FIELD_METRICS)),
        torch.device("cpu"),
        steady_scaling((3, 4), state_mean=10.0, state_std=1.0e10),
    )

    assert values == {learning.training.loop.NONFINITE_CASES_KEY: 1.0}
    assert not learning.training.loop.validation_eligible(values, _GROUP_OBJECTIVE.id)


def test_physical_units_and_named_channel_selection() -> None:
    """
    Give every task output a distinct constant physical error.

    Each TaskSpec-derived field metric must select the matching channel and unit.
    """
    metrics = _metrics(*(metric.id for metric in _PHYSICAL_FIELD_METRICS))
    target = torch.zeros((1, _TASK.out_channels, 2, 2))
    errors = {field: float(2 + 3 * index) for index, field in enumerate(_TASK.output_names)}
    pred = torch.stack(
        tuple(torch.full((1, 2, 2), errors[field]) for field in _TASK.output_names),
        dim=1,
    )
    cases = learning.metrics.metrics.CaseMembership(("case-a",), ("lentil",))
    for metric in metrics.values():
        metric.update(pred, target, batch_index=0, cases=cases)

    for definition in _PHYSICAL_FIELD_METRICS:
        field = definition.fields[0]
        metric = metrics[definition.id]
        assert metric.unit == _TASK.field(field).unit
        assert metric.compute() == pytest.approx(errors[field])


def test_incompatible_physical_aggregate_is_rejected() -> None:
    """
    Resolve one physical aggregate spanning pressure and both velocity fields.

    Configuration must reject the mixed-unit reduction before runtime so a single
    scalar cannot falsely imply a coherent physical unit.
    """
    raw = configs.direct_config()
    defaults = experiments.config.defaults.get_task_defaults(str(raw["task"]))
    aggregate = copy.deepcopy(
        next(metric for metric in defaults["evaluation"]["metrics"] if metric["id"] == _PHYSICAL_FIELD_METRICS[0].id),
    )
    aggregate.pop("direction")
    aggregate["id"] = "physical_rmse_all"
    aggregate["fields"] = list(_TASK.output_names)
    raw["evaluation"] = {
        "metrics": [aggregate],
        "objective": {"id": "physical_rmse_all"},
    }

    with pytest.raises(ValueError, match="exactly one field; selected units"):
        configs.resolve(raw)
