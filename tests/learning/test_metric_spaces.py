"""Protect named field selection, normalized/physical tensor routing, and metric units.

Responsibilities:
  - Check named channels, physical inverse transforms, units, and mixed-unit rejection
  - Check that a prediction that overflows on decoding counts as a non-finite validation case

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
from support.scaling import steady_scaling

from src import domain, experiments, learning

_TASK = domain.tasks.registry.get_task("steady_flow")
_GROUP_OBJECTIVE = next(metric for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_NORMALIZED_FIELD_METRICS = tuple(metric for metric in _TASK.default_metrics if metric.kind == "rmse" and metric.space == "normalized")
_PHYSICAL_FIELD_METRICS = tuple(
    metric for metric in _TASK.default_metrics if metric.kind == "rmse" and metric.space == "physical" and len(metric.fields) == 1
)


class UnitErrorModel(torch.nn.Module):
    """Return a unit normalized error for every TaskSpec output channel."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return task-derived output channels of normalized ones."""
        return torch.ones(
            (inputs.shape[0], _TASK.out_channels, *inputs.shape[-2:]),
            device=inputs.device,
        )


def _metrics(*metric_ids: str) -> dict[str, learning.metrics.metrics.DatasetMetric]:
    """Build a selected subset of resolved default metrics."""
    config = configs.resolve(configs.direct_config())
    selected = [metric for metric in config["evaluation"]["metrics"] if metric["id"] in metric_ids]
    config["evaluation"]["metrics"] = selected
    return learning.metrics.metrics.build_evaluation_metrics(
        config,
        device=torch.device("cpu"),
        output_scales=torch.full((_TASK.out_channels,), 2.0),
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
        "x": torch.zeros((2, _TASK.in_channels, 3, 4)),
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


def test_validation_produces_the_curated_relative_l2_above_the_reference_floors() -> None:
    """
    The curated ``physical_relative_l2`` has a producer: references of 10 and physical errors of 2 give 0.2.

    The steady lifecycle test excludes this key, because its tiny references lie below the TaskSpec floors
    and the metric then has no finite value, which keeps it out of the event.
    """
    metrics = _metrics("physical_relative_l2")
    raw_batch = {
        "x": torch.zeros((2, _TASK.in_channels, 3, 4)),
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
        "x": torch.zeros((2, _TASK.in_channels, 3, 4)),
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

    Each TaskSpec-derived field metric must select the matching channel and unit,
    while a normalized-space update fails instead of mixing tensor spaces.
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
        metric.update(pred, target, space="physical", batch_index=0, cases=cases)

    for definition in _PHYSICAL_FIELD_METRICS:
        field = definition.fields[0]
        metric = metrics[definition.id]
        assert metric.unit == _TASK.field(field).unit
        assert metric.compute() == pytest.approx(errors[field])

    first_metric = metrics[_PHYSICAL_FIELD_METRICS[0].id]
    with pytest.raises(ValueError, match="expects 'physical'"):
        first_metric.update(pred, target, space="normalized", batch_index=1)


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
