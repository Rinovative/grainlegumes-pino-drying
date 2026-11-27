"""Protect the material-case selection objective and the transient state-loss construction.

Responsibilities:
  - Check that the group objective weighs materials equally whatever their case counts and cases equally whatever their lengths
  - Check that the case-resolved objective metric and its transient components follow the objective
  - Check case-resolved MAE accumulation and transient state-loss construction

Design principles:
  - Expected values are computed by hand from test-owned constant errors

This module does NOT:
  - Construct steady physics losses or fit transient scales
"""

from __future__ import annotations

import math

import pytest
import torch

from src import domain, learning

_TASK = domain.tasks.registry.get_task("transient_drying")
_OBJECTIVE = next(metric for metric in _TASK.default_metrics if metric.kind == "group_macro_rmse")
_PHYSICAL_MAE = next(metric for metric in _TASK.default_metrics if metric.id == "physical_mae_T")
_SCALES = (2.0, 0.5, 4.0, 8.0)
_WEIGHTS = (1 / 3, 1 / 3, 1 / 6, 1 / 6)
_METRICS = learning.metrics.metrics


def _metric(definition: domain.tasks.spec.MetricSpec) -> learning.metrics.metrics.DatasetMetric:
    """Build one test-selected transient metric on CPU with test-owned Train scales."""
    config = {"task": _TASK.id, "evaluation": {"metrics": [definition.as_dict(all_fields=_TASK.metric_names)]}}
    return _METRICS.build_evaluation_metrics(config, device=torch.device("cpu"), output_scales=torch.tensor(_SCALES))[definition.id]


def _case_rows(errors: tuple[float, ...], *, steps: int) -> torch.Tensor:
    """Return ``steps`` flattened state rows of one case with one constant physical error per field."""
    return torch.tensor(errors, dtype=torch.float64).view(1, 4, 1, 1).expand(steps, 4, 2, 3)


def _objective(field_mse: dict[str, float]) -> float:
    """Return the hand-computed weighted sum of scaled term RMSEs for single-field drying terms."""
    return sum(weight * math.sqrt(field_mse[name] / scale**2) for name, weight, scale in zip(_TASK.metric_names, _WEIGHTS, _SCALES, strict=True))


def test_group_objective_weighs_materials_equally_whatever_their_case_counts_and_lengths() -> None:
    """
    Accumulate three cases of two materials, split across batches and of different lengths.

    Each case's errors average over its own grid and time; cases of a material then count
    equally, and both materials count equally although one has two cases and one has one.
    """
    metric = _metric(_OBJECTIVE)
    assert isinstance(metric, _METRICS.GroupMacroRMSEMetric)
    assert metric.requires_cases
    long_case = _case_rows((1.0, 1.0, 1.0, 1.0), steps=6)
    short_case = _case_rows((3.0, 3.0, 3.0, 3.0), steps=2)
    other = _case_rows((2.0, 4.0, 6.0, 8.0), steps=3)
    target = torch.zeros(11, 4, 2, 3, dtype=torch.float64)
    # The long case arrives in two batches: its rows still form one case mean.
    batches = (
        (torch.cat((long_case[:4], short_case)), ("a1",) * 4 + ("a2",) * 2, ("A",) * 6),
        (torch.cat((long_case[4:], other)), ("a1",) * 2 + ("b1",) * 3, ("A",) * 2 + ("B",) * 3),
    )
    for index, (prediction, cases, materials) in enumerate(batches):
        membership = _METRICS.CaseMembership(cases, materials)
        metric.update(prediction, target[: len(cases)], batch_index=index, cases=membership)

    material_a = dict.fromkeys(_TASK.metric_names, 0.5 * (1.0 + 9.0))
    material_b = dict(zip(_TASK.metric_names, (4.0, 16.0, 36.0, 64.0), strict=True))
    population = {name: 0.5 * (material_a[name] + material_b[name]) for name in _TASK.metric_names}
    objective = metric.objective()
    assert (objective.case_count, objective.material_count) == (3, 2)
    assert dict(objective.field_mse) == pytest.approx(population)
    assert metric.compute() == pytest.approx(_objective(population))
    components = metric.components()
    assert components["T"] == pytest.approx(math.sqrt(population["T"]) / _SCALES[0])
    assert components["grain_moisture"] == pytest.approx(0.5 * (components["w_surf"] + components["w_int"]))
    with pytest.raises(ValueError, match="one case membership per batch row"):
        metric.update(other, target[:3], batch_index=2)


def test_group_objective_rejects_a_case_that_enters_twice() -> None:
    """Refuse a population in which one case appears twice, so no case can double its weight."""
    terms = _TASK.error_terms
    case = _METRICS.CaseErrors("c1", "A", dict.fromkeys(_TASK.metric_names, 1.0))
    with pytest.raises(ValueError, match="twice"):
        _METRICS.group_objective((case, case), terms=terms, train_scales=dict(zip(_TASK.metric_names, _SCALES, strict=True)))


def test_mae_averages_each_case_over_its_states_before_the_materials() -> None:
    """Finalize transient MAE as each case's mean absolute error over its states, then equal over cases and materials."""
    metric = _metric(_PHYSICAL_MAE)
    first = torch.zeros(2, 4, 1, 2)
    second = torch.zeros(1, 4, 1, 2)
    first[:, 0, 0] = torch.tensor([[1.0, -3.0], [10.0, 10.0]])
    second[:, 0, 0] = torch.tensor([5.0, -7.0])
    metric.update(first, torch.zeros_like(first), batch_index=0, cases=_METRICS.CaseMembership(("a1", "b1"), ("A", "B")))
    metric.update(second, torch.zeros_like(second), batch_index=1, cases=_METRICS.CaseMembership(("a1",), ("A",)))
    assert metric.compute() == pytest.approx(0.5 * (4.0 + 10.0))


def test_transient_training_loss_is_the_data_only_state_loss() -> None:
    """Build the transient state loss from the resolved selection and reject physics terms."""
    config = {"task": _TASK.id, "loss": {"data": {"kind": "mse", "weight": 1.0}, "physics": {"enabled": False}}}
    loss = learning.losses.factory.build_training_loss(config, device=torch.device("cpu"))
    assert isinstance(loss, learning.losses.transient.TransientStateLoss)
    assert loss.physics_enabled is False
    with pytest.raises(ValueError, match="data-only"):
        learning.losses.factory.build_training_loss(
            {**config, "loss": {"data": {"kind": "mse", "weight": 1.0}, "physics": {"enabled": True}}}, device=torch.device("cpu")
        )
