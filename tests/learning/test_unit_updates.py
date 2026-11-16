"""Protect the one optimizer-update path that both tasks train through.

Responsibilities:
  - Check that each update weights its unit losses by case and averages them over the planned k,
    the last group of an epoch over its actual unit count, independent of the memory split
  - Check one optimizer step per group, the per-update telemetry and the training work counts
  - Check that a validation point stops mid-epoch after the crossing group and the epoch resumes at the next group
  - Check that the optimisation clock leaves out loader waits

Design principles:
  - A scalar model and a fake task make every gradient a closed-form weighted sum

This module does NOT:
  - Compute task losses; the loss-composition and transient adapter tests own them
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from src.datasets.runtime import dataset_runtime_units as units
from src.learning.training import learning_training_adapter as adapter_contract
from src.learning.training import learning_training_budget as training_budget
from src.learning.training import learning_training_loop as loop

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# Four units of two cases: case 0 owns three units, case 1 one, so w = 2/3 and 2.
_CASES = (0, 0, 0, 1)
_VALUES = (1.0, 2.0, 4.0, 8.0)
_WEIGHTS = (2 / 3, 2 / 3, 2 / 3, 2.0)


class _Units(Dataset[dict[str, torch.Tensor]]):
    """Return one scalar value per unit."""

    def __len__(self) -> int:
        return len(_VALUES)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor(_VALUES[index], dtype=torch.float64)}


class _Scalar(nn.Module):
    """Hold one trainable scalar."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))


class _ScalarTask:
    """Give every unit the loss ``weight * value``, two supervised and three forward transitions."""

    def __init__(self) -> None:
        self.groups: list[tuple[int, ...]] = []

    def begin_group(self, group: units.UnitGroup, *, progress: float) -> None:
        del progress
        self.groups.append(group.items)

    def unit_losses(self, model: nn.Module, raw_batch: Any, loss: nn.Module, *, positions: tuple[int, ...]) -> Iterator[adapter_contract.UnitLosses]:
        del loss
        total = model.weight * raw_batch["value"]
        yield adapter_contract.UnitLosses(
            components={"total": total, "data": total}, supervised_transitions=2 * len(positions), forward_transitions=3 * len(positions)
        )


class _RecordingSGD(SGD):
    """Record the scalar gradient of every optimizer step."""

    def __init__(self, parameters: Any) -> None:
        super().__init__(parameters, lr=0.0)
        self.gradients: list[float] = []

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        gradient = self.param_groups[0]["params"][0].grad
        self.gradients.append(float(gradient.item()))
        return super().step(closure)


class _SlowUnits(_Units):
    """Wait in the loader before every unit, so the optimisation clock must leave the wait out."""

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        time.sleep(_LOADER_WAIT_SECONDS)
        return super().__getitem__(index)


_LOADER_WAIT_SECONDS = 0.05


def _train(
    memory_batch: int, *, budget: training_budget.StageBudget | None = None, dataset: Dataset[dict[str, torch.Tensor]] | None = None
) -> tuple[units.UnitSampler, _RecordingSGD, dict[str, float], _ScalarTask]:
    """Run one epoch of three-unit updates and return the sampler, the recorded steps, the telemetry and the task."""
    sampler = units.UnitSampler(
        unit_kind="window", cases=_CASES, keys=None, units_per_update=3, memory_batch=memory_batch, seed=4, full_set_transitions=len(_CASES)
    )
    sampler.begin_epoch(0)
    model = _Scalar()
    optimizer = _RecordingSGD(model.parameters())
    task = _ScalarTask()
    values = loop.train_update_groups(
        model,
        DataLoader(dataset or _Units(), batch_sampler=sampler),
        optimizer,
        nn.Identity(),
        torch.device("cpu"),
        task,
        budget=budget or training_budget.StageBudget(unit="units", total=1000),
    )
    return sampler, optimizer, values, task


def test_an_update_averages_case_weighted_unit_losses_over_the_planned_k() -> None:
    """
    Train two updates of three planned units, the last one holding the remaining unit.

    A full group divides by k = 3, not by the sum of its weights; the last group divides by
    its one unit. The memory split changes nothing, and each group is one optimizer step.
    """
    sampler, optimizer, values, task = _train(3)
    expected = [sum(_WEIGHTS[item] * _VALUES[item] for item in group) / len(group) for group in task.groups]

    assert [len(group) for group in task.groups] == [3, 1]
    assert optimizer.gradients == pytest.approx(expected, rel=1e-12)
    assert _train(1)[1].gradients == pytest.approx(expected, rel=1e-12)
    assert sampler.epoch_complete
    assert values["train/loss_total"] == pytest.approx(sum(w * v for w, v in zip(_WEIGHTS, _VALUES, strict=True)) / 4, rel=1e-12)
    distinct = sum(len({_CASES[item] for item in group}) for group in task.groups)
    assert (values["train/updates"], values["train/units_per_update"], values["train/last_group_units"]) == (2.0, 2.0, 1.0)
    assert values["train/distinct_cases_per_update"] == distinct / 2
    assert values["train/supervised_transitions_per_update"] == 4.0
    assert values["train/forward_transitions"] == 12.0
    assert (values["optimizer_steps"], values["gradient_predictions"], values["units"], values["update_distinct_cases"]) == (2.0, 8.0, 4.0, distinct)
    # Each committed update keeps its own units, distinct cases and supervised transitions, so the short last group shows.
    assert values["update_work"] == [[len(group), len({_CASES[item] for item in group}), 2 * len(group)] for group in task.groups]


def test_a_validation_point_stops_mid_epoch_after_the_crossing_group_and_the_epoch_resumes_at_the_next_group() -> None:
    """
    With a budget of 20 units every unit is one validation point.

    The first group of three crosses points 1 to 3, so it stops the epoch after one step; one validation
    there records points 1 and 2 as skipped. The next call trains only the remaining unit, which crosses
    point 4, and completes the epoch.
    """
    budget = training_budget.StageBudget(unit="units", total=20)
    sampler, _optimizer, values, task = _train(2, budget=budget)

    assert values["optimizer_steps"] == 1.0
    assert (budget.units, budget.reached_point, sampler.cursor) == (3, 3, 3)
    assert budget.validation_due
    assert not sampler.epoch_complete
    assert budget.record_validation() == 3
    assert budget.skipped_points == [1, 2]

    model = _Scalar()
    rest = loop.train_update_groups(
        model,
        DataLoader(_Units(), batch_sampler=sampler),
        SGD(model.parameters(), lr=0.0),
        nn.Identity(),
        torch.device("cpu"),
        task,
        budget=budget,
    )
    assert rest["train/updates"] == 1.0
    assert task.groups[-1] == (sampler.permutation[3],)
    assert sampler.epoch_complete
    assert (budget.units, budget.reached_point) == (4, 4)


def test_the_optimisation_clock_leaves_out_loader_waits() -> None:
    """A loader that sleeps before every unit adds its wait to the wall time but not to the budget's train_seconds."""
    budget = training_budget.StageBudget(unit="units", total=1000)
    started = time.perf_counter()
    _train(1, budget=budget, dataset=_SlowUnits())
    wall = time.perf_counter() - started

    assert wall >= len(_VALUES) * _LOADER_WAIT_SECONDS
    assert 0.0 < budget.train_seconds < _LOADER_WAIT_SECONDS
