"""Protect the stage learning-rate schedules and the stage budget with its validation points.

Responsibilities:
  - Check the half cosine at budget fractions 0, 1/2 and 1, its clipping and its rejection of invalid progress
  - Check that a stage restart starts at lr_scale times the A0 start rate and keeps the Adam moments and step counters
  - Check the plateau comparison kind: no floor, stepped only at validation points
  - Check the passes-to-count conversion, independent of k and with the full set's number for a Train subset
  - Check the validation points of unit and supervised-transition budgets, skipped points, and exact state restoration
  - Check that a zero-transition budget (A1 after a first-group divergence) is complete and that seconds never end a stage

Design principles:
  - Small synthetic optimizers and budgets stand in for a training stage; the loop tests own the wiring

This module does NOT:
  - Train models or publish checkpoints; the resume and lifecycle tests own those paths
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
import torch
from torch.optim.adamw import AdamW

from src import learning
from src.datasets.runtime import dataset_runtime_units as units

optim = learning.training.optim
budget_state = learning.training.budget

_START = 1.0e-2


def _config(scheduler: dict[str, Any] | None) -> dict[str, Any]:
    """Return the resolved-config slice that builds a schedule."""
    return {"optimizer": {"lr": _START}, "scheduler": scheduler, "evaluation": {"objective": {"direction": "minimize"}}}


def _stepped_adam() -> AdamW:
    """Return an AdamW with two groups whose moments and step counters already moved."""
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    optimizer = AdamW([{"params": [parameter]} for parameter in parameters], lr=3.0e-4)
    for _ in range(3):
        sum(parameter.square().sum() for parameter in parameters).backward()  # type: ignore[union-attr]
        optimizer.step()
        optimizer.zero_grad()
    return optimizer


@pytest.mark.parametrize(("progress", "factor"), [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (1.7, 0.0)])
def test_cosine_sets_the_half_cosine_rate_of_the_budget_fraction(progress: float, factor: float) -> None:
    """η(p) = η₀ (1 + cos πp) / 2, with p clipped at 1 and no floor."""
    optimizer = _stepped_adam()
    schedule = optim.build_scheduler(optimizer, _config({"kind": "cosine"}))
    assert isinstance(schedule, optim.CosineSchedule)

    schedule.advance(progress)
    schedule.validated(123.0)

    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([_START * factor] * 2, abs=1e-15)


@pytest.mark.parametrize("progress", [-0.1, math.nan, math.inf])
def test_cosine_rejects_invalid_progress(progress: float) -> None:
    """A negative or non-finite budget fraction is a broken budget, never a rate."""
    with pytest.raises(ValueError, match="progress"):
        optim.cosine_factor(progress)


def test_a_stage_restart_starts_at_the_scaled_a0_rate_and_keeps_the_adam_moments() -> None:
    """B and A1 build their own schedule: every group starts at lr_scale · A0's start rate; moments and steps stay."""
    optimizer = _stepped_adam()
    inherited = copy.deepcopy(optimizer.state_dict()["state"])

    schedule = optim.build_scheduler(optimizer, _config({"kind": "cosine"}), lr_scale=0.25)
    assert schedule is not None
    schedule.advance(0.0)

    assert [group["lr"] for group in optimizer.param_groups] == [0.25 * _START] * 2
    assert schedule.state_dict() == {"kind": "cosine", "start_rate": 0.25 * _START}
    for index, state in optimizer.state_dict()["state"].items():
        assert torch.equal(state["step"], inherited[index]["step"])
        assert torch.equal(state["exp_avg"], inherited[index]["exp_avg"])
        assert torch.equal(state["exp_avg_sq"], inherited[index]["exp_avg_sq"])


def test_a_constant_rate_has_no_schedule_and_invalid_scales_are_rejected() -> None:
    """``scheduler: null`` keeps the start rate; a non-positive or non-finite lr_scale fails before any rate changes."""
    optimizer = _stepped_adam()
    assert optim.build_scheduler(optimizer, _config(None), lr_scale=2.0) is None
    assert [group["lr"] for group in optimizer.param_groups] == [2.0 * _START] * 2
    for scale in (0.0, -1.0, math.inf, True):
        with pytest.raises(ValueError, match="lr_scale"):
            optim.build_scheduler(optimizer, _config(None), lr_scale=scale)  # type: ignore[arg-type]
    assert [group["lr"] for group in optimizer.param_groups] == [2.0 * _START] * 2


def test_the_plateau_comparison_halves_at_validation_points_without_a_floor() -> None:
    """The plateau kind ignores budget progress, halves after patience bad points and keeps halving below any floor."""
    optimizer = _stepped_adam()
    schedule = optim.build_scheduler(optimizer, _config({"kind": "reduce_on_plateau", "factor": 0.5, "patience": 2}))
    assert isinstance(schedule, optim.PlateauSchedule)

    schedule.advance(0.9)
    assert optimizer.param_groups[0]["lr"] == _START
    schedule.validated(1.0)
    for _ in range(3 * 20):
        schedule.validated(2.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(_START * 0.5**20)
    restored = optim.build_scheduler(_stepped_adam(), _config({"kind": "reduce_on_plateau", "factor": 0.5, "patience": 2}))
    assert restored is not None
    restored.load_state_dict(schedule.state_dict())
    assert restored.state_dict() == schedule.state_dict()


def test_a_passes_budget_counts_the_full_train_set_independently_of_k_and_of_a_subset() -> None:
    """A budget of E passes counts the full Train set's units (A0) or transitions (B: K x N); a subset sampler keeps both full counts."""
    full = units.UnitSampler(unit_kind="case", cases=range(12), keys=None, units_per_update=4, memory_batch=2, seed=0)
    subset = units.UnitSampler(
        unit_kind="window", cases=range(3), keys=None, units_per_update=1, memory_batch=1, seed=0, full_set_units=12, full_set_transitions=50
    )

    assert budget_state.passes_budget(5, full.full_set_units) == budget_state.passes_budget(5, subset.full_set_units) == 60
    assert (full.full_set_transitions, budget_state.passes_budget(5, subset.full_set_transitions)) == (12, 250)
    with pytest.raises(ValueError, match="full_set_units"):
        units.UnitSampler(unit_kind="case", cases=range(3), keys=None, units_per_update=1, memory_batch=1, seed=0, full_set_units=2)
    with pytest.raises(ValueError, match="full_set_transitions"):
        units.UnitSampler(unit_kind="trajectory", cases=range(3), keys=None, units_per_update=1, memory_batch=1, seed=0)


@pytest.mark.parametrize("unit", ["units", "supervised_transitions"])
def test_points_fall_at_each_twentieth_and_a_multi_point_group_records_the_skipped_points(unit: str) -> None:
    """Point j is due at j/20 of the budget in its own count; one validation covers a group that crossed several and records the rest."""
    budget = budget_state.StageBudget(unit=unit, total=40)  # type: ignore[arg-type]

    def commit(count: int) -> None:
        # The other count moves differently (a B window holds several transitions), so only the budget's own count decides.
        units_, transitions = (count, 5 * count) if unit == "units" else (1, count)
        budget.record_group(units=units_, supervised_transitions=transitions, train_seconds=100.0)

    commit(1)
    assert (budget.reached_point, budget.validation_due) == (0, False)
    commit(1)
    assert budget.record_validation() == 1
    commit(7)
    assert budget.record_validation() == 4
    assert budget.skipped_points == [2, 3]
    with pytest.raises(RuntimeError, match="due"):
        budget.record_validation()


@pytest.mark.parametrize(
    ("unit", "total", "spends", "point"),
    [
        ("supervised_transitions", 3, (1,), 6),
        ("supervised_transitions", 40, (17, 1), 9),
        ("units", 40, (18,), 9),
        ("units", 10**9, (15 * 10**7 - 1,), 2),
        ("units", 10**9, (15 * 10**7,), 3),
        ("supervised_transitions", 10**9, (10**9 - 1,), 19),
    ],
)
def test_a_spend_exactly_at_a_threshold_reaches_that_point(unit: str, total: int, spends: tuple[int, ...], point: int) -> None:
    """Point j is reached exactly once the spend reaches j/20 of the budget, an exact integer comparison in both units."""
    budget = budget_state.StageBudget(unit=unit, total=total)  # type: ignore[arg-type]
    for spend in spends:
        budget.record_group(units=spend, supervised_transitions=spend, train_seconds=0.1)

    assert budget.reached_point == point


def test_the_transition_budget_end_is_point_twenty_and_seconds_never_end_a_stage() -> None:
    """B and A1 end after the group that crosses their transitions, whatever seconds it took; crossing the total is point 20."""
    budget = budget_state.StageBudget(unit="supervised_transitions", total=100)
    budget.record_group(units=1, supervised_transitions=6, train_seconds=1.0e6)
    assert (budget.record_validation(), budget.complete) == (1, False)
    budget.record_group(units=3, supervised_transitions=96, train_seconds=0.0)

    assert (budget.complete, budget.progress, budget.overrun) == (True, 1.0, 2)
    assert budget.record_validation() == budget_state.VALIDATION_POINTS
    assert budget.skipped_points == list(range(2, 20))
    with pytest.raises(RuntimeError, match="completed stage budget"):
        budget.record_group(units=1, supervised_transitions=1, train_seconds=0.1)


def test_a_zero_transition_budget_is_complete_and_only_transition_budgets_may_be_zero() -> None:
    """An A1 matched to a B that diverged in its first group gets 0 transitions: complete from the start, no group admitted."""
    budget = budget_state.StageBudget(unit="supervised_transitions", total=0)

    assert (budget.complete, budget.progress, budget.reached_point) == (True, 1.0, 20)
    with pytest.raises(RuntimeError, match="completed stage budget"):
        budget.record_group(units=1, supervised_transitions=1, train_seconds=0.1)
    for unit, total in (("units", 0), ("supervised_transitions", 2.5), ("train_seconds", 10)):
        with pytest.raises(ValueError, match=r"(?i)budget"):
            budget_state.StageBudget(unit=unit, total=total)  # type: ignore[arg-type]


def test_a_restored_budget_continues_the_exact_rate_sequence_and_rejects_another_budget() -> None:
    """Stopping after four groups and restoring budget and schedule state gives the rates of an uninterrupted stage."""

    def rates(budget: budget_state.StageBudget, schedule: optim.LearningRateSchedule, optimizer: AdamW, groups: int) -> list[float]:
        observed = []
        for _ in range(groups):
            schedule.advance(budget.progress)
            observed.append(optimizer.param_groups[0]["lr"])
            budget.record_group(units=1, supervised_transitions=3, train_seconds=0.5)
        return observed

    def stage() -> tuple[budget_state.StageBudget, optim.LearningRateSchedule, AdamW]:
        optimizer = _stepped_adam()
        schedule = optim.build_scheduler(optimizer, _config({"kind": "cosine"}))
        assert schedule is not None
        return budget_state.StageBudget(unit="supervised_transitions", total=30), schedule, optimizer

    uninterrupted = rates(*stage(), groups=10)
    first, schedule, optimizer = stage()
    head = rates(first, schedule, optimizer, groups=4)
    resumed, resumed_schedule, resumed_optimizer = stage()
    resumed.load_state_dict(first.state_dict())
    resumed_schedule.load_state_dict(schedule.state_dict())

    assert head + rates(resumed, resumed_schedule, resumed_optimizer, groups=6) == uninterrupted
    assert uninterrupted[-1] > 0.0
    assert resumed.state_dict() == {**first.state_dict(), "units": 10, "supervised_transitions": 30, "train_seconds": 5.0}
    for unit, total in (("supervised_transitions", 31), ("units", 30)):
        with pytest.raises(ValueError, match="differs"):
            budget_state.StageBudget(unit=unit, total=total).load_state_dict(first.state_dict())  # type: ignore[arg-type]
