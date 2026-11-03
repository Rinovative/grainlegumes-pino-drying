"""Protect the learning-rate scaling applied to a restored teacher-handoff continuation.

Responsibilities:
  - Check the exact no-op at scale one and the per-group scaling of restored optimizer rates
  - Check that floors scale with the rates while plateau state survives and the last-rate record follows
  - Check rejection of invalid scales without touching optimizer or scheduler

Design principles:
  - A rejected scale leaves optimizer and scheduler untouched
  - Synthetic two-group optimizers with an explicit plateau history stand in for a restored A0 state

This module does NOT:
  - Restore handoff checkpoints or decide which stages are scaled
  - Train models or evaluate the scheduler's reduction schedule
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import learning

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

_RATES = (1.0e-3, 5.0e-4)
_FLOORS = (1.0e-6, 1.0e-5)


def _restored_state(*, floors: tuple[float, ...] = _FLOORS) -> tuple[AdamW, ReduceLROnPlateau]:
    """Return a two-group optimizer and a plateau scheduler that has already seen one bad epoch."""
    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in _RATES]
    optimizer = AdamW([{"params": [parameter], "lr": rate} for parameter, rate in zip(parameters, _RATES, strict=True)])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=list(floors))
    scheduler.step(1.0)
    scheduler.step(2.0)
    return optimizer, scheduler


def _rates(optimizer: Optimizer) -> list[float]:
    """Return the current learning rate of every parameter group."""
    return [group["lr"] for group in optimizer.param_groups]


def test_scale_one_keeps_every_restored_rate_and_scheduler_value() -> None:
    """The default continuation must be bit-identical to the restored state."""
    optimizer, scheduler = _restored_state()
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())

    returned = learning.training.optim.scale_restored_learning_rates(optimizer, scheduler, scale=1)

    assert returned == list(_RATES)
    assert optimizer.state_dict() == before_optimizer
    assert scheduler.state_dict() == before_scheduler


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_scale_multiplies_rates_and_floors_and_keeps_plateau_state(scale: float) -> None:
    """Scale every group and its floor together, align the last-rate record, and keep the plateau history."""
    optimizer, scheduler = _restored_state()
    before = copy.deepcopy(scheduler.state_dict())

    returned = learning.training.optim.scale_restored_learning_rates(optimizer, scheduler, scale=scale)

    expected = [rate * scale for rate in _RATES]
    assert returned == pytest.approx(expected)
    assert _rates(optimizer) == pytest.approx(expected)
    after = scheduler.state_dict()
    assert after.pop("_last_lr") == pytest.approx(expected)
    assert after.pop("min_lrs") == pytest.approx([floor * scale for floor in _FLOORS])
    before.pop("_last_lr")
    before.pop("min_lrs")
    assert after == before
    assert (after["best"], after["num_bad_epochs"]) == (1.0, 1)


def test_rate_at_its_floor_continues_at_the_scaled_floor() -> None:
    """A0 decayed to its floor still continues at a smaller scale and stays at that scaled floor."""
    optimizer, scheduler = _restored_state(floors=(1.0e-6, 5.0e-4))

    learning.training.optim.scale_restored_learning_rates(optimizer, scheduler, scale=0.1)
    for _ in range(4):
        scheduler.step(3.0)

    assert scheduler.min_lrs[1] == pytest.approx(5.0e-5)
    assert _rates(optimizer)[1] == pytest.approx(5.0e-5)
    assert _rates(optimizer)[0] == pytest.approx(0.5e-4)


def test_scale_without_a_scheduler_scales_only_the_rates() -> None:
    """A run without a plateau scheduler has no floor to carry along."""
    optimizer, _scheduler = _restored_state()

    returned = learning.training.optim.scale_restored_learning_rates(optimizer, None, scale=1.0e-4)

    assert returned == pytest.approx([rate * 1.0e-4 for rate in _RATES])
    assert _rates(optimizer) == pytest.approx(returned)


@pytest.mark.parametrize("scale", [0, -0.5, math.nan, math.inf, True, "0.5"])
def test_invalid_scale_is_rejected(scale: object) -> None:
    """Reject zero, negative, non-finite, boolean, and textual factors without touching any state."""
    optimizer, scheduler = _restored_state()
    before_scheduler = copy.deepcopy(scheduler.state_dict())

    with pytest.raises(ValueError, match="positive and finite"):
        learning.training.optim.scale_restored_learning_rates(optimizer, scheduler, scale=scale)  # type: ignore[arg-type]

    assert _rates(optimizer) == list(_RATES)
    assert scheduler.state_dict() == before_scheduler
