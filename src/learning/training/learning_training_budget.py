"""Track one training stage's budget, its optimisation clock and its validation points.

Responsibilities:
  - Hold a stage budget in seen units (steady, A0) or in optimisation-clock seconds (B, A1)
  - Convert a budget in epochs of the full Train set into units, independently of units_per_update
  - Sum every committed optimizer group's units and optimisation seconds
  - Derive the completed budget fraction that drives the learning-rate schedule and the curriculum, and the overrun
  - Mark the validation points at j/20 of the budget and record points a group crossed without its own validation
  - Persist and restore the exact budget state

Design principles:
  - Progress counts only committed optimizer groups, so a resume restores it exactly
  - Point j is due once the committed spend reaches j/20 of the budget; point 20 is the budget end
  - Unit budgets compare exact integers; a seconds budget ends exactly at its total (no negative overrun) and
    compares its intermediate thresholds within a relative tolerance of 1e-9, because j·total/20 rounds
  - One validation covers every point the last group crossed; the others are plan deviations, never validated again

This module does NOT:
  - Measure time or step optimizers; the training loop owns the optimisation clock
  - Decide what a unit is; the unit sampler owns the unit kind
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

BudgetUnit = Literal["units", "train_seconds"]
VALIDATION_POINTS: Final = 20
_STATE_KEYS: Final = frozenset({"unit", "total", "units", "train_seconds", "validated_point", "skipped_points"})
# Intermediate seconds thresholds only: j·total/20 rounds, so a spend exactly at a decimal threshold can land just below it.
_RELATIVE_TOLERANCE: Final = 1e-9


def _positive_int(value: object, *, label: str) -> int:
    """Return one positive integer or raise a labelled error."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{label} must be a positive integer, got {value!r}."
        raise ValueError(msg)
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    """Return one finite non-negative number or raise a labelled error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        msg = f"{label} must be a finite non-negative number, got {value!r}."
        raise ValueError(msg)
    return float(value)


def units_budget(budget_epochs: int, full_set_units: int) -> int:
    """
    Return a budget of ``budget_epochs`` passes over the full Train set in seen units.

    A Train subset (learning curve) passes the full set's unit count, so it sees as
    many units as a full run; the number never depends on ``units_per_update``.
    """
    return _positive_int(budget_epochs, label="budget_epochs") * _positive_int(full_set_units, label="full_set_units")


@dataclass(slots=True)
class StageBudget:
    """
    Account one stage's committed work against its budget.

    ``total`` is the budget in ``unit``: an integer number of seen units, or positive
    finite optimisation-clock seconds. ``units`` and ``train_seconds`` are the committed
    totals of this stage in both measures. ``validated_point`` is the last validation
    point (0 before the first); ``skipped_points`` are crossed points whose group also
    crossed a later point and so were validated together with it.
    """

    unit: BudgetUnit
    total: float
    units: int = 0
    train_seconds: float = 0.0
    validated_point: int = 0
    skipped_points: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject an inconsistent budget state."""
        if self.unit == "units":
            _positive_int(self.total, label="Unit budget total")
        elif self.unit == "train_seconds":
            # A state may hold zero seconds, complete from the start: the persisted budget of an A1 matched to a B that
            # diverged before its first group. Which total a stage may hold is admitted against its configured target
            # (experiments.run), positive everywhere except that A1.
            _finite_nonnegative(self.total, label="Seconds budget total")
        else:
            msg = f"Budget unit must be 'units' or 'train_seconds', got {self.unit!r}."
            raise ValueError(msg)
        if isinstance(self.units, bool) or not isinstance(self.units, int) or self.units < 0:
            msg = "Budget units must be a non-negative integer."
            raise ValueError(msg)
        _finite_nonnegative(self.train_seconds, label="Budget train_seconds")
        if isinstance(self.validated_point, bool) or not isinstance(self.validated_point, int) or not 0 <= self.validated_point <= self.reached_point:
            msg = "Budget validated_point must lie between 0 and the reached point."
            raise ValueError(msg)
        skipped = self.skipped_points
        if (
            not isinstance(skipped, list)
            or any(isinstance(point, bool) or not isinstance(point, int) or not 1 <= point < self.validated_point for point in skipped)
            or skipped != sorted(set(skipped))
        ):
            msg = "Budget skipped_points must be ascending unique points before the validated point."
            raise ValueError(msg)

    @property
    def spent(self) -> float:
        """Return the committed work in the budget's unit."""
        return float(self.units) if self.unit == "units" else self.train_seconds

    @property
    def overrun(self) -> float:
        """Return the committed work beyond the budget in its unit (negative while the budget is not spent)."""
        return self.spent - self.total

    @property
    def complete(self) -> bool:
        """Return whether the committed work reached the budget (validation point 20)."""
        return self._reaches(VALIDATION_POINTS)

    @property
    def progress(self) -> float:
        """Return the completed budget fraction p, clipped at 1 and exactly 1 once the budget is complete."""
        return 1.0 if self.complete else min(1.0, self.spent / self.total)

    @property
    def reached_point(self) -> int:
        """
        Return the number of validation thresholds the committed work reached.

        Threshold j is ``j * total / 20``, the last one ``total`` itself, and every
        threshold, completion included, is reached by the same rule (``_reaches``).
        """
        return sum(1 for point in range(1, VALIDATION_POINTS + 1) if self._reaches(point))

    def _reaches(self, point: int) -> bool:
        """
        Return whether the committed spend reached validation point ``point``.

        A unit budget compares exact integers (``units * 20 >= point * total``, so point 20
        is ``units >= total``). A seconds budget reaches its end exactly at ``total``, so a
        completed stage never spent less than its budget; its intermediate thresholds
        compare within the relative tolerance, because j·total/20 rounds.
        """
        if self.unit == "units":
            return self.units * VALIDATION_POINTS >= point * int(self.total)
        if point == VALIDATION_POINTS:
            return self.train_seconds >= self.total
        return self.train_seconds >= point * self.total / VALIDATION_POINTS * (1.0 - _RELATIVE_TOLERANCE)

    @property
    def validation_due(self) -> bool:
        """Return whether a validation point was reached after the last validation."""
        return self.reached_point > self.validated_point

    def record_group(self, *, units: int, train_seconds: float) -> None:
        """Commit one completed optimizer group's units and optimisation seconds."""
        if self.complete:
            msg = "No optimizer group may follow a completed stage budget."
            raise RuntimeError(msg)
        self.units += _positive_int(units, label="Group units")
        self.train_seconds += _finite_nonnegative(train_seconds, label="Group train_seconds")

    def record_validation(self) -> int:
        """Record a validation at the reached point and return it; earlier crossed points become plan deviations."""
        point = self.reached_point
        if point <= self.validated_point:
            msg = "A validation is due only after a new budget point was reached."
            raise RuntimeError(msg)
        self.skipped_points.extend(range(self.validated_point + 1, point))
        self.validated_point = point
        return point

    def state_dict(self) -> dict[str, Any]:
        """Return the exact budget state."""
        return {
            "unit": self.unit,
            "total": self.total,
            "units": self.units,
            "train_seconds": self.train_seconds,
            "validated_point": self.validated_point,
            "skipped_points": list(self.skipped_points),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state of this budget; reject another unit or total."""
        if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
            msg = f"Budget state keys must be exactly {sorted(_STATE_KEYS)}."
            raise ValueError(msg)
        restored = StageBudget(**{key: state[key] for key in _STATE_KEYS})
        if (restored.unit, restored.total) != (self.unit, self.total):
            msg = f"Saved budget {restored.total} {restored.unit} differs from this stage's {self.total} {self.unit}."
            raise ValueError(msg)
        self.units = restored.units
        self.train_seconds = restored.train_seconds
        self.validated_point = restored.validated_point
        self.skipped_points = list(restored.skipped_points)


def validate_state(state: object) -> dict[str, Any]:
    """Return an isolated copy of a valid budget state; reject a malformed one."""
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        msg = f"Budget state keys must be exactly {sorted(_STATE_KEYS)}."
        raise ValueError(msg)
    return StageBudget(**{key: state[key] for key in _STATE_KEYS}).state_dict()
