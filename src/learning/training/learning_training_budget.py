"""Track one training stage's budget, its committed work and its validation points.

Responsibilities:
  - Hold a stage budget in seen units (steady, A0) or in supervised transitions (B, A1)
  - Convert a budget in passes over the full Train set into its count, independently of units_per_update
  - Sum every committed optimizer group's units, supervised transitions and optimisation seconds
  - Derive the completed budget fraction that drives the learning-rate schedule and the curriculum, and the overrun
  - Mark the validation points at j/20 of the budget, for a matched A1 at B's recorded validations, and record points a
    group crossed without its own validation
  - Validate a replayed validation schedule, the one rule for an A1 budget and for B's receipt
  - Persist and restore the exact budget state

Design principles:
  - Progress counts only committed optimizer groups, so a resume restores it exactly
  - Point j is due once the committed spend reaches j/20 of the budget (point 20 is the budget end), or B's count at j in a replay
  - Both budget units are exact integers, so every threshold is an exact integer comparison
  - Optimisation seconds are measured and reported, never budgeted: hardware and node load change no step count
  - One validation covers every point the last group crossed; the others are plan deviations, never validated again

This module does NOT:
  - Measure time or step optimizers; the training loop owns the optimisation clock
  - Decide what a unit or a supervised transition is; the unit sampler and the task's loss shares own them
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

BudgetUnit = Literal["units", "supervised_transitions"]
VALIDATION_POINTS: Final = 20
# One replayed validation: its point and the committed supervised transitions at it.
_REPLAY_ENTRY: Final = 2
_STATE_KEYS: Final = frozenset({"unit", "total", "units", "supervised_transitions", "train_seconds", "validated_point", "skipped_points"})


def _positive_int(value: object, *, label: str) -> int:
    """Return one positive integer or raise a labelled error."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{label} must be a positive integer, got {value!r}."
        raise ValueError(msg)
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    """Return one non-negative integer or raise a labelled error."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{label} must be a non-negative integer, got {value!r}."
        raise ValueError(msg)
    return value


def _finite_nonnegative(value: object, *, label: str) -> float:
    """Return one finite non-negative number or raise a labelled error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        msg = f"{label} must be a finite non-negative number, got {value!r}."
        raise ValueError(msg)
    return float(value)


def validate_replay(replay: object, *, total: int) -> None:
    """Require a replayed schedule of ``[point, count]`` entries with strictly ascending points (1 to 20) and counts up to ``total``."""
    entries = replay if isinstance(replay, list) and all(isinstance(entry, list) and len(entry) == _REPLAY_ENTRY for entry in replay) else None
    values = [value for entry in entries or () for value in entry]
    points, counts = values[::2], values[1::2]
    if (
        entries is None
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or points != sorted(set(points))
        or counts != sorted(set(counts))
        or not all(1 <= point <= VALIDATION_POINTS for point in points)
        or not all(1 <= count <= total for count in counts)
    ):
        msg = f"Budget replay must list [point, committed transitions] with ascending points 1 to 20 and positive counts, got {replay!r}."
        raise ValueError(msg)


def passes_budget(budget_epochs: int, full_set_count: int) -> int:
    """
    Return a budget of ``budget_epochs`` passes over the full Train set in that set's count.

    ``full_set_count`` is the full Train role's units (steady, A0) or supervised
    transitions N (B: K x N). A Train subset (learning curve) passes the full set's
    count, so it trains as long as a full run; the number never depends on ``units_per_update``.
    """
    return _positive_int(budget_epochs, label="budget_epochs") * _positive_int(full_set_count, label="full_set_count")


@dataclass(slots=True)
class StageBudget:
    """
    Account one stage's committed work against its budget.

    ``total`` is the budget in ``unit``: an integer number of seen units or of supervised
    transitions. ``units``, ``supervised_transitions`` and ``train_seconds`` are the committed
    totals of this stage; the seconds are only reported. ``validated_point`` is the last
    validation point (0 before the first); ``skipped_points`` are crossed points whose group
    also crossed a later point and so were validated together with it. ``replay``, set only for
    a matched A1, is B's validation schedule from its receipt, ``[point, committed transitions]``
    per validation B ran: A1 reaches point j once it committed B's transitions of point j and
    no other point, so it validates once per B validation, at its first group end at or after
    B's count; a group that reaches two of them validates once and skips the earlier point.
    Example: B validated ``[[2, 5], [5, 10]]`` and diverged at 12; A1 in groups of 2 validates
    point 2 at 6 and point 5 at 10, and ends at 12 without a validation.
    """

    unit: BudgetUnit
    total: int
    units: int = 0
    supervised_transitions: int = 0
    train_seconds: float = 0.0
    validated_point: int = 0
    skipped_points: list[int] = field(default_factory=list)
    replay: list[list[int]] | None = None

    def __post_init__(self) -> None:
        """Reject an inconsistent budget state."""
        if self.unit == "units":
            _positive_int(self.total, label="Unit budget total")
        elif self.unit == "supervised_transitions":
            # A state may hold zero transitions, complete from the start: the persisted budget of an A1 matched to a B that
            # diverged before its first group. Which total a stage may hold is admitted against its configured target
            # (experiments.run), positive everywhere except that A1.
            _nonnegative_int(self.total, label="Supervised-transition budget total")
        else:
            msg = f"Budget unit must be 'units' or 'supervised_transitions', got {self.unit!r}."
            raise ValueError(msg)
        if self.replay is not None:
            validate_replay(self.replay, total=self.total)
        _nonnegative_int(self.units, label="Budget units")
        _nonnegative_int(self.supervised_transitions, label="Budget supervised_transitions")
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
    def spent(self) -> int:
        """Return the committed work in the budget's unit."""
        return self.units if self.unit == "units" else self.supervised_transitions

    @property
    def overrun(self) -> int:
        """Return the committed work beyond the budget in its unit (negative while the budget is not spent)."""
        return self.spent - self.total

    @property
    def complete(self) -> bool:
        """Return whether the committed work reached the budget (validation point 20 unless the points replay B's)."""
        return self.spent >= self.total

    @property
    def progress(self) -> float:
        """Return the completed budget fraction p, clipped at 1 and exactly 1 once the budget is complete."""
        return 1.0 if self.complete else min(1.0, self.spent / self.total)

    @property
    def reached_point(self) -> int:
        """
        Return the number of validation thresholds the committed work reached.

        Threshold j is ``j * total / 20``, the last one ``total`` itself, and every threshold,
        completion included, is reached by the same rule (``_reaches``). With ``replay`` it is the
        last replayed point whose count the committed work reached (0 before the first).
        """
        if self.replay is not None:
            return max((point for point, count in self.replay if count <= self.spent), default=0)
        return sum(1 for point in range(1, VALIDATION_POINTS + 1) if self._reaches(point))

    def _reaches(self, point: int) -> bool:
        """Return whether the committed spend reached validation point ``point``: ``spent * 20 >= point * total`` in integers."""
        return self.spent * VALIDATION_POINTS >= point * self.total

    @property
    def validation_due(self) -> bool:
        """Return whether a validation point was reached after the last validation."""
        return self.reached_point > self.validated_point

    @property
    def validated_points(self) -> tuple[int, ...]:
        """Return the points validated so far: every point up to ``validated_point`` that no group skipped."""
        skipped = set(self.skipped_points)
        return tuple(point for point in range(1, self.validated_point + 1) if point not in skipped)

    def record_group(self, *, units: int, supervised_transitions: int, train_seconds: float) -> None:
        """Commit one completed optimizer group's units, supervised transitions and optimisation seconds."""
        if self.complete:
            msg = "No optimizer group may follow a completed stage budget."
            raise RuntimeError(msg)
        self.units += _positive_int(units, label="Group units")
        self.supervised_transitions += _positive_int(supervised_transitions, label="Group supervised_transitions")
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
            "supervised_transitions": self.supervised_transitions,
            "train_seconds": self.train_seconds,
            "validated_point": self.validated_point,
            "skipped_points": list(self.skipped_points),
            **({} if self.replay is None else {"replay": [list(entry) for entry in self.replay]}),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state of this budget; reject another unit, total or replayed schedule."""
        restored = StageBudget(**validate_state(state))
        if (restored.unit, restored.total, restored.replay) != (self.unit, self.total, self.replay):
            msg = f"Saved budget {restored.total} {restored.unit} (replay {restored.replay}) differs from this stage's {self.total} {self.unit}."
            raise ValueError(msg)
        self.units = restored.units
        self.supervised_transitions = restored.supervised_transitions
        self.train_seconds = restored.train_seconds
        self.validated_point = restored.validated_point
        self.skipped_points = list(restored.skipped_points)


def validate_state(state: object) -> dict[str, Any]:
    """Return an isolated copy of a valid budget state, ``replay`` only for a matched A1; reject a malformed one."""
    if not isinstance(state, Mapping) or set(state) - {"replay"} != _STATE_KEYS or state.get("replay", []) is None:
        msg = f"Budget state keys must be exactly {sorted(_STATE_KEYS)}, a matched A1's also replay."
        raise ValueError(msg)
    return StageBudget(**state).state_dict()
