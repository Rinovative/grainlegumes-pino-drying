"""Provide validated stored-time coordinates, master-time resolution, and physical-time axis presentation.

Responsibilities:
  - Validate canonical irregular time coordinates and form their exact union
  - Resolve comparison master timelines without interpolating scientific arrays
  - Format physical-time axes without resampling authoritative coordinates; an hour axis of a day or more ticks in the clock
    step (1 to 24 h) that gives it at most eight steps
  - Define the time windows of a line figure (the EDA start-up view, the evaluation's first, middle and last day, the
    drying phase τ) and style one axis per window: limits, ticks, and the time label under the bottom row only
  - Own the drying-phase rule of every pooled transient view (``phase_cohort``): τ = t / t_end on a fixed 20-point grid, the
    nearest stored state per case and point, the reached-only cohort with its censored and unknown counts and its label, and
    the views' hours-or-phase clock and per-series note

Design principles:
  - Physical time comes only from validated caller-supplied coordinates
  - Axis conversion and locators never change stored scientific values
  - A window keeps the data in hours and lets only its ticks and label read in days, so its unit may follow the shown data
  - A phase cohort is the same at every grid point, one stored state per reached case and point (ties to the earlier), never
    interpolated; a case without completion evidence is unknown, and a reached case without a post-origin hour is refused

This module does NOT:
  - Interpolate, align scientific arrays, or infer time coordinates
  - Navigate times (analysis_ui_navigation.py), render figures, load artifacts, or own panel export state
  - Build the axes grid of the windows (analysis_ui_plot_layout.py)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import matplotlib.ticker as mticker
import numpy as np

from src.analysis.presentation import analysis_field_labels as field_labels

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from matplotlib.axes import Axes

_HOURS_PER_DAY = 24.0
_LONG_RANGE_DAY_THRESHOLD_HOURS = 72.0
_WINDOW_TOLERANCE_HOURS: Final = 1.0e-9  # a point on a window's edge belongs to it
_BEFORE_END_DECIMALS: Final = 9  # hours before an end are rounded so equal offsets of different cases compare equal
_HOUR_WINDOW_WIDTH: Final = 0.55  # an hour column beside a day column of width 1
_HOUR_TICK_STEP_HOURS: Final = 6.0  # an hour window of at least the span below ticks every 6 h counted from its start
_HOUR_TICK_MIN_SPAN_HOURS: Final = 12.0
_HOUR_TICK_STEPS: Final = (1.0, 2.0, 3.0, 6.0, 12.0, 24.0)  # the clock steps an hour axis of a day or more ticks in
_MAX_HOUR_TICK_STEPS: Final = 8
_EMPTY_WINDOW_HOURS: Final = 1.2  # span of an open window without any shown data
_TAU_TICK_STEP: Final = 0.2
_PHASE_POINTS: Final = 20  # the drying-phase grid τ = 0.05, 0.10, ..., 1.00
_PHASE_TIE: Final = 1.0e-9  # in τ: states this close to equally near count as a tie, which the earlier one takes

type Completion = Literal["reached", "censored", "unknown"]


@dataclass(frozen=True, slots=True)
class ResolvedPhysicalTime:
    """Describe the exact per-case snapshot used at one master time."""

    physical_time: float
    exact_master_match: bool
    final_hold: bool


@dataclass(frozen=True, slots=True)
class PhysicalTimeDisplay:
    """Describe one deterministic display-only physical-time axis transform."""

    unit: str
    divisor_hours: float
    minimum_hours: float
    maximum_hours: float
    major_interval_hours: float | None = None

    def values(self, physical_hours: Iterable[float]) -> np.ndarray:
        """Convert finite authoritative hours for display without resampling."""
        values = np.asarray(
            tuple(float(value) for value in physical_hours),
            dtype=np.float64,
        )
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            message = "Physical-time display values must be one non-empty finite sequence."
            raise ValueError(message)
        return values / self.divisor_hours

    def configure(
        self,
        axis: Axes,
        *,
        dimension: Literal["x", "y"] = "x",
    ) -> None:
        """Apply the shared unit label and aligned tick policy to one dimension."""
        if dimension not in {"x", "y"}:
            message = "Physical-time axis dimension must be 'x' or 'y'."
            raise ValueError(message)
        lower = self.minimum_hours / self.divisor_hours
        upper = self.maximum_hours / self.divisor_hours
        if np.isclose(lower, upper):
            padding = max(abs(lower), 1.0) * 0.05
            lower -= padding
            upper += padding
        target_axis = axis.xaxis if dimension == "x" else axis.yaxis
        if dimension == "x":
            axis.set_xlim(lower, upper)
            axis.set_xlabel(f"Time [{self.unit}]")
        else:
            axis.set_ylim(lower, upper)
            axis.set_ylabel(f"Time [{self.unit}]")

        if self.major_interval_hours is not None:
            target_axis.set_major_locator(
                mticker.MultipleLocator(
                    self.major_interval_hours / self.divisor_hours,
                )
            )
            target_axis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
        elif self.unit == "d":
            target_axis.set_major_locator(mticker.MultipleLocator(1.0))
            target_axis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
        elif self.maximum_hours - self.minimum_hours >= _HOURS_PER_DAY:
            target_axis.set_major_locator(mticker.MultipleLocator(_hour_tick_step(self.maximum_hours - self.minimum_hours)))
            target_axis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
        else:
            target_axis.set_major_locator(mticker.MaxNLocator(nbins=6, min_n_ticks=3))

        target_axis.set_minor_locator(mticker.NullLocator())
        axis.grid(axis=dimension, which="major", alpha=0.25)


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """
    Describe one column of a line figure: a stretch of physical time shown in its own unit.

    The data stay in hours; ``unit`` only decides the tick numbers and the label. ``stop_hours`` may be ``inf`` for
    "to the end of the shown data". A ``before_end`` window counts hours before each case's own end (zero or negative),
    so cases of different duration share one last window; ``trim_end_hours`` leaves the last hours of each case to that
    window, so the window beside it stops at each case's own end minus them, and ``not_before_hours`` leaves the clock hours
    up to it to the window before. ``open_start`` leaves the start hour itself to the window before, so a stored hour never
    lies in two windows. A window that needs a case's end refuses a call without ``end_hours``. A window in unit ``"1"``
    (``PHASE_WINDOW``) only styles an axis of the drying phase τ, whose values the caller resolves through ``phase_cohort``.
    """

    title: str
    start_hours: float
    stop_hours: float
    unit: Literal["h", "d", "1"]
    width_ratio: float
    before_end: bool = False
    trim_end_hours: float = 0.0
    not_before_hours: float = -math.inf
    open_start: bool = False

    def _required_end(self, end_hours: float | None) -> float:
        """Return the case's own end that a ``before_end`` or trimmed window needs."""
        if end_hours is None:
            message = f"The window {self.title!r} counts from each case's end and needs end_hours."
            raise ValueError(message)
        return end_hours

    def hours(self, physical_hours: Sequence[float] | np.ndarray, *, end_hours: float | None = None) -> np.ndarray:
        """Return the window coordinate of stored hours: the clock, or the hours before ``end_hours``."""
        values = np.asarray(physical_hours, dtype=np.float64)
        end = self._required_end(end_hours) if self.before_end or self.trim_end_hours else 0.0
        return np.round(values - end, _BEFORE_END_DECIMALS) if self.before_end else values

    def select(self, physical_hours: Sequence[float] | np.ndarray, *, end_hours: float | None = None) -> np.ndarray:
        """
        Return the mask of the stored hours inside the window: its stop included, its start only unless ``open_start``.

        Every bound is compared on the unrounded clock, and the two windows beside a boundary compare against the same
        threshold, so a stored hour never falls between them or into both; only ``hours`` rounds, for the display.
        """
        clock = np.asarray(physical_hours, dtype=np.float64)
        shift = self._required_end(end_hours) if self.before_end else 0.0
        lower, upper, tolerance = self.start_hours + shift, self.stop_hours + shift, _WINDOW_TOLERANCE_HOURS
        inside = (clock > lower + tolerance) if self.open_start else (clock >= lower - tolerance)
        inside &= (clock <= upper + tolerance) & (clock > self.not_before_hours + tolerance)
        if self.trim_end_hours:
            inside &= clock <= (self._required_end(end_hours) - self.trim_end_hours) + tolerance
        return inside


@dataclass(frozen=True, slots=True)
class PhaseCohort:
    """
    The reached cases pooled over the drying-phase grid and the stored state each contributes at every grid point.

    ``cases`` are the input positions of the pooled cases; ``states`` and ``hours`` (``[pooled case, grid point]``) hold the
    index and the hour of the stored state each takes, so ``hours[:, -1]`` is each case's t_end, its last stored hour.
    """

    cases: tuple[int, ...]
    states: np.ndarray
    hours: np.ndarray
    censored: int
    unknown: int

    def take(self, values_by_case: Sequence[np.ndarray]) -> np.ndarray:
        """Return ``[pooled case, grid point]`` values from per-case series aligned with the input cases and their stored times."""
        return np.asarray([np.asarray(values_by_case[case])[states] for case, states in zip(self.cases, self.states, strict=True)])

    def label(self) -> str:
        """Return the effective time of the pooled cases (t_end min-max and median in hours) and the reached, censored and unknown counts."""
        left_out = ", ".join(f"{count} {name}" for name, count in (("censored", self.censored), ("unknown", self.unknown)) if count)
        excluded = f"{left_out} not pooled" if left_out else ""
        if not self.cases:
            return "; ".join(filter(None, ("no reached case to pool", excluded)))
        ends = self.hours[:, -1]
        span = f"t_end {_hours_text(ends.min())}-{_hours_text(ends.max())} h, median {_hours_text(np.median(ends))} h"
        return f"{span}; " + ", ".join(filter(None, (f"{len(self.cases)} reached", excluded)))

    def hours_label(self, phase: float) -> str:
        """Return the range of stored hours the pooled cases resolve at one grid phase."""
        selected = self.hours[:, phase_index(phase)]
        if not selected.size:
            return f"τ = {phase:.2f}: no stored state"
        return f"τ = {phase:.2f}: stored t {_hours_text(selected.min())}-{_hours_text(selected.max())} h"


# The EDA's schedule and trajectory views: the day-scale main view from hour 1, then the hour-scale start-up view.
STARTUP_WINDOWS: Final = (
    TimeWindow("Physical time from 1 h to final support", 1.0, math.inf, "d", 1.0),
    TimeWindow("Startup: 0-1 h", 0.0, 1.0, "h", 0.34),
)
# The pooled views' drying phase τ = t / t_end: its axis, and its fixed grid of 20 post-origin points 0.05, ..., 1.
PHASE_WINDOW: Final = TimeWindow("Drying phase, τ = t / t_end", 0.0, 1.0, "1", 1.0)
PHASE_GRID: Final = np.arange(1, _PHASE_POINTS + 1) / _PHASE_POINTS


def _hour_tick_step(span_hours: float) -> float:
    """Return the tick step of an hour axis over ``span_hours``: the finest of 1, 2, 3, 6, 12 or 24 h with at most 8 steps, else whole days."""
    fitting = [step for step in _HOUR_TICK_STEPS if span_hours / step <= _MAX_HOUR_TICK_STEPS]
    return fitting[0] if fitting else _HOURS_PER_DAY * math.ceil(span_hours / (_HOURS_PER_DAY * _MAX_HOUR_TICK_STEPS))


def _hours_text(hours: float) -> str:
    """Return hours with at most one decimal, the one number format of the phase labels."""
    return f"{round(float(hours), 1):g}"


def is_phase(time_axis: str) -> bool:
    """Return whether a pooled view's clock is the drying phase; a clock other than ``"hours"`` or ``"phase"`` raises ``ValueError``."""
    if time_axis not in {"hours", "phase"}:
        message = f"time_axis must be 'hours' or 'phase', not {time_axis!r}."
        raise ValueError(message)
    return time_axis == "phase"


def phase_note(texts: Mapping[str, str]) -> str | None:
    """Return one «series: text» line per pooled series (the text alone for an unlabelled one), ``None`` without any series."""
    return "\n".join(f"{label}: {text}" if label else text for label, text in texts.items()) or None


def completion_status(*, available: bool, reached: object) -> Completion:
    """Classify a case's reference completion from existing evidence: reached, censored (stopped without the target) or unknown."""
    if not available or not isinstance(reached, bool):
        return "unknown"
    return "reached" if reached else "censored"


def phase_index(phase: float) -> int:
    """Return the position of one phase on ``PHASE_GRID``; a phase off the grid raises ``ValueError``."""
    matches = np.flatnonzero(np.isclose(PHASE_GRID, float(phase), rtol=0.0, atol=_PHASE_TIE))
    if matches.size != 1:
        message = f"Drying phase {phase!r} is not on the 20-point phase grid 0.05, 0.10, ..., 1.00."
        raise ValueError(message)
    return int(matches[0])


def phase_cohort(times_by_case: Sequence[Sequence[float] | np.ndarray], completion: Sequence[Completion]) -> PhaseCohort:
    """
    Pool the reached cases over ``PHASE_GRID``: each contributes its stored state nearest to τ = t / t_end at every grid point.

    t_end is a case's last stored hour (at most one interval before its exact stop), so every reached case spans the whole
    grid; every stored state of ``times_by_case``, the origin included, is a candidate, a tie goes to the earlier state and
    nothing is interpolated. Censored and unknown cases are counted, not pooled. A reached case whose last stored hour is not
    after 0 h has no phase and raises ``ValueError`` naming its position.
    """
    if len(times_by_case) != len(completion) or any(status not in {"reached", "censored", "unknown"} for status in completion):
        message = "A phase cohort needs one completion status (reached, censored or unknown) per case."
        raise ValueError(message)
    cases, states, hours = [], [], []
    for case, (times, status) in enumerate(zip(times_by_case, completion, strict=True)):
        if status != "reached":
            continue
        clock = np.asarray(validated_times(times), dtype=np.float64)
        if clock[-1] <= 0.0:
            message = f"The reached case at position {case} ends at {clock[-1]:g} h, so it has no drying phase τ = t / t_end."
            raise ValueError(message)
        distance = np.abs(clock[:, None] / clock[-1] - PHASE_GRID[None, :])
        nearest = np.argmax(distance <= distance.min(axis=0) + _PHASE_TIE, axis=0)  # the first, earliest, of equally near states
        cases.append(case)
        states.append(nearest)
        hours.append(clock[nearest])
    shape = (len(cases), PHASE_GRID.size)
    return PhaseCohort(
        cases=tuple(cases),
        states=np.asarray(states, dtype=np.int64).reshape(shape),
        hours=np.asarray(hours, dtype=np.float64).reshape(shape),
        censored=sum(status == "censored" for status in completion),
        unknown=sum(status == "unknown" for status in completion),
    )


def format_terminal_physical_time_hours(
    physical_time_hours: float,
    *,
    include_unit: bool = True,
) -> str:
    """Format one final or terminal physical time to the nearest full hour."""
    if isinstance(physical_time_hours, bool):
        message = "Terminal physical time must be one finite non-negative scalar."
        raise TypeError(message)
    value = float(physical_time_hours)
    if not np.isfinite(value) or value < 0.0:
        message = "Terminal physical time must be one finite non-negative scalar."
        raise ValueError(message)
    if not isinstance(include_unit, bool):
        message = "include_unit must be boolean."
        raise TypeError(message)
    formatted = f"{value:.0f}"
    return f"{formatted} h" if include_unit else formatted


def physical_time_display(
    physical_hours: Iterable[float],
    *,
    preferred_unit: str = "auto",
    include_zero: bool = False,
    right_margin_hours: float = 0.0,
    major_interval_hours: float | None = None,
) -> PhysicalTimeDisplay:
    """Resolve one display contract from finite authoritative physical hours."""
    values = np.asarray(
        tuple(float(value) for value in physical_hours),
        dtype=np.float64,
    )
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        message = "Physical-time axes require one non-empty finite sequence in hours."
        raise ValueError(message)
    if preferred_unit not in {"auto", "h", "d"}:
        message = "preferred_unit must be 'auto', 'h', or 'd'."
        raise ValueError(message)
    if not isinstance(include_zero, bool):
        message = "include_zero must be boolean."
        raise TypeError(message)
    margin = float(right_margin_hours)
    if not np.isfinite(margin) or margin < 0.0:
        message = "right_margin_hours must be finite and non-negative."
        raise ValueError(message)
    if major_interval_hours is not None and (
        isinstance(major_interval_hours, bool) or not np.isfinite(float(major_interval_hours)) or float(major_interval_hours) <= 0.0
    ):
        message = "major_interval_hours must be positive finite hours or None."
        raise ValueError(message)
    minimum = float(np.min(values))
    if include_zero:
        minimum = min(0.0, minimum)
    maximum = float(np.max(values)) + margin
    span = maximum - minimum
    unit = "d" if preferred_unit == "d" or (preferred_unit == "auto" and span >= _LONG_RANGE_DAY_THRESHOLD_HOURS) else "h"
    return PhysicalTimeDisplay(
        unit=unit,
        divisor_hours=(_HOURS_PER_DAY if unit == "d" else 1.0),
        minimum_hours=minimum,
        maximum_hours=maximum,
        major_interval_hours=(None if major_interval_hours is None else float(major_interval_hours)),
    )


def drying_windows(end_hours: float, *, before_end: bool) -> tuple[TimeWindow, ...]:
    """
    Return the first 24 h, the middle in days and the last 24 h of a run, or one window for a run below 72 h.

    With ``before_end`` the last window counts hours before each case's own end, so cases of different duration share it;
    ``end_hours`` is then the longest end. Each stored hour of a case lies in exactly one window, with half-open bounds: the
    first is [0, 24] h on the clock, the middle (24 h, end - 24 h] and the last (the later of end - 24 h and 24 h, end], so a
    case ending at 24 h or 48 h has no hour in two windows, a 40 h case is in the last window from the first hour after 24 h
    and a case under 24 h only in the first. A window may start a fraction of an interval after its edge.
    """
    if end_hours < _LONG_RANGE_DAY_THRESHOLD_HOURS:
        return (TimeWindow(f"whole run, {end_hours:.0f} h", 0.0, end_hours, "h", 1.0),)
    tail = end_hours - _HOURS_PER_DAY
    last = (-_HOURS_PER_DAY, 0.0) if before_end else (tail, end_hours)
    return (
        TimeWindow("first 24 h", 0.0, _HOURS_PER_DAY, "h", _HOUR_WINDOW_WIDTH),
        TimeWindow(
            f"day 1 to {tail / _HOURS_PER_DAY:.1f}",
            _HOURS_PER_DAY,
            tail,
            "d",
            1.0,
            trim_end_hours=_HOURS_PER_DAY if before_end else 0.0,
            open_start=True,
        ),
        TimeWindow(
            "last 24 h",
            *last,
            "h",
            _HOUR_WINDOW_WIDTH,
            before_end=before_end,
            not_before_hours=_HOURS_PER_DAY if before_end else -math.inf,
            open_start=True,
        ),
    )


def style_window_axis(axis: Axes, window: TimeWindow, *, shown_end_hours: float | None = None, bottom: bool = True) -> None:
    """
    Set a window's limits, ticks and time label on one axis; ``bottom`` False leaves the tick numbers and the label off.

    An open window (``stop_hours`` ``inf``) ends at ``shown_end_hours``, or spans a short default without shown data. A day
    window that ends before one day reads in hours. An hour window shorter than 12 h has automatic ticks plus one at each end.
    A τ window (unit "1") spans its fractions with ticks every 0.2.
    """
    stop = window.stop_hours
    if not math.isfinite(stop):
        has_data = shown_end_hours is not None and shown_end_hours > window.start_hours
        stop = shown_end_hours if has_data and shown_end_hours is not None else window.start_hours + _EMPTY_WINDOW_HOURS
    unit = "d" if window.unit == "d" and stop >= _HOURS_PER_DAY else "1" if window.unit == "1" else "h"
    axis.set_xlim(window.start_hours, stop)
    if unit == "1":
        axis.xaxis.set_major_locator(mticker.MultipleLocator(_TAU_TICK_STEP))
        axis.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    elif unit == "d":
        axis.xaxis.set_major_locator(mticker.MultipleLocator(_HOURS_PER_DAY))
        axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda hours, _position: f"{hours / _HOURS_PER_DAY:g}"))
    else:
        if stop - window.start_hours >= _HOUR_TICK_MIN_SPAN_HOURS:
            axis.xaxis.set_major_locator(mticker.MultipleLocator(_HOUR_TICK_STEP_HOURS, offset=window.start_hours))
        else:
            automatic = np.asarray(mticker.MaxNLocator(nbins=6, min_n_ticks=3).tick_values(window.start_hours, stop))
            inside = automatic[(automatic >= window.start_hours) & (automatic <= stop)]
            axis.xaxis.set_major_locator(mticker.FixedLocator(sorted({*inside.tolist(), window.start_hours, stop})))
        axis.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    axis.xaxis.set_minor_locator(mticker.NullLocator())
    axis.grid(axis="x", which="major", alpha=0.25)
    axis.tick_params(axis="x", labelbottom=bottom)
    label = f"τ = t / t_end [{field_labels.display_unit('1')}]" if unit == "1" else f"Time{' before end' if window.before_end else ''} [{unit}]"
    axis.set_xlabel(label if bottom else "")


def validated_times(values: Iterable[float]) -> tuple[float, ...]:
    """Return finite, strictly increasing canonical physical-time coordinates."""
    times = tuple(float(value) for value in values)
    array = np.asarray(times, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all() or np.any(np.diff(array) <= 0.0):
        message = "Stored physical-time coordinates must be finite and strictly increasing."
        raise ValueError(message)
    return times


def ordered_time_union(
    time_coordinates: Sequence[Iterable[float]],
) -> tuple[float, ...]:
    """Return the sorted union of authoritative stored physical-time grids."""
    grids = tuple(validated_times(values) for values in time_coordinates)
    if not grids:
        message = "Physical-time union requires at least one coordinate grid."
        raise ValueError(message)
    return tuple(sorted({value for grid in grids for value in grid}))


def resolve_master_physical_time(
    available_times: Iterable[float],
    master_time: float,
) -> ResolvedPhysicalTime:
    """Resolve exact or latest-prior stored evidence without interpolation."""
    times = validated_times(available_times)
    requested = float(master_time)
    if not np.isfinite(requested):
        message = "Master physical time must be finite."
        raise ValueError(message)
    values = np.asarray(times, dtype=np.float64)
    position = int(np.searchsorted(values, requested, side="right") - 1)
    if position < 0:
        message = f"Master time {requested:g} h precedes the first stored case time {times[0]:g} h."
        raise ValueError(message)
    actual = times[position]
    return ResolvedPhysicalTime(
        physical_time=actual,
        exact_master_match=bool(actual == requested),
        final_hold=bool(position == len(times) - 1 and requested >= actual),
    )
