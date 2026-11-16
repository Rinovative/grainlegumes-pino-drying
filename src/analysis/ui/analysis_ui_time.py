"""Provide validated stored-time coordinates, master-time resolution, and physical-time axis presentation.

Responsibilities:
  - Validate canonical irregular time coordinates and form their exact union
  - Resolve comparison master timelines without interpolating scientific arrays
  - Format physical-time axes without resampling authoritative coordinates
  - Define the time windows of a line figure (the EDA start-up view, the evaluation's first, middle and last day, the
    normalized time τ = t / t_stop of each case's own end) and style one axis per window: limits, ticks, and the time
    label under the bottom row only

Design principles:
  - Physical time comes only from validated caller-supplied coordinates
  - Axis conversion and locators never change stored scientific values
  - A window keeps the data in hours and lets only its ticks and label read in days, so its unit may follow the shown data
  - A τ window compares its bounds on the unrounded clock scaled by each case's end, so every stored hour in [0, end] lies in it

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
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes

_HOURS_PER_DAY = 24.0
_LONG_RANGE_DAY_THRESHOLD_HOURS = 72.0
_WINDOW_TOLERANCE_HOURS: Final = 1.0e-9  # a point on a window's edge belongs to it
_BEFORE_END_DECIMALS: Final = 9  # hours before an end are rounded so equal offsets of different cases compare equal
_HOUR_WINDOW_WIDTH: Final = 0.55  # an hour column beside a day column of width 1
_HOUR_TICK_STEP_HOURS: Final = 6.0  # an hour window of at least the span below ticks every 6 h counted from its start
_HOUR_TICK_MIN_SPAN_HOURS: Final = 12.0
_EMPTY_WINDOW_HOURS: Final = 1.2  # span of an open window without any shown data
_TAU_TICK_STEP: Final = 0.2


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
            target_axis.set_major_locator(mticker.MultipleLocator(_HOURS_PER_DAY))
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
    lies in two windows. A window in unit ``"1"`` counts the normalized time τ = t / ``end_hours`` of each case: its bounds
    are fractions of the case's end and its coordinate is τ. A window that needs a case's end refuses a call without
    ``end_hours``.
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
        """Return the window coordinate of stored hours: the clock, the hours before ``end_hours`` or τ = t / ``end_hours`` (unit "1")."""
        values = np.asarray(physical_hours, dtype=np.float64)
        if self.unit == "1":
            return values / self._required_end(end_hours)
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
        scale = self._required_end(end_hours) if self.unit == "1" else 1.0
        lower, upper, tolerance = self.start_hours * scale + shift, self.stop_hours * scale + shift, _WINDOW_TOLERANCE_HOURS
        inside = (clock > lower + tolerance) if self.open_start else (clock >= lower - tolerance)
        inside &= (clock <= upper + tolerance) & (clock > self.not_before_hours + tolerance)
        if self.trim_end_hours:
            inside &= clock <= (self._required_end(end_hours) - self.trim_end_hours) + tolerance
        return inside


# The EDA's schedule and trajectory views: the day-scale main view from hour 1, then the hour-scale start-up view.
STARTUP_WINDOWS: Final = (
    TimeWindow("Physical time from 1 h to final support", 1.0, math.inf, "d", 1.0),
    TimeWindow("Startup: 0-1 h", 0.0, 1.0, "h", 0.34),
)
# The evaluation's whole run in normalized time τ = t / t_stop, t_stop the end each case passes as ``end_hours``.
TAU_WINDOW: Final = TimeWindow("whole run, τ = t / t_stop", 0.0, 1.0, "1", 1.0)


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
    label = f"τ = t / t_stop [{field_labels.display_unit('1')}]" if unit == "1" else f"Time{' before end' if window.before_end else ''} [{unit}]"
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
