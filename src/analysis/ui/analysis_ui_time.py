"""
analysis_ui_time.py

Provide reusable stored-time navigation and physical-time axis presentation.

Responsibilities:
  - Navigate canonical irregular time coordinates by stored position
  - Display and accept authoritative physical time rather than step indices
  - Resolve comparison master timelines without interpolating scientific arrays
  - Format physical-time axes without resampling authoritative coordinates

Design principles:
  - Physical time comes only from validated caller-supplied coordinates
  - Typed values snap deterministically to the nearest available stored time
  - Axis conversion and locators never change stored scientific values

This module does NOT:
  - Interpolate, align scientific arrays, or infer time coordinates
  - Render figures, load artifacts, or own panel export state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import ipywidgets as widgets
import matplotlib.ticker as mticker
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.axes import Axes

_HOURS_PER_DAY = 24.0
_LONG_RANGE_DAY_THRESHOLD_HOURS = 72.0
COMPACT_CONTROL_WIDTH = "390px"
_TIME_FIELD_WIDTH = "150px"
_TIME_BUTTON_WIDTH = "42px"


@dataclass(frozen=True, slots=True)
class TimeStepSelection:
    """Describe one selected stored snapshot and its canonical physical time."""

    position: int
    display_index: int
    physical_time: float


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
    minor_interval_hours: float | None = None

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

        if self.minor_interval_hours is None:
            target_axis.set_minor_locator(mticker.NullLocator())
        else:
            target_axis.set_minor_locator(
                mticker.MultipleLocator(
                    self.minor_interval_hours / self.divisor_hours,
                )
            )
        axis.grid(axis=dimension, which="major", alpha=0.25)
        if self.minor_interval_hours is not None:
            axis.grid(axis=dimension, which="minor", alpha=0.10)


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
    minor_interval_hours: float | None = None,
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
    for label, interval in (
        ("major_interval_hours", major_interval_hours),
        ("minor_interval_hours", minor_interval_hours),
    ):
        if interval is not None and (isinstance(interval, bool) or not np.isfinite(float(interval)) or float(interval) <= 0.0):
            message = f"{label} must be positive finite hours or None."
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
        minor_interval_hours=(None if minor_interval_hours is None else float(minor_interval_hours)),
    )


def configure_physical_time_axis(
    axis: Axes,
    physical_hours: Iterable[float],
    *,
    preferred_unit: str = "auto",
    include_zero: bool = False,
    right_margin_hours: float = 0.0,
    major_interval_hours: float | None = None,
    minor_interval_hours: float | None = None,
) -> PhysicalTimeDisplay:
    """Configure one axis and return its display-only transform."""
    display = physical_time_display(
        physical_hours,
        preferred_unit=preferred_unit,
        include_zero=include_zero,
        right_margin_hours=right_margin_hours,
        major_interval_hours=major_interval_hours,
        minor_interval_hours=minor_interval_hours,
    )
    display.configure(axis)
    return display


def _validated_times(values: Iterable[float]) -> tuple[float, ...]:
    """Return finite, strictly increasing canonical physical-time coordinates."""
    times = tuple(float(value) for value in values)
    array = np.asarray(times, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all() or np.any(np.diff(array) <= 0.0):
        message = "Time-step navigation requires finite, strictly increasing physical times."
        raise ValueError(message)
    return times


def ordered_time_intersection(
    time_coordinates: Sequence[Iterable[float]],
) -> tuple[float, ...]:
    """Return the exact ordered intersection of canonical physical-time grids."""
    grids = tuple(_validated_times(values) for values in time_coordinates)
    if not grids:
        message = "Physical-time intersection requires at least one coordinate grid."
        raise ValueError(message)
    shared = set(grids[0])
    for grid in grids[1:]:
        shared.intersection_update(grid)
    return tuple(value for value in grids[0] if value in shared)


def ordered_time_union(
    time_coordinates: Sequence[Iterable[float]],
) -> tuple[float, ...]:
    """Return the sorted union of authoritative stored physical-time grids."""
    grids = tuple(_validated_times(values) for values in time_coordinates)
    if not grids:
        message = "Physical-time union requires at least one coordinate grid."
        raise ValueError(message)
    return tuple(sorted({value for grid in grids for value in grid}))


def resolve_master_physical_time(
    available_times: Iterable[float],
    master_time: float,
) -> ResolvedPhysicalTime:
    """Resolve exact or latest-prior stored evidence without interpolation."""
    times = _validated_times(available_times)
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


class TimeStepNavigator:
    """Own one physical-time field and manual stored-position navigator."""

    def __init__(
        self,
        physical_times: Iterable[float],
        *,
        callback: Callable[[TimeStepSelection], None] | None = None,
        initial_time: float | None = None,
    ) -> None:
        """Construct controls in physical-time then double/single-arrow order."""
        self._times = _validated_times(physical_times)
        self._callback = callback
        self._updating = False
        self._last_selection: TimeStepSelection | None = None
        self._position = self._nearest_position(initial_time) if initial_time is not None else 0
        button_layout = widgets.Layout(width=_TIME_BUTTON_WIDTH)
        self.time = widgets.FloatText(
            value=self._times[self._position],
            description="Time [h]:",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width=_TIME_FIELD_WIDTH),
            tooltip=("Typed values snap to the nearest available stored physical time; scientific fields are never interpolated."),
        )
        self.backward_ten = widgets.Button(
            description="≪",
            tooltip="Ten stored times backward",
            layout=button_layout,
        )
        self.backward_one = widgets.Button(
            description="←",
            tooltip="Previous stored time",
            layout=button_layout,
        )
        self.forward_one = widgets.Button(
            description="→",
            tooltip="Next stored time",
            layout=button_layout,
        )
        self.forward_ten = widgets.Button(
            description="≫",
            tooltip="Ten stored times forward",
            layout=button_layout,
        )
        self.widget = widgets.HBox(
            (
                self.time,
                self.backward_ten,
                self.backward_one,
                self.forward_one,
                self.forward_ten,
            ),
            layout=widgets.Layout(
                align_items="center",
                width=COMPACT_CONTROL_WIDTH,
            ),
        )
        self.backward_ten.on_click(lambda _button: self.step(-10))
        self.backward_one.on_click(lambda _button: self.step(-1))
        self.forward_one.on_click(lambda _button: self.step(1))
        self.forward_ten.on_click(lambda _button: self.step(10))
        self.time.observe(self._time_changed, names="value")
        self._synchronize_state()

    @property
    def physical_times(self) -> tuple[float, ...]:
        """Return the currently bound canonical time coordinates."""
        return self._times

    @property
    def selection(self) -> TimeStepSelection:
        """Return the current zero-based position, display index, and time."""
        return TimeStepSelection(
            position=self._position,
            display_index=self._position + 1,
            physical_time=self._times[self._position],
        )

    def set_callback(
        self,
        callback: Callable[[TimeStepSelection], None] | None,
    ) -> None:
        """Replace the render callback without changing the current selection."""
        self._callback = callback

    def _nearest_position(self, physical_time: float | None) -> int:
        """Resolve the closest stored time, preferring the earlier tie."""
        if physical_time is None:
            return 0
        requested = float(physical_time)
        if not np.isfinite(requested):
            message = "A preserved physical time must be finite."
            raise ValueError(message)
        distances = np.abs(np.asarray(self._times, dtype=np.float64) - requested)
        return int(np.argmin(distances))

    def _synchronize_state(self) -> None:
        """Synchronize physical-time value and arrow bounds."""
        self._updating = True
        try:
            self.time.value = self._times[self._position]
        finally:
            self._updating = False
        at_start = self._position == 0
        at_end = self._position == len(self._times) - 1
        self.backward_ten.disabled = at_start
        self.backward_one.disabled = at_start
        self.forward_one.disabled = at_end
        self.forward_ten.disabled = at_end

    def _emit(self) -> None:
        """Notify the callback once for a changed accepted selection."""
        selected = self.selection
        if selected == self._last_selection:
            return
        self._last_selection = selected
        if self._callback is not None:
            self._callback(selected)

    def _time_changed(self, _change: dict[str, Any]) -> None:
        """Snap one typed physical time to the nearest authoritative position."""
        if self._updating:
            return
        try:
            self._position = self._nearest_position(float(self.time.value))
        except (TypeError, ValueError):
            self._synchronize_state()
            return
        self._synchronize_state()
        self._emit()

    def step(self, delta: int) -> None:
        """Move by an integer number of stored positions within current bounds."""
        if isinstance(delta, bool) or not isinstance(delta, int):
            message = "Time-step movement must be an integer position delta."
            raise TypeError(message)
        position = max(0, min(len(self._times) - 1, self._position + delta))
        if position == self._position:
            return
        self._position = position
        self._synchronize_state()
        self._emit()

    def rebind(
        self,
        physical_times: Iterable[float],
        *,
        preserve_time: float | None = None,
        notify: bool = True,
    ) -> TimeStepSelection:
        """Replace the grid, preserve a nearest valid time, and notify once."""
        previous_time = self.selection.physical_time if preserve_time is None else preserve_time
        self._times = _validated_times(physical_times)
        self._position = self._nearest_position(previous_time)
        self._synchronize_state()
        if notify:
            self._last_selection = None
            self._emit()
        else:
            self._last_selection = self.selection
        return self.selection
