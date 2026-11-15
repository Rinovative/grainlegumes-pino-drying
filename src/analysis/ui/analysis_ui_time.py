"""Provide validated stored-time coordinates, master-time resolution, and physical-time axis presentation.

Responsibilities:
  - Validate canonical irregular time coordinates and form their exact union
  - Resolve comparison master timelines without interpolating scientific arrays
  - Format physical-time axes without resampling authoritative coordinates

Design principles:
  - Physical time comes only from validated caller-supplied coordinates
  - Axis conversion and locators never change stored scientific values

This module does NOT:
  - Interpolate, align scientific arrays, or infer time coordinates
  - Navigate times (analysis_ui_navigation.py), render figures, load artifacts, or own panel export state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.ticker as mticker
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes

_HOURS_PER_DAY = 24.0
_LONG_RANGE_DAY_THRESHOLD_HOURS = 72.0


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


def _validated_times(values: Iterable[float]) -> tuple[float, ...]:
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
