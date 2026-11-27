"""Protect stored-time coordinate semantics: validation, exact unions, master-time resolution and time windows of the ui time module.

Responsibilities:
  - Check grid validation, exact unions, and latest-snapshot terminal holds
  - Check the choice of time windows by run length and the placement of time numbers and label on a window's axis
  - Check that an hour axis of a day or more ticks in a step that follows its span
  - Check the drying-phase cohort: the 20-point τ grid, nearest stored states with ties to the earlier, a constant cohort
    over cases of different length, censored and unknown cases counted and not pooled, the refusal of a reached case without
    a stored hour after its origin, the t_end summary in fractional hours, the clock and note helpers and the τ axis

Design principles:
  - Canonical physical times are neither interpolated nor aligned by array index

This module does NOT:
  - Construct transient schedules or change stored snapshots
  - Check time navigation, which test_analysis_ui_navigation.py owns
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.analysis.ui import analysis_ui_time as time_ui


@pytest.mark.parametrize(
    "physical_times",
    [(), (0.0, 0.0), (1.0, 0.0), (0.0, float("nan"))],
)
def test_time_grids_reject_empty_nonfinite_or_nonincreasing_coordinates(
    physical_times: tuple[float, ...],
) -> None:
    """Fail closed rather than infer or reorder invalid canonical time grids."""
    with pytest.raises(ValueError, match="strictly increasing"):
        time_ui.ordered_time_union((physical_times,))


def test_union_never_aligns_times_by_array_index() -> None:
    """Keep sorted authoritative union coordinates."""
    grids = (
        (0.0, 0.5, 2.0, 10.0),
        (0.0, 1.0, 2.0, 12.0),
        (0.0, 2.0, 3.0, 14.0),
    )
    assert time_ui.ordered_time_union(grids) == (0.0, 0.5, 1.0, 2.0, 3.0, 10.0, 12.0, 14.0)
    with pytest.raises(ValueError, match="at least one"):
        time_ui.ordered_time_union(())


def test_master_time_resolution_uses_latest_snapshot_and_final_hold() -> None:
    """Resolve a longest-case master time without interpolation or extrapolation."""
    exact = time_ui.resolve_master_physical_time((0.0, 24.0, 72.0), 24.0)
    prior = time_ui.resolve_master_physical_time((0.0, 24.0, 72.0), 50.0)
    final = time_ui.resolve_master_physical_time((0.0, 24.0, 72.0), 96.0)
    assert exact == time_ui.ResolvedPhysicalTime(24.0, True, False)
    assert prior == time_ui.ResolvedPhysicalTime(24.0, False, False)
    assert final == time_ui.ResolvedPhysicalTime(72.0, False, True)


def test_a_short_run_gets_one_window_and_a_long_run_its_first_day_middle_and_last_day() -> None:
    """Below 72 h one hour window shows the whole run; above, hours for the first and last day around a middle in days."""
    (whole,) = time_ui.drying_windows(48.0, before_end=False)
    assert (whole.title, whole.start_hours, whole.stop_hours, whole.unit) == ("whole run, 48 h", 0.0, 48.0, "h")
    first, middle, last = time_ui.drying_windows(300.0, before_end=False)
    assert [window.unit for window in (first, middle, last)] == ["h", "d", "h"]
    assert (first.title, middle.title, last.title) == ("first 24 h", "day 1 to 11.5", "last 24 h")  # the middle ends where the last day starts
    assert (middle.start_hours, middle.stop_hours) == (24.0, 276.0)
    assert (last.start_hours, last.stop_hours, last.before_end) == (276.0, 300.0, False)
    assert first.width_ratio == last.width_ratio < middle.width_ratio  # hour columns are narrower than the day column


def test_the_last_day_before_each_end_gives_cases_of_different_duration_one_coordinate() -> None:
    """Cases that end at 120 h and 96 h meet in the last 24 h as hours before their own end; the other windows keep the clock."""
    first, middle, last = time_ui.drying_windows(120.0, before_end=True)
    assert (last.start_hours, last.stop_hours, last.before_end) == (-24.0, 0.0, True)
    long_times, short_times = np.arange(0.0, 120.1, 0.25), np.arange(0.0, 96.1, 0.25)
    for times, end in ((long_times, 120.0), (short_times, 96.0)):
        shown = times[last.select(times, end_hours=end)]
        np.testing.assert_array_equal(shown, np.arange(end - 24.0 + 0.25, end + 0.1, 0.25))  # (end - 24 h, end]
        np.testing.assert_array_equal(last.hours(shown, end_hours=end), np.arange(-24.0 + 0.25, 0.1, 0.25))
    np.testing.assert_array_equal(first.select(long_times, end_hours=120.0), long_times <= 24.0)
    np.testing.assert_array_equal(middle.select(long_times, end_hours=120.0), (long_times > 24.0) & (long_times <= 96.0))
    # The middle stops at each case's own end minus 24 h: the short case is in the middle up to 72 h and in the last window after it.
    np.testing.assert_array_equal(middle.select(short_times, end_hours=96.0), (short_times > 24.0) & (short_times <= 72.0))
    assert not (middle.select(short_times, end_hours=96.0) & last.select(short_times, end_hours=96.0)).any()


@pytest.mark.parametrize(
    ("end", "before_end"),
    [*((end, True) for end in (20.0, 24.0, 40.0, 47.0, 48.0, 96.0, 100.1, 120.0)), (96.0, False), (100.1, False), (120.0, False)],
)
def test_every_stored_hour_of_a_case_lies_in_exactly_one_window(end: float, before_end: bool) -> None:
    """
    Half-open windows [0, 24], (24, end - 24] and (max(end - 24, 24), end]: a boundary hour at 24 h or 48 h is drawn once.

    Hours a hair either side of both thresholds (24 h and end - 24 h) fall into one window too, whatever the rounding.
    """
    windows = time_ui.drying_windows(120.0 if before_end else end, before_end=before_end)
    grid = np.arange(0.0, end + 0.1, 0.25)
    hairs = np.asarray([sign * size for size in (0.5e-9, 1.2e-9, 1.5e-9, 3e-9) for sign in (-1.0, 1.0)])
    near = np.concatenate([threshold + hairs for threshold in (24.0, end - 24.0)])
    times = np.unique(np.concatenate([grid, near[(near >= 0.0) & (near <= end)]]))
    shown = np.sum([window.select(times, end_hours=end) for window in windows], axis=0)
    assert (shown == 1).all()


def test_a_case_under_48_hours_is_in_the_first_window_and_from_the_first_hour_after_24_h_in_the_last() -> None:
    """A 40 h case is in the first window to 24 h and in the last from 24.25 h (-15.75 to 0 h); a 20 h case is only in the first."""
    first, _middle, last = time_ui.drying_windows(120.0, before_end=True)
    times = np.arange(0.0, 40.1, 0.25)
    np.testing.assert_array_equal(last.hours(times[last.select(times, end_hours=40.0)], end_hours=40.0), np.arange(-15.75, 0.1, 0.25))
    np.testing.assert_array_equal(first.select(times, end_hours=40.0), times <= 24.0)
    short = np.arange(0.0, 20.1, 0.25)
    assert not last.select(short, end_hours=20.0).any()
    assert first.select(short, end_hours=20.0).all()


def test_the_phase_cohort_keeps_every_reached_case_at_every_grid_point_and_counts_the_others() -> None:
    """
    A 4 h and a 40 h reached case both contribute one stored state at each of the 20 grid points; censored and unknown cases are counted, not pooled.

    Each case takes its stored state nearest to τ = t / t_end, t_end its last stored hour; on a 10 h hourly case every grid point
    halfway between two stored τ resolves to the earlier state.
    """
    short, long_, censored, unknown = np.arange(5.0), np.arange(41.0), np.arange(9.0), np.arange(3.0)
    cohort = time_ui.phase_cohort((short, long_, censored, unknown), ("reached", "reached", "censored", "unknown"))
    np.testing.assert_allclose(time_ui.PHASE_GRID, np.arange(1, 21) * 0.05)
    assert (cohort.cases, cohort.censored, cohort.unknown) == ((0, 1), 1, 1)
    assert cohort.hours.shape == cohort.states.shape == (2, 20)
    np.testing.assert_array_equal(cohort.hours[:, -1], [4.0, 40.0])
    np.testing.assert_array_equal(cohort.hours[0, :4], [0.0, 0.0, 1.0, 1.0])  # τ 0.05, 0.10 nearest 0; 0.15, 0.20 nearest 0.25
    np.testing.assert_array_equal(cohort.take((short * 10.0, long_ * 10.0, censored, unknown))[1], cohort.hours[1] * 10.0)
    assert cohort.label() == "t_end 4-40 h, median 22 h; 2 reached, 1 censored, 1 unknown not pooled"
    assert cohort.hours_label(0.1) == "τ = 0.10: stored t 0-4 h"
    halves = time_ui.phase_cohort((np.arange(11.0),), ("reached",)).hours[0]
    np.testing.assert_array_equal(halves[0::2], np.arange(10.0))  # τ 0.05, 0.15, ... lie halfway: the earlier state
    empty = time_ui.phase_cohort((censored,), ("censored",))
    assert (empty.cases, empty.hours.shape, empty.label()) == ((), (0, 20), "no reached case to pool; 1 censored not pooled")
    assert time_ui.phase_cohort((np.arange(5.0), np.arange(6.0)), ("reached",) * 2).label() == "t_end 4-5 h, median 4.5 h; 2 reached"
    with pytest.raises(ValueError, match="position 1"):  # a reached case stored only at its origin has no phase
        time_ui.phase_cohort((short, np.zeros(1)), ("reached", "reached"))
    assert time_ui.phase_cohort((np.zeros(1),), ("censored",)).censored == 1  # a case left out is only counted
    assert (time_ui.is_phase("phase"), time_ui.is_phase("hours")) == (True, False)
    with pytest.raises(ValueError, match="time_axis"):
        time_ui.is_phase("tau")
    assert time_ui.phase_note({"A": "x", "": "y"}) == "A: x\ny"
    assert time_ui.phase_note({}) is None
    assert time_ui.phase_index(0.5) == 9
    with pytest.raises(ValueError, match="grid"):
        time_ui.phase_index(0.52)
    assert [time_ui.completion_status(available=a, reached=r) for a, r in ((True, True), (True, False), (False, True), (True, None))] == [
        "reached",
        "censored",
        "unknown",
        "unknown",
    ]
    figure, axis = plt.subplots()
    try:
        time_ui.style_window_axis(axis, time_ui.PHASE_WINDOW)
        assert axis.get_xlim() == (0.0, 1.0)
        assert axis.get_xlabel() == "τ = t / t_end [-]"
        np.testing.assert_allclose(_ticks(axis), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    finally:
        plt.close(figure)


def test_a_window_that_counts_from_each_cases_end_refuses_a_call_without_it() -> None:
    """The before-end and the trimmed window name the missing end_hours; windows on the clock need none."""
    first, middle, last = time_ui.drying_windows(120.0, before_end=True)
    times = np.arange(0.0, 120.1, 1.0)
    for window in (last, middle):
        with pytest.raises(ValueError, match="end_hours"):
            window.select(times)
    for window in (last, middle):
        with pytest.raises(ValueError, match="end_hours"):
            window.hours(times)
    np.testing.assert_array_equal(first.select(times), times <= 24.0)
    np.testing.assert_array_equal(time_ui.drying_windows(120.0, before_end=False)[2].select(times), times > 96.0)


def test_window_axes_keep_time_numbers_and_the_label_under_the_bottom_row_only() -> None:
    """A row above the bottom has neither tick numbers nor label; the unit follows the window; a before-end window says so."""
    figure, axes = plt.subplots(2, 3)
    try:
        windows = time_ui.drying_windows(120.0, before_end=True)
        for row, bottom in enumerate((False, True)):
            for axis, window in zip(axes[row], windows, strict=True):
                time_ui.style_window_axis(axis, window, bottom=bottom)
        assert [axis.get_xlabel() for axis in axes[0]] == ["", "", ""]
        assert not any(axis.xaxis.get_tick_params()["labelbottom"] for axis in axes[0])
        assert [axis.get_xlabel() for axis in axes[1]] == ["Time [h]", "Time [d]", "Time before end [h]"]
        assert all(axis.xaxis.get_tick_params()["labelbottom"] for axis in axes[1])
        assert axes[1, 2].get_xlim() == (-24.0, 0.0)
        np.testing.assert_array_equal(_ticks(axes[1, 0]), [0.0, 6.0, 12.0, 18.0, 24.0])
        assert axes[1, 1].xaxis.get_major_formatter()(48.0, 0) == "2"  # the middle window counts days on hour data
        np.testing.assert_array_equal(_ticks(axes[1, 1]), [24.0, 48.0, 72.0, 96.0])
        np.testing.assert_array_equal(_ticks(axes[1, 2]), [-24.0, -18.0, -12.0, -6.0, 0.0])
    finally:
        plt.close(figure)


def _ticks(axis: Any) -> np.ndarray:
    """Return the major ticks of one axis inside its limits."""
    lower, upper = axis.get_xlim()
    return np.asarray([tick for tick in axis.xaxis.get_major_locator()() if lower - 1e-9 <= tick <= upper + 1e-9])


@pytest.mark.parametrize(("end", "step"), [(24.0, 3.0), (48.0, 6.0), (70.0, 12.0)])
def test_an_hour_axis_of_a_day_or_more_ticks_in_a_step_that_follows_its_span(end: float, step: float) -> None:
    """A 24 h, 48 h and 70 h hour axis tick every 3, 6 and 12 h, at least four ticks each, never only every 24 h."""
    figure, axis = plt.subplots()
    try:
        time_ui.physical_time_display((0.0, end), preferred_unit="h").configure(axis)
        ticks = _ticks(axis)
        np.testing.assert_allclose(np.diff(ticks), step)
        assert len(ticks) >= 4
    finally:
        plt.close(figure)


def test_hour_windows_tick_every_six_hours_from_their_start() -> None:
    """A clock last window of 94 to 118 h ticks 94, 100, ..., 118 and a 60 h whole run 0, 6, ..., 60; a window under 12 h has nice steps."""
    figure, axes = plt.subplots(1, 3)
    try:
        last = time_ui.drying_windows(118.0, before_end=False)[2]
        (whole,) = time_ui.drying_windows(60.0, before_end=False)
        for axis, window in zip(axes, (last, whole, time_ui.STARTUP_WINDOWS[1]), strict=True):
            time_ui.style_window_axis(axis, window)
        np.testing.assert_array_equal(_ticks(axes[0]), [94.0, 100.0, 106.0, 112.0, 118.0])
        np.testing.assert_array_equal(_ticks(axes[1]), np.arange(0.0, 60.1, 6.0))
        assert len(_ticks(axes[2])) >= 3  # 0 to 1 h: the nice steps of a short window, not 6 h
    finally:
        plt.close(figure)


def test_an_open_main_window_ends_at_the_shown_data_and_a_day_window_under_one_day_reads_in_hours() -> None:
    """The EDA's main window has no fixed end: it stops at the shown support, and below one day its ticks and label are hours."""
    main, startup = time_ui.STARTUP_WINDOWS
    assert math.isinf(main.stop_hours)
    figure, axes = plt.subplots(1, 3)
    try:
        time_ui.style_window_axis(axes[0], main, shown_end_hours=8.0)
        time_ui.style_window_axis(axes[1], main, shown_end_hours=72.0)
        time_ui.style_window_axis(axes[2], startup)
        assert (axes[0].get_xlabel(), axes[0].get_xlim()) == ("Time [h]", (1.0, 8.0))
        assert (axes[1].get_xlabel(), axes[1].get_xlim()) == ("Time [d]", (1.0, 72.0))
        assert axes[2].get_xlim() == (0.0, 1.0)
        time_ui.style_window_axis(axes[0], main, shown_end_hours=1.55)  # a support ending just after hour 1 clips to its data ...
        assert axes[0].get_xlim() == (1.0, 1.55)
        assert 1.55 in _ticks(axes[0])  # ... and its end point keeps a tick
        time_ui.style_window_axis(axes[1], main)  # without any data the window spans a short default
        assert axes[1].get_xlim() == (1.0, 2.2)
    finally:
        plt.close(figure)
