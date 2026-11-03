"""Protect stored-time navigation and terminal-time display semantics.

Responsibilities:
  - Check stored-position moves, typed-time snapping, exact unions, and terminal holds

Design principles:
  - Canonical physical times are neither interpolated nor aligned by array index

This module does NOT:
  - Construct transient schedules or change stored snapshots
"""

from __future__ import annotations

import ipywidgets as widgets
import pytest

from src.analysis.ui import analysis_ui_time as time_ui


def test_position_controls_use_stored_snapshots_and_typed_time_snaps_nearest() -> None:
    """Move by one or ten stored positions and snap typed hours without interpolation."""
    events: list[time_ui.TimeStepSelection] = []
    times = (0.0, 0.25, 1.0, 1.5, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0)
    navigator = time_ui.TimeStepNavigator(times, callback=events.append)

    assert navigator.selection == time_ui.TimeStepSelection(0, 1, 0.0)
    navigator.forward_one.click()
    assert navigator.selection.physical_time == 0.25
    navigator.forward_ten.click()
    assert navigator.selection.position == 11
    navigator.backward_one.click()
    assert navigator.selection.physical_time == 55.0
    navigator.backward_ten.click()
    assert navigator.selection.position == 0
    navigator.time.value = 5.1
    assert navigator.selection == time_ui.TimeStepSelection(5, 6, 5.0)
    assert navigator.time.value == 5.0
    assert events[-1].physical_time == 5.0
    assert not any(isinstance(control, widgets.Play) for control in navigator.widget.children)


@pytest.mark.parametrize(
    "physical_times",
    [(), (0.0, 0.0), (1.0, 0.0), (0.0, float("nan"))],
)
def test_time_grids_reject_empty_nonfinite_or_nonincreasing_coordinates(
    physical_times: tuple[float, ...],
) -> None:
    """Fail closed rather than infer or reorder invalid canonical time grids."""
    with pytest.raises(ValueError, match="strictly increasing"):
        time_ui.TimeStepNavigator(physical_times)


def test_rebind_preserves_nearest_stored_time_and_notifies_once() -> None:
    """Rebind an irregular grid around authoritative physical time."""
    events: list[time_ui.TimeStepSelection] = []
    navigator = time_ui.TimeStepNavigator(
        (0.0, 0.3, 2.0, 9.0),
        callback=events.append,
        initial_time=2.0,
    )
    selected = navigator.rebind((0.0, 1.0, 4.0, 10.0))
    assert selected.physical_time == 1.0
    assert events == [selected]
    selected = navigator.rebind((0.0, 1.0, 3.0), preserve_time=2.0, notify=False)
    assert selected.physical_time == 1.0
    assert events == [events[0]]


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
