"""Protect stored-time coordinate semantics: validation, exact unions and master-time resolution of the ui time module.

Responsibilities:
  - Check grid validation, exact unions, and latest-snapshot terminal holds

Design principles:
  - Canonical physical times are neither interpolated nor aligned by array index

This module does NOT:
  - Construct transient schedules or change stored snapshots
  - Check time navigation, which test_analysis_ui_navigation.py owns
"""

from __future__ import annotations

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
