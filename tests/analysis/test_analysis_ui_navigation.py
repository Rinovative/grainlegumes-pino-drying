"""Protect the case, time, and count navigators of the notebook control standard.

Responsibilities:
  - Check nearest-case replacement of typed absent IDs, step clamping, edge disabling, and labels on the status
  - Check time snapping, ten-position steps, the status time, rebinding, and callbacks
  - Check the count control's edge rules and that every navigator sits on the column grid

Design principles:
  - One case per contract branch, without a browser

This module does NOT:
  - Check interpolation or physical-time axes, which no navigator performs
"""

from __future__ import annotations

from src.analysis.ui import analysis_ui_controls as controls
from src.analysis.ui import analysis_ui_navigation as navigation
from src.analysis.ui import analysis_ui_style as style


def on_grid(width: int) -> bool:
    """Return whether ``width`` spans whole grid columns."""
    return (width + style.GAP_PX) % (style.TRACK_PX + style.GAP_PX) == 0


def _buttons(navigator: navigation.CaseNavigator | navigation.TimeNavigator) -> list:
    return list(navigator.widget.children[1:])


def test_typed_absent_case_selects_the_nearest_with_a_warning() -> None:
    status = controls.StatusLine()
    seen: list[int] = []
    navigator = navigation.CaseNavigator((3, 7, 12), status=status, on_change=seen.append)
    navigator.field.value = 8
    assert navigator.value == 7
    assert status.level == "warning"
    assert "8" in status.text
    navigator.field.value = 99
    assert navigator.value == 12
    navigator.field.value = 5
    assert navigator.value == 3
    assert seen == [7, 12, 3]
    navigator.field.value = 7
    assert seen == [7, 12, 3, 7]
    assert status.level == "normal"


def test_case_arrows_step_one_position_and_disable_at_the_ends() -> None:
    navigator = navigation.CaseNavigator((3, 7, 12))
    back, forward = _buttons(navigator)
    assert back.disabled
    assert not forward.disabled
    forward.click()
    forward.click()
    assert navigator.value == 12
    assert forward.disabled
    navigator.step(-5)
    assert navigator.position == 0
    navigator.disabled = True
    assert forward.disabled
    assert navigator.field.disabled


def test_case_label_goes_to_the_status_line() -> None:
    status = controls.StatusLine()
    navigator = navigation.CaseNavigator((3, 7), labels={3: "lentil · case 3", 7: "chickpea · case 7"}, status=status)
    assert status.text == "lentil · case 3"
    navigator.step(1)
    assert status.text == "chickpea · case 7"


def test_case_rebind_keeps_a_valid_case_and_replaces_a_removed_one() -> None:
    status = controls.StatusLine()
    seen: list[int] = []
    navigator = navigation.CaseNavigator((3, 7, 12), value=7, status=status, on_change=seen.append)
    assert navigator.rebind((1, 7, 40), notify=False) == 7
    assert seen == []
    navigator.step(1)
    assert navigator.rebind((1, 30, 50, 100)) == 30
    assert status.level == "warning"
    assert seen == [40, 30]
    assert navigator.rebind((5, 6), preserve=False, notify=False) == 5
    assert on_grid(controls.element_width(navigator.widget))


def test_typed_time_snaps_to_the_nearest_stored_time_with_a_warning() -> None:
    status = controls.StatusLine()
    navigator = navigation.TimeNavigator((0.0, 1.0, 3.0, 6.0), status=status)
    navigator.field.value = 2.7
    assert navigator.value == 3.0
    assert status.level == "warning"
    assert "3 h" in status.text
    navigator.field.value = 6.0
    assert status.level == "normal"
    assert "6 h" in status.text


def test_time_steps_move_ten_positions_and_clamp_at_the_ends() -> None:
    times = [float(index) for index in range(50)]
    navigator = navigation.TimeNavigator(times, value=3.0)
    back_ten, back_one, forward_one, forward_ten = _buttons(navigator)
    forward_ten.click()
    assert navigator.position == 13
    back_one.click()
    assert navigator.position == 12
    navigator.step(100)
    assert navigator.position == 49
    assert forward_one.disabled
    assert forward_ten.disabled
    back_ten.click()
    assert navigator.position == 39
    assert not back_ten.disabled


def test_time_rebind_preserves_the_nearest_time_and_notifies_once() -> None:
    seen: list[navigation.TimeStepSelection] = []
    navigator = navigation.TimeNavigator((0.0, 2.0, 4.0), value=4.0)
    navigator.set_callback(seen.append)
    selection = navigator.rebind((0.0, 1.0, 3.5, 8.0))
    assert selection == navigation.TimeStepSelection(position=2, display_index=3, physical_time=3.5)
    assert seen == [selection]
    assert on_grid(controls.element_width(navigator.widget))


def test_count_control_buttons_disable_at_the_bounds() -> None:
    seen: list[int] = []
    count = navigation.CountControl(value=5, minimum=1, maximum=10, step=5, on_change=seen.append)
    count.increment.click()
    assert count.value == 10
    assert count.increment.disabled
    count.decrement.click()
    count.decrement.click()
    assert count.value == 1
    assert count.decrement.disabled
    assert seen == [10, 5, 1]
    assert on_grid(controls.element_width(count.widget))
