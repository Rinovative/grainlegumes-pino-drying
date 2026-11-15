"""Protect the primitive controls, the column grid, and the stylesheet of the notebook control standard.

Responsibilities:
  - Check grid arithmetic, text estimates, and that every constructor sizes its element on whole grid columns
  - Check dropdown, checkbox, choice-group, and action-button sizing and styles
  - Check channel rebinding, status levels, the failure display, and the busy guard
  - Check the stylesheet rules that carry the decided layout rules

Design principles:
  - One case per contract branch, without a browser

This module does NOT:
  - Check rendered pixel geometry in a frontend
"""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pytest

from src.analysis.ui import analysis_ui_controls as controls
from src.analysis.ui import analysis_ui_style as style

PITCH = style.TRACK_PX + style.GAP_PX


def on_grid(width: int) -> bool:
    """Return whether ``width`` spans whole grid columns."""
    return width >= style.TRACK_PX and (width + style.GAP_PX) % PITCH == 0


def test_grid_span_width_offset_and_snap() -> None:
    grid = style.GRID
    assert grid.width(1) == 40
    assert grid.width(3) == 3 * 40 + 2 * 16
    assert grid.span(40) == 1
    assert grid.span(41) == 2
    assert grid.span(0) == 1
    assert grid.offset(2) == 112
    assert grid.snap(96) == grid.width(2)
    assert grid.snap(97) == grid.width(3)
    with pytest.raises(ValueError, match="positive integer"):
        grid.width(0)


def test_text_estimate_is_generous_for_upper_case_and_wide_letters() -> None:
    assert style.text_ch("ab") == 2.0
    assert style.text_ch("A") == 1.25
    assert style.text_ch("mW@") == 4.5
    assert style.text_px("abc", allowance_ch=1) == 4 * style.CH_PX


def test_dropdown_width_is_label_plus_longest_entry_rounded_to_the_grid() -> None:
    dropdown = controls.dropdown(["short", "a much longer option"], description="Model")
    longest = style.text_px("a much longer option", allowance_ch=style.DROPDOWN_ALLOWANCE_CH)
    expected = style.GRID.snap(style.text_px("Model") + style.LABEL_GAP_PX + longest)
    assert controls.element_width(dropdown) == expected
    assert dropdown.style.description_width == f"{style.text_px('Model')}px"
    wider = controls.dropdown(["short"], description="Model", width_texts=["a much longer option"])
    assert controls.element_width(wider) == expected
    with pytest.raises(ValueError, match="not one of the options"):
        controls.dropdown(["a"], value="b")


def test_every_constructor_sizes_its_element_on_the_grid() -> None:
    elements = [
        controls.dropdown(["x"]),
        controls.toggle("Show residual"),
        controls.checkbox("Normalise"),
        controls.choice_group(("Mean", "Median", "Max"), description="Statistic"),
        controls.error_mode_group("steady"),
        controls.scale_mode_group(),
        controls.kappa_scale_group(),
        controls.scope_group(),
        controls.display_stride_dropdown(),
        controls.channel_dropdown(),
        controls.scale_lock_checkbox(value=False),
        controls.action_button("Close", kind="close"),
        controls.action_button("Steady flow - Open", kind="open"),
        controls.ChannelChecks(("T", "phi")).widget,
    ]
    assert all(on_grid(controls.element_width(element)) for element in elements)


def test_checkbox_label_is_never_cut() -> None:
    text = "Normalise each field to its own reference"
    box = controls.checkbox(text)
    assert controls.element_width(box) >= style.text_px(text)
    assert box.indent is False


def test_vocabularies_keep_their_values_and_defaults() -> None:
    assert controls.error_mode_group("transient").options == ("Signed error", "Absolute error", "Normalized absolute error")
    assert controls.error_mode_group("steady").options == ("Signed error", "MAE", "Relative [%]", "Normalized absolute error")
    assert [controls.error_mode_group(task).value for task in ("steady", "transient")] == ["Signed error"] * 2  # the signed error is the default
    assert controls.scale_mode_group().options == ("Independent", "Shared (GT)")
    assert controls.kappa_scale_group().value == "log10(kappa)"
    assert controls.scope_group("single").value == "single"
    assert controls.channel_dropdown(("p", "|u|")).value == "|u|"
    with pytest.raises(ValueError, match="steady"):
        controls.error_mode_group("other")  # type: ignore[arg-type]


def test_action_buttons_share_one_width_and_their_kind_colour() -> None:
    kinds: tuple[tuple[str, controls.ActionKind], ...] = (("Close", "close"), ("Export PDF", "export"), ("Export GIF", "export"))
    buttons = {label: controls.action_button(label, kind=kind) for label, kind in kinds}
    buttons["Berechnen"] = controls.action_button(controls.COMPUTE_LABEL, kind="compute")
    assert len({controls.element_width(button) for button in buttons.values()}) == 1
    assert buttons["Close"].button_style == "danger"
    assert buttons["Export PDF"].button_style == ""
    assert buttons["Berechnen"].button_style == "primary"
    opener = controls.action_button("Generation inputs - Open", kind="open")
    assert opener.button_style == "primary"
    assert controls.element_width(opener) == style.GRID.snap(style.text_px("Generation inputs - Open", allowance_ch=style.BUTTON_ALLOWANCE_CH))


def test_channel_checks_rebind_keeps_valid_choices_and_names_removed_ones() -> None:
    checks = controls.ChannelChecks(("T", "phi", "w_gr"))
    calls: list[tuple[str, ...]] = []
    checks.observe(lambda: calls.append(checks.selected))
    checks.boxes["phi"].value = False
    assert calls == [("T", "w_gr")]
    removed = checks.rebind(("T", "phi", "w_int"))
    assert removed == ("w_gr",)
    assert checks.selected == ("T", "w_int")
    assert len(calls) == 1
    assert tuple(checks.boxes) == ("T", "phi", "w_int")
    assert checks.widget.children[1:] == tuple(checks.boxes.values())
    with pytest.raises(ValueError, match="no repeated"):
        checks.rebind(("T", "T"))


def test_channel_checks_rebind_detaches_the_removed_checkboxes() -> None:
    checks = controls.ChannelChecks(("T", "phi"))
    calls: list[tuple[str, ...]] = []
    checks.observe(lambda: calls.append(checks.selected))
    old = checks.boxes["T"]
    checks.rebind(("T", "w_gr"))
    old.value = False
    assert calls == []
    assert checks.selected == ("T", "w_gr")


def test_status_line_levels_escape_and_progress() -> None:
    status = controls.StatusLine()
    assert status.widget.value == "&nbsp;"
    status.set("<b>x</b>", "warning")
    assert status.text == "<b>x</b>"
    assert status.level == "warning"
    assert "&lt;b&gt;" in status.widget.value
    assert "glp-status-warning" in status.widget._dom_classes
    status.progress("[GIF] Frame", 3, 10)
    assert status.text == "[GIF] Frame 3/10"
    assert "<progress" in status.widget.value
    assert "glp-status-warning" not in status.widget._dom_classes
    with pytest.raises(ValueError, match="level"):
        status.set("x", "loud")  # type: ignore[arg-type]


def test_failure_display_shows_traceback_and_releases_the_export_figure() -> None:
    display = controls.FailureDisplay()
    assert not display.shown
    figure = plt.figure()
    state = {"fig": figure, "figures": (figure,), "plot_name": "kept"}
    display.show(ValueError("bad value"), state)
    assert display.shown
    assert "ValueError: bad value" in display.widget.value
    assert "<details>" in display.widget.value
    assert state["fig"] is None
    assert state["figures"] == ()
    assert state["plot_name"] == "kept"
    assert not plt.fignum_exists(figure.number)
    display.clear()
    assert not display.shown


def test_busy_guard_disables_during_the_action_and_restores_after_a_failure() -> None:
    button = widgets.Button()
    other = widgets.Button(disabled=True)
    status = controls.StatusLine()
    guard = controls.BusyGuard([button, other], status)
    seen: list[bool] = []
    errors: list[BaseException] = []

    def failing() -> str:
        seen.append(button.disabled)
        assert guard.run(lambda: "nested") is False
        message = "boom"
        raise RuntimeError(message)

    assert guard.run(failing, on_error=errors.append) is True
    assert seen == [True]
    assert button.disabled is False
    assert other.disabled is True
    assert status.level == "error"
    assert "RuntimeError: boom" in status.text
    assert isinstance(errors[0], RuntimeError)
    guard.run(lambda: "done")
    assert status.text == "done"


def test_set_visible_hides_and_restores_an_element() -> None:
    box = widgets.HBox((widgets.Button(), widgets.Checkbox()))
    controls.set_visible(box, False)
    assert box.layout.display == "none"
    controls.set_visible(box, True)
    assert box.layout.display is None


def test_stylesheet_carries_the_decided_layout_rules() -> None:
    sheet = style.stylesheet()
    assert sheet.startswith("<style>")
    assert "flex-flow: row nowrap" in sheet
    assert "overflow-x: auto" in sheet
    assert f"margin-right: {style.LABEL_GAP_PX}px" in sheet
    assert f"margin: 0 0 {style.PLOT_GAP_PX}px 0" in sheet
    assert "white-space: normal" in sheet
    assert "-webkit-appearance: none" in sheet
    assert "text-align: left; }" in sheet
    assert "position: absolute; left: 0; z-index: 20" in sheet
    assert f"max-height: {style.POPOVER_MAX_PX}px" in sheet
    assert ".glp-ui.glp-tabw-36 .lm-TabBar-tab" in sheet
    assert "glp-guide" not in sheet
    assert "@" not in sheet
    carrier = style.stylesheet_widget()
    assert carrier.layout.height == "0px"
    assert carrier.value == sheet
