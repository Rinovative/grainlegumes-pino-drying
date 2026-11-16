"""Protect view panels, popovers, rendering lifecycle, notebook widths, and lazy panels of the control standard.

Responsibilities:
  - Check row validation, first-label alignment, grid membership, reserved height, and popover placement (F1, F2)
  - Check that hidden views stay idle, batches redraw once, and only the newest render publishes (rules 3, 4, 12)
  - Check re-entrant requests from a plot, an observer, or a nested view's host, and a failing display, without a lock
  - Check on-demand drawing with «Berechnen» and its cache, empty-selection clearing, and failures in the plot's place
  - Check notebook-wide tab and selector widths, view sections, and the lazy panel shell with persistent control rows
  - Check the case-count viewer built on view panels, with its dataset selector and channel control
  - Check that tables show as HTML and that a failure display escapes markup and keeps its traceback collapsed

Design principles:
  - One case per contract branch, without a browser; the items plot slots show are recorded

This module does NOT:
  - Check rendered pixel geometry or CSS reach in a frontend
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis import presentation
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.ui import analysis_ui_controls as controls
from src.analysis.ui import analysis_ui_navigation as navigation
from src.analysis.ui import analysis_ui_notebook as notebook_ui
from src.analysis.ui import analysis_ui_panel as panel
from src.analysis.ui import analysis_ui_selectors as selectors
from src.analysis.ui import analysis_ui_style as style
from src.analysis.ui import analysis_ui_viewers as viewers

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence


@pytest.fixture
def shown(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every item a plot slot shows, in order, while the slot still builds its elements."""
    captured: list[Any] = []
    show = panel.PlotSlot.show

    def record(slot: panel.PlotSlot, items: Sequence[Any]) -> None:
        captured.extend(items)
        show(slot, items)

    monkeypatch.setattr(panel.PlotSlot, "show", record)
    return captured


def _children(widget: widgets.Widget) -> tuple[widgets.Widget, ...]:
    """Return the children of a box found while walking a generic widget tree."""
    return cast("widgets.Box", widget).children


def on_grid(width: int) -> bool:
    """Return whether ``width`` spans whole grid columns."""
    return (width + style.GAP_PX) % (style.TRACK_PX + style.GAP_PX) == 0


def _three_row_panel() -> tuple[panel.ViewPanel, selectors.ChoiceBlock, selectors.ChoiceBlock]:
    runs = selectors.ChoiceBlock("Läufe", [("FNO", [("fno_a", "a")])], multiple=True, selected=["a"])
    other = selectors.ChoiceBlock("Datensätze", [("ID", [("lentil, ID", "l")])], multiple=True, selected=["l"])
    rows = [
        panel.row(controls.dropdown(["x", "y"], description="Model")),
        panel.row(runs.header, other.header),
        panel.row(navigation.CaseNavigator((3, 7)).widget, controls.checkbox("Normalise")),
    ]
    view = panel.ViewPanel(rows, slot=panel.PlotSlot(), popovers=[(runs.header, runs.body), (other.header, other.body)])
    return view, runs, other


def test_panel_refuses_non_rows_and_stacked_radio_groups() -> None:
    with pytest.raises(TypeError, match=r"panel\.row"):
        panel.ViewPanel([widgets.HBox((controls.toggle("x"),))])
    with pytest.raises(TypeError, match="radio"):
        panel.ViewPanel([panel.row(widgets.RadioButtons(options=["a", "b"]))])


def test_first_labels_align_and_every_element_sits_on_the_grid() -> None:
    view, _runs, _other = _three_row_panel()
    firsts = [_children(row)[0] for row in view.rows]
    labels = {controls.element_label(first) for first in firsts} - {None}
    assert len(labels) == 1
    widths = [controls.element_width(cast("widgets.DOMWidget", element)) for row in view.rows for element in _children(row)]
    assert all(on_grid(width) for width in widths)


def test_panel_reserves_its_rows_and_hidden_rows_keep_their_space() -> None:
    view, _runs, _other = _three_row_panel()
    block = view.content.children[0]
    assert block.layout.height == f"{3 * style.ROW_PITCH_PX}px"
    controls.set_visible(cast("widgets.DOMWidget", view.rows[2]), False)
    assert block.layout.height == f"{3 * style.ROW_PITCH_PX}px"
    assert view.slot is not None
    assert view.content.children[1] is view.status.widget
    assert view.content.children[3] is view.slot.widget


def test_popover_body_overlays_the_content_in_one_grid_cell_below_its_own_row() -> None:
    view, runs, other = _three_row_panel()
    assert isinstance(view.widget, widgets.GridBox)
    assert view.widget.children == (view.content, runs.body, other.body)
    assert all(runs.body not in _children(row) for row in view.rows)
    assert {child.layout.grid_area for child in view.widget.children} == {"1 / 1"}
    assert runs.body.layout.margin == f"{2 * style.ROW_PITCH_PX}px 0 0 0"
    assert "glp-popover" in runs.body._dom_classes
    assert view.content.children[0].layout.height == f"{3 * style.ROW_PITCH_PX}px"  # the rows, so the plot below never moves
    runs.header.value = True
    other.header.value = True
    assert runs.header.value is False
    assert runs.body.layout.display == "none"


def test_panel_without_popovers_is_not_wrapped() -> None:
    plain = panel.ViewPanel([panel.row(controls.toggle("x"))])
    assert plain.widget is plain.content
    assert style.ROOT_CLASS in plain.widget._dom_classes
    view, _runs, _other = _three_row_panel()
    assert view.widget is not view.content
    assert {"glp-panel", style.ROOT_CLASS} <= set(view.widget._dom_classes)


def test_popover_overlay_validates_before_changing_anything() -> None:
    first_header, first_body = panel.disclosure("a", widgets.VBox())
    bad_header, bad_body = panel.disclosure("b", widgets.VBox())
    rows = [panel.row(first_header)]
    notifiers = len(first_header._trait_notifiers.get("value", {}).get("change", []))
    with pytest.raises(ValueError, match="rows"):
        panel.popover_overlay(widgets.VBox(), rows, [(first_header, first_body), (bad_header, bad_body)])
    assert first_body.layout.grid_area is None
    assert len(first_header._trait_notifiers.get("value", {}).get("change", [])) == notifiers
    second_header, second_body = panel.disclosure("c", widgets.VBox(), opened=True)
    first_header.value = True
    with pytest.raises(ValueError, match="At most one"):
        panel.popover_overlay(widgets.VBox(), [panel.row(first_header, second_header)], [(first_header, first_body), (second_header, second_body)])
    assert first_body.layout.grid_area is None


def test_popover_header_must_stand_in_a_row() -> None:
    header, body = panel.disclosure("x", widgets.VBox())
    with pytest.raises(ValueError, match="rows"):
        panel.ViewPanel([panel.row(controls.toggle("y"))], popovers=[(header, body)])


def test_disclosure_shows_its_mark_and_body() -> None:
    header, body = panel.disclosure("Details", widgets.VBox())
    assert header.description == "▸ Details"
    assert body.layout.display == "none"
    header.value = True
    assert header.description == "▾ Details"
    assert body.layout.display is None


def test_inactive_panel_renders_nothing_and_a_batch_renders_once() -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    calls: list[str] = []
    view.deactivate()
    view.request(lambda: calls.append("first"))
    view.request(lambda: calls.append("latest"))
    assert calls == []
    view.activate()
    assert calls == ["latest"]
    with view.batch():
        view.request(lambda: calls.append("a"))
        view.request(lambda: calls.append("b"))
    assert calls == ["latest", "b"]


def test_render_view_failure_takes_the_plot_place_and_clears_export(shown: list[Any]) -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    state: dict[str, Any] = {}
    figure = plt.figure()
    viewers.render_view(view, lambda: figure, export_state=state, export_plot_name="p", export_title="t")
    assert state["fig"] is figure
    assert shown == [figure]

    def failing() -> None:
        message = "bad"
        raise ValueError(message)

    viewers.render_view(view, failing, export_state=state, export_plot_name="p", export_title="t")
    assert view.slot is not None
    assert view.failure.shown
    assert view.slot.widget.layout.display == "none"
    assert state["fig"] is None
    viewers.render_view(view, lambda: "table", export_state=state)
    assert not view.failure.shown
    assert view.slot.widget.layout.display is None


@pytest.mark.parametrize("inner", ["render", "clear", "fail"])
def test_a_request_from_inside_the_plot_wins_and_closes_the_outer_figure(inner: str, shown: list[Any]) -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    state: dict[str, Any] = {}
    outer = plt.figure()
    newer = plt.figure()
    requests = {
        "render": lambda: viewers.render_view(view, lambda: newer, export_state=state, export_plot_name="newer"),
        "clear": lambda: viewers.clear_view(view, "empty", export_state=state, export_plot_name="newer"),
        "fail": lambda: viewers.fail_view(view, RuntimeError("newer failed"), export_state=state, export_plot_name="newer"),
    }

    def plot() -> Any:
        requests[inner]()
        return outer

    assert viewers.render_view(view, plot, export_state=state, export_plot_name="outer") is False
    assert not plt.fignum_exists(outer.number)
    assert outer not in shown
    assert state["plot_name"] == "newer"
    assert state["fig"] is (newer if inner == "render" else None)
    assert view.failure.shown is (inner == "fail")


@pytest.mark.parametrize("end", ["supersede", "deactivate"])
def test_a_nested_view_publishes_nothing_once_its_host_render_is_over(end: str, shown: list[Any]) -> None:
    host = panel.ViewPanel([], slot=panel.PlotSlot())
    nested = panel.ViewPanel([], slot=panel.PlotSlot(), host=host)
    kept = plt.figure()
    state: dict[str, Any] = {"fig": kept, "figures": (kept,), "plot_name": "kept", "title": "Kept"}
    figure = plt.figure()

    def plot() -> Any:
        getattr(host, end)()
        return figure

    assert viewers.render_view(nested, plot, export_state=state, export_plot_name="nested", export_title="Nested") is False
    assert state == {"fig": kept, "figures": (kept,), "plot_name": "kept", "title": "Kept"}
    assert shown == []
    assert not plt.fignum_exists(figure.number)


def test_a_failing_display_shows_the_failure_and_releases_the_export(monkeypatch: pytest.MonkeyPatch) -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    figure = plt.figure()
    note = widgets.HTML("note")

    element = panel.item_widget

    def display(item: Any) -> widgets.Widget:
        if item is note:
            message = "cannot display the note"
            raise RuntimeError(message)
        return element(item)

    monkeypatch.setattr(panel, "item_widget", display)
    state: dict[str, Any] = {}
    assert viewers.render_view(view, lambda: (figure, note), export_state=state, export_plot_name="p", export_title="t") is True
    assert view.failure.shown
    assert "cannot display the note" in view.failure.widget.value
    assert state["fig"] is None
    assert (state["plot_name"], state["title"]) == ("p", "t")


def test_a_render_an_observer_starts_inside_a_publication_wins_without_deadlock(shown: list[Any]) -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    state: dict[str, Any] = {}
    older = plt.figure()
    newer = plt.figure()

    def failing() -> None:
        message = "bad"
        raise ValueError(message)

    viewers.render_view(view, failing, export_state=state)
    started: list[bool] = []

    def failure_cleared(_change: object) -> None:
        if not started:
            started.append(True)
            viewers.render_view(view, lambda: newer, export_state=state)

    view.failure.widget.observe(failure_cleared, names="value")
    # A deadlock fails the test after the timeout instead of hanging the session.
    publication = threading.Thread(target=viewers.render_view, args=(view, lambda: older), kwargs={"export_state": state}, daemon=True)
    publication.start()
    publication.join(timeout=5)
    assert not publication.is_alive()
    assert started == [True]
    assert state["fig"] is newer
    assert shown[-1] is newer
    assert not view.failure.shown


def test_clear_view_explains_the_empty_selection_and_releases_export() -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    figure = plt.figure()
    state: dict[str, Any] = {"fig": figure, "figures": (figure,)}
    viewers.clear_view(view, "Select at least one dataset.", export_state=state, export_plot_name="p")
    assert state["fig"] is None
    assert state["plot_name"] == "p"


def test_view_cache_drops_the_least_recently_used_entry() -> None:
    cache = panel.ViewCache()
    for key in range(8):
        cache.put(key, key)
    cache.get(0)
    cache.put(8, 8)
    assert len(cache) == 8
    assert 0 in cache
    assert 1 not in cache


@pytest.mark.usefixtures("shown")
def test_on_demand_view_draws_cheap_at_once_and_expensive_on_compute() -> None:
    draws: list[str] = []

    def draw(key: Any) -> str:
        draws.append(key)
        return f"plot {key}"

    cheap_panel = panel.ViewPanel([], slot=panel.PlotSlot())
    cheap = panel.OnDemandView(cheap_panel, draw, lambda: "a", expensive=False)
    cheap.refresh()
    assert draws == ["a"]
    assert cheap.button.layout.display == "none"

    key = {"value": "x"}
    button = panel.compute_button()
    expensive_panel = panel.ViewPanel([panel.row(button)], slot=panel.PlotSlot())
    view = panel.OnDemandView(expensive_panel, draw, lambda: key["value"], expensive=True, button=button)
    view.refresh()
    assert draws == ["a"]
    button.click()
    assert draws == ["a", "x"]
    key["value"] = "y"
    view.refresh()
    key["value"] = "x"
    view.refresh()
    assert draws == ["a", "x"]
    assert panel.CACHE_NOTE in expensive_panel.status.text
    with pytest.raises(ValueError, match="Berechnen"):
        panel.OnDemandView(expensive_panel, draw, lambda: "z", expensive=True)


def test_tab_width_covers_the_longest_title() -> None:
    title = "6. Spectral & Representation Analysis"
    assert panel.tab_width_ch([title, "1. Overview"]) == 36
    assert panel.tab_width_ch([title]) * 1.0 >= (style.text_ch(title) + style.TAB_PADDING_CH) * style.TAB_WIDTH_FACTOR
    tab = panel.tabs(["a", "b"], [widgets.VBox(), widgets.VBox()], width_ch=12)
    panel.set_tab_width(tab, 20)
    assert [name for name in tab._dom_classes if name.startswith("glp-tabw-")] == ["glp-tabw-20"]


def test_width_scope_shares_the_maximum_and_follows_replacement() -> None:
    widths = notebook_ui.WidthScope()
    short = panel.tabs(["A"], [widgets.VBox()], width_ch=1)
    long = panel.tabs(["6. Spectral & Representation Analysis"], [widgets.VBox()], width_ch=1)
    widths.register_tabs(short, ["A"])
    widths.register_tabs(long, ["6. Spectral & Representation Analysis"])
    assert "glp-tabw-36" in short._dom_classes
    widths.register_tabs(long, ["B"])
    assert widths.tab_width_ch == panel.tab_width_ch(["A", "B"])
    first, second = controls.dropdown(["x"]), controls.dropdown(["y"])
    widths.register_selectors([first], ["a very long view title in the first panel"])
    widths.register_selectors([second], ["short"])
    assert first.layout.width == second.layout.width
    third = controls.dropdown(["z"])
    widths.replace([first], [third], ["tiny"])
    assert second.layout.width == third.layout.width == f"{controls.dropdown_width('', ['short', 'tiny'])}px"


@pytest.mark.usefixtures("shown")
def test_view_section_renders_its_first_entry_only_when_activated() -> None:
    calls: list[str] = []
    state: dict[str, Any] = {}

    def entry(name: str) -> Any:
        calls.append(name)
        return plt.figure()

    section = notebook_ui.view_section([("One", lambda: entry("one"), "one"), ("Two", lambda: entry("two"), "two")], export_state=state)
    assert calls == []
    section.activate()
    assert calls == ["one"]
    assert state["plot_name"] == "one"
    first = state["fig"]
    section.selector.value = 1
    assert calls == ["one", "two"]
    section.deactivate()
    state["fig"] = None
    section.activate()
    assert state["fig"] is not first
    assert state["plot_name"] == "two"


@pytest.mark.usefixtures("shown")
def test_hidden_section_defers_a_selector_change_until_it_is_shown_again() -> None:
    calls: list[str] = []
    state: dict[str, Any] = {}
    entries = [(name, lambda name=name: calls.append(name), name) for name in ("one", "two", "three")]
    section = notebook_ui.view_section(entries, export_state=state)
    section.activate()
    section.deactivate()
    state["plot_name"] = "visible elsewhere"
    section.selector.value = 1
    section.selector.value = 2
    assert calls == ["one"]
    assert state["plot_name"] == "visible elsewhere"
    section.activate()
    assert calls == ["one", "three"]
    assert state["plot_name"] == "three"


@pytest.mark.usefixtures("shown")
def test_replaced_section_ignores_its_old_selector() -> None:
    calls: list[str] = []
    state: dict[str, Any] = {}
    old = notebook_ui.view_section([("A", lambda: calls.append("a"), "a"), ("B", lambda: calls.append("b"), "b")], export_state=state)
    lazy = notebook_ui.LazyPanel([old], ["1. Old"], open_label="Panel - Open", export_state=state)
    lazy.open()
    new = notebook_ui.view_section([("C", lambda: calls.append("c"), "c")], export_state=state)
    lazy.replace_tabs([new], ["2. New"])
    old.selector.value = 1
    assert old._stale is False
    assert calls == ["a", "c"]
    assert state["plot_name"] == "c"


@pytest.mark.usefixtures("shown")
def test_a_detached_section_cannot_be_adopted_again() -> None:
    state: dict[str, Any] = {}
    old = notebook_ui.view_section([("A", plt.figure, "a")], export_state=state)
    lazy = notebook_ui.LazyPanel([old], ["1. Old"], open_label="Panel - Open", export_state=state)
    lazy.replace_tabs([notebook_ui.view_section([("B", plt.figure, "b")], export_state=state)], ["2. New"])
    with pytest.raises(ValueError, match="detached"):
        lazy.replace_tabs([old], ["1. Old"])
    with pytest.raises(ValueError, match="detached"):
        notebook_ui.LazyPanel([old], ["1. Old"], open_label="Other - Open", export_state=state)


@pytest.mark.usefixtures("shown")
def test_view_section_shows_a_view_panel_result_and_a_failure_in_place() -> None:
    returned = panel.ViewPanel([], slot=panel.PlotSlot())

    def failing() -> None:
        message = "broken"
        raise RuntimeError(message)

    section = notebook_ui.view_section([("Panel", lambda: returned, "panel"), ("Broken", failing, "broken")], export_state=None)
    section.activate()
    assert section.status is returned.status
    section.selector.value = 1
    assert section.status is not returned.status
    assert not returned.active


def test_no_analysis_module_or_notebook_builds_an_output_widget() -> None:
    """VS Code's Jupyter extension stalls a kernel on overlapping Output-capture updates, so no view may build one."""
    root = Path(__file__).resolve().parents[2]
    sources = [*root.joinpath("src", "analysis").rglob("*.py"), *root.joinpath("notebooks").glob("*.ipynb")]
    assert [path.name for path in sources if re.search(r"\bOutput\(", path.read_text(encoding="utf-8"))] == []


def test_plot_slot_closes_the_elements_it_built_once_replaced() -> None:
    slot = panel.PlotSlot()
    figure = plt.figure()
    kept = widgets.HTML("owned by the caller")
    slot.show((figure, "text", kept))
    built = slot.widget.children[:2]
    slot.show(("next",))
    plt.close(figure)
    assert all(part.comm is None for element in built for part in (element, element.layout))
    assert kept.comm is not None
    assert slot.widget.children[0].comm is not None


def test_view_section_stacks_a_list_result_and_exports_its_figures(shown: list[Any]) -> None:
    table = pd.DataFrame({"case": [1, 2]})
    figure = plt.figure()
    state: dict[str, Any] = {}
    section = notebook_ui.view_section([("Both", lambda: [table, None, figure], "both")], export_state=state)
    section.activate()
    assert shown == [table, figure]
    assert state["fig"] is figure
    assert state["figures"] == (figure,)
    assert not plt.fignum_exists(figure.number)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("path", ["render_view", "show_result", "view_section"])
def test_a_list_and_a_tuple_result_show_and_export_the_same_items_on_every_path(container: type, path: str, shown: list[Any]) -> None:
    first, second = plt.figure(), plt.figure()
    result = container([widgets.HTML("a"), None, first, second])
    state: dict[str, Any] = {}
    if path == "view_section":
        section = notebook_ui.view_section([("S", lambda: result, "s")], export_state=state)
        section.activate()
    else:
        view = panel.ViewPanel([], slot=panel.PlotSlot())
        export = {"export_state": state, "export_plot_name": "s", "export_title": "S"}
        if path == "render_view":
            assert viewers.render_view(view, lambda: result, **export)
        else:
            assert viewers.show_result(view, result, **export)
    assert len(shown) == 3
    assert shown[1:] == [first, second]
    assert state["fig"] is first
    assert state["figures"] == (first, second)
    assert not plt.fignum_exists(first.number)
    assert not plt.fignum_exists(second.number)


def test_a_single_result_and_none_are_one_item_and_nothing(shown: list[Any]) -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    state: dict[str, Any] = {}
    viewers.show_result(view, "text", export_state=state)
    assert shown == ["text"]
    assert state["figures"] == ()
    viewers.show_result(view, None, export_state=state)
    assert shown == ["text"]
    viewers.show_result(view, ["a", "b"])
    assert shown == ["text", "a", "b"]


def test_show_result_on_a_hidden_view_publishes_nothing_and_closes_the_figure(shown: list[Any]) -> None:
    # show_result supersedes and publishes in one synchronous call, so a hidden view is the only way it stops being current.
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    view.deactivate()
    figure = plt.figure()
    assert not viewers.show_result(view, figure)
    assert shown == []
    assert not plt.fignum_exists(figure.number)


def test_lazy_panel_keeps_control_rows_and_exports_to_the_view_status(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, Any] = {}
    block = selectors.ChoiceBlock("Datensätze", [("ID", [("lentil, ID", "l")])], multiple=True, selected=["l"])
    controls_row = panel.row(block.header)
    sections = [notebook_ui.view_section([("One", plt.figure, "one")], export_state=state)]
    lazy = notebook_ui.LazyPanel(
        sections, ["1. One"], open_label="Panel - Open", export_state=state, controls=[controls_row], popovers=[(block.header, block.body)]
    )
    assert lazy.widget.children[0].children[1] is lazy.open_button
    lazy.open_button.click()
    assert lazy.expanded
    overlay = lazy.widget.children[0].children[1]
    content = overlay.children[0]
    assert overlay.children[1] is block.body
    assert content.children[-1] is lazy.tabs
    assert block.body.layout.margin == f"{2 * style.ROW_PITCH_PX}px 0 0 0"
    exported: list[Any] = []
    monkeypatch.setattr(notebook_ui, "write_export", lambda status, **_kwargs: exported.append(status))
    lazy.export_button.click()
    assert exported == [lazy.status]
    replacement = [notebook_ui.view_section([("Two", plt.figure, "two")], export_state=state)]
    lazy.replace_tabs(replacement, ["2. Two"])
    assert controls_row in content.children[0].children
    assert lazy.tabs.children == tuple(replacement)
    assert state["plot_name"] == "two"


def test_export_reports_on_a_status_line(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    status = controls.StatusLine()
    notebook_ui.export_pdf(status=status, export_state={})
    assert status.level == "warning"
    (tmp_path / "notebooks").mkdir()
    monkeypatch.setattr(notebook_ui.common.paths, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(notebook_ui.common.paths, "source_tree_id", lambda *_args, **_kwargs: None)

    def failing(_output: object) -> None:
        message = "encoder failed"
        raise OSError(message)

    notebook_ui.write_export(status=status, stem="x", suffix=".gif", write=failing)
    assert status.level == "error"
    assert "OSError: encoder failed" in status.text

    def writing(output: IO[bytes]) -> None:
        output.write(b"GIF")

    notebook_ui.write_export(status, stem="x", suffix=".gif", write=writing)
    assert status.text.startswith("[Export] Saved:")


def _inventory() -> presentation.dataset_selection.DatasetInventory:
    items = [presentation.dataset_selection.DatasetItem(key, key, "id", f"{key}, ID") for key in ("lentil", "chickpea")]
    return presentation.dataset_selection.DatasetInventory(items, group_order=("id",), group_labels={"id": "ID"}, poolable=True)


@pytest.mark.usefixtures("shown")
def test_case_count_view_hands_labelled_frames_to_the_plot_unchanged() -> None:
    frames = {"A · lentil, ID": pd.DataFrame({"x": range(3)})}
    labelled = dataset_cuts.DisplayFrames(frames, {"A · lentil, ID": ("lentil, ID", "A")})
    calls: list[dict[str, Any]] = []

    viewers.case_count_view(
        lambda **kwargs: calls.append(kwargs), datasets=cast("Mapping[Hashable, pd.DataFrame]", labelled), start_cases=5, step_size=1
    )

    assert isinstance(calls[-1]["datasets"], dataset_cuts.DisplayFrames)
    assert calls[-1]["datasets"].labels == labelled.labels
    selector = selectors.DatasetSelector(_inventory(), multiple=True, selected=["lentil"], pool_reason="not here")
    with pytest.raises(TypeError, match="display labels"):
        viewers.case_count_view(
            lambda **_kwargs: None,
            datasets=cast("Mapping[Hashable, pd.DataFrame]", labelled),
            start_cases=1,
            step_size=1,
            selector=selector,
        )


@pytest.mark.usefixtures("shown")
def test_case_count_view_draws_selected_frames_and_explains_an_empty_selection() -> None:
    frames: dict[Hashable, pd.DataFrame] = {"lentil": pd.DataFrame({"x": range(5)}), "chickpea": pd.DataFrame({"x": range(3)})}
    selector = selectors.DatasetSelector(_inventory(), multiple=True, selected=["lentil", "chickpea"], pool_reason="not here")
    calls: list[dict[str, Any]] = []
    view = viewers.case_count_view(lambda **kwargs: calls.append(kwargs), datasets=frames, start_cases=10, step_size=2, selector=selector)
    assert calls[-1]["max_cases"] == 3
    assert list(calls[-1]["datasets"]) == ["lentil, ID", "chickpea, ID"]
    selector.select(["chickpea"])
    assert list(calls[-1]["datasets"]) == ["chickpea, ID"]
    selector.select([])
    assert len(calls) == 2
    assert view.slot is not None
    with pytest.raises(ValueError, match="pool_reason"):
        viewers.case_count_view(
            lambda **_kwargs: None,
            datasets=frames,
            start_cases=1,
            step_size=1,
            selector=selectors.DatasetSelector(_inventory(), multiple=True, selected=["lentil"]),
        )


@pytest.mark.usefixtures("shown")
def test_case_count_view_passes_the_selected_channels_and_redraws_once_per_toggle() -> None:
    frames: dict[Hashable, pd.DataFrame] = {"lentil": pd.DataFrame({"x": range(5)})}
    channels = controls.ChannelChecks(("T", "phi"))
    calls: list[dict[str, Any]] = []
    viewers.case_count_view(lambda **kwargs: calls.append(kwargs), datasets=frames, start_cases=2, step_size=1, controls={"channels": channels})
    assert len(calls) == 1
    assert calls[-1]["channels"] == ("T", "phi")
    channels.boxes["phi"].value = False
    assert len(calls) == 2
    assert calls[-1]["channels"] == ("T",)


def test_tables_show_as_html_in_the_plot_slot() -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    frame = pd.DataFrame({"column": [1]})
    assert viewers.render_view(view, lambda: (frame, frame.style)) is True
    assert view.slot is not None
    tables = view.slot.widget.children
    assert len(tables) == 2
    assert all(isinstance(element, widgets.HTML) and "<table" in element.value for element in tables)


def test_a_failure_display_escapes_markup_and_keeps_the_traceback_collapsed() -> None:
    view = panel.ViewPanel([], slot=panel.PlotSlot())
    stale = plt.figure()
    state: dict[str, Any] = {"fig": stale, "figures": (stale,)}

    def failing() -> None:
        message = "<boom>"
        raise RuntimeError(message)

    viewers.render_view(view, failing, export_state=state, export_plot_name="bad")
    markup = view.failure.widget.value
    assert "&lt;boom&gt;" in markup
    assert "<boom>" not in markup
    assert "<details>" in markup
    assert "<pre>" in markup
    assert state["fig"] is None
    assert state["plot_name"] == "bad"


@pytest.mark.usefixtures("shown")
def test_view_section_shows_a_failing_entry_in_place_and_a_later_entry_replaces_it() -> None:
    figure = plt.figure()
    state: dict[str, Any] = {}

    def failing() -> None:
        message = "entry failed"
        raise ValueError(message)

    section = notebook_ui.view_section([("Bad", failing, "bad"), ("Good", lambda: figure, "good")], export_state=state)
    section.activate()
    failure = section._plain.failure
    assert failure.shown
    assert "entry failed" in failure.widget.value
    assert state["fig"] is None
    assert state["plot_name"] == "bad"
    section.selector.value = 1
    assert not failure.shown
    assert state["fig"] is figure
    assert state["plot_name"] == "good"


def test_view_section_releases_the_export_of_a_result_that_cannot_be_shown() -> None:
    class Unshowable:
        def _repr_html_(self) -> str:
            message = "cannot render"
            raise RuntimeError(message)

    state: dict[str, Any] = {}
    section = notebook_ui.view_section([("Bad", lambda: [plt.figure(), Unshowable()], "bad")], export_state=state)
    section.activate()
    failure = section._plain.failure
    assert failure.shown
    assert "cannot render" in failure.widget.value
    assert state["fig"] is None
    assert state["figures"] == ()
