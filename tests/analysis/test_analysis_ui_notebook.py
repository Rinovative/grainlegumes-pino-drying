"""Protect notebook render ownership, failure display, notebooks-folder export, and shared controls.

Responsibilities:
  - Check render supersession, in-place failure display, export destination and naming
  - Check the shared panel header, channel rebinding, and scope controls

Design principles:
  - Only the latest admitted selection owns displayed figures and export state
  - Exports reach only the repository's notebooks folder, resolved at export time

This module does NOT:
  - Validate scientific field values or manage persisted artifacts
"""

from __future__ import annotations

from threading import Event, Thread
from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from src.analysis.generation_inputs import generation_input_panel
from src.analysis.ui import analysis_ui_components as components
from src.analysis.ui import analysis_ui_notebook as notebook_ui
from src.analysis.ui import analysis_ui_viewers as viewers

if TYPE_CHECKING:
    from pathlib import Path

FIXED_TIMESTAMP = "20261015_123000"


class _FixedTimestamp:
    def strftime(self, _format: str) -> str:
        return FIXED_TIMESTAMP


class _FixedDateTime:
    @staticmethod
    def now(_tz: object) -> _FixedTimestamp:
        return _FixedTimestamp()


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temporary project root with a real ``notebooks`` folder and no source-tree mode."""
    (tmp_path / "notebooks").mkdir()
    monkeypatch.setattr(notebook_ui.common.paths, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(notebook_ui.common.paths, "source_tree_id", lambda *_args, **_kwargs: None)
    return tmp_path


def _open_panel(monkeypatch: pytest.MonkeyPatch, state: dict[str, Any], *sections: widgets.Widget) -> widgets.VBox:
    """Build a lazy panel, press its opener, and return the rendered panel."""
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)
    notebook_ui.make_lazy_panel_with_tabs(sections or (widgets.VBox(),), export_state=state)
    opener = displayed[-1]
    assert isinstance(opener, widgets.Button)
    opener.click()
    rendered = displayed[-1]
    assert isinstance(rendered, widgets.VBox)
    return rendered


def _export_button(panel: widgets.VBox) -> widgets.Button:
    """Return the rendered panel's PDF export trigger."""
    return next(child for child in panel.children[0].children if isinstance(child, widgets.Button) and child.description == "Export PDF")


def test_older_completed_render_cannot_publish_after_newer_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the latest callback figure enters the widget and its export state."""
    started = Event()
    release = Event()
    displayed = []
    errors = []
    output = widgets.Output()
    export_state = {}
    older = Figure()
    newer = Figure()
    monkeypatch.setattr(viewers, "display", displayed.append)

    def old_plot() -> Figure:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError
        return older

    def old_callback() -> None:
        try:
            viewers.render_figure(out=output, plot_func=old_plot, export_state=export_state, export_plot_name="old")
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    thread = Thread(target=old_callback, daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    viewers.render_figure(out=output, plot_func=lambda: newer, export_state=export_state, export_plot_name="new")
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert displayed == [newer]
    assert export_state["fig"] is newer
    assert export_state["plot_name"] == "new"


def test_failing_render_shows_failure_clears_export_and_is_replaced_by_next_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising plot function is shown in place with an escaped traceback; a later render replaces it."""
    displayed: list[Any] = []
    monkeypatch.setattr(viewers, "display", displayed.append)
    output = widgets.Output()
    stale = plt.figure()
    state: dict[str, Any] = {"fig": stale, "figures": (stale,)}

    def failing() -> Figure:
        message = "<boom>"
        raise RuntimeError(message)

    viewers.render_figure(out=output, plot_func=failing, export_state=state, export_plot_name="bad")

    html = displayed[-1].data
    assert "View failed" in html
    assert "RuntimeError: &lt;boom&gt;" in html
    assert "<details><summary>Traceback</summary><pre>" in html
    assert "<boom>" not in html
    assert state["fig"] is None
    assert state["figures"] == ()
    assert state["plot_name"] == "bad"

    good = Figure()
    viewers.render_figure(out=output, plot_func=lambda: good, export_state=state, export_plot_name="good")
    assert displayed[-1] is good
    assert state["fig"] is good


def test_failing_dropdown_entry_leaves_other_entries_renderable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising entry shows its failure while another entry of the section still renders."""
    shown: list[Any] = []
    monkeypatch.setattr(viewers, "display", shown.append)
    monkeypatch.setattr(notebook_ui, "display", shown.append)
    figure = Figure()
    state: dict[str, Any] = {}

    def failing() -> Figure:
        message = "entry failed"
        raise ValueError(message)

    section = notebook_ui.make_dropdown_section(
        [("Bad", failing, "bad"), ("Good", lambda: figure, "good")],
        export_state=state,
        select_first=True,
    )
    dropdown = section.children[0]
    dropdown.value = 0
    assert "ValueError: entry failed" in shown[-1].data
    dropdown.value = 1
    assert shown[-1] is figure
    assert state["fig"] is figure
    assert state["plot_name"] == "good"


def test_view_selector_has_one_width_in_both_modes() -> None:
    """The first-entry and prompt dropdowns share the standard selector width."""
    sections = [
        notebook_ui.make_dropdown_section([("Good", Figure, "good")], export_state={}, select_first=select_first) for select_first in (True, False)
    ]
    widths = {section.children[0].layout.width for section in sections}
    assert widths == {f"{notebook_ui.STANDARD_VIEW_SELECTOR_WIDTH_PX}px"}


def test_control_row_wraps_keeps_child_order_and_spaces_children() -> None:
    """Every control row wraps, preserves order and gives each child the shared margin."""
    first = widgets.Dropdown(options=["a"])
    second = widgets.Checkbox()
    row = components.ui_control_row(first, second)
    assert row.children == (first, second)
    assert row.layout.flex_flow == "row wrap"
    assert row.layout.width == "100%"
    assert first.layout.margin == second.layout.margin == "0 12px 6px 0"


def test_panel_header_holds_close_export_and_status_without_directory_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shell header is one row of close button, export button and status; no text field exists."""
    rendered = _open_panel(monkeypatch, {})
    header = rendered.children[0]
    close, export, status = header.children
    assert isinstance(close, widgets.Button)
    assert close.description == "Close"
    assert isinstance(export, widgets.Button)
    assert export.description == "Export PDF"
    assert isinstance(status, widgets.Output)

    def walk(widget: widgets.Widget) -> list[widgets.Widget]:
        return [widget, *(item for child in getattr(widget, "children", ()) for item in walk(child))]

    assert not any(isinstance(item, widgets.Text) for item in walk(rendered))


def test_export_lands_in_notebooks_folder_with_sanitized_stem_and_never_overwrites(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use viewer context for the stem; a second export in the same second keeps the first file."""
    monkeypatch.setattr(notebook_ui, "datetime", _FixedDateTime)
    figure = plt.figure()
    state = {
        "fig": figure,
        "figures": (),
        "plot_name": "Generic Plot",
        "title": "plot",
        "filename_stem": "Drying / ID: Case #7",
        "filename_prefix": "ignored-prefix",
    }
    button = _export_button(_open_panel(monkeypatch, state))
    button.click()
    button.click()
    assert sorted(path.name for path in (project_root / "notebooks").iterdir()) == [
        f"drying_id_case_7_{FIXED_TIMESTAMP}.pdf",
        f"drying_id_case_7_{FIXED_TIMESTAMP}_1.pdf",
    ]
    assert sorted(path.name for path in project_root.iterdir()) == ["notebooks"]
    plt.close(figure)


def test_export_collision_skips_dangling_symlink(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dangling export-name symlink is occupied and cannot redirect PDF writing."""
    monkeypatch.setattr(notebook_ui, "datetime", _FixedDateTime)
    folder = project_root / "notebooks"
    elsewhere = project_root / "elsewhere"
    elsewhere.mkdir()
    followed_target = elsewhere / "redirected.pdf"
    dangling_candidate = folder / f"collision-check_{FIXED_TIMESTAMP}.pdf"
    dangling_candidate.symlink_to(followed_target)
    figure = plt.figure()
    _export_button(_open_panel(monkeypatch, {"fig": figure, "figures": (), "plot_name": "collision-check"})).click()

    assert (folder / f"collision-check_{FIXED_TIMESTAMP}_1.pdf").is_file()
    assert dangling_candidate.readlink() == followed_target
    assert not followed_target.exists()
    plt.close(figure)


def test_export_reserves_final_filename_without_following_new_symlink(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An output entry appearing after name selection cannot redirect the actual write."""
    target = project_root / "must-not-exist.pdf"
    chosen = project_root / "notebooks" / "new-entry.pdf"

    def occupied_after_selection(_directory: Path, **_kwargs: object) -> Path:
        chosen.symlink_to(target)
        return chosen

    monkeypatch.setattr(notebook_ui, "_next_export_path", occupied_after_selection)
    figure = Figure()
    _export_button(_open_panel(monkeypatch, {"fig": figure})).click()
    assert chosen.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("case", ["source_tree", "missing", "symlink"])
def test_export_refusal_is_visible_and_writes_nothing(
    case: str, project_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A frozen source tree, a missing notebooks folder and a symlinked one are refused with a message."""
    folder = project_root / "notebooks"
    if case == "source_tree":
        monkeypatch.setattr(notebook_ui.common.paths, "source_tree_id", lambda *_args, **_kwargs: "abc123def456-0123456789ab")
    elif case == "missing":
        folder.rmdir()
    else:
        folder.rmdir()
        real = project_root / "real"
        real.mkdir()
        folder.symlink_to(real, target_is_directory=True)
    figure = Figure()
    _export_button(_open_panel(monkeypatch, {"fig": figure})).click()
    assert "[Export]" in capsys.readouterr().out
    assert not any(project_root.rglob("*.pdf"))
    assert not (folder.exists() and not folder.is_symlink() and any(folder.iterdir()))


def test_write_export_uses_suffix_and_removes_file_of_failed_writer(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared writer sanitizes the stem and serves other suffixes; a failing writer leaves no file and shows the failure."""
    shown: list[Any] = []
    monkeypatch.setattr(viewers, "display", shown.append)
    folder = project_root / "notebooks"

    def writing(output: Any) -> None:
        output.write(b"GIF89a")

    notebook_ui.write_export(widgets.Output(), stem="Sample / A", suffix=".gif", write=writing)
    (written,) = folder.iterdir()
    assert written.name.startswith("sample_a_")
    assert written.suffix == ".gif"
    assert written.read_bytes() == b"GIF89a"

    def failing(_output: object) -> None:
        message = "encoder failed"
        raise OSError(message)

    notebook_ui.write_export(widgets.Output(), stem="bad", suffix=".gif", write=failing)
    assert "OSError: encoder failed" in shown[-1].data
    assert [path.name for path in folder.iterdir()] == [written.name]


def test_write_export_open_failure_leaves_existing_file_untouched(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An open error other than an occupied name is shown and never unlinks the chosen path."""
    shown: list[Any] = []
    monkeypatch.setattr(viewers, "display", shown.append)
    foreign = project_root / "notebooks" / "foreign.pdf"
    foreign.write_bytes(b"keep")
    monkeypatch.setattr(notebook_ui, "_next_export_path", lambda *_args, **_kwargs: foreign)

    def refuse_open(_path: Path, *_args: object, **_kwargs: object) -> None:
        message = "open refused"
        raise PermissionError(message)

    monkeypatch.setattr("pathlib.Path.open", refuse_open)
    called: list[object] = []
    notebook_ui.write_export(widgets.Output(), stem="x", suffix=".pdf", write=called.append)
    assert "PermissionError: open refused" in shown[-1].data
    assert not called
    monkeypatch.undo()
    assert foreign.read_bytes() == b"keep"


def test_generation_input_export_uses_the_shared_destination_rule(project_root: Path) -> None:
    """The generation-input shell exports through the same notebooks-folder writer."""
    shell = object.__new__(generation_input_panel._GenerationInputPanelShell)
    figure = plt.figure()
    shell._status_output = widgets.Output()
    shell._export_state = {"fig": figure, "plot_name": "Input View"}
    shell._export_current()
    (written,) = (project_root / "notebooks").iterdir()
    assert written.name.startswith("input_view_")
    assert written.suffix == ".pdf"
    plt.close(figure)


def test_empty_selection_supersedes_inflight_figure_and_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cleared selection cannot be overwritten by an older slow callback."""
    output = widgets.Output()
    export_state: dict[str, object] = {"fig": None, "figures": (), "plot_name": None}
    displayed: list[object] = []
    started = Event()
    release = Event()
    monkeypatch.setattr(viewers, "display", displayed.append)

    def slow_plot() -> Figure:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError
        return Figure()

    worker = Thread(target=lambda: viewers.render_figure(out=output, plot_func=slow_plot, export_state=export_state), daemon=True)
    worker.start()
    assert started.wait(timeout=5)
    viewers.clear_rendered_figure(out=output, message="No selected channels.", export_state=export_state)
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert displayed == []
    assert export_state["fig"] is None
    assert export_state["figures"] == ()


def test_export_stem_uses_prefix_then_generic_fallback() -> None:
    """Keep legacy prefix and generic plot names backward-compatible."""
    assert notebook_ui._export_stem({"filename_prefix": "Transient – OOD", "plot_name": "generic"}) == "transient_ood_generic"  # noqa: RUF001
    assert notebook_ui._export_stem({"plot_name": "Generic / Plot"}) == "generic_plot"


def test_channel_rebind_and_scope_choices_preserve_selected_meaning() -> None:
    """Keep selected channels and aggregate/single-case scope through rebinding."""
    case, previous, following = components.ui_step_case_index(n_cases=3)
    row = components.ui_compact_case_row(case, previous, following)

    retained: dict[str, tuple[str, ...] | None] = {"value": None}
    state = components.ChannelCheckboxState(
        title="Channels",
        callback=lambda: None,
        selection_getter=lambda: retained["value"],
        selection_setter=lambda values: retained.__setitem__("value", values),
    )
    state.rebind(("T", "phi", "w_surf", "w_int"))
    group = state.group
    assert group is not None
    assert tuple(group.boxes) == ("T", "phi", "w_surf", "w_int")
    group.boxes["phi"].value = False
    state.rebind(("T", "w_surf", "future"), labels={"future": "Future [1]"})
    assert state.selected == ("T", "w_surf", "future")
    assert "phi" not in state.status.value
    group = state.group
    assert group is not None
    group.boxes["w_surf"].value = False
    state.rebind(("T",))
    assert state.selected == ("T",)
    assert "future" in state.status.value

    scope = components.ui_scope_toggle()
    _scope_row, detail = components.ui_compact_scope_controls(scope)
    assert tuple(scope.options) == (("Aggregate", "aggregate"), ("Single case", "single"))
    aggregate = widgets.IntSlider()
    components.ui_set_scope_detail(detail, (aggregate,))
    assert detail.children == (aggregate,)
    components.ui_set_scope_detail(detail, tuple(row.children))
    assert detail.children == row.children
