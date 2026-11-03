"""Protect notebook render ownership, PDF naming, and shared selection controls.

Responsibilities:
  - Check render supersession, PDF naming, channel rebinding, and scope controls

Design principles:
  - Only the latest admitted selection owns displayed figures and export state

This module does NOT:
  - Validate scientific field values or manage persisted artifacts
"""

from __future__ import annotations

from threading import Event, Thread
from typing import TYPE_CHECKING

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.analysis.ui import analysis_ui_components as components
from src.analysis.ui import analysis_ui_notebook as notebook_ui
from src.analysis.ui import analysis_ui_viewers as viewers

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _export_button(panel: widgets.VBox) -> widgets.Button:
    """Return the rendered panel's PDF export trigger."""
    return next(child for child in panel.children[0].children if isinstance(child, widgets.Button) and child.description == "Export PDF")


def _export_directory(panel: widgets.VBox) -> widgets.Text:
    """Return the visible output-directory field below the panel header."""
    row = panel.children[1]
    return next(child for child in row.children if isinstance(child, widgets.Text))


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


def test_export_prefers_sanitized_viewer_stem_and_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use viewer context over generic names while retaining timestamp collision protection."""
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)
    figure = plt.figure()
    state = {
        "fig": figure,
        "figures": (),
        "plot_name": "Generic Plot",
        "title": "plot",
        "filename_stem": "Drying / ID: Case #7",
        "filename_prefix": "ignored-prefix",
    }
    notebook_ui.make_lazy_panel_with_tabs((widgets.VBox(),), export_state=state, export_dir=str(tmp_path))
    opener = displayed[-1]
    assert isinstance(opener, widgets.Button)
    opener.click()
    rendered = displayed[-1]
    assert isinstance(rendered, widgets.VBox)
    button = _export_button(rendered)
    button.click()
    button.click()
    names = sorted(path.name for path in tmp_path.glob("*.pdf"))
    assert len(names) == 2
    assert names[0].startswith("drying_id_case_7_")
    assert names[0] != names[1]
    plt.close(figure)


def test_export_collision_skips_dangling_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dangling export-name symlink is occupied and cannot redirect PDF writing."""

    class FixedTimestamp:
        def strftime(self, _format: str) -> str:
            return "20261015_123000"

    class FixedDateTime:
        @staticmethod
        def now(_tz: object) -> FixedTimestamp:
            return FixedTimestamp()

    monkeypatch.setattr(notebook_ui, "datetime", FixedDateTime)
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)
    allowed = tmp_path / "exports"
    allowed.mkdir()
    protected = tmp_path / "immutable-artifacts"
    protected.mkdir()
    followed_target = protected / "redirected.pdf"
    timestamp = FixedTimestamp().strftime("%Y%m%d_%H%M%S")
    stem = "collision-check"
    dangling_candidate = allowed / f"{stem}_{timestamp}.pdf"
    dangling_candidate.symlink_to(followed_target)
    figure = plt.figure()
    state = {"fig": figure, "figures": (), "plot_name": stem}

    notebook_ui.make_lazy_panel_with_tabs(
        (widgets.VBox(),),
        export_state=state,
        export_dir=str(allowed),
        protected_export_roots=(protected,),
    )
    opener = displayed[-1]
    assert isinstance(opener, widgets.Button)
    opener.click()
    rendered = displayed[-1]
    assert isinstance(rendered, widgets.VBox)
    _export_button(rendered).click()

    exported = allowed / f"{stem}_{timestamp}_1.pdf"
    assert exported.is_file()
    assert dangling_candidate.is_symlink()
    assert dangling_candidate.readlink() == followed_target
    assert not followed_target.exists()
    assert tuple(protected.iterdir()) == ()
    plt.close(figure)


def test_export_writes_only_to_explicitly_selected_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Viewing creates no folder; an explicit user-selected folder receives the PDF."""
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)
    requested = tmp_path / "initial-choice"
    selected = tmp_path / "user-selected"
    figure = plt.figure()
    state: dict[str, object] = {}
    section = notebook_ui.make_dropdown_section(
        [("Small figure", lambda: figure, "small_figure")],
        export_state=state,
        select_first=True,
    )

    notebook_ui.make_lazy_panel_with_tabs(
        (section,),
        export_state=state,
        export_dir=str(requested),
    )
    assert not requested.exists()
    opener = displayed[-1]
    assert isinstance(opener, widgets.Button)
    opener.click()
    rendered = displayed[-1]
    assert isinstance(rendered, widgets.VBox)
    assert not requested.exists()
    assert state["fig"] is figure

    directory = _export_directory(rendered)
    directory.value = str(selected)
    _export_button(rendered).click()

    assert directory.value == str(selected)
    assert not requested.exists()
    pdfs = tuple(selected.glob("*.pdf"))
    assert len(pdfs) == 1
    assert pdfs[0].is_file()
    plt.close(figure)


def test_export_requires_nonblank_directory_and_protects_artifact_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank and immutable artifact destinations produce no filesystem writes."""
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)
    state = {"fig": Figure(), "figures": (), "plot_name": "Small figure"}
    protected = tmp_path / "immutable-artifacts"
    protected.mkdir()
    notebook_ui.make_lazy_panel_with_tabs(
        (widgets.VBox(),),
        export_state=state,
        export_dir="",
        protected_export_roots=(protected,),
    )
    opener = displayed[-1]
    assert isinstance(opener, widgets.Button)
    opener.click()
    rendered = displayed[-1]
    assert isinstance(rendered, widgets.VBox)
    _export_button(rendered).click()
    assert tuple(tmp_path.iterdir()) == (protected,)

    directory = _export_directory(rendered)
    directory.value = str(protected)
    _export_button(rendered).click()
    assert tuple(protected.iterdir()) == ()

    requested = protected / "export"
    directory.value = str(requested)
    _export_button(rendered).click()
    assert not requested.exists()
    assert tuple(protected.iterdir()) == ()

    link = tmp_path / "artifact-link"
    link.symlink_to(protected, target_is_directory=True)
    directory.value = str(link / "through-link")
    _export_button(rendered).click()
    assert not (link / "through-link").exists()
    assert tuple(protected.iterdir()) == ()


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
    assert notebook_ui._export_stem({"filename_prefix": "Transient \u2013 OOD", "plot_name": "generic"}) == "transient_ood_generic"
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


def test_export_reserves_final_filename_without_following_new_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An output entry appearing after name selection cannot redirect the actual write."""
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)
    protected = tmp_path / "immutable"
    protected.mkdir()
    target = protected / "must-not-exist.pdf"
    chosen = tmp_path / "comparison" / "new-entry.pdf"

    def occupied_after_selection(_directory: Path, **_kwargs: object) -> Path:
        chosen.symlink_to(target)
        return chosen

    monkeypatch.setattr(notebook_ui, "_next_export_path", occupied_after_selection)
    figure = Figure()
    notebook_ui.make_lazy_panel_with_tabs(
        (widgets.VBox(),),
        export_state={"fig": figure},
        export_dir=str(chosen.parent),
        protected_export_roots=(protected,),
    )
    open_button = displayed[-1]
    assert isinstance(open_button, widgets.Button)
    open_button.click()
    panel = displayed[-1]
    assert isinstance(panel, widgets.VBox)
    _export_button(panel).click()
    assert chosen.is_symlink()
    assert not target.exists()
    assert tuple(protected.iterdir()) == ()
    plt.close(figure)
