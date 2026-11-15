"""Protect figure display, notebooks-folder export, and shared controls.

Responsibilities:
  - Check the PNG display of a figure, export destination and naming
  - Check the panel export controls, the export writer, and the refusals that leave the notebooks folder untouched

Design principles:
  - Exports reach only the repository's notebooks folder, resolved at export time

This module does NOT:
  - Validate scientific field values or manage persisted artifacts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.analysis.ui import analysis_ui_controls as controls
from src.analysis.ui import analysis_ui_notebook as notebook_ui
from src.analysis.ui import analysis_ui_panel as panel_ui

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


def _open_panel(state: dict[str, Any]) -> notebook_ui.LazyPanel:
    """Build a lazy panel exporting ``state`` over one empty view, open it, and return it."""
    section = notebook_ui.view_section([("View", lambda: None, "view")], export_state=None)
    lazy = notebook_ui.LazyPanel([section], ["1. View"], open_label="Open", export_state=state)
    lazy.open()
    return lazy


def _png_size(image: widgets.Widget) -> tuple[int, int]:
    """Return the pixel width and height of a PNG image widget."""
    assert isinstance(image, widgets.Image)
    assert image.format == "png"
    value = bytes(image.value)
    assert value[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(value[16:20], "big"), int.from_bytes(value[20:24], "big")


def test_rendered_figure_becomes_a_png_image_of_its_own_pixel_size() -> None:
    """A figure is shown as one PNG at its own dpi, not at the savefig dpi, and keeps that size in the slot."""
    figure = Figure(figsize=(4.0, 3.0), dpi=50)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_axis_off()
    axes.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=axes.transAxes))
    slot = panel_ui.PlotSlot()
    with plt.rc_context({"savefig.dpi": 300}):
        slot.show((figure,))
    (image,) = slot.widget.children
    pad = round(2 * plt.rcParams["savefig.pad_inches"] * figure.dpi)
    assert _png_size(image) == (200 + pad, 150 + pad)
    assert image.layout.width is None
    assert image.layout.max_width is None


def test_panel_exports_without_directory_field() -> None:
    """The open panel offers a close and an export button; no text field exists for an export directory."""
    lazy = _open_panel({})
    assert lazy.close_button.description == "Close"
    assert lazy.export_button.description == "Export PDF"

    def walk(widget: widgets.Widget) -> list[widgets.Widget]:
        return [widget, *(item for child in getattr(widget, "children", ()) for item in walk(child))]

    assert not any(isinstance(item, widgets.Text) for item in walk(lazy.widget))


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
    button = _open_panel(state).export_button
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
    _open_panel({"fig": figure, "figures": (), "plot_name": "collision-check"}).export_button.click()

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
    _open_panel({"fig": figure}).export_button.click()
    assert chosen.is_symlink()
    assert not target.exists()


@pytest.mark.parametrize("case", ["source_tree", "missing", "symlink"])
def test_export_refusal_is_visible_and_writes_nothing(case: str, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    rendered = _open_panel({"fig": figure})
    rendered.export_button.click()
    assert "[Export]" in rendered.status.text
    assert not any(project_root.rglob("*.pdf"))
    assert not (folder.exists() and not folder.is_symlink() and any(folder.iterdir()))


def test_write_export_uses_suffix_and_removes_file_of_failed_writer(project_root: Path) -> None:
    """The shared writer sanitizes the stem and serves other suffixes; a failing writer leaves no file and shows the failure."""
    status = controls.StatusLine()
    folder = project_root / "notebooks"

    def writing(output: Any) -> None:
        output.write(b"GIF89a")

    notebook_ui.write_export(status, stem="Sample / A", suffix=".gif", write=writing)
    (written,) = folder.iterdir()
    assert written.name.startswith("sample_a_")
    assert written.suffix == ".gif"
    assert written.read_bytes() == b"GIF89a"

    def failing(_output: object) -> None:
        message = "encoder failed"
        raise OSError(message)

    notebook_ui.write_export(status, stem="bad", suffix=".gif", write=failing)
    assert "OSError: encoder failed" in status.text
    assert [path.name for path in folder.iterdir()] == [written.name]


def test_write_export_open_failure_leaves_existing_file_untouched(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An open error other than an occupied name is shown and never unlinks the chosen path."""
    status = controls.StatusLine()
    foreign = project_root / "notebooks" / "foreign.pdf"
    foreign.write_bytes(b"keep")
    monkeypatch.setattr(notebook_ui, "_next_export_path", lambda *_args, **_kwargs: foreign)

    def refuse_open(_path: Path, *_args: object, **_kwargs: object) -> None:
        message = "open refused"
        raise PermissionError(message)

    monkeypatch.setattr("pathlib.Path.open", refuse_open)
    called: list[object] = []
    notebook_ui.write_export(status, stem="x", suffix=".pdf", write=called.append)
    assert "PermissionError: open refused" in status.text
    assert not called
    monkeypatch.undo()
    assert foreign.read_bytes() == b"keep"


def test_export_stem_uses_prefix_then_generic_fallback() -> None:
    """Keep legacy prefix and generic plot names backward-compatible."""
    assert notebook_ui._export_stem({"filename_prefix": "Transient – OOD", "plot_name": "generic"}) == "transient_ood_generic"  # noqa: RUF001
    assert notebook_ui._export_stem({"plot_name": "Generic / Plot"}) == "generic_plot"
