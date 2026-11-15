"""Build notebook view sections, lazy panels, notebook-wide widths and figure exports.

Responsibilities:
  - Assemble plot functions into view sections whose first view renders when its tab first opens
  - Build collapsible tabbed notebook panels with one header row and persistent panel-owned control rows
  - Keep tab and view-selector widths equal over every panel of one notebook
  - Manage Matplotlib figure export state and write exports to the notebooks folder
  - Show a failing view in place while the other views keep working

Design principles:
  - Panels render lazily to keep notebooks responsive; hidden views stay idle
  - Export state is passed explicitly between callbacks
  - Exports go to the repository's notebooks folder, never to a user-chosen path
  - Panels and sections swap container children and show results through plot slots; no output widget or capture exists

This module does NOT:
  - Define primitive widgets or domain-specific scientific controls
  - Load artifact cases or implement case-level plot mathematics
"""

import logging
import os
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from inspect import signature
from pathlib import Path
from typing import IO, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from src import common

from . import analysis_ui_controls as ui_controls
from . import analysis_ui_panel as panel
from . import analysis_ui_style as style
from . import analysis_ui_viewers as viewers

_LOGGER = logging.getLogger(__name__)
_EXPORT_KEYS = ("fig", "figures", "plot_name", "title", "filename_stem", "filename_prefix")


def _sanitize_name(name: str) -> str:
    """
    Convert a display label to the module's minimal export filename stem.

    Text is lowercased. Spaces become underscores. Unicode dashes become ASCII
    hyphens, and forward slashes become underscores. No broader path-policy
    validation is performed here.
    """
    return name.lower().replace(" ", "_").replace("–", "-").replace("—", "-").replace("/", "_")  # noqa: RUF001


def _sanitize_export_basename(value: Any) -> str:
    """Return one safe concise basename for context-aware PDF export."""
    if not isinstance(value, str):
        return "plot"
    normalized = _sanitize_name(value).strip()
    normalized = re.sub(r"[^a-z0-9._-]+", "_", normalized)
    normalized = re.sub(r"[_-]{2,}", "_", normalized).strip("._-")
    return normalized[:96] or "plot"


def _export_stem(export_state: dict[str, Any]) -> str:
    """Resolve a dynamic stem or combine stable panel context with the plot kind."""
    dynamic = export_state.get("filename_stem")
    if isinstance(dynamic, str) and dynamic:
        return _sanitize_export_basename(dynamic)
    context = tuple(
        value
        for value in (
            export_state.get("filename_prefix"),
            export_state.get("plot_name"),
        )
        if isinstance(value, str) and value
    )
    return _sanitize_export_basename("__".join(context) or "plot")


def _next_export_path(directory: Path, *, stem: str, timestamp: str, suffix: str) -> Path:
    """Return one timestamped path with ``suffix`` without overwriting an existing export."""
    candidate = directory / f"{stem}_{timestamp}{suffix}"
    index = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = directory / f"{stem}_{timestamp}_{index}{suffix}"
        index += 1
    return candidate


def _export_destination() -> Path:
    """Return the repository's ``notebooks`` folder or raise why exporting there is refused."""
    if common.paths.source_tree_id() is not None:
        message = "Export is unavailable in a frozen source tree."
        raise RuntimeError(message)
    folder = common.paths.get_project_root() / "notebooks"
    if folder.is_symlink() or not folder.is_dir():
        message = f"The notebooks folder {folder} is missing or not a real directory."
        raise RuntimeError(message)
    return folder


def _show_export_failure(status: ui_controls.StatusLine, error: BaseException) -> None:
    """Report a failed export on the status line."""
    _LOGGER.exception("Export failed: %s", error, exc_info=error)
    status.set(f"[Export] Failed — {type(error).__name__}: {error}", "error")


def write_export(status: ui_controls.StatusLine, *, stem: str, suffix: str, write: Callable[[IO[bytes]], None]) -> None:
    """
    Write one export file into the notebooks folder and report the outcome in ``status``.

    The file is named ``<sanitized stem>_<UTC timestamp><suffix>`` and created exclusively, so no
    existing file is overwritten and no symlink is followed. ``write`` receives the opened
    binary file. A refused destination, a failing writer or a name replaced during the write
    is shown in the status instead of a saved path.
    """
    try:
        directory = _export_destination()
    except RuntimeError as error:
        status.set(f"[Export] {error}", "warning")
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = _next_export_path(directory, stem=_sanitize_export_basename(stem), timestamp=timestamp, suffix=suffix)
    try:
        output = path.open("xb")
    except FileExistsError:
        status.set("[Export] The output filename became occupied. Export again to choose a free filename.", "warning")
        return
    except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
        _show_export_failure(status, error)
        return
    with output:
        try:
            write(output)
        except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
            path.unlink(missing_ok=True)
            _show_export_failure(status, error)
            return
        # The bytes went to the reserved open file; report success only while the name still points to it.
        reserved = os.fstat(output.fileno())
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if current is None or (current.st_dev, current.st_ino) != (reserved.st_dev, reserved.st_ino):
            status.set(f"[Export] {path.name} was replaced during the export; nothing was saved there. Export again.", "warning")
            return
    status.set(f"[Export] Saved: {path}")


def _write_pdf(output: IO[bytes], figures: tuple[Figure, ...]) -> None:
    """Write one figure as a PDF or several figures as pages of one PDF."""
    if len(figures) == 1:
        figures[0].savefig(output, format="pdf", bbox_inches="tight")
        return
    with PdfPages(output) as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")


def export_pdf(status: ui_controls.StatusLine, export_state: dict[str, Any] | None) -> None:
    """Export the current figure bundle of one panel as a PDF into the notebooks folder."""
    if export_state is None:
        status.set("[Export] Export is unavailable for this panel.", "warning")
        return
    primary = export_state.get("fig")
    bundled = export_state.get("figures")
    figures = tuple(figure for figure in bundled if isinstance(figure, Figure)) if isinstance(bundled, (list, tuple)) else ()
    if not figures and isinstance(primary, Figure):
        figures = (primary,)
    if not figures:
        status.set("[Export] No figure is selected. Render a plot first.", "warning")
        return
    write_export(status, stem=_export_stem(export_state), suffix=".pdf", write=partial(_write_pdf, figures=figures))


def _result_items(result: Any) -> tuple[Any, ...]:
    """Return what a view result shows: a list stacks its items in order, ``None`` shows nothing (also inside a list), anything else is one item."""
    return tuple(item for item in (result if isinstance(result, list) else (result,)) if item is not None)


def _show_result(slot: panel.PlotSlot, result: Any) -> None:
    """Show ``result`` in ``slot`` (a list stacks its items, ``None`` empties it); every figure is closed once it is encoded."""
    items = _result_items(result)
    try:
        slot.show(items)
    finally:
        for item in items:
            if isinstance(item, Figure):
                plt.close(item)


def _invoke_dropdown_entry(
    plot_func: Callable[..., Any],
    *,
    export_state: dict[str, Any] | None,
    export_plot_name: str,
    export_title: str,
) -> Any:
    """Invoke one entry with explicit export context when it declares support."""
    try:
        parameters = signature(plot_func).parameters
    except (TypeError, ValueError):
        return plot_func()
    context = {
        "export_state": export_state,
        "export_plot_name": export_plot_name,
        "export_title": export_title,
    }
    return plot_func(**{name: value for name, value in context.items() if name in parameters})


def make_toggle_shortcut(
    dfs: dict[str, pd.DataFrame] | list[pd.DataFrame],
) -> Callable:
    """
    Create a dropdown-entry factory that injects labelled datasets into viewers.

    Parameters
    ----------
    dfs : dict[str, pandas.DataFrame] | list[pandas.DataFrame]
        Dataset mapping, or an ordered list assigned stable ``df0``, ``df1``, ...
        labels within this factory.

    Returns
    -------
    Callable
        Factory returning ``(title, lazy callable, export_name)``. The callable
        accepts panel-local export context. A ``datasets`` keyword is injected
        only when the target declares it. Other caller-supplied keyword
        arguments pass through unchanged.

    Notes
    -----
    Missing export names are generated monotonically. Supplied names use the
    module's minimal filename-stem sanitizer.

    """
    counter = {"i": 0}

    # normalize dfs to dict[str, DataFrame]
    dataset_map = dfs if isinstance(dfs, dict) else {f"df{i}": df for i, df in enumerate(dfs)}

    def toggle(title: str, func: Callable[..., Any], plot_name: str | None = None, **kwargs: Any) -> tuple[str, Callable[..., Any], str]:
        """Bind one target callable and panel-local export identity lazily."""
        if plot_name is None:
            resolved_plot_name = f"plot_{counter['i']:03d}"
            counter["i"] += 1
        else:
            resolved_plot_name = _sanitize_name(plot_name)

        parameters = signature(func).parameters
        bound_kwargs = dict(kwargs)
        if "datasets" in parameters:
            bound_kwargs.setdefault("datasets", dataset_map)

        def invoke(
            *,
            export_state: dict[str, Any] | None = None,
            export_plot_name: str | None = None,
            export_title: str | None = None,
        ) -> Any:
            """Invoke the target with context only when its API declares it."""
            call_kwargs = dict(bound_kwargs)
            context = {
                "export_state": export_state,
                "export_plot_name": export_plot_name,
                "export_title": export_title,
            }
            call_kwargs.update({name: value for name, value in context.items() if name in parameters})
            return func(**call_kwargs)

        return title, invoke, resolved_plot_name

    return toggle


# =============================================================================
# CONTROL-STANDARD SHELL: notebook widths, view sections, lazy panels
# =============================================================================


class WidthScope:
    """
    Keep the tab headers and the view selectors of one notebook equally wide, as wide as their longest title.

    Every registration recomputes both widths over all registered titles and updates every registered widget. Registering
    a tab set again replaces its titles; ``replace`` swaps a panel's view selectors and their titles.
    """

    def __init__(self) -> None:
        """Create an empty scope."""
        self._tabs: dict[int, tuple[widgets.Tab, tuple[str, ...]]] = {}
        self._selectors: dict[int, tuple[widgets.Dropdown, tuple[str, ...]]] = {}

    @property
    def tab_width_ch(self) -> int | None:
        """Return the shared tab header width in ch, or None before any tab set is registered."""
        titles = [title for _tab, names in self._tabs.values() for title in names]
        return panel.tab_width_ch(titles) if titles else None

    def register_tabs(self, tab: widgets.Tab, titles: Sequence[str]) -> None:
        """Register or re-register the titles of one tab set and update every registered width."""
        self._tabs[id(tab)] = (tab, tuple(titles))
        self._apply()

    def register_selectors(self, dropdowns: Sequence[widgets.Dropdown], titles: Sequence[str]) -> None:
        """Register view selectors whose entries come from ``titles`` and update every registered width."""
        for dropdown in dropdowns:
            self._selectors[id(dropdown)] = (dropdown, tuple(titles))
        self._apply()

    def replace(self, old: Sequence[widgets.Dropdown], dropdowns: Sequence[widgets.Dropdown], titles: Sequence[str]) -> None:
        """Drop the registrations of ``old`` and register ``dropdowns`` with ``titles`` in their place."""
        for dropdown in old:
            self._selectors.pop(id(dropdown), None)
        self.register_selectors(dropdowns, titles)

    def _apply(self) -> None:
        width = self.tab_width_ch
        if width is not None:
            for tab, _titles in self._tabs.values():
                panel.set_tab_width(tab, width)
        texts = [title for _dropdown, titles in self._selectors.values() for title in titles]
        for dropdown, _titles in self._selectors.values():
            dropdown.layout.width = f"{ui_controls.dropdown_width(dropdown.description, texts)}px"


class ViewSection(widgets.VBox):
    """
    The content of one tab: a view-selector row and the selected view below it.

    The first entry renders when the tab first opens (rule 2). An entry returning a ``ViewPanel`` is shown as that
    panel; any other result (figure, table, widget) is shown in the section's own plot slot. A failing entry shows its
    failure in place while the other entries stay selectable.
    """

    def __init__(self, entries: Sequence[tuple[str, Callable[..., Any], str]], *, export_state: dict[str, Any] | None) -> None:
        """Create the section; nothing renders before ``activate``."""
        resolved = tuple(entries)
        if not resolved:
            message = "A view section needs at least one entry."
            raise ValueError(message)
        selector = ui_controls.dropdown([(title, index) for index, (title, _func, _name) in enumerate(resolved)], value=0)
        slot = panel.PlotSlot()
        plain = panel.ViewPanel((), slot=slot)
        body = widgets.VBox((plain.widget,))
        super().__init__((panel.control_rows([panel.row(selector)]), body))
        self.entries = resolved
        self.export_state = export_state
        self.selector = selector
        self._slot = slot
        self._plain = plain
        self._body = body
        self._current = plain
        self._active = False
        self._stale = True  # the selected entry has not been rendered yet
        self._detached = False
        self._snapshot: dict[str, Any] | None = None
        selector.observe(self._selected, names="value")

    @property
    def titles(self) -> tuple[str, ...]:
        """Return the entry titles in selector order."""
        return tuple(title for title, _func, _name in self.entries)

    @property
    def status(self) -> ui_controls.StatusLine:
        """Return the status line of the view shown now."""
        return self._current.status

    def activate(self) -> None:
        """
        Show the section: render the selected entry when it is new or was changed while hidden, else restore its export target.

        A selector change made while the section was hidden is rendered here once, with the latest selection (rule 12).
        """
        if self._active:
            return
        self._active = True
        if self._stale:
            self._stale = False
            self._render(self._selected_index())
        elif self.export_state is not None and self._snapshot is not None:
            self.export_state.update(self._snapshot)
        self._current.activate()

    def deactivate(self) -> None:
        """Remember the export target of the shown view and let the view idle."""
        if not self._active:
            return
        self._active = False
        if self.export_state is not None:
            self._snapshot = {key: self.export_state.get(key) for key in _EXPORT_KEYS}
        self._current.deactivate()

    @property
    def detached(self) -> bool:
        """Return whether the section was retired by a tab replacement and can no longer be shown."""
        return self._detached

    def detach(self) -> None:
        """Retire a replaced section for good: let it idle and stop following its selector."""
        self.deactivate()
        if not self._detached:
            self._detached = True
            self.selector.unobserve(self._selected, names="value")

    def show_failure(self, error: BaseException) -> None:
        """Show ``error`` in the place of the shown view's plot, superseding its running renders, under the selected entry's name."""
        title, _func, plot_name = self.entries[self._selected_index()]
        viewers.fail_view(self._current, error, export_state=self.export_state, export_plot_name=plot_name, export_title=title)

    def _selected_index(self) -> int:
        """Return the selected entry's index; the dropdown always holds one."""
        value = self.selector.value
        if value is None:
            message = "A view section's selector lost its selection; it is built with a selected first entry and never cleared."
            raise RuntimeError(message)
        return int(value)

    def _selected(self, change: dict[str, Any]) -> None:
        if not self._active:
            self._stale = True
            return
        self._current.deactivate()
        self._render(int(change["new"]))
        self._current.activate()

    def _show(self, view: panel.ViewPanel) -> None:
        self._current = view
        self._body.children = (view.widget,)
        if view is self._plain:
            # The section's own slot has no callbacks; marked shown, a failure can take its place (rule 12).
            view.activate()

    def _render(self, index: int) -> None:
        title, plot_func, plot_name = self.entries[index]
        state = self.export_state
        ui_controls.release_export(state)
        if state is not None:
            state.update(plot_name=plot_name, title=title)
        self._plain.clear_failure()
        try:
            result = _invoke_dropdown_entry(plot_func, export_state=state, export_plot_name=plot_name, export_title=title)
        except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
            self._show(self._plain)
            viewers.fail_view(self._plain, error, export_state=state, export_plot_name=plot_name, export_title=title)
            return
        primary = result[0] if isinstance(result, tuple) and result else result
        if isinstance(primary, panel.ViewPanel):
            self._show(primary)
            return
        self._show(self._plain)
        figures = tuple(item for item in _result_items(primary) if isinstance(item, Figure))
        if figures and state is not None:
            state.update(fig=figures[0], figures=figures)
        try:
            _show_result(self._slot, primary)
        except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
            viewers.fail_view(self._plain, error, export_state=state, export_plot_name=plot_name, export_title=title)


def view_section(entries: Sequence[tuple[str, Callable[..., Any], str]], *, export_state: dict[str, Any] | None) -> ViewSection:
    """Return one tab's view section over ``(title, callable, export_name)`` entries; its first entry renders when the tab opens."""
    return ViewSection(entries, export_state=export_state)


class LazyPanel:
    """
    One collapsible notebook panel: an open button, then the header row, persistent control rows, and the tabs.

    ``controls`` rows belong to the panel and survive ``replace_tabs``; their popover bodies overlay the tabs. Hidden
    tabs stay idle. The PDF export reports on the status line of the shown view (Q6). ``widths`` is the notebook's
    ``WidthScope``; without one the panel uses its own.
    """

    def __init__(
        self,
        sections: Sequence[ViewSection],
        titles: Sequence[str],
        *,
        open_label: str,
        export_state: dict[str, Any] | None,
        widths: WidthScope | None = None,
        controls: Sequence[widgets.HBox] = (),
        popovers: Sequence[tuple[widgets.ToggleButton, widgets.DOMWidget]] = (),
    ) -> None:
        """Build the panel collapsed; nothing renders before it opens."""
        current = self._validated(sections, titles)
        self.export_state = export_state
        self.widths = widths if widths is not None else WidthScope()
        self._sections = current
        self._expanded = False
        self.widget = widgets.VBox()
        self.widget.add_class(style.ROOT_CLASS)
        self.widget.add_class("glp-shell")
        self.open_button = ui_controls.action_button(open_label, kind="open")
        self.close_button = ui_controls.action_button("Close", kind="close")
        self.export_button = ui_controls.action_button("Export PDF", kind="export")
        self.tabs = panel.tabs(titles, current, width_ch=panel.tab_width_ch(titles))
        rows = [panel.row(self.close_button, self.export_button), *controls]
        content = widgets.VBox((panel.control_rows(rows), *panel.popover_bodies(rows, popovers), self.tabs))
        content.add_class(style.ROOT_CLASS)
        content.add_class("glp-panel")
        carrier = style.stylesheet_widget()
        self._expanded_view = widgets.VBox((carrier, content))
        self._collapsed_view = widgets.VBox((carrier, self.open_button))
        for view in (self._expanded_view, self._collapsed_view):
            view.add_class(style.ROOT_CLASS)
        self.widths.register_tabs(self.tabs, titles)
        self.widths.register_selectors(self._selectors(), self._view_titles())
        self.export_button.on_click(lambda _button: export_pdf(self.status, self.export_state))
        self.open_button.on_click(lambda _button: self.open())
        self.close_button.on_click(lambda _button: self.close())
        self.tabs.observe(self._tab_changed, names="selected_index")
        self._display(self._collapsed_view)

    @property
    def expanded(self) -> bool:
        """Return whether the panel is open."""
        return self._expanded

    @property
    def sections(self) -> tuple[ViewSection, ...]:
        """Return the current view sections in tab order."""
        return tuple(self._sections)

    @property
    def status(self) -> ui_controls.StatusLine:
        """Return the status line of the view shown in the selected tab."""
        return self._selected().status

    def open(self) -> None:
        """Show the panel and activate the selected tab's view."""
        self._expanded = True
        self._activate_selected()
        self._display(self._expanded_view)

    def close(self) -> None:
        """Let the selected view idle and show only the open button."""
        if self._expanded:
            self._selected().deactivate()
        self._expanded = False
        self._display(self._collapsed_view)

    def replace_tabs(self, sections: Sequence[ViewSection], titles: Sequence[str]) -> None:
        """Replace the tabs below the persistent control rows; the control rows and their state stay."""
        replacement = self._validated(sections, titles)
        if self._expanded:
            self._selected().deactivate()
        for section in self._sections:
            if all(section is not kept for kept in replacement):
                section.detach()
        old_selectors = self._selectors()
        self.tabs.unobserve(self._tab_changed, names="selected_index")
        try:
            self._sections = replacement
            self.tabs.children = tuple(replacement)
            self.tabs.titles = tuple(titles)
            self.tabs.selected_index = 0
        finally:
            self.tabs.observe(self._tab_changed, names="selected_index")
        self.widths.register_tabs(self.tabs, titles)
        self.widths.replace(old_selectors, self._selectors(), self._view_titles())
        ui_controls.release_export(self.export_state)
        if self.export_state is not None:
            self.export_state.update(plot_name=None, title=None)
        if self._expanded:
            self._activate_selected()

    @staticmethod
    def _validated(sections: Sequence[ViewSection], titles: Sequence[str]) -> list[ViewSection]:
        current = list(sections)
        if not current or len(current) != len(titles):
            message = "Lazy panels require matching non-empty sections and tab titles."
            raise ValueError(message)
        if not all(isinstance(section, ViewSection) for section in current):
            message = "Lazy panel sections must be built with view_section(...)."
            raise TypeError(message)
        if any(section.detached for section in current):
            message = "A detached view section was replaced and no longer follows its selector; build a new one with view_section(...)."
            raise ValueError(message)
        return current

    def _selected(self) -> ViewSection:
        index = self.tabs.selected_index
        return self._sections[index if isinstance(index, int) and 0 <= index < len(self._sections) else 0]

    def _selectors(self) -> list[widgets.Dropdown]:
        return [section.selector for section in self._sections]

    def _view_titles(self) -> list[str]:
        return [title for section in self._sections for title in section.titles]

    def _activate_selected(self) -> None:
        section = self._selected()
        try:
            section.activate()
        except Exception as error:  # noqa: BLE001 -- UI boundary shows the failure in place
            section.show_failure(error)

    def _tab_changed(self, change: dict[str, Any]) -> None:
        old = change.get("old")
        if self._expanded and isinstance(old, int) and 0 <= old < len(self._sections):
            self._sections[old].deactivate()
        if self._expanded:
            self._activate_selected()

    def _display(self, view: widgets.Widget) -> None:
        self.widget.children = (view,)
