"""
analysis_ui_notebook.py

Build notebook dropdown sections, lazy panels and figure exports.

Responsibilities:
  - Assemble plot functions into dropdown sections
  - Build collapsible tabbed notebook panels
  - Manage Matplotlib figure export state

Design principles:
  - Panels render lazily to keep notebooks responsive
  - Export state is passed explicitly between callbacks
  - Display helpers handle figures, widgets and rich objects uniformly

This module does NOT:
  - Define primitive widgets or domain-specific scientific controls
  - Load artifact cases or implement case-level plot mathematics
"""

import re
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from . import analysis_ui_components as components

COMPACT_VIEW_SELECTOR_WIDTH_PX = components.COMPACT_CONTROL_WIDTH_PX
STANDARD_VIEW_SELECTOR_WIDTH_PX = 360


class _DropdownSectionVBox(widgets.VBox):
    """Expose lifecycle deactivation for the currently rendered dropdown view."""

    def __init__(
        self,
        children: Sequence[widgets.Widget],
        *,
        activate: Callable[[], None],
        deactivate: Callable[[], None],
    ) -> None:
        """Retain normal section construction beside lifecycle callbacks."""
        super().__init__(children=tuple(children))
        self._activate_callback = activate
        self._deactivate_callback = deactivate

    def activate(self) -> None:
        """Reactivate the currently rendered view after it becomes visible."""
        self._activate_callback()

    def deactivate(self) -> None:
        """Deactivate the currently rendered view before it is hidden."""
        self._deactivate_callback()


def _activate_result(result: Any) -> None:
    """Invoke one explicit activation hook when the result provides it."""
    primary = result[0] if isinstance(result, tuple) and result else result
    activate = getattr(primary, "activate", None)
    if callable(activate):
        activate()


def _deactivate_result(result: Any) -> None:
    """Invoke one explicit deactivation hook when the result provides it."""
    primary = result[0] if isinstance(result, tuple) and result else result
    deactivate = getattr(primary, "deactivate", None)
    if callable(deactivate):
        deactivate()


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


def _next_export_path(directory: Path, *, stem: str, timestamp: str) -> Path:
    """Return one timestamped PDF path without overwriting an existing export."""
    candidate = directory / f"{stem}_{timestamp}.pdf"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{timestamp}_{index}.pdf"
        index += 1
    return candidate


def _show_anything(result: Any) -> None:
    """
    Display one supported result in the active notebook output context.

    Matplotlib figures are displayed and closed, objects with ``show`` invoke it,
    strings are printed, and other non-``None`` values use IPython display.
    """
    if isinstance(result, Figure):
        display(result)
        plt.close(result)
    elif hasattr(result, "show") and callable(result.show):
        result.show()
    elif isinstance(result, str):
        print(result)
    elif result is not None:
        display(result)


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


def make_dropdown_section(
    plots: list,
    *,
    export_state: dict | None = None,
    select_first: bool = False,
) -> Any:
    """
    Build one lazy dropdown whose entries render notebook views on selection.

    Parameters
    ----------
    plots : list
        Ordered ``(title, lazy callable, export_name)`` entries.
    export_state : dict | None, optional
        Shared mutable state receiving the current title/name and direct or
        viewer-rendered Matplotlib figure for later PDF export.
    select_first : bool, optional
        Omit the prompt and mark the first entry for activation when its tab opens.

    Returns
    -------
    ipywidgets.VBox
        Dropdown and output area. A plot callable runs only when its entry is
        selected for the first time after a different selection.

    Notes
    -----
    Selecting the prompt clears output. A first-entry section remains lazy until
    its tab opens. Before rendering, the prior export figure is cleared. Non-figure
    widgets may populate it later through viewer callbacks.

    """
    plot_options = [(title, i) for i, (title, _, _) in enumerate(plots)]
    dropdown = widgets.Dropdown(
        options=plot_options if select_first else [("Choose a view…", -1), *plot_options],
        value=None if select_first else -1,
        description="" if select_first else "View:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width=(f"{COMPACT_VIEW_SELECTOR_WIDTH_PX}px" if select_first else f"{STANDARD_VIEW_SELECTOR_WIDTH_PX}px")),
    )
    output = widgets.Output()
    last_idx: dict[str, int | None] = {"idx": None}
    active_result: dict[str, Any] = {"value": None}

    def activate_current() -> None:
        """Refresh the cached current view against shared state when visible."""
        _activate_result(active_result["value"])

    def deactivate_current() -> None:
        """Stop active view resources without discarding cached widget state."""
        _deactivate_result(active_result["value"])

    def on_plot_change(change: dict) -> None:
        """
        Render a newly selected entry and synchronize panel-local export state.

        Repeated selection is ignored, the prompt clears output, and direct
        figures replace the export target while viewer widgets may update it
        later through the explicitly supplied viewer state.
        """
        idx = change["new"]
        if last_idx["idx"] == idx:
            return
        deactivate_current()
        if idx == -1:
            if export_state is not None:
                export_state["fig"] = None
                export_state["figures"] = ()
                export_state["plot_name"] = None
                export_state["title"] = None
                export_state["filename_stem"] = None
            with output:
                output.clear_output(wait=True)
            last_idx["idx"] = idx
            return

        title, plot_func, plot_name = plots[idx]
        with output:
            output.clear_output(wait=True)

            if export_state is not None:
                previous = export_state.get("fig")
                if isinstance(previous, Figure):
                    plt.close(previous)
                export_state["fig"] = None
                export_state["figures"] = ()
                export_state["plot_name"] = plot_name
                export_state["title"] = title
                export_state["filename_stem"] = None

            result = _invoke_dropdown_entry(
                plot_func,
                export_state=export_state,
                export_plot_name=plot_name,
                export_title=title,
            )
            active_result["value"] = result
            if isinstance(result, tuple):
                result = result[0]

            # update export target for direct Figure returns
            if export_state is not None:
                export_state["title"] = title
                export_state["plot_name"] = plot_name

                # Viewer callbacks may already have stored their rendered Figure
                # in this panel's explicit state. Preserve it for non-Figure widgets.
                if isinstance(result, Figure):
                    export_state["fig"] = result

            if isinstance(result, Figure):
                display(result)
                plt.close(result)
            else:
                _show_anything(result)

        last_idx["idx"] = idx

    dropdown.observe(on_plot_change, names="value")

    return _DropdownSectionVBox(
        (dropdown, output),
        activate=activate_current,
        deactivate=deactivate_current,
    )


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
        accepts panel-local export context from :func:`make_dropdown_section`. A
        ``datasets`` keyword is injected only when the target declares it. Other
        caller-supplied keyword arguments pass through unchanged.

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


class LazyTabbedPanelOutput(widgets.Output):
    """Expose controlled replacement of one lazy tab collection."""

    def __init__(self) -> None:
        """Initialize before the panel factory binds its replacement callback."""
        super().__init__()
        self._replace_tabs_callback: (
            Callable[
                [Sequence[widgets.Widget], Sequence[str]],
                None,
            ]
            | None
        ) = None
        self.tabs: widgets.Tab | None = None

    def bind_tab_replacement(
        self,
        callback: Callable[[Sequence[widgets.Widget], Sequence[str]], None],
    ) -> None:
        """Bind the sole panel-owned section replacement callback."""
        if self._replace_tabs_callback is not None:
            message = "Lazy panel tab replacement is already initialized."
            raise RuntimeError(message)
        self._replace_tabs_callback = callback

    def replace_tabs(
        self,
        sections: Sequence[widgets.Widget],
        tab_titles: Sequence[str],
    ) -> None:
        """Replace visible sections without rebuilding the panel."""
        if self._replace_tabs_callback is None:
            message = "Lazy panel tab replacement is not initialized."
            raise RuntimeError(message)
        self._replace_tabs_callback(sections, tab_titles)


def make_lazy_panel_with_tabs(  # noqa: C901, PLR0915 -- coordinated lifecycle
    sections: Sequence[widgets.Widget],
    tab_titles: Sequence[str] | None = None,
    open_btn_text: str = "Open section",
    close_btn_text: str = "Close",
    *,
    panel_controls: Sequence[widgets.Widget] = (),
    export_state: dict | None = None,
    export_dir: str = "exports",
    export_btn_text: str = "Export PDF",
) -> LazyTabbedPanelOutput:
    """Build one collapsible, replaceable lazy tab panel with PDF export."""
    current_sections = list(sections)
    titles = list(tab_titles) if tab_titles is not None else [f"Tab {index + 1}" for index in range(len(current_sections))]
    if not current_sections or len(titles) != len(current_sections):
        message = "Lazy panels require matching non-empty sections and tab titles."
        raise ValueError(message)

    main_out = LazyTabbedPanelOutput()
    open_btn = widgets.Button(
        description=open_btn_text,
        button_style="primary",
        layout=widgets.Layout(width="auto"),
    )
    close_btn = widgets.Button(
        description=close_btn_text,
        button_style="danger",
        layout=widgets.Layout(width="145px"),
    )
    tabs = widgets.Tab(children=tuple(current_sections))
    main_out.tabs = tabs

    def set_titles(values: Sequence[str]) -> None:
        """Install one exact title for each current section."""
        if len(values) != len(tabs.children):
            message = "Tab titles must match the current section count."
            raise ValueError(message)
        for index, value in enumerate(values):
            tabs.set_title(index, value)

    set_titles(titles)
    status_out = widgets.Output(
        layout=widgets.Layout(
            display="none",
            flex="0 0 auto",
        )
    )
    export_btn = widgets.Button(
        description=export_btn_text,
        button_style="success",
        layout=widgets.Layout(width="145px"),
    )

    def do_export(_: object = None) -> None:
        """Write the current single- or multi-page figure bundle to PDF."""
        status_out.layout.display = "block"
        with status_out:
            status_out.clear_output(wait=True)
            if export_state is None:
                print("[Export] Export is unavailable for this panel.")
                return
            primary = export_state.get("fig")
            bundled = export_state.get("figures")
            figures = tuple(figure for figure in bundled if isinstance(figure, Figure)) if isinstance(bundled, (list, tuple)) else ()
            if not figures and isinstance(primary, Figure):
                figures = (primary,)
            if not figures:
                print("[Export] No figure is selected. Render a plot first.")
                return
            out_dir = Path(export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = _export_stem(export_state)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = _next_export_path(out_dir, stem=stem, timestamp=timestamp)
            if len(figures) == 1:
                figures[0].savefig(out_path, bbox_inches="tight")
            else:
                with PdfPages(out_path) as pdf:
                    for figure in figures:
                        pdf.savefig(figure, bbox_inches="tight")
            print(f"[Export] Saved: {out_path}")

    export_btn.on_click(do_export)
    header = widgets.HBox((close_btn, export_btn))
    controls_box = widgets.VBox(tuple(panel_controls))
    panel_children: list[widgets.Widget] = [header]
    if panel_controls:
        panel_children.append(controls_box)
    panel_children.extend((status_out, tabs))
    panel = widgets.VBox(tuple(panel_children))
    initialized_sections: set[int] = set()
    section_export_states: dict[int, dict[str, Any]] = {}
    expanded = {"value": False}

    def current_section(index: int | None) -> widgets.Widget | None:
        """Resolve one current section index without stale-list assumptions."""
        if index is None or not 0 <= index < len(current_sections):
            return None
        return current_sections[index]

    def snapshot_section(index: int | None) -> None:
        """Preserve one section's current export target before hiding it."""
        section = current_section(index)
        if export_state is None or section is None:
            return
        section_export_states[id(section)] = {
            "fig": export_state.get("fig"),
            "figures": export_state.get("figures", ()),
            "plot_name": export_state.get("plot_name"),
            "title": export_state.get("title"),
            "filename_stem": export_state.get("filename_stem"),
            "filename_prefix": export_state.get("filename_prefix"),
        }

    def deactivate_section(index: int | None) -> None:
        """Deactivate one visible section before tab replacement."""
        section = current_section(index)
        deactivate = None if section is None else getattr(section, "deactivate", None)
        if callable(deactivate):
            deactivate()

    def activate_selected_view(_: object = None) -> None:
        """Initialize or reactivate the currently selected lazy section."""
        selected_index = tabs.selected_index
        section = current_section(selected_index)
        if section is None:
            return
        section_children = getattr(section, "children", ())
        if not section_children or not isinstance(section_children[0], widgets.Dropdown):
            return
        section_id = id(section)
        dropdown = section_children[0]
        if section_id not in initialized_sections:
            if dropdown.index is None and dropdown.options:
                dropdown.index = 0
            initialized_sections.add(section_id)
            snapshot_section(selected_index)
            return
        if export_state is not None:
            export_state.update(
                section_export_states.get(
                    section_id,
                    {
                        "fig": None,
                        "figures": (),
                        "plot_name": None,
                        "title": None,
                        "filename_stem": None,
                        "filename_prefix": None,
                    },
                )
            )
        activate = getattr(section, "activate", None)
        if callable(activate):
            activate()
        snapshot_section(selected_index)

    def show_panel(_: object = None) -> None:
        """Display the expanded panel and activate its selected view."""
        expanded["value"] = True
        snapshot_section(tabs.selected_index)
        activate_selected_view()
        with main_out:
            clear_output()
            display(panel)

    def show_open(_: object = None) -> None:
        """Display only the established collapsed open button."""
        expanded["value"] = False
        snapshot_section(tabs.selected_index)
        deactivate_section(tabs.selected_index)
        with main_out:
            clear_output()
            display(open_btn)

    def on_tab_change(change: dict[str, object]) -> None:
        """Deactivate the hidden section and activate the newly selected one."""
        old_index = change.get("old")
        previous = old_index if isinstance(old_index, int) else None
        snapshot_section(previous)
        deactivate_section(previous)
        if expanded["value"]:
            activate_selected_view()

    def replace_tabs(
        replacement_sections: Sequence[widgets.Widget],
        replacement_titles: Sequence[str],
    ) -> None:
        """Swap visible sections while preserving cached view state."""
        resolved_sections = list(replacement_sections)
        resolved_titles = list(replacement_titles)
        if not resolved_sections or len(resolved_sections) != len(resolved_titles):
            message = "Replacement tabs require matching non-empty sections and titles."
            raise ValueError(message)
        snapshot_section(tabs.selected_index)
        deactivate_section(tabs.selected_index)
        tabs.unobserve(on_tab_change, names="selected_index")
        try:
            current_sections[:] = resolved_sections
            tabs.children = tuple(resolved_sections)
            set_titles(resolved_titles)
            tabs.selected_index = 0
        finally:
            tabs.observe(on_tab_change, names="selected_index")
        if export_state is not None:
            export_state.update(
                {
                    "fig": None,
                    "figures": (),
                    "plot_name": None,
                    "title": None,
                    "filename_stem": None,
                    "filename_prefix": export_state.get("filename_prefix"),
                }
            )
        if expanded["value"]:
            activate_selected_view()

    main_out.bind_tab_replacement(replace_tabs)
    tabs.observe(on_tab_change, names="selected_index")
    open_btn.on_click(show_panel)
    close_btn.on_click(show_open)
    show_open()
    return main_out
