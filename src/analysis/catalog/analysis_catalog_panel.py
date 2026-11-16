"""Collect the plot-catalog entries of all groups and show them in one notebook panel.

Responsibilities:
  - Collect the entries of the group modules in base-type order and refuse duplicate keys
  - Look up one entry by key
  - Build the catalog panel: one tab per base type, one view per entry, through the maintained notebook panel

Design principles:
  - The panel is the same lazy panel the workflow notebooks use, so the catalog shows plots as the notebooks show them
  - Nothing draws before its tab and view are opened

This module does NOT:
  - Hold entries or synthetic data; the group modules own them
  - Show the control catalog; ``analysis.ui.catalog`` owns it
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from IPython.display import display

from src.analysis.ui import analysis_ui_notebook as ui_notebook

from . import analysis_catalog_checks as checks
from . import analysis_catalog_entries as entries
from . import analysis_catalog_evaluation as evaluation
from . import analysis_catalog_inputs as inputs
from . import analysis_catalog_transient as transient

if TYPE_CHECKING:
    import ipywidgets as widgets


def plot_entries() -> tuple[entries.CatalogPlot, ...]:
    """
    Return every plot entry, ordered by base type and key.

    Raises
    ------
    ValueError
        If two entries share a key.

    """
    collected = (*inputs.ENTRIES, *checks.ENTRIES, *evaluation.ENTRIES, *transient.ENTRIES)
    keys = [entry.key for entry in collected]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        message = f"Plot-catalog keys must be unique; duplicated: {duplicates}."
        raise ValueError(message)
    return tuple(sorted(collected, key=lambda entry: (entries.BASE_TYPES.index(entry.base_type), entry.key)))


def plot_entry(key: str) -> entries.CatalogPlot:
    """Return the plot entry ``key``; raise ``KeyError`` naming the known keys."""
    for candidate in plot_entries():
        if candidate.key == key:
            return candidate
    message = f"Unknown plot-catalog entry {key!r}; known entries: {[candidate.key for candidate in plot_entries()]}."
    raise KeyError(message)


def plot_panel() -> widgets.Widget:
    """Return the collapsed catalog panel: a tab per base type with entries, a view per entry, titled by key and sub-type."""
    export_state: dict[str, object] = {}
    sections: list[ui_notebook.ViewSection] = []
    titles: list[str] = []
    for base_type in entries.BASE_TYPES:
        members = [entry for entry in plot_entries() if entry.base_type == base_type]
        if not members:
            continue
        views = [(f"{entry.key} · {entry.title}", entry.draw, entry.key) for entry in members]
        sections.append(ui_notebook.view_section(views, export_state=export_state))
        titles.append(f"{len(titles) + 1}. {base_type}")
    return ui_notebook.LazyPanel(sections, titles, open_label="Plot catalog - Open", export_state=export_state).widget


def show_plots() -> None:
    """Display the catalog panel."""
    display(plot_panel())
