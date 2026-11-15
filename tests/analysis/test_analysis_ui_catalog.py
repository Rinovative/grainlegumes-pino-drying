"""Protect the control catalog behind the plot-standard notebook.

Responsibilities:
  - Check that every control entry renders once without a frontend and that keys are unique and complete
  - Check the unknown-key error, figure rendering into a plot slot, and refusal of other results

Design principles:
  - Entries are built by the real constructors on synthetic data; display calls are captured

This module does NOT:
  - Check the visual result in a browser
"""

from __future__ import annotations

from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pytest

from src.analysis.ui import analysis_ui_catalog as catalog

PLANNED_KEYS = (
    "dropdowns",
    "column-grid",
    "channels",
    "case-navigator",
    "time-navigator",
    "transition",
    "count",
    "choice-groups",
    "toggles",
    "actions",
    "status",
    "failure",
    "runs-multiple",
    "run-single",
    "datasets-multiple",
    "datasets-pool-disabled",
    "dataset-single",
    "on-demand",
    "tabs",
    "lazy-panel",
    "view-panel",
)


@pytest.fixture(autouse=True)
def quiet_display(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture the display calls of the catalog."""
    shown: list[Any] = []
    monkeypatch.setattr(catalog, "display", shown.append)
    return shown


def test_every_entry_renders_once() -> None:
    entries = catalog.control_entries()
    assert tuple(entry.key for entry in entries) == PLANNED_KEYS
    known = {entry.key for entry in entries}
    for entry in entries:
        assert set(entry.uses) <= known
        rendered = catalog.render(entry)
        assert isinstance(rendered, widgets.VBox)
        assert "<details>" in rendered.children[1].value
        assert entry.build.__name__ in rendered.children[1].value


def test_unknown_key_lists_the_known_keys() -> None:
    with pytest.raises(KeyError, match="dropdowns"):
        catalog.entry("missing")


def test_figure_result_renders_into_a_plot_slot_and_other_results_are_refused() -> None:
    figure = plt.figure()
    rendered = catalog.render(catalog.CatalogEntry("figure", "Figure", lambda: figure))
    assert "glp-plot" in rendered.children[2]._dom_classes
    (image,) = rendered.children[2].children
    assert isinstance(image, widgets.Image)
    assert not plt.fignum_exists(figure.number)
    with pytest.raises(TypeError, match="expected a widget"):
        catalog.render(catalog.CatalogEntry("text", "Text", lambda: "not a widget"))


def test_show_displays_the_rendered_entry(quiet_display: list[Any]) -> None:
    catalog.show("actions")
    assert isinstance(quiet_display[-1], widgets.VBox)
