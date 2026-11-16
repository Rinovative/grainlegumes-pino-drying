"""Protect the plot catalog behind the plot-standard notebook.

Responsibilities:
  - Draw every plot entry once through its real builder and check that it shows something
  - Check the entry contract (base type, key form), unique keys, resolvable builder names and the panel build

Design principles:
  - One case per entry, parametrized by group and key, so ``-k <group>`` selects one group module

This module does NOT:
  - Check the visual result or the builders' own contracts; their tests do
"""

from __future__ import annotations

import importlib
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler

from src.analysis import catalog
from src.analysis.catalog import analysis_catalog_entries as entries

ENTRIES = catalog.panel.plot_entries()


def _group(entry: entries.CatalogPlot) -> str:
    return entry.draw.__module__.rsplit("_", 1)[-1]


def _items(result: Any) -> list[Any]:
    return [item for item in (result if isinstance(result, (list, tuple)) else (result,)) if item is not None]


@pytest.mark.parametrize("entry", ENTRIES, ids=[f"{_group(entry)}-{entry.key}" for entry in ENTRIES])
def test_every_entry_draws_through_its_builder(entry: entries.CatalogPlot) -> None:
    items = _items(entry.draw())
    try:
        assert items, f"{entry.key} drew nothing"
        for item in items:
            if isinstance(item, Figure):
                assert any(axis.has_data() or axis.images or axis.collections for axis in item.axes), f"{entry.key}: empty figure"
            elif isinstance(item, pd.DataFrame):
                assert not item.empty, f"{entry.key}: empty table"
            elif isinstance(item, Styler):
                assert "<td" in item.to_html(), f"{entry.key}: empty styled table"
            else:
                assert isinstance(item, (str, widgets.Widget)), f"{entry.key}: unexpected item {type(item).__name__}"
                assert item, f"{entry.key}: empty item"
    finally:
        for item in items:
            if isinstance(item, Figure):
                plt.close(item)


def test_keys_are_unique_ordered_by_base_type_and_builders_resolve() -> None:
    keys = [entry.key for entry in ENTRIES]
    assert len(keys) == len(set(keys))
    assert [entries.BASE_TYPES.index(entry.base_type) for entry in ENTRIES] == sorted(entries.BASE_TYPES.index(entry.base_type) for entry in ENTRIES)
    for entry in ENTRIES:
        for builder in entry.builders:
            module_name, _, function_name = builder.rpartition(".")
            assert callable(getattr(importlib.import_module(module_name), function_name)), builder


def test_entry_contract_refuses_an_unknown_base_type_a_bad_number_and_no_builder() -> None:
    assert entries.CatalogPlot("map-plus-lines_02", "x", ("src.x.f",), lambda: None).base_type == "map-plus-lines"
    with pytest.raises(ValueError, match="base type"):
        entries.CatalogPlot("heatmap_01", "x", ("src.x.f",), lambda: None)
    with pytest.raises(ValueError, match="base type"):
        entries.CatalogPlot("table_1", "x", ("src.x.f",), lambda: None)
    with pytest.raises(ValueError, match="names no builder"):
        entries.CatalogPlot("table_01", "x", (), lambda: None)


def test_lookup_names_the_known_keys_and_the_panel_builds_collapsed() -> None:
    with pytest.raises(KeyError, match="known entries"):
        catalog.panel.plot_entry("no-such_01")
    if ENTRIES:
        assert catalog.panel.plot_entry(ENTRIES[0].key) is ENTRIES[0]
        assert isinstance(catalog.panel.plot_panel(), widgets.Widget)
