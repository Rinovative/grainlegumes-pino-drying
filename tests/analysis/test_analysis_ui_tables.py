"""Protect the numeric fills of styled analysis tables.

Responsibilities:
  - Check that a constant numeric column keeps a pale fill unless the caller switches it off
  - Check that the fill can be limited to named columns

Design principles:
  - The fill is read from the rendered HTML, the only interface the notebooks see

This module does NOT:
  - Check table layout, titles or the row-local colour matrix
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from src.analysis.ui import analysis_ui_tables as tables

if TYPE_CHECKING:
    import ipywidgets as widgets


def _html(box: widgets.VBox) -> str:
    """Return the table HTML of a styled-table widget."""
    return str(cast("Any", box.children[1]).value)


def test_constant_numeric_columns_are_filled_pale_only_when_shade_constant_holds() -> None:
    """A varying column is always filled; a constant one only with ``shade_constant``."""
    table = pd.DataFrame({"varying": [1.0, 2.0, 3.0], "constant": [5.0, 5.0, 5.0]})
    shaded = _html(tables.styled_dataframe(table, title="t"))
    plain = _html(tables.styled_dataframe(table, title="t", shade_constant=False))
    rules = [html.split("</style>")[0] for html in (shaded, plain)]  # Styler lists the filled cells as CSS selectors
    assert [rule.count("_col0") for rule in rules] == [len(table)] * 2
    assert [rule.count("_col1") for rule in rules] == [len(table), 0]


def test_only_the_named_columns_are_filled_and_an_empty_cell_stays_empty() -> None:
    """``shade_columns`` limits the fill to its columns; a text cell among the numbers is neither filled nor printed as nan."""
    table = pd.DataFrame({"unit": [1.0, 2.0, 3.0], "ratio": [0.5, "", 2.0]})
    rule = _html(tables.styled_dataframe(table, title="t", shade_columns=("ratio",))).split("</style>")[0]
    assert (rule.count("_col0"), rule.count("_col1")) == (0, 2)
    assert "nan" not in _html(tables.styled_dataframe(table, title="t", shade_columns=("ratio",)))


def test_a_column_of_tiny_values_is_not_taken_for_a_constant_one() -> None:
    """Values near 1e-9 differ by a factor of four, so they are filled even without ``shade_constant``."""
    tiny = pd.DataFrame({"tiny": [2.0e-9, 8.0e-9]})
    rule = _html(tables.styled_dataframe(tiny, title="t", shade_constant=False)).split("</style>")[0]
    assert rule.count("_col0") == len(tiny)
