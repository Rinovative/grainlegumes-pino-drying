"""Protect the numeric fills, their key and the index display of styled analysis tables.

Responsibilities:
  - Check that a table is shaded only where its caller opts in, with a key naming the shading, once per grouped table
  - Check that a constant numeric column keeps a pale fill unless the caller switches it off
  - Check that text and boolean cells are never filled and that the fill can be limited to named columns
  - Check that a positional index is hidden and that no index name forms a header row of its own

Design principles:
  - The fill and the index are read from the rendered HTML, the only interface the notebooks see

This module does NOT:
  - Check table layout or the row-local colour matrix
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


def _heading(box: widgets.VBox) -> str:
    """Return the heading HTML of a styled-table widget."""
    return str(cast("Any", box.children[0]).value)


def _all_html(widget: Any) -> str:
    """Return the HTML of every HTML widget inside ``widget``."""
    return "".join(_all_html(child) for child in widget.children) if hasattr(widget, "children") else str(widget.value)


def test_constant_numeric_columns_are_filled_pale_only_when_shade_constant_holds() -> None:
    """A varying column is always filled; a constant one only with ``shade_constant``."""
    table = pd.DataFrame({"varying": [1.0, 2.0, 3.0], "constant": [5.0, 5.0, 5.0]})
    shaded = _html(tables.styled_dataframe(table, title="t", shade_columns=tuple(table.columns)))
    plain = _html(tables.styled_dataframe(table, title="t", shade_constant=False, shade_columns=tuple(table.columns)))
    rules = [html.split("</style>")[0] for html in (shaded, plain)]  # Styler lists the filled cells as CSS selectors
    assert [rule.count("_col0") for rule in rules] == [len(table)] * 2
    assert [rule.count("_col1") for rule in rules] == [len(table), 0]


def test_only_the_named_columns_are_filled_and_an_empty_cell_stays_empty() -> None:
    """``shade_columns`` limits the fill to its columns; a text cell among the numbers is neither filled nor printed as nan."""
    table = pd.DataFrame({"unit": [1.0, 2.0, 3.0], "ratio": [0.5, "", 2.0]})
    rule = _html(tables.styled_dataframe(table, title="t", shade_columns=("ratio",))).split("</style>")[0]
    assert (rule.count("_col0"), rule.count("_col1")) == (0, 2)
    assert "nan" not in _html(tables.styled_dataframe(table, title="t", shade_columns=("ratio",)))


def test_a_single_row_has_nothing_to_compare_and_stays_unshaded_without_a_key() -> None:
    """A column shades against its own values, so a one-row table (one run, one dataset) carries no fill and no key."""
    single = tables.styled_dataframe(pd.DataFrame({"rmse": [0.1], "mae": [0.05]}), title="t", shade_columns=("rmse", "mae"))
    assert "background-color" not in _html(single)
    assert "Shading" not in _heading(single)


def test_a_column_of_tiny_values_is_not_taken_for_a_constant_one() -> None:
    """Values near 1e-9 differ by a factor of four, so they are filled even without ``shade_constant``."""
    tiny = pd.DataFrame({"tiny": [2.0e-9, 8.0e-9]})
    rule = _html(tables.styled_dataframe(tiny, title="t", shade_constant=False, shade_columns=("tiny",))).split("</style>")[0]
    assert rule.count("_col0") == len(tiny)


def test_a_table_is_shaded_only_where_its_caller_opts_in_with_a_key_and_never_in_text_or_boolean_cells() -> None:
    """By default nothing is filled and no key shows; named columns are filled and keyed, their text and boolean cells never."""
    table = pd.DataFrame({"rmse": [0.1, 0.3], "available": [True, False], "case": ["41", "1"]})
    plain = tables.styled_dataframe(table, title="t")
    assert "background-color" not in _html(plain)
    assert "Shading" not in _heading(plain)
    shaded = tables.styled_dataframe(table, title="t", shade_columns=tuple(table.columns))
    rule = _html(shaded).split("</style>")[0]
    assert (rule.count("_col0"), rule.count("_col1"), rule.count("_col2")) == (2, 0, 0)
    assert "Shading" in _heading(shaded)
    groups = [(name, pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 1.0]}, index=["p", "q"])) for name in ("first", "second")]
    assert _all_html(tables.grouped_styled_dataframes(groups, title="t", row_local=True)).count("Shading") == 1


def test_a_positional_index_is_hidden_and_no_index_name_forms_a_header_row_of_its_own() -> None:
    """Row positions 0..n-1 are incidental; named row labels stay visible, but neither axis name gets a header cell."""
    positional = _html(tables.styled_dataframe(pd.DataFrame({"rmse": [0.1, 0.3]}, index=[3, 7]), title="t"))
    assert "row_heading" not in positional
    named = pd.DataFrame({"rmse": [0.1, 0.3]}, index=pd.Index(["FNO", "U-NO"], name="label"))
    for table in (named, named.transpose()):  # a transposed table carries the index name on its columns
        html = _html(tables.styled_dataframe(table, title="t"))
        assert "FNO" in html
        assert ">label</th>" not in html
