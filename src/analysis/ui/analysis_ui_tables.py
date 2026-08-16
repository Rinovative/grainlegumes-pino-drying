"""
===============================================================================
analysis_ui_tables.py
===============================================================================
Render reusable styled DataFrame tables for analysis notebooks.
Responsibilities:
  - Format numeric table values consistently for notebook display
  - Derive reusable readable row-local colors for HTML and figure tables
  - Compose compact single or grouped table widget layouts with safe titles
Design principles:
  - Table styling is presentation-only
  - Numeric color scaling is configurable by column or per-row
  - Callers retain ownership of table content, grouping, and semantic labels
This module does NOT:
  - Compute scientific statistics or choose displayed columns
  - Compose notebook sections or manage figure-export state
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any


def _blue_numeric_styles(table: pd.DataFrame, *, shade_constant: bool) -> pd.DataFrame:
    """Return quantile-bounded blue fills for finite numeric cells, column-local."""
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    colormap = plt.get_cmap("Blues")
    for column in table.columns:
        numeric = np.asarray(pd.to_numeric(table[column], errors="coerce"), dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            continue
        lower, upper = np.quantile(finite, (0.05, 0.95))
        if np.isclose(lower, upper) and not shade_constant:
            continue
        for row, value in enumerate(numeric):
            if not np.isfinite(value):
                continue
            if np.isclose(lower, upper):
                continue
            fraction = float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
            red, green, blue, _alpha = colormap(0.05 + 0.90 * fraction)
            column_index = cast("int", styles.columns.get_loc(column))
            styles.iloc[row, column_index] = f"background-color: rgba({int(red * 255)}, {int(green * 255)}, {int(blue * 255)}, 0.55)"
    return styles


ROW_EQUAL_RELATIVE_TOLERANCE = 1.0e-9
_NEUTRAL_BACKGROUND_COLOR = "#eef1f5"
_DARK_TEXT_COLOR = "#1f2933"
_LIGHT_TEXT_COLOR = "#ffffff"
_DARK_BACKGROUND_LUMINANCE = 0.48


@dataclass(frozen=True, slots=True)
class TableCellColors:
    """Define readable background and text colors for one numeric table cell."""

    background: str
    foreground: str


_NEUTRAL_CELL_COLORS = TableCellColors(
    background=_NEUTRAL_BACKGROUND_COLOR,
    foreground=_DARK_TEXT_COLOR,
)


def _finite_number(value: object) -> float | None:
    """Return one finite non-boolean numeric value or None."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.number),
    ):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _cell_colors(red: float, green: float, blue: float) -> TableCellColors:
    """Return an opaque blue background with readable contrast text."""
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return TableCellColors(
        background=to_hex((red, green, blue), keep_alpha=False),
        foreground=(_LIGHT_TEXT_COLOR if luminance < _DARK_BACKGROUND_LUMINANCE else _DARK_TEXT_COLOR),
    )


def row_local_color_matrix(
    table: pd.DataFrame,
    *,
    shade_constant: bool = True,
) -> pd.DataFrame:
    """
    Return canonical readable colors from one normalization per numeric row.

    Categorical, identity, boolean, missing, or non-finite rows have no value
    fill. Numerically equal rows use one common neutral fill when requested.
    The caller-owned values are never modified.
    """
    colors = pd.DataFrame(
        [[None] * len(table.columns) for _row in table.index],
        index=table.index,
        columns=table.columns,
        dtype=object,
    )
    colormap = plt.get_cmap("Blues")
    for row_position, (_index, row) in enumerate(table.iterrows()):
        numeric = tuple(_finite_number(value) for value in row)
        if any(value is None for value in numeric):
            continue
        values = np.asarray(numeric, dtype=np.float64)
        lower = float(np.min(values))
        upper = float(np.max(values))
        tolerance = ROW_EQUAL_RELATIVE_TOLERANCE * max(
            abs(lower),
            abs(upper),
            float(np.finfo(np.float64).tiny),
        )
        if upper - lower <= tolerance:
            if shade_constant:
                colors.iloc[row_position, :] = _NEUTRAL_CELL_COLORS
            continue
        for column_position, value in enumerate(values):
            fraction = float((value - lower) / (upper - lower))
            red, green, blue, _alpha = colormap(0.12 + 0.78 * fraction)
            colors.iloc[row_position, column_position] = _cell_colors(
                red,
                green,
                blue,
            )
    return colors


def row_local_style_matrix(
    table: pd.DataFrame,
    *,
    shade_constant: bool = True,
) -> pd.DataFrame:
    """Return CSS styles derived from the canonical row-local color matrix."""
    colors = row_local_color_matrix(
        table,
        shade_constant=shade_constant,
    )
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    for row in range(len(table.index)):
        for column in range(len(table.columns)):
            cell_colors = colors.iloc[row, column]
            if isinstance(cell_colors, TableCellColors):
                styles.iloc[row, column] = f"background-color: {cell_colors.background}; color: {cell_colors.foreground}"
    return styles


def styled_dataframe(
    table: pd.DataFrame,
    *,
    title: str,
    format_spec: str = "{:.4g}",
    shade_constant: bool = True,
    row_local: bool = False,
    heading_level: int = 2,
) -> widgets.VBox:
    """
    Render one analysis table using the maintained blue numeric style.

    Parameters
    ----------
    table : pandas.DataFrame
        Caller-owned table to render without mutation.
    title : str
        Human-readable heading escaped before HTML insertion.
    format_spec : str, optional
        Pandas Styler format applied to numeric columns.
    shade_constant : bool, optional
        Apply a pale fill to constant numeric columns. Disable this when constant
        columns should remain visually neutral.
    row_local : bool, optional
        Normalize each row independently across numeric cells before applying
        coloring.
    heading_level : int, optional
        HTML heading level, from 2 through 4, used for the table title.

    Returns
    -------
    ipywidgets.VBox
        Heading and styled HTML table.

    """
    if not isinstance(table, pd.DataFrame):
        message = "Styled analysis tables require a pandas DataFrame."
        raise TypeError(message)
    if not isinstance(title, str) or not title.strip():
        message = "Styled analysis tables require a non-empty title."
        raise ValueError(message)
    if heading_level not in {2, 3, 4}:
        message = "Styled analysis table headings must use HTML level 2, 3, or 4."
        raise ValueError(message)
    display_table = table.copy(deep=True)
    styles = (
        row_local_style_matrix(
            display_table,
            shade_constant=shade_constant,
        )
        if row_local
        else _blue_numeric_styles(
            display_table,
            shade_constant=shade_constant,
        )
    )

    def format_value(value: object) -> str:
        """Format numeric values even when a mixed table has object columns."""
        number = _finite_number(value)
        return format_spec.format(number) if number is not None else str(value)

    formats: dict[Any, Callable[[object], str]] = dict.fromkeys(
        display_table.columns,
        format_value,
    )
    styler = display_table.style
    styler.format(formats)
    styler.apply(lambda _table: styles, axis=None)
    heading = widgets.HTML(f"<h{heading_level}>{escape(title)}</h{heading_level}>")
    return widgets.VBox(
        (heading, widgets.HTML(styler.to_html())),
        layout=widgets.Layout(width="100%", overflow="auto"),
    )


def grouped_styled_dataframes(
    groups: Sequence[tuple[str, pd.DataFrame]],
    *,
    title: str,
    columns: int = 2,
    format_spec: str = "{:.4g}",
    shade_constant: bool = True,
    row_local: bool = False,
) -> widgets.VBox:
    """
    Render category-grouped tables in compact responsive rows.

    Parameters
    ----------
    groups : Sequence[tuple[str, pandas.DataFrame]]
        Ordered group titles and caller-owned tables.
    title : str
        Shared heading rendered once above all groups.
    columns : int, optional
        Maximum number of tables placed beside one another.
    format_spec, shade_constant, row_local
        Styling options forwarded unchanged to each grouped table.

    Returns
    -------
    ipywidgets.VBox
        Shared heading followed by one or more compact table rows.

    """
    selected = tuple(groups)
    if not selected:
        message = "Grouped analysis tables require at least one table."
        raise ValueError(message)
    if columns <= 0:
        message = "Grouped analysis tables require a positive column count."
        raise ValueError(message)
    rendered = tuple(
        styled_dataframe(
            table,
            title=group_title,
            format_spec=format_spec,
            shade_constant=shade_constant,
            row_local=row_local,
            heading_level=3,
        )
        for group_title, table in selected
    )
    rows = []
    for start in range(0, len(rendered), columns):
        children = rendered[start : start + columns]
        for child in children:
            child.layout = widgets.Layout(
                flex="1 1 0",
                min_width="0",
                overflow="auto",
            )
        rows.append(
            widgets.HBox(
                children,
                layout=widgets.Layout(
                    width="100%",
                    align_items="flex-start",
                    justify_content="space-between",
                ),
            )
        )
    return widgets.VBox(
        (widgets.HTML(f"<h2>{escape(title)}</h2>"), *rows),
        layout=widgets.Layout(width="100%", align_items="stretch"),
    )
