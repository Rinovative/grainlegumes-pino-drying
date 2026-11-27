"""Render reusable styled DataFrame tables for analysis notebooks.

Responsibilities:
  - Format numeric table values consistently for notebook display
  - Shade only the columns or rows a caller opts into, with one key naming the shading, never a text or boolean cell
  - Hide a positional index and the axis names, so no index metadata forms a header row or corner of its own
  - Derive reusable readable row-local colors for HTML and figure tables
  - Compose compact single or grouped table widget layouts with safe titles

Design principles:
  - Table styling is presentation-only
  - Shading is opt-in: only the caller knows which values are comparable and which are identities
  - Callers retain ownership of table content, grouping, and semantic labels

This module does NOT:
  - Compute scientific statistics or choose displayed columns
  - Compose notebook sections or manage figure-export state
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


def _blue_numeric_styles(table: pd.DataFrame, *, shade_constant: bool, columns: Sequence[str]) -> pd.DataFrame:
    """Return quantile-bounded blue fills for the finite numeric cells of ``columns``, column-local; text and booleans stay unfilled."""
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    colormap = plt.get_cmap("Blues")
    for column in columns:
        numeric = np.array([np.nan if (number := _finite_number(value)) is None else number for value in table[column]], dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            continue
        lower, upper = np.quantile(finite, (0.05, 0.95))
        constant = bool(np.isclose(lower, upper, atol=0.0))  # relative only: a column of tiny values still varies
        if constant and not shade_constant:
            continue
        for row, value in enumerate(numeric):
            if not np.isfinite(value):
                continue
            fraction = 0.0 if constant else float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
            red, green, blue, _alpha = colormap(0.05 + 0.90 * fraction)
            column_index = cast("int", styles.columns.get_loc(column))
            styles.iloc[row, column_index] = f"background-color: rgba({int(red * 255)}, {int(green * 255)}, {int(blue * 255)}, 0.55)"
    return styles


ROW_EQUAL_RELATIVE_TOLERANCE = 1.0e-9
_SHADING_KEY = '<p style="margin: 0 0 6px 0; color: #52606d">Shading compares the cells of each {}: darker means larger.</p>'
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
    colors: list[list[TableCellColors | None]] = [[None for _column in table.columns] for _row in table.index]
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
                colors[row_position] = [_NEUTRAL_CELL_COLORS for _column in table.columns]
            continue
        for column_position, value in enumerate(values):
            fraction = float((value - lower) / (upper - lower))
            red, green, blue, _alpha = colormap(0.12 + 0.78 * fraction)
            colors[row_position][column_position] = _cell_colors(
                red,
                green,
                blue,
            )
    return pd.DataFrame(
        colors,
        index=table.index,
        columns=table.columns,
        dtype=object,
    )


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


def _render_table(
    table: pd.DataFrame,
    *,
    title: str,
    format_spec: str,
    shade_constant: bool,
    row_local: bool,
    heading_level: int,
    shade_columns: Sequence[str] | None,
    keyed: bool,
    row_heading: str | None = None,
) -> tuple[widgets.VBox, bool]:
    """Return the heading and table of ``styled_dataframe`` and whether a cell is shaded; ``keyed`` writes the shading key."""
    if not isinstance(table, pd.DataFrame):
        message = "Styled analysis tables require a pandas DataFrame."
        raise TypeError(message)
    if not isinstance(title, str) or not title.strip():
        message = "Styled analysis tables require a non-empty title."
        raise ValueError(message)
    if heading_level not in {2, 3, 4}:
        message = "Styled analysis table headings must use HTML level 2, 3, or 4."
        raise ValueError(message)
    if shade_columns is not None and row_local:
        message = "shade_columns applies to column-local fills and cannot be combined with row_local."
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
            # A column shades against its own values: a single row has nothing to compare and stays plain.
            columns=(shade_columns or ()) if len(display_table) > 1 else (),
        )
    )

    def format_value(value: object) -> str:
        """Format numeric values even when a mixed table has object columns."""
        number = _finite_number(value)
        return format_spec.format(number) if number is not None else str(value)

    formats: dict[Any, str | Callable[[object], str] | None] = dict.fromkeys(
        display_table.columns,
        format_value,
    )
    index = display_table.index
    positions = index.nlevels == 1 and index.name is None and pd.api.types.is_integer_dtype(index.dtype)
    # Only the caller's row heading heads the corner cell; a pandas axis name forms no header row or cell of its own.
    shown = display_table if row_heading is None else display_table.rename_axis(index=None, columns=row_heading)
    styler = shown.style
    styler.format(formats)
    styler.apply(lambda _table: styles, axis=None)
    styler.hide(axis="index", names=True)
    if row_heading is None:
        styler.hide(axis="columns", names=True)
    if positions:
        styler.hide(axis="index")  # row positions, not row labels
    shaded = bool((styles != "").to_numpy().any())
    key = _SHADING_KEY.format("row" if row_local else "column") if keyed and shaded else ""
    heading = widgets.HTML(f"<h{heading_level}>{escape(title)}</h{heading_level}>{key}")
    return widgets.VBox(
        (heading, widgets.HTML(styler.to_html())),
        layout=widgets.Layout(width="100%", overflow="auto"),
    ), shaded


def styled_dataframe(
    table: pd.DataFrame,
    *,
    title: str,
    format_spec: str = "{:.4g}",
    shade_constant: bool = True,
    row_local: bool = False,
    heading_level: int = 2,
    shade_columns: Sequence[str] | None = None,
    row_heading: str | None = None,
) -> widgets.VBox:
    """
    Render one analysis table, shaded in the maintained blue style only where the caller opts in.

    A positional index (unnamed integers) is hidden, named row labels stay, and no pandas axis name forms a header row or
    corner cell of its own; only ``row_heading`` heads the corner cell. A shaded table carries a key under its heading. Text
    and boolean cells are never shaded.

    Parameters
    ----------
    table : pandas.DataFrame
        Caller-owned table to render without mutation.
    title : str
        Human-readable heading escaped before HTML insertion.
    format_spec : str, optional
        Pandas Styler format applied to numeric columns.
    shade_constant : bool, optional
        Apply a pale fill to constant numeric columns or rows that are shaded. Disable this when constant ones should
        remain visually neutral.
    row_local : bool, optional
        Shade each row, normalized independently across its numeric cells.
    heading_level : int, optional
        HTML heading level, from 2 through 4, used for the table title.
    shade_columns : Sequence[str], optional
        Shade these columns, each against its own values, when the table has at least two rows; the default shades none. Not
        combined with ``row_local``.
    row_heading : str, optional
        What the rows are («Dataset», «Model»), written in the corner cell above the row labels of a single-level table.

    Returns
    -------
    ipywidgets.VBox
        Heading, with the shading key when a cell is shaded, and the styled HTML table.

    """
    return _render_table(
        table,
        title=title,
        format_spec=format_spec,
        shade_constant=shade_constant,
        row_local=row_local,
        heading_level=heading_level,
        shade_columns=shade_columns,
        keyed=True,
        row_heading=row_heading,
    )[0]


def grouped_styled_dataframes(
    groups: Sequence[tuple[str, pd.DataFrame]],
    *,
    title: str,
    columns: int = 2,
    format_spec: str = "{:.4g}",
    shade_constant: bool = True,
    row_local: bool = False,
    shade_columns: Sequence[str] | None = None,
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
    format_spec, shade_constant, row_local, shade_columns
        Styling options forwarded unchanged to each grouped table.

    Returns
    -------
    ipywidgets.VBox
        Shared heading, with the shading key once when a cell is shaded, followed by one or more compact table rows.

    """
    selected = tuple(groups)
    if not selected:
        message = "Grouped analysis tables require at least one table."
        raise ValueError(message)
    if columns <= 0:
        message = "Grouped analysis tables require a positive column count."
        raise ValueError(message)
    results = [
        _render_table(
            table,
            title=group_title,
            format_spec=format_spec,
            shade_constant=shade_constant,
            row_local=row_local,
            heading_level=3,
            shade_columns=shade_columns,
            keyed=False,
        )
        for group_title, table in selected
    ]
    rendered = tuple(box for box, _shaded in results)
    key = _SHADING_KEY.format("row" if row_local else "column") if any(shaded for _box, shaded in results) else ""
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
        (widgets.HTML(f"<h2>{escape(title)}</h2>{key}"), *rows),
        layout=widgets.Layout(width="100%", align_items="stretch"),
    )
