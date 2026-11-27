"""Render coordinated A/B/B-minus-A blocks for generation-input fields.

Responsibilities:
  - Group maintained channels as rows of the shared map grid under three columns headed once by the two case captions
    and B minus A, each row named once at its left and the figure title carrying the scale state
  - Use independent A/B physical scales unless the local scale lock is enabled
  - Give every map its own unit-bearing slot colour bar as tall as the map, and note the range of a difference map whose
    cells share one sign and of an anisotropy map whose cells fill little of its scale, and a bar not written in its row's
    power of ten
  - Compose one lower row under the maps: a distribution per field and an optional side plot, then a column with the key
    of the case and dataset-mean identities above the summary table

Design principles:
  - B-minus-A remains physical, symmetric, and exactly centred at zero; maps, differences and distributions use the
    shared display scales (relative humidity in %, its differences in pp) under the active label style
  - The lower row is as wide as the maps together and never takes part in the map grid; the identities its panels share
    stand once in the key right of them, top-aligned with their frames, and a panel keeps only its own entries
  - A field's distribution, its summary rows and the bars of its map row are written in one power of ten, the one of the
    distribution's value axis; only a wide log bar keeps its decades and a bar whose steps would need more than three
    decimals in that power its own, and their maps say so
  - Summary-table numbers are written with three significant digits, zeros kept, in the row's power of ten (a row with a
    value far from it writes each number with its own remaining power, so one row never mixes notations) and in columns
    wide enough that they never overlap, on unfilled cells, since a figure table has no key that would give a shade a
    meaning; its unit brackets read as the figure label style writes them
  - The two cases of a distribution are unfilled outlines in their own colours, Case B's dashed, so coincident bars or
    uniform-field lines never blend into a colour the key lacks or hide Case A, and the count axis is labelled once, at
    the left of the lower row
  - Incompatible grids are reported without interpolation or resampling

This module does NOT:
  - Render single-case alternatives, derive fields, or load input files
  - Pool unrelated fields or display dataset-mean maps
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Final

import ipywidgets as widgets
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_histograms as histograms
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as shared_layout

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

_MAP_COLUMNS: Final = 3  # case A, case B, B minus A
_ROW_LABEL_WIDTH: Final = 16  # characters per line of a row label: a long field name stays within its map's height
_MATH_SPAN: Final = re.compile(r"\$[^$]*\$")  # a mathtext span, which a row label never splits
_UNBROKEN_SPAN: Final = re.compile(r"\$[^$]*\$|\[[^\]]*\]")  # a mathtext span or a unit bracket, which a panel text never splits
_COMPRESSION_NOTE_FIELDS: Final = frozenset({"K_anisotropy"})  # a ratio whose cells may crowd one end of its state bands
_CASE_LINESTYLES: Final = ("-", "--")  # Case A solid, Case B dashed: a coincident outline or line never hides Case A
_SUMMARY_VALUE_COLUMN_COUNT: Final = 4
# The summary column of the lower row: its longest row label (2.0 in) and its widest number (a negative value with its own
# negative power, such as -1.23e-10 with true minus signs, 0.65 in) each fit their cell inside the cell padding at the table font.
_SUMMARY_WIDTH_IN: Final = 6.3
_SUMMARY_ROW_HEIGHT_IN: Final = 0.34  # every row of the summary table, the two-line header included
_KEY_LINE_IN: Final = 0.22  # one entry of the key above the summary table at the legend font, its spacing included
_KEY_GAP_IN: Final = 0.2  # the key to the summary table
_PANEL_GAP_IN: Final = 0.75  # between the lower row's panels, and before its summary column: y tick labels and a y label
_TITLE_LINE_IN: Final = 0.25  # above the lower row: the second line of a panel title broken to its panel's width
_SUMMARY_LABEL_COLUMN_FRACTION: Final = 0.45
_SUMMARY_VALUE_COLUMN_FRACTION: Final = (1.0 - _SUMMARY_LABEL_COLUMN_FRACTION) / _SUMMARY_VALUE_COLUMN_COUNT
_SUMMARY_COLUMN_WIDTHS: Final = (
    _SUMMARY_LABEL_COLUMN_FRACTION,
    *(_SUMMARY_VALUE_COLUMN_FRACTION,) * _SUMMARY_VALUE_COLUMN_COUNT,
)


@dataclass(frozen=True, slots=True)
class _Identities:
    """Hold the captions and colours of case A and case B and the dataset means drawn beside them, as every lower panel names them."""

    captions: tuple[str, str]
    colors: tuple[str, str]
    means: tuple[tuple[diagnostics.DatasetDiagnostics, str, str], ...]  # (dataset, label, colour) of each drawn mean


def _identities(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
) -> _Identities:
    """Return the identities of one comparison; two cases of one dataset share one mean in the neutral colour."""
    colors = layout.case_colors(mean_a, mean_b)
    means = (
        ((mean_a, f"Dataset mean, n = {mean_a.case_count}", layout.DATASET_MEAN_COLOR),)
        if mean_a.batch_identity == mean_b.batch_identity
        else ((mean_a, f"Mean A, n = {mean_a.case_count}", colors[0]), (mean_b, f"Mean B, n = {mean_b.case_count}", colors[1]))
    )
    return _Identities((layout.case_caption(first, mean_a, "A"), layout.case_caption(second, mean_b, "B")), colors, means)


def _validate_quantities(
    record: diagnostics.GenerationInputDiagnostics,
    quantities: Sequence[str],
) -> tuple[str, ...]:
    """Return one non-empty tuple of profile-available display fields."""
    selected = tuple(quantities)
    if not selected:
        msg = "A spatial comparison block requires at least one field."
        raise ValueError(msg)
    unavailable = tuple(quantity for quantity in selected if quantity not in diagnostics.display_field_names(record))
    if unavailable:
        msg = f"Spatial fields are unavailable for the selected profile: {unavailable}."
        raise ValueError(msg)
    return selected


def _histogram_bounds(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    """Return finite shared histogram bounds with a constant-field margin."""
    combined = np.concatenate((first.ravel(), second.ravel()))
    if not np.isfinite(combined).all():
        msg = "Spatial distributions require finite physical values."
        raise ValueError(msg)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if lower == upper:
        padding = max(abs(lower), 1.0) * 1.0e-6
        lower -= padding
        upper += padding
    return lower, upper


def _fit_to_axis(text: Text, axis: Axes) -> None:
    """Break a panel title or axis label between words, never inside mathtext or a unit, so that no drawn line is wider than its axis."""
    limit = axis.get_window_extent().width
    words = _UNBROKEN_SPAN.sub(lambda match: match.group(0).replace(" ", "\0"), text.get_text()).split(" ")
    lines = [words[0]]
    for word in words[1:]:
        text.set_text(f"{lines[-1]} {word}".replace("\0", " "))
        if text.get_window_extent().width <= limit:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    text.set_text("\n".join(lines).replace("\0", " "))


def _finish_panel(axis: Axes, title: str) -> None:
    """Title one lower-row panel, fit its title and x label to its width, and set its tick size and grid."""
    axis.set_title(title, fontsize=layout.FIGURE_LAYOUT.axis_title_size)
    for text in (axis.title, axis.xaxis.label):
        _fit_to_axis(text, axis)
    axis.tick_params(labelsize=layout.FIGURE_LAYOUT.tick_size)
    axis.grid(alpha=0.22)


def _draw_distribution(
    axis: Axes,
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantity: str,
    identities: _Identities,
    exponent: int,
    *,
    count_label: bool,
) -> None:
    """
    Draw the A/B distributions of one field and its dataset-mean markers on the display scale, written in 10**``exponent``.

    The cases are unfilled outlines in their colours, so coincident bars never blend into a colour the key does not show, and
    Case B's outline or uniform-field line is dashed, so equal counts or values never hide Case A's; the count axis is
    labelled only when ``count_label`` is set, once at the left of the lower row.
    """
    scale = 10.0**exponent
    values = (layout.display_field(first, quantity) / scale, layout.display_field(second, quantity) / scale)
    lower, upper = _histogram_bounds(*values)
    bins = tuple(float(value) for value in np.linspace(lower, upper, 31))
    width = layout.FIGURE_LAYOUT.line_width
    for case_values, color, caption, linestyle in zip(values, identities.colors, identities.captions, _CASE_LINESTYLES, strict=True):
        drawn = histograms.plot_histogram(
            axis, case_values.ravel(), bins=bins, color=color, label=caption, fill=False, edgecolor=color, linewidth=width, linestyle=linestyle
        )
        if drawn.constant_line is not None:  # a uniform field is drawn as one line, which the shared key names as well
            drawn.constant_line.set(label=caption, linestyle=linestyle)
    for dataset, label, color in identities.means:
        mean = diagnostics.display_value(quantity, dataset.field_summary_means[(quantity, "mean")], diagnostics.FIELD_UNITS[quantity])
        axis.axvline(mean / scale, color=color, linestyle="--", linewidth=layout.FIGURE_LAYOUT.line_width, label=label)
    axis.set_xlabel(layout.quantity_axis_label(quantity, exponent), fontsize=layout.FIGURE_LAYOUT.label_size)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)  # the power of ten stands in the unit bracket, never as offset text
    if count_label:
        axis.set_ylabel("Counts", fontsize=layout.FIGURE_LAYOUT.label_size)
    _finish_panel(axis, f"{layout.quantity_label(quantity)} distribution")


def _formatted_table_row(values: Sequence[object], exponent: int) -> tuple[str, ...]:
    """
    Return the values of one figure-table row in 10**``exponent``, the power of ten of its row, with three significant digits.

    A row with a value outside 10^-3 to 10^4 in that power, where a state bar takes a power of its own and the written-out
    digits could overflow the cell, writes each of its numbers with its own remaining power instead, with the true minus sign,
    so that a row keeps one notation: 12345 and 0.5 in a row without factor read 1.23e4 and 5.00e-1. Zero reads 0, a
    non-finite statistic a dash.
    """
    numeric = {
        index: float(value)
        for index, value in enumerate(values)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
    }
    # Rounded as ``significant`` rounds, from the shortest repr in decimal, so notation, power and digits agree: 0.0009995 is 1.00e-3.
    rounded = {index: Decimal(f"{Decimal(repr(number)).scaleb(-exponent):.2e}") for index, number in numeric.items() if np.isfinite(number)}
    scientific = any(field_maps.state_exponent(float(number), float(number)) for number in rounded.values())
    texts = []
    for index, value in enumerate(values):
        number = rounded.get(index)
        if number is None:
            texts.append("\N{EN DASH}" if index in numeric else str(value))
            continue
        own = number.adjusted() if scientific and number else 0  # the power of the rounded number
        text = shared_layout.significant(numeric[index], exponent=exponent + own)
        texts.append(f"{text}e{own}".replace("-", "\N{MINUS SIGN}") if scientific and number else text)
    return tuple(texts)


def _table_row_label(quantity: str, label: str) -> str:
    """Return a table row label with its unit bracket in the figure label style (``[%]``, not the control text's ``[% RH]``)."""
    unit = diagnostics.FIELD_UNITS[quantity]
    control = f"[{diagnostics.display_unit(quantity, unit, control=True)}]"
    return label.replace(control, f"[{diagnostics.display_unit(quantity, unit)}]") if unit else label


def _draw_summary_table(
    axis: Axes,
    table: pd.DataFrame,
    rows: Sequence[tuple[str, int]],
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
) -> None:
    """Draw one compact four-value summary at the top of its axes, each row of a (field, power of ten) pair of ``rows`` written in that power."""
    if not isinstance(table, pd.DataFrame):
        msg = "Spatial summary tables require a pandas DataFrame."
        raise TypeError(msg)
    axis.set_axis_off()
    labels = [_table_row_label(quantity, index[2]) for index, (quantity, _exponent) in zip(table.index, rows, strict=True)]
    cell_text = tuple(
        (
            shared_layout.label_with_exponent(label, exponent) if exponent else label,
            *_formatted_table_row(values, exponent),
        )
        for label, (_quantity, exponent), values in zip(labels, rows, table.itertuples(index=False, name=None), strict=True)
    )
    column_labels = (
        "Summary",
        f"Case {first.case.case_index}\n(A)",
        f"Mean A\n(n={mean_a.case_count})",
        f"Case {second.case.case_index}\n(B)",
        f"Mean B\n(n={mean_b.case_count})",
    )
    figure = axis.get_figure()
    if figure is None:
        msg = "The summary table needs an axis on a figure."
        raise RuntimeError(msg)
    axis_height_in = axis.get_position().height * figure.bbox.height / figure.dpi
    height = min(1.0, (len(cell_text) + 1) * _SUMMARY_ROW_HEIGHT_IN / axis_height_in)
    artist = axis.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=_SUMMARY_COLUMN_WIDTHS,
        bbox=Bbox.from_bounds(0.0, 1.0 - height, 1.0, height),
    )
    artist.auto_set_font_size(False)
    artist.set_fontsize(layout.FIGURE_LAYOUT.table_size)
    for (row, column), cell in artist.get_celld().items():  # value cells stay unfilled: a figure table has no key for a shade
        if row == 0:
            cell.set_facecolor("#e7edf3")
            cell.set_text_props(weight="bold", color="#1f2933")
        elif column == 0:
            cell.set_facecolor("#f5f7fa")
            cell.set_text_props(ha="left", color="#1f2933")


def _row_label(label: str) -> str:
    """Wrap a row label between words to lines of ``_ROW_LABEL_WIDTH`` characters, never inside a mathtext span."""
    protected = _MATH_SPAN.sub(lambda match: match.group(0).replace(" ", "\0"), label)
    return textwrap.fill(protected, _ROW_LABEL_WIDTH, break_long_words=False, break_on_hyphens=False).replace("\0", " ")


def _spatial_incompatibility(
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantities: Sequence[str],
) -> str | None:
    """Return the first exact grid incompatibility across selected fields."""
    for quantity in quantities:
        message = diagnostics.spatial_difference_compatibility(
            first,
            second,
            quantity,
        )
        if message is not None:
            return message
    return None


def _power_text(power: int) -> str:
    """Return how a note writes the power of ten of a bar or row: ``in 10⁻⁵``, and ``without factor`` for 10⁰."""
    return f"in {shared_layout.power_of_ten(power)}" if power else "without factor"


def _bar_note(scale: field_maps.FieldMapDefinition, exponent: int) -> str:
    """Return the note of a bar not written in its row's 10**``exponent`` (``shared_layout.bar_exponent``): at its decades or in its own power."""
    power = shared_layout.bar_exponent(scale, exponent)
    if power == exponent:
        return ""
    return f"Bar {'at decades' if power is None else _power_text(power)} (row {_power_text(exponent)})"


def _draw_comparison_map_row(
    figure: Figure,
    axes: np.ndarray,
    slots: np.ndarray,
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantity: str,
    *,
    exponent: int,
    lock_scale: bool,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> tuple[str, str, str]:
    """
    Draw one actual-case A/B/difference row of the shared map grid, every map with its own unit-bearing colour bar.

    The three bars write their labels in the row's one power of ten ``exponent``, the one of the field's distribution and summary
    rows, except where ``shared_layout.bar_exponent`` names an exception. Return the note of each map: the range of an anisotropy
    map whose cells fill little of its scale, and of a difference whose cells share one sign, which its symmetric scale shows on one
    side only, followed by the exception its bar uses, if any (``_bar_note``).
    """
    difference = field_labels.display_values(
        diagnostics.compatible_field_difference(first, second, quantity),
        diagnostics.FIELD_UNITS[quantity],
        field=quantity,
        quantity_kind="difference",
    )
    first_values, second_values = layout.display_field(first, quantity), layout.display_field(second, quantity)
    first_norm, second_norm = layout.comparison_norms(quantity, first_values, second_values, lock_scale=lock_scale)
    difference_norm = layout.signed_norm(difference)
    unit, difference_unit = layout.quantity_colorbar_label(quantity), layout.quantity_colorbar_label(quantity, difference=True)
    for axis, slot, values, norm, label in (
        (axes[0], slots[0], first_values, first_norm, unit),
        (axes[1], slots[1], second_values, second_norm, unit),
        (axes[2], slots[2], difference, difference_norm, difference_unit),
    ):
        image = layout.draw_field(axis, first, values, norm, display_spatial_stride=display_spatial_stride)
        shared_layout.add_field_map_colorbar(figure, image, slot, norm, label=label, exponent=exponent)
    compressed = quantity in _COMPRESSION_NOTE_FIELDS
    notes = (
        layout.compressed_note(first_values, first_norm) if compressed else "",
        layout.compressed_note(second_values, second_norm) if compressed else "",
        layout.one_sign_note(difference, difference_norm, difference_unit, exponent=exponent),
    )
    first_note, second_note, difference_note = (
        "; ".join(part for part in (note, _bar_note(norm, exponent)) if part)
        for note, norm in zip(notes, (first_norm, second_norm, difference_norm), strict=True)
    )
    return first_note, second_note, difference_note


def _draw_map_rows(
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    quantities: Sequence[str],
    identities: _Identities,
    *,
    exponents: Mapping[str, int],
    below_in: float,
    lock_scale: bool,
    display_spatial_stride: display_grid.DisplayStride,
) -> tuple[Figure, shared_layout.MapGeometry]:
    """
    Draw the A, B and B-minus-A map rows on the shared map grid above a lower band of ``below_in`` and return the figure and its geometry.

    The columns are headed once above the first row by the case captions and each row is named once at its left; a map's
    note stands under its header in the first row and above the map in the others. Each row's bars take its field's power of
    ten of ``exponents``.
    """
    extent = display_grid.physical_extent(first.fields["x"], first.fields["y"])
    geometry = shared_layout.plan_geometry(len(quantities), _MAP_COLUMNS, (extent[1] - extent[0], extent[3] - extent[2]), below_in=below_in)
    figure, axes, slots = shared_layout.map_grid(geometry, extent=extent)
    notes = [
        _draw_comparison_map_row(
            figure,
            axes[row],
            slots[row],
            first,
            second,
            quantity,
            exponent=exponents[quantity],
            lock_scale=lock_scale,
            display_spatial_stride=display_spatial_stride,
        )
        for row, quantity in enumerate(quantities)
    ]
    difference_header = f"Case {second.case.case_index} (B) \N{MINUS SIGN} Case {first.case.case_index} (A)"
    shared_layout.set_column_headers(axes, (*identities.captions, difference_header), notes[0])
    for row, row_notes in enumerate(notes[1:], start=1):
        for column, note in enumerate(row_notes):
            if note:
                shared_layout.add_map_note(axes[row, column], note)
    shared_layout.set_row_labels(axes, [_row_label(layout.quantity_label(quantity)) for quantity in quantities])
    return figure, geometry


def _lower_row(figure: Figure, geometry: shared_layout.MapGeometry, panels: int) -> tuple[list[Axes], Bbox]:
    """Return the lower row's ``panels`` axes of one width and the figure-fraction box of the summary column right of them."""
    left, bottom, width, height = shared_layout.band_rect(geometry, "below")
    panel_width = (geometry.grid_width - _SUMMARY_WIDTH_IN - panels * _PANEL_GAP_IN) / panels
    grid = figure.add_gridspec(
        1,
        panels + 1,
        left=left,
        right=left + width,
        bottom=bottom,
        top=bottom + height - _TITLE_LINE_IN / geometry.figure_height,
        width_ratios=(*(panel_width,) * panels, _SUMMARY_WIDTH_IN),
        wspace=_PANEL_GAP_IN * (panels + 1) / (geometry.grid_width - panels * _PANEL_GAP_IN),  # a fraction of the mean column width
    )
    return [figure.add_subplot(grid[0, column]) for column in range(panels)], grid[0, panels].get_position(figure)


def _dataset_pressure_line(
    dataset: diagnostics.DatasetDiagnostics,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact pointwise empirical inlet-pressure mean."""
    supports_and_values = tuple(diagnostics.inlet_pressure_boundary(record) for record in dataset.records)
    support = supports_and_values[0][0]
    if any(not np.array_equal(other_support, support) for other_support, _values in supports_and_values[1:]):
        msg = "Dataset inlet-pressure mean requires identical persisted x support."
        raise ValueError(msg)
    values = np.mean(
        np.stack(tuple(item[1] for item in supports_and_values)),
        axis=0,
    )
    return support, values


def _draw_pressure_comparison(
    axis: Axes,
    *,
    first: diagnostics.GenerationInputDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    identities: _Identities,
) -> None:
    """Draw the inlet-pressure profiles of both cases and the exact pointwise means of their datasets."""
    support, first_values = diagnostics.inlet_pressure_boundary(first)
    second_support, second_values = diagnostics.inlet_pressure_boundary(second)
    if not np.array_equal(support, second_support):
        msg = "Case pressure-line comparison requires identical persisted x support."
        raise ValueError(msg)
    for values, color, caption in zip((first_values, second_values), identities.colors, identities.captions, strict=True):
        axis.plot(support, values, color=color, linewidth=layout.FIGURE_LAYOUT.line_width, label=caption)
    for dataset, label, color in identities.means:
        mean_support, mean_values = _dataset_pressure_line(dataset)
        if not np.array_equal(mean_support, support):
            msg = "A dataset pressure mean must match the selected case support exactly."
            raise ValueError(msg)
        axis.plot(support, mean_values, color=color, linestyle="--", linewidth=layout.FIGURE_LAYOUT.line_width, label=label)
    axis.set_xlabel("x [m]", fontsize=layout.FIGURE_LAYOUT.label_size)
    axis.set_ylabel(layout.quantity_axis_label("p_in_bc"), fontsize=layout.FIGURE_LAYOUT.label_size)
    _finish_panel(axis, f"{layout.quantity_label('p_in_bc')} profile")


def basic_comparison(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    *,
    lock_scale: bool,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure | widgets.HTML:
    """Render a porosity map row above its distribution, the inlet-pressure profiles, and the summary of both fields."""
    return comparison_block(
        first,
        mean_a,
        second,
        mean_b,
        ("eps_bed",),
        title="Basic spatial inputs",
        lock_scale=lock_scale,
        side_plot=partial(_draw_pressure_comparison, first=first, second=second, identities=_identities(first, mean_a, second, mean_b)),
        summary_quantities=diagnostics.SPATIAL_OVERVIEW_FIELD_NAMES,
        display_spatial_stride=display_spatial_stride,
    )


def comparison_block(
    first: diagnostics.GenerationInputDiagnostics,
    mean_a: diagnostics.DatasetDiagnostics,
    second: diagnostics.GenerationInputDiagnostics,
    mean_b: diagnostics.DatasetDiagnostics,
    quantities: Sequence[str],
    *,
    title: str,
    lock_scale: bool,
    side_plot: Callable[[Axes], None] | None = None,
    summary_quantities: Sequence[str] | None = None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure | widgets.HTML:
    """
    Render A/B/difference map rows above one lower row of a distribution per field and the optional side plot.

    Each map has its own unit-bearing colour bar. Right of the lower panels stands the key of the case and dataset-mean
    identities they share, top-aligned with their frames, and below it the summary of ``summary_quantities`` (default: the
    mapped fields), each row in the power of ten of its field's distribution. The side plot keeps its own entries inside.
    """
    selected = _validate_quantities(first, quantities)
    _validate_quantities(second, selected)
    summarized = selected if summary_quantities is None else _validate_quantities(first, summary_quantities)
    _validate_quantities(second, summarized)
    fields = tuple(dict.fromkeys((*selected, *summarized)))
    incompatibility = _spatial_incompatibility(first, second, fields)
    if incompatibility is not None:
        return widgets.HTML(f"<p><b>Spatial comparison unavailable.</b> {incompatibility} Scalar field summaries remain available.</p>")
    identities = _identities(first, mean_a, second, mean_b)
    table = diagnostics.field_summary_comparison_table(first, mean_a, second, mean_b, quantities=summarized)
    exponents = {
        quantity: layout.histogram_exponent(layout.display_field(first, quantity), layout.display_field(second, quantity)) for quantity in fields
    }
    key_in = (len(identities.captions) + len(identities.means)) * _KEY_LINE_IN + _KEY_GAP_IN
    below_in = _TITLE_LINE_IN + max(layout.FIGURE_LAYOUT.distribution_row_height, key_in + (len(table) + 1) * _SUMMARY_ROW_HEIGHT_IN)
    figure, geometry = _draw_map_rows(
        first,
        second,
        selected,
        identities,
        exponents=exponents,
        below_in=below_in,
        lock_scale=lock_scale,
        display_spatial_stride=display_spatial_stride,
    )
    panels, column = _lower_row(figure, geometry, len(selected) + (side_plot is not None))
    for axis, quantity in zip(panels[: len(selected)], selected, strict=True):
        _draw_distribution(axis, first, second, quantity, identities, exponents[quantity], count_label=axis is panels[0])
    if side_plot is not None:
        side_plot(panels[-1])
    handles, labels = figure_layout.legend_entries(panels)
    figure.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(column.x0, column.y1),
        bbox_transform=figure.transFigure,
        frameon=False,
        borderaxespad=0.0,
        fontsize=layout.FIGURE_LAYOUT.legend_size,
    )
    table_axis = figure.add_axes((column.x0, column.y0, column.width, column.height - key_in / geometry.figure_height))
    row_fields = [quantity for quantity, _statistic in diagnostics.field_summary_specs(first) if quantity in summarized]
    _draw_summary_table(table_axis, table, [(quantity, exponents[quantity]) for quantity in row_fields], first, mean_a, second, mean_b)
    scale_state = "locked A/B scales" if lock_scale else "independent A/B scales"
    shared_layout.set_map_title(figure, f"{title} — {scale_state}")
    return figure
