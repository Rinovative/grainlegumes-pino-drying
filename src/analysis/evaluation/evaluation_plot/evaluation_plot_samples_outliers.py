"""Render case, permeability, outlier, and input-extreme views.

Responsibilities:
  - Rank explicit predictive metrics with canonical identity ties
  - Rank supported scientific metadata independently at low and high extremes
  - Render one case's reference, prediction, and selected error of the displayed fields (every learned field, or a ``fields``
    selection) on the shared fixed map grid, every panel on discrete round colour levels
  - Render fixed permeability and two-model field layouts
  - Overlay velocity streamlines on regular display-only physical coordinates, integrated once per drawn field
  - Build colored outlier and side-by-side metadata-extreme tables

Design principles:
  - Case selection follows exact saved membership and canonical case identity
  - TaskSpec fields and vector groups define p, u, v, and |u| presentation
  - Sample titles name material, role, compact case and, in a comparison, the model from the frame's own identity
  - State levels cover the 5th to the 95th percentile of the reference, error levels follow the shared percentile rule,
    arrows mark values beyond; non-finite prediction cells stay masked and are noted
  - Permeability remains physical context and is never an error aggregate
  - Scaling, clipping, contouring, and regular coordinates affect display only

This module does NOT:
  - Parse unvalidated payloads or infer case identity from filenames
  - Compare models by row position without matching identity and reference fields
  - Redefine predictive, physics, or metadata-ranking semantics
  - Own interactive notebook controls or public panel composition
"""

from __future__ import annotations

from html import escape
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import FancyArrowPatch

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_run_summary as run_summary
from src.analysis.presentation import analysis_dataset_selection as dataset_selection
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_plot_layout as map_layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas.io.formats.style import Styler

_DEFAULT_TOP_K = 5
_MINIMUM_COMPARISON_MODELS = 2
_CONTOUR_LEVELS = 10
_MINIMUM_TABLE_NUMERIC_ROWS = 2
_MINIMUM_VELOCITY_COMPONENTS = 2
_OVERVIEW_COLUMNS = 3  # reference, prediction, error
_OVERVIEW_ERROR_HEADERS = {
    error_diagnostics.SIGNED_ERROR: "Signed error",
    "MAE": "Absolute error",
    "Relative [%]": "Relative error",
    error_diagnostics.NORMALIZED_ABSOLUTE_ERROR: error_diagnostics.NORMALIZED_ERROR_HEADER,
}
_RELATIVE_ERROR_COLORMAP = "magma"  # the field-map relative-error colormap the steady maps used before
_VELOCITY_COLORMAP = "turbo"  # owner 14.11.: the velocity components' colormap before the plot standard rework
_MASKED_NOTE = "Non-finite cells masked"  # the transient state maps' wording
_STREAMLINE_STYLE: dict[str, Any] = {"color": "white", "linewidth": 0.55}
_STREAMLINE_ARROW_SIZE = 0.55
_STANDARD_NUMBER_LOWER = 1e-4
_STANDARD_NUMBER_UPPER = 1e4


def _rank_positions(frame: pd.DataFrame, values: np.ndarray, *, descending: bool) -> np.ndarray:
    """Return deterministic value order with canonical case/source tie breaking."""
    numeric = np.asarray(values, dtype=float)
    if numeric.shape != (len(frame),) or not np.isfinite(numeric).all():
        msg = "Ranked values must be a finite vector aligned with artifact membership."
        raise ValueError(msg)
    primary = -numeric if descending else numeric
    return np.lexsort(
        (
            frame["source_index"].to_numpy(dtype=np.int64),
            frame["case_index"].to_numpy(dtype=np.int64),
            primary,
        )
    )


def _fmt_number(value: Any) -> str:
    """Format one table value with the compact numerical style."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    return f"{number:.3g}" if _STANDARD_NUMBER_LOWER <= abs(number) < _STANDARD_NUMBER_UPPER else f"{number:.2e}"


def _rank_style(table: pd.DataFrame, *, reference_row: int) -> pd.DataFrame:
    """Return blue/purple cell fills diverging from the reference."""
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    blue = colormaps["Blues"]
    purple = colormaps["Purples"]
    for column in table.columns:
        source = table[column]
        if not isinstance(source, pd.Series):
            msg = "Rank styling requires unique scalar-valued table columns."
            raise TypeError(msg)
        numeric = pd.to_numeric(source, errors="coerce")
        if not isinstance(numeric, pd.Series):
            msg = "Rank styling requires a numeric pandas Series."
            raise TypeError(msg)
        if numeric.notna().sum() < _MINIMUM_TABLE_NUMERIC_ROWS or pd.isna(numeric.iloc[reference_row]):
            continue
        reference = float(numeric.iloc[reference_row])
        lower = numeric < reference
        higher = numeric > reference
        for mask, cmap in ((lower, blue), (higher, purple)):
            selected = numeric[mask]
            if not isinstance(selected, pd.Series):
                msg = "Rank styling requires a pandas Series after boolean selection."
                raise TypeError(msg)
            if selected.empty:
                continue
            ranks = selected.rank(method="average", pct=True)
            for index, rank in ranks.items():
                red, green, blue_value, _alpha = cmap(0.25 + 0.5 * float(rank))
                styles.loc[index, column] = f"background-color: rgba({int(red * 255)}, {int(green * 255)}, {int(blue_value * 255)}, 0.65)"
    styles.iloc[reference_row] = "background-color: white"
    return styles


def _channel_outlier_styler(frame: pd.DataFrame, field: presentation.DisplayField, *, top_k: int) -> Styler:
    """Build one worst-five-plus-reference channel table."""
    values = np.asarray(pd.to_numeric(frame[field.metric_column], errors="raise"), dtype=float)
    positions = list(_rank_positions(frame, values, descending=True)[:top_k])
    reference_position = presentation.reference_case_position(frame)
    parameters = presentation.metadata_parameters((frame,))
    rows: list[dict[str, Any]] = []
    for position in (*positions, reference_position):
        source = frame.iloc[int(position)]
        is_reference = int(position) == reference_position
        row: dict[str, Any] = {
            "Case": f"{int(source['case_index'])} (Ref)" if is_reference else int(source["case_index"]),
            f"normalized RMSE [{field.label}]": float(source[field.metric_column]),
            "relative L2": float(source["physical_relative_l2"]),
        }
        row.update({presentation.metadata_label(column): float(source[column]) for column in parameters})
        rows.append(row)
    table = pd.DataFrame(rows)
    reference_row = len(table) - 1
    styler = table.style
    styler.format(_fmt_number)
    styler.apply(_rank_style, axis=None, reference_row=reference_row)
    return styler


def plot_outlier_table(
    *,
    datasets: Mapping[str, pd.DataFrame],
    top_k: int = _DEFAULT_TOP_K,
) -> widgets.VBox:
    """Return automatically rendered colored channel outlier tables."""
    dataframe.validate_comparison(datasets)
    if len(datasets) != 1:
        msg = "The outlier-table view displays one selected dataset."
        raise ValueError(msg)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        msg = "top_k must be a positive integer."
        raise ValueError(msg)
    label, frame = next(iter(datasets.items()))
    title = layout.case_title("Worst per-channel cases", case_count=len(frame))
    children: list[widgets.Widget] = [widgets.HTML(f"<h2>{title}</h2>"), widgets.HTML(f"<h3>{escape(label)}</h3>")]
    for field in presentation.display_fields(frame):
        styler = _channel_outlier_styler(frame, field, top_k=top_k)
        children.append(widgets.HTML(f"<h4>Channel {escape(field.label)}</h4>{styler.to_html()}"))
    return widgets.VBox(children)


def plot_input_extremes_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Return side-by-side min/reference/max metadata tables."""
    dataframe.validate_comparison(datasets)
    parameters = presentation.metadata_parameters(tuple(datasets.values()))
    if not parameters:
        msg = "No supported metadata parameters are available."
        raise ValueError(msg)
    title, count_headings = layout.aggregate_title_context(
        "Extreme input parameters",
        layout.effective_case_counts(datasets),
    )
    tables: list[widgets.Widget] = []
    for label, frame in datasets.items():
        reference = frame.iloc[presentation.reference_case_position(frame, parameters)]
        rows = []
        for parameter in parameters:
            values = np.asarray(pd.to_numeric(frame[parameter], errors="raise"), dtype=float)
            reference_value = float(reference[parameter])
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            rows.append(
                {
                    "Parameter": presentation.metadata_label(parameter),
                    "Min": minimum,
                    "Reference": reference_value,
                    "Max": maximum,
                    "Min / ref": minimum / reference_value if reference_value != 0.0 else np.nan,
                    "Max / ref": maximum / reference_value if reference_value != 0.0 else np.nan,
                }
            )
        table = pd.DataFrame(rows).set_index("Parameter")
        heading = count_headings[label] or label
        styler = table.style
        styler.format(_fmt_number)
        tables.append(widgets.HTML(f"<h3>{escape(heading)}</h3>{styler.to_html()}"))
    return widgets.VBox((widgets.HTML(f"<h2>{title}</h2>"), widgets.HBox(tables)))


def _selected_position(frame: pd.DataFrame, row_position: int) -> int:
    """Validate one zero-based saved-membership position."""
    if isinstance(row_position, bool) or not isinstance(row_position, Integral):
        msg = "row_position must be an integer."
        raise TypeError(msg)
    position = int(row_position)
    if not 0 <= position < len(frame):
        msg = "row_position is outside the artifact membership."
        raise IndexError(msg)
    return position


def _single_dataset(datasets: Mapping[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Return the one dataset selected by a local dropdown."""
    dataframe.validate_comparison(datasets)
    if len(datasets) != 1:
        msg = "This case view requires one selected dataset."
        raise ValueError(msg)
    return next(iter(datasets.items()))


def _levels(values: np.ndarray, *, count: int = _CONTOUR_LEVELS, nonnegative: bool = False) -> np.ndarray:
    """Return finite stable contour levels without changing field values."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        msg = "A displayed field contains no finite values."
        raise ValueError(msg)
    low = 0.0 if nonnegative else float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high):
        delta = max(abs(low), 1.0) * np.finfo(float).eps * 16.0
        low = 0.0 if nonnegative else low - delta
        high += delta
    return np.linspace(low, high, count)


def _format_axes(axis: Axes, case: cases.EvaluationCase) -> None:
    """Apply exact physical extents and geometry without changing labels."""
    x_values, y_values = case.coordinates
    axis.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
    axis.set_ylim(float(np.min(y_values)), float(np.max(y_values)))
    axis.set_aspect("equal", adjustable="box")


def _clip_to_levels(values: np.ndarray, levels: np.ndarray) -> np.ma.MaskedArray:
    """Saturate finite out-of-range values without creating white masked bands."""
    return np.ma.clip(np.ma.asarray(values), float(levels[0]), float(levels[-1]))


def _contour(
    figure: Figure,
    axis: Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    *,
    levels: np.ndarray,
    cmap: str,
    title: str,
    unit: str,
    case: cases.EvaluationCase,
    display_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> Any:
    """Draw one filled-contour panel and its unit-labelled colorbar."""
    displayed = _clip_to_levels(values, levels)
    if display_indices is not None:
        x_grid = display_grid.display_sample(x_grid, display_indices)
        y_grid = display_grid.display_sample(y_grid, display_indices)
        displayed = np.ma.asarray(display_grid.display_sample(displayed, display_indices))
    image = axis.contourf(x_grid, y_grid, displayed, levels=levels, cmap=cmap)
    axis.set_title(title)
    layout.add_map_colorbar(figure, image, axis, label=f"[{field_labels.display_unit(unit)}]")
    _format_axes(axis, case)
    return image


def _velocity_indices(frame: pd.DataFrame, case: cases.EvaluationCase) -> tuple[int, int]:
    """Return the exact first two TaskSpec velocity-component indices."""
    fields = dataframe.output_group_fields(frame, group_id="velocity")
    if len(fields) < _MINIMUM_VELOCITY_COMPONENTS or any(field not in case.fields for field in fields[:_MINIMUM_VELOCITY_COMPONENTS]):
        msg = "Velocity streamlines require two declared components."
        raise dataframe.ComparisonCompatibilityError(msg)
    return case.fields.index(fields[0]), case.fields.index(fields[1])


def _overlay_streamlines(
    axis: Axes,
    frame: pd.DataFrame,
    case: cases.EvaluationCase,
    values: np.ndarray,
    *,
    display_indices: tuple[np.ndarray, np.ndarray],
    segments: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """
    Overlay velocity streamlines on the authoritative equally spaced display coordinates and return their segments.

    An endpoint that the display stride appends would break the equal spacing streamlines need, so it stays off their
    grid. Given ``segments`` (of the same field on another panel) are drawn again instead of integrating anew, each
    with one arrow at its middle as ``streamplot`` places it.
    """
    if segments is None:
        first, second = _velocity_indices(frame, case)
        x, y, _x_grid, _y_grid = presentation.display_grid(case)
        y_indices, x_indices = (index[:-1] if np.unique(np.diff(index)).size > 1 else index for index in display_indices)
        grid = np.ix_(y_indices, x_indices)
        stream = axis.streamplot(
            x[x_indices], y[y_indices], values[first][grid], values[second][grid], density=0.65, arrowsize=_STREAMLINE_ARROW_SIZE, **_STREAMLINE_STYLE
        )
        return stream.lines.get_segments()
    axis.add_collection(LineCollection(segments, **_STREAMLINE_STYLE))
    for points in segments:
        distance = np.cumsum(np.hypot(*np.diff(points, axis=0).T))
        middle = int(np.searchsorted(distance, distance[-1] / 2.0))
        tail, head = points[middle], points[middle : middle + 2].mean(axis=0)
        axis.add_patch(FancyArrowPatch(tail, head, arrowstyle="-|>", mutation_scale=10.0 * _STREAMLINE_ARROW_SIZE, **_STREAMLINE_STYLE))
    return segments


def _require_error_mode(error_mode: str) -> error_diagnostics.ErrorMode:
    """Require one current error-presentation mode of the case overviews."""
    if error_mode not in error_diagnostics.STEADY_ERROR_MODES:
        msg = f"error_mode must be one of {error_diagnostics.STEADY_ERROR_MODES}."
        raise ValueError(msg)
    return cast("error_diagnostics.ErrorMode", error_mode)


def _require_scale_mode(scale_mode: str) -> str:
    """Require one current prediction/reference scale mode."""
    if scale_mode not in {"Independent", "Shared (GT)"}:
        msg = "scale_mode must be 'Independent' or 'Shared (GT)'."
        raise ValueError(msg)
    return scale_mode


def _error_values(
    reference: np.ndarray, prediction: np.ndarray, *, error_mode: str, frame: pd.DataFrame, field: presentation.DisplayField
) -> error_diagnostics.SelectedErrorMap:
    """Compute only the selected existing-location field error."""
    mode = _require_error_mode(error_mode)
    scale = None
    if mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        scale = error_diagnostics.steady_reference_scale(dataframe.require_complete_provenance(frame), field=field.key, unit=field.unit)
    floor = dataframe.reference_floor(frame, field.component_fields) if mode == "Relative [%]" else None
    return error_diagnostics.selected_error_map(
        reference, prediction, mode=mode, field=field.key, physical_unit=field.unit, scale=scale, reference_floor=floor
    )


def _state_boundaries(values: np.ndarray) -> np.ndarray | None:
    """
    Return round boundaries covering the 5th to the 95th percentile of the finite ``values``; ``None`` without any.

    A degenerate percentile range, such as a constant field, falls back to the full finite range (``layout.state_boundaries``).
    """
    scale = field_maps.percentile_scale(values, "state")
    if scale is None:
        return None
    finite = values[np.isfinite(values)]
    return layout.state_boundaries(*(scale if scale[1] > scale[0] else (float(np.min(finite)), float(np.max(finite)))))


def _discrete_colours(boundaries: np.ndarray, colormap: str, values: np.ndarray) -> tuple[dict[str, Any], field_maps.FieldMapExtend]:
    """Return one colour per band between ``boundaries`` and the bar's arrows at the ends that the finite ``values`` pass."""
    finite = values[np.isfinite(values)]
    below, above = bool(np.any(finite < boundaries[0])), bool(np.any(finite > boundaries[-1]))
    extend: field_maps.FieldMapExtend = "both" if below and above else "min" if below else "max" if above else "neither"
    cmap = colormaps[colormap].resampled(len(boundaries) - 1)
    return {"cmap": cmap, "norm": BoundaryNorm(boundaries, cmap.N)}, extend


def _error_scale(result: error_diagnostics.SelectedErrorMap) -> tuple[np.ndarray, str, str]:
    """
    Return the boundaries, colormap and colour-bar label of one available steady error map.

    The signed error runs symmetric to the larger absolute one of its 5th and 95th percentile on a blue-white-red scale,
    every other mode from 0 to its 95th percentile; errors beyond get the bar's arrows.
    """
    unit = f"[{field_labels.display_unit(result.unit, quantity_kind='difference')}]"
    signed = result.label == error_diagnostics.SIGNED_ERROR
    semantics = visual_semantics.field_visual_semantics("error", role="signed_error" if signed else "absolute_error")
    colormap = _RELATIVE_ERROR_COLORMAP if result.unit == "%" else semantics.colormap
    return (
        layout.error_boundaries(np.asarray(result.map_values, dtype=float), signed=signed),
        colormap,
        f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} {unit}" if result.scale is not None else unit,
    )


def _error_contour(
    figure: Figure,
    axis: Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    result: error_diagnostics.SelectedErrorMap,
    *,
    field: presentation.DisplayField,
    case: cases.EvaluationCase,
    display_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Render one selected error on its round discrete levels, or its reason, in the existing axis."""
    axis.set_title(f"{field.matplotlib_label} {result.label.removesuffix(' [%]')}")
    if result.map_values is None:
        axis.text(0.5, 0.5, f"Unavailable: {result.reason}", ha="center", va="center", wrap=True, transform=axis.transAxes)
        _format_axes(axis, case)
        return
    boundaries, colormap, label = _error_scale(result)
    colours, extend = _discrete_colours(boundaries, colormap, result.map_values)
    map_values = result.map_values
    if display_indices is not None:
        x_grid = display_grid.display_sample(x_grid, display_indices)
        y_grid = display_grid.display_sample(y_grid, display_indices)
        map_values = display_grid.display_sample(map_values, display_indices)
    image = axis.contourf(x_grid, y_grid, map_values, levels=boundaries, extend=extend, **colours)
    colorbar = layout.add_map_colorbar(figure, image, axis, label=label)
    layout.set_fixed_decimal_ticks(colorbar, layout.error_ticks(boundaries, signed=result.label == error_diagnostics.SIGNED_ERROR))
    _format_axes(axis, case)


def _resolve_display_field(frame: pd.DataFrame, field: str | None) -> presentation.DisplayField:
    """Resolve one canonical key or concise display label."""
    available = presentation.display_fields(frame)
    if field is None:
        return available[-1]
    for candidate in available:
        aliases = {candidate.key, candidate.label}
        if candidate.key == "velocity_magnitude":
            aliases.add("U")
        if field in aliases:
            return candidate
    msg = f"Unknown display field {field!r}; expected {[item.label for item in available]}."
    raise ValueError(msg)


def _model_display_name(frame: pd.DataFrame) -> str:
    """Return the architecture label of the frame's model, ``PI-`` prefixed when it trained with physics loss terms."""
    family, _trainable, physics_enabled = run_summary.model_metadata(frame)
    return visual_semantics.architecture_label(f"pi-{family}" if physics_enabled else family)


def _sample_title(frame: pd.DataFrame, position: int, label: str, context: str | None) -> str:
    """
    Return material · role · model · compact case like the transient sample viewer, then an optional selection context.

    Like the transient title, the model is named only in a comparison: a single-model panel keys the drawn frame by its
    own dataset label, a comparison by the model, so the model is named unless the frame's ``label`` is that dataset label.
    It is named by its display name, such as ``PI-U-NO``, never by the run's configuration name.
    """
    row = frame.iloc[position]
    material, case_id = row.get("material_family"), row.get("source_case_id")
    if not isinstance(material, str) or not material or not isinstance(case_id, str) or not case_id:
        msg = "A sample title requires the row's material_family and source_case_id metadata."
        raise dataframe.ComparisonCompatibilityError(msg)
    role = dataframe.dataset_role(frame)
    model = None if label == dataset_selection.dataset_label(material, role) else _model_display_name(frame)
    parts = (display_labels.material_display_label(material), role, model, case_id, context)
    return " · ".join(part for part in parts if part is not None)


def _draw_map(
    figure: Figure,
    axis: Axes,
    slot: Axes,
    grids: tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray],
    *,
    boundaries: np.ndarray | None,
    colormap: str,
    label: str,
    signed: bool = False,
) -> None:
    """
    Draw one map on discrete round ``boundaries`` with its slot bar, non-finite cells masked and noted; ``None`` states no finite values.

    The bar labels the boundaries, a ``signed`` one the band centres, at one decimal (more only for a step below 0.1).
    """
    if boundaries is None:
        axis.text(0.5, 0.5, "unavailable: no finite values", ha="center", va="center", transform=axis.transAxes)
        return
    colours, extend = _discrete_colours(boundaries, colormap, values)
    image = axis.pcolormesh(*grids, np.ma.masked_invalid(display_grid.display_sample(values, indices)), shading="auto", **colours)
    if not np.isfinite(values).all():
        axis.text(0.02, 0.02, _MASKED_NOTE, transform=axis.transAxes, fontsize=8)
    colorbar = map_layout.add_slot_colorbar(figure, image, slot, label=label, extend=extend)
    layout.set_fixed_decimal_ticks(colorbar, layout.error_ticks(boundaries, signed=signed))


def _plot_prediction_overview(
    *,
    label: str,
    frame: pd.DataFrame,
    row_position: int,
    error_mode: str,
    context: str | None,
    display_spatial_stride: display_grid.DisplayStride,
    fields: Sequence[str] | None = None,
) -> Figure:
    """
    Plot reference, prediction, and the selected error of the displayed fields on the shared fixed map grid.

    By default these are the learned fields except the velocity magnitude, else the given ``fields`` (such as ("p", "U")).

    Every state panel has discrete round levels over the 5th to the 95th percentile of the reference; the prediction takes
    the reference's levels, and arrows mark the values beyond them. The error panel's levels follow the error's own
    percentile rule (``_error_scale``). Velocity components are drawn on the turbo colormap with streamlines of the drawn
    field pair, integrated once per column; the velocity magnitude ``U`` keeps its own colormap and gets the same
    streamlines. Only the error maps carry a metric, one mean |e| line above each; one note line under the title names the
    scale rule.
    """
    position = _selected_position(frame, row_position)
    case = cases.load_case(frame, position)
    if fields is None:
        shown = tuple(field for field in presentation.display_fields(frame) if not field.is_magnitude)
    else:
        shown = tuple(_resolve_display_field(frame, name) for name in fields)
    error = _require_error_mode(error_mode)
    title = _sample_title(frame, position, label, context)
    velocity = dataframe.output_group_fields(frame, group_id="velocity")
    _x, _y, x_grid, y_grid = presentation.display_grid(case)
    indices = display_grid.display_indices(x_grid.shape, display_spatial_stride)
    grids = (display_grid.display_sample(x_grid, indices), display_grid.display_sample(y_grid, indices))
    x_values, y_values = case.coordinates
    extent = (float(np.min(x_values)), float(np.max(x_values)), float(np.min(y_values)), float(np.max(y_values)))
    geometry = map_layout.plan_geometry(len(shown), _OVERVIEW_COLUMNS, (extent[1] - extent[0], extent[3] - extent[2]))
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    streamlines: dict[int, list[np.ndarray]] = {}  # per reference and prediction column, reused on every velocity row
    error_texts = []
    for row, field in enumerate(shown):
        reference = presentation.case_field(case, field, source="reference")
        prediction = presentation.case_field(case, field, source="prediction")
        colormap = _VELOCITY_COLORMAP if field.key in velocity else visual_semantics.field_visual_semantics(field.key).colormap
        reference_boundaries = _state_boundaries(reference)
        unit = f"[{field_labels.display_unit(field.unit)}]"
        for column, (values, tensor) in enumerate(((reference, case.reference), (prediction, case.prediction))):
            _draw_map(
                figure, axes[row, column], slots[row, column], grids, values, indices, boundaries=reference_boundaries, colormap=colormap, label=unit
            )
            if set(field.component_fields) <= set(velocity):  # a velocity component or the magnitude of the pair
                streamlines[column] = _overlay_streamlines(
                    axes[row, column], frame, case, tensor, display_indices=indices, segments=streamlines.get(column)
                )
        result = _error_values(reference, prediction, error_mode=error, frame=frame, field=field)
        error_unit = None if result.scale is not None else field_labels.display_unit(result.unit, quantity_kind="difference")
        if result.map_values is None:
            error_texts.append(layout.mean_error_text(None, unit=error_unit))
            axes[row, -1].text(0.5, 0.5, f"Unavailable: {result.reason}", ha="center", va="center", wrap=True, transform=axes[row, -1].transAxes)
            continue
        boundaries, error_colormap, error_label = _error_scale(result)
        exponent = layout.error_format(boundaries, signed=error == error_diagnostics.SIGNED_ERROR).exponent
        error_texts.append(layout.mean_error_text(result.map_values, unit=error_unit, exponent=exponent))
        _draw_map(
            figure,
            axes[row, -1],
            slots[row, -1],
            grids,
            result.map_values,
            indices,
            boundaries=boundaries,
            colormap=error_colormap,
            label=error_label,
            signed=error == error_diagnostics.SIGNED_ERROR,
        )
    layout.set_error_headers(axes, ("Reference", "Prediction", _OVERVIEW_ERROR_HEADERS[error]), error_texts)
    map_layout.set_row_labels(
        axes, [field.matplotlib_label if field.is_magnitude else field_labels.field_label(field.key, mathtext=True) for field in shown]
    )
    map_layout.set_figure_title(figure, title)
    layout.add_scale_note(figure)
    return figure


def plot_sample_prediction_overview(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    display_spatial_stride: display_grid.DisplayStride = "auto",
    fields: Sequence[str] | None = None,
) -> Figure:
    """Plot reference, prediction, and the selected error of every learned field for one case, or of ``fields`` ("p", "U")."""
    label, frame = _single_dataset(datasets)
    return _plot_prediction_overview(
        label=label,
        frame=frame,
        row_position=row_position,
        error_mode=error_mode,
        context=None,
        display_spatial_stride=display_spatial_stride,
        fields=fields,
    )


def plot_permeability_error_overlay(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    channel: str = "U",
    kappa_scale: str = "log10(kappa)",
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Plot the 2 x 3 permeability tensor, target, and error view."""
    _label, frame = _single_dataset(datasets)
    position = _selected_position(frame, row_position)
    case = cases.load_case(frame, position)
    if case.permeability is None:
        msg = "Permeability context is unavailable."
        raise dataframe.ComparisonCompatibilityError(msg)
    selected = _resolve_display_field(frame, channel)
    reference = presentation.case_field(case, selected, source="reference")
    prediction = presentation.case_field(case, selected, source="prediction")
    selected_error = _error_values(reference, prediction, error_mode=error_mode, frame=frame, field=selected)
    # The red contours mark large errors of either sign, so they follow the magnitude of the displayed error in every mode.
    displayed_error = None if selected_error.map_values is None else np.abs(selected_error.map_values)
    contour_levels = np.unique(np.quantile(displayed_error, (0.75, 0.95))) if displayed_error is not None else np.array([])
    _x, _y, x_grid, y_grid = presentation.display_grid(case)
    display_indices = display_grid.display_indices(x_grid.shape, display_spatial_stride)
    shown_x = display_grid.display_sample(x_grid, display_indices)
    shown_y = display_grid.display_sample(y_grid, display_indices)
    shown_error = None if displayed_error is None else display_grid.display_sample(displayed_error, display_indices)
    by_name = dict(zip(case.permeability_names, case.permeability, strict=True))
    if not {"Kxx", "Kxy", "Kyy"}.issubset(by_name):
        msg = "The tensor overlay requires kxx, kxy, and kyy."
        raise dataframe.ComparisonCompatibilityError(msg)
    components = (("Kxx", by_name["Kxx"]), ("Kxy", by_name["Kxy"]), ("Kyx", by_name["Kxy"]), ("Kyy", by_name["Kyy"]))
    if kappa_scale not in {"kappa", "log10(kappa)"}:
        msg = "kappa_scale must be 'kappa' or 'log10(kappa)'."
        raise ValueError(msg)
    figure, axes = layout.map_subplots(rows=2, columns=3)
    for index, (name, values) in enumerate(components):
        row, column = divmod(index, 2)
        shown = values
        title = field_labels.field_label(name, mathtext=True)
        unit = "m²"
        if kappa_scale == "log10(kappa)" and name in {"Kxx", "Kyy"}:
            positive = values[values > 0.0]
            if positive.size == 0:
                msg = f"{name} must be positive for logarithmic display."
                raise ValueError(msg)
            shown = np.log10(np.maximum(values, float(np.min(positive))))
            title = f"log10({title} / 1 m²)"
            unit = "1"
        _contour(
            figure,
            axes[row, column],
            x_grid,
            y_grid,
            shown,
            levels=_levels(shown, count=11),
            cmap="viridis",
            title=title,
            unit=unit,
            case=case,
            display_indices=display_indices,
        )
        if contour_levels.size:
            axes[row, column].contour(shown_x, shown_y, shown_error, levels=contour_levels, colors="red", linewidths=1.0)
    _contour(
        figure,
        axes[0, 2],
        x_grid,
        y_grid,
        reference,
        levels=_levels(reference, count=11),
        cmap="turbo",
        title=f"{selected.matplotlib_label} true",
        unit=selected.unit,
        case=case,
        display_indices=display_indices,
    )
    if contour_levels.size:
        axes[0, 2].contour(shown_x, shown_y, shown_error, levels=contour_levels, colors="red", linewidths=1.0)
    _error_contour(figure, axes[1, 2], x_grid, y_grid, selected_error, field=selected, case=case, display_indices=display_indices)
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{case.coordinate_units[0]}]",
        y_label=f"y [{case.coordinate_units[1]}]",
    )
    figure.suptitle(layout.case_title("Kappa tensor with error overlay", case_number=case.case_index), fontsize=14)
    return figure


def plot_pressure_velocity_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    model_1: str | None = None,
    model_2: str | None = None,
    scale_mode: str = "Independent",
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Plot 3 x 2 pressure/speed truth and two-model predictions."""
    dataframe.validate_comparison(datasets)
    if len(datasets) < _MINIMUM_COMPARISON_MODELS:
        msg = "Pressure/velocity comparison requires at least two models."
        raise ValueError(msg)
    labels = tuple(datasets)
    first_label = labels[0] if model_1 is None else model_1
    second_label = labels[1] if model_2 is None else model_2
    if first_label not in datasets or second_label not in datasets or first_label == second_label:
        msg = "model_1 and model_2 must name two distinct displayed models."
        raise ValueError(msg)
    position = _selected_position(datasets[first_label], row_position)
    first_case = cases.load_case(datasets[first_label], position)
    second_case = cases.load_case(datasets[second_label], position)
    if (
        first_case.case_index != second_case.case_index
        or first_case.source_index != second_case.source_index
        or first_case.fields != second_case.fields
        or not np.array_equal(first_case.reference, second_case.reference)
    ):
        msg = "Model comparison requires exact shared case identity and reference fields."
        raise dataframe.ComparisonCompatibilityError(msg)
    try:
        display_grid.require_common_grid(
            display_grid.spatial_grid_identity(first_case.coordinates, coordinate_units=first_case.coordinate_units, field_order=first_case.fields),
            display_grid.spatial_grid_identity(
                second_case.coordinates, coordinate_units=second_case.coordinate_units, field_order=second_case.fields
            ),
        )
    except ValueError as error:
        raise dataframe.ComparisonCompatibilityError(str(error)) from error
    display = presentation.shared_display_fields((datasets[first_label], datasets[second_label]))
    pressure = next(field for field in display if field.label == "p")
    speed = next(field for field in display if field.key == "velocity_magnitude")
    pressure_values = (
        presentation.case_field(first_case, pressure, source="reference"),
        presentation.case_field(first_case, pressure, source="prediction"),
        presentation.case_field(second_case, pressure, source="prediction"),
    )
    speed_values = (
        presentation.case_field(first_case, speed, source="reference"),
        presentation.case_field(first_case, speed, source="prediction"),
        presentation.case_field(second_case, speed, source="prediction"),
    )
    row_labels = ("GT", first_label, second_label)
    row_cases = (first_case, first_case, second_case)
    scale = _require_scale_mode(scale_mode)
    _x, _y, x_grid, y_grid = presentation.display_grid(first_case)
    display_indices = display_grid.display_indices(x_grid.shape, display_spatial_stride)
    figure, axes = layout.map_subplots(rows=3, columns=2)
    for row, row_label in enumerate(row_labels):
        pressure_levels = _levels(pressure_values[0] if scale == "Shared (GT)" else pressure_values[row])
        speed_levels = _levels(speed_values[0] if scale == "Shared (GT)" else speed_values[row])
        _contour(
            figure,
            axes[row, 0],
            x_grid,
            y_grid,
            pressure_values[row],
            levels=pressure_levels,
            cmap="turbo",
            title=f"{row_label} {pressure.matplotlib_label}",
            unit=pressure.unit,
            case=first_case,
            display_indices=display_indices,
        )
        _contour(
            figure,
            axes[row, 1],
            x_grid,
            y_grid,
            speed_values[row],
            levels=speed_levels,
            cmap="turbo",
            title=f"{row_label} {speed.matplotlib_label}",
            unit=speed.unit,
            case=first_case,
            display_indices=display_indices,
        )
        tensor = row_cases[row].reference if row == 0 else row_cases[row].prediction
        _overlay_streamlines(
            axes[row, 1],
            datasets[first_label] if row < _MINIMUM_COMPARISON_MODELS else datasets[second_label],
            row_cases[row],
            tensor,
            display_indices=display_indices,
        )
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_case.coordinate_units[0]}]",
        y_label=f"y [{first_case.coordinate_units[1]}]",
    )
    figure.suptitle(layout.case_title("Pressure and velocity comparison", case_number=first_case.case_index), fontsize=14)
    return figure


def _outlier_positions(frame: pd.DataFrame, field: presentation.DisplayField, *, top_k: int) -> tuple[int, ...]:
    """Return worst channel positions followed by the parameter-centre reference."""
    values = np.asarray(pd.to_numeric(frame[field.metric_column], errors="raise"), dtype=float)
    worst = tuple(int(position) for position in _rank_positions(frame, values, descending=True)[:top_k])
    return (*worst, presentation.reference_case_position(frame))


def plot_linked_outlier_cases(
    *,
    datasets: Mapping[str, pd.DataFrame],
    channel: str = "U",
    selection_index: int = 0,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    top_k: int = _DEFAULT_TOP_K,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Render one of the worst-five-plus-reference cases for a selected channel."""
    label, frame = _single_dataset(datasets)
    field = _resolve_display_field(frame, channel)
    positions = _outlier_positions(frame, field, top_k=top_k)
    if isinstance(selection_index, bool) or not isinstance(selection_index, int) or not 0 <= selection_index < len(positions):
        msg = "selection_index is outside the outlier/reference sequence."
        raise IndexError(msg)
    context = "Reference field view" if selection_index == len(positions) - 1 else f"{field.matplotlib_label} outlier field view"
    return _plot_prediction_overview(
        label=label,
        frame=frame,
        row_position=positions[selection_index],
        error_mode=error_mode,
        context=context,
        display_spatial_stride=display_spatial_stride,
    )


def plot_linked_input_extreme_cases(
    *,
    datasets: Mapping[str, pd.DataFrame],
    parameter: str,
    selection_index: int = 0,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Render the minimum or maximum case for one supported metadata parameter."""
    label, frame = _single_dataset(datasets)
    parameters = presentation.metadata_parameters((frame,))
    if parameter not in parameters:
        msg = f"Unsupported metadata parameter {parameter!r}."
        raise ValueError(msg)
    if selection_index not in {0, 1}:
        msg = "selection_index must be 0 (minimum) or 1 (maximum)."
        raise IndexError(msg)
    values = np.asarray(pd.to_numeric(frame[parameter], errors="raise"), dtype=float)
    order = _rank_positions(frame, values, descending=selection_index == 1)
    context = f"Input extreme: {presentation.metadata_label(parameter)} {'minimum' if selection_index == 0 else 'maximum'}"
    return _plot_prediction_overview(
        label=label,
        frame=frame,
        row_position=int(order[0]),
        error_mode=error_mode,
        context=context,
        display_spatial_stride=display_spatial_stride,
    )
