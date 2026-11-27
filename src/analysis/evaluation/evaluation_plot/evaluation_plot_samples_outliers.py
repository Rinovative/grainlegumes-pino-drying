"""Render case, permeability, outlier, and input-extreme views.

Responsibilities:
  - Rank explicit predictive metrics with canonical identity ties
  - Rank supported scientific metadata independently at low and high extremes
  - Render one case's reference, prediction, and selected error of the displayed fields (every learned field, or a ``fields``
    selection) on the shared fixed map grid, every panel on discrete round colour levels
  - Render the fixed permeability tensor, error and reference view, the red error contours it draws named in the note, and the
    pressure and speed of the reference beside each selected model on the same grid
  - Overlay velocity streamlines on regular display-only physical coordinates, integrated once per drawn field
  - Build colored outlier tables, the case column written as whole case numbers, and unshaded side-by-side metadata-extreme
    tables, whose parameter ranges rank nothing

Design principles:
  - Case selection follows exact saved membership and canonical case identity
  - TaskSpec fields and vector groups define p, u, v, and |u| presentation
  - Every case map is captioned «view · Material · role — Case N» by the display case number that the case navigator and the
    outlier table show, and ends with the compared models' workspace short labels, «A vs B» for two
  - State levels cover the 5th to the 95th percentile of the reference, error levels follow the shared percentile rule,
    arrows mark values beyond; non-finite prediction cells stay masked and are noted
  - Permeability remains physical context and is never an error aggregate; Kxx and Kyy have eleven geometric bands, written as
    log10 values on round ticks on request, the shear components round bands symmetric about 0, a constant component its one value
  - Scaling, contouring, and regular coordinates affect display only

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
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation.evaluation_plot import evaluation_plot_error_behavior as error_behavior
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_plot_layout as map_layout
from src.analysis.ui import analysis_ui_tables as tables

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from pandas.io.formats.style import Styler

_DEFAULT_TOP_K = 5
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
_SIGNED_TENSOR_COMPONENTS = frozenset({"Kxy", "Kyx"})  # the shear components may be negative; Kxx and Kyy are positive
_STANDARD_NUMBER_LOWER = 1e-4
_STANDARD_NUMBER_UPPER = 1e4
_MINIMUM_ROUND_TICKS = 2  # a log10 permeability bar labels round values only when at least two fall inside its bands
_TICK_TOLERANCE = 1e-9  # a round tick at a band end, within rounding, stands inside the bar
_CONTOUR_PERCENTILES = {"75th": 0.75, "95th": 0.95}  # the red lines of the permeability overlay: |error| at these quantiles


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


def _rank_style(table: pd.DataFrame, *, reference_row: int, columns: Sequence[str]) -> pd.DataFrame:
    """Return blue/purple cell fills of the error ``columns`` diverging from the reference; every other column stays unshaded."""
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    blue = colormaps["Blues"]
    purple = colormaps["Purples"]
    for column in columns:
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
    """Build one worst-five-plus-reference channel table: only its error columns are shaded (from two rows on) and no row index shows."""
    values = np.asarray(pd.to_numeric(frame[field.metric_column], errors="raise"), dtype=float)
    positions = list(_rank_positions(frame, values, descending=True)[:top_k])
    reference_position = presentation.reference_case_position(frame)
    parameters = presentation.metadata_parameters((frame,))
    rmse_column = f"normalized RMSE [{field.label}]"
    rows: list[dict[str, Any]] = []
    for position in (*positions, reference_position):
        source = frame.iloc[int(position)]
        is_reference = int(position) == reference_position
        row: dict[str, Any] = {
            # The case number is an identity, written whole, never rounded by the numeric formatter.
            "Case": f"{int(source['case_index'])}{' (Ref)' if is_reference else ''}",
            rmse_column: float(source[field.metric_column]),
            "relative L2": float(source["physical_relative_l2"]),
        }
        row.update({presentation.metadata_label(column): float(source[column]) for column in parameters})
        rows.append(row)
    table = pd.DataFrame(rows)
    reference_row = len(table) - 1
    styler = table.style.hide(axis="index")
    styler.format(_fmt_number)
    if len(table) >= _MINIMUM_TABLE_NUMERIC_ROWS:
        styler.apply(_rank_style, axis=None, reference_row=reference_row, columns=[rmse_column, "relative L2"])
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
    blocks: list[tuple[str, pd.DataFrame]] = []
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
                    "Min / ref": minimum / reference_value if reference_value != 0.0 else "",  # no ratio to a zero reference: an empty cell
                    "Max / ref": maximum / reference_value if reference_value != 0.0 else "",
                }
            )
        blocks.append((count_headings[label] or label, pd.DataFrame(rows).set_index("Parameter")))
    return tables.grouped_styled_dataframes(blocks, title=title, columns=len(blocks))


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


def _models_suffix(datasets: Mapping[str, pd.DataFrame], keys: Sequence[str]) -> str:
    """
    Return the title suffix naming the compared models ``keys`` by the workspace short labels the panel's ``DisplayFrames`` carry.

    The rule is ``layout.models_suffix``; a bare mapping without labels, or a model without an explicit label, gives none.
    """
    return layout.models_suffix([datasets.title_labels[key] for key in keys] if isinstance(datasets, dataset_cuts.DisplayFrames) else [])


def _single_dataset(datasets: Mapping[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Return the model title suffix (``_models_suffix``) and the frame of the one dataset selected by a local dropdown."""
    dataframe.validate_comparison(datasets)
    if len(datasets) != 1:
        msg = "This case view requires one selected dataset."
        raise ValueError(msg)
    key = next(iter(datasets))
    return _models_suffix(datasets, (key,)), datasets[key]


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

    A degenerate percentile range, such as a constant field, falls back to the full finite range (``field_maps.state_boundaries``).
    """
    scale = field_maps.percentile_scale(values, "state")
    if scale is None:
        return None
    finite = values[np.isfinite(values)]
    return field_maps.state_boundaries(*(scale if scale[1] > scale[0] else (float(np.min(finite)), float(np.max(finite)))))


def _state_scale(values: np.ndarray, reference: np.ndarray, colormap: str) -> field_maps.FieldMapDefinition | None:
    """Return the state scale of ``values`` on the bands of ``reference`` (``_state_boundaries``); ``None`` without a finite reference."""
    boundaries = _state_boundaries(reference)
    return None if boundaries is None else _discrete_scale(values, boundaries, colormap)


def _discrete_scale(
    values: np.ndarray,
    boundaries: np.ndarray,
    colormap: str,
    semantic: field_maps.FieldMapSemantic = field_maps.FieldMapSemantic.LINEAR_POSITIVE,
) -> field_maps.FieldMapDefinition:
    """
    Return one map's scale: a colour per band of ``boundaries`` and arrows at the ends its finite ``values`` pass.

    The bands come from the percentile rules, not from ``values``, which may lie beyond them or be constant, so the scale is built
    here rather than by ``field_maps.build_field_map``. Its semantic only picks the bar's labels (``add_field_map_colorbar``): an
    error map keeps its error semantic, a linear state map (of either sign) is a linear positive one, labelled as a state.
    """
    cmap = colormaps[colormap].resampled(len(boundaries) - 1)
    return field_maps.FieldMapDefinition(
        semantic=semantic,
        field="",
        unit="",
        display_transform="identity",
        comparison_scope=None,
        bound_policy="percentile",
        extend=field_maps.locked_extend(values[np.isfinite(values)], boundaries),
        locked=True,
        boundaries=boundaries,
        colormap=cmap,
        normalizer=BoundaryNorm(boundaries, cmap.N),
        ticks=boundaries,
    )


def _error_scale(result: error_diagnostics.SelectedErrorMap) -> tuple[field_maps.FieldMapDefinition, str]:
    """
    Return the scale and colour-bar label of one available steady error map.

    The signed error runs symmetric to the larger absolute one of its 5th and 95th percentile on a blue-white-red scale,
    every other mode from 0 to its 95th percentile; errors beyond get the bar's arrows.
    """
    unit = f"[{field_labels.display_unit(result.unit, quantity_kind='difference')}]"
    signed = result.label == error_diagnostics.SIGNED_ERROR
    relative = result.unit == "%"
    semantics = visual_semantics.field_visual_semantics("error", role="signed_error" if signed else "absolute_error")
    values = np.asarray(result.map_values, dtype=float)
    semantic = field_maps.FieldMapSemantic
    return (
        _discrete_scale(
            values,
            field_maps.error_boundaries(values, signed=signed),
            _RELATIVE_ERROR_COLORMAP if relative else semantics.colormap,
            semantic.SIGNED_ERROR if signed else semantic.RELATIVE_ERROR if relative else semantic.ABSOLUTE_ERROR,
        ),
        f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} {unit}" if result.scale is not None else unit,
    )


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


def _sample_title(frame: pd.DataFrame, position: int, suffix: str, view: str | None = None) -> str:
    """
    Return «view · Material, role — Case N» of one saved row and the model ``suffix`` (``_models_suffix``), the caption of every case map.

    The material and the saved split role read as every view writes a dataset (``material_role_display_label``: ``Lentil, ID``).
    The case is named by its display case number (``case_index``), the number the case navigator and the outlier table show, never by
    its generation case ID, an identifier. Like the transient title, the model is named by the workspace short label
    (``FNO · B · seed 9``) for a single run and in a comparison alike, never by the run's full name.
    """
    material = frame.iloc[position].get("material_family")
    if not isinstance(material, str) or not material:
        msg = "A sample title requires the row's material_family metadata."
        raise dataframe.ComparisonCompatibilityError(msg)
    role = str(frame.attrs.get("dataset_role") or "unspecified")
    parts = (view, display_labels.material_role_display_label(material, role))
    caption = " · ".join(part for part in parts if part is not None)
    return layout.case_title(caption, case_number=presentation.case_number(frame, position)) + suffix


def _draw_map(
    figure: Figure,
    axis: Axes,
    slot: Axes,
    grids: tuple[np.ndarray, np.ndarray],
    values: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray],
    *,
    scale: field_maps.FieldMapDefinition | None,
    label: str,
    exponent: int | None = None,
) -> Colorbar | None:
    """
    Draw one map on ``scale`` and return its labelled slot bar, non-finite cells masked and noted; ``None`` states no finite values.

    A given ``exponent`` is the power of ten the bar's labels are written in (``map_layout.add_field_map_colorbar``).
    """
    if scale is None:
        axis.text(0.5, 0.5, "unavailable: no finite values", ha="center", va="center", transform=axis.transAxes)
        return None
    image = axis.pcolormesh(*grids, np.ma.masked_invalid(display_grid.display_sample(values, indices)), shading="auto", **scale.pcolormesh_kwargs())
    if not np.isfinite(values).all():
        axis.text(0.02, 0.02, _MASKED_NOTE, transform=axis.transAxes, fontsize=8)
    return map_layout.add_field_map_colorbar(figure, image, slot, scale, label=label, exponent=exponent)


def _field_colormap(field: presentation.DisplayField, velocity: Sequence[str]) -> str:
    """Return the colormap of a state map: turbo for a velocity component, else the field's semantic one."""
    return _VELOCITY_COLORMAP if field.key in velocity else visual_semantics.field_visual_semantics(field.key).colormap


def _draw_error_panel(
    figure: Figure,
    axis: Axes,
    slot: Axes,
    grids: tuple[np.ndarray, np.ndarray],
    indices: tuple[np.ndarray, np.ndarray],
    result: error_diagnostics.SelectedErrorMap,
    exponent: int | None = None,
) -> str:
    """Draw one selected error map, or why it is unavailable, and return its mean-error line, both in 10**``exponent`` if given."""
    error_unit = None if result.scale is not None else field_labels.display_unit(result.unit, quantity_kind="difference")
    if result.map_values is None:
        axis.text(0.5, 0.5, f"Unavailable: {result.reason}", ha="center", va="center", wrap=True, transform=axis.transAxes)
        return layout.mean_error_text(None, unit=error_unit)
    scale, label = _error_scale(result)
    _draw_map(figure, axis, slot, grids, result.map_values, indices, scale=scale, label=label, exponent=exponent)
    signed = scale.semantic in field_maps.SIGNED_SEMANTICS
    shown = map_layout.error_format(scale.boundaries, signed=signed, exponent=exponent)
    return layout.mean_error_text(result.map_values, unit=error_unit, exponent=shown.exponent)


def _tensor_scale(name: str, values: np.ndarray, *, log: bool) -> tuple[np.ndarray, field_maps.FieldMapDefinition]:
    """
    Return the shown values and the scale of one permeability component, both from the one permeability scale of its values.

    Kxx and Kyy are positive: eleven geometric bands span the field's range, and ``log`` shows the log10 values on the log10 of exactly
    these bands. A near-uniform one whose log10 bands would be finer than a state label's last decimal is widened in log10 by the
    state-band rule (``field_maps.state_boundaries``), so its labels stay distinct. The shear components are signed: round bands
    symmetric about 0 on the signed colormap. A constant component keeps the truthful one-value scale of that permeability scale,
    the shear its signed colormap, ``log`` at the log10 of its value. A positive component with a value of zero or below raises
    ``ValueError``.
    """
    signed = name in _SIGNED_TENSOR_COMPONENTS
    if not signed and np.any(values <= 0.0):
        msg = f"{name} must be positive for the permeability scales."
        raise ValueError(msg)
    semantic = field_maps.FieldMapSemantic
    colormap = visual_semantics.field_visual_semantics(name, role="signed_error" if signed else None).colormap
    options: dict[str, Any] = {
        "semantic": semantic.LINEAR_SIGNED if signed else semantic.LOG_POSITIVE,
        "field": name,
        "unit": "m²",
        "colormap_name": colormap,
    }
    kappa = field_maps.build_field_map(values, **options)
    exponents = np.log10(kappa.boundaries) if kappa.semantic is semantic.LOG_POSITIVE else None
    if exponents is not None and float(np.min(np.diff(exponents))) < 10.0**-field_maps.MAX_TICK_DECIMALS:
        # Near-uniform: its log10 bands are widened like any state's, so their labels stay distinct; kappa takes them back.
        kappa = field_maps.build_field_map(values, locked_boundaries=10.0 ** field_maps.state_boundaries(exponents[0], exponents[-1]), **options)
    if not log:
        return values, kappa
    shown = np.log10(values)
    if kappa.semantic is semantic.CONSTANT:
        return shown, field_maps.build_field_map(
            shown, semantic=semantic.CONSTANT, field=name, unit="1", display_transform="log10", colormap_name=colormap
        )
    return shown, _discrete_scale(shown, np.log10(kappa.boundaries), colormap)


def _label_log10_bands(colorbar: Colorbar, boundaries: np.ndarray) -> None:
    """
    Label a log10 permeability bar at the round values inside its bands, which stay the log10 of the geometric kappa bands.

    Those boundaries are not round, so labelling them would write three-decimal ticks; the round ticks name the scale without moving
    a band. A range too narrow for two round ticks keeps its boundary labels.
    """
    ticks = field_maps.rounded_boundaries(float(boundaries[0]), float(boundaries[-1]), max_decimals=field_maps.MAX_TICK_DECIMALS)
    if ticks is not None:
        inside = ticks[(ticks >= boundaries[0] - _TICK_TOLERANCE) & (ticks <= boundaries[-1] + _TICK_TOLERANCE)]
        if inside.size >= _MINIMUM_ROUND_TICKS:
            map_layout.set_fixed_decimal_ticks(colorbar, inside)


def _channel_header(field: presentation.DisplayField, header: str) -> str:
    """Return the shared column ``header`` («Reference», «Signed error» ...) of ``field`` in the active label style, with a capital first letter."""
    label = field.metric_label(header[0].lower() + header[1:])
    return label[0].upper() + label[1:]


def _contour_note(
    field: presentation.DisplayField, result: error_diagnostics.SelectedErrorMap, percentiles: np.ndarray, drawn: Sequence[float]
) -> str | None:
    """
    Return the note naming the red lines: the |error| of ``field`` at those of its ``percentiles`` whose level drew a line.

    A level that two percentiles share is written once; a level whose contour has no path is not named, and without a drawn line
    there is no note.
    """
    if not drawn:  # also an unavailable error map, which has no percentiles
        return None
    named = [name for name, level in zip(_CONTOUR_PERCENTILES, percentiles, strict=True) if level in drawn]
    unit = "" if result.scale is not None else f" {field_labels.display_unit(result.unit, quantity_kind='difference')}"
    symbol = error_diagnostics.NORMALIZED_ERROR_SYMBOL if result.scale is not None else "|e|"
    values = ("both " if len(drawn) < len(named) else "") + " and ".join(layout.significant(level) for level in drawn)
    lines = "lines" if len(drawn) > 1 else "line"
    return f"red {lines}: {symbol} of {field.matplotlib_label} at its {' and '.join(named)} percentile, {values}{unit}"


def _plot_prediction_overview(
    *,
    suffix: str,
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
    title = _sample_title(frame, position, suffix, context)
    velocity = dataframe.output_group_fields(frame, group_id="velocity")
    _x, _y, x_grid, y_grid = presentation.display_grid(case)
    indices = display_grid.display_indices(x_grid.shape, display_spatial_stride)
    grids = (display_grid.display_sample(x_grid, indices), display_grid.display_sample(y_grid, indices))
    extent = display_grid.physical_extent(*case.coordinates)
    geometry = map_layout.plan_geometry(len(shown), _OVERVIEW_COLUMNS, (extent[1] - extent[0], extent[3] - extent[2]))
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    streamlines: dict[int, list[np.ndarray]] = {}  # per reference and prediction column, reused on every velocity row
    error_texts = []
    arrays = [
        (presentation.case_field(case, field, source="reference"), presentation.case_field(case, field, source="prediction")) for field in shown
    ]
    results = [
        _error_values(reference, prediction, error_mode=error, frame=frame, field=field)
        for field, (reference, prediction) in zip(shown, arrays, strict=True)
    ]
    # The error bars of rows with one unit share one power of ten where their tick steps lie close.
    exponents = error_behavior.error_bar_exponents(
        [(result.unit, None if result.map_values is None else _error_scale(result)[0]) for result in results]
    )
    for row, field in enumerate(shown):
        reference, prediction = arrays[row]
        colormap = _field_colormap(field, velocity)
        unit = f"[{field_labels.display_unit(field.unit)}]"
        for column, (values, tensor) in enumerate(((reference, case.reference), (prediction, case.prediction))):
            scale = _state_scale(values, reference, colormap)
            _draw_map(figure, axes[row, column], slots[row, column], grids, values, indices, scale=scale, label=unit)
            if set(field.component_fields) <= set(velocity):  # a velocity component or the magnitude of the pair
                streamlines[column] = _overlay_streamlines(
                    axes[row, column], frame, case, tensor, display_indices=indices, segments=streamlines.get(column)
                )
        error_texts.append(_draw_error_panel(figure, axes[row, -1], slots[row, -1], grids, indices, results[row], exponents[row]))
    layout.set_error_headers(axes, ("Reference", "Prediction", _OVERVIEW_ERROR_HEADERS[error]), error_texts)
    map_layout.set_row_labels(
        axes, [field.matplotlib_label if field.is_magnitude else field_labels.field_label(field.key, mathtext=True) for field in shown]
    )
    map_layout.set_map_title(figure, title)
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
    suffix, frame = _single_dataset(datasets)
    return _plot_prediction_overview(
        suffix=suffix,
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
    """
    Plot the 2 x 3 permeability tensor, error, and reference view on the shared fixed map grid.

    Kxx and Kyy are drawn on eleven geometric bands over their range, written as log10 values on round ticks with ``kappa_scale``
    "log10(kappa)"; the shear components Kxy and Kyx on round bands symmetric about 0. The selected error stands right of the top row
    on its own scale (``_error_scale``), its mean |e| line under its header; the reference under it has the state levels of the sample
    overview. Red lines mark the displayed error at its 75th and 95th percentile on every tensor and reference panel, and the note
    names those that draw a line with their values (``_contour_note``).
    """
    suffix, frame = _single_dataset(datasets)
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
    percentiles = np.quantile(displayed_error, tuple(_CONTOUR_PERCENTILES.values())) if displayed_error is not None else np.array([])
    _x, _y, x_grid, y_grid = presentation.display_grid(case)
    indices = display_grid.display_indices(x_grid.shape, display_spatial_stride)
    grids = (display_grid.display_sample(x_grid, indices), display_grid.display_sample(y_grid, indices))
    shown_error = None if displayed_error is None else display_grid.display_sample(displayed_error, indices)
    by_name = dict(zip(case.permeability_names, case.permeability, strict=True))
    if not {"Kxx", "Kxy", "Kyy"}.issubset(by_name):
        msg = "The tensor overlay requires kxx, kxy, and kyy."
        raise dataframe.ComparisonCompatibilityError(msg)
    components = (("Kxx", by_name["Kxx"]), ("Kxy", by_name["Kxy"]), ("Kyx", by_name["Kxy"]), ("Kyy", by_name["Kyy"]))
    if kappa_scale not in {"kappa", "log10(kappa)"}:
        msg = "kappa_scale must be 'kappa' or 'log10(kappa)'."
        raise ValueError(msg)
    extent = display_grid.physical_extent(*case.coordinates)
    geometry = map_layout.plan_geometry(2, 3, (extent[1] - extent[0], extent[3] - extent[2]), row_labels=False)
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    titles: tuple[list[str], list[str]] = ([], [])
    for index, (name, values) in enumerate(components):
        row, column = divmod(index, 2)
        log = kappa_scale == "log10(kappa)" and name not in _SIGNED_TENSOR_COMPONENTS
        shown, scale = _tensor_scale(name, values, log=log)
        label = f"[{field_labels.display_unit('1' if log else 'm²')}]"
        bar = _draw_map(figure, axes[row, column], slots[row, column], grids, shown, indices, scale=scale, label=label)
        if log and bar is not None and scale.semantic is not field_maps.FieldMapSemantic.CONSTANT:
            _label_log10_bands(bar, scale.boundaries)
        symbol = field_labels.field_label(name, mathtext=True)
        titles[row].append(rf"$\log_{{10}}$({symbol} / 1 m²)" if log else symbol)  # typeset like the spectral transfer label
    # The error stands in the top row, its mean line in the header band above the grid; the narrow row gap holds one header only.
    error_text = _draw_error_panel(figure, axes[0, 2], slots[0, 2], grids, indices, selected_error)
    titles[0].append(_channel_header(selected, _OVERVIEW_ERROR_HEADERS[error_mode]))
    velocity = dataframe.output_group_fields(frame, group_id="velocity")
    target = _state_scale(reference, reference, _field_colormap(selected, velocity))
    _draw_map(figure, axes[1, 2], slots[1, 2], grids, reference, indices, scale=target, label=f"[{field_labels.display_unit(selected.unit)}]")
    titles[1].append(_channel_header(selected, "Reference"))
    drawn: list[float] = []
    if percentiles.size:
        contours = [
            axis.contour(*grids, shown_error, levels=np.unique(percentiles), colors="red", linewidths=1.0) for axis in (*axes[:, :2].flat, axes[1, 2])
        ]
        # Every panel contours the same field, so the levels with a path in one are the red lines a reader sees in all.
        drawn = [float(level) for level, path in zip(contours[0].levels, contours[0].get_paths(), strict=True) if len(path.vertices)]
    map_layout.set_column_headers(axes[:1], titles[0], ("", "", error_text))
    map_layout.set_column_headers(axes[1:], titles[1], ())
    map_layout.set_map_title(figure, _sample_title(frame, position, suffix, "Kappa tensor with error overlay"))
    layout.add_scale_note(
        figure,
        shared=False,
        scale=f"permeability over its full range, reference and error {layout.PERCENTILE_SCALE}",
        extra=_contour_note(selected, selected_error, percentiles, drawn),
    )
    return figure


def plot_pressure_velocity_comparison(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    models: Sequence[str] | None = None,
    scale_mode: str = "Independent",
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """
    Plot pressure and speed of the reference and of each selected model: rows p and |u|, columns Reference, then the models in order.

    ``models`` names the displayed models (default: all of ``datasets``); they must share the case and its reference fields. Every
    map has discrete round levels over the 5th to the 95th percentile of its own values ("Independent") or of the reference
    ("Shared (GT)"), arrows marking the values beyond; the speed maps carry streamlines of their field pair.
    """
    dataframe.validate_comparison(datasets)
    labels = tuple(datasets) if models is None else tuple(models)
    if not labels or len(set(labels)) != len(labels) or any(label not in datasets for label in labels):
        msg = "models must name distinct displayed models."
        raise ValueError(msg)
    frames = [datasets[label] for label in labels]
    position = _selected_position(frames[0], row_position)
    model_cases = [cases.load_case(frame, position) for frame in frames]
    first_case = model_cases[0]
    first_grid = display_grid.spatial_grid_identity(
        first_case.coordinates, coordinate_units=first_case.coordinate_units, field_order=first_case.fields
    )
    for other in model_cases[1:]:
        if (
            first_case.case_index != other.case_index
            or first_case.source_index != other.source_index
            or first_case.fields != other.fields
            or not np.array_equal(first_case.reference, other.reference)
        ):
            msg = "Model comparison requires exact shared case identity and reference fields."
            raise dataframe.ComparisonCompatibilityError(msg)
        try:
            display_grid.require_common_grid(
                first_grid, display_grid.spatial_grid_identity(other.coordinates, coordinate_units=other.coordinate_units, field_order=other.fields)
            )
        except ValueError as error:
            raise dataframe.ComparisonCompatibilityError(str(error)) from error
    display = presentation.shared_display_fields(tuple(frames))
    fields = (next(field for field in display if field.label == "p"), next(field for field in display if field.key == "velocity_magnitude"))
    scale = _require_scale_mode(scale_mode)
    _x, _y, x_grid, y_grid = presentation.display_grid(first_case)
    indices = display_grid.display_indices(x_grid.shape, display_spatial_stride)
    grids = (display_grid.display_sample(x_grid, indices), display_grid.display_sample(y_grid, indices))
    extent = display_grid.physical_extent(*first_case.coordinates)
    geometry = map_layout.plan_geometry(len(fields), len(labels) + 1, (extent[1] - extent[0], extent[3] - extent[2]), width_cap=None)
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    velocity = dataframe.output_group_fields(frames[0], group_id="velocity")
    truth = [presentation.case_field(first_case, field, source="reference") for field in fields]
    columns = [(frames[0], first_case, True), *((frame, case, False) for frame, case in zip(frames, model_cases, strict=True))]
    for column, (frame, case, is_truth) in enumerate(columns):
        for row, field in enumerate(fields):
            values = presentation.case_field(case, field, source="reference" if is_truth else "prediction")
            state = _state_scale(values, truth[row] if scale == "Shared (GT)" else values, _field_colormap(field, velocity))
            _draw_map(
                figure, axes[row, column], slots[row, column], grids, values, indices, scale=state, label=f"[{field_labels.display_unit(field.unit)}]"
            )
        _overlay_streamlines(axes[1, column], frame, case, case.reference if is_truth else case.prediction, display_indices=indices)
    map_layout.set_column_headers(axes, ["Reference", *labels], ())
    map_layout.set_row_labels(axes, [field.matplotlib_label for field in fields])
    map_layout.set_map_title(figure, _sample_title(frames[0], position, _models_suffix(datasets, labels), "Pressure and velocity comparison"))
    layout.add_scale_note(figure, shared=scale == "Shared (GT)")
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
    suffix, frame = _single_dataset(datasets)
    field = _resolve_display_field(frame, channel)
    positions = _outlier_positions(frame, field, top_k=top_k)
    if isinstance(selection_index, bool) or not isinstance(selection_index, int) or not 0 <= selection_index < len(positions):
        msg = "selection_index is outside the outlier/reference sequence."
        raise IndexError(msg)
    context = "Reference field view" if selection_index == len(positions) - 1 else f"{field.matplotlib_label} outlier field view"
    # The ranking channel |u| is no learned field, so the default rows omit it: show it after the learned fields.
    fields = [item.key for item in presentation.display_fields(frame) if not item.is_magnitude] + [field.key] if field.is_magnitude else None
    return _plot_prediction_overview(
        suffix=suffix,
        frame=frame,
        row_position=positions[selection_index],
        error_mode=error_mode,
        context=context,
        display_spatial_stride=display_spatial_stride,
        fields=fields,
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
    suffix, frame = _single_dataset(datasets)
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
        suffix=suffix,
        frame=frame,
        row_position=int(order[0]),
        error_mode=error_mode,
        context=context,
        display_spatial_stride=display_spatial_stride,
    )
