"""Render case, permeability, outlier, and input-extreme views.

Responsibilities:
  - Rank explicit predictive metrics with canonical identity ties
  - Rank supported scientific metadata independently at low and high extremes
  - Render fixed sample, permeability, and two-model field layouts
  - Overlay velocity streamlines on regular display-only physical coordinates
  - Build colored outlier and side-by-side metadata-extreme tables

Design principles:
  - Case selection follows exact saved membership and canonical case identity
  - TaskSpec fields and vector groups define p, u, v, and |u| presentation
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

from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_maps as field_maps

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas.io.formats.style import Styler

_DEFAULT_TOP_K = 5
_MINIMUM_COMPARISON_MODELS = 2
_RELATIVE_DENOMINATOR_FLOOR = 1e-12
_CONTOUR_LEVELS = 10
_MINIMUM_TABLE_NUMERIC_ROWS = 2
_MINIMUM_VELOCITY_COMPONENTS = 2
_REQUIRED_DISPLAY_FIELDS = 4
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
            "relative L2": float(source["rel_l2"]),
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
    case: cases.EvaluationCase,
    display_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> Any:
    """Draw one filled-contour panel and colorbar."""
    displayed = _clip_to_levels(values, levels)
    if display_indices is not None:
        x_grid = display_grid.display_sample(x_grid, display_indices)
        y_grid = display_grid.display_sample(y_grid, display_indices)
        displayed = np.ma.asarray(display_grid.display_sample(displayed, display_indices))
    image = axis.contourf(x_grid, y_grid, displayed, levels=levels, cmap=cmap)
    axis.set_title(title)
    layout.add_map_colorbar(figure, image, axis)
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
    axis: Axes, frame: pd.DataFrame, case: cases.EvaluationCase, values: np.ndarray, *, display_indices: tuple[np.ndarray, np.ndarray] | None = None
) -> None:
    """Overlay velocity using authoritative equally spaced display coordinates."""
    first, second = _velocity_indices(frame, case)
    x, y, _x_grid, _y_grid = presentation.display_grid(case)
    horizontal = values[first]
    vertical = values[second]
    if display_indices is not None:
        y_indices, x_indices = display_indices
        x = x[x_indices]
        y = y[y_indices]
        horizontal = display_grid.display_sample(horizontal, display_indices)
        vertical = display_grid.display_sample(vertical, display_indices)
    if len(x) > 1 and len(y) > 1 and (not np.allclose(np.diff(x), np.diff(x)[0]) or not np.allclose(np.diff(y), np.diff(y)[0])):
        axis.quiver(x, y, horizontal, vertical, color="white", alpha=0.7)
        return
    axis.streamplot(
        x,
        y,
        horizontal,
        vertical,
        color="white",
        density=0.65,
        linewidth=0.55,
        arrowsize=0.55,
    )


def _aggregate_diagonal_permeability(case: cases.EvaluationCase) -> np.ndarray:
    """Return the diagonal-only permeability display aggregate."""
    if case.permeability is None:
        msg = "Permeability context is unavailable."
        raise dataframe.ComparisonCompatibilityError(msg)
    by_name = dict(zip(case.permeability_names, case.permeability, strict=True))
    diagonals = [by_name[name] for name in ("Kxx", "Kyy", "Kzz") if name in by_name]
    if not diagonals:
        msg = "No diagonal permeability components are available."
        raise dataframe.ComparisonCompatibilityError(msg)
    return np.mean(np.stack(diagonals), axis=0)


def _require_error_mode(error_mode: str) -> error_diagnostics.SteadyErrorMode:
    """Require one current error-presentation mode."""
    if error_mode not in error_diagnostics.STEADY_ERROR_MODES:
        msg = f"error_mode must be one of {error_diagnostics.STEADY_ERROR_MODES}."
        raise ValueError(msg)
    return cast("error_diagnostics.SteadyErrorMode", error_mode)


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
    return error_diagnostics.selected_error_map(reference, prediction, mode=mode, field=field.key, physical_unit=field.unit, scale=scale)


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
    """Render one selected 11-band error or its reason in the existing axis."""
    title = f"{field.matplotlib_label} {result.label} [{result.unit}]"
    axis.set_title(title)
    if result.map_values is None:
        axis.text(0.5, 0.5, f"Unavailable: {result.reason}", ha="center", va="center", wrap=True, transform=axis.transAxes)
        _format_axes(axis, case)
        return
    definition_values = np.concatenate((result.map_values.ravel(), np.array((0.0, 1.0)))) if result.scale is not None else result.map_values
    definition = field_maps.build_field_map(
        definition_values,
        semantic=field_maps.FieldMapSemantic.RELATIVE_ERROR if result.unit == "%" else field_maps.FieldMapSemantic.ABSOLUTE_ERROR,
        field=field.key,
        unit=result.unit,
        display_transform=result.label,
        comparison_scope=None if result.scale is None else result.scale.dataset_identity,
        locked_boundaries=np.linspace(0.0, 1.0, 12) if result.scale is not None else None,
        bound_policy="fixed 0..1 with extension" if result.scale is not None else "full selected field",
    )
    map_values = result.map_values
    if display_indices is not None:
        x_grid = display_grid.display_sample(x_grid, display_indices)
        y_grid = display_grid.display_sample(y_grid, display_indices)
        map_values = display_grid.display_sample(map_values, display_indices)
    image = axis.contourf(x_grid, y_grid, map_values, **definition.contourf_kwargs())
    label = "1 (train state std; >1 extended)" if result.scale is not None else result.unit
    layout.add_map_colorbar(figure, image, axis, label=label)
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


def _plot_prediction_overview(
    *,
    frame: pd.DataFrame,
    row_position: int,
    error_mode: str,
    scale_mode: str,
    title: str,
    display_spatial_stride: display_grid.DisplayStride,
) -> Figure:
    """Render the fixed 4 x 4 prediction/reference/error/input figure."""
    position = _selected_position(frame, row_position)
    case = cases.load_case(frame, position)
    fields = presentation.display_fields(frame)
    if len(fields) != _REQUIRED_DISPLAY_FIELDS:
        msg = "The overview requires four supported display fields."
        raise dataframe.ComparisonCompatibilityError(msg)
    scale = _require_scale_mode(scale_mode)
    error = _require_error_mode(error_mode)
    _x, _y, x_grid, y_grid = presentation.display_grid(case)
    display_indices = display_grid.display_indices(x_grid.shape, display_spatial_stride)
    figure, axes = layout.map_subplots(rows=4, columns=4)
    permeability = _aggregate_diagonal_permeability(case)
    positive = permeability[permeability > 0.0]
    if positive.size == 0:
        msg = "Permeability must contain positive values for logarithmic display."
        raise ValueError(msg)
    permeability_log = np.log10(np.maximum(permeability, float(np.min(positive))))

    for row, field in enumerate(fields):
        reference = presentation.case_field(case, field, source="reference")
        prediction = presentation.case_field(case, field, source="prediction")
        if scale == "Shared (GT)":
            reference_levels = _levels(reference)
            prediction_levels = reference_levels
            prediction_plot = prediction
        else:
            reference_levels = _levels(reference)
            prediction_levels = _levels(prediction)
            prediction_plot = np.ma.asarray(prediction)
        _contour(
            figure,
            axes[row, 0],
            x_grid,
            y_grid,
            prediction_plot,
            levels=prediction_levels,
            cmap="turbo",
            title=f"{field.matplotlib_label} pred [{field.unit}]",
            case=case,
            display_indices=display_indices,
        )
        _contour(
            figure,
            axes[row, 1],
            x_grid,
            y_grid,
            reference,
            levels=reference_levels,
            cmap="turbo",
            title=f"{field.matplotlib_label} true [{field.unit}]",
            case=case,
            display_indices=display_indices,
        )
        if field.key in {"u", "v", "velocity_magnitude"}:
            _overlay_streamlines(axes[row, 0], frame, case, case.prediction, display_indices=display_indices)
            _overlay_streamlines(axes[row, 1], frame, case, case.reference, display_indices=display_indices)
        _error_contour(
            figure,
            axes[row, 2],
            x_grid,
            y_grid,
            _error_values(reference, prediction, error_mode=error, frame=frame, field=field),
            field=field,
            case=case,
            display_indices=display_indices,
        )
        if row == 0:
            _contour(
                figure,
                axes[row, 3],
                x_grid,
                y_grid,
                permeability,
                levels=_levels(permeability),
                cmap="viridis",
                title="kappa [m²]",
                case=case,
                display_indices=display_indices,
            )
        elif row == 1:
            _contour(
                figure,
                axes[row, 3],
                x_grid,
                y_grid,
                permeability_log,
                levels=_levels(permeability_log),
                cmap="viridis",
                title="log10(kappa / 1 m²)",
                case=case,
                display_indices=display_indices,
            )
        else:
            axes[row, 3].axis("off")
    x_label = f"x [{case.coordinate_units[0]}]"
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=x_label,
        y_label=f"y [{case.coordinate_units[1]}]",
    )
    figure.suptitle(layout.case_title(title, case_number=case.case_index), fontsize=14)
    layout.add_shortened_column_x_decorations(axes, x_label=x_label)
    return figure


def plot_sample_prediction_overview(
    *,
    datasets: Mapping[str, pd.DataFrame],
    row_position: int = 0,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    scale_mode: str = "Independent",
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Plot the 4 x 4 selected-case prediction overview."""
    _label, frame = _single_dataset(datasets)
    return _plot_prediction_overview(
        frame=frame,
        row_position=row_position,
        error_mode=error_mode,
        scale_mode=scale_mode,
        title="Sample GT vs prediction",
        display_spatial_stride=display_spatial_stride,
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
    displayed_error = selected_error.map_values
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
        title = f"{name} [m²]"
        if kappa_scale == "log10(kappa)" and name in {"Kxx", "Kyy"}:
            positive = values[values > 0.0]
            if positive.size == 0:
                msg = f"{name} must be positive for logarithmic display."
                raise ValueError(msg)
            shown = np.log10(np.maximum(values, float(np.min(positive))))
            title = f"log10({name} / 1 m²)"
        _contour(
            figure,
            axes[row, column],
            x_grid,
            y_grid,
            shown,
            levels=_levels(shown, count=11),
            cmap="viridis",
            title=title,
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
        title=f"{selected.matplotlib_label} true [{selected.unit}]",
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
            title=f"{row_label} p [{pressure.unit}]",
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
            title=f"{row_label} {speed.matplotlib_label} [{speed.unit}]",
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
    scale_mode: str = "Independent",
    top_k: int = _DEFAULT_TOP_K,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Render one of the worst-five-plus-reference cases for a selected channel."""
    _label, frame = _single_dataset(datasets)
    field = _resolve_display_field(frame, channel)
    positions = _outlier_positions(frame, field, top_k=top_k)
    if isinstance(selection_index, bool) or not isinstance(selection_index, int) or not 0 <= selection_index < len(positions):
        msg = "selection_index is outside the outlier/reference sequence."
        raise IndexError(msg)
    title = "Reference field view" if selection_index == len(positions) - 1 else f"{field.label} outlier field view"
    return _plot_prediction_overview(
        frame=frame,
        row_position=positions[selection_index],
        error_mode=error_mode,
        scale_mode=scale_mode,
        title=title,
        display_spatial_stride=display_spatial_stride,
    )


def plot_linked_input_extreme_cases(
    *,
    datasets: Mapping[str, pd.DataFrame],
    parameter: str,
    selection_index: int = 0,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    scale_mode: str = "Independent",
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Render the minimum or maximum case for one supported metadata parameter."""
    _label, frame = _single_dataset(datasets)
    parameters = presentation.metadata_parameters((frame,))
    if parameter not in parameters:
        msg = f"Unsupported metadata parameter {parameter!r}."
        raise ValueError(msg)
    if selection_index not in {0, 1}:
        msg = "selection_index must be 0 (minimum) or 1 (maximum)."
        raise IndexError(msg)
    values = np.asarray(pd.to_numeric(frame[parameter], errors="raise"), dtype=float)
    order = _rank_positions(frame, values, descending=selection_index == 1)
    title = f"Input extreme: {presentation.metadata_label(parameter)} {'minimum' if selection_index == 0 else 'maximum'}"
    return _plot_prediction_overview(
        frame=frame,
        row_position=int(order[0]),
        error_mode=error_mode,
        scale_mode=scale_mode,
        title=title,
        display_spatial_stride=display_spatial_stride,
    )
