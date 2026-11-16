"""Render physical-consistency CDFs, maps, and tables from current evidence.

Responsibilities:
  - Summarize dimensionless momentum, flux-balance, and pressure-boundary metrics
  - Plot scalar residual CDFs on separate panels with one legend beside them
  - Compare the median pressure-boundary components and the pressure-drop error as bars on two panels
  - Render the full-grid momentum magnitude map of every dataset side by side on the shared fixed map grid, with the residual
    CDFs in a band above them or one flux-balance profile in a band right of them on the maps' y range
  - Calculate pressure-drop mismatch from admitted pressure and boundary fields
  - Pair only scalar distributions and maps that represent the same quantity
  - Present descriptive physical statistics in colored tables

Design principles:
  - Scalar metrics and full-grid arrays retain their declared distinct semantics
  - Residual metrics are dimensionless and pressure-drop evidence stays in pascals, so the two never share an axis
  - Complete or bounded case reductions are owned by EvaluationSession
  - Display clipping never changes immutable residual or pressure evidence
  - Per-case distributions leave out non-finite cases, which the invalid-prediction warning names
  - Residual maps share one round scale locked over the datasets, each bar with its unit and an arrow only where its map has values
    beyond; every line or bar takes its model's fixed series style (``evaluation_plot_series``)
  - Line and bar figures draw a row of panels per dataset and the models as lines or bars inside it

This module does NOT:
  - Recompute residual equations, derivatives, crops, or boundary masks
  - Fabricate an unavailable independent ground-truth physics baseline
  - Derive a composite physics score
  - Admit artifacts or own notebook control composition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as map_layout
from src.analysis.ui import analysis_ui_tables as tables

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import ipywidgets as widgets
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.analysis.evaluation import evaluation_datasets as dataset_cuts
    from src.analysis.presentation import analysis_visual_semantics as visual_semantics

_DEFAULT_CASE_LIMIT = 100
_CDF_BAND_IN = 3.0  # height of the residual-CDF band above the maps
_PROFILE_BAND_IN = 3.0  # width of the flux-balance profile band right of the maps
_DIMENSIONLESS = field_labels.display_unit("1")
_HIDDEN_NOTE_GID = "cdf-values-not-shown"
_RESIDUAL_COLORBAR_LABEL = rf"mean $\sqrt{{R_x^2+R_y^2}}$ [{_DIMENSIONLESS}]"  # R_x, R_y are already scaled by the pressure gradient
_RESIDUAL_LABELS = {  # the per-case scalar of each residual, in the symbols of the artifact contract
    "momentum_residual_mse": r"$\mathrm{mean}(R_x^2+R_y^2)$",
    "flux_balance_mse": r"$\mathrm{mean}(((q(y)-\bar{q})/Q)^2)$",
    "pressure_boundary_mse": r"$\mathrm{mean}_{in}((p-p_{bc})^2)/\Delta p^2+(\bar{p}_{out}/\Delta p)^2$",
}
_PRESSURE_COMPONENTS = (  # the dimensionless medians of the pressure-boundary summary: column, frame column
    ("pressure inlet MSE / Δp²", "pressure_inlet_mse"),
    ("squared outlet mean / Δp²", "pressure_outlet_mean_square"),
    ("pressure boundary MSE / Δp²", "pressure_boundary_mse"),
)
_PRESSURE_DROP_COLUMN = "pressure drop absolute error"


def _values(frame: pd.DataFrame, column: str, max_cases: int) -> np.ndarray:
    """Return the non-negative values of one saved prefix without its non-finite cases."""
    values = validity_contract.finite_case_values(pd.to_numeric(frame.iloc[:max_cases][column], errors="raise"))
    if np.any(values < 0.0):
        msg = f"Physical metric {column!r} must be non-negative."
        raise ValueError(msg)
    return values


def _log_axis_values(values: np.ndarray) -> np.ndarray:
    """Return the values for a log axis with non-positive entries as gaps, never raised to a fake positive value."""
    values = np.asarray(values, dtype=float)
    return np.where(values > 0.0, values, np.nan)


def _cdf(axis: Axes, values: np.ndarray, style: visual_semantics.SeriesStyle, *, name: str, case_count: int, label: str | None = None) -> None:
    """
    Draw one series' CDF on a log axis; a series without a finite case, and every value ≤ 0 a log axis cannot place, are named on the panel.

    The values ≤ 0 keep their share of the CDF but get no point, so the panel says how many each series has.
    """
    if values.size == 0:
        layout.mark_empty_prefix(axis, case_count=case_count, model=name)
        return
    hidden = int(np.count_nonzero(values <= 0.0))
    if hidden:
        earlier = sum(item.get_gid() == _HIDDEN_NOTE_GID for item in axis.texts)
        axis.text(
            0.98,
            0.02 + 0.08 * earlier,
            f"{name}: {hidden} of {values.size} values ≤ 0 not shown",
            ha="right",
            va="bottom",
            transform=axis.transAxes,
            fontsize=figure_layout.TICK_PT,
            gid=_HIDDEN_NOTE_GID,
        )
    if hidden < values.size:
        layout.plot_ecdf(axis, values, color=style.color, linestyle=style.linestyle, marker=style.marker, label=label, positive_only=True)


def _finish(
    figure: Figure,
    axes: np.ndarray,
    title: str,
    datasets: dataset_cuts.DisplayFrames,
    groups: Mapping[str, Mapping[str, str]],
    styles: series.SeriesStyles,
    count_headings: Mapping[str, str],
    *,
    bars: bool = False,
) -> None:
    """Write the dataset header of every row, the legend naming the models (and differing roles) and the figure title."""
    figure_layout.set_row_headers(figure, axes, [count_headings[dataset] or dataset for dataset in groups])
    layout.add_series_legend(figure, axes, datasets, groups, styles, bars=bars)
    figure_layout.set_figure_title(figure, axes, title, note=layout.series_note(styles))


def build_physical_consistency_summary_table(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build current named residual statistics without a composite score."""
    dataframe.validate_comparison(datasets, require_physics=True)
    statistics = (
        ("median", np.median),
        ("mean", np.mean),
        ("q90", lambda values: np.quantile(values, 0.90)),
        ("q95", lambda values: np.quantile(values, 0.95)),
    )
    rows: list[dict[str, Any]] = []
    for label, frame in datasets.items():
        row: dict[str, Any] = {"Model": label}
        for metric in dataframe.STEADY_PHYSICS_METRICS:
            values = _values(frame, metric, len(frame))
            for statistic, reducer in statistics:
                row[f"{metric} {statistic}"] = float(reducer(values)) if values.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def plot_physical_consistency_summary_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Return the automatically rendered colored summary table."""
    summary = build_physical_consistency_summary_table(datasets)
    title, count_headings = layout.aggregate_title_context(
        "Physical consistency summary",
        layout.effective_case_counts(datasets),
    )
    if any(count_headings.values()):
        summary = summary.rename(index={label: count_headings[label] for label in datasets})
    return tables.styled_dataframe(summary, title=title)


def plot_flux_balance(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot the flux-balance distribution beside its mean per-row profile: a row per dataset, the models as lines."""
    dataframe.validate_comparison(datasets, require_physics=True)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {key: active_session.full_summary(frame, max_cases).require_residuals() for key, frame in datasets.items()}
    figure_title, count_headings = layout.dataset_title_context(
        "Mass conservation: cross-section flux balance", groups, {key: summary.sample_count for key, summary in summaries.items()}
    )
    figure, axes = figure_layout.panel_figure(len(groups), 2)
    for (cdf_axis, profile_axis), models in zip(axes, groups.values(), strict=True):
        for model, key in models.items():
            style, summary = styles[key], summaries[key]
            _cdf(cdf_axis, _values(datasets[key], "flux_balance_mse", max_cases), style, name=model, case_count=min(max_cases, len(datasets[key])))
            profile_axis.plot(
                _log_axis_values(cast("np.ndarray", summary.flux_balance_mean)),
                summary.grid.coordinates[1, :, 0],
                linewidth=2,
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                markevery=layout.MARK_EVERY,
            )
        cdf_axis.set_xlabel(f"{_RESIDUAL_LABELS['flux_balance_mse']} [{_DIMENSIONLESS}]")
        cdf_axis.set_ylabel("CDF")
        figure_layout.set_panel_title(cdf_axis, "Per-case distribution")
        profile_axis.set_xlabel(rf"mean $|q(y)-\bar{{q}}|/Q$ [{_DIMENSIONLESS}]")
        profile_axis.set_ylabel(f"y [{summaries[next(iter(models.values()))].grid.coordinate_units[1]}]")
        figure_layout.set_panel_title(profile_axis, "Mean profile over the bed height")
    for axis in axes.flat:
        figure_layout.set_log_scale(axis, "x")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    _finish(figure, axes, figure_title, datasets, groups, styles, count_headings)
    return figure


def plot_brinkman_residual(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot momentum-residual distribution and its corresponding map."""
    return _metric_map_composite(
        datasets=datasets,
        max_cases=max_cases,
        metric="momentum_residual_mse",
        attribute="momentum_mean",
        title="Darcy–Brinkman momentum residual",  # noqa: RUF001
        colorbar_label=_RESIDUAL_COLORBAR_LABEL,
    )


def _pressure_drop_values(
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
) -> dict[str, np.ndarray]:
    """Return the pressure-drop mismatch relative to the declared drop per admitted case; a non-positive declared drop fails."""
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_pressure() for frame in datasets.values()}
    result = {}
    for label, frame in datasets.items():
        summary = summaries[id(frame)]
        declared = cast("np.ndarray", summary.pressure_declared)
        absolute = cast("np.ndarray", summary.pressure_absolute_error)
        if not np.all(declared > 0.0):
            msg = f"{label}: every declared pressure drop must be positive, got a minimum of {float(np.min(declared)):.3g} Pa."
            raise ValueError(msg)
        result[label] = validity_contract.finite_case_values(absolute / declared)
    return result


def plot_pressure_consistency(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Place pressure-drop and boundary-condition CDFs in one honest view: a row per dataset, the models as lines."""
    dataframe.validate_comparison(datasets, require_physics=True)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    pressure_drop = _pressure_drop_values(datasets, max_cases)
    pressure_boundary = {key: _values(frame, "pressure_boundary_mse", max_cases) for key, frame in datasets.items()}
    figure_title, count_headings = layout.dataset_title_context(
        "Pressure and boundary-condition consistency", groups, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    figure, axes = figure_layout.panel_figure(len(groups), 2)
    specifications = (
        (0, pressure_drop, "Pressure-drop mismatch", rf"$|\Delta p_{{pred}}-\Delta p_{{bc}}|/\Delta p_{{bc}}$ [{_DIMENSIONLESS}]"),
        (1, pressure_boundary, "Pressure-boundary mismatch", f"{_RESIDUAL_LABELS['pressure_boundary_mse']} [{_DIMENSIONLESS}]"),
    )
    for row, models in enumerate(groups.values()):
        for column, values, axis_title, xlabel in specifications:
            axis = axes[row, column]
            for model, key in models.items():
                _cdf(axis, values[key], styles[key], name=model, case_count=min(max_cases, len(datasets[key])))
            figure_layout.set_log_scale(axis, "x")
            axis.set_xlabel(xlabel)
            figure_layout.set_panel_title(axis, axis_title)
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
    figure_layout.label_rows_once(axes, "CDF")
    _finish(figure, axes, figure_title, datasets, groups, styles, count_headings)
    return figure


def _residual_maps(
    summaries: Sequence[sessions.FullEvaluationSummary],
    headings: Sequence[str],
    *,
    title: str,
    attribute: str,
    colorbar_label: str,
    above_in: float = 0.0,
    right_in: float = 0.0,
) -> tuple[Figure, map_layout.MapGeometry]:
    """
    Draw one mean residual map per summary in a row on the shared fixed map grid and return the figure and its geometry.

    The columns follow the selection, so the figure grows with them (no width cap). The maps share one round scale, locked over
    them, from 0 to the rounded 95th percentile of their pooled finite values; each bar carries ``colorbar_label`` with its unit and an
    arrow only if its own map has values beyond. ``above_in`` and ``right_in`` reserve the band of a line panel.
    """
    values = [np.asarray(getattr(summary, attribute), dtype=np.float64) for summary in summaries]
    extents = {summary.grid.extent for summary in summaries}
    if len(extents) != 1:
        msg = "The maps of one figure share one physical extent, but the selected datasets have different grids."
        raise ValueError(msg)
    extent = next(iter(extents))
    pooled = np.concatenate([field[np.isfinite(field)] for field in values])
    boundaries = field_maps.error_boundaries(pooled, signed=False)
    definition = field_maps.build_field_map(
        np.append(pooled, boundaries[[0, -1]]),  # the scale's own ends keep a uniform residual from being a constant map
        semantic=field_maps.FieldMapSemantic.ABSOLUTE_ERROR,
        field=attribute,
        unit="1",
        locked_boundaries=boundaries,
    )
    geometry = map_layout.plan_geometry(
        1, len(summaries), (extent[1] - extent[0], extent[3] - extent[2]), row_labels=False, width_cap=None, above_in=above_in, right_in=right_in
    )
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    for column, (summary, field) in enumerate(zip(summaries, values, strict=True)):
        image = axes[0, column].pcolormesh(*summary.grid.coordinates, np.ma.masked_invalid(field), shading="auto", **definition.pcolormesh_kwargs())
        extend = field_maps.locked_extend(field[np.isfinite(field)], boundaries)
        map_layout.add_field_map_colorbar(figure, image, slots[0, column], definition, label=colorbar_label, extend=extend)
    map_layout.set_column_headers(axes, headings, ())
    map_layout.set_map_title(figure, title)
    layout.add_scale_note(figure, shared=False)
    return figure, geometry


def _metric_map_composite(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
    metric: str,
    attribute: str,
    title: str,
    colorbar_label: str,
) -> Figure:
    """Compose the scalar-residual CDFs in a band above one row of matching maps, a column per dataset."""
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_residuals() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        f"{title}: distribution and map",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    headings = {label: count_headings[label] or label for label in datasets}
    figure, geometry = _residual_maps(
        [summaries[id(frame)] for frame in datasets.values()],
        list(headings.values()),
        title=figure_title,
        attribute=attribute,
        colorbar_label=colorbar_label,
        above_in=_CDF_BAND_IN,
    )
    styles = series.series_styles(datasets)
    line_axis = figure.add_axes(map_layout.band_rect(geometry, "above"))
    for label, frame in datasets.items():
        _cdf(line_axis, _values(frame, metric, max_cases), styles[label], name=label, case_count=min(max_cases, len(frame)), label=headings[label])
    figure_layout.set_log_scale(line_axis, "x")
    line_axis.set_xlabel(f"{_RESIDUAL_LABELS[metric]} [{_DIMENSIONLESS}]")
    line_axis.set_ylabel("CDF")
    line_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    if line_axis.lines:
        line_axis.legend(loc="best")
    return figure


def plot_physical_consistency_cdf_grid(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Render the four residual CDFs of every dataset as a row of panels, the models as lines, with one legend beside them."""
    dataframe.validate_comparison(datasets, require_physics=True)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    pressure_drop = _pressure_drop_values(datasets, max_cases)
    figure_title, count_headings = layout.dataset_title_context(
        "Physical consistency distributions", groups, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    specifications = (
        ("flux_balance_mse", _RESIDUAL_LABELS["flux_balance_mse"], "Mass conservation (flux balance)"),
        ("momentum_residual_mse", _RESIDUAL_LABELS["momentum_residual_mse"], "Brinkman residual"),
        (None, r"$|\Delta p_{pred}-\Delta p_{bc}|/\Delta p_{bc}$", "Pressure drop"),
        ("pressure_boundary_mse", _RESIDUAL_LABELS["pressure_boundary_mse"], "Pressure BC"),
    )
    figure, axes = figure_layout.panel_figure(len(groups), len(specifications))
    for row, models in enumerate(groups.values()):
        for axis, (metric, xlabel, title) in zip(axes[row], specifications, strict=True):
            for model, key in models.items():
                values = pressure_drop[key] if metric is None else _values(datasets[key], metric, max_cases)
                _cdf(axis, values, styles[key], name=model, case_count=min(max_cases, len(datasets[key])))
            figure_layout.set_log_scale(axis, "x")
            axis.set_xlabel(f"{xlabel} [{_DIMENSIONLESS}]")
            figure_layout.set_panel_title(axis, title)
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
    figure_layout.label_rows_once(axes, "CDF")
    _finish(figure, axes, figure_title, datasets, groups, styles, count_headings)
    return figure


def plot_spatial_residuals(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int | None = None,
) -> Figure:
    """
    Plot the mean momentum-residual map of every dataset side by side and one flux-balance profile of all of them right of the maps.

    The profile panel stands in a band of the maps' height and y range, so each profile row lies at the height of its map row; its lines
    take the datasets' fixed colours and line styles and its legend stands inside it.
    """
    limit = min(len(frame) for frame in datasets.values()) if max_cases is None else max_cases
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, limit).require_residuals() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Spatial residuals",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    headings = {label: count_headings[label] or label for label in datasets}
    figure, geometry = _residual_maps(
        [summaries[id(frame)] for frame in datasets.values()],
        list(headings.values()),
        title=figure_title,
        attribute="momentum_mean",
        colorbar_label=_RESIDUAL_COLORBAR_LABEL,
        right_in=_PROFILE_BAND_IN,
    )
    styles = series.series_styles(datasets)
    profile_axis = figure.add_axes(map_layout.band_rect(geometry, "right"))
    for label, frame in datasets.items():
        summary = summaries[id(frame)]
        profile_axis.plot(
            _log_axis_values(cast("np.ndarray", summary.flux_balance_mean)),
            summary.grid.coordinates[1, :, 0],
            linewidth=2,
            label=headings[label],
            color=styles[label].color,
            linestyle=styles[label].linestyle,
            marker=styles[label].marker,
            markevery=layout.MARK_EVERY,
        )
    extent = next(iter(summaries.values())).grid.extent
    profile_axis.set_ylim(extent[2], extent[3])
    profile_axis.tick_params(labelleft=False)
    figure_layout.set_log_scale(profile_axis, "x")
    profile_axis.set_xlabel(r"mean $|q(y)-\bar{q}|/Q$ [-]")
    profile_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    profile_axis.legend(loc="best")
    map_layout.set_column_headers(np.array([[profile_axis]], dtype=object), ["Mean flux-balance profile"], ())
    return figure


def _median(values: np.ndarray) -> float:
    """Return the median of the finite case values, NaN without any."""
    return float(np.median(values)) if values.size else float("nan")


def build_pressure_boundary_summary(datasets: Mapping[str, pd.DataFrame], max_cases: int = _DEFAULT_CASE_LIMIT) -> pd.DataFrame:
    """Build the medians of the pressure-boundary components and of the pressure-drop error over the finite cases of the saved prefix."""
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {label: active_session.full_summary(frame, max_cases).require_pressure() for label, frame in datasets.items()}
    rows = []
    for label, frame in datasets.items():
        rows.append(
            {
                "Model": label,
                **{name: _median(_values(frame, column, max_cases)) for name, column in _PRESSURE_COMPONENTS},
                _PRESSURE_DROP_COLUMN: _median(validity_contract.finite_case_values(summaries[label].pressure_absolute_error)),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def plot_pressure_boundary_summary(*, datasets: dataset_cuts.DisplayFrames, max_cases: int = _DEFAULT_CASE_LIMIT) -> Figure:
    """Compare the median pressure-boundary components ([-]) and the pressure-drop error (Pa) on two panels per dataset row, a bar per model."""
    table = build_pressure_boundary_summary(datasets, max_cases)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    figure_title, count_headings = layout.dataset_title_context(
        "Pressure boundary and drop summary", groups, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    figure, axes = figure_layout.panel_figure(len(groups), 2, width_ratios=(1.6, 1.0))
    panels = (
        (0, [name for name, _column in _PRESSURE_COMPONENTS], f"Median [{_DIMENSIONLESS}]"),
        (1, [_PRESSURE_DROP_COLUMN], f"Median absolute error [{field_labels.display_unit('Pa')}]"),
    )
    for row, models in enumerate(groups.values()):
        width = 0.8 / len(models)
        for column, columns, ylabel in panels:
            axis = axes[row, column]
            positions = np.arange(len(columns), dtype=float)
            shown: list[float] = []
            for offset, (model, key) in enumerate(models.items()):
                medians = table.loc[key, columns].to_numpy(dtype=float)
                if not np.isfinite(medians).any():
                    layout.mark_empty_prefix(axis, case_count=min(max_cases, len(datasets[key])), model=model)
                    continue
                shown.extend(medians[np.isfinite(medians)])
                bars = positions + (offset - (len(models) - 1) / 2.0) * width
                axis.bar(bars, medians, width, color=styles[key].color, edgecolor="white", hatch=styles[key].hatch)
            axis.set_xticks(positions, columns, rotation=25, ha="right")
            axis.set_ylabel(ylabel)
            axis.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
            if shown and min(shown) > 0.0:
                figure_layout.set_log_scale(axis, "y")
    _finish(figure, axes, figure_title, datasets, groups, styles, count_headings, bars=True)
    return figure
