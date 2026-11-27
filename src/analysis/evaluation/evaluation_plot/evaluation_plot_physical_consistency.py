"""Render physical-consistency CDFs, maps, and tables from current evidence.

Responsibilities:
  - Summarize dimensionless momentum, flux-balance, and pressure-boundary metrics, the table's rows headed by what they are
  - Plot scalar residual CDFs on separate panels with one legend beside them
  - Compare the median pressure-boundary components and the pressure-drop error as points on two panels, naming every model and
    component without a finite median
  - Render the full-grid momentum magnitude map of every dataset side by side on the shared fixed map grid, with the residual
    CDFs in a band above them or one flux-balance profile in a band right of them on the maps' y range; a comparison's title
    names the one dataset whose models are the columns
  - Draw flux-balance profiles as thin lines with about ten markers each (a dotted role thinner, with half as many between the
    others'), their key placed by the shared single-panel legend in a free corner, else with headroom from the profile's own value
    axis, never from the maps' y range, and in more columns where one column would stand taller than the panel
  - Calculate pressure-drop mismatch from admitted pressure and boundary fields
  - Pair only scalar distributions and maps that represent the same quantity
  - Present descriptive physical statistics in colored tables

Design principles:
  - Scalar metrics and full-grid arrays retain their declared distinct semantics
  - Residual metrics are dimensionless and pressure-drop evidence stays in pascals, so the two never share an axis
  - Complete or bounded case reductions are owned by EvaluationSession
  - Display clipping never changes immutable residual or pressure evidence
  - Per-case distributions leave out non-finite cases, which the invalid-prediction warning names
  - Residual maps share one round scale per dataset, locked over its runs and named in the scale note, each bar with its unit and an
    arrow only where its map has values beyond, a map without a finite value unavailable; every line or bar takes its model's fixed
    series style (``evaluation_plot_series``)
  - Line and point figures draw a row of panels per dataset and the models as lines or points inside it

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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_error_behavior as error_behavior
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as map_layout
from src.analysis.ui import analysis_ui_tables as tables

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import ipywidgets as widgets
    from matplotlib.axes import Axes
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    from src.analysis.presentation import analysis_visual_semantics as visual_semantics

_DEFAULT_CASE_LIMIT = 100
_CDF_BAND_IN = 3.0  # height of the residual-CDF band above the maps
_PROFILE_BAND_IN = 3.0  # least width of the flux-balance profile band right of the maps
# The most of that band the padded key may take: the data, with Matplotlib's 5 % margin below them, keep (1 - 0.36) / 1.05 = 61 %
# of a value axis widened for the key.
_KEY_SHARE = 0.36
_KEY_PAD_IN = 0.15  # beside and under the key's frame: Matplotlib's border pad (4.5 pt), the legend helper's clearance (3 pt) and a margin
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
_POINT_SIZE = 7.0  # a median of the pressure-boundary summary
_POINT_MARGIN = 0.12  # the y margin of its panels, as a share of their range
_PANEL_GAP = 0.05  # extra space between the summary's two panels (of the mean panel width): the right y label clears the left frame
_PROFILE_WIDTH = 1.2
_DOTTED_PROFILE_SCALE = 0.75  # a dotted profile's stroke relative to a solid one
_PROFILE_MARKERS = 10  # about this many markers on a flux-balance profile, whatever its row count


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
    points: bool = False,
    note: str | None = None,
) -> None:
    """Write the dataset header of every row, the legend naming the models (and differing roles of lines) and the title with its ``note``."""
    figure_layout.set_row_headers(figure, axes, [count_headings[dataset] or dataset for dataset in groups])
    layout.add_series_legend(figure, axes, datasets, groups, styles, points=points, name_roles=not points)  # a row header names its role
    figure_layout.set_figure_title(figure, axes, title, note=layout.series_note(styles, note))


def _row_heading(labels: Iterable[tuple[str, str]]) -> str:
    """Return what the rows of a table of frames with these (dataset, model) ``labels`` are: datasets, models, or model · dataset pairs."""
    pairs = list(labels)
    if len({model for _dataset, model in pairs}) == 1:
        return "Dataset"
    return "Model" if len({dataset for dataset, _model in pairs}) == 1 else "Model · dataset"


def _key_size_in(entries: Sequence[tuple[str, visual_semantics.SeriesStyle]], panel_in: float) -> tuple[int, float, float]:
    """
    Return the columns of a key naming the (label, style) ``entries`` of profile lines, as the shared legend draws it, and its padded size.

    The size is the width and height in inches with ``_KEY_PAD_IN``; the key takes one more column while it would stand taller than a
    panel ``panel_in`` high, at most one per entry.
    """
    probe = Figure()
    axis = probe.add_subplot()
    handles = [Line2D([], [], color=style.color, linestyle=style.linestyle, marker=style.marker, linewidth=_PROFILE_WIDTH) for _, style in entries]
    columns = 0
    while True:
        columns += 1
        legend = axis.legend(handles, [label for label, _ in entries], fontsize=figure_layout.TICK_PT, ncols=columns)  # laid out once, so anew
        probe.draw_without_rendering()
        box = legend.get_window_extent()
        width, height = box.width / probe.dpi + _KEY_PAD_IN, box.height / probe.dpi + _KEY_PAD_IN
        if height <= panel_in or columns >= len(entries):
            return columns, width, height


def _clear_profile_key(axis: Axes, key: tuple[float, float], *, widen: bool) -> None:
    """
    Widen the profile panel's log value axis on its high side until a key of ``key`` (padded width, height as panel shares) clears the right.

    Unless ``widen`` is set, a free corner comes first: within the band of rows a corner's key spans, a profile reaches farthest at a
    vertex there or where it crosses the band's edge, as a line is straight between rows. The widening puts the finite data maximum at
    ``1 - key[0]`` of the axis and its minimum a 5 % margin above the low end, both from the finite data, never from the autoscaled
    limits, so the data's own extrema fill at least 60 % of it (``_KEY_SHARE``) whatever the margins; a flat profile spans one decade.
    """
    low, high = np.log10(axis.get_xlim())
    bottom, top = axis.get_ylim()
    profiles = []
    for line in axis.lines:
        x, y = (np.asarray(values, dtype=float) for values in line.get_data())
        kept = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
        order = np.argsort(y[kept])
        if order.size:
            profiles.append((((np.log10(x[kept]) - low) / (high - low))[order], ((y[kept] - bottom) / (top - bottom))[order]))
    if not profiles:
        return
    width, height = key
    for lower, upper in ((0.0, height), (1.0 - height, 1.0)):
        reach = np.concatenate([np.append(x[(y >= lower) & (y <= upper)], np.interp((lower, upper), y, x)) for x, y in profiles])
        if not widen and (reach.max() < 1.0 - width or reach.min() > width):
            return
    # The limits come from the finite data extrema, never the autoscaled ones: a 5 % margin below them, the key's share above.
    first, last = (low + (high - low) * edge for edge in (min(float(x.min()) for x, _y in profiles), max(float(x.max()) for x, _y in profiles)))
    if last <= first:  # a flat profile spans one decade around its value
        first, last = first - 0.5, last + 0.5
    left = first - 0.05 * (last - first)
    axis.set_xlim(10.0**left, 10.0 ** (left + (last - left) / (1.0 - width)))


def _plot_profile(axis: Axes, summary: sessions.FullEvaluationSummary, style: visual_semantics.SeriesStyle, label: str | None = None) -> None:
    """
    Draw one mean flux-balance profile over the bed height as a thin line in ``style`` with about ten markers, whatever its row count.

    A dotted role (material OOD) is drawn thinner with half the markers, set off by half a step from the others' markers, so its dots
    and markers stay a readable trace beside the solid lines instead of a cloud.
    """
    heights = summary.grid.coordinates[1, :, 0]
    dotted = style.linestyle in {":", "dotted"}
    step = max(1, heights.size // _PROFILE_MARKERS)
    axis.plot(
        _log_axis_values(cast("np.ndarray", summary.flux_balance_mean)),
        heights,
        linewidth=_PROFILE_WIDTH * (_DOTTED_PROFILE_SCALE if dotted else 1.0),
        label=label,
        color=style.color,
        linestyle=style.linestyle,
        marker=style.marker,
        markevery=(step // 2, 2 * step) if dotted else step,
    )


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
    """
    Return the summary table with display headings, each residual column shaded against its own values (lower is better).

    The row heading names what the rows of the displayed frames are; the builder keeps its raw ``<metric> <statistic>`` keys.
    """
    summary = build_physical_consistency_summary_table(datasets).rename(columns=lambda key: display_labels.table_heading(f"{key} [1]"))
    title, count_headings = layout.aggregate_title_context(
        "Physical consistency summary",
        layout.effective_case_counts(datasets),
    )
    if any(count_headings.values()):
        summary = summary.rename(index={label: count_headings[label] for label in datasets})
    heading = _row_heading(datasets.labels.values()) if isinstance(datasets, dataset_cuts.DisplayFrames) else None
    return tables.styled_dataframe(summary, title=title, row_heading=heading, shade_columns=list(summary.columns))


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
        "Mass conservation: cross-section flux balance", datasets, {key: summary.sample_count for key, summary in summaries.items()}
    )
    figure, axes = figure_layout.panel_figure(len(groups), 2)
    for (cdf_axis, profile_axis), models in zip(axes, groups.values(), strict=True):
        for model, key in models.items():
            values = _values(datasets[key], "flux_balance_mse", max_cases)
            _cdf(cdf_axis, values, styles[key], name=model, case_count=min(max_cases, len(datasets[key])))
            _plot_profile(profile_axis, summaries[key], styles[key])
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
        "Pressure and boundary-condition consistency", datasets, layout.effective_case_counts(datasets, max_cases=max_cases)
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
    summaries: Mapping[str, sessions.FullEvaluationSummary],
    headings: Sequence[str],
    *,
    groups: Sequence[Sequence[str]],
    title: str,
    attribute: str,
    colorbar_label: str,
    above_in: float = 0.0,
    right_in: float = 0.0,
) -> tuple[Figure, map_layout.MapGeometry]:
    """
    Draw one mean residual map per summary in a row on the shared fixed map grid and return the figure and its geometry.

    The columns follow the selection, so the figure grows with them (no width cap). The maps of the keys of one of ``groups`` (the runs
    of a dataset) share one round scale, locked over them, from 0 to the rounded 95th percentile of their pooled finite values; each
    bar carries ``colorbar_label`` with its unit and an arrow only if its own map has values beyond. A map without a finite value, as
    of a dataset whose cases are all non-finite, says it is unavailable. ``above_in`` and ``right_in`` reserve the band of a line panel.
    """
    values = {key: np.asarray(getattr(summary, attribute), dtype=np.float64) for key, summary in summaries.items()}
    extents = {summary.grid.extent for summary in summaries.values()}
    if len(extents) != 1:
        msg = "The maps of one figure share one physical extent, but the selected datasets have different grids."
        raise ValueError(msg)
    extent = next(iter(extents))
    definitions: dict[str, tuple[field_maps.FieldMapDefinition, np.ndarray] | None] = {}
    for group in groups:
        pooled = np.concatenate([values[key][np.isfinite(values[key])] for key in group])
        if not pooled.size:
            definitions.update(dict.fromkeys(group))
            continue
        boundaries = field_maps.error_boundaries(pooled, signed=False)
        definition = field_maps.build_field_map(
            np.append(pooled, boundaries[[0, -1]]),  # the scale's own ends keep a uniform residual from being a constant map
            semantic=field_maps.FieldMapSemantic.ABSOLUTE_ERROR,
            field=attribute,
            unit="1",
            locked_boundaries=boundaries,
        )
        definitions.update(dict.fromkeys(group, (definition, boundaries)))
    geometry = map_layout.plan_geometry(
        1, len(summaries), (extent[1] - extent[0], extent[3] - extent[2]), row_labels=False, width_cap=None, above_in=above_in, right_in=right_in
    )
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    for column, (key, summary) in enumerate(summaries.items()):
        field, scale = values[key], definitions[key]
        if scale is None or not np.isfinite(field).any():
            map_layout.empty_cell(axes[0, column], "Unavailable: no finite values")
            continue
        definition, boundaries = scale
        image = axes[0, column].pcolormesh(*summary.grid.coordinates, np.ma.masked_invalid(field), shading="auto", **definition.pcolormesh_kwargs())
        extend = field_maps.locked_extend(field[np.isfinite(field)], boundaries)
        map_layout.add_field_map_colorbar(figure, image, slots[0, column], definition, label=colorbar_label, extend=extend)
    map_layout.set_column_headers(axes, headings, ())
    map_layout.set_map_title(figure, title)
    layout.add_scale_note(figure, shared=False, scale=error_behavior.dataset_scale_note(signed=False, compared=len(summaries) > 1))
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
        summaries = {label: active_session.full_summary(frame, max_cases).require_residuals() for label, frame in datasets.items()}
    figure_title, count_headings = layout.aggregate_title_context(
        f"{title}: distribution and map{layout.dataset_suffix(datasets)}",
        {label: summary.sample_count for label, summary in summaries.items()},
        models=datasets.title_labels.values() if isinstance(datasets, dataset_cuts.DisplayFrames) else (),
        eligible={label: len(frame) for label, frame in datasets.items()},  # a bounded prefix reads «first n of N cases»
    )
    headings = {label: count_headings[label] or label for label in datasets}
    figure, geometry = _residual_maps(
        summaries,
        list(headings.values()),
        groups=error_behavior.dataset_scale_groups(datasets),
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
        "Physical consistency distributions", datasets, layout.effective_case_counts(datasets, max_cases=max_cases)
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

    The profile panel stands in a band of the maps' height and shares their y axis (``sharey``), so each profile row lies at the height of
    its map row; its lines take the datasets' fixed colours and line styles. Its key stands inside it, placed by the shared single-panel
    legend in a free corner; without one, the profile's own log value (x) axis widens on its high side until the key is clear
    (``_clear_profile_key``), the band being wide enough that the profiles' extrema keep at least 60 % of that axis. A key that would
    stand taller than the panel takes more columns, which the shared legend, drawing one, cannot give it: the axis then always widens
    and the key stands in the upper right.
    """
    limit = min(len(frame) for frame in datasets.values()) if max_cases is None else max_cases
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {label: active_session.full_summary(frame, limit).require_residuals() for label, frame in datasets.items()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Spatial residuals" + layout.dataset_suffix(datasets),
        {label: summary.sample_count for label, summary in summaries.items()},
        models=datasets.title_labels.values() if isinstance(datasets, dataset_cuts.DisplayFrames) else (),
        eligible={label: len(frame) for label, frame in datasets.items()},  # a bounded prefix reads «first n of N cases»
    )
    headings = {label: count_headings[label] or label for label in datasets}
    styles = series.series_styles(datasets)
    extent = next(iter(summaries.values())).grid.extent
    domain = (extent[1] - extent[0], extent[3] - extent[2])
    panel_in = map_layout.plan_geometry(1, len(summaries), domain, row_labels=False, width_cap=None).grid_height
    columns, key_width, key_height = _key_size_in([(headings[label], styles[label]) for label in datasets], panel_in)
    band_in = max(_PROFILE_BAND_IN, key_width / _KEY_SHARE)
    figure, geometry = _residual_maps(
        summaries,
        list(headings.values()),
        groups=error_behavior.dataset_scale_groups(datasets),
        title=figure_title,
        attribute="momentum_mean",
        colorbar_label=_RESIDUAL_COLORBAR_LABEL,
        right_in=band_in,
    )
    profile_axis = figure.add_axes(map_layout.band_rect(geometry, "right"))
    panel = np.array([[profile_axis]], dtype=object)
    for label, summary in summaries.items():
        _plot_profile(profile_axis, summary, styles[label], label=headings[label])
    profile_axis.set_ylim(extent[2], extent[3])
    profile_axis.sharey(figure.axes[0])  # the first map: the shared-axis guard of the key then never moves y, only x
    profile_axis.tick_params(labelleft=False)
    figure_layout.set_log_scale(profile_axis, "x")
    # The shared legend places a one-column key: one taller than the panel needs the high side cleared for its columns.
    _clear_profile_key(profile_axis, (key_width / band_in, key_height / panel_in), widen=columns > 1)
    profile_axis.set_xlabel(r"mean $|q(y)-\bar{q}|/Q$ [-]")
    profile_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    entries = figure_layout.legend_entries([profile_axis])
    if columns > 1:  # in the upper right, which the widening cleared over the panel's full height
        profile_axis.legend(*entries, loc="upper right", ncols=columns, fontsize=figure_layout.TICK_PT)
    else:
        figure_layout.add_legend(figure, panel, *entries)  # the single-panel key rule places it
    map_layout.set_column_headers(panel, ["Mean flux-balance profile"], ())
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
    """
    Compare the median pressure-boundary components ([-]) and the pressure-drop error (Pa) on two panels per dataset row, a point per model.

    Points, not bars: on the log axis of positive medians a bar's length would suggest a baseline that does not exist. A zero median
    keeps the panel linear, where its point stands at 0. A model without any finite median of a panel is named on the panel; one
    that lacks only some keeps its other points and their scale, and the note names it and the components it lacks.
    """
    table = build_pressure_boundary_summary(datasets, max_cases)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    figure_title, count_headings = layout.dataset_title_context(
        "Pressure boundary and drop summary", datasets, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    figure, axes = figure_layout.panel_figure(len(groups), 2, width_ratios=(1.6, 1.0))
    cast("ConstrainedLayoutEngine", figure.get_layout_engine()).set(wspace=_PANEL_GAP)
    panels = (
        (0, [name for name, _column in _PRESSURE_COMPONENTS], f"Median [{_DIMENSIONLESS}]"),
        (1, [_PRESSURE_DROP_COLUMN], f"Median absolute error [{field_labels.display_unit('Pa')}]"),
    )
    gaps: list[str] = []
    for row, (dataset, models) in enumerate(groups.items()):
        width = 0.8 / len(models)
        for column, columns, ylabel in panels:
            axis = axes[row, column]
            names = [name[0].upper() + name[1:] for name in columns]  # the table's keys, capitalized only where drawn
            positions = np.arange(len(columns), dtype=float)
            shown: list[float] = []
            for offset, (model, key) in enumerate(models.items()):
                medians = table.loc[key, columns].to_numpy(dtype=float)
                finite = np.isfinite(medians)
                if not finite.any():
                    layout.mark_empty_prefix(axis, case_count=min(max_cases, len(datasets[key])), model=model)
                    continue
                if not finite.all():
                    gaps.append(f"{model} on {dataset} ({', '.join(name for name, ok in zip(names, finite, strict=True) if not ok)})")
                shown.extend(medians[finite])
                x = positions + (offset - (len(models) - 1) / 2.0) * width
                style = styles[key]  # the row header names the role
                axis.plot(x, medians, linestyle="", marker=style.marker or "o", markersize=_POINT_SIZE, color=style.color)
            axis.set_xticks(positions, names, rotation=25, ha="right")
            axis.set_xlim(-0.5, len(columns) - 0.5)
            axis.margins(y=_POINT_MARGIN)  # no point touches the frame
            axis.set_ylabel(ylabel)
            axis.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
            if shown and min(shown) > 0.0:
                figure_layout.set_log_scale(axis, "y")
    note = "Not plotted, no finite median: " + "; ".join(gaps) if gaps else None
    _finish(figure, axes, figure_title, datasets, groups, styles, count_headings, points=True, note=note)
    return figure
