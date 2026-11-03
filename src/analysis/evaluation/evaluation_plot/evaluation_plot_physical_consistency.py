"""Render physical-consistency CDFs, maps, and tables from current evidence.

Responsibilities:
  - Summarize dimensionless momentum, flux-balance, and pressure-boundary metrics
  - Plot scalar residual CDFs with separate axes and legends
  - Render full-grid momentum magnitude maps and per-row flux-balance profiles
  - Calculate pressure-drop mismatch from admitted pressure and boundary fields
  - Pair only scalar distributions and maps that represent the same quantity
  - Present descriptive physical statistics in colored tables

Design principles:
  - Scalar metrics and full-grid arrays retain their declared distinct semantics
  - Residual metrics are dimensionless; pressure-drop evidence stays in pascals
  - Complete or bounded case reductions are owned by EvaluationSession
  - Display clipping never changes immutable residual or pressure evidence

This module does NOT:
  - Recompute residual equations, derivatives, crops, or boundary masks
  - Fabricate an unavailable independent ground-truth physics baseline
  - Derive a composite physics score
  - Admit artifacts or own notebook control composition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 100
_POSITIVE_FLOOR = np.finfo(float).tiny
_RESIDUAL_LABELS = {
    "momentum_residual_mse": (r"MSE of Brinkman momentum residual $(R_x^2+R_y^2)/G^2$", "1"),
    "flux_balance_mse": (r"MSE of cross-section flux deviation $(q(y)-\bar{q})/Q$", "1"),
}


def _values(frame: pd.DataFrame, column: str, max_cases: int) -> np.ndarray:
    """Return finite non-negative scalar evidence from one saved prefix."""
    values = np.asarray(pd.to_numeric(frame.iloc[:max_cases][column], errors="raise"), dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        msg = f"Physical metric {column!r} must be finite and non-negative."
        raise ValueError(msg)
    return values


def _cdf(axis: Axes, values: np.ndarray, *, label: str, color: object | None = None) -> Line2D:
    """Draw one log-safe empirical CDF."""
    ordered = np.maximum(np.sort(np.asarray(values, dtype=float)), _POSITIVE_FLOOR)
    cumulative = np.linspace(0.0, 1.0, ordered.size)
    (line,) = axis.plot(ordered, cumulative, linewidth=2, label=label, color=color)
    return line


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
                row[f"{metric} {statistic}"] = float(reducer(values))
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def _blue_style(table: pd.DataFrame) -> pd.DataFrame:
    """Return quantile-bounded blue numeric cell fills."""
    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    cmap = plt.get_cmap("Blues")
    for column in table.columns:
        values = np.asarray(pd.to_numeric(table[column], errors="coerce"), dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        low, high = np.quantile(finite, (0.05, 0.95))
        if np.isclose(low, high):
            continue
        for row, value in enumerate(values):
            if not np.isfinite(value):
                continue
            fraction = float(np.clip((value - low) / (high - low), 0.0, 1.0))
            red, green, blue, _alpha = cmap(0.05 + 0.90 * fraction)
            column_index = cast("int", styles.columns.get_loc(column))
            styles.iloc[row, column_index] = f"background-color: rgba({int(red * 255)}, {int(green * 255)}, {int(blue * 255)}, 0.55)"
    return styles


def plot_physical_consistency_summary_table(*, datasets: Mapping[str, pd.DataFrame]) -> widgets.VBox:
    """Return the automatically rendered colored summary table."""
    summary = build_physical_consistency_summary_table(datasets)
    title, count_headings = layout.aggregate_title_context(
        "Physical consistency summary",
        layout.effective_case_counts(datasets),
    )
    if any(count_headings.values()):
        summary = summary.rename(index={label: count_headings[label] for label in datasets})
    styler = summary.style
    styler.format("{:.4g}")
    styler.apply(lambda _table: _blue_style(summary), axis=None)
    return widgets.VBox((widgets.HTML(f"<h2>{title}</h2>"), widgets.HTML(styler.to_html())))


def plot_flux_balance(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot the flux-balance distribution beside its mean per-row profile."""
    dataframe.validate_comparison(datasets, require_physics=True)
    metric_label, unit = _RESIDUAL_LABELS["flux_balance_mse"]
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_residuals() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Mass conservation: cross-section flux balance",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    colors = {label: plt.get_cmap("tab10")(index % 10) for index, label in enumerate(datasets)}
    figure = plt.figure(figsize=(13.0, 4.8))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.30), wspace=0.28)
    cdf_axis = figure.add_subplot(grid[0, 0])
    profile_axis = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[0, 2])
    legend_axis.axis("off")
    for label, frame in datasets.items():
        summary = summaries[id(frame)]
        heading = count_headings[label] or label
        _cdf(cdf_axis, _values(frame, "flux_balance_mse", max_cases), label=heading, color=colors[label])
        rows = summary.grid.coordinates[1, :, 0]
        profile_axis.plot(np.maximum(cast("np.ndarray", summary.flux_balance_mean), _POSITIVE_FLOOR), rows, linewidth=2, color=colors[label])
    cdf_axis.set_xscale("log")
    cdf_axis.set_xlabel(f"{metric_label} [{unit}]")
    cdf_axis.set_ylabel("CDF")
    cdf_axis.set_title("Per-case distribution")
    profile_axis.set_xscale("log")
    profile_axis.set_xlabel(r"mean $|q(y)-\bar{q}|/Q$ [1]")
    profile_axis.set_ylabel(f"y [{summaries[id(next(iter(datasets.values())))].grid.coordinate_units[1]}]")
    profile_axis.set_title("Mean profile over the bed height")
    for axis in (cdf_axis, profile_axis):
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    handles = [Line2D([0], [0], color=colors[label], linewidth=2, label=count_headings[label] or label) for label in datasets]
    legend_axis.legend(handles, [str(handle.get_label()) for handle in handles], loc="upper left", frameon=True)
    figure.subplots_adjust(top=0.84, bottom=0.16, left=0.07, right=0.98)
    layout.title_over_axes(figure, figure_title, (cdf_axis, profile_axis), y=0.96)
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
        colorbar_label=r"mean $\sqrt{R_x^2+R_y^2}/G$ [1]",
    )


def _pressure_drop_values(
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
) -> dict[str, np.ndarray]:
    """Return current absolute-relative pressure-drop mismatch per admitted case."""
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_pressure() for frame in datasets.values()}
    result = {}
    for label, frame in datasets.items():
        summary = summaries[id(frame)]
        declared = cast("np.ndarray", summary.pressure_declared)
        absolute = cast("np.ndarray", summary.pressure_absolute_error)
        result[label] = absolute / (np.abs(declared) + 1e-12)
    return result


def plot_pressure_consistency(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Place pressure-drop and boundary-condition CDFs in one honest view."""
    dataframe.validate_comparison(datasets, require_physics=True)
    pressure_drop = _pressure_drop_values(datasets, max_cases)
    pressure_boundary = {label: _values(frame, "pressure_boundary_mse", max_cases) for label, frame in datasets.items()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Pressure and boundary-condition consistency",
        layout.effective_case_counts(datasets, max_cases=max_cases),
    )
    colors = {label: plt.get_cmap("tab10")(index % 10) for index, label in enumerate(datasets)}
    figure = plt.figure(figsize=(13.0, 4.8))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.30), wspace=0.28)
    axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    legend_axis = figure.add_subplot(grid[0, 2])
    legend_axis.axis("off")
    specifications = (
        (
            axes[0],
            pressure_drop,
            "Pressure-drop mismatch",
            r"$|\Delta p_{pred}-\Delta p_{bc}|/(|\Delta p_{bc}|+\epsilon)$",
        ),
        (
            axes[1],
            pressure_boundary,
            "Pressure-boundary mismatch",
            r"$\mathrm{MSE}(p_\Gamma-p_{bc})/\Delta p^2$ [1]",
        ),
    )
    for axis, values, axis_title, xlabel in specifications:
        for label in datasets:
            _cdf(axis, values[label], label=count_headings[label] or label, color=colors[label])
        axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("CDF")
        axis.set_title(axis_title)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    handles = [Line2D([0], [0], color=colors[label], linewidth=2, label=count_headings[label] or label) for label in datasets]
    legend_axis.legend(handles, [str(handle.get_label()) for handle in handles], loc="upper left", frameon=True)
    figure.subplots_adjust(top=0.84, bottom=0.18, left=0.07, right=0.98)
    layout.title_over_axes(figure, figure_title, axes, y=0.96)
    return figure


def _metric_map_composite(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
    metric: str,
    attribute: str,
    title: str,
    colorbar_label: str,
) -> Figure:
    """Compose one scalar-residual line region directly above matching maps."""
    dataframe.validate_comparison(datasets, require_physics=True)
    metric_label, unit = _RESIDUAL_LABELS[metric]
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_residuals() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        f"{title}: distribution and map",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    figure = plt.figure(figsize=layout.MAP_LAYOUT.composite_size(map_columns=len(datasets)))
    grid = figure.add_gridspec(
        2,
        len(datasets),
        height_ratios=layout.MAP_LAYOUT.composite_height_ratios,
        hspace=layout.MAP_LAYOUT.composite_hspace,
        wspace=layout.MAP_LAYOUT.composite_group_spacing,
    )
    line_axis = figure.add_subplot(grid[0, :])
    handles = []
    for label, frame in datasets.items():
        handles.append(_cdf(line_axis, _values(frame, metric, max_cases), label=count_headings[label] or label))
    line_axis.set_xscale("log")
    line_axis.set_xlabel(f"{metric_label} [{unit}]")
    line_axis.set_ylabel("CDF")
    line_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    line_axis.legend(handles=handles, labels=[str(handle.get_label()) for handle in handles], loc="best", ncol=min(3, len(handles)))

    map_axes: list[Axes] = []
    for column, (label, frame) in enumerate(datasets.items()):
        summary = summaries[id(frame)]
        values = cast("np.ndarray", getattr(summary, attribute))
        upper = max(float(np.nanpercentile(values, 99.5)), np.finfo(float).eps)
        map_group = grid[1, column].subgridspec(1, 1)
        axis = figure.add_subplot(map_group[0, 0])
        map_axes.append(axis)
        image = axis.imshow(
            np.clip(values, 0.0, upper),
            origin="lower",
            extent=summary.grid.extent,
            aspect="equal",
            vmin=0.0,
            vmax=upper,
            interpolation="nearest",
        )
        heading = count_headings[label] or label
        axis.set_title(f"{heading} — mean spatial residual")
        layout.add_map_colorbar(figure, image, axis, label=colorbar_label)
    first_summary = summaries[id(next(iter(datasets.values())))]
    layout.apply_map_grid_axis_labels(
        (map_axes,),
        x_label=f"x [{first_summary.grid.coordinate_units[0]}]",
        y_label=f"y [{first_summary.grid.coordinate_units[1]}]",
    )
    figure.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.98)
    figure.suptitle(figure_title)
    return figure


def plot_physical_consistency_cdf_grid(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Render the approved 2 x 2 residual grid and external legend."""
    dataframe.validate_comparison(datasets, require_physics=True)
    colors = {label: plt.get_cmap("tab10")(index % 10) for index, label in enumerate(datasets)}
    pressure_drop = _pressure_drop_values(datasets, max_cases)
    figure_title, count_headings = layout.aggregate_title_context(
        "Physical consistency distributions",
        layout.effective_case_counts(datasets, max_cases=max_cases),
    )
    specifications = (
        ("flux_balance_mse", r"$\mathrm{MSE}((q(y)-\bar{q})/Q)$", "Mass conservation (flux balance)"),
        ("momentum_residual_mse", r"$\mathrm{MSE}((R_x^2+R_y^2)/G^2)$", "Brinkman residual"),
        (None, r"$|\Delta p_{pred}-\Delta p_{bc}|/(|\Delta p_{bc}|+\epsilon)$", "Pressure drop"),
        ("pressure_boundary_mse", r"$\mathrm{MSE}(p_\Gamma-p_{bc})/\Delta p^2$", "Pressure BC"),
    )
    figure = plt.figure(figsize=(15.5, 8.5))
    grid = figure.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.30), wspace=0.25, hspace=0.30)
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    )
    legend_axis = figure.add_subplot(grid[:, 2])
    legend_axis.axis("off")
    for axis, (metric, xlabel, title) in zip(axes, specifications, strict=True):
        for label, frame in datasets.items():
            values = pressure_drop[label] if metric is None else _values(frame, metric, max_cases)
            _cdf(axis, values, label=label, color=colors[label])
        axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("CDF")
        axis.set_title(title)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    handles = [Line2D([0], [0], color=colors[label], linewidth=2, label=count_headings[label] or label) for label in datasets]
    legend_axis.legend(handles, [str(handle.get_label()) for handle in handles], loc="upper left", frameon=True)
    figure.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.98)
    layout.title_over_axes(figure, figure_title, axes, y=0.96)
    return figure


def plot_spatial_residuals(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int | None = None,
) -> Figure:
    """Retain a direct momentum-map and flux-profile family outside approved panel scope."""
    limit = min(len(frame) for frame in datasets.values()) if max_cases is None else max_cases
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, limit).require_residuals() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Spatial residuals",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    figure, axes = plt.subplots(len(datasets), 2, figsize=(12.0, 4.0 * len(datasets)), squeeze=False, width_ratios=(2.0, 1.0))
    for row, (label, frame) in enumerate(datasets.items()):
        summary = summaries[id(frame)]
        heading = count_headings[label] or label
        momentum = cast("np.ndarray", summary.momentum_mean)
        upper = max(float(np.quantile(momentum, 0.99)), np.finfo(float).eps)
        image = axes[row, 0].imshow(momentum, origin="lower", extent=summary.grid.extent, aspect="equal", cmap="magma", vmin=0.0, vmax=upper)
        axes[row, 0].set_title(f"{heading} — momentum")
        axes[row, 0].set_xlabel(f"x [{summary.grid.coordinate_units[0]}]")
        axes[row, 0].set_ylabel(f"y [{summary.grid.coordinate_units[1]}]")
        layout.add_map_colorbar(figure, image, axes[row, 0], label=r"mean $\sqrt{R_x^2+R_y^2}/G$ [1]")
        rows = summary.grid.coordinates[1, :, 0]
        axes[row, 1].plot(np.maximum(cast("np.ndarray", summary.flux_balance_mean), _POSITIVE_FLOOR), rows, linewidth=2)
        axes[row, 1].set_xscale("log")
        axes[row, 1].set_title(f"{heading} — flux balance")
        axes[row, 1].set_xlabel(r"mean $|q(y)-\bar{q}|/Q$ [1]")
        axes[row, 1].set_ylabel(f"y [{summary.grid.coordinate_units[1]}]")
        axes[row, 1].grid(True, which="both", linestyle="--", alpha=0.3)
    figure.suptitle(figure_title)
    figure.tight_layout()
    return figure


def build_pressure_boundary_summary(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build current pressure-boundary components and pressure-drop error."""
    dataframe.validate_comparison(datasets, require_physics=True)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame).require_pressure() for frame in datasets.values()}
    rows = []
    for label, frame in datasets.items():
        summary = summaries[id(frame)]
        rows.append(
            {
                "Model": label,
                "pressure inlet MSE / Δp² [1]": float(np.median(frame["pressure_inlet_mse"])),
                "squared outlet mean / Δp² [1]": float(np.median(frame["pressure_outlet_mean_square"])),
                "pressure boundary MSE / Δp² [1]": float(np.median(frame["pressure_boundary_mse"])),
                "pressure drop absolute error [Pa]": float(np.median(cast("np.ndarray", summary.pressure_absolute_error))),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def plot_pressure_boundary_summary(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """Retain the direct pressure summary diagnostic outside panel scope."""
    table = build_pressure_boundary_summary(datasets)
    figure_title, count_headings = layout.aggregate_title_context(
        "Pressure boundary and drop summary",
        layout.effective_case_counts(datasets),
    )
    if any(count_headings.values()):
        table = table.rename(index={label: count_headings[label] for label in datasets})
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    table.plot.bar(ax=axis)
    axis.set_yscale("log")
    figure.suptitle(figure_title)
    return figure
