"""Render predictive-error views from current session reducers.

Responsibilities:
  - Compare secondary relative L2 and H1 metrics without redefining the objective
  - Render global/local distributions and GT-versus-prediction field means
  - Plot p, u, v, and TaskSpec velocity-magnitude spatial error reductions
  - Relate field error to target magnitude and left/right boundary distance

Design principles:
  - Learned fields, groups, order, and units come from admitted TaskSpec evidence
  - Complete and bounded-prefix reductions are reused through EvaluationSession
  - Local relative error normalizes each case field by its own reference RMS
  - Percentile clipping changes display scales only, never numerical summaries

This module does NOT:
  - Parse NPZ payloads or reconstruct the authoritative primary aggregate
  - Combine fields with incompatible physical units
  - Own notebook controls, dataset identity, or public panel composition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_maps as field_maps

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 100
_DISTRIBUTION_DEFAULT_CASE_LIMIT = 50
_TARGET_MAGNITUDE_DEFAULT_CASE_LIMIT = 50
_MINIMUM_KDE_SAMPLES = 2
_BAND_LABELS = ("0-5 %", "5-10 %", "10-20 %", "20-40 %", ">40 %")


def _finite(frame: pd.DataFrame, column: str, *, limit: int | None = None) -> np.ndarray:
    """Return the non-negative metric values of the finite cases in saved order; non-finite cases are left out, so it may be empty."""
    selected = frame if limit is None else frame.iloc[:limit]
    values = validity_contract.finite_case_values(pd.to_numeric(selected[column], errors="raise"))
    if np.any(values < 0.0):
        msg = f"{column} must not contain a negative value."
        raise ValueError(msg)
    return values


def _display_fields(datasets: Mapping[str, pd.DataFrame]) -> tuple[presentation.DisplayField, ...]:
    """Return one shared current display-field order."""
    return presentation.shared_display_fields(tuple(datasets.values()))


def _magnitude_id(field: presentation.DisplayField) -> str:
    """Return the canonical TaskSpec group id for a magnitude display field."""
    if not field.key.endswith("_magnitude"):
        msg = f"Display field {field.key!r} is not a group magnitude."
        raise ValueError(msg)
    return field.key.removesuffix("_magnitude")


def _full_array(summary: sessions.FullEvaluationSummary, field: presentation.DisplayField, attribute: str) -> np.ndarray:
    """Return one full-summary learned or group-magnitude array."""
    if field.is_magnitude:
        return np.asarray(getattr(summary.magnitudes[_magnitude_id(field)], attribute))
    values = cast("np.ndarray", getattr(summary, attribute))
    return np.asarray(values[summary.grid.fields.index(field.component_fields[0])])


def _case_means(
    summary: sessions.FullEvaluationSummary,
    field: presentation.DisplayField,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned reference/prediction spatial means for one display field."""
    if field.is_magnitude:
        magnitude = summary.magnitudes[_magnitude_id(field)]
        return magnitude.case_reference_means, magnitude.case_prediction_means
    index = summary.grid.fields.index(field.component_fields[0])
    return summary.case_reference_means[:, index], summary.case_prediction_means[:, index]


def _prefix_field(
    summary: sessions.PrefixEvaluationSummary,
    field: presentation.DisplayField,
    attribute: str,
) -> Any:
    """Return one learned or magnitude bounded reduction."""
    if field.is_magnitude:
        return getattr(summary.magnitudes[_magnitude_id(field)], attribute)
    mapping = getattr(summary, attribute)
    return mapping[field.component_fields[0]]


def plot_global_error_metrics(*, datasets: Mapping[str, pd.DataFrame]) -> Figure:
    """Compare current secondary relative L2/H1 in the fixed 3 x 2 layout."""
    dataframe.validate_comparison(datasets)
    names = tuple(datasets)
    title, count_headings = layout.aggregate_title_context(
        "Global error metrics",
        layout.effective_case_counts(datasets),
    )
    display_names = tuple(count_headings[name] or name for name in names)
    if not names:
        msg = "At least one artifact frame is required."
        raise ValueError(msg)
    metrics = (("Relative L2", "rel_l2"), ("Relative H1", "rel_h1"))
    populations = {column: [_finite(datasets[name], column) for name in names] for _label, column in metrics}
    # A dataset without one finite case stays in the comparison, is drawn nowhere and says so in the legend.
    display_names = tuple(
        f"{display_name}: unavailable (0 finite of {len(datasets[name])} cases)"
        if all(populations[column][index].size == 0 for _label, column in metrics)
        else display_name
        for index, (name, display_name) in enumerate(zip(names, display_names, strict=True))
    )
    palette = [plt.get_cmap("tab10")(index % 10) for index in range(len(names))]
    figure = plt.figure(figsize=(21, 10))
    grid = figure.add_gridspec(3, 3, width_ratios=(1.0, 1.0, 0.35), hspace=0.35, wspace=0.25)
    plot_axes: list[Axes] = []
    for column_index, (display_name, column) in enumerate(metrics):
        values_by_dataset = populations[column]
        box_axis = figure.add_subplot(grid[0, column_index])
        density_axis = figure.add_subplot(grid[1, column_index])
        cdf_axis = figure.add_subplot(grid[2, column_index])
        plot_axes.extend((box_axis, density_axis, cdf_axis))
        boxes = box_axis.boxplot(
            values_by_dataset,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, color in zip(boxes["boxes"], palette, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        box_axis.set_xticks([])
        box_axis.set_title(f"{display_name} - Boxplot")
        box_axis.set_ylabel(f"{display_name} (dimensionless)")
        for values, name, color in zip(values_by_dataset, names, palette, strict=True):
            if values.size == 0:
                continue
            if values.size >= _MINIMUM_KDE_SAMPLES and not np.allclose(values, values[0]):
                coordinates = np.linspace(float(np.min(values)), float(np.max(values)), 400)
                density_axis.plot(coordinates, gaussian_kde(values)(coordinates), color=color, label=name)
            else:
                density_axis.axvline(float(values[0]), color=color, label=name)
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
            cdf_axis.plot(ordered, cumulative, color=color, label=name)
        density_axis.set_title(f"{display_name} - KDE Density")
        density_axis.set_xlabel(f"{display_name} (dimensionless)")
        density_axis.set_ylabel("Density")
        cdf_axis.set_title(f"{display_name} - CDF")
        cdf_axis.set_xlabel(f"{display_name} (dimensionless)")
        cdf_axis.set_ylabel("CDF")
        if all(np.all(values > 0.0) for values in values_by_dataset):
            box_axis.set_yscale("log")
            density_axis.set_xscale("log")
            cdf_axis.set_xscale("log")
        for axis in (box_axis, density_axis, cdf_axis):
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
    legend_axis = figure.add_subplot(grid[:, 2])
    legend_axis.axis("off")
    legend_axis.legend(
        [Line2D([0], [0], color=color, linewidth=8) for color in palette],
        display_names,
        loc="upper left",
    )
    figure.subplots_adjust(top=0.90)
    layout.title_over_axes(figure, title, plot_axes, y=0.97)
    return figure


def _plot_ecdf(axis: Any, values: np.ndarray, *, color: object, label: str | None = None) -> None:
    """Plot one finite empirical cumulative distribution without smoothing."""
    ordered = np.sort(np.asarray(values, dtype=float))
    cumulative = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    axis.step(ordered, cumulative, where="post", color=color, linewidth=2.0, label=label)


def plot_predictive_error_distributions(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DISTRIBUTION_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot transparent casewise and local predictive-error distributions."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.prefix_summary(frame, max_cases) for frame in datasets.values()}
    title, count_headings = layout.aggregate_title_context(
        "Predictive error distributions (lower is better)",
        layout.effective_case_counts(datasets, max_cases=max_cases),
    )
    colors = {field.key: plt.get_cmap("tab10")(index % 10) for index, field in enumerate(fields)}
    figure, axes = plt.subplots(
        2,
        len(datasets),
        figsize=(6.4 * len(datasets), 7.4),
        squeeze=False,
        sharey="row",
    )
    for column, (dataset_label, frame) in enumerate(datasets.items()):
        count = min(max_cases, len(frame))
        summary = summaries[id(frame)]
        global_axis = axes[0, column]
        local_axis = axes[1, column]
        if not summary.used_count:
            layout.mark_empty_prefix(global_axis, local_axis, case_count=summary.case_count)
        else:
            _plot_ecdf(global_axis, _finite(frame, "rel_l2", limit=count), color="black")
        for field in fields if summary.used_count else ():
            local = _prefix_field(summary, field, "local_relative_error")
            local_axis.plot(
                local.quantiles,
                local.probabilities,
                color=colors[field.key],
                linewidth=2.0,
                label=field.matplotlib_label,
            )
        global_axis.set_title(count_headings[dataset_label] or dataset_label)
        global_axis.set_xlabel("Casewise channel-balanced relative L2 [-]")
        local_axis.set_xlabel("Pointwise |error| / case reference-field RMS [-]")
        for axis in (global_axis, local_axis):
            axis.set_xscale("symlog", linthresh=1e-6)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
        if column == 0:
            global_axis.set_ylabel("Fraction of cases")
            local_axis.set_ylabel("Fraction of case-gridpoint values")
    handles = [Line2D([0], [0], color=colors[field.key], linewidth=2.6) for field in fields]
    figure.legend(
        handles,
        [field.matplotlib_label for field in fields],
        title="Local-error field",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(fields),
        frameon=True,
    )
    figure.subplots_adjust(top=0.80, bottom=0.10, left=0.08, right=0.98, hspace=0.38, wspace=0.22)
    layout.title_over_axes(figure, title, tuple(axes.flat), y=0.985)
    return figure


def plot_mean_field_bias(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DISTRIBUTION_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Render four-row ground-truth versus prediction case-mean panels."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases) for frame in datasets.values()}
    title, count_headings = layout.aggregate_title_context(
        "GT vs prediction mean",
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    figure, axes = plt.subplots(
        len(fields),
        len(datasets),
        figsize=(6 * len(datasets), 9),
        squeeze=False,
    )
    for column, (label, frame) in enumerate(datasets.items()):
        summary = summaries[id(frame)]
        for row, field in enumerate(fields):
            reference, prediction = _case_means(summary, field)
            low = float(min(np.min(reference), np.min(prediction)))
            high = float(max(np.max(reference), np.max(prediction)))
            if np.isclose(low, high):
                high += np.finfo(float).eps
            rmse = float(np.sqrt(np.mean((prediction - reference) ** 2)))
            denominator = float(np.sum((reference - np.mean(reference)) ** 2))
            r2 = 1.0 - float(np.sum((prediction - reference) ** 2)) / denominator if denominator > 0.0 else float("nan")
            axis = axes[row, column]
            axis.plot((low, high), (low, high), "k--", linewidth=1, alpha=0.7)
            axis.scatter(reference, prediction, s=18, alpha=0.45)
            heading = count_headings[label] or label
            prefix = f"{heading}\n" if row == 0 else ""
            axis.set_title(f"{prefix}{field.matplotlib_label}: RMSE={rmse:.3g}, R2={r2:.3g}", fontsize=11)
            if column == 0:
                axis.set_ylabel("Prediction mean")
            if row == len(fields) - 1:
                axis.set_xlabel("GT mean")
            axis.grid(alpha=0.3)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.98, hspace=0.35, wspace=0.25)
    return figure


def _map_figure(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
    attribute: str,
    metric_label: str,
    title: str,
    relative_percent: bool = False,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Render one four-row spatial statistic family."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_spatial() for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        title,
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    figure, axes = layout.map_subplots(rows=len(fields), columns=len(datasets))
    for row, field in enumerate(fields):
        full = {
            label: _full_array(summaries[id(frame)], field, attribute) * (100.0 if relative_percent else 1.0) for label, frame in datasets.items()
        }
        definition = field_maps.build_field_map(
            np.concatenate([values.ravel() for values in full.values()]),
            semantic=field_maps.FieldMapSemantic.RELATIVE_ERROR if relative_percent else field_maps.FieldMapSemantic.ABSOLUTE_ERROR,
            field=field.key,
            unit="%" if relative_percent else field.unit,
            display_transform=metric_label,
            comparison_scope="selected Evaluation models",
            bound_policy="full selected model/case grids",
        )
        for column, (label, frame) in enumerate(datasets.items()):
            summary = summaries[id(frame)]
            values = full[label]
            indices = display_grid.display_indices(values.shape, display_spatial_stride)
            x_grid, y_grid = summary.grid.coordinates
            image = axes[row, column].contourf(
                display_grid.display_sample(x_grid, indices),
                display_grid.display_sample(y_grid, indices),
                display_grid.display_sample(values, indices),
                **definition.contourf_kwargs(),
            )
            heading = count_headings[label] or label
            prefix = f"{heading}\n" if row == 0 else ""
            axes[row, column].set_title(f"{prefix}{field.matplotlib_label} {metric_label}", fontsize=11)
            axes[row, column].set_aspect("equal")
            layout.add_map_colorbar(figure, image, axes[row, column], label="%" if relative_percent else field.unit)
    first_summary = summaries[id(next(iter(datasets.values())))]
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_summary.grid.coordinate_units[0]}]",
        y_label=f"y [{first_summary.grid.coordinate_units[1]}]",
    )
    figure.suptitle(figure_title)
    return figure


def plot_mean_error_maps(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Plot one selected mean-error representation in the fixed layout."""
    if error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        return _plot_normalized_mean_error_maps(datasets=datasets, max_cases=max_cases, display_spatial_stride=display_spatial_stride)
    if error_mode == "MAE":
        return _map_figure(
            datasets=datasets,
            max_cases=max_cases,
            attribute="absolute_error_mean",
            metric_label="MAE",
            title="Mean absolute error maps",
            display_spatial_stride=display_spatial_stride,
        )
    if error_mode == "Relative [%]":
        return _map_figure(
            datasets=datasets,
            max_cases=max_cases,
            attribute="local_relative_error_mean",
            metric_label="relative error [% of reference RMS]",
            title="Mean relative error maps",
            relative_percent=True,
            display_spatial_stride=display_spatial_stride,
        )
    msg = f"error_mode must be one of {error_diagnostics.STEADY_ERROR_MODES}."
    raise ValueError(msg)


def _plot_normalized_mean_error_maps(
    *, datasets: Mapping[str, pd.DataFrame], max_cases: int, display_spatial_stride: display_grid.DisplayStride
) -> Figure:
    """Use complete-grid mean absolute errors and one common train-only scale."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_spatial() for frame in datasets.values()}
    figure, axes = layout.map_subplots(rows=len(fields), columns=len(datasets))
    for row, field in enumerate(fields):
        scales = [
            error_diagnostics.steady_reference_scale(dataframe.require_complete_provenance(frame), field=field.key, unit=field.unit)
            for frame in datasets.values()
        ]
        reason = next((item for item in scales if isinstance(item, str)), None)
        if reason is None:
            try:
                common_scale = error_diagnostics.require_common_reference_scale(cast("list[error_diagnostics.ReferenceScale]", scales))
            except ValueError as error:
                reason = str(error)
        if reason is None:
            all_values = [_full_array(summaries[id(other)], field, "absolute_error_mean") / common_scale.value for other in datasets.values()]
            definition = field_maps.build_field_map(
                np.concatenate([item.ravel() for item in all_values] + [np.array((0.0, 1.0))]),
                semantic=field_maps.FieldMapSemantic.ABSOLUTE_ERROR,
                field=field.key,
                unit="1",
                display_transform="train_state_std_normalized_absolute_error",
                comparison_scope=common_scale.dataset_identity,
                locked_boundaries=np.linspace(0.0, 1.0, 12),
                bound_policy="fixed 0..1 with extension",
            )
        for column, (label, frame) in enumerate(datasets.items()):
            axis = axes[row, column]
            axis.set_title(f"{field.matplotlib_label} normalized absolute error [1]" + (f"\n{label}" if row == 0 else ""))
            if reason is not None:
                axis.text(0.5, 0.5, f"Unavailable: {reason}", ha="center", va="center", wrap=True, transform=axis.transAxes)
                continue
            summary = summaries[id(frame)]
            values = _full_array(summary, field, "absolute_error_mean") / common_scale.value
            indices = display_grid.display_indices(values.shape, display_spatial_stride)
            x_grid, y_grid = summary.grid.coordinates
            image = axis.contourf(
                display_grid.display_sample(x_grid, indices),
                display_grid.display_sample(y_grid, indices),
                display_grid.display_sample(values, indices),
                **definition.contourf_kwargs(),
            )
            axis.set_aspect("equal")
            layout.add_map_colorbar(figure, image, axis, label="1 (train state std; >1 extended)")
    first_summary = summaries[id(next(iter(datasets.values())))]
    layout.apply_map_grid_axis_labels(
        axes,
        x_label=f"x [{first_summary.grid.coordinate_units[0]}]",
        y_label=f"y [{first_summary.grid.coordinate_units[1]}]",
    )
    figure.suptitle("Mean normalized absolute error maps")
    return figure


def plot_std_error_maps(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Plot current signed-error standard deviation in the fixed layout."""
    return _map_figure(
        datasets=datasets,
        max_cases=max_cases,
        attribute="signed_error_std",
        metric_label="STD error",
        title="Error standard-deviation maps",
        display_spatial_stride=display_spatial_stride,
    )


def plot_error_vs_target_magnitude(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _TARGET_MAGNITUDE_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Render four-channel target-magnitude trends with a dedicated legend."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.prefix_summary(frame, max_cases) for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Error vs GT magnitude",
        {label: summaries[id(frame)].case_count for label, frame in datasets.items()},
    )
    figure = plt.figure(figsize=(9.5, 9))
    grid = figure.add_gridspec(len(fields), 2, width_ratios=(1.0, 0.35), hspace=0.35, wspace=0.25)
    axes = [figure.add_subplot(grid[row, 0]) for row in range(len(fields))]
    legend_axis = figure.add_subplot(grid[:, 1])
    legend_axis.axis("off")
    handles: list[Line2D] = []
    for row, field in enumerate(fields):
        axis = axes[row]
        for label, frame in datasets.items():
            summary = summaries[id(frame)]
            display_label = count_headings[label] or label
            if not summary.used_count:
                if row == 0:
                    handles.append(Line2D([], [], linestyle="none", label=f"{display_label}: {layout.empty_prefix_text(summary.case_count)}"))
                continue
            trend = _prefix_field(summary, field, "target_magnitude_error")
            (line,) = axis.plot(trend.centers, trend.medians, marker="o", markersize=4, label=display_label, alpha=0.9)
            if row == 0:
                handles.append(line)
        if field.label != "p":
            all_positive = all(
                np.all(_prefix_field(summaries[id(frame)], field, "target_magnitude_error").centers > 0.0)
                and np.all(_prefix_field(summaries[id(frame)], field, "target_magnitude_error").medians > 0.0)
                for frame in datasets.values()
                if summaries[id(frame)].used_count
            )
            if all_positive:
                axis.set_xscale("log")
                axis.set_yscale("log")
        axis.set_title(f"{field.matplotlib_label}: MAE vs |GT|")
        axis.set_ylabel(f"Median MAE [{field.unit}]")
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("|GT| (bin center)")
    legend_axis.legend(handles, [str(handle.get_label()) for handle in handles], loc="upper left")
    figure.subplots_adjust(top=0.92, bottom=0.07, left=0.10, right=0.97)
    layout.title_over_axes(figure, figure_title, axes, y=0.98)
    return figure


def plot_boundary_error_decomposition(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    channels: Sequence[str] = ("p", "u", "v", "U"),
) -> Figure:
    """Render channel-checkbox left/right boundary ratios and an external legend."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    by_label = {field.label: field for field in fields}
    magnitude = next((field for field in fields if field.key == "velocity_magnitude"), None)
    if magnitude is not None:
        by_label["U"] = magnitude
    active = tuple(channels)
    if not active or any(channel not in by_label for channel in active):
        msg = f"Select at least one channel from {tuple(by_label)}."  # noqa: S608
        raise ValueError(msg)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.prefix_summary(frame, max_cases) for frame in datasets.values()}
    figure_title, count_headings = layout.aggregate_title_context(
        "Left/right boundary error ratio vs x-distance (interior reference: >40 %)",
        {label: summaries[id(frame)].case_count for label, frame in datasets.items()},
    )
    figure = plt.figure(figsize=(6 * len(datasets) + 2.5, 5))
    grid = figure.add_gridspec(1, len(datasets) + 1, width_ratios=(*([1.0] * len(datasets)), 0.35), wspace=0.25)
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(datasets))]
    legend_axis = figure.add_subplot(grid[0, -1])
    legend_axis.axis("off")
    bar_handles: list[Any] = []
    positions = np.arange(len(active))
    width = 0.8 / len(_BAND_LABELS)
    for axis, (label, frame) in zip(axes, datasets.items(), strict=True):
        summary = summaries[id(frame)]
        if not summary.used_count:
            layout.mark_empty_prefix(axis, case_count=summary.case_count)
            axis.set_title(count_headings[label] or label)
            continue
        means = np.stack([np.asarray(_prefix_field(summary, by_label[channel], "boundary_region_error").means) for channel in active])
        ratios = np.divide(
            means,
            means[:, -1, None],
            out=np.full_like(means, np.nan),
            where=means[:, -1, None] > 0.0,
        )
        for band_index, band_label in enumerate(_BAND_LABELS):
            bars = axis.bar(
                positions + (band_index - (len(_BAND_LABELS) - 1) / 2) * width,
                ratios[:, band_index],
                width,
                label=band_label,
            )
            if len(bar_handles) < len(_BAND_LABELS):
                bar_handles.append(bars[0])
        axis.axhline(1.0, color="black", linestyle="--", alpha=0.4)
        axis.set_xticks(positions, [by_label[channel].matplotlib_label for channel in active])
        axis.set_ylabel("Boundary error ratio (MAE / interior MAE)")
        axis.set_xlabel("Channel")
        axis.set_title(count_headings[label] or label)
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
    if bar_handles:
        legend_axis.legend(bar_handles, _BAND_LABELS, title="x-distance from left/right boundary", loc="upper left")
    figure.subplots_adjust(top=0.87, bottom=0.07, left=0.001, right=0.98, wspace=0.25)
    layout.title_over_axes(figure, figure_title, axes, y=0.97)
    return figure
