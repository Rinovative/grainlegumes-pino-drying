"""Render predictive-error views from current session reducers.

Responsibilities:
  - Compare secondary relative L2 and H1 metrics without redefining the objective
  - Render global/local distributions and reference-versus-prediction field means
  - Plot p, u, v, and TaskSpec velocity-magnitude spatial error reductions as field rows by dataset columns on the shared fixed
    map grid, every map on a round discrete scale locked over its row and every mean error map under one mean-error line
  - Relate field error to target magnitude and left/right boundary distance

Design principles:
  - Line, bar and distribution figures draw datasets as panels and models as series, in the one fixed style of each model
  - Learned fields, groups, order, and units come from admitted TaskSpec evidence
  - Complete and bounded-prefix reductions are reused through EvaluationSession
  - Local relative error normalizes each case field by its own reference RMS; cases below the field's floor are left
    out and their count is shown
  - Percentile clipping changes display scales only, never numerical summaries

This module does NOT:
  - Parse NPZ payloads or reconstruct the authoritative primary aggregate
  - Combine fields with incompatible physical units
  - Own notebook controls, dataset identity, or public panel composition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as map_layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure

_DEFAULT_CASE_LIMIT = 100
_DISTRIBUTION_DEFAULT_CASE_LIMIT = 50
_TARGET_MAGNITUDE_DEFAULT_CASE_LIMIT = 50
_MINIMUM_KDE_SAMPLES = 2
_BAND_LABELS = ("0-5 %", "5-10 %", "10-20 %", "20-40 %", ">40 %")
_UNAVAILABLE_CASE_NOTE = "Unavailable: a field reference below its floor, or a non-finite prediction"


def _finite(frame: pd.DataFrame, column: str, *, limit: int | None = None) -> np.ndarray:
    """Return the non-negative metric values of the finite cases in saved order; non-finite cases are left out, so it may be empty."""
    selected = frame if limit is None else frame.iloc[:limit]
    values = validity_contract.finite_case_values(pd.to_numeric(selected[column], errors="raise"))
    if np.any(values < 0.0):
        msg = f"{column} must not contain a negative value."
        raise ValueError(msg)
    return values


def _case_support_text(label: str, available: int, total: int) -> str:
    """Return how many cases one per-case distribution shows and how many it leaves out as unavailable."""
    return f"{label}: {available} of {total} cases, {total - available} unavailable"


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


def _relative_unavailable_cases(summary: sessions.FullEvaluationSummary, field: presentation.DisplayField) -> int:
    """Return the number of cases below the reference floor of one learned field or magnitude."""
    if field.is_magnitude:
        return summary.magnitudes[_magnitude_id(field)].local_relative_unavailable_cases
    counts = cast("np.ndarray", summary.local_relative_unavailable_cases)
    return int(counts[summary.grid.fields.index(field.component_fields[0])])


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


def plot_global_error_metrics(*, datasets: dataset_cuts.DisplayFrames) -> Figure:
    """Compare current secondary relative L2/H1: per dataset and norm one row of box plot, density and CDF, the models as series."""
    dataframe.validate_comparison(datasets)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    title, count_headings = layout.dataset_title_context("Global error metrics", groups, {key: len(frame) for key, frame in datasets.items()})
    metrics = (("Relative L2", "physical_relative_l2"), ("Relative H1", "physical_relative_h1"))
    unit = f"[{field_labels.display_unit('1')}]"
    figure, axes = figure_layout.panel_figure(len(groups) * len(metrics), 3)
    headers: list[str] = []
    for block, (dataset, models) in enumerate(groups.items()):
        heading = count_headings[dataset] or dataset
        for offset, (name, column) in enumerate(metrics):
            box_axis, density_axis, cdf_axis = axes[block * len(metrics) + offset]
            headers.append(f"{heading} · {name}")
            populations = {model: _finite(datasets[key], column) for model, key in models.items()}
            boxes = box_axis.boxplot(list(populations.values()), patch_artist=True, showfliers=True, medianprops={"color": "black", "linewidth": 2})
            for patch, key in zip(boxes["boxes"], models.values(), strict=True):
                patch.set_facecolor(styles[key].color)
                patch.set_hatch(styles[key].hatch)
                patch.set_alpha(0.65)
            box_axis.set_xticks([])
            box_axis.set_ylabel(f"{name} {unit}")
            for key, values in zip(models.values(), populations.values(), strict=True):
                if values.size == 0:
                    continue
                style = styles[key]
                if values.size >= _MINIMUM_KDE_SAMPLES and not np.allclose(values, values[0]):
                    coordinates = np.linspace(float(np.min(values)), float(np.max(values)), 400)
                    density_axis.plot(coordinates, gaussian_kde(values)(coordinates), color=style.color, linestyle=style.linestyle)
                else:
                    density_axis.axvline(float(values[0]), color=style.color, linestyle=style.linestyle)
                layout.plot_ecdf(cdf_axis, values, color=style.color, linestyle=style.linestyle)
            for axis, text in ((box_axis, "Box plot"), (density_axis, "Kernel density"), (cdf_axis, "CDF")):
                figure_layout.set_panel_title(axis, text)
                axis.grid(True, which="both", linestyle="--", alpha=0.3)
            density_axis.set_xlabel(f"{name} {unit}")
            density_axis.set_ylabel("Density")
            cdf_axis.set_xlabel(f"{name} {unit}")
            cdf_axis.set_ylabel("Fraction of cases")
            # Every model names, per norm, the cases it shows and the unavailable ones; one without any is drawn nowhere.
            support = [
                _case_support_text(model if len(models) > 1 else name, values.size, len(datasets[key]))
                for (model, key), values in zip(models.items(), populations.values(), strict=True)
            ]
            cdf_axis.text(0.98, 0.02, "\n".join(support), transform=cdf_axis.transAxes, ha="right", va="bottom", fontsize=figure_layout.TICK_PT)
            if all(np.all(values > 0.0) for values in populations.values()):
                figure_layout.set_log_scale(box_axis, "y")
                figure_layout.set_log_scale(density_axis, "x")
                figure_layout.set_log_scale(cdf_axis, "x")
    figure_layout.set_row_headers(figure, axes, headers)
    layout.add_series_legend(figure, axes, datasets, groups, styles)
    figure_layout.set_figure_title(figure, axes, title, note=layout.series_note(styles, _UNAVAILABLE_CASE_NOTE))
    return figure


def plot_predictive_error_distributions(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DISTRIBUTION_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot transparent casewise and local predictive-error distributions: a column per dataset, a row for the casewise norm and one per field."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {key: active_session.prefix_summary(frame, max_cases) for key, frame in datasets.items()}
    title, count_headings = layout.dataset_title_context(
        "Predictive error distributions (lower is better)", groups, layout.effective_case_counts(datasets, max_cases=max_cases)
    )
    figure, axes = figure_layout.panel_figure(1 + len(fields), len(groups), panel_in=(4.2, 2.6), sharey="row")
    zero_relative_l2: set[int] = set()  # columns with an exact prediction, whose zero a log axis could not draw
    for column, (dataset, models) in enumerate(groups.items()):
        support: list[str] = []
        below_floor: list[list[str]] = [[] for _field in fields]
        for model, key in models.items():
            style, summary = styles[key], summaries[key]
            if not summary.used_count:
                layout.mark_empty_prefix(*axes[:, column], case_count=summary.case_count, model=model)
                continue
            count = min(max_cases, len(datasets[key]))
            relative_l2 = _finite(datasets[key], "physical_relative_l2", limit=count)
            if np.any(relative_l2 <= 0.0):
                zero_relative_l2.add(column)
            if relative_l2.size:
                layout.plot_ecdf(axes[0, column], relative_l2, color=style.color, linestyle=style.linestyle)
            support.append(_case_support_text(model if len(models) > 1 else "Relative L2", relative_l2.size, count))
            for row, field in enumerate(fields):
                local = _prefix_field(summary, field, "local_relative_error")
                axes[row + 1, column].plot(local.quantiles, local.probabilities, color=style.color, linestyle=style.linestyle, linewidth=2.0)
                if local.unavailable_case_count:
                    prefix = f"{model}: " if len(models) > 1 else ""
                    noun = "case" if local.unavailable_case_count == 1 else "cases"
                    below_floor[row].append(f"{prefix}{local.unavailable_case_count} {noun} below the reference floor, left out")
        if support:
            axes[0, column].text(0.02, 0.02, "\n".join(support), transform=axes[0, column].transAxes, fontsize=figure_layout.TICK_PT, va="bottom")
        for row, counts in enumerate(below_floor):
            if counts:
                axes[row + 1, column].text(
                    0.02,
                    0.02,
                    "\n".join(counts),
                    transform=axes[row + 1, column].transAxes,
                    fontsize=figure_layout.TICK_PT,
                    va="bottom",
                )
        figure_layout.set_panel_title(axes[0, column], count_headings[dataset] or dataset)
    for column, axis in enumerate(axes[0]):  # a log axis labels a range inside one decade too, but draws no exact prediction
        if column not in zero_relative_l2:
            figure_layout.set_log_scale(axis, "x")
    # A pointwise error, and a relative L2 of an exact prediction, can be 0 and is never negative: symlog from 0.
    for axis in (*(axes[0, column] for column in sorted(zero_relative_l2)), *axes[1:].flat):
        axis.set_xscale("symlog", linthresh=1e-6)
        axis.set_xlim(left=0.0)
    for axis in axes.flat:
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
    for axis in axes[0]:  # the x axes are not shared, so every column carries its label
        axis.set_xlabel(f"Casewise group-weighted relative L2 [{field_labels.display_unit('1')}]")
    for axis in axes[1:].flat:
        axis.set_xlabel(f"Pointwise |error| / case reference-field RMS [{field_labels.display_unit('1')}]")
    figure_layout.label_rows_once(axes[:1], "Fraction of cases")
    figure_layout.label_rows_once(axes[1:], "Fraction of case-gridpoint values")
    figure_layout.set_row_headers(figure, axes, ["Casewise relative L2", *(f"Local error · {field.matplotlib_label}" for field in fields)])
    layout.add_series_legend(figure, axes, datasets, groups, styles)
    figure_layout.set_figure_title(figure, axes, title, note=layout.series_note(styles, _UNAVAILABLE_CASE_NOTE))
    return figure


def plot_mean_field_bias(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DISTRIBUTION_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Render a row per field and a column per dataset of reference versus prediction case means, the models as series with their RMSE and R2."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {key: active_session.full_summary(frame, max_cases) for key, frame in datasets.items()}
    title, count_headings = layout.dataset_title_context(
        "Reference vs prediction mean", groups, {key: summary.sample_count for key, summary in summaries.items()}
    )
    figure, axes = figure_layout.panel_figure(len(fields), len(groups))
    for column, (dataset, models) in enumerate(groups.items()):
        for row, field in enumerate(fields):
            axis, means = axes[row, column], {}
            for model, key in models.items():
                reference, prediction = _case_means(summaries[key], field)
                means[model] = (
                    field_labels.display_values(reference, field.unit, field=field.key),
                    field_labels.display_values(prediction, field.unit, field=field.key),
                )
            low = min(float(min(np.min(reference), np.min(prediction))) for reference, prediction in means.values())
            high = max(float(max(np.max(reference), np.max(prediction))) for reference, prediction in means.values())
            if np.isclose(low, high):
                high += np.finfo(float).eps
            axis.plot((low, high), (low, high), "k--", linewidth=1, alpha=0.7)
            for line, (model, key) in enumerate(models.items()):
                reference, prediction = means[model]
                style = styles[key]
                rmse = float(np.sqrt(np.mean((prediction - reference) ** 2)))
                denominator = float(np.sum((reference - np.mean(reference)) ** 2))
                r2 = 1.0 - float(np.sum((prediction - reference) ** 2)) / denominator if denominator > 0.0 else float("nan")
                axis.scatter(reference, prediction, s=18, alpha=0.45, color=style.color, marker=style.marker or "o")
                axis.text(
                    0.03,
                    0.97 - 0.08 * line,
                    f"{model + ': ' if len(models) > 1 else ''}{field.metric_label('RMSE')} = {rmse:.3g}, R² = {r2:.3g}",
                    transform=axis.transAxes,
                    color=style.color,
                    fontsize=figure_layout.TICK_PT,
                    va="top",
                )
            axis.set_xlabel(f"Reference mean [{field_labels.display_unit(field.unit, field=field.key)}]")
            axis.grid(alpha=0.3)
        figure_layout.set_panel_title(axes[0, column], count_headings[dataset] or dataset)
    for row, field in enumerate(fields):
        figure_layout.label_rows_once(axes[row : row + 1], f"Prediction mean [{field_labels.display_unit(field.unit, field=field.key)}]")
    figure_layout.set_row_headers(figure, axes, [field.matplotlib_label for field in fields])
    layout.add_series_legend(figure, axes, datasets, groups, styles, name_roles=False)  # the panel title names the dataset and so its role
    figure_layout.set_figure_title(figure, axes, title, note=layout.series_note(styles))
    return figure


def _normalized_scale(datasets: Mapping[str, pd.DataFrame], field: presentation.DisplayField) -> error_diagnostics.ReferenceScale | str:
    """Return the one train-only reference scale that every dataset shares for ``field``, else the reason there is none."""
    scales = [
        error_diagnostics.steady_reference_scale(dataframe.require_complete_provenance(frame), field=field.key, unit=field.unit)
        for frame in datasets.values()
    ]
    reason = next((item for item in scales if isinstance(item, str)), None)
    if reason is not None:
        return reason
    try:
        return error_diagnostics.require_common_reference_scale(cast("list[error_diagnostics.ReferenceScale]", scales))
    except ValueError as error:
        return str(error)


def _map_figure(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
    attribute: str,
    metric_label: str,
    title: str,
    relative_percent: bool = False,
    signed: bool = False,
    normalized: bool = False,
    mean_line: bool = True,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """
    Render one spatial statistic family: a row per display field, a column per dataset, one colour bar per map.

    The columns follow the selection, so the figure grows with them (no width cap). Each row has one round scale locked over its
    maps, with arrows only at the ends of the bars that have values beyond: a ``signed`` statistic is symmetric to the larger absolute
    5th or 95th percentile, blue-white-red, any other runs from 0 to the 95th percentile. A ``normalized`` statistic is divided by the one
    train-only reference scale the datasets share; a ``relative_percent`` one names the cases below the field's reference floor. With
    ``mean_line`` (the error maps, not the standard deviation) one line above each map gives its mean |e|, or "n/a" without a map.
    """
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {id(frame): active_session.full_summary(frame, max_cases).require_spatial() for frame in datasets.values()}
    # The columns of a comparison are the models of one dataset, which the title then names.
    dataset_labels = {datasets.labels[key][0] for key in datasets} if isinstance(datasets, dataset_cuts.DisplayFrames) else set()
    if len(dataset_labels) == 1 and (dataset_label := next(iter(dataset_labels))) not in datasets:
        title = f"{title} · {dataset_label}"
    figure_title, count_headings = layout.aggregate_title_context(
        title,
        {label: summaries[id(frame)].sample_count for label, frame in datasets.items()},
    )
    extents = {summary.grid.extent for summary in summaries.values()}
    if len(extents) != 1:
        msg = "The maps of one figure share one physical extent, but the selected datasets have different grids."
        raise ValueError(msg)
    extent = next(iter(extents))
    geometry = map_layout.plan_geometry(len(fields), len(datasets), (extent[1] - extent[0], extent[3] - extent[2]), width_cap=None)
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    semantic = (
        field_maps.FieldMapSemantic.SIGNED_ERROR
        if signed
        else field_maps.FieldMapSemantic.RELATIVE_ERROR
        if relative_percent
        else field_maps.FieldMapSemantic.ABSOLUTE_ERROR
    )
    colorbar_unit = "%" if relative_percent else "1" if normalized else None
    top_notes: list[str] = []
    for row, field in enumerate(fields):
        scale = _normalized_scale(datasets, field) if normalized else None
        reason = scale if isinstance(scale, str) else None
        divisor = scale.value if isinstance(scale, error_diagnostics.ReferenceScale) else 1.0
        full = {
            label: _full_array(summaries[id(frame)], field, attribute) * (100.0 if relative_percent else 1.0) / divisor
            for label, frame in datasets.items()
        }
        error_unit = field_labels.display_unit(colorbar_unit or field.unit, quantity_kind="difference")
        unit = f"[{error_unit}]"
        bar_label = f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} {unit}" if normalized else unit
        # A relative mean map is all NaN when every case lies below the field's reference floor.
        available = np.concatenate([values[np.isfinite(values)] for values in full.values()])
        boundaries = field_maps.error_boundaries(available, signed=signed) if reason is None and available.size else None
        definition = (
            field_maps.build_field_map(
                np.append(available, boundaries[[0, -1]]),  # the scale's own ends keep a uniform (all-zero) statistic from being a constant map
                semantic=semantic,
                field=field.key,
                unit=colorbar_unit or field.unit,
                display_transform=metric_label,
                comparison_scope=scale.dataset_identity if isinstance(scale, error_diagnostics.ReferenceScale) else "selected Evaluation models",
                locked_boundaries=boundaries,
                bound_policy="rounded 5th to 95th percentile, symmetric" if signed else "0 to the rounded 95th percentile",
            )
            if boundaries is not None
            else None
        )
        for column, (label, frame) in enumerate(datasets.items()):
            summary = summaries[id(frame)]
            values = full[label]
            finite = np.isfinite(values)
            below_floor = _relative_unavailable_cases(summary, field) if relative_percent else 0
            note = f"{below_floor} of {summary.sample_count} cases below the reference floor, left out" if below_floor else ""
            if mean_line:  # computed from the displayed map over all its finite cells, "n/a" for an unavailable map
                exponent = 0 if definition is None else map_layout.error_format(definition.boundaries, signed=signed).exponent
                shown = values if definition is not None else None
                mean = layout.mean_error_text(shown, unit=None if normalized else error_unit, exponent=exponent)
                note = f"{mean}; {note}" if note else mean
            if row == 0:
                top_notes.append(note)
            elif note:
                map_layout.add_map_note(axes[row, column], note)
            if definition is None or not finite.any():
                lacking = "every case lies below the reference floor" if relative_percent else "no finite values"
                map_layout.empty_cell(axes[row, column], f"Unavailable: {reason or lacking}")
                continue
            indices = display_grid.display_indices(values.shape, display_spatial_stride)
            x_grid, y_grid = summary.grid.coordinates
            image = axes[row, column].pcolormesh(
                display_grid.display_sample(x_grid, indices),
                display_grid.display_sample(y_grid, indices),
                np.ma.masked_invalid(display_grid.display_sample(values, indices)),
                shading="auto",
                **definition.pcolormesh_kwargs(),
            )
            extend = field_maps.locked_extend(values[finite], definition.boundaries)
            map_layout.add_field_map_colorbar(figure, image, slots[row, column], definition, label=bar_label, extend=extend)
    map_layout.set_column_headers(axes, [count_headings[label] or label for label in datasets], top_notes)
    map_layout.set_row_labels(axes, [field.matplotlib_label for field in fields])
    map_layout.set_map_title(figure, figure_title)
    layout.add_scale_note(figure, shared=False)
    return figure


def plot_mean_error_maps(
    *,
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int = _DEFAULT_CASE_LIMIT,
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    display_spatial_stride: display_grid.DisplayStride = "auto",
) -> Figure:
    """Plot one selected mean-error representation in the fixed layout; the signed one is the mean of prediction minus reference."""
    if error_mode == error_diagnostics.SIGNED_ERROR:
        return _map_figure(
            datasets=datasets,
            max_cases=max_cases,
            attribute="signed_error_mean",
            metric_label="mean signed error",
            title="Mean signed error maps",
            signed=True,
            display_spatial_stride=display_spatial_stride,
        )
    if error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        return _map_figure(
            datasets=datasets,
            max_cases=max_cases,
            attribute="absolute_error_mean",
            metric_label="train_state_std_normalized_absolute_error",
            title="Mean normalized absolute error maps",
            normalized=True,
            display_spatial_stride=display_spatial_stride,
        )
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
            metric_label="relative error (of reference RMS)",
            title="Mean relative error maps",
            relative_percent=True,
            display_spatial_stride=display_spatial_stride,
        )
    msg = f"error_mode must be one of {error_diagnostics.STEADY_ERROR_MODES}."
    raise ValueError(msg)


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
        mean_line=False,
        display_spatial_stride=display_spatial_stride,
    )


def plot_error_vs_target_magnitude(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _TARGET_MAGNITUDE_DEFAULT_CASE_LIMIT,
) -> Figure:
    """Render a row per field and a column per dataset of median error against reference magnitude, the models as series."""
    dataframe.validate_comparison(datasets)
    fields = _display_fields(datasets)
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {key: active_session.prefix_summary(frame, max_cases) for key, frame in datasets.items()}
    figure_title, count_headings = layout.dataset_title_context(
        "Error vs reference magnitude", groups, {key: summary.case_count for key, summary in summaries.items()}
    )
    figure, axes = figure_layout.panel_figure(len(fields), len(groups))
    for row, field in enumerate(fields):
        unit = field_labels.display_unit(field.unit, field=field.key, quantity_kind="difference")
        trends = {key: _prefix_field(summary, field, "target_magnitude_error") for key, summary in summaries.items() if summary.used_count}
        log_scale = field.label != "p" and all(np.all(trend.centers > 0.0) and np.all(trend.medians > 0.0) for trend in trends.values())
        for column, (dataset, models) in enumerate(groups.items()):
            axis = axes[row, column]
            for model, key in models.items():
                if key not in trends:
                    layout.mark_empty_prefix(axis, case_count=summaries[key].case_count, model=model)
                    continue
                axis.plot(
                    trends[key].centers,
                    trends[key].medians,
                    color=styles[key].color,
                    linestyle=styles[key].linestyle,
                    marker=styles[key].marker or "o",
                    markersize=4,
                    alpha=0.9,
                )
            if log_scale:
                figure_layout.set_log_scale(axis, "x")
                figure_layout.set_log_scale(axis, "y")
            axis.grid(True, which="both", linestyle="--", alpha=0.3)
            if row == 0:
                figure_layout.set_panel_title(axis, count_headings[dataset] or dataset)
            magnitude_label = field.matplotlib_label if field.is_magnitude else f"|{field.matplotlib_label}|"
            axis.set_xlabel(f"{magnitude_label} reference (bin centre) [{field_labels.display_unit(field.unit, field=field.key)}]")
        figure_layout.label_rows_once(axes[row : row + 1], f"Median {field.metric_label('MAE')} [{unit}]")
    layout.add_series_legend(figure, axes, datasets, groups, styles)
    figure_layout.set_figure_title(figure, axes, figure_title, note=layout.series_note(styles))
    return figure


def plot_boundary_error_decomposition(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
    channels: Sequence[str] = ("p", "u", "v", "U"),
) -> Figure:
    """Render channel-checkbox left/right boundary ratios, a panel per dataset with the models as bar clusters shaded by distance band."""
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
    groups, styles = series.panel_groups(datasets.labels), series.series_styles(datasets)
    with sessions.scoped_session(datasets) as active_session:
        summaries = {key: active_session.prefix_summary(frame, max_cases) for key, frame in datasets.items()}
    figure_title, count_headings = layout.dataset_title_context(
        "Left/right boundary error ratio vs x-distance (interior reference: >40 %)",
        groups,
        {key: summary.case_count for key, summary in summaries.items()},
    )
    figure, axes = figure_layout.panel_figure(1, len(groups), sharey="row")
    positions = np.arange(len(active))
    bar_count = max(len(models) for models in groups.values()) * len(_BAND_LABELS)
    width = 0.8 / bar_count
    shades = np.linspace(0.3, 1.0, len(_BAND_LABELS))  # nearer the boundary is lighter

    def blend(color: str, shade: float) -> tuple[float, float, float]:
        """Return ``color`` mixed toward white by ``shade`` as an opaque colour, so edge and hatch keep full strength."""
        red, green, blue = (1.0 - shade * (1.0 - channel) for channel in to_rgb(color))
        return red, green, blue

    for axis, (dataset, models) in zip(axes[0], groups.items(), strict=True):
        for slot, (model, key) in enumerate(models.items()):
            style, summary = styles[key], summaries[key]
            if not summary.used_count:
                layout.mark_empty_prefix(axis, case_count=summary.case_count, model=model)
                continue
            means = np.stack([np.asarray(_prefix_field(summary, by_label[channel], "boundary_region_error").means) for channel in active])
            ratios = np.divide(means, means[:, -1, None], out=np.full_like(means, np.nan), where=means[:, -1, None] > 0.0)
            for band_index, shade in enumerate(shades):
                offset = slot * len(_BAND_LABELS) + band_index - (bar_count - 1) / 2
                axis.bar(
                    positions + offset * width,
                    ratios[:, band_index],
                    width,
                    color=blend(style.color, float(shade)),
                    edgecolor="white" if shade == 1.0 else style.color,  # the top band is the model colour itself: a white hatch contrasts
                    hatch=style.hatch,
                )
        axis.axhline(1.0, color="black", linestyle="--", alpha=0.4)
        axis.set_xticks(positions, [by_label[channel].matplotlib_label for channel in active])
        axis.set_xlabel("Channel")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)
        figure_layout.set_panel_title(axis, count_headings[dataset] or dataset)
    figure_layout.label_rows_once(axes, f"Boundary error ratio [{field_labels.display_unit('1')}]\n(MAE / interior MAE)")
    bands = {label: Patch(facecolor=blend("0.0", float(shade))) for label, shade in zip(_BAND_LABELS, shades, strict=True)}
    layout.add_series_legend(figure, axes, datasets, groups, styles, bars=True, extra={"x-distance from left/right boundary": bands})
    figure_layout.set_figure_title(figure, axes, figure_title, note=layout.series_note(styles))
    return figure
