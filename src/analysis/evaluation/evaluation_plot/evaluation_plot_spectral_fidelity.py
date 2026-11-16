"""Render output-spectrum views from current bounded reducers.

Responsibilities:
  - Render session-reduced Hann-windowed radial spectra from physical coordinate spacing
  - Compare reference, prediction, and error spectra for p, u, v, and |u|, a panel per dataset and a row per model
  - Plot prediction-to-reference transfer ratios with casewise uncertainty
  - Draw every channel, its spectra normalised per case
  - Mask ratios where reference energy is too small for stable interpretation

Design principles:
  - Spectral evidence is architecture-independent and session-owned
  - Frequencies use inverse coordinate units; spectra are normalised per case, so channels of different power units share one
    dimensionless axis and no figure mixes units
  - TaskSpec vector magnitude is derived explicitly from declared components
  - Channels remain separate
  - The colour is the channel in every panel and curve kind; the curve kind (reference, prediction, error) is the line style

This module does NOT:
  - Parse artifact cases or invent physical coordinate spacing
  - Inspect learned layers, latent activations, or model internals
  - Provide learned-layer or latent spectral hooks
  - Own notebook controls or public panel composition
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.lines import Line2D

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_presentation as presentation
from src.analysis.evaluation import evaluation_session as sessions
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.ui import analysis_ui_figure_layout as figure_layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.analysis.evaluation import evaluation_datasets as dataset_cuts

_DEFAULT_CASE_LIMIT = 50
_REFERENCE_ENERGY_FLOOR = 1e-12
_SPECTRAL_DIMENSIONS = 2
_DIMENSIONLESS = field_labels.display_unit("1")
_FREQUENCY_LABEL = "Spatial frequency k [{unit}]"
_TRANSFER_HEIGHTS = (3.0, 1.0)  # a model's transfer panel and the support panel under it
_SUPPORT_NOTE = "Support requires k > 0 and reference power above 10⁻¹² x its case maximum.\nZero predictions use that numerical floor."


def _quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q10, median, and q90 while preserving wholly masked bins."""
    valid_columns = np.isfinite(values).any(axis=0)
    q10 = np.full(values.shape[1], np.nan)
    median = np.full(values.shape[1], np.nan)
    q90 = np.full(values.shape[1], np.nan)
    if valid_columns.any():
        selected = values[:, valid_columns]
        q10[valid_columns] = np.nanquantile(selected, 0.1, axis=0)
        median[valid_columns] = np.nanquantile(selected, 0.5, axis=0)
        q90[valid_columns] = np.nanquantile(selected, 0.9, axis=0)
    return q10, median, q90


def _normalize(values: np.ndarray) -> np.ndarray:
    """Normalize every case spectrum to unit sum, preserving zero spectra."""
    denominator = np.sum(values, axis=1, keepdims=True)
    return np.divide(values, denominator, out=np.array(values, copy=True), where=denominator > 0.0)


def _log_spectral_transfer(
    reference: np.ndarray,
    prediction: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return supported log10 prediction/GT transfer and per-bin support."""
    reference_values = np.asarray(reference, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    frequency_values = np.asarray(frequencies, dtype=float)
    if reference_values.ndim != _SPECTRAL_DIMENSIONS or prediction_values.shape != reference_values.shape:
        msg = "Reference and prediction spectra must have equal (case, frequency) shape."
        raise ValueError(msg)
    if frequency_values.shape != (reference_values.shape[1],):
        msg = "Frequencies must align with the spectral-bin dimension."
        raise ValueError(msg)
    if (
        not np.isfinite(reference_values).all()
        or not np.isfinite(prediction_values).all()
        or not np.isfinite(frequency_values).all()
        or np.any(reference_values < 0.0)
        or np.any(prediction_values < 0.0)
    ):
        msg = "Spectra and frequencies must be finite and non-negative."
        raise ValueError(msg)
    case_floor = _REFERENCE_ENERGY_FLOOR * np.max(reference_values, axis=1, keepdims=True)
    support = (frequency_values[None, :] > 0.0) & (reference_values > case_floor)
    transfer = np.full(reference_values.shape, np.nan, dtype=float)
    stabilized_prediction = np.maximum(prediction_values, case_floor)
    np.log10(
        np.divide(
            stabilized_prediction,
            reference_values,
            out=np.ones_like(reference_values),
            where=support,
        ),
        out=transfer,
        where=support,
    )
    support_count = np.sum(support, axis=0, dtype=int)
    support_fraction = support_count.astype(float) / max(reference_values.shape[0], 1)
    return transfer, support_fraction, support_count


def _plot_band(
    axis: Axes,
    frequencies: np.ndarray,
    values: np.ndarray,
    *,
    color: object,
    linestyle: str,
    linewidth: float = 2.2,
) -> None:
    """Plot current casewise median and q10-q90 band in the defined styling."""
    q10, median, q90 = _quantiles(values)
    valid = (frequencies > 0.0) & np.isfinite(median) & (median > 0.0)
    if not valid.any():
        return
    axis.plot(frequencies[valid], median[valid], color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95)
    band = valid & np.isfinite(q10) & np.isfinite(q90) & (q10 > 0.0)
    axis.fill_between(frequencies[band], q10[band], q90[band], color=color, alpha=0.12)


def _summaries(
    datasets: Mapping[str, pd.DataFrame],
    max_cases: int,
) -> dict[str, sessions.PrefixEvaluationSummary]:
    """Return one validated bounded-prefix summary per frame key."""
    dataframe.validate_comparison(datasets)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        msg = "max_cases must be a positive integer."
        raise ValueError(msg)
    with sessions.scoped_session(datasets) as active_session:
        return {key: active_session.prefix_summary(frame, max_cases) for key, frame in datasets.items()}


def _spectral(
    summary: sessions.PrefixEvaluationSummary,
    field: presentation.DisplayField,
) -> sessions.SpectralFieldSummary:
    """Return learned-field or TaskSpec magnitude spectral evidence."""
    if field.is_magnitude:
        return summary.magnitudes[field.key.removesuffix("_magnitude")].spectrum
    return summary.spectra[field.component_fields[0]]


def _models(groups: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Return the models of all dataset panels in order of first appearance; each one is a row of panels."""
    return list(dict.fromkeys(model for by_model in groups.values() for model in by_model))


def _title_columns(axes: np.ndarray, groups: Mapping[str, Mapping[str, str]], models: Sequence[str], count_headings: Mapping[str, str]) -> None:
    """Title every dataset column on its first visible panel, which is not the top one where the first model lacks the dataset."""
    for column, (dataset, by_model) in enumerate(groups.items()):
        first = min(models.index(model) for model in by_model)
        figure_layout.set_panel_title(axes[2 * first, column], count_headings[dataset] or dataset)


def _channel_legend(
    figure: Figure,
    axes: np.ndarray,
    fields: Sequence[presentation.DisplayField],
    colors: Mapping[str, tuple[float, float, float, float]],
    curves: Mapping[str, Line2D],
) -> None:
    """Write the legend of ``curves`` (black entries, the group "Curves") and of every channel in its colour (the group "Channels")."""
    channels = {field.matplotlib_label: Line2D([0], [0], color=colors[field.key], linewidth=2.6) for field in fields}
    figure_layout.add_legend(
        figure, axes, [*curves.values(), *channels.values()], [*curves, *channels], groups={"Curves": list(curves), "Channels": list(channels)}
    )


def plot_spectral_demand_prediction_error(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Render every channel's normalised demand/prediction and error spectra: a column per dataset, a spectrum and an error row per model.

    Every case spectrum is normalised to unit sum, so channels of different units (Pa², (m/s)²) share one dimensionless axis.
    """
    summaries = _summaries(datasets, max_cases)
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    colors = layout.channel_colors(fields)
    groups = series.panel_groups(datasets.labels)
    models = _models(groups)
    figure_title, count_headings = layout.dataset_title_context(
        "Spectral demand vs prediction and error", groups, {key: summary.case_count for key, summary in summaries.items()}
    )
    figure, axes = figure_layout.panel_figure(2 * len(models), len(groups))
    for column, by_model in enumerate(groups.values()):
        for row, model in enumerate(models):
            top_axis, bottom_axis = axes[2 * row, column], axes[2 * row + 1, column]
            if model not in by_model:
                top_axis.set_visible(False)
                bottom_axis.set_visible(False)
                continue
            summary = summaries[by_model[model]]
            if not summary.used_count:
                layout.mark_empty_prefix(top_axis, bottom_axis, case_count=summary.case_count)
            for field in fields if summary.used_count else ():
                spectrum = _spectral(summary, field)
                color = colors[field.key]
                _plot_band(top_axis, spectrum.frequencies, _normalize(spectrum.reference), color=color, linestyle="--", linewidth=2.0)
                _plot_band(top_axis, spectrum.frequencies, _normalize(spectrum.prediction), color=color, linestyle="-")
                _plot_band(bottom_axis, spectrum.frequencies, _normalize(spectrum.error), color=color, linestyle="-")
            for axis in (top_axis, bottom_axis):
                figure_layout.set_log_scale(axis, "x")
                figure_layout.set_log_scale(axis, "y")
                axis.grid(True, which="both", linestyle="--", alpha=0.3)
            bottom_axis.set_xlabel(_FREQUENCY_LABEL.format(unit=field_labels.display_unit(f"1/{summary.coordinate_unit}")))
    _title_columns(axes, groups, models, count_headings)
    figure_layout.label_rows_once(axes[0::2], f"Spectral power (normalised) [{_DIMENSIONLESS}]")
    figure_layout.label_rows_once(axes[1::2], f"Error spectral power (normalised) [{_DIMENSIONLESS}]")
    if len(models) > 1:
        figure_layout.set_row_headers(figure, axes[0::2], models)
    curves = {
        "Reference demand": Line2D([0], [0], color="black", linewidth=2.2, linestyle="--"),
        "Prediction": Line2D([0], [0], color="black", linewidth=2.2, linestyle="-"),
        "Error": Line2D([0], [0], color="black", linewidth=2.2, linestyle="-"),
    }
    _channel_legend(figure, axes, fields, colors, curves)
    figure_layout.set_figure_title(figure, axes, figure_title)
    return figure


def plot_spectral_transfer_ratio(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot every channel's supported log spectral transfer: a column per dataset, a transfer and a support row per model."""
    summaries = _summaries(datasets, max_cases)
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    colors = layout.channel_colors(fields)
    groups = series.panel_groups(datasets.labels)
    models = _models(groups)
    figure_title, count_headings = layout.dataset_title_context(
        "Log spectral transfer with q10-q90 uncertainty and support", groups, {key: summary.case_count for key, summary in summaries.items()}
    )
    figure, axes = figure_layout.panel_figure(2 * len(models), len(groups), panel_in=(4.2, 1.0), height_ratios=_TRANSFER_HEIGHTS * len(models))
    for column, by_model in enumerate(groups.values()):
        for row, model in enumerate(models):
            transfer_axis, support_axis = axes[2 * row, column], axes[2 * row + 1, column]
            if model not in by_model:
                transfer_axis.set_visible(False)
                support_axis.set_visible(False)
                continue
            support_axis.sharex(transfer_axis)
            summary = summaries[by_model[model]]
            if not summary.used_count:
                layout.mark_empty_prefix(transfer_axis, support_axis, case_count=summary.case_count)
            for field in fields if summary.used_count else ():
                spectrum = _spectral(summary, field)
                transfer, support_fraction, _support_count = _log_spectral_transfer(
                    spectrum.reference,
                    spectrum.prediction,
                    spectrum.frequencies,
                )
                q10, median, q90 = _quantiles(transfer)
                valid = (spectrum.frequencies > 0.0) & np.isfinite(median)
                color = colors[field.key]
                transfer_axis.plot(spectrum.frequencies[valid], median[valid], color=color, linewidth=2.2)
                band = valid & np.isfinite(q10) & np.isfinite(q90)
                transfer_axis.fill_between(spectrum.frequencies[band], q10[band], q90[band], color=color, alpha=0.14)
                support_axis.plot(spectrum.frequencies[valid], support_fraction[valid], color=color, linewidth=1.8)
            transfer_axis.axhline(0.0, linewidth=1.6, linestyle="--", color="black", alpha=0.6)
            figure_layout.set_log_scale(transfer_axis, "x")
            figure_layout.set_log_scale(support_axis, "x")
            for axis in (transfer_axis, support_axis):
                axis.grid(True, which="both", linestyle="--", alpha=0.3)
            support_axis.set_ylim(0.0, 1.05)
            support_axis.set_xlabel(_FREQUENCY_LABEL.format(unit=field_labels.display_unit(f"1/{summary.coordinate_unit}")))
            transfer_axis.tick_params(labelbottom=False)
    _title_columns(axes, groups, models, count_headings)
    figure_layout.label_rows_once(axes[0::2], rf"$\log_{{10}}(S_{{pred}}/S_{{ref}})$ [{_DIMENSIONLESS}]")
    figure_layout.label_rows_once(axes[1::2], f"Supported fraction [{_DIMENSIONLESS}]")
    if len(models) > 1:
        figure_layout.set_row_headers(figure, axes[0::2], models)
    _channel_legend(figure, axes, fields, colors, {"equal spectral power": Line2D([0], [0], color="black", linewidth=1.6, linestyle="--")})
    figure_layout.set_figure_title(figure, axes, figure_title, note=_SUPPORT_NOTE)
    return figure
