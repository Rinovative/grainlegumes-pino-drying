"""Render output-spectrum views from current bounded reducers.

Responsibilities:
  - Render session-reduced Hann-windowed radial spectra from physical coordinate spacing
  - Compare reference, prediction, and error spectra for p, u, v, and |u|, a panel per dataset and a row per model
  - Plot prediction-to-reference transfer ratios with casewise uncertainty
  - Draw every channel, its spectra normalized per case
  - Mask ratios where reference energy is too small for stable interpretation

Design principles:
  - Spectral evidence is architecture-independent and session-owned
  - Frequencies use inverse coordinate units; spectra are normalized per case, so channels of different power units share one
    dimensionless axis and no figure mixes units
  - TaskSpec vector magnitude is derived explicitly from declared components
  - Channels remain separate: each later channel is drawn thinner and with its own sparse marker, in the plot and in its key glyph,
    on points of its own wherever a trace has a point per channel, so one that coincides with another stays visible; a curve's
    q10-q90 band is its two edges, thinner lines in its style and colour with the channel's marker open at its median's bins, and
    no fill, so overlapping bands never mix into a colour the key lacks; the figure note names them
  - The colour is the channel in every panel and curve kind; the curve kind (reference, prediction, error) is the line style, and
    every legend swatch has the style of its curves

This module does NOT:
  - Parse artifact cases or invent physical coordinate spacing
  - Inspect learned layers, latent activations, or model internals
  - Provide learned-layer or latent spectral hooks
  - Own notebook controls or public panel composition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    from src.analysis.presentation import analysis_visual_semantics as visual_semantics

    _Color = tuple[float, float, float, float]

_DEFAULT_CASE_LIMIT = 50
_REFERENCE_ENERGY_FLOOR = 1e-12
_SPECTRAL_DIMENSIONS = 2
_DIMENSIONLESS = field_labels.display_unit("1")
_FREQUENCY_LABEL = "Spatial frequency k [{unit}]"
_TRANSFER_HEIGHTS = (3.0, 1.0)  # a model's transfer panel and the support panel under it
_SUPPORT_NOTE = (
    "Support requires k > 0 and reference power above 10⁻¹² \N{MULTIPLICATION SIGN} its case maximum.\nZero predictions use that numerical floor."
)
_CURVE_STYLES: Mapping[str, visual_semantics.LineStyle] = {"Reference demand": "--", "Prediction": "-", "Error": "-."}  # the colour is the channel
_BAND_EDGE_ALPHA = 0.7  # a q10-q90 band edge stays lighter than the medians
_BAND_EDGE_WIDTH = 0.6  # and thinner than the thinnest channel
_BAND_NOTE = "Thin lines with open markers bound each curve's q10\N{EN DASH}q90 band over the cases, in its line style and colour"
_WIDEST_CHANNEL = 2.8  # the first channel's line width; each later channel is thinner, so one drawn over another leaves it visible
_CHANNEL_WIDTH_STEP = 0.6
_THINNEST_CHANNEL = 1.0
_CHANNEL_MARKERS = ("o", "s", "^", "D")  # sparse markers tell channels apart where their medians coincide, in the plot and in the key
_MARKERS_PER_TRACE = 6
_MARKER_PT = 4.0


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


def _channel_marks(fields: Sequence[presentation.DisplayField]) -> dict[str, tuple[str, int, int]]:
    """Return each channel's marker by field key with its position and the channel count, which set where it marks (``_marker_keywords``)."""
    return {field.key: (_CHANNEL_MARKERS[index % len(_CHANNEL_MARKERS)], index, len(fields)) for index, field in enumerate(fields)}


def _marker_keywords(marks: tuple[str, int, int], count: int) -> dict[str, Any]:
    """
    Return the keywords of a channel trace of ``count`` points: its marker about ``_MARKERS_PER_TRACE`` times, from an offset of its own.

    The step spans at least one point per channel wherever the trace has that many, so the channels' offsets differ and coincident
    traces never mark one point twice.
    """
    marker, index, channels = marks
    step = max(1, count // _MARKERS_PER_TRACE, min(channels, count))
    return {"marker": marker, "markersize": _MARKER_PT, "markevery": (index * step // channels, step)}


def _band_edges(
    axis: Axes, frequencies: np.ndarray, edges: tuple[np.ndarray, np.ndarray], *, color: _Color, linestyle: str, marks: tuple[str, int, int]
) -> None:
    """
    Draw one curve's q10-q90 band as its two ``edges`` over its median's ``frequencies``, NaN where the band has no value.

    Each edge is a thin line in the curve's style and its channel's colour with the channel's marker open at the bins the median
    marks, so coinciding bands stay told apart by channel; no fill, whose overlapping colours the key would lack (``_BAND_NOTE``).
    A finite edge value without a finite neighbour, which no line segment draws, takes the open marker as well.
    """
    keywords = _marker_keywords(marks, frequencies.size)
    for edge in edges:
        finite = np.isfinite(edge)
        isolated = finite & ~(np.r_[False, finite[:-1]] | np.r_[finite[1:], False])
        style = {
            "color": color,
            "alpha": _BAND_EDGE_ALPHA,
            "markerfacecolor": "none",
            "marker": keywords["marker"],
            "markersize": keywords["markersize"],
        }
        axis.plot(frequencies, edge, linestyle=linestyle, linewidth=_BAND_EDGE_WIDTH, **style, markevery=keywords["markevery"])
        if isolated.any():
            axis.plot(frequencies[isolated], edge[isolated], linestyle="none", **style)


def _plot_band(
    axis: Axes,
    frequencies: np.ndarray,
    values: np.ndarray,
    *,
    color: _Color,
    linestyle: str,
    linewidth: float,
    marks: tuple[str, int, int],
) -> None:
    """Plot current casewise median and q10-q90 band in the defined styling."""
    q10, median, q90 = _quantiles(values)
    valid = (frequencies > 0.0) & np.isfinite(median) & (median > 0.0)
    if not valid.any():
        return
    axis.plot(
        frequencies[valid],
        median[valid],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.95,
        **_marker_keywords(marks, int(valid.sum())),
    )
    low, high = np.where(q10 > 0.0, (q10, q90), np.nan)[:, valid]  # a log axis has no band edge at or below zero
    _band_edges(axis, frequencies[valid], (low, high), color=color, linestyle=linestyle, marks=marks)


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


def _channel_widths(fields: Sequence[presentation.DisplayField]) -> dict[str, float]:
    """Return each channel's line width by field key, in display order: each later channel thinner, so one drawn over another stays visible."""
    return {field.key: max(_THINNEST_CHANNEL, _WIDEST_CHANNEL - _CHANNEL_WIDTH_STEP * index) for index, field in enumerate(fields)}


def _channel_legend(
    figure: Figure,
    axes: np.ndarray,
    fields: Sequence[presentation.DisplayField],
    colors: Mapping[str, tuple[float, float, float, float]],
    curves: Mapping[str, Line2D],
) -> None:
    """Write the legend of ``curves`` (black entries, the group "Curves") and of every channel in its colour and width (the group "Channels")."""
    widths, marks = _channel_widths(fields), _channel_marks(fields)
    channels = {
        field.matplotlib_label: Line2D(
            [0], [0], color=colors[field.key], linewidth=widths[field.key], marker=marks[field.key][0], markersize=_MARKER_PT
        )
        for field in fields
    }
    figure_layout.add_legend(
        figure,
        axes,
        [*curves.values(), *channels.values()],
        [*curves, *channels],
        groups={"Curves": list(curves.values()), "Channels": list(channels.values())},
    )


def plot_spectral_demand_prediction_error(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """
    Render every channel's normalized demand/prediction and error spectra: a column per dataset, a spectrum and an error row per model.

    Every case spectrum is normalized to unit sum, so channels of different units (Pa², (m/s)²) share one dimensionless axis.
    """
    summaries = _summaries(datasets, max_cases)
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    colors, widths, marks = layout.channel_colors(fields), _channel_widths(fields), _channel_marks(fields)
    groups = series.panel_groups(datasets.labels)
    models = _models(groups)
    figure_title, count_headings = layout.dataset_title_context(
        "Spectral demand vs prediction and error", datasets, {key: summary.case_count for key, summary in summaries.items()}
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
                for axis, values, curve in (
                    (top_axis, spectrum.reference, "Reference demand"),
                    (top_axis, spectrum.prediction, "Prediction"),
                    (bottom_axis, spectrum.error, "Error"),
                ):
                    style = _CURVE_STYLES[curve]
                    _plot_band(
                        axis,
                        spectrum.frequencies,
                        _normalize(values),
                        color=colors[field.key],
                        linestyle=style,
                        linewidth=widths[field.key],
                        marks=marks[field.key],
                    )
            for axis in (top_axis, bottom_axis):
                figure_layout.set_log_scale(axis, "x")
                figure_layout.set_log_scale(axis, "y")
                axis.grid(True, which="both", linestyle="--", alpha=0.3)
            bottom_axis.set_xlabel(_FREQUENCY_LABEL.format(unit=field_labels.display_unit(f"1/{summary.coordinate_unit}")))
    _title_columns(axes, groups, models, count_headings)
    figure_layout.label_rows_once(axes[0::2], f"Normalized spectral power [{_DIMENSIONLESS}]")
    figure_layout.label_rows_once(axes[1::2], f"Normalized error spectral power [{_DIMENSIONLESS}]")
    if len(models) > 1:
        figure_layout.set_row_headers(figure, axes[0::2], models)
    curves = {curve: Line2D([0], [0], color="black", linewidth=2.0, linestyle=style) for curve, style in _CURVE_STYLES.items()}
    _channel_legend(figure, axes, fields, colors, curves)
    figure_layout.set_figure_title(figure, axes, figure_title, note=_BAND_NOTE)
    return figure


def plot_spectral_transfer_ratio(
    *,
    datasets: dataset_cuts.DisplayFrames,
    max_cases: int = _DEFAULT_CASE_LIMIT,
) -> Figure:
    """Plot every channel's supported log spectral transfer: a column per dataset, a transfer and a support row per model."""
    summaries = _summaries(datasets, max_cases)
    fields = presentation.shared_display_fields(tuple(datasets.values()))
    colors, widths, marks = layout.channel_colors(fields), _channel_widths(fields), _channel_marks(fields)
    groups = series.panel_groups(datasets.labels)
    models = _models(groups)
    figure_title, count_headings = layout.dataset_title_context(
        "Log spectral transfer with q10-q90 uncertainty and support", datasets, {key: summary.case_count for key, summary in summaries.items()}
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
                color, width = colors[field.key], widths[field.key]
                marker = _marker_keywords(marks[field.key], int(valid.sum()))
                transfer_axis.plot(spectrum.frequencies[valid], median[valid], color=color, linewidth=width, **marker)
                _band_edges(transfer_axis, spectrum.frequencies[valid], (q10[valid], q90[valid]), color=color, linestyle="-", marks=marks[field.key])
                support_axis.plot(spectrum.frequencies[valid], support_fraction[valid], color=color, linewidth=width, **marker)
            transfer_axis.axhline(0.0, linewidth=1.6, linestyle="--", color="black", alpha=0.6)
            figure_layout.set_log_scale(transfer_axis, "x")
            figure_layout.set_log_scale(support_axis, "x")
            for axis in (transfer_axis, support_axis):
                axis.grid(True, which="both", linestyle="--", alpha=0.3)
            support_axis.set_ylim(0.0, 1.05)
            support_axis.set_xlabel(_FREQUENCY_LABEL.format(unit=field_labels.display_unit(f"1/{summary.coordinate_unit}")))
            transfer_axis.tick_params(which="both", labelbottom=False)  # the support panel under it labels k, minor ticks included
    _title_columns(axes, groups, models, count_headings)
    figure_layout.label_rows_once(axes[0::2], rf"$\log_{{10}}(S_{{pred}}/S_{{ref}})$ [{_DIMENSIONLESS}]")
    figure_layout.label_rows_once(axes[1::2], f"Support [{_DIMENSIONLESS}]")  # the share of cases; the note defines support
    if len(models) > 1:
        figure_layout.set_row_headers(figure, axes[0::2], models)
    _channel_legend(figure, axes, fields, colors, {"Equal spectral power": Line2D([0], [0], color="black", linewidth=1.6, linestyle="--")})
    figure_layout.set_figure_title(figure, axes, figure_title, note=f"{_SUPPORT_NOTE}\n{_BAND_NOTE}")
    return figure
