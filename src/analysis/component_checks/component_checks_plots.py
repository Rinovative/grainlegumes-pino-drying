"""Render component-check tables as static scientific figures.

Responsibilities:
  - Plot steady residual floors, crop sweeps, edge profiles, and term decompositions
  - Plot residual maps with the shared field-map scale contract
  - Plot derivative-backend differences and perturbation sensitivity ratios
  - Plot Fourier-mode truncation curves against the maintained mode counts
  - Plot transient balance closures, field-versus-COMSOL differences, and sorption residuals

Design principles:
  - Figures consume component-check tables and never recompute physics
  - Categorical colours follow one fixed colour-blind-checked order, never a cycle
  - Logarithmic magnitude axes; one quantity per axis
  - Line figures take their geometry, legend and titles from the shared figure layout; the legend of several panels stands once, outside right

This module does NOT:
  - Load data, evaluate residuals or balances, or save files
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from matplotlib.lines import Line2D

from src.analysis import presentation
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as plot_layout
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as map_layout

from . import component_checks_steady as steady

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure

_LOSS_TERMS: Final = ("momentum", "flux_balance", "boundary")
_TERM_LABELS: Final = {
    "momentum": "Momentum (loss)",
    "pointwise_divergence": "Pointwise div u",
    "porous_divergence": "Pointwise div(eps u)",
    "flux_balance": "Flux balance (loss)",
    "boundary": "Pressure boundary (loss)",
}
# Quadrature rule of the sorption residual tables, its line style and its key entry.
_QUADRATURE_RULES: Final[tuple[tuple[str, Literal["-", "--"], str], ...]] = (
    ("trapezoid", "-", "Trapezoidal rule"),
    ("left", "--", "Left-endpoint rule"),
)
_EDGE_LABELS: Final = {"inlet": "Inlet (y = 0)", "outlet": "Outlet (y = L_y)", "wall_x_min": "Wall x = 0", "wall_x_max": "Wall x = L_x"}
_REFERENCE_LINE: Final = {"color": "0.5", "linewidth": 1.0, "linestyle": ":"}
_SINGLE_PANEL_IN: Final = (6.5, 4.2)  # one-panel figures, larger than a cell of a panel row
_PANEL_ROW_IN: Final = (3.4, 3.0)  # cell of a row of four or five panels
# Okabe-Ito hues in an order whose adjacent pairs pass the colour-vision-deficiency check.
SERIES_COLORS: Final = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9")
MAP_BOUNDARIES: Final = tuple(float(value) for value in np.geomspace(1e-5, 1.0, 12))


def series_color(position: int) -> str:
    """Return the categorical colour of a series position, refusing to cycle."""
    if not 0 <= position < len(SERIES_COLORS):
        msg = f"At most {len(SERIES_COLORS)} categorical series are supported; facet or fold the rest."
        raise ValueError(msg)
    return SERIES_COLORS[position]


def _case_style(position: int, count: int, label: str) -> dict[str, object]:
    """Return per-case line style: categorical up to the palette size, otherwise neutral and unlabelled."""
    if count <= len(SERIES_COLORS):
        return {"color": series_color(position), "label": label}
    return {"color": "0.35", "alpha": 0.5, "label": None}


def _legend(
    figure: Figure, axes: np.ndarray, *, extra: Sequence[tuple[Line2D, str]] = (), groups: Mapping[str, Sequence[Artist]] | None = None
) -> None:
    """Write the one legend of every labelled artist of ``axes`` plus the ``extra`` key entries, ``groups`` by handle; no entries, no legend."""
    handles, labels = figure_layout.legend_entries(axes.flat)
    handles += [handle for handle, _label in extra]
    labels += [label for _handle, label in extra]
    if labels:
        figure_layout.add_legend(figure, axes, handles, labels, groups=groups)


def _case_legend(
    figure: Figure, axes: np.ndarray, count: int, *, extra: Sequence[tuple[Line2D, str]] = (), groups: Mapping[str, Sequence[Artist]] | None = None
) -> None:
    """Write the legend of the cases when they carry categorical colours, otherwise name their count in the first panel title."""
    if count > len(SERIES_COLORS):
        first = axes[0, 0]
        figure_layout.set_panel_title(first, f"{first.get_title()} ({count} cases)")
    _legend(figure, axes, extra=extra, groups=groups)


def backend_colors() -> dict[str, str]:
    """Return one fixed colour per declared derivative backend label."""
    return {backend.label: series_color(position) for position, backend in enumerate(steady.DERIVATIVE_BACKENDS)}


def _description(label: str) -> str:
    """Return the readable description of a backend label with a capital first letter, as a legend entry."""
    text = steady.backend_by_label(label).description
    return text[:1].upper() + text[1:]


def plot_crop_sweep(residuals: pd.DataFrame, *, dtype: str = "float32") -> Figure:
    """
    Plot reference momentum terms against the interior crop per backend.

    Parameters
    ----------
    residuals : pandas.DataFrame
        Output of :func:`component_checks_steady.summarize_reference_residuals`.
    dtype : str, optional
        Precision rows to show.

    Returns
    -------
    matplotlib.figure.Figure
        Median over cases with the case range as a band.

    """
    colors = backend_colors()
    selected = residuals[residuals["dtype"] == dtype]
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=_SINGLE_PANEL_IN)
    axis = axes[0, 0]
    for backend, group in selected.groupby("backend", sort=False):
        stats = group.groupby("crop")["momentum"].agg(["median", "min", "max"])
        color = colors[str(backend)]
        axis.plot(stats.index, stats["median"], "-", color=color, linewidth=2.0, marker="o", markersize=5, label=_description(str(backend)))
        axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.15, linewidth=0)
    figure_layout.set_log_scale(axis, "y")
    axis.set_xlabel("Interior crop [cells per edge]")
    axis.set_ylabel("Mean (Rx² + Ry²) / G²")
    figure_layout.set_panel_title(axis, f"Reference momentum residual vs crop ({dtype})")
    axis.grid(visible=True, which="major", color="0.9")
    _legend(figure, axes)
    return figure


def plot_reference_terms(residuals: pd.DataFrame, *, settings: Sequence[tuple[str, int]], dtype: str = "float32") -> Figure:
    """
    Plot every reference residual term for selected backend and crop settings.

    Returns
    -------
    matplotlib.figure.Figure
        One marker per case; terms on the y-axis, magnitude on a log x-axis.

    """
    terms = tuple(_TERM_LABELS)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=_SINGLE_PANEL_IN)
    axis = axes[0, 0]
    offsets = np.linspace(-0.25, 0.25, len(settings))
    markers = ("o", "s", "D", "^", "v")
    for position, (backend, crop) in enumerate(settings):
        rows = residuals[(residuals["dtype"] == dtype) & (residuals["backend"] == backend) & (residuals["crop"] == crop)]
        for index, term in enumerate(terms):
            values = rows[term].to_numpy()
            axis.scatter(
                values,
                np.full(values.shape, index + offsets[position]),
                color=series_color(position),
                marker=markers[position % len(markers)],
                s=36,
                label=f"{_description(backend)}, crop {crop}" if index == 0 else None,
            )
    axis.set_yticks(range(len(terms)), [_TERM_LABELS[term] for term in terms])
    axis.invert_yaxis()
    figure_layout.set_log_scale(axis, "x")
    axis.set_xlabel("Dimensionless mean square (per case)")
    figure_layout.set_panel_title(axis, f"Reference residual terms ({dtype})")
    axis.grid(visible=True, axis="x", which="major", color="0.9")
    _legend(figure, axes)
    return figure


def plot_edge_profiles(profiles: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot momentum-residual RMS against the distance from each grid edge.

    Parameters
    ----------
    profiles : Mapping[str, pandas.DataFrame]
        Backend label to :func:`component_checks_steady.edge_profiles` output.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per edge with a shared log y-axis.

    """
    colors = backend_colors()
    figure, axes = figure_layout.panel_figure(1, len(steady.EDGES), panel_in=_PANEL_ROW_IN, sharey="row")
    for axis, edge in zip(axes[0], steady.EDGES, strict=True):
        for backend, frame in profiles.items():
            rows = frame[frame["edge"] == edge]
            axis.plot(rows["distance"], rows["rms"], color=colors[backend], linewidth=2.0, marker="o", markersize=4, label=_description(backend))
        figure_layout.set_panel_title(axis, _EDGE_LABELS[edge])
        axis.set_xlabel("Distance from edge [cells]")
        figure_layout.set_log_scale(axis, "y")
        axis.grid(visible=True, which="major", color="0.9")
    figure_layout.label_rows_once(axes, "RMS |R| / G")
    _legend(figure, axes)
    return figure


def plot_residual_maps(
    fields: Sequence[steady.SteadyResidualFields], *, case_index: int, case_label: str, extent: tuple[float, float, float, float]
) -> Figure:
    """
    Map the momentum-residual magnitude of one case for several backends.

    Parameters
    ----------
    fields : Sequence[SteadyResidualFields]
        Backend evaluations of the same cases.
    case_index : int
        Case position in the evaluations.
    case_label : str
        Case label for the title.
    extent : tuple[float, float, float, float]
        Physical ``(x_min, x_max, y_min, y_max)`` in m.

    Returns
    -------
    matplotlib.figure.Figure
        One map per backend on the common map grid, each with its own bar on the shared logarithmic eleven-colour scale,
        an arrow only where its map has values beyond and then one note line saying so.

    """
    magnitudes = [steady.residual_magnitude(item)[case_index] for item in fields]
    positive = np.concatenate([value[value > 0].ravel() for value in magnitudes])
    scale = presentation.field_maps.build_field_map(positive, semantic="log_positive", field="|R|/G", unit="1", locked_boundaries=MAP_BOUNDARIES)
    geometry = map_layout.plan_geometry(1, len(fields), (extent[1] - extent[0], extent[3] - extent[2]), row_labels=False, width_cap=None)
    figure, axes, slots = map_layout.map_grid(geometry, extent=extent)
    label = f"|R|/G [{presentation.field_labels.display_unit('1')}]"
    arrows = False
    for column, magnitude in enumerate(magnitudes):
        image = axes[0, column].imshow(
            np.maximum(magnitude, MAP_BOUNDARIES[0]), origin="lower", extent=extent, aspect="equal", **scale.imshow_kwargs()
        )
        extend = presentation.field_maps.locked_extend(magnitude[np.isfinite(magnitude)], scale.boundaries)  # exact zeros lie below the floor too
        map_layout.add_field_map_colorbar(figure, image, slots[0, column], scale, label=label, extend=extend)
        arrows = arrows or extend != "neither"
    map_layout.set_column_headers(axes, [steady.backend_by_label(item.backend).description for item in fields], [])
    map_layout.set_map_title(figure, f"Momentum residual magnitude, {case_label}")
    if arrows:
        low, high = (map_layout.power_of_ten(round(math.log10(bound))) for bound in (MAP_BOUNDARIES[0], MAP_BOUNDARIES[-1]))
        plot_layout.add_scale_note(figure, shared=False, scale=f"fixed {low} to {high}")
    return figure


def plot_term_decomposition(terms: pd.DataFrame) -> Figure:
    """
    Plot the median RMS of each momentum term per backend.

    Returns
    -------
    matplotlib.figure.Figure
        Horizontal bars on a log axis with case ranges as error bars.

    """
    colors = backend_colors()
    columns = ("pressure", "drag", "viscous", "trace", "residual")
    backends = list(dict.fromkeys(terms["backend"]))
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=_SINGLE_PANEL_IN)
    axis = axes[0, 0]
    height = 0.8 / max(len(backends), 1)
    for position, backend in enumerate(backends):
        rows = terms[terms["backend"] == backend]
        median = rows[list(columns)].median().to_numpy()
        low = median - rows[list(columns)].min().to_numpy()
        high = rows[list(columns)].max().to_numpy() - median
        crop = int(rows["crop"].iloc[0])
        axis.barh(
            np.arange(len(columns)) + position * height,
            median,
            height=height * 0.9,
            xerr=np.vstack((low, high)),
            color=colors[backend],
            label=f"{_description(backend)}, crop {crop}",
        )
    axis.set_yticks(np.arange(len(columns)) + 0.4 - height / 2, [column.capitalize() for column in columns])
    axis.invert_yaxis()
    figure_layout.set_log_scale(axis, "x")
    axis.set_xlabel("Interior RMS / G")
    figure_layout.set_panel_title(axis, "Momentum term decomposition of the reference fields")
    _legend(figure, axes)
    return figure


def plot_derivative_difference(comparison: pd.DataFrame) -> Figure:
    """
    Plot backend derivative differences against the FD baseline per boundary band.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per output field; lines per backend and direction (median
        over cases).

    """
    colors = backend_colors()
    fields = list(dict.fromkeys(comparison["field"]))
    figure, axes = figure_layout.panel_figure(1, len(fields), sharey="row")
    positions = {band: index for index, band in enumerate(steady.BAND_LABELS)}
    for axis, field in zip(axes[0], fields, strict=True):
        rows = comparison[comparison["field"] == field]
        for (backend, direction), group in rows.groupby(["backend", "direction"], sort=False):
            median = group.groupby("band")["relative_difference"].median().reindex(steady.BAND_LABELS)
            axis.plot(
                [positions[band] for band in median.index],
                median.to_numpy(),
                "-" if direction == "x" else "--",
                color=colors[str(backend)],
                marker="o",
                markersize=4,
                linewidth=2.0,
                label=f"{str(backend).upper()} d/d{direction}",
            )
        axis.set_xticks(range(len(steady.BAND_LABELS)), steady.BAND_LABELS)
        axis.set_xlabel("Distance from nearest edge [cells]")
        figure_layout.set_panel_title(axis, presentation.field_labels.field_label(str(field), mathtext=True))
        figure_layout.set_log_scale(axis, "y")
        axis.grid(visible=True, which="major", color="0.9")
    figure_layout.label_rows_once(axes, "RMS(D - D_fd) / interior RMS(D_fd)")
    _legend(figure, axes)
    return figure


def plot_sensitivity(sensitivity: pd.DataFrame) -> Figure:
    """
    Plot perturbed-to-reference ratios of the three loss terms.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per loss term; perturbations on the y-axis, the median ratio
        over cases and seeds per backend setting on a log x-axis, the case and
        seed range as a line. Dotted lines mark ratios 1 and 10.

    """
    perturbations = list(dict.fromkeys(sensitivity["perturbation"]))
    settings = list(dict.fromkeys(zip(sensitivity["backend"], sensitivity["crop"], strict=True)))
    figure, axes = figure_layout.panel_figure(1, len(_LOSS_TERMS), panel_in=(4.2, 0.42 * len(perturbations) + 1.6), sharey="row")
    offsets = np.linspace(-0.25, 0.25, len(settings))
    markers = ("o", "s", "D", "^")
    for axis, term in zip(axes[0], _LOSS_TERMS, strict=True):
        for position, (backend, crop) in enumerate(settings):
            rows = sensitivity[(sensitivity["backend"] == backend) & (sensitivity["crop"] == crop)]
            stats = rows.groupby("perturbation")[f"{term}_ratio"].agg(["median", "min", "max"]).reindex(perturbations)
            y = np.arange(len(perturbations)) + offsets[position]
            color = series_color(position)
            axis.hlines(y, stats["min"], stats["max"], color=color, linewidth=1.5)
            axis.scatter(stats["median"], y, color=color, marker=markers[position % len(markers)], s=30, label=f"{backend.upper()}, crop {crop}")
        axis.axvline(1.0, **_REFERENCE_LINE)
        axis.axvline(10.0, **_REFERENCE_LINE)
        figure_layout.set_log_scale(axis, "x")
        figure_layout.set_panel_title(axis, _TERM_LABELS[term])
        axis.set_xlabel("Perturbed / reference")
        axis.grid(visible=True, axis="x", which="major", color="0.9")
    axes[0, 0].set_yticks(range(len(perturbations)), perturbations)
    axes[0, 0].invert_yaxis()
    _legend(figure, axes)
    return figure


def plot_mode_curves(errors: pd.DataFrame, settings: pd.DataFrame, *, title: str) -> Figure:
    """
    Plot the worst truncation error against the kept mode count per axis.

    Parameters
    ----------
    errors : pandas.DataFrame
        Steady or transient truncation-error rows of :mod:`component_checks_spectral`.
    settings : pandas.DataFrame
        :func:`component_checks_spectral.maintained_mode_settings`; its mode
        counts are marked by dotted lines.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per axis; colour per field, solid without and dashed with
        domain padding; maximum over cases (and states and increments).

    """
    fields = list(dict.fromkeys(errors["field"]))
    paddings = sorted(set(errors["padding"]))
    figure, axes = figure_layout.panel_figure(1, 2, panel_in=(5.0, 4.0), sharey="row")
    for axis, name in zip(axes[0], ("y", "x"), strict=True):
        curves = errors[errors["axis"] == name]
        for position, field in enumerate(fields):
            for padding, group in curves[curves["field"] == field].groupby("padding"):
                worst = group.groupby(f"modes_{name}")["relative_l2"].max()
                axis.plot(worst.index, worst.to_numpy(), "-" if padding == 0 else "--", color=series_color(position), marker="o", markersize=3)
        for count in sorted(set(settings[f"modes_{name}"])):
            axis.axvline(count, **_REFERENCE_LINE)
        axis.set_xscale("log", base=2)
        figure_layout.set_log_scale(axis, "y")
        axis.set_xlabel(f"Number of modes along {name} (other axis complete)")
        axis.grid(visible=True, which="major", color="0.9")
    figure_layout.label_rows_once(axes, "Relative L2 error / field std (worst case)")
    field_names = [presentation.field_labels.field_label(str(field), mathtext=True) for field in fields]
    padding_names = [f"Padding {padding:g}" for padding in paddings]
    keys = [(Line2D([], [], color=series_color(position), marker="o", markersize=3), name) for position, name in enumerate(field_names)]
    for padding, name in zip(paddings, padding_names, strict=True):
        keys.append((Line2D([], [], color="0.35", linestyle="-" if padding == 0 else "--"), name))
    handles = [handle for handle, _name in keys]
    _legend(figure, axes, extra=keys, groups={"Field": handles[: len(field_names)], "Domain padding": handles[len(field_names) :]})
    figure_layout.set_figure_title(figure, axes, title)
    return figure


def plot_transient_balances(balances: Mapping[str, pd.DataFrame], *, rule: str = "trapezoid") -> Figure:
    """
    Plot per-interval COMSOL balance closures relative to the grain water change.

    Returns
    -------
    matplotlib.figure.Figure
        Panels for grain, gas, and total closure; one line per case.

    """
    figure, axes = figure_layout.panel_figure(1, 3, sharey="row")
    for axis, (name, quantity) in zip(axes[0], (("grain", "grain water"), ("gas", "gas vapour"), ("total", "total water")), strict=True):
        for position, (label, frame) in enumerate(balances.items()):
            scale = frame["grain_change_kg"].abs().clip(lower=np.finfo(float).tiny)
            style = _case_style(position, len(balances), label)
            axis.plot(frame["end_h"], (frame[f"{name}_closure_kg_{rule}"].abs() / scale), linewidth=1.5, **style)
        figure_layout.set_log_scale(axis, "y")
        figure_layout.set_panel_title(axis, f"{quantity.capitalize()} balance ({rule})")
        axis.set_xlabel("Time [h]")
        axis.grid(visible=True, which="major", color="0.9")
    figure_layout.label_rows_once(axes, "|closure| / |grain water change|")
    _case_legend(figure, axes, len(balances))
    return figure


def plot_field_vs_comsol(comparisons: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot relative differences of field reconstructions against COMSOL globals.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per reconstructed quantity; one line per case.

    """
    quantities = ("grain_water", "gas_vapour", "evaporation", "inlet_flux", "outlet_flux")
    figure, axes = figure_layout.panel_figure(1, len(quantities), panel_in=_PANEL_ROW_IN, sharey="row")
    for axis, quantity in zip(axes[0], quantities, strict=True):
        for position, (label, frame) in enumerate(comparisons.items()):
            style = _case_style(position, len(comparisons), label)
            axis.plot(frame["time_h"], frame[f"{quantity}_relative_difference"].abs(), linewidth=1.5, **style)
        figure_layout.set_log_scale(axis, "y")
        figure_layout.set_panel_title(axis, quantity.replace("_", " ").capitalize())
        axis.set_xlabel("Time [h]")
        axis.grid(visible=True, which="major", color="0.9")
    figure_layout.label_rows_once(axes, "|field / COMSOL - 1|")
    _case_legend(figure, axes, len(comparisons))
    return figure


def plot_sorption_residuals(residuals: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot relative sorption-equation residuals over time for both quadrature rules.

    Returns
    -------
    matplotlib.figure.Figure
        Panels for the internal, surface, and grain equations; solid lines use
        the trapezoidal rule, dashed lines the left-endpoint rule.

    """
    figure, axes = figure_layout.panel_figure(1, 3, sharey="row")
    for axis, (name, field) in zip(axes[0], (("internal", "w_int"), ("surface", "w_surf"), ("grain", "w_gr")), strict=True):
        for position, (label, frame) in enumerate(residuals.items()):
            case_style = _case_style(position, len(residuals), label)
            for rule, line, _name in _QUADRATURE_RULES:
                rows = frame[frame["rule"] == rule]
                style = case_style if rule == "trapezoid" else {**case_style, "label": None}
                axis.plot(rows["end_h"], rows[f"{name}_relative"], line, linewidth=1.5, **style)
        figure_layout.set_log_scale(axis, "y")
        figure_layout.set_panel_title(axis, f"{presentation.field_labels.field_label(field, mathtext=True)} equation")
        axis.set_xlabel("Time [h]")
        axis.grid(visible=True, which="major", color="0.9")
    figure_layout.label_rows_once(axes, "RMS residual / RMS storage rate")
    keys = [(Line2D([], [], color="0.35", linestyle=line), name) for _rule, line, name in _QUADRATURE_RULES]
    cases, _labels = figure_layout.legend_entries(axes.flat)  # each case's labelled trapezoid line, owned by handle: a case may share a rule's name
    groups: dict[str, Sequence[Artist]] = {"Case": cases, "Quadrature rule": [handle for handle, _name in keys]}
    _case_legend(figure, axes, len(residuals), extra=keys, groups=groups)
    return figure
