"""Render physics-diagnostic tables as static scientific figures.

Responsibilities:
  - Plot steady residual floors, crop sweeps, edge profiles, and term decompositions
  - Plot residual maps with the shared field-map scale contract
  - Plot derivative-backend differences and perturbation sensitivity ratios
  - Plot transient balance closures, field-versus-COMSOL differences, and sorption residuals

Design principles:
  - Figures consume diagnostic tables and never recompute physics
  - Categorical colours follow one fixed colour-blind-checked order, never a cycle
  - Logarithmic magnitude axes; one quantity per axis

This module does NOT:
  - Load data, evaluate residuals or balances, or save files
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation

from src.analysis import presentation

from . import physics_diagnostics_steady as steady

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_LOSS_TERMS: Final = ("momentum", "flux_balance", "boundary")
_TERM_LABELS: Final = {
    "momentum": "momentum (loss)",
    "pointwise_divergence": "pointwise div u",
    "porous_divergence": "pointwise div(eps u)",
    "flux_balance": "flux balance (loss)",
    "boundary": "pressure boundary (loss)",
}
_EDGE_LABELS: Final = {"inlet": "inlet (y = 0)", "outlet": "outlet (y = L_y)", "wall_x_min": "wall x = 0", "wall_x_max": "wall x = L_x"}
_REFERENCE_LINE: Final = {"color": "0.5", "linewidth": 1.0, "linestyle": ":"}
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


def _case_legend(axis: Axes, count: int) -> None:
    """Add a case legend when cases carry categorical colours, otherwise a case count."""
    if count <= len(SERIES_COLORS):
        axis.legend(frameon=False, fontsize="small")
    else:
        axis.set_title(f"{axis.get_title()} ({count} cases)")


def backend_colors() -> dict[str, str]:
    """Return one fixed colour per declared derivative backend label."""
    return {backend.label: series_color(position) for position, backend in enumerate(steady.DERIVATIVE_BACKENDS)}


def _description(label: str) -> str:
    """Return the readable description of a backend label."""
    return steady.backend_by_label(label).description


def plot_crop_sweep(residuals: pd.DataFrame, *, dtype: str = "float32") -> Figure:
    """
    Plot reference momentum terms against the interior crop per backend.

    Parameters
    ----------
    residuals : pandas.DataFrame
        Output of :func:`physics_diagnostics_steady.summarize_reference_residuals`.
    dtype : str, optional
        Precision rows to show.

    Returns
    -------
    matplotlib.figure.Figure
        Median over cases with the case range as a band.

    """
    colors = backend_colors()
    selected = residuals[residuals["dtype"] == dtype]
    figure, axis = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    for backend, group in selected.groupby("backend", sort=False):
        stats = group.groupby("crop")["momentum"].agg(["median", "min", "max"])
        color = colors[str(backend)]
        axis.plot(stats.index, stats["median"], "-", color=color, linewidth=2.0, marker="o", markersize=5, label=_description(str(backend)))
        axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.15, linewidth=0)
    axis.set_yscale("log")
    axis.set_xlabel("interior crop [cells per edge]")
    axis.set_ylabel("mean (Rx² + Ry²) / G²")
    axis.set_title(f"Reference momentum residual vs crop ({dtype})")
    axis.grid(visible=True, which="major", color="0.9")
    axis.legend(frameon=False)
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
    figure, axis = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
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
    axis.set_xscale("log")
    axis.set_xlabel("dimensionless mean square (per case)")
    axis.set_title(f"Reference residual terms ({dtype})")
    axis.grid(visible=True, axis="x", which="major", color="0.9")
    axis.legend(frameon=False, loc="lower right", fontsize="small")
    return figure


def plot_edge_profiles(profiles: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot momentum-residual RMS against the distance from each grid edge.

    Parameters
    ----------
    profiles : Mapping[str, pandas.DataFrame]
        Backend label to :func:`physics_diagnostics_steady.edge_profiles` output.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per edge with a shared log y-axis.

    """
    colors = backend_colors()
    figure, axes = plt.subplots(1, len(steady.EDGES), figsize=(13.0, 3.6), sharey=True, layout="constrained")
    for axis, edge in zip(axes, steady.EDGES, strict=True):
        for backend, frame in profiles.items():
            rows = frame[frame["edge"] == edge]
            axis.plot(rows["distance"], rows["rms"], color=colors[backend], linewidth=2.0, marker="o", markersize=4, label=_description(backend))
        axis.set_title(_EDGE_LABELS[edge])
        axis.set_xlabel("distance from edge [cells]")
        axis.set_yscale("log")
        axis.grid(visible=True, which="major", color="0.9")
    axes[0].set_ylabel("RMS |R| / G")
    axes[0].legend(frameon=False, fontsize="small")
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
        Maps with one shared logarithmic eleven-colour scale.

    """
    magnitudes = [steady.residual_magnitude(item)[case_index] for item in fields]
    positive = np.concatenate([value[value > 0].ravel() for value in magnitudes])
    scale = presentation.field_maps.build_field_map(positive, semantic="log_positive", field="|R|/G", unit="1", locked_boundaries=MAP_BOUNDARIES)
    figure, axes = plt.subplots(1, len(fields), figsize=(5.6 * len(fields) + 1.2, 4.4), sharey=True, layout="constrained", squeeze=False)
    image = None
    for axis, item, magnitude in zip(axes[0], fields, magnitudes, strict=True):
        image = axis.imshow(np.maximum(magnitude, MAP_BOUNDARIES[0]), origin="lower", extent=extent, aspect="equal", **scale.imshow_kwargs())
        axis.set_title(_description(item.backend), fontsize="small")
        axis.set_xlabel("x [m]")
    axes[0][0].set_ylabel("y [m]")
    if image is not None:
        colorbar = figure.colorbar(image, ax=list(axes[0]), shrink=0.9, extend=scale.extend)
        colorbar.set_ticks([10.0**exponent for exponent in range(-5, 1)])
        colorbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation())
        colorbar.set_label("|R| / G")
    figure.suptitle(f"Momentum residual magnitude, {case_label}")
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
    figure, axis = plt.subplots(figsize=(8.0, 4.4), layout="constrained")
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
    axis.set_yticks(np.arange(len(columns)) + 0.4 - height / 2, columns)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("interior RMS / G")
    axis.set_title("Momentum term decomposition of the reference fields")
    axis.legend(frameon=False, fontsize="small")
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
    figure, axes = plt.subplots(1, len(fields), figsize=(12.0, 3.8), sharey=True, layout="constrained")
    positions = {band: index for index, band in enumerate(steady.BAND_LABELS)}
    for axis, field in zip(np.atleast_1d(axes), fields, strict=True):
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
                label=f"{backend} d/d{direction}",
            )
        axis.set_xticks(range(len(steady.BAND_LABELS)), steady.BAND_LABELS)
        axis.set_xlabel("distance from nearest edge [cells]")
        axis.set_title(f"field {field}")
        axis.set_yscale("log")
        axis.grid(visible=True, which="major", color="0.9")
    np.atleast_1d(axes)[0].set_ylabel("RMS(D - D_fd) / interior RMS(D_fd)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize="small")
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
    figure, axes = plt.subplots(1, len(_LOSS_TERMS), figsize=(14.0, 0.42 * len(perturbations) + 1.6), sharey=True, layout="constrained")
    offsets = np.linspace(-0.25, 0.25, len(settings))
    markers = ("o", "s", "D", "^")
    for axis, term in zip(axes, _LOSS_TERMS, strict=True):
        for position, (backend, crop) in enumerate(settings):
            rows = sensitivity[(sensitivity["backend"] == backend) & (sensitivity["crop"] == crop)]
            stats = rows.groupby("perturbation")[f"{term}_ratio"].agg(["median", "min", "max"]).reindex(perturbations)
            y = np.arange(len(perturbations)) + offsets[position]
            color = series_color(position)
            axis.hlines(y, stats["min"], stats["max"], color=color, linewidth=1.5)
            axis.scatter(stats["median"], y, color=color, marker=markers[position % len(markers)], s=30, label=f"{backend}, crop {crop}")
        axis.axvline(1.0, **_REFERENCE_LINE)
        axis.axvline(10.0, **_REFERENCE_LINE)
        axis.set_xscale("log")
        axis.set_title(_TERM_LABELS[term])
        axis.set_xlabel("perturbed / reference")
        axis.grid(visible=True, axis="x", which="major", color="0.9")
    axes[0].set_yticks(range(len(perturbations)), perturbations)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize="small")
    return figure


def plot_transient_balances(balances: Mapping[str, pd.DataFrame], *, rule: str = "trapezoid") -> Figure:
    """
    Plot per-interval COMSOL balance closures relative to the grain water change.

    Returns
    -------
    matplotlib.figure.Figure
        Panels for grain, gas, and total closure; one line per case.

    """
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), sharey=True, layout="constrained")
    for axis, name in zip(axes, ("grain", "gas", "total"), strict=True):
        for position, (label, frame) in enumerate(balances.items()):
            scale = frame["grain_change_kg"].abs().clip(lower=np.finfo(float).tiny)
            style = _case_style(position, len(balances), label)
            axis.plot(frame["end_h"], (frame[f"{name}_closure_kg_{rule}"].abs() / scale), linewidth=1.5, **style)
        axis.set_yscale("log")
        axis.set_title(f"{name} balance ({rule})")
        axis.set_xlabel("time [h]")
        axis.grid(visible=True, which="major", color="0.9")
    axes[0].set_ylabel("|closure| / |grain water change|")
    _case_legend(axes[0], len(balances))
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
    figure, axes = plt.subplots(1, len(quantities), figsize=(16.0, 3.6), sharey=True, layout="constrained")
    for axis, quantity in zip(axes, quantities, strict=True):
        for position, (label, frame) in enumerate(comparisons.items()):
            style = _case_style(position, len(comparisons), label)
            axis.plot(frame["time_h"], frame[f"{quantity}_relative_difference"].abs(), linewidth=1.5, **style)
        axis.set_yscale("log")
        axis.set_title(quantity.replace("_", " "))
        axis.set_xlabel("time [h]")
        axis.grid(visible=True, which="major", color="0.9")
    axes[0].set_ylabel("|field / COMSOL - 1|")
    _case_legend(axes[0], len(comparisons))
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
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), sharey=True, layout="constrained")
    for axis, name in zip(axes, ("internal", "surface", "grain"), strict=True):
        for position, (label, frame) in enumerate(residuals.items()):
            case_style = _case_style(position, len(residuals), label)
            for rule, line in (("trapezoid", "-"), ("left", "--")):
                rows = frame[frame["rule"] == rule]
                style = case_style if rule == "trapezoid" else {**case_style, "label": None}
                axis.plot(rows["end_h"], rows[f"{name}_relative"], line, linewidth=1.5, **style)
        axis.set_yscale("log")
        axis.set_title(f"{name} equation")
        axis.set_xlabel("time [h]")
        axis.grid(visible=True, which="major", color="0.9")
    axes[0].set_ylabel("RMS residual / RMS storage rate")
    _case_legend(axes[0], len(residuals))
    return figure
