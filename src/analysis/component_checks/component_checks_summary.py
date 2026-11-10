"""Apply the proposed component criteria to the measured component-check tables.

Responsibilities:
  - Name every proposed criterion threshold once
  - Reduce the crop, timing, mode-truncation, stride, and increment tables to one row per component
  - Mark components whose tables were not measured instead of guessing them

Design principles:
  - Criteria are proposals while the decision criteria are open; a status is a prior, never a decision
  - Each criterion constant is applied in exactly one place

This module does NOT:
  - Measure anything, change a configuration, or persist results
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pandas as pd

if TYPE_CHECKING:
    from .component_checks_increments import TransientComponentTables

CRITERIA_STATE: Final = "proposed, decision criteria open"
FC_FLOOR_RATIO_MAX: Final = 1.0
FC_COST_RATIO_MAX: Final = 2.0
FC_CROP: Final = 2
CROP_SENSITIVITY_MIN: Final = 100.0
MODES_LOWER_BOUND_ERROR: Final = 1e-3
MODES_SUFFICIENT_ERROR: Final = 1e-4
PADDING_NO_EFFECT_RELATIVE: Final = 0.10
STRIDE_FLOOR_MAX: Final = 1e-3
HUBER_ABOVE_BETA_MAX: Final = 0.0
NOT_MEASURED: Final = "not measured"


def _row(component: str, measured: str, criterion: str, status: str) -> dict[str, str]:
    """Return one summary row with the criterion marked as proposed."""
    return {"component": component, "measured": measured, "criterion": f"{criterion} ({CRITERIA_STATE})", "status": status}


def _backend_row(crop: pd.DataFrame | None, timing: pd.DataFrame | None) -> dict[str, str]:
    """Return the FC-arm row from the crop floors and the measured batch cost."""
    criterion = f"FC crop-{FC_CROP} floor <= {FC_FLOOR_RATIO_MAX:g} x FD and batch cost <= {FC_COST_RATIO_MAX:g} x FD"
    measured = timing[timing["status"] == "measured"] if timing is not None else None
    if crop is None or measured is None or measured.empty:
        return _row("Derivative backend: FC arm", NOT_MEASURED, criterion, NOT_MEASURED)
    floors = crop["momentum_floor_median"].xs(FC_CROP, level="crop")
    floor = float(floors["fc"]) / float(floors["fd"])
    device = "cuda" if (measured["device"] == "cuda").any() else "cpu"
    cost_ms = measured[measured["device"] == device].set_index("backend")["median_ms"]
    cost = float(cost_ms["fc"]) / float(cost_ms["fd"])
    keep = floor <= FC_FLOOR_RATIO_MAX and cost <= FC_COST_RATIO_MAX
    return _row(
        "Derivative backend: FC arm",
        f"floor ratio {floor:.3g}, cost ratio {cost:.3g} ({device})",
        criterion,
        "keep FC arm" if keep else "drop FC arm",
    )


def _crop_row(crop: pd.DataFrame | None) -> dict[str, str]:
    """Return the smallest crop per backend whose case-median smooth 1 % ratio reaches the threshold."""
    criterion = f"smallest crop with smooth 1 % momentum ratio >= {CROP_SENSITIVITY_MIN:g}"
    if crop is None:
        return _row("Interior crop", NOT_MEASURED, criterion, NOT_MEASURED)
    measured, status = [], []
    for backend, rows in crop.groupby(level="backend", sort=False):
        ratios = rows["smooth_1pct_ratio_median"].droplevel("backend")
        passing = ratios[ratios >= CROP_SENSITIVITY_MIN]
        measured.append(f"{backend}: " + ", ".join(f"crop {index} {value:.3g}" for index, value in ratios.items()))
        status.append(f"{backend}: crop {passing.index.min()}" if not passing.empty else f"{backend}: none")
    return _row("Interior crop", "; ".join(measured), criterion, "; ".join(status))


def _smallest(curve: pd.Series, threshold: float) -> str:
    """Return the smallest mode count whose worst error stays within ``threshold``."""
    passing = curve[curve <= threshold]
    return str(int(passing.index.min())) if not passing.empty else "none"


def _mode_rows(task: str, errors: pd.DataFrame | None) -> list[dict[str, str]]:
    """Return per-axis mode bounds from the worst relative error over cases, fields, and quantities."""
    criterion = f"smallest count with error <= {MODES_LOWER_BOUND_ERROR:g} bounds HPO from below; <= {MODES_SUFFICIENT_ERROR:g} sufficient"
    if errors is None:
        return [_row(f"n_modes {task}", NOT_MEASURED, criterion, NOT_MEASURED)]
    rows = []
    for axis in ("y", "x"):
        curves = errors[errors["axis"] == axis]
        measured, status = [], []
        for padding, group in curves.groupby("padding"):
            worst = group.groupby(f"modes_{axis}")["relative_l2"].max()
            worst.index = worst.index.astype(int)
            lower, sufficient = _smallest(worst, MODES_LOWER_BOUND_ERROR), _smallest(worst, MODES_SUFFICIENT_ERROR)
            measured.append(f"padding {padding:g}: worst error {worst.min():.2g} at {worst.idxmin()} modes")
            status.append(f"padding {padding:g}: lower bound {lower}, sufficient {sufficient}")
        rows.append(_row(f"n_modes {task}, axis {axis}", "; ".join(measured), criterion, "; ".join(status)))
    return rows


def _padding_row(task: str, errors: pd.DataFrame | None) -> dict[str, str]:
    """Return the largest relative edge-band difference between paddings at the maintained mode pairs."""
    criterion = f"no effect if edge-band errors differ by < {PADDING_NO_EFFECT_RELATIVE:.0%}"
    configured = None if errors is None else errors[errors["axis"] == "config"]
    if configured is None or configured.empty or configured["padding"].nunique() < 2:  # noqa: PLR2004 -- two paddings compared
        return _row(f"Domain padding {task}", NOT_MEASURED, criterion, NOT_MEASURED)
    keys = [column for column in ("case", "quantity", "field", "config") if column in configured.columns]
    edges = configured.pivot_table(index=keys, columns="padding", values="edge_relative", aggfunc="max")
    peak = edges.max(axis=1)
    # Rows whose errors are all zero agree exactly; their 0/0 counts as no difference.
    difference = ((peak - edges.min(axis=1)) / peak).where(peak > 0, 0.0)
    largest = float(difference.max())
    status = "no effect" if largest < PADDING_NO_EFFECT_RELATIVE else "effect"
    return _row(f"Domain padding {task}", f"largest relative edge-band difference {largest:.3g}", criterion, status)


def _stride_row(stride: pd.DataFrame | None) -> dict[str, str]:
    """Return the worst full-grid stride floor per stride."""
    criterion = f"stride admissible if floor <= {STRIDE_FLOOR_MAX:g} normalized RMSE (full-grid evaluation only)"
    if stride is None:
        return _row("Spatial stride", NOT_MEASURED, criterion, NOT_MEASURED)
    measured, status = [], []
    for value, rows in stride.groupby("stride"):
        if (rows["status"] != "admissible").any():
            measured.append(f"stride {value}: refused by the endpoint rule")
            status.append(f"stride {value}: refused")
            continue
        worst = float(rows["max_normalized_rmse"].max())
        measured.append(f"stride {value}: {worst:.3g}")
        status.append(f"stride {value}: {'admissible' if worst <= STRIDE_FLOOR_MAX else 'not admissible'}")
    return _row("Spatial stride", "; ".join(measured), criterion, "; ".join(status))


def _huber_row(increments: pd.DataFrame | None) -> dict[str, str]:
    """Return whether any scaled increment reaches the linear Huber branch."""
    criterion = f"Huber = MSE prior if the fraction above beta is {HUBER_ABOVE_BETA_MAX:g}"
    if increments is None:
        return _row("Huber regime", NOT_MEASURED, criterion, NOT_MEASURED)
    largest = float(increments["above_beta_fraction"].max())
    status = "Huber = MSE prior" if largest <= HUBER_ABOVE_BETA_MAX else "Huber linear branch reached"
    return _row("Huber regime", f"largest fraction above beta {largest:.3g}", criterion, status)


def summary_table(
    *,
    crop: pd.DataFrame | None = None,
    timing: pd.DataFrame | None = None,
    steady_modes: pd.DataFrame | None = None,
    transient: TransientComponentTables | None = None,
) -> pd.DataFrame:
    """
    Return one row per component: measured value, proposed criterion, and status.

    Parameters
    ----------
    crop : pandas.DataFrame, optional
        :func:`component_checks_steady.crop_overview`.
    timing : pandas.DataFrame, optional
        :func:`component_checks_timing.time_physics_loss`.
    steady_modes : pandas.DataFrame, optional
        :func:`component_checks_spectral.steady_truncation_errors`.
    transient : TransientComponentTables, optional
        :func:`component_checks_increments.scan_transient_cases`.

    Returns
    -------
    pandas.DataFrame
        Columns ``component``, ``measured``, ``criterion`` and ``status``;
        a missing table yields ``not measured``.

    """
    transient_modes = None if transient is None else transient.spectral
    rows = [
        _backend_row(crop, timing),
        _crop_row(crop),
        *_mode_rows("steady", steady_modes),
        _padding_row("steady", steady_modes),
        *_mode_rows("transient", transient_modes),
        _padding_row("transient", transient_modes),
        _stride_row(None if transient is None else transient.stride),
        _huber_row(None if transient is None else transient.increments),
    ]
    return pd.DataFrame(rows).set_index("component")
