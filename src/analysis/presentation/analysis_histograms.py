"""
analysis_histograms.py

Render maintained analysis histograms with exact constant-data semantics.

Responsibilities:
  - Preserve caller-owned histogram normalization and styling
  - Render exact constant values as one vertical line without histogram patches
  - Return the created histogram artists for downstream scientific annotation

Design principles:
  - Constant classification uses exact equality after caller validity filtering
  - Non-constant inputs delegate unchanged to Matplotlib histogram binning
  - Constant line heights preserve count, weight, or density semantics

This module does NOT:
  - Filter invalid values, choose non-constant bins, or infer scientific units
  - Own subplot layout, legends, titles, or dataset colors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.lines import Line2D


@dataclass(frozen=True, slots=True)
class HistogramArtists:
    """Describe one histogram and its optional exact-constant line."""

    heights: np.ndarray
    bin_edges: np.ndarray
    bars: BarContainer | None
    constant_line: Line2D | None
    constant_value: float | None


def plot_histogram(
    axis: Axes,
    values: Sequence[float] | np.ndarray,
    *,
    bins: int | Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
    density: bool = False,
    **kwargs: Any,
) -> HistogramArtists:
    """Plot one maintained histogram with exact line-only constant handling."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        message = "Histogram values must be one non-empty finite numeric sequence."
        raise ValueError(message)
    resolved_weights: np.ndarray | None = None
    if weights is not None:
        resolved_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if resolved_weights.shape != array.shape or not np.isfinite(resolved_weights).all():
            message = "Histogram weights must be finite and align exactly with values."
            raise ValueError(message)
    if not isinstance(density, bool):
        message = "Histogram density mode must be boolean."
        raise TypeError(message)

    constant = bool(np.all(array == array[0]))
    constant_value = float(array[0]) if constant else None
    if constant_value is not None:
        total = float(array.size) if resolved_weights is None else float(np.sum(resolved_weights))
        if density:
            reference_width = 0.02 * max(1.0, abs(constant_value))
            height = 0.0 if total == 0.0 else 1.0 / reference_width
        else:
            height = total
        color = kwargs.get("color")
        zorder = float(kwargs.get("zorder", 1.0)) + 1.0
        (line,) = axis.plot(
            (constant_value, constant_value),
            (0.0, height),
            color=color,
            linewidth=1.6,
            label="_nolegend_",
            zorder=zorder,
        )
        return HistogramArtists(
            heights=np.asarray((height,), dtype=np.float64),
            bin_edges=np.empty(0, dtype=np.float64),
            bars=None,
            constant_line=line,
            constant_value=constant_value,
        )

    heights, bin_edges, bars = axis.hist(
        array,
        bins=bins,
        weights=resolved_weights,
        density=density,
        **kwargs,
    )
    return HistogramArtists(
        heights=np.asarray(heights, dtype=np.float64),
        bin_edges=np.asarray(bin_edges, dtype=np.float64),
        bars=bars,
        constant_line=None,
        constant_value=None,
    )
