"""
===============================================================================
generation_input_plot_layout.py
===============================================================================
Define notebook-width map, normalization, and colorbar contracts.
Responsibilities:
  - Create bounded generation-input map and composite figure grids
  - Attach every map colorbar directly to its associated map axis
  - Centralize independent, locked, and signed physical normalization
  - Record map-to-colorbar bindings for focused geometry verification
Design principles:
  - Colorbar geometry follows settled map-axis geometry after canvas drawing
  - Every map retains its own axes-coupled colorbar
  - Presentation helpers never transform or clip displayed scientific values
This module does NOT:
  - Load generation inputs, derive fields, or select cases
  - Import evaluation internals or use manually positioned colorbar axes
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from weakref import WeakKeyDictionary

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore[import-untyped]

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec


@dataclass(frozen=True, slots=True)
class GenerationInputMapLayout:
    """Define one centrally configured generation-input notebook layout."""

    notebook_width: float = 17.4
    minimum_width: float = 5.8
    map_column_width: float = 5.8
    map_row_height: float = 3.6
    composite_map_row_height: float = 3.6
    distribution_row_height: float = 2.75
    schedule_row_height: float = 2.75
    table_row_height: float = 2.2
    horizontal_spacing: float = 0.135
    vertical_spacing: float = 0.11
    relation_vertical_spacing: float = 0.04
    relation_height_ratios: tuple[float, float] = (0.78, 0.22)
    colorbar_width_fraction: float = 0.046
    colorbar_pad_fraction: float = 0.025
    outer_right_reserve_inches: float = 0.72
    title_size: float = 15.0
    axis_title_size: float = 12.5
    label_size: float = 11.25
    tick_size: float = 10.0
    legend_size: float = 9.0
    table_size: float = 9.0
    line_width: float = 1.4

    def figure_width(self, columns: int) -> float:
        """Return a positive notebook-bounded figure width."""
        if columns <= 0:
            message = "Generation-input figures require at least one column."
            raise ValueError(message)
        return min(self.notebook_width, max(self.minimum_width, self.map_column_width * columns))

    def figure_height(self, row_heights: Sequence[float]) -> float:
        """Return the central height for one explicit row composition."""
        heights = tuple(float(value) for value in row_heights)
        if not heights or not np.isfinite(heights).all() or any(value <= 0.0 for value in heights):
            message = "Generation-input figure row heights must be positive and finite."
            raise ValueError(message)
        return float(sum(heights) + self.vertical_spacing * (len(heights) - 1))


MAP_LAYOUT: Final = GenerationInputMapLayout()
DATASET_A_COLOR: Final = "tab:blue"
DATASET_B_COLOR: Final = "tab:orange"
DATASET_MEAN_COLOR: Final = "0.30"
_COLORBAR_BINDINGS: WeakKeyDictionary[Figure, list[MapColorbarBinding]]
_COMPACT_QUANTITY_LABELS: Final = {
    "K_anisotropy": "Anisotropy ratio",
    "X_0_db_field": "Dry-basis moisture",
    "phi_eq": "Equilibrium RH",
    "rho_bu_dry": "Dry bulk density",
    "w_gr0": "Initial water content",
}


@dataclass(frozen=True, slots=True)
class MapColorbarBinding:
    """Bind one colorbar axis to its anchor and comparable map axes."""

    colorbar: Colorbar
    anchor_axis: Axes
    map_axes: tuple[Axes, ...]
    label: str


_COLORBAR_BINDINGS = WeakKeyDictionary()


def _reserve_outer_colorbar_labels(figure: Figure) -> None:
    """Reserve a fixed physical right margin for appended colorbar artists."""
    engine = figure.get_layout_engine()
    if not isinstance(engine, ConstrainedLayoutEngine):
        message = "Generation-input figures require constrained layout."
        raise TypeError(message)
    right = 1.0 - MAP_LAYOUT.outer_right_reserve_inches / figure.get_figwidth()
    if not 0.0 < right < 1.0:
        message = "Generation-input figure width cannot hold the colorbar label reserve."
        raise ValueError(message)
    engine.set(rect=(0.0, 0.0, right, 1.0))


def figure_grid(
    *,
    columns: int,
    row_heights: Sequence[float],
) -> tuple[Figure, GridSpec]:
    """Create one constrained notebook-width figure and explicit GridSpec."""
    heights = tuple(float(value) for value in row_heights)
    figure = plt.figure(
        figsize=(MAP_LAYOUT.figure_width(columns), MAP_LAYOUT.figure_height(heights)),
        layout="constrained",
    )
    grid = figure.add_gridspec(
        len(heights),
        columns,
        height_ratios=heights,
        wspace=MAP_LAYOUT.horizontal_spacing,
        hspace=MAP_LAYOUT.vertical_spacing,
    )
    return figure, grid


def map_subplots(*, rows: int, columns: int) -> tuple[Figure, np.ndarray]:
    """Create one all-map grid at the shared notebook-native scale."""
    if rows <= 0 or columns <= 0:
        message = "Generation-input map grids require positive dimensions."
        raise ValueError(message)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            MAP_LAYOUT.figure_width(columns),
            MAP_LAYOUT.figure_height((MAP_LAYOUT.map_row_height,) * rows),
        ),
        squeeze=False,
        layout="constrained",
        gridspec_kw={
            "wspace": MAP_LAYOUT.horizontal_spacing,
            "hspace": MAP_LAYOUT.vertical_spacing,
        },
    )
    return figure, axes


def linear_norm(*values: np.ndarray) -> Normalize:
    """Return one finite linear normalization shared by all supplied arrays."""
    if not values:
        message = "A physical normalization requires at least one array."
        raise ValueError(message)
    combined = np.concatenate(tuple(np.ravel(np.asarray(value, dtype=np.float64)) for value in values))
    if not np.isfinite(combined).all():
        message = "Map normalization requires finite physical values."
        raise ValueError(message)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if lower == upper:
        padding = max(abs(lower), 1.0) * 1.0e-12
        lower -= padding
        upper += padding
    return Normalize(vmin=lower, vmax=upper)


def signed_norm(*values: np.ndarray) -> TwoSlopeNorm:
    """Return one symmetric finite normalization centred exactly at zero."""
    if not values:
        message = "A signed normalization requires at least one array."
        raise ValueError(message)
    combined = np.concatenate(tuple(np.ravel(np.asarray(value, dtype=np.float64)) for value in values))
    if not np.isfinite(combined).all():
        message = "Signed map normalization requires finite physical values."
        raise ValueError(message)
    limit = max(float(np.max(np.abs(combined))), float(np.finfo(np.float64).eps))
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def physical_norm(quantity: str, *values: np.ndarray) -> Normalize:
    """Return one data-driven physical normalization for a quantity."""
    if quantity == "Kxy":
        return signed_norm(*values)
    return linear_norm(*values)


def comparison_norms(
    quantity: str,
    first: np.ndarray,
    second: np.ndarray,
    *,
    lock_scale: bool,
) -> tuple[Normalize, Normalize]:
    """
    Return independent or exactly shared A/B physical normalizations.

    Signed Kxy fields remain zero-centred. Locking uses the same normalization
    object for both cases; unlocked maps use separate data-driven objects.
    """
    if lock_scale:
        shared = physical_norm(quantity, first, second)
        return shared, shared
    return physical_norm(quantity, first), physical_norm(quantity, second)


def draw_array_map(
    axis: Axes,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    norm: Normalize,
    cmap: str | None = None,
) -> ScalarMappable:
    """Draw one exact structured-grid array without creating a colorbar."""
    resolved_cmap = "coolwarm" if isinstance(norm, TwoSlopeNorm) else "viridis"
    image = axis.pcolormesh(
        x_coordinates,
        y_coordinates,
        values,
        shading="auto",
        cmap=resolved_cmap if cmap is None else cmap,
        norm=norm,
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]", fontsize=MAP_LAYOUT.label_size)
    axis.set_ylabel("y [m]", fontsize=MAP_LAYOUT.label_size)
    axis.set_title(title, fontsize=MAP_LAYOUT.axis_title_size)
    axis.tick_params(labelsize=MAP_LAYOUT.tick_size)
    return image


def draw_map(
    axis: Axes,
    record: diagnostics.GenerationInputDiagnostics,
    quantity: str,
    *,
    title: str,
    norm: Normalize | None = None,
    cmap: str | None = None,
) -> ScalarMappable:
    """Draw one exact structured-grid field without creating a colorbar."""
    if quantity not in diagnostics.display_field_names(record):
        message = f"Unknown generation-input map quantity {quantity!r}."
        raise ValueError(message)
    values = record.fields[quantity]
    resolved_norm = physical_norm(quantity, values) if norm is None else norm
    return draw_array_map(
        axis,
        record.fields["x"],
        record.fields["y"],
        values,
        title=title,
        norm=resolved_norm,
        cmap=cmap,
    )


def _bindings(figure: Figure) -> list[MapColorbarBinding]:
    """Return the weakly held colorbar-binding list owned by one figure."""
    try:
        return _COLORBAR_BINDINGS[figure]
    except KeyError:
        bindings: list[MapColorbarBinding] = []
        _COLORBAR_BINDINGS[figure] = bindings
        return bindings


def add_map_colorbar(
    figure: Figure,
    image: ScalarMappable,
    anchor_axis: Axes,
    *,
    label: str,
    map_axes: Sequence[Axes] | None = None,
) -> Colorbar:
    """Attach one axes-coupled colorbar under the central width/padding contract."""
    _reserve_outer_colorbar_labels(figure)
    associated = tuple(map_axes) if map_axes is not None else (anchor_axis,)
    if not associated or anchor_axis not in associated:
        message = "A map colorbar binding must include its anchor axis."
        raise ValueError(message)
    for comparable_axis in associated:
        if comparable_axis is anchor_axis:
            continue
        spacer_divider = make_axes_locatable(comparable_axis)
        spacer_axis = spacer_divider.append_axes(
            "right",
            size=f"{100.0 * MAP_LAYOUT.colorbar_width_fraction:g}%",
            pad=f"{100.0 * MAP_LAYOUT.colorbar_pad_fraction:g}%",
        )
        spacer_axis.set_axis_off()
    divider = make_axes_locatable(anchor_axis)
    colorbar_axis = divider.append_axes(
        "right",
        size=f"{100.0 * MAP_LAYOUT.colorbar_width_fraction:g}%",
        pad=f"{100.0 * MAP_LAYOUT.colorbar_pad_fraction:g}%",
    )
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(label, fontsize=MAP_LAYOUT.label_size)
    colorbar.ax.tick_params(labelsize=MAP_LAYOUT.tick_size)
    _bindings(figure).append(
        MapColorbarBinding(
            colorbar=colorbar,
            anchor_axis=anchor_axis,
            map_axes=associated,
            label=label,
        )
    )
    return colorbar


def map_colorbar_bindings(figure: Figure) -> tuple[MapColorbarBinding, ...]:
    """Return immutable map/colorbar bindings for geometry verification."""
    return tuple(_bindings(figure))


def quantity_axis_label(quantity: str) -> str:
    """Return one concise physical quantity and canonical unit label."""
    label = _COMPACT_QUANTITY_LABELS.get(
        quantity,
        diagnostics.FIELD_LABELS[quantity],
    )
    unit = diagnostics.FIELD_UNITS[quantity]
    return f"{label} [{unit}]" if unit else label


def quantity_colorbar_label(quantity: str) -> str:
    """Return only the canonical unit because each map title owns the quantity."""
    unit = diagnostics.FIELD_UNITS[quantity]
    return f"[{unit}]" if unit else ""


def align_axis_to_references(
    figure: Figure,
    axis: Axes,
    vertical_reference: Axes,
    *,
    right_reference: Axes | None = None,
) -> None:
    """Align one axis vertically and optionally extend it to a settled right edge."""

    def align(_event: object = None) -> None:
        """Copy settled reference bounds after constrained layout resolves."""
        position = axis.get_position()
        vertical = vertical_reference.get_position()
        right = position.x1 if right_reference is None else right_reference.get_position().x1
        if right <= position.x0:
            message = "Aligned generation-input axis requires a positive width."
            raise ValueError(message)
        axis.set_position(
            (
                position.x0,
                vertical.y0,
                right - position.x0,
                vertical.height,
            )
        )

    align()
    figure.canvas.mpl_connect("draw_event", align)


def readable_case_title(
    quantity: str,
    record: diagnostics.GenerationInputDiagnostics,
) -> str:
    """Return a concise map title without source paths or technical identities."""
    return f"{diagnostics.FIELD_LABELS[quantity]} — {diagnostics.case_display_label(record)}"
