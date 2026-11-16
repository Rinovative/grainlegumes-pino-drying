"""Show exact generated spatial fields for one admitted case.

Responsibilities:
  - Admit domain fields separately from boundary, scalar, and time support
  - Prepare full-resolution discrete scales once for a selected case
  - Render independently selected fields without reloading case evidence, each as a header-labelled map with its
    unit-bearing colour bar on the shared fixed-inch map grid, at most three per row, in the order of a 2 x 3 grid

Design principles:
  - Persisted arrays and exact tensor diagnostics determine availability
  - Missing stochastic fields are disclosed rather than regenerated
  - Constant density belongs to metadata, not a fabricated spatial map
  - Fields show on their shared display scale (equilibrium relative humidity in %) under the active label style

This module does NOT:
  - Discover files, run Generation, normalize model inputs, or compute errors
  - Turn inlet strips or schedules into domain-wide fields
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as shared_layout

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure

_MAXIMUM_COLUMNS = 3
_SPATIAL_DIMENSIONS = 2
_ATLAS_ORDER = ("Kxx", "Kxy", "eps_bed", "Kyy", "K_orientation", "X_0_db_field")  # fills a 2 x 3 grid; other fields follow in their natural order
# Half of arctan2 lies in [-90, 90] degrees, both ends included and naming the same axial direction: nine 20-degree bands.
_ORIENTATION_BOUNDARIES = np.linspace(-90.0, 90.0, 10)


@dataclass(frozen=True, slots=True)
class GeneratedFieldAtlas:
    """Bind one admitted case to immutable arrays and reusable full-grid scales."""

    record: diagnostics.GenerationInputDiagnostics
    fields: Mapping[str, np.ndarray]
    labels: Mapping[str, str]
    units: Mapping[str, str]
    scales: Mapping[str, field_maps.FieldMapDefinition]
    notes: tuple[str, ...]


def prepare_field_atlas(record: diagnostics.GenerationInputDiagnostics) -> GeneratedFieldAtlas:
    """Admit true domain fields in deterministic order without any source reload."""
    fields: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    units: dict[str, str] = {}
    notes = [
        "Pressure uses its inlet profile; temperature and humidity use their schedules.",
        "Backbone z and background z_bg are unavailable: their exact arrays were not persisted.",
    ]
    shape = record.fields["x"].shape
    for name in diagnostics.display_field_names(record):
        if name == "p_in_bc":
            continue
        values = record.fields[name]
        if values.ndim != _SPATIAL_DIMENSIONS or values.shape != shape or not np.isfinite(values).all():
            message = f"Generated domain field {name!r} must match the finite coordinate grid."
            raise ValueError(message)
        if name == "rho_bu_dry" and np.all(values == values.flat[0]):
            notes.append(f"Constant dry bulk density: {float(values.flat[0]):g} kg/m³ (metadata).")
            continue
        fields[name] = layout.display_field(record, name)
        labels[name] = layout.quantity_label(name)
        units[name] = diagnostics.FIELD_UNITS[name]

    # The major eigenvector direction is defined modulo 180 degrees. An exactly
    # isotropic tensor has no physical principal direction, so do not invent one.
    diagonal_difference = record.fields["Kxx"] - record.fields["Kyy"]
    cross = record.fields["Kxy"]
    if np.all((diagonal_difference != 0.0) | (cross != 0.0)):
        orientation = np.rad2deg(0.5 * np.arctan2(2.0 * cross, diagonal_difference))
        orientation.setflags(write=False)
        fields["K_orientation"] = orientation
        labels["K_orientation"] = "Major permeability direction (modulo 180°)"
        units["K_orientation"] = "degree"
    else:
        notes.append("Principal-direction map unavailable: isotropic cells have no unique direction.")

    fields = dict(sorted(fields.items(), key=lambda item: _ATLAS_ORDER.index(item[0]) if item[0] in _ATLAS_ORDER else len(_ATLAS_ORDER)))
    scales = {name: layout.physical_norm(name, values) for name, values in fields.items() if name != "K_orientation"}
    if "K_orientation" in fields:
        # Axial angles wrap at +-90 degrees, so the cyclic colormap's two ends meet there. A constant orientation keeps the
        # truthful one-value scale.
        orientation = fields["K_orientation"]
        scales["K_orientation"] = field_maps.build_field_map(
            orientation,
            semantic=field_maps.FieldMapSemantic.LINEAR_SIGNED,
            field="K_orientation",
            unit=units["K_orientation"],
            display_transform="principal_axis_modulo_180_degrees",
            comparison_scope="axial_direction",
            locked_boundaries=None if np.all(orientation == orientation.flat[0]) else _ORIENTATION_BOUNDARIES,
            colormap_name="twilight",
        )
    return GeneratedFieldAtlas(
        record,
        MappingProxyType(fields),
        MappingProxyType(labels),
        MappingProxyType(units),
        MappingProxyType(scales),
        tuple(notes),
    )


def field_atlas(atlas: GeneratedFieldAtlas, selected: Sequence[str], *, display_spatial_stride: display_grid.DisplayStride = "auto") -> Figure | None:
    """Render selected fields using unchanged full-grid scientific scale objects."""
    if len(set(selected)) != len(selected) or any(name not in atlas.fields for name in selected):
        message = "Atlas selection must contain unique admitted domain fields."
        raise ValueError(message)
    if not selected:
        return None
    ordered = tuple(name for name in atlas.fields if name in selected)
    columns = min(_MAXIMUM_COLUMNS, len(ordered))
    rows = (len(ordered) + columns - 1) // columns
    extent = display_grid.physical_extent(atlas.record.fields["x"], atlas.record.fields["y"])
    geometry = shared_layout.plan_geometry(rows, columns, (extent[1] - extent[0], extent[3] - extent[2]), row_labels=False)
    figure, axes, slots = shared_layout.map_grid(geometry, extent=extent)
    for index, name in enumerate(ordered):
        row, column = divmod(index, columns)
        image = layout.draw_field(
            axes[row, column], atlas.record, atlas.fields[name], atlas.scales[name], display_spatial_stride=display_spatial_stride
        )
        shared_layout.add_field_map_colorbar(
            figure, image, slots[row, column], atlas.scales[name], label=f"[{field_labels.display_unit(atlas.units[name], field=name)}]"
        )
        shared_layout.set_column_headers(axes[row : row + 1, column : column + 1], [atlas.labels[name]], [])
    for index in range(len(ordered), rows * columns):  # cells the last row leaves empty: the maps above them close the grid
        row, column = divmod(index, columns)
        axes[row, column].set_visible(False)
        axes[row - 1, column].tick_params(labelbottom=True)
        axes[row - 1, column].set_xlabel("x [m]", fontsize=figure_layout.LABEL_PT)
    shared_layout.set_map_title(figure, f"Generated fields — case {atlas.record.case.case_index}")
    return figure
