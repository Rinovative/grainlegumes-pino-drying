"""Show exact generated spatial fields for one admitted case.

Responsibilities:
  - Admit domain fields separately from boundary, scalar, and time support
  - Prepare full-resolution discrete scales once for a selected case
  - Render independently selected fields without reloading case evidence

Design principles:
  - Persisted arrays and exact tensor diagnostics determine availability
  - Missing stochastic fields are disclosed rather than regenerated
  - Constant density belongs to metadata, not a fabricated spatial map

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
from src.analysis.presentation import analysis_field_maps as field_maps

from . import generation_input_plot_layout as layout

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure

_MAXIMUM_COLUMNS = 3
_SPATIAL_DIMENSIONS = 2
_LOG_FIELDS = frozenset({"Kxx", "Kyy", "K_min", "K_max"})


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
        fields[name] = values
        labels[name] = diagnostics.FIELD_LABELS[name]
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

    scales = {}
    for name, values in fields.items():
        semantic = field_maps.FieldMapSemantic.LINEAR_POSITIVE
        if name in _LOG_FIELDS:
            semantic = field_maps.FieldMapSemantic.LOG_POSITIVE
        elif name in {"Kxy", "K_orientation"}:
            semantic = field_maps.FieldMapSemantic.LINEAR_SIGNED
        if name == "K_orientation":
            # Axial angles wrap after 180 degrees; the two endpoint colors must
            # meet. Constant orientations retain the truthful one-value scale.
            boundaries = None if np.all(values == values.flat[0]) else np.linspace(-90.0, 90.0, field_maps.N_COLOR_BINS + 1)
            scales[name] = field_maps.build_field_map(
                values,
                semantic=semantic,
                field=name,
                unit=units[name],
                display_transform="principal_axis_modulo_180_degrees",
                comparison_scope="axial_direction",
                locked_boundaries=boundaries,
                colormap_name="twilight",
            )
            continue
        scales[name] = field_maps.build_field_map(
            values,
            semantic=semantic,
            field=name,
            unit=units[name],
            comparison_scope=f"case:{record.case.payload['case_input_id']}",
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
    figure, grid = layout.figure_grid(columns=columns, row_heights=(layout.MAP_LAYOUT.map_row_height,) * rows)
    for index, name in enumerate(ordered):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        scale = atlas.scales[name]
        indices = display_grid.display_indices(atlas.fields[name].shape, display_spatial_stride)
        image = axis.pcolormesh(
            display_grid.display_sample(atlas.record.fields["x"], indices),
            display_grid.display_sample(atlas.record.fields["y"], indices),
            display_grid.display_sample(atlas.fields[name], indices),
            shading="auto",
            **scale.pcolormesh_kwargs(),
        )
        axis.set_aspect("equal")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(atlas.labels[name])
        colorbar = figure.colorbar(image, ax=axis, **scale.colorbar_kwargs())
        colorbar.set_label(atlas.units[name])
    figure.suptitle(f"Generated fields — case {atlas.record.case.case_index}")
    return figure
