"""Render sequence-aware transient-drying Evaluation evidence.

Responsibilities:
  - Compare reference, prediction, and signed-error fields at exact physical times
  - Draw state maps on the fixed map-grid geometry: state scales from the 5th to the 95th percentile, per drawn time or,
    per field, pooled over all stored times (``fixed_time_fields``; ``default_fixed_time_fields`` names all but moisture), with arrows
    for the cells beyond (on the state bars fixed over the scaled times), error scales (signed by default) fixed over the
    scaled times, tick decimals fixed per colour bar over the scaled times, one scale note under the title and a mean |e|
    line above each error map
  - Plot physical state trajectories and exact grain-moisture derivations
  - Plot central rollout error across time, horizons, and reduction scopes
  - Plot per-field relative norms over physical time with their supported case counts
  - Present target, matched-compute, pipeline-degradation, and timing evidence

Design principles:
  - Dynamic states follow the shared EDA labels, units, and display conversions in the active label style: phi in
    percent with errors in pp, and the water densities w_surf, w_int and w_gr as surface, internal and grain moisture in
    percent of wet basis with errors in pp, converted per case with its dry bulk density; cells with negative or
    non-finite water are masked
  - Normalized errors stay on the stored states and saved train scales
  - Physical times are selected exactly and never nearest-neighbour substituted
  - Endpoint and cumulative reductions remain visually and semantically distinct

This module does NOT:
  - Load artifacts, run inference, refit scaling, or calculate speedup formulas
  - Alter steady Evaluation plots or silently fill unavailable evidence
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize

from src import datasets, domain, learning
from src.analysis.evaluation import evaluation_transient_artifact as artifact
from src.analysis.evaluation import evaluation_transient_comparison as comparison
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as plot_layout
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_moisture_display as moisture_display
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_plot_layout as layout
from src.analysis.ui import analysis_ui_time as time_axis

if TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

_STATE_ARRAY_RANK = 4
_STATE_MAP_COLUMNS = 3  # reference, prediction, error
_SPATIAL_MAP_ERROR_COLUMN = 2
_ERROR_HEADERS = {
    error_diagnostics.SIGNED_ERROR: "Signed error",
    "Absolute error": "Absolute error",
    error_diagnostics.NORMALIZED_ABSOLUTE_ERROR: error_diagnostics.NORMALIZED_ERROR_HEADER,
}
_BEYOND_SCALE_COLOUR = "black"  # owner 14.11.: errors beyond the scale in a colour of their own
# A scale this narrow (relative to its value, at least 1) is the padded range of a constant field, see ``layout.linear_norm``.
_CONSTANT_SPAN = 4.0e-12
_MINIMUM_TRAJECTORY_TIMES = 2
_TARGET_MAGNITUDE_BIN_COUNT = 12
_TARGET_MAGNITUDE_POINT_LIMIT = 50_000
_GRAIN_MOISTURE_FIELD = "w_gr"
_MAX_ERROR_BATCH_INPUT_BYTES = 64 * 1024 * 1024
_ERROR_PAIR_MASK_BYTES = 17
_MIN_BAND_CASES = 2


def _value(record: Any, name: str) -> Any:
    """Read one field from a validated sequence object or mapping."""
    if isinstance(record, Mapping):
        if name not in record:
            msg = f"Transient plot record lacks {name!r}."
            raise ValueError(msg)
        return record[name]
    if not hasattr(record, name):
        msg = f"Transient plot record lacks {name!r}."
        raise ValueError(msg)
    return getattr(record, name)


def _material_label(value: Any) -> str:
    """Return one shared human material label from authoritative identity evidence."""
    if not isinstance(value, str) or not value:
        msg = "Transient plot material_family must be non-empty text."
        raise ValueError(msg)
    return display_labels.material_display_label(value)


def _record_material(record: Any) -> str:
    """Return one record's authoritative material identity."""
    identity = _value(record, "identity")
    material = identity.get("material_family") if isinstance(identity, Mapping) else None
    if not isinstance(material, str) or not material:
        msg = "Transient plot record identity requires material_family."
        raise ValueError(msg)
    return material


def _material_role_label(
    frame_name: Any,
    material_family: Any,
    dataset_role: Any,
    *,
    include_frame: bool,
) -> str:
    """Return material-first role context with model identity only when needed."""
    material = _material_label(material_family)
    label = (
        display_labels.material_role_display_label(material_family, dataset_role)
        if isinstance(dataset_role, str) and dataset_role in {"id", "material_ood", "parameter_ood"}
        else material
    )
    return f"{label} · {frame_name}" if include_frame else label


def _selected_fields(fields: Sequence[str] | None) -> tuple[str, ...]:
    """Return unique compatible stored and derived states in canonical order."""
    canonical = (*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD)
    requested = set(artifact.STATE_ORDER if fields is None else fields)
    if not requested or requested.difference(canonical):
        msg = "Transient state selection must contain known stored or derived states."
        raise ValueError(msg)
    return tuple(field for field in canonical if field in requested)


def _time_index(record: Any, physical_time: float | None) -> tuple[int, float]:
    """Resolve only an exact stored physical time."""
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
        msg = "Transient plot times must be finite and strictly increasing."
        raise ValueError(msg)
    selected = float(times[-1]) if physical_time is None else float(physical_time)
    matches = np.flatnonzero(times == selected)
    if matches.size != 1:
        msg = f"Requested physical time {selected:g} is unavailable; valid times are {times.tolist()}."
        raise ValueError(msg)
    return int(matches[0]), selected


def _state_unit(field: str) -> str:
    """Return the exact physical unit for one persisted or derived state."""
    if field == _GRAIN_MOISTURE_FIELD:
        surface_unit = artifact.STATE_UNITS[artifact.STATE_ORDER.index("w_surf")]
        internal_unit = artifact.STATE_UNITS[artifact.STATE_ORDER.index("w_int")]
        if surface_unit != internal_unit:
            msg = "Surface and internal moisture units must agree for grain moisture."
            raise ValueError(msg)
        return surface_unit
    try:
        return artifact.STATE_UNITS[artifact.STATE_ORDER.index(field)]
    except ValueError as error:
        msg = f"Unknown transient state field {field!r}."
        raise ValueError(msg) from error


def _display_state_values(
    field: str,
    values: np.ndarray,
    *,
    error: bool,
) -> np.ndarray:
    """Apply the shared EDA display conversion without changing artifact arrays."""
    return field_labels.display_values(
        values,
        _state_unit(field),
        field=field,
        quantity_kind="difference" if error else "absolute",
    )


def _absolute_display(record: Any, field: str, raw: np.ndarray) -> np.ndarray:
    """Return displayed absolute values: wet-basis % with masked unphysical cells for water densities, else the EDA conversion."""
    if moisture_display.is_moisture_field(field):
        return moisture_display.wet_basis_percent(raw, _dry_bulk_density(record))
    return _display_state_values(field, raw, error=False)


def _difference_display(field: str, values: np.ndarray) -> np.ndarray:
    """Return displayed differences; wet-basis differences are already in percentage points."""
    return values if moisture_display.is_moisture_field(field) else _display_state_values(field, values, error=True)


def _display_unit(field: str, *, difference: bool) -> str:
    """Return the displayed unit of one state's values or differences."""
    return moisture_display.field_display_unit(field, _state_unit(field), difference=difference)


def _display_symbol(field: str) -> str:
    """Return the label of one displayed state in the active style (water as wet-basis moisture)."""
    return moisture_display.display_name(field, mathtext=True)


def _display_label(field: str) -> str:
    """Return the label of one displayed state in the active style with its absolute unit."""
    return f"{_display_symbol(field)} [{_display_unit(field, difference=False)}]"


def _spatial_means(record: Any, field: str, reference: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return displayed per-time spatial means [time] of reference and prediction [time,Y,X].

    Wet-basis moisture is the bulk value 100 Σw/(Σρ_bu,dry + Σw) over the cells valid in both at that time, so the two
    lines share one cell set; other states average all cells.
    """
    if not moisture_display.is_moisture_field(field):
        with np.errstate(invalid="ignore", over="ignore"):
            return (
                np.mean(_display_state_values(field, reference, error=False), axis=(-2, -1)),
                np.mean(_display_state_values(field, prediction, error=False), axis=(-2, -1)),
            )
    density = _dry_bulk_density(record)
    valid = np.isfinite(_absolute_display(record, field, reference)) & np.isfinite(_absolute_display(record, field, prediction))
    reference_means, prediction_means = (
        np.asarray([moisture_display.bulk_wet_basis_percent(water, density, cells) for water, cells in zip(values, valid, strict=True)])
        for values in (reference, prediction)
    )
    return reference_means, prediction_means


def _grain_moisture(record: Any, states: np.ndarray) -> np.ndarray:
    """Derive grain moisture from persisted states and material evidence, unphysical or non-finite water passed through as stored."""
    if states.ndim != _STATE_ARRAY_RANK or states.shape[1] != len(artifact.STATE_ORDER):
        msg = "Grain-moisture trajectories require [time,4,Y,X] state arrays."
        raise ValueError(msg)
    scalars = np.asarray(_value(record, "scalar_conditioning"), dtype=np.float64)
    if scalars.shape != (len(artifact.SCALAR_ORDER),):
        msg = "Grain-moisture trajectories require exact scalar conditioning."
        raise ValueError(msg)
    fraction = scalars[artifact.SCALAR_ORDER.index("f_surf")]
    return domain.moisture.granular_water_content(
        states[:, artifact.STATE_ORDER.index("w_surf")],
        states[:, artifact.STATE_ORDER.index("w_int")],
        fraction,
        require_physical_states=False,
    )


def _field_snapshots(record: Any, field: str, reference: np.ndarray, prediction: np.ndarray, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return raw reference and prediction values [time,Y,X] of one field at the given time indices."""
    selected = list(indices)
    if field == _GRAIN_MOISTURE_FIELD:
        return _grain_moisture(record, reference[selected]), _grain_moisture(record, prediction[selected])
    channel = artifact.STATE_ORDER.index(field)
    return reference[selected, channel], prediction[selected, channel]


def _state_arrays(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the validated reference and prediction state arrays of one record."""
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK or reference.shape[1] != len(artifact.STATE_ORDER):
        msg = "Transient state maps require matching [time,4,Y,X] arrays."
        raise ValueError(msg)
    return reference, prediction


def _selected_error(
    record: Any,
    field: str,
    mode: str,
    raw_reference: np.ndarray,
    raw_prediction: np.ndarray,
    scale: error_diagnostics.ReferenceScale | str | None,
) -> error_diagnostics.SelectedErrorMap:
    """
    Compute one displayed error map over the full grid, with diverging prediction cells masked.

    A signed or absolute water-density error is the difference of the wet-basis fields in pp, masked where either is
    unphysical; the normalized error stays on the stored states, so it needs no ``record``.
    """
    reference, prediction, unit = raw_reference, raw_prediction, _state_unit(field)
    finite = np.isfinite(raw_prediction)
    if moisture_display.is_moisture_field(field) and mode != error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        reference, prediction = _absolute_display(record, field, raw_reference), _absolute_display(record, field, raw_prediction)
        unit = moisture_display.ABSOLUTE_UNIT
        finite = np.isfinite(reference) & np.isfinite(prediction)
    # A fully diverging field keeps the whole grid, so the shared map reports it as non-finite.
    support = finite if finite.any() else np.ones(raw_prediction.shape, dtype=bool)
    return error_diagnostics.selected_error_map(
        reference,
        prediction,
        mode=mode,  # type: ignore[arg-type]
        field=field,
        physical_unit=unit,
        scale=scale,
        support=support,
    )


def _error_display(field: str, error_mode: str, error_result: error_diagnostics.SelectedErrorMap) -> np.ndarray | None:
    """Return the displayed error values of one selected error map."""
    if error_result.map_values is None or error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR:
        return error_result.map_values
    return _difference_display(field, error_result.map_values)


def _error_definition(field: str, error_mode: str, finite_errors: np.ndarray) -> field_maps.FieldMapDefinition:
    """Build one field's error colour scale from its finite displayed errors; errors beyond it set the bar's arrows."""
    signed = error_mode == error_diagnostics.SIGNED_ERROR
    normalized = error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    boundaries = plot_layout.error_boundaries(finite_errors, signed=signed)
    return field_maps.build_field_map(
        # The scale's own ends keep a uniform error (all zero included) from being a constant field, whose map refuses locked boundaries.
        np.append(finite_errors, boundaries[[0, -1]]),
        semantic=field_maps.FieldMapSemantic.SIGNED_ERROR if signed else field_maps.FieldMapSemantic.ABSOLUTE_ERROR,
        field=field,
        unit="1" if normalized else _display_unit(field, difference=True),
        display_transform=error_mode,
        locked_boundaries=boundaries,
        bound_policy="rounded 5th to 95th percentile, symmetric" if signed else "0 to the rounded 95th percentile",
    )


@dataclass(frozen=True, slots=True)
class StateMapNorm:
    """
    One field's error colour scale, state tick format and state-bar arrows, fixed over the scaled times; no error scale without finite errors.

    ``arrows`` are the (reference, prediction) bar's arrows: an end is drawn when any scaled time has values beyond it.
    ``fixed`` is the (reference, prediction) state scale pooled over all times, set only for a field of ``fixed_time_fields``.
    """

    error: field_maps.FieldMapDefinition | None
    state_format: plot_layout.TickFormat
    arrows: tuple[field_maps.FieldMapExtend, field_maps.FieldMapExtend]
    fixed: tuple[Normalize | None, Normalize | None] | None = None


def _is_constant(norm: Normalize) -> bool:
    """Return whether ``norm`` is the padded scale of a constant field."""
    lower, upper = np.asarray((norm.vmin, norm.vmax), dtype=np.float64)
    return bool(upper - lower <= _CONSTANT_SPAN * max(abs(lower), 1.0))


def _state_ticks(norm: Normalize) -> np.ndarray:
    """Return the ticks of a state bar: its band boundaries; one tick at the value of a constant field."""
    if isinstance(norm, BoundaryNorm):
        return np.asarray(norm.boundaries, dtype=np.float64)
    lower, upper = np.asarray((norm.vmin, norm.vmax), dtype=np.float64)
    return np.array([(lower + upper) / 2.0])


def _fixed_fields(fields: Sequence[str], fixed_time_fields: Collection[str]) -> frozenset[str]:
    """Return ``fixed_time_fields`` as a set of drawn ``fields``; a bare string or a name that is not drawn is refused."""
    if isinstance(fixed_time_fields, str):
        msg = "fixed_time_fields must be a collection of field names, not one string."
        raise TypeError(msg)
    fixed = frozenset(fixed_time_fields)
    if undrawn := fixed.difference(fields):
        msg = f"Fixed-over-time fields must be drawn fields, not {sorted(undrawn)}."
        raise ValueError(msg)
    return fixed


def default_fixed_time_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Return the ``fields`` whose state scale is fixed over all stored times by default: all but the moisture fields, which follow each time."""
    return tuple(field for field in fields if not moisture_display.is_moisture_field(field))


def state_map_norms(
    record: Any,
    *,
    state_fields: Sequence[str] | None,
    time_indices: Sequence[int],
    error_mode: str,
    reference_scales: Mapping[str, error_diagnostics.ReferenceScale | str] | None = None,
    lock_scale: bool = True,
    fixed_time_fields: Collection[str] = (),
) -> dict[str, StateMapNorm]:
    """
    Return per-field error colour scales and state tick decimals from all given time indices.

    The state colour scales are not fixed here unless the field is in ``fixed_time_fields``: ``plot_state_maps`` derives
    them from each drawn time. Their tick decimals are the most any time's ticks need, at most three and never a power of
    ten (``plot_layout.state_format``); a near-uniform time does not set them but is widened (``lock_scale`` as in
    ``plot_state_maps``). A fixed field's state scale is the one of the 5th to 95th percentile of the displayed values
    pooled over all given times (the prediction's own without ``lock_scale``), and its decimals come from that scale.
    A name in ``fixed_time_fields`` that is not a drawn field raises ``ValueError``, a bare string ``TypeError``.
    """
    fields = _selected_fields(state_fields)
    fixed_fields = _fixed_fields(fields, fixed_time_fields)
    indices = tuple(time_indices)
    if not indices:
        msg = "State-map colour scales require at least one time index."
        raise ValueError(msg)
    if error_mode not in error_diagnostics.TRANSIENT_ERROR_MODES:
        msg = f"Transient error mode must be one of {error_diagnostics.TRANSIENT_ERROR_MODES}."
        raise ValueError(msg)
    reference, prediction = _state_arrays(record)
    norms: dict[str, StateMapNorm] = {}
    for field in fields:
        raw_reference, raw_prediction = _field_snapshots(record, field, reference, prediction, indices)
        scale = None if reference_scales is None else reference_scales.get(field)
        errors = [
            values[np.isfinite(values)]
            for position in range(len(indices))
            if (
                values := _error_display(
                    field, error_mode, _selected_error(record, field, error_mode, raw_reference[position], raw_prediction[position], scale)
                )
            )
            is not None
        ]
        finite_errors = np.concatenate(errors) if errors else np.empty(0)
        frames_reference = [_absolute_display(record, field, raw) for raw in raw_reference]
        frames_prediction = [_absolute_display(record, field, raw) for raw in raw_prediction]
        shown_reference, shown_prediction = frames_reference, frames_prediction
        pooled = field in fixed_fields
        if pooled:  # one pooled "frame" per panel kind
            shown_reference, shown_prediction = [np.concatenate(frames_reference)], [np.concatenate(frames_prediction)]
        steps = []
        for shown in shown_reference if lock_scale else (*shown_reference, *shown_prediction):
            state_range = _frame_range(shown)
            if state_range is None:  # a constant frame keeps one tick and never sets the format
                continue
            # A near-uniform frame records the finer step it needs: it never sets the format, but keeps three decimals if no frame fits.
            bands = layout.rounded_boundaries(*state_range, max_decimals=plot_layout.MAX_TICK_DECIMALS)
            bands = layout.rounded_boundaries(*state_range) if bands is None else bands
            if bands is not None:
                steps.append(float(bands[1] - bands[0]))
        state_format = plot_layout.state_format(steps)
        fixed = _state_norms(shown_reference[0], shown_prediction[0], shared=lock_scale, decimals=state_format.decimals) if pooled else None
        frame_norms = [
            fixed or _state_norms(shown, other, shared=lock_scale, decimals=state_format.decimals)
            for shown, other in zip(frames_reference, frames_prediction, strict=True)
        ]
        norms[field] = StateMapNorm(
            _error_definition(field, error_mode, finite_errors) if finite_errors.size else None,
            state_format,
            (_union_extend(frames_reference, frame_norms, 0), _union_extend(frames_prediction, frame_norms, 1)),
            fixed,
        )
    return norms


def _union_extend(
    frames: Sequence[np.ndarray], frame_norms: Sequence[tuple[Normalize | None, Normalize | None]], column: int
) -> field_maps.FieldMapExtend:
    """Return the arrows of one state bar over all frames: an end is drawn when any frame has finite values beyond that frame's scale."""
    extends = [
        field_maps.locked_extend(values[np.isfinite(values)], np.array([norm.vmin, norm.vmax]))
        for values, norms in zip(frames, frame_norms, strict=True)
        if (norm := norms[column]) is not None
    ]
    below, above = (any(extend in ends for extend in extends) for ends in (("min", "both"), ("max", "both")))
    return "both" if below and above else "min" if below else "max" if above else "neither"


def _frame_range(values: np.ndarray) -> tuple[float, float] | None:
    """Return the 5th to 95th percentile of the finite ``values``, else their full range; ``None`` without finite values or when constant."""
    scale = field_maps.percentile_scale(values, "state")
    if scale is None or scale[1] > scale[0]:
        return scale
    full = layout.linear_norm(values[np.isfinite(values)])
    return None if _is_constant(full) else (float(np.asarray(full.vmin)), float(np.asarray(full.vmax)))


def _state_norm(values: np.ndarray, decimals: int) -> Normalize | None:
    """
    Return the discrete scale over the 5th to the 95th percentile of the finite ``values`` (``None`` without any).

    The bands are the round ones of ``plot_layout.state_boundaries``, with a ``BoundaryNorm`` of one colour per band.
    A degenerate percentile range falls back to the values' own full finite range, and a constant field to a padded
    continuous scale of one tick. A near-uniform scale whose step is finer than the bar's ``decimals`` is widened to
    11 steps of 10**-``decimals``.
    """
    scale = _frame_range(values)
    if scale is None:
        finite = values[np.isfinite(values)]
        return layout.linear_norm(finite) if finite.size else None
    boundaries = plot_layout.state_boundaries(*scale)
    if plot_layout.is_finer_than_decimals(float(boundaries[1] - boundaries[0]), decimals):
        boundaries = np.linspace(*plot_layout.widened_range(*scale, decimals), field_maps.N_COLOR_BINS + 1)
    return BoundaryNorm(boundaries, boundaries.size - 1)


def _state_norms(reference: np.ndarray, prediction: np.ndarray, *, shared: bool, decimals: int) -> tuple[Normalize | None, Normalize | None]:
    """
    Return one time's reference and prediction colour scales (``None`` for a panel without finite values).

    Shared: both panels take the reference's percentile scale, so one colour means one value and the prediction never
    widens it; a reference without finite values leaves each panel its own scale. Otherwise each panel scales to its own
    5th to 95th percentile.
    """
    if shared and (together := _state_norm(reference, decimals)) is not None:
        return together, together
    return _state_norm(reference, decimals), _state_norm(prediction, decimals)


def _static_conditioning(record: Any) -> np.ndarray:
    """Return one record's exact static conditioning [7,Y,X]."""
    static = np.asarray(_value(record, "static_conditioning"), dtype=np.float64)
    if static.shape != (len(artifact.STATIC_ORDER), *np.shape(_value(record, "reference_states"))[-2:]):
        msg = "Transient state maps require exact [7,Y,X] static conditioning."
        raise ValueError(msg)
    return static


def _static_coordinates(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y grid coordinates [Y,X] of one record's exact static conditioning."""
    static = _static_conditioning(record)
    return static[0], static[1]


def _dry_bulk_density(record: Any) -> np.ndarray:
    """Return one record's dry bulk density [Y,X] from its exact static conditioning."""
    return _static_conditioning(record)[artifact.STATIC_ORDER.index("rho_bu_dry")]


def _map_extent(record: Any) -> tuple[float, float, float, float]:
    """Return the physical extent (x0, x1, y0, y1) spanned by the grid coordinates of one record."""
    x_values, y_values = _static_coordinates(record)
    return float(np.min(x_values)), float(np.max(x_values)), float(np.min(y_values)), float(np.max(y_values))


def _fixed_width_time(record: Any, selected_time: float) -> str:
    """Return ``selected_time`` with the decimals and width of the record's widest stored time, padded with figure spaces.

    Digits and figure spaces are equally wide, so a title carrying this text keeps its width over all animation frames.
    """
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    decimals = max(len(np.format_float_positional(value, precision=6, trim="-").partition(".")[2]) for value in times)
    width = max(len(f"{value:.{decimals}f}") for value in times)
    return f"{selected_time:.{decimals}f}".rjust(width, "\N{FIGURE SPACE}")


def state_map_geometry(record: Any, *, state_fields: Sequence[str] | None = None) -> layout.MapGeometry:
    """Return the fixed map-grid geometry that ``plot_state_maps`` draws for one record and state selection."""
    x0, x1, y0, y1 = _map_extent(record)
    return layout.plan_geometry(len(_selected_fields(state_fields)), _STATE_MAP_COLUMNS, (x1 - x0, y1 - y0))


def plot_state_maps(
    record: Any,
    *,
    state_fields: Sequence[str] | None = None,
    physical_time: float | None = None,
    lock_scale: bool = True,
    fixed_time_fields: Collection[str] = (),
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    reference_scales: Mapping[str, error_diagnostics.ReferenceScale | str] | None = None,
    display_spatial_stride: display_grid.DisplayStride = "auto",
    norms: Mapping[str, StateMapNorm] | None = None,
    figure: Figure | None = None,
    title_context: str | None = None,
) -> Figure:
    """
    Plot reference, prediction, and one selected full-grid error mode on the fixed map-grid geometry.

    Reference and prediction follow this time: with ``lock_scale`` both share the 5th to 95th percentile of the
    reference's finite displayed values, else each its own; a bar's arrows mark the cells beyond it. ``norms`` (from ``state_map_norms``)
    fixes the error scales, the state tick decimals and the state bars' arrows (drawn when any scaled time has values beyond that
    end) over several times; the default derives them from this time alone.
    A drawn field of ``fixed_time_fields`` has the state scale pooled over all stored times instead of this time's; any
    such field makes the default norms span all stored times, and given ``norms`` must cover every drawn field and carry
    exactly the fixed fields' pooled scales (``state_map_norms`` with the same fields), else ``ValueError``. Names in
    ``fixed_time_fields`` must be drawn fields, as in ``state_map_norms``.
    Only the error maps carry a metric, one mean |e| line above each; one note line under the title names the colour-scale
    rule. A given ``figure`` (from ``layout.map_figure`` with
    ``state_map_geometry``) is cleared and reused so animation frames keep one size and identical
    axes positions. ``title_context`` (case, model) replaces the generic figure title before the
    time, which is then written at the fixed width of the record's stored times. Panels without
    finite values state that instead of raising.
    """
    fields = _selected_fields(state_fields)
    if not isinstance(lock_scale, bool):
        msg = "lock_scale must be boolean."
        raise TypeError(msg)
    fixed_fields = _fixed_fields(fields, fixed_time_fields)
    index, selected_time = _time_index(record, physical_time)
    reference, prediction = _state_arrays(record)
    x_values, y_values = _static_coordinates(record)
    if norms is None:
        norms = state_map_norms(
            record,
            state_fields=fields,
            time_indices=range(reference.shape[0]) if fixed_fields else (index,),
            error_mode=error_mode,
            reference_scales=reference_scales,
            lock_scale=lock_scale,
            fixed_time_fields=fixed_fields,
        )
    elif missing := set(fields).difference(norms):
        msg = f"State-map colour scales lack the drawn fields {sorted(missing)}."
        raise ValueError(msg)
    elif {field for field in fields if norms[field].fixed is not None} != fixed_fields:
        msg = "State-map colour scales do not fix the requested fixed_time_fields."
        raise ValueError(msg)
    indices = display_grid.display_indices(reference.shape[-2:], display_spatial_stride)
    x_display = display_grid.display_sample(x_values, indices)
    y_display = display_grid.display_sample(y_values, indices)
    geometry = state_map_geometry(record, state_fields=fields)
    figure, axes, slots = layout.map_grid(geometry, extent=_map_extent(record), figure=figure)
    error_texts = []
    normalized_error = error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    signed_error = error_mode == error_diagnostics.SIGNED_ERROR
    for row, field in enumerate(fields):
        snapshot_reference, snapshot_prediction = _field_snapshots(record, field, reference, prediction, (index,))
        raw_reference, raw_prediction = snapshot_reference[0], snapshot_prediction[0]
        scale = None if reference_scales is None else reference_scales.get(field)
        error_result = _selected_error(record, field, error_mode, raw_reference, raw_prediction, scale)
        norm = norms[field]
        values = (
            _absolute_display(record, field, raw_reference),
            _absolute_display(record, field, raw_prediction),
            _error_display(field, error_mode, error_result),
        )
        exponent = plot_layout.error_format(None if norm.error is None else norm.error.boundaries, signed=signed_error).exponent
        error_unit = None if normalized_error else _display_unit(field, difference=True)
        error_texts.append(plot_layout.mean_error_text(values[2], unit=error_unit, exponent=exponent))
        masked_note = "Negative or non-finite water masked" if moisture_display.is_moisture_field(field) else "Non-finite cells masked"
        reference_norm, prediction_norm = (
            _state_norms(values[0], values[1], shared=lock_scale, decimals=norm.state_format.decimals) if norm.fixed is None else norm.fixed
        )
        for column, value in enumerate(values):
            axis = axes[row, column]
            is_error = column == _SPATIAL_MAP_ERROR_COLUMN
            error_scale = norm.error if is_error else None
            colormap_name = visual_semantics.field_visual_semantics(field).colormap
            colormap: Colormap | str = colormap_name
            normalizer: Normalize | None = None if is_error else reference_norm if column == 0 else prediction_norm
            if isinstance(normalizer, BoundaryNorm):  # one colour per band of this frame's scale
                colormap = colormaps[colormap_name].resampled(normalizer.boundaries.size - 1)
            if error_scale is not None:
                # Errors beyond the robust scale stand out in one colour of their own instead of the darkest scale colour.
                colormap = error_scale.colormap.with_extremes(over=_BEYOND_SCALE_COLOUR, under=_BEYOND_SCALE_COLOUR)
                normalizer = error_scale.normalizer
            image: ScalarMappable
            if value is None or not np.isfinite(value).any():
                reason = "unavailable: no finite values" if value is not None else f"Unavailable: {error_result.reason}"
                axis.text(0.5, 0.5, reason, ha="center", va="center", wrap=True, transform=axis.transAxes)
                if normalizer is None:
                    continue
                # An error scale fixed over several frames keeps its bar in a frame without finite values.
                image = ScalarMappable(norm=normalizer, cmap=colormap)
            else:
                if is_error and error_scale is None:
                    msg = "State-map colour scales do not cover this time."
                    raise ValueError(msg)
                image = axis.pcolormesh(
                    x_display,
                    y_display,
                    np.ma.masked_invalid(display_grid.display_sample(np.asarray(value), indices)),
                    shading="auto",
                    cmap=colormap,
                    norm=normalizer,
                )
                if not is_error and not np.isfinite(value).all():
                    axis.text(0.02, 0.02, masked_note, transform=axis.transAxes, fontsize=8)
            unit = f"[{_display_unit(field, difference=is_error)}]"
            slot = slots[row, column]
            if error_scale is None:
                extend: field_maps.FieldMapExtend = "neither" if normalizer is None else norm.arrows[column]
                ticks = np.empty(0) if normalizer is None else _state_ticks(normalizer)
                plot_layout.set_fixed_decimal_ticks(
                    layout.add_slot_colorbar(figure, image, slot, label=unit, extend=extend), ticks, norm.state_format
                )
                continue
            colorbar = layout.add_slot_colorbar(
                figure,
                image,
                slot,
                label=f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} [{field_labels.display_unit('1')}]" if normalized_error else unit,
                extend=error_scale.extend,
            )
            plot_layout.set_fixed_decimal_ticks(colorbar, plot_layout.error_ticks(error_scale.boundaries, signed=signed_error))
    plot_layout.set_error_headers(axes, ("Reference", "Prediction", _ERROR_HEADERS[error_mode]), error_texts)
    layout.set_row_labels(axes, [_display_symbol(field) for field in fields])
    layout.set_figure_title(
        figure,
        f"Transient state comparison at t={selected_time:g} h"
        if title_context is None
        else f"{title_context} · t = {_fixed_width_time(record, selected_time)} h",
    )
    plot_layout.add_scale_note(figure, shared=lock_scale)
    return figure


def _trajectory_values(
    record: Any,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact time, reference, and prediction values of one field on the full grid."""
    if field not in {*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD}:
        msg = f"Transient trajectory field is unsupported: {field!r}."
        raise ValueError(msg)
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if (
        reference.shape != prediction.shape
        or reference.ndim != _STATE_ARRAY_RANK
        or times.shape != (reference.shape[0],)
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0.0)
    ):
        msg = "Transient trajectories require aligned state arrays and physical times."
        raise ValueError(msg)
    if field == _GRAIN_MOISTURE_FIELD:
        reference_values = _grain_moisture(record, reference)
        prediction_values = _grain_moisture(record, prediction)
    else:
        channel = artifact.STATE_ORDER.index(field)
        reference_values = reference[:, channel]
        prediction_values = prediction[:, channel]
    return times, reference_values, prediction_values


def _union_record_times(records: Sequence[Any], *, exclude_origin: bool) -> np.ndarray:
    """Return every exact supported physical time, without filling short tails."""
    admitted = tuple(records)
    if not admitted:
        msg = "Transient trajectory summaries require at least one record."
        raise ValueError(msg)
    support: set[float] = set()
    for record in admitted:
        values = np.asarray(_value(record, "physical_times"), dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0.0):
            msg = "Transient physical times must be finite and strictly increasing."
            raise ValueError(msg)
        support.update(float(value) for value in values[1 if exclude_origin else 0 :])
    times = np.asarray(sorted(support), dtype=np.float64)
    if not times.size:
        msg = "Transient trajectory records contain no required exact physical times."
        raise ValueError(msg)
    return times


@dataclass(frozen=True, slots=True)
class _ExactTimeSummary:
    """One exact-time cohort summary with separate presence and invalid counts."""

    times: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    count: np.ndarray
    invalid_count: np.ndarray


def _exact_time_groups(rows: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, tuple[tuple[tuple[int, float], ...], ...]]:
    """Return the sorted present exact times with the ``(row, value)`` contributions at each."""
    by_time: dict[float, list[tuple[int, float]]] = {}
    for row, (times, values) in enumerate(rows):
        if times.ndim != 1 or values.shape != times.shape or not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
            msg = "Exact-time aggregate rows require aligned strictly increasing times and values."
            raise ValueError(msg)
        for time, value in zip(times, values, strict=True):
            by_time.setdefault(float(time), []).append((row, float(value)))
    if not by_time:
        msg = "Exact-time aggregate requires at least one supported physical time."
        raise ValueError(msg)
    times = np.asarray(sorted(by_time), dtype=np.float64)
    return times, tuple(tuple(by_time[float(time)]) for time in times)


def _aggregate_exact_time_values(rows: Sequence[tuple[np.ndarray, np.ndarray]]) -> _ExactTimeSummary:
    """Average only present exact times; one nonfinite contributor voids that score."""
    times, groups = _exact_time_groups(rows)
    mean, lower, upper = (np.full(times.shape, np.nan, dtype=np.float64) for _ in range(3))
    count = np.asarray([len(group) for group in groups], dtype=np.int64)
    invalid = np.asarray([sum(not math.isfinite(value) for _row, value in group) for group in groups], dtype=np.int64)
    for index, group in enumerate(groups):
        if invalid[index]:
            continue
        values = np.asarray([value for _row, value in group], dtype=np.float64)
        mean[index] = float(np.mean(values))
        if values.size >= _MIN_BAND_CASES:
            lower[index], upper[index] = np.quantile(values, (0.1, 0.9))
    return _ExactTimeSummary(times, mean, lower, upper, count, invalid)


def _aggregate_exact_time_errors(rows: Sequence[tuple[np.ndarray, np.ndarray]], *, materials: Sequence[str]) -> _ExactTimeSummary:
    """
    Reduce per-case mean squared errors at every exact time like every reported field RMSE.

    The central value is the root of :func:`learning.metrics.metrics.material_case_mean` (equal
    over a material's cases, then over the materials); the band holds the 10th and 90th
    quantiles of the per-case RMSEs. One nonfinite contributor voids that time.
    """
    times, groups = _exact_time_groups(rows)
    central, lower, upper = (np.full(times.shape, np.nan, dtype=np.float64) for _ in range(3))
    count = np.asarray([len(group) for group in groups], dtype=np.int64)
    invalid = np.asarray([sum(not math.isfinite(value) for _row, value in group) for group in groups], dtype=np.int64)
    for index, group in enumerate(groups):
        if invalid[index]:
            continue
        mean_squared = learning.metrics.metrics.material_case_mean((materials[row], value) for row, value in group)
        central[index] = math.nan if mean_squared is None else math.sqrt(mean_squared)
        if len(group) >= _MIN_BAND_CASES:
            lower[index], upper[index] = np.quantile(np.sqrt([value for _row, value in group]), (0.1, 0.9))
    return _ExactTimeSummary(times, central, lower, upper, count, invalid)


def plot_state_trajectory_summary(
    series: Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
) -> Figure:
    """Plot material-aware aggregate or single-case trajectories on exact times."""
    fields = _selected_fields(state_fields)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not label or not records for label, records in admitted.items()):
        msg = "Trajectory summaries require labelled non-empty record groups."
        raise ValueError(msg)
    all_times = np.concatenate([_union_record_times(records, exclude_origin=False) for records in admitted.values()])
    display_time = time_axis.physical_time_display(all_times, preferred_unit="auto")
    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(9.2, max(4.8, 3.8 * len(fields))),
        layout="constrained",
        squeeze=False,
    )
    colors = plt.colormaps["tab10"]
    for row, field in enumerate(fields):
        axis = axes[row, 0]
        support_axis = axis.twinx() if row == 0 else None
        for series_index, (label, records) in enumerate(admitted.items()):
            reference_rows: list[tuple[np.ndarray, np.ndarray]] = []
            prediction_rows: list[tuple[np.ndarray, np.ndarray]] = []
            for record in records:
                record_times, reference, prediction = _trajectory_values(record, field)
                reference_means, prediction_means = _spatial_means(record, field, reference, prediction)
                reference_rows.append((record_times, reference_means))
                prediction_rows.append((record_times, prediction_means))
            reference_summary = _aggregate_exact_time_values(reference_rows)
            prediction_summary = _aggregate_exact_time_values(prediction_rows)
            plotted_time = display_time.values(reference_summary.times)
            color = colors(series_index % colors.N)
            axis.plot(
                plotted_time,
                reference_summary.mean,
                color=color,
                linewidth=2.0,
                label=f"{label} · Reference",
            )
            axis.plot(
                plotted_time,
                prediction_summary.mean,
                color=color,
                linewidth=2.0,
                linestyle="--",
                label=f"{label} · Prediction",
            )
            axis.fill_between(plotted_time, reference_summary.lower, reference_summary.upper, color=color, alpha=0.08)
            axis.fill_between(plotted_time, prediction_summary.lower, prediction_summary.upper, color=color, alpha=0.08)
            if support_axis is not None:
                support_axis.step(plotted_time, reference_summary.count, where="mid", color=color, alpha=0.55, label=f"{label} n(t)")
                unavailable = (reference_summary.invalid_count + prediction_summary.invalid_count) > 0
                if unavailable.any():
                    support_axis.scatter(plotted_time[unavailable], reference_summary.count[unavailable], color=color, marker="x", s=28)
        display_time.configure(axis)
        axis.set_ylabel(_display_label(field))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
        if support_axis is not None:
            support_axis.set_ylabel("Exact cases n(t); x marker = non-finite unavailable")
            support_axis.set_ylim(bottom=0)
            support_axis.legend(loc="lower right", frameon=False, fontsize="small")
    figure.suptitle("Reference vs prediction trajectories — exact-time spatial means; bands require n(t) ≥ 2")
    return figure


def _normalized_squared_error_trajectory(
    record: Any,
    *,
    scaling_state: Mapping[str, Any],
    fields: Sequence[str],
) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    """Return the saved-scale per-channel mean squared error over the grid at exact post-origin times."""
    selected_fields = tuple(fields)
    if not selected_fields or set(selected_fields).difference(artifact.STATE_ORDER):
        msg = "Normalized error trajectories require stored transient states."
        raise ValueError(msg)
    scaling = datasets.preprocessing.normalization.TrainScaling.from_state_dict(scaling_state)
    reference = torch.as_tensor(np.asarray(_value(record, "reference_states"), dtype=np.float32))
    prediction = torch.as_tensor(np.asarray(_value(record, "predicted_states"), dtype=np.float32))
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK:
        msg = "Normalized error trajectories require aligned state arrays."
        raise ValueError(msg)
    normalized_error = (scaling.encode_state(prediction) - scaling.encode_state(reference)).detach().cpu().numpy()
    result = {
        field: np.mean(np.square(normalized_error[1:, artifact.STATE_ORDER.index(field)], dtype=np.float64), axis=(1, 2)) for field in selected_fields
    }
    return times[1:], result


def plot_error_over_physical_time(
    series: Mapping[str, Sequence[Any]],
    *,
    scaling_states: Mapping[str, Mapping[str, Any]],
    state_fields: Sequence[str],
) -> Figure:
    """
    Plot saved-scale channel errors over exact physical time by context.

    The line is the normalized RMSE of the cases present at each time, reduced like every
    reported field RMSE (root after the material-case mean of the per-case mean squared
    error); the band spans the 10th to 90th quantile of the per-case RMSEs.
    """
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not fields or not admitted or set(admitted) != set(scaling_states):
        msg = "Error-over-time series require stored fields and matching scaling contexts."
        raise ValueError(msg)
    all_times = np.concatenate([_union_record_times(records, exclude_origin=True) for records in admitted.values()])
    display_time = time_axis.physical_time_display(all_times, preferred_unit="auto")
    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(9.2, max(4.8, 3.5 * len(fields))),
        layout="constrained",
        squeeze=False,
    )
    colors = plt.colormaps["tab10"]
    for row, field in enumerate(fields):
        axis = axes[row, 0]
        support_axis = axis.twinx() if row == 0 else None
        for series_index, (label, records) in enumerate(admitted.items()):
            rows: list[tuple[np.ndarray, np.ndarray]] = []
            materials: list[str] = []
            for record in records:
                record_times, values = _normalized_squared_error_trajectory(
                    record,
                    scaling_state=scaling_states[label],
                    fields=fields,
                )
                rows.append((record_times, values[field]))
                materials.append(_record_material(record))
            summary = _aggregate_exact_time_errors(rows, materials=materials)
            plotted_time = display_time.values(summary.times)
            color = colors(series_index % colors.N)
            axis.plot(plotted_time, summary.mean, color=color, linewidth=2.0, label=f"{label} · material-case RMSE")
            axis.fill_between(plotted_time, summary.lower, summary.upper, color=color, alpha=0.1, label=f"{label} · per-case RMSE 10th-90th")
            if support_axis is not None:
                support_axis.step(plotted_time, summary.count, where="mid", color=color, alpha=0.55, label=f"{label} n(t)")
                unavailable = summary.invalid_count > 0
                if unavailable.any():
                    support_axis.scatter(plotted_time[unavailable], summary.count[unavailable], color=color, marker="x", s=28)
        display_time.configure(axis)
        axis.set_ylabel(f"{field_labels.metric_label('nRMSE', field, mathtext=True)} [{field_labels.display_unit('1')}]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
        if support_axis is not None:
            support_axis.set_ylabel("Exact cases n(t); x marker = non-finite unavailable")
            support_axis.set_ylim(bottom=0)
            support_axis.legend(loc="lower right", frameon=False, fontsize="small")
    figure.suptitle(
        "Error over physical time — root of the material-case mean squared error at each exact time; per-case RMSE 10th-90th bands require n(t) ≥ 2"
    )
    return figure


def plot_central_error_vs_time(
    record: Any,
    *,
    scaling_state: Mapping[str, Any],
) -> Figure:
    """Plot central and per-channel normalized errors against physical time."""
    scaling = datasets.preprocessing.normalization.TrainScaling.from_state_dict(scaling_state)
    reference = torch.as_tensor(
        np.asarray(_value(record, "reference_states"), dtype=np.float32),
        dtype=torch.float32,
    )
    prediction = torch.as_tensor(
        np.asarray(_value(record, "predicted_states"), dtype=np.float32),
        dtype=torch.float32,
    )
    if reference.shape != prediction.shape or reference.ndim != _STATE_ARRAY_RANK:
        msg = "Transient error curves require aligned state arrays."
        raise ValueError(msg)
    normalized_error = (scaling.encode_state(prediction) - scaling.encode_state(reference)).detach().cpu().numpy()
    channel_rmse = np.sqrt(np.mean(np.square(normalized_error[1:], dtype=np.float64), axis=(2, 3)))
    weights = np.asarray((1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0))
    central = channel_rmse @ weights
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)[1:]
    display_time = time_axis.physical_time_display(times, preferred_unit="auto")
    plotted_time = display_time.values(times)
    figure, axis = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    axis.plot(
        plotted_time,
        central,
        marker="o",
        linewidth=2.2,
        label="Central Drying metric",
    )
    for channel, field in enumerate(artifact.STATE_ORDER):
        axis.plot(
            plotted_time,
            channel_rmse[:, channel],
            alpha=0.75,
            label=field_labels.field_label(field, mathtext=True),
        )
    display_time.configure(axis)
    axis.set_ylabel("Normalized RMSE [-]")
    figure.legend(loc="outside right upper", frameon=False)
    axis.grid(axis="y", alpha=0.25)
    return figure


def plot_relative_norms_over_time(curves: Mapping[str, pd.DataFrame]) -> Figure:
    """
    Plot the per-field relative L2 and H1 of complete rollouts over physical time with their supported case counts.

    ``curves`` maps a label to :meth:`TransientEvaluationSession.relative_norm_curve`. A gap is
    a time without a supported case in some material; the step line counts the supported cases.
    """
    if not curves or any(frame.empty for frame in curves.values()):
        msg = "Relative-norm curves require at least one labelled non-empty curve."
        raise ValueError(msg)
    all_times = np.concatenate([frame["physical_time"].to_numpy(dtype=np.float64) for frame in curves.values()])
    display_time = time_axis.physical_time_display(all_times, preferred_unit="auto")
    figure, axes = plt.subplots(2, len(artifact.STATE_ORDER), figsize=(5.0 * len(artifact.STATE_ORDER), 7.6), layout="constrained", squeeze=False)
    colors = plt.colormaps["tab10"]
    for row, norm in enumerate(("l2", "h1")):
        for column, field in enumerate(artifact.STATE_ORDER):
            axis = axes[row, column]
            support_axis = axis.twinx()
            for index, (label, frame) in enumerate(curves.items()):
                selected = frame.loc[frame["metric_id"] == f"physical_relative_{norm}_{field}"].sort_values("physical_time")
                plotted_time = display_time.values(selected["physical_time"].to_numpy(dtype=np.float64))
                axis.plot(plotted_time, selected["value"].to_numpy(dtype=np.float64), color=colors(index % colors.N), linewidth=2.0, label=label)
                support_axis.step(plotted_time, selected["supported_cases"], where="mid", color=colors(index % colors.N), alpha=0.4)
            display_time.configure(axis)
            axis.set_title(field_labels.metric_label(f"rel. {norm.upper()}", field, mathtext=True))
            axis.grid(alpha=0.25)
            support_axis.set_ylim(bottom=0)
            if column == len(artifact.STATE_ORDER) - 1:
                support_axis.set_ylabel("Supported cases n(t)")
        axes[row, 0].set_ylabel(f"Relative {norm.upper()} [-]")
    axes[0, 0].legend(frameon=False, fontsize="small")
    figure.suptitle("Relative norms over physical time: per-case ratios, equal over cases and materials; gaps are unavailable")
    return figure


def _aggregate_snapshot_evidence(  # noqa: PLR0915 -- one bounded physical snapshot reduction
    records: Sequence[Any],
    *,
    state_field: str,
    physical_time: float,
    compute_device: str = "cpu",
    device_evidence: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool exact-time case evidence without stacking complete trajectories.

    Returns the grid coordinates, the pooled mean absolute and signed-error standard deviation maps in displayed
    difference units, and the displayed case spatial means (see ``_spatial_means``) of reference and prediction.
    """
    if state_field not in {*artifact.STATE_ORDER, _GRAIN_MOISTURE_FIELD}:
        msg = f"Aggregate transient snapshot field is unsupported: {state_field!r}."
        raise ValueError(msg)
    admitted = tuple(records)
    if not admitted:
        msg = "Aggregate transient snapshots require at least one complete-rollout record."
        raise ValueError(msg)
    error_sum: np.ndarray | None = None
    absolute_error_sum: np.ndarray | None = None
    error_square_sum: np.ndarray | None = None
    counts: np.ndarray | None = None
    divergent: np.ndarray | None = None
    x_values: np.ndarray | None = None
    y_values: np.ndarray | None = None
    reference_means: list[float] = []
    prediction_means: list[float] = []
    cuda_reference: list[np.ndarray] = []
    cuda_prediction: list[np.ndarray] = []
    cuda_masks: list[np.ndarray] = []
    selected_device: compute.AnalysisDevice | None = None
    batch_cases = 1
    routes: set[str] = set()

    def flush_cuda_batch() -> None:
        nonlocal error_sum, absolute_error_sum, error_square_sum, counts, selected_device
        if not cuda_reference or selected_device is None:
            return
        if error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None:
            msg = "Aggregate transient snapshot accumulators were not initialized."
            raise RuntimeError(msg)
        reduced = compute.reduce_error_batch(
            np.stack(cuda_reference),
            np.stack(cuda_prediction),
            np.stack(cuda_masks),
            device=selected_device,
        )
        error_sum += reduced.signed_sum
        absolute_error_sum += reduced.absolute_sum
        error_square_sum += reduced.squared_sum
        counts += reduced.support_count
        routes.add(reduced.device.label)
        selected_device = reduced.device
        cuda_reference.clear()
        cuda_prediction.clear()
        cuda_masks.clear()

    for record in admitted:
        if _value(record, "mode") != "autonomous_full" or _value(record, "requested_horizon") != "full":
            msg = "Aggregate transient snapshots require complete autonomous rollouts only."
            raise ValueError(msg)
        time_index, _selected_time = _time_index(record, physical_time)
        reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
        prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
        static = np.asarray(_value(record, "static_conditioning"), dtype=np.float64)
        if (
            reference.shape != prediction.shape
            or reference.ndim != _STATE_ARRAY_RANK
            or reference.shape[1] != len(artifact.STATE_ORDER)
            or static.shape != (len(artifact.STATIC_ORDER), *reference.shape[-2:])
        ):
            msg = "Aggregate transient snapshots require aligned state and coordinate arrays."
            raise ValueError(msg)
        # Every case is evaluated on its full grid.
        mask = np.ones(reference.shape[-2:], dtype=bool)
        current_x = static[0]
        current_y = static[1]
        if x_values is None:
            x_values = current_x
            y_values = current_y
            error_sum = np.zeros(mask.shape, dtype=np.float64)
            absolute_error_sum = np.zeros(mask.shape, dtype=np.float64)
            error_square_sum = np.zeros(mask.shape, dtype=np.float64)
            counts = np.zeros(mask.shape, dtype=np.int64)
            divergent = np.zeros(mask.shape, dtype=bool)
            batch_cases = max(1, _MAX_ERROR_BATCH_INPUT_BYTES // (mask.size * _ERROR_PAIR_MASK_BYTES))
            selected_device = compute.resolve_analysis_device(compute_device)
        elif y_values is None or not np.array_equal(current_x, x_values) or not np.array_equal(current_y, y_values):
            msg = "Aggregate transient spatial maps require one exact shared coordinate grid."
            raise ValueError(msg)
        snapshots = _field_snapshots(record, state_field, reference, prediction, (time_index,))
        reference_snapshot, prediction_snapshot = snapshots[0][0], snapshots[1][0]
        if not np.isfinite(reference_snapshot[mask]).all():
            msg = "Aggregate transient snapshots require finite reference values on every grid."
            raise ValueError(msg)
        if error_sum is None or absolute_error_sum is None or error_square_sum is None or counts is None or divergent is None:
            msg = "Aggregate transient snapshot accumulators were not initialized."
            raise RuntimeError(msg)
        reference_mean, prediction_mean = _spatial_means(record, state_field, *snapshots)
        reference_means.append(float(reference_mean[0]))
        prediction_means.append(float(prediction_mean[0]))
        # Each case converts to its displayed values before pooling (water with its own dry bulk density), so the pooled
        # errors carry the displayed difference unit, such as pp for phi and water.
        reference_snapshot = _absolute_display(record, state_field, reference_snapshot)
        prediction_snapshot = _absolute_display(record, state_field, prediction_snapshot)
        # A diverging or unphysical cell gets no pooled error; the remaining cells keep their evidence.
        finite = mask & np.isfinite(reference_snapshot) & np.isfinite(prediction_snapshot)
        divergent |= mask & ~finite
        if selected_device is not None and selected_device.backend == "cuda":
            cuda_reference.append(reference_snapshot)
            cuda_prediction.append(prediction_snapshot)
            cuda_masks.append(finite)
            if len(cuda_reference) >= batch_cases:
                flush_cuda_batch()
        else:
            error = prediction_snapshot - reference_snapshot
            error_sum[finite] += error[finite]
            absolute_error_sum[finite] += np.abs(error[finite])
            error_square_sum[finite] += np.square(error[finite], dtype=np.float64)
            counts[finite] += 1
            if selected_device is not None:
                routes.add(selected_device.label)
    if (
        x_values is None
        or y_values is None
        or error_sum is None
        or absolute_error_sum is None
        or error_square_sum is None
        or counts is None
        or divergent is None
    ):
        msg = "Aggregate transient snapshot evidence was not initialized."
        raise RuntimeError(msg)
    flush_cuda_batch()
    if device_evidence is not None:
        device_evidence["route"] = "; ".join(sorted(routes))
    if not all(np.isfinite(values[counts > 0]).all() for values in (error_sum, absolute_error_sum, error_square_sum)):
        msg = "Aggregate transient error statistics are non-finite on required support."
        raise ValueError(msg)
    mean_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    mean_absolute_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    mean_square_error = np.full(error_sum.shape, np.nan, dtype=np.float64)
    np.divide(error_sum, counts, out=mean_error, where=counts > 0)
    np.divide(absolute_error_sum, counts, out=mean_absolute_error, where=counts > 0)
    np.divide(error_square_sum, counts, out=mean_square_error, where=counts > 0)
    error_variance = np.maximum(mean_square_error - np.square(mean_error), 0.0)
    mean_absolute_error[divergent] = np.nan
    error_variance[divergent] = np.nan
    return (
        x_values,
        y_values,
        mean_absolute_error,
        np.sqrt(error_variance),
        np.asarray(reference_means, dtype=np.float64),
        np.asarray(prediction_means, dtype=np.float64),
    )


def _labelled_record_series(
    value: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    allow_empty_contexts: bool = False,
) -> dict[str, tuple[Any, ...]]:
    """Normalize one sequence or labelled multi-context aggregate."""
    if isinstance(value, Mapping):
        series: dict[str, tuple[Any, ...]] = {}
        for label, records in value.items():
            if not isinstance(label, str) or not label:
                msg = "Transient aggregate context labels must be non-empty text."
                raise ValueError(msg)
            admitted = tuple(records)
            if not admitted and not allow_empty_contexts:
                msg = "Transient aggregate contexts must contain at least one record."
                raise ValueError(msg)
            series[label] = admitted
    else:
        admitted = tuple(value)
        if not admitted:
            msg = "Transient aggregate plots require at least one record."
            raise ValueError(msg)
        series = {"": admitted}
    if not series or not any(series.values()):
        msg = "Transient aggregate plots require at least one context."
        raise ValueError(msg)
    return series


def plot_aggregate_error_maps(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    physical_time: float,
    statistic: str,
    compute_device: str = "cpu",
) -> Figure:
    """Plot exact-time mean or standard-deviation error maps by context."""
    fields = _selected_fields(state_fields)
    if statistic not in {"mean", "std"}:
        msg = "Aggregate error-map statistic must be 'mean' or 'std'."
        raise ValueError(msg)
    series = _labelled_record_series(records, allow_empty_contexts=True)
    figure, axes = layout.map_subplots(
        rows=len(fields),
        columns=len(series),
    )
    total_cases = 0
    routes: set[str] = set()
    for column, (context, current_records) in enumerate(series.items()):
        total_cases += len(current_records)
        for row, field in enumerate(fields):
            if not current_records:
                axis = axes[row, column]
                axis.text(0.5, 0.5, "No case at this exact time", ha="center", va="center", transform=axis.transAxes)
                if row == 0 and context:
                    axis.set_title(context)
                continue
            device_evidence: dict[str, str] = {}
            (
                x_values,
                y_values,
                mean_error,
                std_error,
                _reference_means,
                _prediction_means,
            ) = _aggregate_snapshot_evidence(
                current_records,
                state_field=field,
                physical_time=physical_time,
                compute_device=compute_device,
                device_evidence=device_evidence,
            )
            routes.update(device_evidence.values())
            displayed = mean_error if statistic == "mean" else std_error
            axis = axes[row, column]
            if not np.isfinite(displayed).any():
                axis.text(0.5, 0.5, "Prediction non-finite in every cell", ha="center", va="center", transform=axis.transAxes)
                if row == 0 and context:
                    axis.set_title(context)
                continue
            image = axis.pcolormesh(
                x_values,
                y_values,
                displayed,
                shading="auto",
                cmap=visual_semantics.field_visual_semantics(
                    field,
                    role="absolute_error",
                ).colormap,
                norm=layout.linear_norm(displayed[np.isfinite(displayed)]),
            )
            axis.set_aspect("equal")
            if row == 0 and context:
                axis.set_title(context)
            layout.add_map_colorbar(figure, image, axis, label=_display_unit(field, difference=True))
            if column == 0:
                layout.add_channel_row_label(axis, _display_label(field))
    layout.apply_map_grid_axis_labels(
        axes,
        x_label="x [m]",
        y_label="y [m]",
    )
    statistic_label = "Mean absolute" if statistic == "mean" else "Standard deviation of signed"
    layout.set_suptitle_over_axes(
        figure,
        (f"{statistic_label} error at t={physical_time:g} h — {total_cases} case-context observations\n" + "; ".join(sorted(routes))),
        axes.flat,
    )
    return figure


def plot_predicted_vs_reference_channels(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    physical_time: float,
) -> Figure:
    """Plot exact-time case spatial means by channel and material context."""
    fields = _selected_fields(state_fields)
    series = _labelled_record_series(records, allow_empty_contexts=True)
    figure, raw_axes = plt.subplots(
        len(fields),
        len(series),
        figsize=(
            max(7.2, 5.8 * len(series)),
            max(5.2, 4.6 * len(fields)),
        ),
        layout="constrained",
        squeeze=False,
    )
    total_cases = 0
    for column, (context, current_records) in enumerate(series.items()):
        total_cases += len(current_records)
        for row, field in enumerate(fields):
            if not current_records:
                axis = raw_axes[row, column]
                axis.text(0.5, 0.5, "No case at this exact time", ha="center", va="center", transform=axis.transAxes)
                if row == 0 and context:
                    axis.set_title(context)
                continue
            (
                _x,
                _y,
                _mean_error,
                _std_error,
                reference_display,
                prediction_display,
            ) = _aggregate_snapshot_evidence(
                current_records,
                state_field=field,
                physical_time=physical_time,
            )
            finite_prediction = np.isfinite(prediction_display)
            # A wet-basis reference mean shares the prediction's valid cells, so it is NaN where no cell of the case is valid.
            shown = np.concatenate((reference_display, prediction_display))
            shown = shown[np.isfinite(shown)]
            lower, upper = (float(np.min(shown)), float(np.max(shown))) if shown.size else (0.0, 0.0)
            if lower == upper:
                padding = max(abs(lower) * 0.05, 1.0)
                lower -= padding
                upper += padding
            axis = raw_axes[row, column]
            axis.scatter(
                reference_display,
                prediction_display,
                s=55,
                alpha=0.8,
            )
            axis.plot(
                (lower, upper),
                (lower, upper),
                color="black",
                linestyle="--",
            )
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            if not finite_prediction.all():
                hidden = int(np.count_nonzero(~finite_prediction))
                axis.text(0.02, 0.98, f"{hidden} non-finite predicted case mean(s) not shown", va="top", fontsize="small", transform=axis.transAxes)
            label = _display_label(field)
            axis.set_xlabel(f"Reference case spatial mean — {label}")
            axis.set_ylabel(f"Predicted case spatial mean — {label}")
            if row == 0 and context:
                axis.set_title(context)
            axis.grid(alpha=0.25)
    figure.suptitle(f"Prediction versus reference at t={physical_time:g} h — {total_cases} case-context observations")
    return figure


def plot_error_distributions(
    case_frame: pd.DataFrame,
    *,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
) -> Figure:
    """Show complete-rollout case-error distributions by stored channel."""
    selected = _full_autonomous_cumulative(case_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient error distributions require at least one stored channel metric."
        raise ValueError(msg)
    values = []
    for field in fields:
        column = f"normalized_rmse_{field}"
        if column not in selected:
            msg = f"Transient error distributions lack {column!r}."
            raise ValueError(msg)
        current = selected[column].to_numpy(dtype=np.float64)
        if not np.isfinite(current).all():
            msg = "Transient case errors must be finite."
            raise ValueError(msg)
        values.append(current)
    figure, axis = plt.subplots(figsize=(9.0, 5.2), layout="constrained")
    axis.boxplot(values, tick_labels=tuple(field_labels.field_label(field, mathtext=True) for field in fields), showfliers=True)
    axis.set_ylabel("Normalized RMSE [-]")
    axis.set_title("Full-rollout case error distributions")
    axis.grid(axis="y", alpha=0.25)
    return figure


def _thinned_finite_errors(record: Any, field: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Return one record's thinned finite target magnitudes and absolute errors, its element count and its non-finite count.

    Finite errors are kept at fixed element positions up to the point limit; index arrays copy
    them, so the record's full arrays are released when this returns.
    """
    _times, reference, prediction = _trajectory_values(record, field)
    shown_reference = _absolute_display(record, field, reference[1:])
    target = np.abs(shown_reference.ravel())
    if moisture_display.is_moisture_field(field):
        # Each record converts with its own dry bulk density; a cell unphysical in either field counts as non-finite.
        difference = _absolute_display(record, field, prediction[1:]) - shown_reference
    else:
        difference = _display_state_values(field, prediction[1:] - reference[1:], error=True)
    error = np.abs(difference.ravel())
    finite = np.isfinite(error)
    kept = np.arange(0, error.size, max(1, int(np.ceil(error.size / _TARGET_MAGNITUDE_POINT_LIMIT))))
    kept = kept[finite[kept]]
    return target[kept], error[kept], error.size, error.size - int(np.count_nonzero(finite))


def _finite_error_trend(records: Sequence[Any], field: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Return target-magnitude bin centers and median finite absolute errors, plus the exact non-finite and total counts.

    The bins and medians describe the thinned finite errors only. The non-finite (diverging) errors
    are counted over every element, so the plot names them instead of hiding them.
    """
    target_chunks: list[np.ndarray] = []
    error_chunks: list[np.ndarray] = []
    total = 0
    diverging = 0
    for record in records:
        target, error, count, non_finite = _thinned_finite_errors(record, field)
        target_chunks.append(target)
        error_chunks.append(error)
        total += count
        diverging += non_finite
    targets = np.concatenate(target_chunks)
    errors = np.concatenate(error_chunks)
    if not targets.size:
        return np.empty(0), np.empty(0), diverging, total
    edges = np.unique(np.quantile(targets, np.linspace(0.0, 1.0, _TARGET_MAGNITUDE_BIN_COUNT + 1)))
    count = max(1, len(edges) - 1)
    bins = np.clip(np.digitize(targets, edges[1:-1], right=True), 0, count - 1)
    present = [index for index in range(count) if np.any(bins == index)]
    centers = np.asarray([float(np.median(targets[bins == index])) for index in present])
    medians = np.asarray([float(np.median(errors[bins == index])) for index in present])
    return centers, medians, diverging, total


def plot_error_vs_target_magnitude(
    series: Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
) -> Figure:
    """Plot bounded trajectory-element error trends against target magnitude."""
    fields = _selected_fields(state_fields)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not records for records in admitted.values()):
        msg = "Target-magnitude error trends require labelled non-empty record groups."
        raise ValueError(msg)
    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(8.8, max(4.8, 3.6 * len(fields))),
        layout="constrained",
        squeeze=False,
    )
    colors = plt.colormaps["tab10"]
    for row, field in enumerate(fields):
        axis = axes[row, 0]
        for series_index, (label, records) in enumerate(admitted.items()):
            centers, medians, diverging, total = _finite_error_trend(records, field)
            # The medians describe finite errors; the exact count of diverging values is named, never hidden.
            shown = f"{label} · {diverging} of {total} values non-finite, excluded" if diverging else label
            axis.plot(
                centers,
                medians,
                marker="o",
                linewidth=2.0,
                color=colors(series_index % colors.N),
                label=shown,
            )
        axis.set_xlabel(f"|{_display_symbol(field)}| [{_display_unit(field, difference=False)}]")
        axis.set_ylabel(f"Median absolute error [{_display_unit(field, difference=True)}]")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize="small")
    figure.suptitle("Error vs |GT| magnitude — all stored post-origin times; deterministic bounded bins")
    return figure


def maximum_error_physical_time(
    record: Any,
    *,
    state_fields: Sequence[str],
) -> float:
    """Return the stored post-origin time with largest scale-normalized field error."""
    fields = _selected_fields(state_fields)
    reference = np.asarray(_value(record, "reference_states"), dtype=np.float64)
    prediction = np.asarray(_value(record, "predicted_states"), dtype=np.float64)
    times = np.asarray(_value(record, "physical_times"), dtype=np.float64)
    if (
        reference.shape != prediction.shape
        or reference.ndim != _STATE_ARRAY_RANK
        or times.shape != (reference.shape[0],)
        or len(times) < _MINIMUM_TRAJECTORY_TIMES
    ):
        msg = "Error-maximizing time requires aligned non-empty transient evidence."
        raise ValueError(msg)
    scores = np.zeros(len(times) - 1, dtype=np.float64)
    for field in fields:
        if field == _GRAIN_MOISTURE_FIELD:
            reference_values = _grain_moisture(record, reference)
            prediction_values = _grain_moisture(record, prediction)
        else:
            channel = artifact.STATE_ORDER.index(field)
            reference_values = reference[:, channel]
            prediction_values = prediction[:, channel]
        selected_error = prediction_values - reference_values
        scale = max(float(np.std(reference_values, dtype=np.float64)), np.finfo(np.float64).eps)
        scores += np.sqrt(np.mean(np.square(selected_error[1:], dtype=np.float64), axis=(1, 2))) / scale
    # A non-finite score is a diverging prediction and counts as the worst time.
    return float(times[1 + int(np.argmax(np.where(np.isfinite(scores), scores, np.inf)))])


def _full_autonomous_cumulative(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the comparable complete-rollout cumulative rows."""
    required = {
        "frame",
        "material_family",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame):
        msg = "Transient aggregate plot requires complete-rollout summary columns."
        raise ValueError(msg)
    selected = frame.loc[(frame["mode"] == "autonomous_full") & (frame["requested_horizon"] == "full") & (frame["scope"] == "cumulative")].copy()
    if selected.empty:
        msg = "Transient aggregate plot requires full-autonomous cumulative evidence."
        raise ValueError(msg)
    return selected


def plot_channel_error(
    summary_frame: pd.DataFrame,
    *,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
) -> Figure:
    """Compare complete-rollout normalized RMSE for selected stored states."""
    selected = _full_autonomous_cumulative(summary_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient channel-error plots require at least one stored state."
        raise ValueError(msg)
    columns = tuple(f"normalized_rmse_{field}" for field in fields)
    if any(column not in selected for column in columns):
        msg = "Transient channel-error plot lacks per-state normalized RMSE."
        raise ValueError(msg)
    labels = tuple(field_labels.field_label(field, mathtext=True) for field in fields)
    positions = np.arange(len(columns), dtype=np.float64)
    width = 0.8 / len(selected)
    figure, axis = plt.subplots(figsize=(9.0, 5.2), layout="constrained")
    role_count = selected["dataset_role"].nunique() if "dataset_role" in selected else 0
    include_frame = selected["frame"].nunique() > role_count
    for offset, (_, row) in enumerate(selected.iterrows()):
        values = row.loc[list(columns)].to_numpy(dtype=np.float64)
        axis.bar(
            positions - 0.4 + width / 2.0 + offset * width,
            values,
            width=width,
            label=_material_role_label(
                row["frame"],
                row["material_family"],
                row.get("dataset_role"),
                include_frame=include_frame,
            ),
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Normalized RMSE [-]")
    axis.set_title("Full-rollout per-state error")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    return figure


def plot_horizon_error(summary_frame: pd.DataFrame) -> Figure:
    """Plot rollout-index and physical-time views without conflating axes."""
    required = {
        "frame",
        "material_family",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
        "elapsed_physical_time_median",
        "contributing_record_count",
    }
    if not isinstance(summary_frame, pd.DataFrame) or not required.issubset(summary_frame):
        msg = "Horizon-error plot requires a transient dataset summary frame."
        raise ValueError(msg)
    # A material without records at a horizon has no error or elapsed time there; its missing cases show in the tables.
    selected = summary_frame.loc[(summary_frame["mode"] == "rolling_origin") & (summary_frame["contributing_record_count"] > 0)].copy()
    if selected.empty:
        msg = "Horizon-error plot requires rolling-origin summaries."
        raise ValueError(msg)
    horizons = tuple(
        sorted(
            selected["requested_horizon"].unique(),
            key=lambda value: float("inf") if value == "full" else int(value),
        )
    )
    horizon_positions = {value: position for position, value in enumerate(horizons)}
    physical_times = selected["elapsed_physical_time_median"].to_numpy(dtype=np.float64)
    display_time = time_axis.physical_time_display(
        physical_times,
        preferred_unit="auto",
    )
    figure, axes_grid, legend_axis = layout.subplots_with_legend_column(
        rows=1,
        columns=2,
        column_width=5.2,
        row_height=5.0,
        legend_width=3.0,
    )
    axes = axes_grid[0]
    handles = []
    role_count = selected["dataset_role"].nunique() if "dataset_role" in selected else 0
    include_frame = selected["frame"].nunique() > role_count
    role_by_context = (
        selected.groupby(["frame", "material_family"], sort=False)["dataset_role"].first().to_dict() if "dataset_role" in selected else {}
    )
    for (frame, material, scope), group in selected.groupby(
        ["frame", "material_family", "scope"],
        sort=False,
    ):
        ordered = group.assign(_position=group["requested_horizon"].map(horizon_positions)).sort_values("_position")
        label = f"{_material_role_label(frame, material, role_by_context.get((frame, material)), include_frame=include_frame)} — {scope}"
        (handle,) = axes[0].plot(
            ordered["_position"],
            ordered["normalized_drying_group_macro_rmse"],
            marker="o",
            label=label,
        )
        handles.append(handle)
        axes[1].plot(
            display_time.values(ordered["elapsed_physical_time_median"]),
            ordered["normalized_drying_group_macro_rmse"],
            marker="o",
            color=handle.get_color(),
        )
    axes[0].set_xticks(
        np.arange(len(horizons)),
        [str(value) for value in horizons],
    )
    axes[0].set_xlabel("Requested horizon [transitions]")
    display_time.configure(axes[1])
    axes[1].set_xlabel(f"Median elapsed physical time [{display_time.unit}]")
    for axis in axes:
        axis.set_ylabel("Normalized Drying group macro RMSE [-]")
        axis.grid(axis="y", alpha=0.25)
    legend_axis.legend(handles=handles, loc="upper left", frameon=False)
    return figure


def plot_endpoint_vs_cumulative(summary_frame: pd.DataFrame) -> Figure:
    """Compare endpoint and cumulative central error without conflating scopes."""
    required = {
        "frame",
        "material_family",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(summary_frame, pd.DataFrame) or not required.issubset(summary_frame):
        msg = "Endpoint comparison requires transient summary columns."
        raise ValueError(msg)
    pivot = summary_frame.pivot_table(
        index=["frame", "material_family", "mode", "requested_horizon"],
        columns="scope",
        values="normalized_drying_group_macro_rmse",
        aggfunc="first",
    ).dropna(subset=["cumulative", "endpoint"])
    if pivot.empty:
        msg = "Endpoint comparison requires paired cumulative and endpoint evidence."
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(6.5, 6.0), layout="constrained")
    for frame_name, group in pivot.groupby(level="frame", sort=False):
        axis.scatter(group["cumulative"], group["endpoint"], label=str(frame_name))
    axis.legend()
    maximum = float(np.max(pivot[["cumulative", "endpoint"]].to_numpy()))
    axis.plot((0.0, maximum), (0.0, maximum), linestyle="--", color="black")
    axis.set_xlabel("Cumulative central error [-]")
    axis.set_ylabel("Endpoint central error [-]")
    axis.grid(alpha=0.25)
    return figure


def plot_target_time(records: Sequence[Any] | Mapping[str, Sequence[Any]]) -> Figure:
    """Plot reached target pairs by saved model and retain censored/unavailable counts."""
    groups = records if isinstance(records, Mapping) else {"Selected model": records}
    pairs: dict[str, tuple[list[float], list[float]]] = {}
    notices = []
    all_times: list[float] = []
    for label, selected in groups.items():
        reference: list[float] = []
        prediction: list[float] = []
        censored = unavailable = 0
        for record in selected:
            target = _value(record, "target")
            if not isinstance(target, Mapping):
                msg = "Transient target plot requires mapping evidence."
                raise TypeError(msg)
            if target.get("reference_available") is not True or target.get("predicted_available") is not True:
                unavailable += 1
                continue
            ref_time = target.get("reference_time_to_target")
            pred_time = target.get("predicted_time_to_target")
            if isinstance(ref_time, Real) and isinstance(pred_time, Real):
                reference.append(float(ref_time))
                prediction.append(float(pred_time))
            else:
                censored += 1
        pairs[str(label)] = (reference, prediction)
        all_times.extend((*reference, *prediction))
        notices.append(f"{label}: paired reached {len(reference)}, censored {censored}, unavailable {unavailable}")
    figure, axis = plt.subplots(figsize=(8.5, 6.0), layout="constrained")
    display_time = time_axis.physical_time_display(tuple(all_times) if all_times else (0.0,), preferred_unit="auto")
    for label, (reference, prediction) in pairs.items():
        if reference:
            axis.scatter(display_time.values(reference), display_time.values(prediction), label=label)
    if all_times:
        lower, upper = display_time.values((min(all_times), max(all_times)))
        axis.plot((lower, upper), (lower, upper), linestyle="--", color="black")
        axis.legend()
    display_time.configure(axis)
    display_time.configure(axis, dimension="y")
    axis.set_xlabel(f"Reference time to target [{display_time.unit}]")
    axis.set_ylabel(f"Predicted time to target [{display_time.unit}]")
    axis.set_title("\n".join(notices), fontsize="small")
    axis.grid(alpha=0.25)
    return figure


def plot_training_performance_vs_compute(performance: pd.DataFrame, *, lineage: comparison.LineageComparison) -> Figure:
    """
    Plot the admitted A0/B/A1 arms with a result in their actual clock, and every arm without one as a status line.

    A B or A1 without an eligible checkpoint has no rows; the admitted comparison names its
    status, budget progress and eligible points in the title, so A0 alone still plots.
    """
    unavailable = [arm for arm in (lineage.b, lineage.a1) if not isinstance(arm, comparison.TrainingArmEvidence)]
    required = {
        "comparison_arm",
        "material_family",
        "optimizer_device_compute",
        "normalized_drying_group_macro_rmse",
        "checkpoint_role",
        "mode",
        "requested_horizon",
        "scope",
    }
    if not isinstance(performance, pd.DataFrame) or not required.issubset(performance):
        msg = "Matched-compute plot requires arm, compute, metric, checkpoint, mode, horizon, and scope."
        raise ValueError(msg)
    arms = set(performance["comparison_arm"])
    if arms != {"A0", "A1", "B"}.difference(arm.arm for arm in unavailable):
        msg = "Matched-compute plot requires rows for exactly the A0/A1/B arms that have a result."
        raise ValueError(msg)
    matched_rows = performance["comparison_arm"].isin(("A1", "B"))
    clocks = set(performance.loc[matched_rows, "compute_clock"]) if "compute_clock" in performance else set()
    if matched_rows.any() and clocks != {"train_seconds"}:
        msg = "Matched A1/B plot requires persisted committed train_seconds for every arm with a result."
        raise ValueError(msg)
    selected = performance.loc[performance["scope"] == "cumulative"].copy()
    if selected.empty:
        msg = "Matched-compute plot requires cumulative Evaluation evidence."
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(10.0, 6.0), layout="constrained")
    for (material, mode, horizon), group in selected.groupby(
        ["material_family", "mode", "requested_horizon"],
        sort=False,
    ):
        if group["comparison_arm"].duplicated().any():
            msg = "Matched-compute series requires at most one value per arm, material, mode, and horizon."
            raise ValueError(msg)
        ordered = group.assign(_arm_order=group["comparison_arm"].map({"A0": 0, "A1": 1, "B": 2})).sort_values("_arm_order")
        axis.plot(
            ordered["training_compute"] if "training_compute" in ordered else ordered["optimizer_device_compute"],
            ordered["normalized_drying_group_macro_rmse"],
            marker="o",
            label=f"{_material_label(material)} · {mode} / {horizon}",
        )
        for row in ordered.itertuples(index=False):
            axis.annotate(
                str(row.comparison_arm),
                (row.training_compute if hasattr(row, "training_compute") else row.optimizer_device_compute, row.normalized_drying_group_macro_rmse),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
    axis.set_xlabel("Committed training time [s]")
    axis.set_ylabel("Cumulative normalized Drying group macro RMSE [-]")
    if unavailable:
        axis.set_title(
            "\n".join(
                f"{arm.arm}: {arm.training_status} at {100.0 * arm.budget_progress:.0f} % of its budget, "
                f"{arm.eligible_points} eligible points, no result"
                for arm in unavailable
            ),
            fontsize="small",
        )
    axis.legend()
    axis.grid(alpha=0.25)
    return figure


def plot_pipeline_degradation(
    evidence: Sequence[comparison.AirflowDegradationMetrics],
) -> Figure:
    """Plot B/A, C/A, and C/B errors while retaining unavailable C evidence."""
    values = tuple(evidence)
    if not values:
        msg = "Pipeline degradation plot requires metric evidence."
        raise ValueError(msg)
    figure, axes = plt.subplots(
        len(values),
        1,
        figsize=(8.5, 3.8 * len(values)),
        squeeze=False,
        layout="constrained",
    )
    for axis, item in zip(axes[:, 0], values, strict=True):
        heights = [
            item.drying_surrogate_error,
            np.nan if item.complete_pipeline_error is None else item.complete_pipeline_error,
            np.nan if item.airflow_substitution_discrepancy is None else item.airflow_substitution_discrepancy,
        ]
        axis.bar(("B/A", "C/A", "C/B"), heights)
        axis.set_ylabel(item.metric_id)
        if not item.c_available:
            axis.text(
                0.98,
                0.95,
                f"C unavailable: {item.c_unavailable_reason}",
                transform=axis.transAxes,
                ha="right",
                va="top",
            )
    return figure


def plot_timing_speedups(report: timing.TransientTimingReport) -> Figure:
    """Plot paired per-case values for every component-composed speedup."""
    if not isinstance(report, timing.TransientTimingReport):
        msg = "Timing plot requires one validated TransientTimingReport."
        raise TypeError(msg)
    names = tuple(report.speedups)
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=(10.0, 3.6 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes[:, 0], names, strict=True):
        summary = report.speedups[name]
        available = [(case.case_id, case.speedup) for case in summary.cases if case.speedup is not None]
        axis.bar(
            [case_id for case_id, _speedup in available],
            [speedup for _case_id, speedup in available],
        )
        axis.set_ylabel("Duration ratio [-]")
        axis.set_title(
            f"{summary.reference_boundary} /\n{summary.surrogate_boundary}\n"
            f"{summary.available_count}/{summary.total_count} paired; ratio of sums={summary.ratio_of_sums}",
            fontsize="small",
        )
        axis.tick_params(axis="x", rotation=45)
    return figure


def plot_timing_distributions(report: timing.TransientTimingReport) -> Figure:
    """Plot paired reference and surrogate runtime distributions per formula."""
    if not isinstance(report, timing.TransientTimingReport):
        msg = "Timing distribution plot requires one validated TransientTimingReport."
        raise TypeError(msg)
    names = tuple(report.speedups)
    figure, axes = plt.subplots(
        len(names),
        1,
        figsize=(10.0, 3.8 * len(names)),
        squeeze=False,
        layout="constrained",
    )
    for axis, name in zip(axes[:, 0], names, strict=True):
        summary = report.speedups[name]
        pairs = [
            (case.reference_seconds, case.surrogate_seconds)
            for case in summary.cases
            if case.reference_seconds is not None and case.surrogate_seconds is not None
        ]
        if pairs:
            reference = [pair[0] for pair in pairs]
            surrogate = [pair[1] for pair in pairs]
            axis.boxplot((reference, surrogate), positions=(0, 1), widths=0.5)
            for reference_seconds, surrogate_seconds in pairs:
                axis.plot(
                    (0, 1),
                    (reference_seconds, surrogate_seconds),
                    marker="o",
                    alpha=0.45,
                )
        else:
            axis.text(
                0.5,
                0.5,
                "No paired runtime evidence available",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_xticks((0, 1), (summary.reference_boundary, summary.surrogate_boundary))
        axis.set_ylabel("Recorded duration [s]")
        axis.set_title(f"{summary.comparison_scope} — {summary.available_count}/{summary.total_count} paired cases", fontsize="small")
        axis.grid(axis="y", alpha=0.25)
    return figure


def _case_accuracy_values(accuracy: pd.DataFrame, *, label: str) -> Mapping[str, float]:
    """Admit one finite non-negative central-accuracy value per case."""
    required = {"case_id", "normalized_drying_group_macro_rmse"}
    if not isinstance(accuracy, pd.DataFrame) or not required.issubset(accuracy):
        msg = f"{label} requires case identity and central metric."
        raise ValueError(msg)
    values: dict[str, float] = {}
    for case_id, value in zip(
        accuracy["case_id"],
        accuracy["normalized_drying_group_macro_rmse"],
        strict=True,
    ):
        if not isinstance(case_id, str) or not case_id or case_id in values:
            msg = f"{label} requires unique non-empty case identities."
            raise ValueError(msg)
        if isinstance(value, bool) or not isinstance(value, Real):
            msg = f"{label} central metrics must be real scalars."
            raise TypeError(msg)
        admitted = float(value)
        if not np.isfinite(admitted) or admitted < 0.0:
            msg = f"{label} central metrics must be finite and non-negative."
            raise ValueError(msg)
        values[case_id] = admitted
    return values


def plot_accuracy_runtime_comparison(
    accuracy: pd.DataFrame,
    reports: Mapping[str, timing.TransientTimingReport],
    *,
    measure: str = "prepared_wall",
) -> Figure:
    """Compare saved models using exact paired case accuracy and a named measured clock."""
    labels = {
        "prepared_wall": "Prepared rollout request wall, warmed median [s]",
        "model": "Rollout model calls only, warmed median [s]",
        "ratio": "COMSOL reported drying solution / model-call time [ratio]",
    }
    if measure not in labels or "frame" not in accuracy:
        msg = "Accuracy/runtime comparison requires a named clock and frame identity."
        raise ValueError(msg)
    figure, axis = plt.subplots(figsize=(10.0, 6.5), layout="constrained")
    notices = []
    for index, (frame_name, report) in enumerate(reports.items()):
        selected = accuracy.loc[accuracy["frame"] == frame_name]
        values = _case_accuracy_values(selected, label=f"Accuracy/runtime comparison {frame_name}")
        durations: Mapping[str, float]
        if measure == "ratio":
            durations = {case.case_id: case.speedup for case in report.speedups["drying_only_solver_speedup"].cases if case.speedup is not None}
        else:
            component = "drying_no_prepared_rollout_wall_seconds" if measure == "prepared_wall" else "drying_no_rollout_model_seconds"
            durations = timing.component_case_medians(report, component)
        paired = [(durations[case_id], value) for case_id, value in values.items() if case_id in durations]
        if paired:
            axis.scatter(
                [point[0] for point in paired],
                [point[1] for point in paired],
                color=plt.colormaps["tab10"](index % 10),
                label=f"{frame_name} ({len(paired)} cases)",
            )
        notices.append(f"{frame_name}: {len(paired)}/{len(values)} paired")
    axis.set_xlabel(labels[measure])
    axis.set_ylabel("Cumulative full-rollout normalized Drying group macro RMSE [-]")
    axis.set_title("\n".join(notices), fontsize="small")
    if axis.collections:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No paired saved accuracy/runtime evidence", transform=axis.transAxes, ha="center")
    axis.grid(alpha=0.25)
    return figure
