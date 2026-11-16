"""Render sequence-aware transient-drying Evaluation evidence.

Responsibilities:
  - Compare reference, prediction, and signed-error fields at exact physical times
  - Draw state maps on the fixed map-grid geometry: state scales from the 5th to the 95th percentile, per drawn time or,
    per field, pooled over all stored times (``fixed_time_fields``; ``default_fixed_time_fields`` names all but moisture), with arrows
    for the cells beyond (on the state bars fixed over the scaled times), error scales (signed by default) fixed over the
    scaled times, tick decimals fixed per colour bar over the scaled times, one scale note under the title and a mean |e|
    line above each error map
  - Draw exact-time aggregate error maps on the same geometry: a row per field, a column per context, the signed mean by
    default (or the absolute or train-sigma-normalized one), one round scale per row locked over the contexts, a unit on every bar
  - Plot physical state trajectories and exact grain-moisture derivations, and the error over physical time, in time
    windows (the first 24 h, the middle in days, the last 24 h): a prediction-minus-reference row under each state mean,
    one labelled line per distinct target moisture on the grain moisture, a case-count row, one legend outside at the top right;
    the error over time alternatively over the normalized time τ = t / t_stop in equal τ bins
  - Plot rollout error across horizons and reduction scopes
  - Plot per-field relative norms over physical time with their supported case counts
  - Present target, matched-compute, pipeline-degradation, and timing evidence; duration ratios as the median of the paired
    per-case ratios with their 10th to 90th percentile as error bar

Design principles:
  - Styles are resolved once per comparison from all its loaded runs (``series.comparison_styles``) and passed to every view, which
    looks each series up by its entities (model kind, Dataset role, material, selected checkpoint, read alike from record
    identity and session provenance) and never resolves its own rows: a run (model kind and checkpoint) keeps one colour in every
    view, also where a view holds evidence for only some runs; a figure of run colours says «colour = run» in its context line;
    legends follow ``figure_layout.add_legend``
  - Dynamic states follow the shared EDA labels, units, and display conversions in the active label style: phi in
    percent with errors in pp, and the water densities w_surf, w_int and w_gr as surface, internal and grain moisture in
    percent of wet basis with errors in pp, converted per case with its dry bulk density; cells with negative or
    non-finite water are masked
  - Normalized errors stay on the stored states and saved train scales
  - Physical times are selected exactly and never nearest-neighbour substituted
  - Cases of different duration share the last window as hours before their own end; the windows only select stored times
  - In τ, every case starts at t = 0 and keeps its own stored times: a bin averages each case's values inside it before the
    cases are reduced; a case of the error over time without a time after its origin is refused, never dropped
  - A duration ratio is drawn on its log scale only when its median and percentiles are positive and finite
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
from typing import TYPE_CHECKING, Any, Final, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src import datasets, domain, learning
from src.analysis.evaluation import evaluation_transient_artifact as artifact
from src.analysis.evaluation import evaluation_transient_comparison as comparison
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.analysis.evaluation.evaluation_plot import evaluation_plot_layout as plot_layout
from src.analysis.evaluation.evaluation_plot import evaluation_plot_series as series
from src.analysis.presentation import analysis_compute as compute
from src.analysis.presentation import analysis_display_grid as display_grid
from src.analysis.presentation import analysis_display_labels as display_labels
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation import analysis_moisture_display as moisture_display
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.ui import analysis_ui_figure_layout as figure_layout
from src.analysis.ui import analysis_ui_plot_layout as layout
from src.analysis.ui import analysis_ui_time as time_ui

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
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
_WINDOW_PANEL_WIDTH_IN = 4.4  # one time window's column; the legend grows the figure by its own width
_WINDOW_ROW_HEIGHT_IN = 2.6
_HALF_ROW = 0.5  # a difference or case-count row against a full row of 1
_SAME_END_HOURS = 1.0e-6  # cases ending within this of each other share the clock in the last window
_TAU_TOLERANCE = 1.0e-9  # in bin widths: a τ on a bin edge belongs to the bin below, whatever the rounding
_REFERENCE_LINE: Final = {"linewidth": 3.2, "alpha": 0.35}  # the reference is wide and pale beside the prediction in its series style
_PREDICTION_WIDTH = 1.6
_SCOPE_MARKERS: Final = {"cumulative": "o", "endpoint": "s"}  # A4: colour and line style belong to entity and role
_CONDITION_MARKERS: Final = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*")  # mode and horizon of one material
_STAGE_COLOURS: Final = {"B/A": "#1B9E77", "C/A": "#D95F02", "C/B": "#7570B3"}  # airflow-substitution stages, no entity colours
_NEUTRAL_COLOUR = "0.55"
_GROUPED_PANEL_IN = (8.0, 4.2)  # one channel comparison of bars or boxes per frame and material
_TARGET_COLOUR = "0.35"
_TARGET_LABEL_GAP_PT = 11.0  # target values closer than this are pushed apart
_TARGET_LABEL_OFFSET_PT = 4.0  # frame to the target value
_UNAVAILABLE_LABEL = "non-finite, unavailable"  # the legend shows the x marker beside it
_UNAVAILABLE_HEIGHT = 0.06  # of the row, above its x axis


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


def _record_entity(record: Any) -> tuple[Any, ...]:
    """Return one record's entity facts from its identity, ordered as ``series.ENTITY_COLUMNS``; the checkpoint is the selected one."""
    material = _record_material(record)  # also admits the identity as a mapping
    identity = _value(record, "identity")
    checkpoint = identity.get("checkpoint_identity")
    selected = checkpoint.get("best_checkpoint_sha256") if isinstance(checkpoint, Mapping) else None
    return identity.get("model_kind"), identity.get("dataset_role"), material, selected


def _record_styles(groups: Mapping[str, Sequence[Any]], styles: series.SeriesStyles) -> series.SeriesStyles:
    """Return the comparison style of every labelled record series by its records' identity."""
    return series.select_styles({label: [_record_entity(record) for record in records] for label, records in groups.items()}, styles)


def _frame_styles(groups: Mapping[str, pd.DataFrame], styles: series.SeriesStyles) -> series.SeriesStyles:
    """Return the comparison style of every labelled series of session summary rows by their provenance columns ``series.ENTITY_COLUMNS``."""
    columns = list(series.ENTITY_COLUMNS)
    for label, rows in groups.items():
        if missing := [column for column in columns if column not in rows]:
            msg = f"Summary rows of series {label!r} lack the session provenance columns {missing}."
            raise ValueError(msg)
    return series.select_styles({label: list(rows[columns].itertuples(index=False, name=None)) for label, rows in groups.items()}, styles)


def _summary_series(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group summary rows into one series per frame and material, labelled by material and role, plus the frame where roles do not separate them."""
    include_frame = rows["frame"].nunique() > rows["dataset_role"].nunique()
    return {
        _material_role_label(frame, material, group["dataset_role"].iloc[0], include_frame=include_frame): group
        for (frame, material), group in rows.groupby(["frame", "material_family"], sort=False)
    }


def _legend(figure: Figure, axes: np.ndarray, *, outside: bool = False) -> None:
    """Write one legend of every labelled artist of ``axes`` by ``figure_layout.add_legend`` (L1); a figure without one draws none."""
    handles, labels = figure_layout.legend_entries(axes.flat)
    if labels:
        figure_layout.add_legend(figure, axes, handles, labels, outside=outside)


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
    boundaries = field_maps.error_boundaries(finite_errors, signed=signed)
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
    state_format: layout.TickFormat
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
    them from each drawn time. Their tick decimals are the most any time's ticks need, at most three (``layout.state_format``);
    a near-uniform time does not set them but is widened (``lock_scale`` as in ``plot_state_maps``). A state bar takes its
    power of ten from the size of its values (``field_maps.state_exponent``); the transient states lie where that is 0, and
    the transient bars carry none, so a field whose state bands would need one raises ``ValueError`` naming it. A fixed
    field's state scale is the one of the 5th to 95th percentile of the displayed values pooled over all given times (the
    prediction's own without ``lock_scale``), and its decimals come from that scale.
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
            bands = field_maps.rounded_boundaries(*state_range, max_decimals=field_maps.MAX_TICK_DECIMALS)
            bands = field_maps.rounded_boundaries(*state_range) if bands is None else bands
            if bands is not None:
                steps.append(float(bands[1] - bands[0]))
        state_format = layout.state_format(steps)
        fixed = _state_norms(field, shown_reference[0], shown_prediction[0], shared=lock_scale, decimals=state_format.decimals) if pooled else None
        frame_norms = [
            fixed or _state_norms(field, shown, other, shared=lock_scale, decimals=state_format.decimals)
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


def _state_norm(field: str, values: np.ndarray, decimals: int) -> Normalize | None:
    """
    Return the discrete scale of ``field`` over the 5th to the 95th percentile of the finite ``values`` (``None`` without any).

    The bands are the round ones of ``field_maps.state_boundaries``, with a ``BoundaryNorm`` of one colour per band.
    A degenerate percentile range falls back to the values' own full finite range, and a constant field to a padded
    continuous scale of one tick. A near-uniform scale whose step is finer than the bar's ``decimals`` is widened to
    11 steps of 10**-``decimals``. The bar is written without a power of ten, so a field whose state bands (a constant:
    its value) or whose widened bands have a nonzero ``field_maps.state_exponent`` (largest magnitude outside 10^-3 to
    10^4) raises ``ValueError``.
    """
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    scale = _frame_range(values)
    banded = np.array([finite.min(), finite.max()]) if scale is None else field_maps.state_boundaries(*scale)
    boundaries = banded
    if scale is not None and layout.is_finer_than_decimals(float(banded[1] - banded[0]), decimals):
        boundaries = np.linspace(*field_maps.widened_range(*scale, decimals), field_maps.N_COLOR_BINS + 1)
    if any(field_maps.state_exponent(float(drawn[0]), float(drawn[-1])) for drawn in (banded, boundaries)):
        msg = f"The transient state {field!r} spans {boundaries[0]!r} to {boundaries[-1]!r}: its bar would need a power of ten it never carries."
        raise ValueError(msg)
    return layout.linear_norm(finite) if scale is None else BoundaryNorm(boundaries, boundaries.size - 1)


def _state_norms(
    field: str, reference: np.ndarray, prediction: np.ndarray, *, shared: bool, decimals: int
) -> tuple[Normalize | None, Normalize | None]:
    """
    Return one time's reference and prediction colour scales (``None`` for a panel without finite values).

    Shared: both panels take the reference's percentile scale, so one colour means one value and the prediction never
    widens it; a reference without finite values leaves each panel its own scale. Otherwise each panel scales to its own
    5th to 95th percentile.
    """
    if shared and (together := _state_norm(field, reference, decimals)) is not None:
        return together, together
    return _state_norm(field, reference, decimals), _state_norm(field, prediction, decimals)


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
        exponent = layout.error_format(None if norm.error is None else norm.error.boundaries, signed=signed_error).exponent
        error_unit = None if normalized_error else _display_unit(field, difference=True)
        error_texts.append(plot_layout.mean_error_text(values[2], unit=error_unit, exponent=exponent))
        masked_note = "Negative or non-finite water masked" if moisture_display.is_moisture_field(field) else "Non-finite cells masked"
        reference_norm, prediction_norm = (
            _state_norms(field, values[0], values[1], shared=lock_scale, decimals=norm.state_format.decimals) if norm.fixed is None else norm.fixed
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
                layout.set_fixed_decimal_ticks(layout.add_slot_colorbar(figure, image, slot, label=unit, extend=extend), ticks, norm.state_format)
                continue
            colorbar = layout.add_slot_colorbar(
                figure,
                image,
                slot,
                label=f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} [{field_labels.display_unit('1')}]" if normalized_error else unit,
                extend=error_scale.extend,
            )
            layout.set_fixed_decimal_ticks(colorbar, layout.error_ticks(error_scale.boundaries, signed=signed_error))
    plot_layout.set_error_headers(axes, ("Reference", "Prediction", _ERROR_HEADERS[error_mode]), error_texts)
    layout.set_row_labels(axes, [_display_symbol(field) for field in fields])
    layout.set_map_title(
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


def _record_end_hours(records: Sequence[Any], *, exclude_origin: bool) -> np.ndarray:
    """
    Return the end of every record: its last stored time, at most one interval before its exact stop.

    The last stored time, not the exact stop, keeps the offsets before the end on the shared time grid, so cases still aggregate. The
    stored times are validated by the shared time owner; with ``exclude_origin`` every record must hold a time after its origin, so
    none silently leaves the cohort.
    """
    admitted = tuple(records)
    if not admitted:
        msg = "Transient trajectory summaries require at least one record."
        raise ValueError(msg)
    grids = [time_ui.validated_times(_value(record, "physical_times")) for record in admitted]
    short = [str(_value(record, "case_id")) for record, grid in zip(admitted, grids, strict=True) if len(grid) < _MINIMUM_TRAJECTORY_TIMES]
    if exclude_origin and short:
        msg = f"Transient records of cases {short} hold no stored time after its origin; a case is refused, not dropped from the cohort."
        raise ValueError(msg)
    return np.asarray([grid[-1] for grid in grids], dtype=np.float64)


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


def _shared_windows(
    admitted: Mapping[str, tuple[Any, ...]], *, exclude_origin: bool, normalized: bool = False
) -> tuple[tuple[time_ui.TimeWindow, ...], dict[str, np.ndarray]]:
    """
    Return the time windows of all contexts and the final hour of every record (its t_stop), per context.

    ``normalized`` gives the one τ window; otherwise cases of one duration share the clock, and the last window counts hours
    before each case's own end.
    """
    ends = {label: _record_end_hours(records, exclude_origin=exclude_origin) for label, records in admitted.items()}
    if normalized:
        return (time_ui.TAU_WINDOW,), ends
    longest = max(float(values.max()) for values in ends.values())
    shortest = min(float(values.min()) for values in ends.values())
    return time_ui.drying_windows(longest, before_end=longest - shortest > _SAME_END_HOURS), ends


def _window_rows(window: time_ui.TimeWindow, rows: Sequence[tuple[np.ndarray, np.ndarray]], ends: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return every row's exact points inside the window on its coordinate; a row keeps its place when none of its points lies inside."""
    shown = []
    for (times, values), end in zip(rows, ends, strict=True):
        inside = window.select(times, end_hours=float(end))
        shown.append((window.hours(times[inside], end_hours=float(end)), values[inside]))
    return shown


def _tau_bin_count(records: Iterable[Any]) -> int:
    """
    Return the number of equal τ bins, each at least as wide as the widest stored step of any case divided by its end.

    Every case must start at t = 0 and hold a time after it: its steps in τ are then no wider than a bin, so it has a stored
    time in every bin, and τ = t / t_stop is the fraction of its run. A case starting later is refused, since it would miss the
    early bins and shrink their cohort unseen.
    """
    widest = 0.0
    for record in records:
        grid = np.asarray(_value(record, "physical_times"), dtype=np.float64)
        if grid[0] != 0.0:
            msg = f"τ = t / t_stop requires every case to start at t = 0; case {_value(record, 'case_id')!r} starts at {grid[0]:g} h."
            raise ValueError(msg)
        widest = max(widest, float(np.max(np.diff(grid)) / grid[-1]))
    return max(1, math.floor(1.0 / widest + _TAU_TOLERANCE))


def _tau_binned(rows: Sequence[tuple[np.ndarray, np.ndarray]], bin_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Average each row's values over the τ bins ((k - 1) / ``bin_count``, k / ``bin_count``], each drawn at its upper edge.

    A row keeps its own stored times: its bin value is the mean over its stored τ inside, non-finite when one of them is.
    """
    binned = []
    for tau, values in rows:
        index = np.clip(np.ceil(tau * bin_count - _TAU_TOLERANCE).astype(np.int64) - 1, 0, bin_count - 1)
        present = np.unique(index)
        binned.append(((present + 1) / bin_count, np.asarray([np.mean(values[index == bin_index]) for bin_index in present], dtype=np.float64)))
    return binned


def _window_figure(windows: Sequence[time_ui.TimeWindow], heights: Sequence[float]) -> tuple[Figure, np.ndarray]:
    """
    Return a figure on the shared panel engine and its axes ``[row, window]``, a row ``heights`` of a full row high.

    The axes take the shared text sizes; the legend grows the figure later. The engine's gaps are paddings in inches only (no
    figure fraction), so a figure grown for its legend keeps every panel's size.
    """
    figure = plt.figure(figsize=(_WINDOW_PANEL_WIDTH_IN * len(windows), sum(heights) * _WINDOW_ROW_HEIGHT_IN), layout=figure_layout.panel_engine())
    axes = layout.window_axes(figure, figure.add_gridspec(1, 1)[0, 0], windows, heights)
    figure_layout.apply_text_sizes(axes.flat)
    return figure, axes


def _count_symbol(windows: Sequence[time_ui.TimeWindow]) -> str:
    """Return the symbol of the case count over the windows' time: n(τ) over normalized time, else n(t)."""
    return "n(τ)" if windows[0].unit == "1" else "n(t)"


def _draw_case_count(axis: Axes, time: np.ndarray, count: np.ndarray, *, style: visual_semantics.SeriesStyle, label: str) -> None:
    """Draw the cases that have a stored time at each exact time (or τ bin); every field has the same count."""
    axis.step(time, count, where="mid", color=style.color, linestyle=style.linestyle, alpha=0.55, label=label)


def _mark_unavailable(axis: Axes, time: np.ndarray, unavailable: np.ndarray, *, color: Any) -> None:
    """Mark on a field's own row the exact times at which a non-finite value voids its score."""
    if unavailable.any():
        axis.plot(
            time[unavailable],
            np.full(int(unavailable.sum()), _UNAVAILABLE_HEIGHT),
            transform=axis.get_xaxis_transform(),
            linestyle="none",
            marker="x",
            markersize=5,
            color=color,
            clip_on=False,
            label=_UNAVAILABLE_LABEL,
        )


def _draw_target_lines(axes: Sequence[Axes], records: Sequence[Any]) -> list[float]:
    """
    Draw one horizontal line per distinct wet-basis target moisture of the shown cases, in the plot's percent, and return the levels.

    The target is a property of the case: records of one case that carry different targets are refused.
    """
    by_case: dict[str, set[float]] = {}
    for record in records:
        by_case.setdefault(str(_value(record, "case_id")), set()).add(float(_value(_value(record, "target"), "target_moisture_wb")))
    for case_id, case_targets in by_case.items():
        if len(case_targets) > 1:
            msg = f"Transient plot records of case {case_id!r} carry different target_moisture_wb values: {sorted(case_targets)}."
            raise ValueError(msg)
    targets = sorted({target for case_targets in by_case.values() for target in case_targets})
    levels = [float(level) for level in field_labels.display_values(targets, "1", field="target_moisture_wb")]
    for index, level in enumerate(levels):
        for column, axis in enumerate(axes):
            legend = f"Target {_display_symbol(_GRAIN_MOISTURE_FIELD)}" if index == column == 0 else "_nolegend_"
            axis.axhline(level, color=_TARGET_COLOUR, linestyle=":", linewidth=1.2, label=legend)
    return levels


def _label_target_lines(figure: Figure, axis: Axes, levels: Sequence[float]) -> None:
    """
    Write each target's value inside ``axis`` at the left end of its line; values closer than one text line are pushed apart and joined to their line.

    The left end is the start of the run, where the grain moisture stands well above every target, so the values cover no curve.
    """
    engine = figure.get_layout_engine()
    if engine is not None:
        engine.execute(figure)  # the final size and y scale of the row
    lines = [float(axis.transData.transform((0.0, level))[1]) for level in levels]
    placed = list(lines)
    gap = _TARGET_LABEL_GAP_PT * figure.dpi / 72.0
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap)
    centring = (sum(lines) - sum(placed)) / len(placed)
    unit = _display_unit(_GRAIN_MOISTURE_FIELD, difference=False)
    for level, line, pixel in zip(levels, lines, placed, strict=True):
        text = axis.annotate(
            f"{level:g} {unit}",
            xy=(0.0, level),
            xycoords=axis.get_yaxis_transform(),
            xytext=(_TARGET_LABEL_OFFSET_PT, (pixel + centring - line) * 72.0 / figure.dpi),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize="small",
            color=_TARGET_COLOUR,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": _TARGET_COLOUR, "linewidth": 0.6, "relpos": (0.0, 0.5), "shrinkA": 0, "shrinkB": 0},
        )
        text.set_in_layout(False)  # inside the panel: the windows keep their widths


def _finish_window_figure(
    figure: Figure,
    axes: np.ndarray,
    windows: Sequence[time_ui.TimeWindow],
    title: str,
    *,
    note: str | None,
    target_axis: Axes | None = None,
    target_levels: Sequence[float] = (),
) -> Figure:
    """
    Title the windows and the figure, style every time axis and write the one legend outside right, top aligned with the top frame.

    Tick numbers and the time label stay under the bottom row; the case-count row (last) gets its own label and integer ticks.
    The legend (``figure_layout.add_legend``) collects the entries of every axis once by label and grows the figure, never a row.
    """
    last_row = axes.shape[0] - 1
    for column, window in enumerate(windows):
        figure_layout.set_panel_title(axes[0, column], window.title)
        for row in range(axes.shape[0]):
            time_ui.style_window_axis(axes[row, column], window, bottom=row == last_row)
            axes[row, column].grid(axis="y", alpha=0.25)
    axes[last_row, 0].set_ylabel(_count_symbol(windows))
    axes[last_row, 0].set_ylim(bottom=0)
    axes[last_row, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    figure_layout.set_figure_title(figure, axes, title, note=note)
    _legend(figure, axes)
    if target_axis is not None:
        _label_target_lines(figure, target_axis, target_levels)  # after the legend, on the final y scale of the row
    return figure


def plot_state_trajectory_summary(
    series: Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    styles: series.SeriesStyles,
) -> Figure:
    """
    Plot material-aware aggregate or single-case trajectories on exact times, in time windows.

    Each state has a row of reference and prediction means with a half-height row of prediction minus reference below it,
    where a non-finite value that voids a score is marked; the grain moisture also carries one line per distinct wet-basis
    target moisture of the shown cases. A last row counts the cases with a stored time at each exact time, the same for every state.
    A series takes its style from the comparison's ``styles`` (``series.comparison_styles``) by its records' identity; its reference is
    drawn wide and pale under its prediction.
    """
    fields = _selected_fields(state_fields)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not label or not records for label, records in admitted.items()):
        msg = "Trajectory summaries require labelled non-empty record groups."
        raise ValueError(msg)
    styles = _record_styles(admitted, styles)
    windows, ends = _shared_windows(admitted, exclude_origin=False)
    figure, axes = _window_figure(windows, [height for _field in fields for height in (1.0, _HALF_ROW)] + [_HALF_ROW])
    try:
        target_axis, target_levels = None, []
        for row, field in enumerate(fields):
            quantity_axes, difference_axes = axes[2 * row], axes[2 * row + 1]
            for label, records in admitted.items():
                reference_rows: list[tuple[np.ndarray, np.ndarray]] = []
                prediction_rows: list[tuple[np.ndarray, np.ndarray]] = []
                for record in records:
                    record_times, reference, prediction = _trajectory_values(record, field)
                    reference_means, prediction_means = _spatial_means(record, field, reference, prediction)
                    reference_rows.append((record_times, reference_means))
                    prediction_rows.append((record_times, prediction_means))
                style = styles[label]
                line = {"color": style.color, "linestyle": style.linestyle}
                for column, window in enumerate(windows):
                    shown_reference = _window_rows(window, reference_rows, ends[label])
                    if not any(times.size for times, _values in shown_reference):
                        continue
                    reference_summary = _aggregate_exact_time_values(shown_reference)
                    prediction_summary = _aggregate_exact_time_values(_window_rows(window, prediction_rows, ends[label]))
                    time = reference_summary.times
                    axis = quantity_axes[column]
                    axis.plot(time, reference_summary.mean, **line, **_REFERENCE_LINE, label=f"{label} · Reference")
                    axis.plot(time, prediction_summary.mean, **line, linewidth=_PREDICTION_WIDTH, label=f"{label} · Prediction")
                    axis.fill_between(time, reference_summary.lower, reference_summary.upper, color=style.color, alpha=0.08)
                    axis.fill_between(time, prediction_summary.lower, prediction_summary.upper, color=style.color, alpha=0.08)
                    difference_axes[column].plot(time, prediction_summary.mean - reference_summary.mean, **line, linewidth=1.3)
                    _mark_unavailable(
                        difference_axes[column], time, (reference_summary.invalid_count + prediction_summary.invalid_count) > 0, color=style.color
                    )
                    if row == 0:
                        _draw_case_count(axes[-1, column], time, reference_summary.count, style=style, label=f"{label} n(t)")
            if field == _GRAIN_MOISTURE_FIELD:
                target_axis = quantity_axes[0]
                target_levels = _draw_target_lines(quantity_axes, [record for records in admitted.values() for record in records])
            quantity_axes[0].set_ylabel(_display_label(field))
            difference_axes[0].set_ylabel(f"pred \N{MINUS SIGN} ref [{_display_unit(field, difference=True)}]")
            for axis in difference_axes:
                axis.axhline(0.0, color="0.6", linewidth=0.8)
        return _finish_window_figure(
            figure,
            axes,
            windows,
            "Reference vs prediction trajectories",
            note=plot_layout.series_note(styles, "exact-time spatial means; bands require n(t) ≥ 2"),
            target_axis=target_axis,
            target_levels=target_levels,
        )
    except Exception:
        plt.close(figure)
        raise


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
    styles: series.SeriesStyles,
    time_axis: Literal["hours", "tau"] = "hours",
) -> Figure:
    """
    Plot saved-scale channel errors over exact physical time, or over the normalized time τ, by context, in the comparison's ``styles``.

    In hours the line is the normalized RMSE of the cases present at each exact time, reduced like every reported field RMSE
    (root after the material-case mean of the per-case mean squared error); the band spans the 10th to 90th quantile of the
    per-case RMSEs. With ``time_axis="tau"`` each case's times become τ = t / t_stop, t_stop its last stored time, so every case
    ends at τ = 1; τ is cut into equal bins no narrower than any case's widest stored step (``_tau_bin_count``), a case's value
    in a bin is the mean of its per-time mean squared errors there, and the cases are reduced per bin as per exact time above.
    A record without a time after its origin, and in τ a case that does not start at t = 0, raise ``ValueError``.
    """
    if time_axis not in {"hours", "tau"}:
        msg = f"time_axis must be 'hours' or 'tau', not {time_axis!r}."
        raise ValueError(msg)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not fields or not admitted or set(admitted) != set(scaling_states):
        msg = "Error-over-time series require stored fields and matching scaling contexts."
        raise ValueError(msg)
    styles = _record_styles(admitted, styles)
    normalized = time_axis == "tau"
    windows, ends = _shared_windows(admitted, exclude_origin=True, normalized=normalized)
    bin_count = _tau_bin_count(record for records in admitted.values() for record in records) if normalized else 0
    count_symbol = _count_symbol(windows)
    figure, axes = _window_figure(windows, [*([1.0] * len(fields)), _HALF_ROW])
    try:
        for label, records in admitted.items():
            trajectories = [_normalized_squared_error_trajectory(record, scaling_state=scaling_states[label], fields=fields) for record in records]
            materials = [_record_material(record) for record in records]
            style = styles[label]
            for row, field in enumerate(fields):
                rows = [(record_times, values[field]) for record_times, values in trajectories]
                for column, window in enumerate(windows):
                    shown = _window_rows(window, rows, ends[label])
                    if not any(times.size for times, _values in shown):
                        continue
                    summary = _aggregate_exact_time_errors(_tau_binned(shown, bin_count) if normalized else shown, materials=materials)
                    axis = axes[row, column]
                    line = {"color": style.color, "linestyle": style.linestyle, "linewidth": 2.0}
                    axis.plot(summary.times, summary.mean, **line, label=f"{label} · material-case RMSE")
                    axis.fill_between(
                        summary.times, summary.lower, summary.upper, color=style.color, alpha=0.1, label=f"{label} · per-case RMSE 10th-90th"
                    )
                    _mark_unavailable(axis, summary.times, summary.invalid_count > 0, color=style.color)
                    if row == 0:
                        _draw_case_count(axes[-1, column], summary.times, summary.count, style=style, label=f"{label} {count_symbol}")
        for row, field in enumerate(fields):
            axes[row, 0].set_ylabel(f"{field_labels.metric_label('nRMSE', field, mathtext=True)} [{field_labels.display_unit('1')}]")
        where = f"in each of {bin_count} equal τ bins, each case first averaged over its stored times there" if normalized else "at each exact time"
        note = f"root of the material-case mean squared error {where}; per-case RMSE 10th-90th bands require {count_symbol} ≥ 2"
        title = "Error over normalized time" if normalized else "Error over physical time"
        return _finish_window_figure(figure, axes, windows, title, note=plot_layout.series_note(styles, note))
    except Exception:
        plt.close(figure)
        raise


def plot_relative_norms_over_time(curves: Mapping[str, pd.DataFrame], *, styles: series.SeriesStyles) -> Figure:
    """
    Plot the per-field relative L2 and H1 of complete rollouts over physical time with their supported case counts.

    ``curves`` maps a session frame name to :meth:`TransientEvaluationSession.relative_norm_curve`. A gap is
    a time without a supported case in some material; the step line counts the supported cases. A curve takes its frame's style
    of the comparison's ``styles`` (:meth:`TransientEvaluationSession.series_styles`); a name the comparison lacks raises
    ``ValueError``. One legend stands outside right.
    """
    if not curves or any(frame.empty for frame in curves.values()):
        msg = "Relative-norm curves require at least one labelled non-empty curve."
        raise ValueError(msg)
    if missing := [label for label in curves if label not in styles]:
        msg = f"Relative-norm curves {missing} are no frames of the comparison's styles."
        raise ValueError(msg)
    all_times = np.concatenate([frame["physical_time"].to_numpy(dtype=np.float64) for frame in curves.values()])
    display_time = time_ui.physical_time_display(all_times, preferred_unit="auto")
    figure, axes = plt.subplots(2, len(artifact.STATE_ORDER), figsize=(5.0 * len(artifact.STATE_ORDER), 7.6), layout="constrained", squeeze=False)
    for row, norm in enumerate(("l2", "h1")):
        for column, field in enumerate(artifact.STATE_ORDER):
            axis = axes[row, column]
            support_axis = axis.twinx()
            for label, frame in curves.items():
                selected = frame.loc[frame["metric_id"] == f"physical_relative_{norm}_{field}"].sort_values("physical_time")
                plotted_time = display_time.values(selected["physical_time"].to_numpy(dtype=np.float64))
                style = styles[label]
                line = {"color": style.color, "linestyle": style.linestyle}
                axis.plot(plotted_time, selected["value"].to_numpy(dtype=np.float64), **line, linewidth=2.0, label=label)
                support_axis.step(plotted_time, selected["supported_cases"], where="mid", **line, alpha=0.4)
            display_time.configure(axis)
            figure_layout.set_panel_title(axis, field_labels.metric_label(f"rel. {norm.upper()}", field, mathtext=True))
            axis.grid(alpha=0.25)
            support_axis.set_ylim(bottom=0)
            if column == len(artifact.STATE_ORDER) - 1:
                support_axis.set_ylabel("Supported cases n(t)")
        axes[row, 0].set_ylabel(f"Relative {norm.upper()} [-]")
    note = plot_layout.series_note(styles, "per-case ratios, equal over cases and materials; gaps are unavailable")
    figure_layout.set_figure_title(figure, axes, "Relative norms over physical time", note=note)
    _legend(figure, axes)
    return figure


def _aggregate_snapshot_evidence(  # noqa: PLR0915 -- one bounded physical snapshot reduction
    records: Sequence[Any],
    *,
    state_field: str,
    physical_time: float,
    compute_device: str = "cpu",
    device_evidence: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool exact-time case evidence without stacking complete trajectories.

    Returns the grid coordinates, the pooled mean absolute, mean signed (``error_sum`` over the count) and signed-error standard
    deviation maps in displayed difference units, and the displayed case spatial means (see ``_spatial_means``) of reference and
    prediction. A cell any case left non-finite is NaN in all three maps.
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
    mean_error[divergent] = np.nan
    error_variance[divergent] = np.nan
    return (
        x_values,
        y_values,
        mean_absolute_error,
        mean_error,
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


def _aggregate_divisors(
    series: Mapping[str, Sequence[Any]],
    fields: Sequence[str],
    reference_scales: Mapping[str, Mapping[str, error_diagnostics.ReferenceScale | str]] | None,
) -> dict[str, float]:
    """Return each field's train sigma in displayed difference units, the one every shown context shares, else raise ``ValueError``."""
    divisors: dict[str, float] = {}
    for field in fields:
        if moisture_display.is_moisture_field(field):
            msg = f"Normalized aggregate errors are defined on the stored states, but {field!r} is shown in wet-basis percentage points."
            raise ValueError(msg)
        scales: list[error_diagnostics.ReferenceScale] = []
        for context in (name for name, current_records in series.items() if current_records):
            scale = (reference_scales or {}).get(context, {}).get(field)
            if scale is None or isinstance(scale, str):  # a missing scale or its unavailability reason, "" included
                msg = scale or f"No valid train reference scale was given for {field!r} in context {context!r}."
                raise ValueError(msg)
            if scale.field != field or scale.unit != _state_unit(field):
                msg = f"The reference scale of {field!r} in context {context!r} belongs to a different physical field or unit."
                raise ValueError(msg)
            scales.append(scale)
        common = error_diagnostics.require_common_reference_scale(scales)
        divisors[field] = float(_difference_display(field, np.asarray(common.value)))
    return divisors


def plot_aggregate_error_maps(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    physical_time: float,
    statistic: str,
    compute_device: str = "cpu",
    error_mode: str = error_diagnostics.DEFAULT_ERROR_MODE,
    reference_scales: Mapping[str, Mapping[str, error_diagnostics.ReferenceScale | str]] | None = None,
) -> Figure:
    """
    Plot exact-time mean or standard-deviation error maps on the fixed map-grid geometry: a row per field, a column per context.

    The mean map follows ``error_mode``: the signed mean of prediction minus reference (default, blue-white-red), the mean absolute
    error, or the mean absolute error over the train sigma, which needs ``reference_scales`` (context, then field) and is refused
    unless every shown context has the same one, and for water, which is shown in wet-basis percentage points. The standard
    deviation map is that of the signed error whatever the mode. Each row has one round scale locked over its contexts, and every bar
    carries its unit; an arrow stands only where that map has values beyond.
    """
    fields = _selected_fields(state_fields)
    if statistic not in {"mean", "std"}:
        msg = "Aggregate error-map statistic must be 'mean' or 'std'."
        raise ValueError(msg)
    if error_mode not in error_diagnostics.TRANSIENT_ERROR_MODES:
        msg = f"Aggregate error mode must be one of {error_diagnostics.TRANSIENT_ERROR_MODES}, not {error_mode!r}."
        raise ValueError(msg)
    series = _labelled_record_series(records, allow_empty_contexts=True)
    signed = statistic == "mean" and error_mode == error_diagnostics.SIGNED_ERROR
    normalized = statistic == "mean" and error_mode == error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    divisors = _aggregate_divisors(series, fields, reference_scales) if normalized else {}
    cells: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    routes: set[str] = set()
    for column, current_records in enumerate(series.values()):
        for row, field in enumerate(fields):
            if current_records:
                device_evidence: dict[str, str] = {}
                x_values, y_values, absolute, signed_mean, std_error, *_ = _aggregate_snapshot_evidence(
                    current_records,
                    state_field=field,
                    physical_time=physical_time,
                    compute_device=compute_device,
                    device_evidence=device_evidence,
                )
                routes.update(device_evidence.values())
                displayed = std_error if statistic == "std" else signed_mean if signed else absolute / divisors.get(field, 1.0)
                cells[row, column] = (x_values, y_values, displayed)
    extents = {(float(x.min()), float(x.max()), float(y.min()), float(y.max())) for x, y, _ in cells.values()}
    if len(extents) != 1:
        msg = "The maps of one figure share one physical extent, but the shown contexts have different grids."
        raise ValueError(msg)
    extent = next(iter(extents))
    geometry = layout.plan_geometry(len(fields), len(series), (extent[1] - extent[0], extent[3] - extent[2]), width_cap=None)
    figure, axes, slots = layout.map_grid(geometry, extent=extent)
    top_notes: list[str] = []
    for row, field in enumerate(fields):
        row_values = [values[np.isfinite(values)] for (cell_row, _), (_, _, values) in cells.items() if cell_row == row]
        available = np.concatenate(row_values)
        unit = "1" if normalized else _display_unit(field, difference=True)
        definition = _error_definition(field, error_mode if statistic == "mean" else "Absolute error", available) if available.size else None
        bar_label = f"{error_diagnostics.NORMALIZED_ERROR_SYMBOL} [{field_labels.display_unit('1')}]" if normalized else f"[{unit}]"
        exponent = 0 if definition is None else layout.error_format(definition.boundaries, signed=signed).exponent
        for column in range(len(series)):
            cell = cells.get((row, column))
            # R16: one small mean line above each mean map, "n/a" without one; the std maps have none.
            shown = None if cell is None else cell[2]
            note = "" if statistic == "std" else plot_layout.mean_error_text(shown, unit=None if normalized else unit, exponent=exponent)
            if row == 0:
                top_notes.append(note)
            elif note:
                layout.add_map_note(axes[row, column], note)
            if cell is None:
                layout.empty_cell(axes[row, column], "No case at this exact time")
                continue
            values = cell[2]
            finite = np.isfinite(values)
            if definition is None or not finite.any():
                layout.empty_cell(axes[row, column], "Prediction non-finite in every cell")
                continue
            image = axes[row, column].pcolormesh(cell[0], cell[1], np.ma.masked_invalid(values), shading="auto", **definition.pcolormesh_kwargs())
            extend = field_maps.locked_extend(values[finite], definition.boundaries)
            layout.add_field_map_colorbar(figure, image, slots[row, column], definition, label=bar_label, extend=extend)
    layout.set_column_headers(axes, list(series), top_notes)
    layout.set_row_labels(axes, [_display_symbol(field) for field in fields])
    statistic_label = f"Mean {_ERROR_HEADERS[error_mode].lower()}" if statistic == "mean" else "Standard deviation of signed error"
    total_cases = sum(len(current_records) for current_records in series.values())
    layout.set_map_title(figure, f"{statistic_label} at t={physical_time:g} h — {total_cases} case-context observations")
    plot_layout.add_scale_note(figure, shared=False)
    if routes:
        figure.text(0.5, 0.01, "; ".join(sorted(routes)), ha="center", va="bottom", fontsize=8, color="0.3")
    return figure


def plot_predicted_vs_reference_channels(
    records: Sequence[Any] | Mapping[str, Sequence[Any]],
    *,
    state_fields: Sequence[str],
    physical_time: float,
    styles: series.SeriesStyles,
) -> Figure:
    """Plot exact-time case spatial means by channel (rows, the water states side by side) and context (columns), each in its comparison style."""
    fields = _selected_fields(state_fields)
    contexts = _labelled_record_series(records, allow_empty_contexts=True)
    styles = _record_styles(contexts, styles)
    figure, raw_axes = plt.subplots(
        len(fields),
        len(contexts),
        figsize=(
            max(7.2, 5.8 * len(contexts)),
            max(5.2, 4.6 * len(fields)),
        ),
        layout="constrained",
        squeeze=False,
    )
    total_cases = 0
    for column, (context, current_records) in enumerate(contexts.items()):
        total_cases += len(current_records)
        for row, field in enumerate(fields):
            if not current_records:
                axis = raw_axes[row, column]
                axis.text(0.5, 0.5, "No case at this exact time", ha="center", va="center", transform=axis.transAxes)
                if row == 0 and context:
                    figure_layout.set_panel_title(axis, context)
                continue
            (
                _x,
                _y,
                _mean_absolute_error,
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
            style = styles[context]
            axis.scatter(
                reference_display,
                prediction_display,
                s=55,
                alpha=0.8,
                color=style.color,
                marker=style.marker or "o",
                label=context or "_nolegend_",
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
                figure_layout.set_panel_title(axis, context)
            axis.grid(alpha=0.25)
    note = plot_layout.series_note(styles, f"{total_cases} case-context observations")
    figure_layout.set_figure_title(figure, raw_axes, f"Prediction versus reference at t={physical_time:g} h", note=note)
    _legend(figure, raw_axes)
    return figure


def plot_error_distributions(
    case_frame: pd.DataFrame,
    *,
    styles: series.SeriesStyles,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
) -> Figure:
    """Show complete-rollout case-error distributions by stored channel, one box per frame and material in its comparison colour and role hatch."""
    selected = _full_autonomous_cumulative(case_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient error distributions require at least one stored channel metric."
        raise ValueError(msg)
    columns = [f"normalized_rmse_{field}" for field in fields]
    if missing := [column for column in columns if column not in selected]:
        msg = f"Transient error distributions lack {missing}."
        raise ValueError(msg)
    if not np.isfinite(selected[columns].to_numpy(dtype=np.float64)).all():
        msg = "Transient case errors must be finite."
        raise ValueError(msg)
    groups = _summary_series(selected)
    styles = _frame_styles(groups, styles)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=_GROUPED_PANEL_IN)
    axis, width = axes[0, 0], 0.8 / len(groups)
    for offset, (label, rows) in enumerate(groups.items()):
        style = styles[label]
        positions = np.arange(len(fields)) - 0.4 + width * (offset + 0.5)
        values = [rows[column].to_numpy(dtype=np.float64) for column in columns]
        boxes = axis.boxplot(values, positions=positions, widths=0.9 * width, patch_artist=True)
        for box in boxes["boxes"]:
            box.set(facecolor=style.color, hatch=style.hatch, alpha=0.75)
    axis.set_xticks(np.arange(len(fields)), [field_labels.field_label(field, mathtext=True) for field in fields])
    axis.set_ylabel(f"Normalized RMSE [{field_labels.display_unit('1')}]")
    axis.grid(axis="y", alpha=0.25)
    figure_layout.set_figure_title(figure, axes, "Full-rollout case error distributions", note=plot_layout.series_note(styles))
    figure_layout.add_legend(figure, axes, [Patch(facecolor=styles[label].color, hatch=styles[label].hatch) for label in groups], list(groups))
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
    styles: series.SeriesStyles,
) -> Figure:
    """Plot bounded trajectory-element error trends against target magnitude, one panel per field, each series in its comparison style."""
    fields = _selected_fields(state_fields)
    admitted = {label: tuple(records) for label, records in series.items()}
    if not admitted or any(not records for records in admitted.values()):
        msg = "Target-magnitude error trends require labelled non-empty record groups."
        raise ValueError(msg)
    styles = _record_styles(admitted, styles)
    figure, axes = figure_layout.panel_figure(len(fields), 1, panel_in=(7.0, 3.6))
    for row, field in enumerate(fields):
        axis, notes = axes[row, 0], []
        for label, records in admitted.items():
            centers, medians, diverging, total = _finite_error_trend(records, field)
            style = styles[label]
            axis.plot(centers, medians, color=style.color, linestyle=style.linestyle, marker=style.marker or "o", linewidth=2.0, label=label)
            if diverging:  # the medians describe finite errors; the exact count of diverging values is named, never hidden
                notes.append(f"{label} · {diverging} of {total} values non-finite, excluded")
        if notes:
            axis.text(0.02, 0.98, "\n".join(notes), va="top", fontsize="small", transform=axis.transAxes)
        axis.set_xlabel(f"|{_display_symbol(field)}| [{_display_unit(field, difference=False)}]")
        axis.set_ylabel(f"Median absolute error [{_display_unit(field, difference=True)}]")
        axis.grid(alpha=0.25)
    figure_layout.set_figure_title(
        figure, axes, "Error vs reference magnitude", note=plot_layout.series_note(styles, "all stored post-origin times; deterministic bounded bins")
    )
    _legend(figure, axes)
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
        "dataset_role",
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
    styles: series.SeriesStyles,
    state_fields: Sequence[str] = artifact.STATE_ORDER,
) -> Figure:
    """Compare complete-rollout normalized RMSE for selected stored states, one bar per frame and material in its comparison colour and role hatch."""
    selected = _full_autonomous_cumulative(summary_frame)
    fields = tuple(field for field in _selected_fields(state_fields) if field in artifact.STATE_ORDER)
    if not fields:
        msg = "Transient channel-error plots require at least one stored state."
        raise ValueError(msg)
    columns = [f"normalized_rmse_{field}" for field in fields]
    if any(column not in selected for column in columns):
        msg = "Transient channel-error plot lacks per-state normalized RMSE."
        raise ValueError(msg)
    groups = _summary_series(selected)
    styles = _frame_styles(groups, styles)
    positions = np.arange(len(columns), dtype=np.float64)
    width = 0.8 / len(groups)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=_GROUPED_PANEL_IN)
    axis = axes[0, 0]
    for offset, (label, rows) in enumerate(groups.items()):
        if len(rows) != 1:
            msg = f"Transient channel-error plot requires one summary row per frame and material; {label!r} has {len(rows)}."
            raise ValueError(msg)
        style = styles[label]
        values = rows[columns].to_numpy(dtype=np.float64)[0]
        axis.bar(positions - 0.4 + width * (offset + 0.5), values, width=width, color=style.color, hatch=style.hatch, edgecolor="0.2", label=label)
    axis.set_xticks(positions, [field_labels.field_label(field, mathtext=True) for field in fields])
    axis.set_ylabel(f"Normalized RMSE [{field_labels.display_unit('1')}]")
    axis.grid(axis="y", alpha=0.25)
    figure_layout.set_figure_title(figure, axes, "Full-rollout per-state error", note=plot_layout.series_note(styles))
    _legend(figure, axes)
    return figure


def plot_horizon_error(summary_frame: pd.DataFrame, *, styles: series.SeriesStyles) -> Figure:
    """
    Plot rollout-index and physical-time views without conflating axes.

    A frame and material keeps its comparison colour and role line style; the reduction scope is the marker (cumulative, endpoint).
    """
    required = {
        "frame",
        "material_family",
        "dataset_role",
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
    display_time = time_ui.physical_time_display(
        physical_times,
        preferred_unit="auto",
    )
    figure, grid = figure_layout.panel_figure(1, 2, panel_in=(5.2, 4.2))
    axes = grid[0]
    groups = _summary_series(selected)
    styles = _frame_styles(groups, styles)
    for series_label, rows in groups.items():
        style = styles[series_label]
        for scope, group in rows.groupby("scope", sort=False):
            ordered = group.assign(_position=group["requested_horizon"].map(horizon_positions)).sort_values("_position")
            line = {"color": style.color, "linestyle": style.linestyle, "marker": _SCOPE_MARKERS.get(str(scope), "o")}
            axes[0].plot(ordered["_position"], ordered["normalized_drying_group_macro_rmse"], **line, label=f"{series_label} — {scope}")
            axes[1].plot(display_time.values(ordered["elapsed_physical_time_median"]), ordered["normalized_drying_group_macro_rmse"], **line)
    axes[0].set_xticks(
        np.arange(len(horizons)),
        [str(value) for value in horizons],
    )
    axes[0].set_xlabel("Requested horizon [transitions]")
    display_time.configure(axes[1])
    axes[1].set_xlabel(f"Median elapsed physical time [{display_time.unit}]")
    figure_layout.label_rows_once(grid, f"Normalized Drying group macro RMSE [{field_labels.display_unit('1')}]")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure_layout.set_figure_title(figure, grid, "Error vs rollout horizon", note=plot_layout.series_note(styles))
    _legend(figure, grid)
    return figure


def plot_endpoint_vs_cumulative(summary_frame: pd.DataFrame, *, styles: series.SeriesStyles) -> Figure:
    """Compare endpoint and cumulative central error without conflating scopes: a point per mode and horizon of a frame and material, in its style."""
    required = {
        "frame",
        "material_family",
        "dataset_role",
        "mode",
        "requested_horizon",
        "scope",
        "normalized_drying_group_macro_rmse",
    }
    if not isinstance(summary_frame, pd.DataFrame) or not required.issubset(summary_frame):
        msg = "Endpoint comparison requires transient summary columns."
        raise ValueError(msg)
    groups = _summary_series(summary_frame)
    pairs = {
        label: rows.pivot_table(index=["mode", "requested_horizon"], columns="scope", values="normalized_drying_group_macro_rmse", aggfunc="first")
        for label, rows in groups.items()
    }
    pairs = {label: pivot.dropna(subset=["cumulative", "endpoint"]) for label, pivot in pairs.items() if {"cumulative", "endpoint"} <= set(pivot)}
    pairs = {label: pivot for label, pivot in pairs.items() if not pivot.empty}
    if not pairs:
        msg = "Endpoint comparison requires paired cumulative and endpoint evidence."
        raise ValueError(msg)
    styles = _frame_styles({label: groups[label] for label in pairs}, styles)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(5.5, 5.0))
    axis = axes[0, 0]
    for label, pivot in pairs.items():
        style = styles[label]
        axis.scatter(pivot["cumulative"], pivot["endpoint"], color=style.color, marker=style.marker or "o", label=label)
    maximum = max(float(np.max(pivot[["cumulative", "endpoint"]].to_numpy())) for pivot in pairs.values())
    axis.plot((0.0, maximum), (0.0, maximum), linestyle="--", color="black")
    axis.set_xlabel(f"Cumulative central error [{field_labels.display_unit('1')}]")
    axis.set_ylabel(f"Endpoint central error [{field_labels.display_unit('1')}]")
    axis.grid(alpha=0.25)
    figure_layout.set_figure_title(figure, axes, "Endpoint vs cumulative error", note=plot_layout.series_note(styles))
    _legend(figure, axes)
    return figure


def plot_target_time(records: Sequence[Any] | Mapping[str, Sequence[Any]], *, styles: series.SeriesStyles) -> Figure:
    """Plot reached target pairs by saved model, each in its comparison style by its records' identity, and retain censored/unavailable counts."""
    groups = {str(label): tuple(selected) for label, selected in (records if isinstance(records, Mapping) else {"Selected model": records}).items()}
    styles = _record_styles(groups, styles)
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
        pairs[label] = (reference, prediction)
        all_times.extend((*reference, *prediction))
        notices.append(f"{label}: paired reached {len(reference)}, censored {censored}, unavailable {unavailable}")
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(6.5, 6.0))
    axis = axes[0, 0]
    display_time = time_ui.physical_time_display(tuple(all_times) if all_times else (0.0,), preferred_unit="auto")
    for label, (reference, prediction) in pairs.items():
        if reference:
            style = styles[label]
            axis.scatter(display_time.values(reference), display_time.values(prediction), color=style.color, marker=style.marker or "o", label=label)
    if all_times:
        lower, upper = display_time.values((min(all_times), max(all_times)))
        axis.plot((lower, upper), (lower, upper), linestyle="--", color="black")
    display_time.configure(axis)
    display_time.configure(axis, dimension="y")
    axis.set_xlabel(f"Reference time to target [{display_time.unit}]")
    axis.set_ylabel(f"Predicted time to target [{display_time.unit}]")
    axis.grid(alpha=0.25)
    figure_layout.set_figure_title(figure, axes, "Time to target", note=plot_layout.series_note(styles, "\n".join(notices)))
    _legend(figure, axes)
    return figure


def plot_training_performance_vs_compute(performance: pd.DataFrame, *, lineage: comparison.LineageComparison) -> Figure:
    """
    Plot the admitted A0/B/A1 arms with a result in their actual clock, and every arm without one as a status line.

    A B or A1 without an eligible checkpoint has no rows; the admitted comparison names its
    status, budget progress and eligible points in the panel title, so A0 alone still plots. The arms are one lineage evaluated on
    its ID artifacts, so a line takes its material's fixed colour and the ID line style; its mode and horizon are the marker. The
    panel is 12 x 7 in with its legend outside right at the top frame.
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
    if matched_rows.any() and clocks != {"supervised_transitions"}:
        msg = "Matched A1/B plot requires persisted committed supervised transitions for every arm with a result."
        raise ValueError(msg)
    selected = performance.loc[performance["scope"] == "cumulative"].copy()
    if selected.empty:
        msg = "Matched-compute plot requires cumulative Evaluation evidence."
        raise ValueError(msg)
    lines = dict(tuple(selected.groupby(["material_family", "mode", "requested_horizon"], sort=False)))
    conditions = list(dict.fromkeys((mode, horizon) for _material, mode, horizon in lines))
    if len(conditions) > len(_CONDITION_MARKERS):
        msg = f"Matched-compute plots tell at most {len(_CONDITION_MARKERS)} modes and horizons apart, not {len(conditions)}."
        raise ValueError(msg)
    labels = {key: f"{_material_label(key[0])} · {key[1]} / {key[2]}" for key in lines}
    styles = series.resolve_styles([series.SeriesEntity(labels[key], "", False, "id", (str(key[0]),), "lineage") for key in lines])
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(12.0, 7.0))
    axis = axes[0, 0]
    for (material, mode, horizon), group in lines.items():
        if group["comparison_arm"].duplicated().any():
            msg = "Matched-compute series requires at most one value per arm, material, mode, and horizon."
            raise ValueError(msg)
        ordered = group.assign(_arm_order=group["comparison_arm"].map({"A0": 0, "A1": 1, "B": 2})).sort_values("_arm_order")
        style = styles[labels[material, mode, horizon]]
        axis.plot(
            ordered["training_compute"] if "training_compute" in ordered else ordered["optimizer_device_compute"],
            ordered["normalized_drying_group_macro_rmse"],
            color=style.color,
            linestyle=style.linestyle,
            marker=_CONDITION_MARKERS[conditions.index((mode, horizon))],
            label=labels[material, mode, horizon],
        )
        for row in ordered.itertuples(index=False):
            axis.annotate(
                str(row.comparison_arm),
                (row.training_compute if hasattr(row, "training_compute") else row.optimizer_device_compute, row.normalized_drying_group_macro_rmse),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize="small",
            )
    axis.set_xlabel(f"Committed post-handoff supervised transitions [{field_labels.display_unit('1')}]")
    axis.set_ylabel(f"Cumulative normalized Drying group macro RMSE [{field_labels.display_unit('1')}]")
    if unavailable:  # the panel's own status: its title, at the shared panel-title size
        figure_layout.set_panel_title(
            axis,
            "\n".join(
                f"{arm.arm}: {arm.training_status} at {100.0 * arm.budget_progress:.0f} % of its budget, "
                f"{arm.eligible_points} eligible points, no result"
                for arm in unavailable
            ),
        )
    axis.grid(alpha=0.25)
    _legend(figure, axes, outside=True)  # S:33 names this plot: legend outside right at the top frame, unlike other single panels
    return figure


def plot_pipeline_degradation(
    evidence: Sequence[comparison.AirflowDegradationMetrics],
) -> Figure:
    """Plot B/A, C/A, and C/B errors in their fixed stage colours, named in the first panel's legend (L10), while retaining unavailable C evidence."""
    values = tuple(evidence)
    if not values:
        msg = "Pipeline degradation plot requires metric evidence."
        raise ValueError(msg)
    figure, axes = figure_layout.panel_figure(len(values), 1, panel_in=(6.0, 3.6))
    for axis, item in zip(axes[:, 0], values, strict=True):
        heights = {
            "B/A": item.drying_surrogate_error,
            "C/A": np.nan if item.complete_pipeline_error is None else item.complete_pipeline_error,
            "C/B": np.nan if item.airflow_substitution_discrepancy is None else item.airflow_substitution_discrepancy,
        }
        for stage, height in heights.items():
            axis.bar(stage, height, color=_STAGE_COLOURS[stage], label=stage)
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
    _legend(figure, axes[:1, :1])  # every panel holds the same stages: one legend inside the first, also of several panels
    return figure


def plot_timing_speedups(report: timing.TransientTimingReport) -> Figure:
    """
    Plot every component-composed duration ratio as one neutral bar on a log scale, a formula without paired cases as a note.

    The bar is the median of the paired per-case ratios and its error bar spans their 10th to 90th percentile, both the timing
    report's own statistics (NumPy's linear quantile); the label names the paired count and the ratio of the summed durations.
    A ratio whose median or percentiles are not all positive and finite has no place on the log scale and is a note as well;
    without any drawn ratio the scale stays linear.
    """
    if not isinstance(report, timing.TransientTimingReport):
        msg = "Timing plot requires one validated TransientTimingReport."
        raise TypeError(msg)
    summaries = [report.speedups[name] for name in report.speedups]
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(7.0, 0.9 * len(summaries) + 1.0))
    axis = axes[0, 0]
    drawn = False
    for position, summary in enumerate(summaries):
        lower, median, upper = summary.q10, summary.median, summary.q90
        if lower is None or median is None or upper is None:
            note = "no paired case"
        elif not all(math.isfinite(value) and value > 0.0 for value in (lower, median, upper)):
            note = "no positive finite ratio"
        else:
            axis.barh(position, median, xerr=[[median - lower], [upper - median]], color=_NEUTRAL_COLOUR, capsize=4.0, error_kw={"elinewidth": 1.0})
            drawn = True
            continue
        axis.text(0.02, position, note, transform=axis.get_yaxis_transform(), va="center", fontsize="small")
    axis.set_yticks(
        range(len(summaries)),
        [
            f"{summary.reference_boundary} /\n{summary.surrogate_boundary}\n{summary.available_count}/{summary.total_count} paired"
            + ("" if summary.ratio_of_sums is None else f"; ratio of sums {summary.ratio_of_sums:.3g}")
            for summary in summaries
        ],
        fontsize="small",
    )
    axis.invert_yaxis()
    if drawn:  # every drawn median and percentile is positive and finite
        figure_layout.set_log_scale(axis, "x")
    axis.set_xlabel(f"Duration ratio, median and 10th-90th percentile of the paired cases [{field_labels.display_unit('1')}]")
    axis.grid(axis="x", which="both", alpha=0.25)
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
    styles: series.SeriesStyles,
    measure: str = "prepared_wall",
) -> Figure:
    """Compare saved models using exact paired case accuracy and a named measured clock, each model frame in its comparison style."""
    labels = {
        "prepared_wall": "Prepared rollout request wall, warmed median [s]",
        "model": "Rollout model calls only, warmed median [s]",
        "ratio": "COMSOL reported drying solution / model-call time [ratio]",
    }
    if measure not in labels or "frame" not in accuracy:
        msg = "Accuracy/runtime comparison requires a named clock and frame identity."
        raise ValueError(msg)
    frames = {frame_name: accuracy.loc[accuracy["frame"] == frame_name] for frame_name in reports}
    styles = _frame_styles(frames, styles)
    figure, axes = figure_layout.panel_figure(1, 1, panel_in=(8.0, 5.5))
    axis = axes[0, 0]
    notices = []
    for frame_name, report in reports.items():
        values = _case_accuracy_values(frames[frame_name], label=f"Accuracy/runtime comparison {frame_name}")
        durations: Mapping[str, float]
        if measure == "ratio":
            durations = {case.case_id: case.speedup for case in report.speedups["drying_only_solver_speedup"].cases if case.speedup is not None}
        else:
            component = "drying_no_prepared_rollout_wall_seconds" if measure == "prepared_wall" else "drying_no_rollout_model_seconds"
            durations = timing.component_case_medians(report, component)
        paired = [(durations[case_id], value) for case_id, value in values.items() if case_id in durations]
        if paired:
            style = styles[frame_name]
            axis.scatter(
                [point[0] for point in paired],
                [point[1] for point in paired],
                color=style.color,
                marker=style.marker or "o",
                label=f"{frame_name} ({len(paired)} cases)",
            )
        notices.append(f"{frame_name}: {len(paired)}/{len(values)} paired")
    axis.set_xlabel(labels[measure])
    axis.set_ylabel(f"Cumulative full-rollout normalized Drying group macro RMSE [{field_labels.display_unit('1')}]")
    if not axis.collections:
        axis.text(0.5, 0.5, "No paired saved accuracy/runtime evidence", transform=axis.transAxes, ha="center")
    axis.grid(alpha=0.25)
    figure_layout.set_figure_title(figure, axes, "Accuracy vs measured runtime", note=plot_layout.series_note(styles, "\n".join(notices)))
    _legend(figure, axes)
    return figure
