"""
eda_capabilities.py

Resolve generated-output dataset capabilities and scientific display fields.

Responsibilities:
  - Read stored and derived field availability from authoritative frame metadata
  - Order the union of selected-dataset fields by shared scientific semantics
  - Identify the compatible dataset subset for each requested field
  - Extract direct or nested spatial values without fabricating missing data

Design principles:
  - Dataset and task schemas remain the source of field names, roles, and units
  - Mixed selections preserve unavailable combinations as explicit omissions
  - Transient state extraction uses retained stored positions without resampling

This module does NOT:
  - Discover datasets, construct widgets, or choose presentation views
  - Fill missing fields, align trajectories, or reinterpret another channel
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from src import domain, generation
from src.analysis.presentation import analysis_channel_semantics as channel_semantics
from src.analysis.presentation import analysis_field_labels as field_labels

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

_SPATIAL_DIMENSIONS = 2
_DIMENSIONLESS_REPRESENTATIONS = frozenset(
    {
        "dimensionless_log10_ratio_to_1_m2",
        "dimensionless_cross_component_ratio_to_geometric_mean",
    }
)

FieldView = Literal[
    "field_statistics",
    "spatial_map",
    "spectral",
    "state_snapshot",
    "state_trajectory",
]

_SUPPORTED_VIEWS = frozenset(
    {
        "field_statistics",
        "spatial_map",
        "spectral",
        "state_snapshot",
        "state_trajectory",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedFieldSelection:
    """Hold ordered fields and exact compatible/omitted dataset labels."""

    fields: tuple[str, ...]
    datasets_by_field: Mapping[str, tuple[str, ...]]
    omitted_by_field: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        """Freeze defensive copies and require complete field mappings."""
        if not self.fields or len(self.fields) != len(set(self.fields)):
            message = "Resolved EDA fields must be a unique non-empty sequence."
            raise ValueError(message)
        compatible = {field: tuple(self.datasets_by_field[field]) for field in self.fields}
        omitted = {field: tuple(self.omitted_by_field[field]) for field in self.fields}
        if any(not compatible[field] for field in self.fields):
            message = "Every resolved EDA field requires one compatible dataset."
            raise ValueError(message)
        object.__setattr__(self, "datasets_by_field", MappingProxyType(compatible))
        object.__setattr__(self, "omitted_by_field", MappingProxyType(omitted))


def is_transient_frame(frame: pd.DataFrame) -> bool:
    """Return whether one frame exposes the authoritative transient field schema."""
    categories = frame.attrs.get("field_categories")
    return isinstance(categories, Mapping) and "dynamic_state" in categories


def available_fields(
    frame: pd.DataFrame,
    *,
    view: FieldView,
) -> tuple[str, ...]:
    """
    Return one frame's fields available to a maintained plot family.

    Parameters
    ----------
    frame : pandas.DataFrame
        Non-empty task-native EDA frame with field metadata.
    view : FieldView
        Scientific plot family whose stored/derived field roles are requested.

    Returns
    -------
    tuple[str, ...]
        Available non-coordinate fields in canonical semantic order.

    """
    _validate_view(view)
    if frame.empty:
        message = "EDA field discovery requires a non-empty frame."
        raise ValueError(message)
    if is_transient_frame(frame):
        fields = _transient_fields(frame, view=view)
    elif view in {"state_snapshot", "state_trajectory"}:
        fields = ()
    else:
        fields = _direct_fields(frame)
    if not fields:
        return ()
    metadata = {name: _field_presentation_metadata(frame, name) for name in fields}
    return channel_semantics.ordered_channels(fields, metadata=metadata)


def resolve_fields(
    datasets: Mapping[str, pd.DataFrame],
    *,
    view: FieldView,
    requested: Sequence[str] | None = None,
) -> ResolvedFieldSelection:
    """
    Resolve the selected-dataset field union and per-field compatible subsets.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Labelled selected frames in deterministic presentation order.
    view : FieldView
        Scientific plot family used for field discovery.
    requested : Sequence[str] | None, optional
        Explicit field selection. Omission selects the complete capability union.

    Returns
    -------
    ResolvedFieldSelection
        Canonically ordered fields plus exact compatible and omitted labels.

    Raises
    ------
    ValueError
        If no dataset/field is available or a requested field is unsupported by
        every selected dataset.

    """
    _validate_view(view)
    if not datasets:
        message = "EDA field resolution requires at least one selected dataset."
        raise ValueError(message)
    labels = tuple(datasets)
    if any(not isinstance(label, str) or not label for label in labels):
        message = "EDA dataset labels must be non-empty text."
        raise ValueError(message)
    fields_by_dataset = {label: available_fields(frame, view=view) for label, frame in datasets.items()}
    metadata: dict[str, channel_semantics.ChannelPresentationMetadata] = {}
    declared_order: list[str] = []
    for label, frame in datasets.items():
        for field in fields_by_dataset[label]:
            resolved = _field_presentation_metadata(frame, field)
            previous = metadata.setdefault(field, resolved)
            if previous.category != resolved.category:
                message = f"EDA field {field!r} has conflicting presentation roles across selected datasets."
                raise ValueError(message)
            if field not in declared_order:
                declared_order.append(field)
    union = channel_semantics.ordered_channels(declared_order, metadata=metadata)
    if not union:
        message = "Selected datasets expose no fields for this EDA view."
        raise ValueError(message)
    fields = _resolve_requested_fields(union, requested=requested, metadata=metadata)
    compatible = {field: tuple(label for label in labels if field in fields_by_dataset[label]) for field in fields}
    omitted = {field: tuple(label for label in labels if field not in fields_by_dataset[label]) for field in fields}
    return ResolvedFieldSelection(
        fields=fields,
        datasets_by_field=compatible,
        omitted_by_field=omitted,
    )


def case_row(frame: pd.DataFrame, case_id: str) -> pd.Series:
    """Return one exact selected case row without positional reinterpretation."""
    matches = np.flatnonzero(frame.index.astype(str).to_numpy() == str(case_id))
    if matches.size != 1:
        message = f"EDA frame has no unique case {case_id!r}."
        raise KeyError(message)
    return frame.iloc[int(matches[0])]


def spatial_coordinates(
    frame: pd.DataFrame,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return authoritative direct or nested Cartesian coordinates."""
    if is_transient_frame(frame):
        static = row.get("static_fields")
        if not isinstance(static, Mapping):
            message = "Transient EDA coordinates require static_fields evidence."
            raise TypeError(message)
        x_value = static.get("x")
        y_value = static.get("y")
    else:
        x_value = row.get("x")
        y_value = row.get("y")
    x = np.asarray(x_value, dtype=float)
    y = np.asarray(y_value, dtype=float)
    if x.shape != y.shape or x.ndim != _SPATIAL_DIMENSIONS or not np.isfinite(x).all() or not np.isfinite(y).all():
        message = "EDA spatial coordinates must be aligned finite two-dimensional arrays."
        raise ValueError(message)
    return x, y


def inlet_pressure_boundary(
    frame: pd.DataFrame,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the authoritative one-dimensional inlet-pressure profile."""
    x_coordinates = np.asarray(field_values(frame, row, "x"), dtype=float)
    pressure = np.asarray(field_values(frame, row, "p_in_bc"), dtype=float)
    if (
        x_coordinates.shape != pressure.shape
        or pressure.ndim != _SPATIAL_DIMENSIONS
        or not np.isfinite(x_coordinates).all()
        or not np.isfinite(pressure).all()
    ):
        message = "Inlet pressure requires matching finite two-dimensional coordinate and boundary fields."
        raise ValueError(message)
    return (
        np.ascontiguousarray(x_coordinates[0, :]),
        np.ascontiguousarray(pressure[0, :]),
    )


def is_dynamic_field(frame: pd.DataFrame, field: str) -> bool:
    """Return whether one field is stored on the transient time dimension."""
    if not is_transient_frame(frame):
        return False
    categories = cast("Mapping[str, object]", frame.attrs["field_categories"])
    dynamic = _text_tuple(categories.get("dynamic_state"), label="dynamic_state")
    return field in dynamic


def field_values_at_physical_time(
    frame: pd.DataFrame,
    row: pd.Series,
    field: str,
    physical_time_hours: float,
) -> np.ndarray:
    """Return one exact stored transient state or time-invariant spatial field."""
    if not is_dynamic_field(frame, field):
        return field_values(frame, row, field)
    requested = float(physical_time_hours)
    if not np.isfinite(requested):
        message = "EDA snapshot physical time must be finite."
        raise ValueError(message)
    time = row.get("time")
    if not isinstance(time, Mapping):
        message = "Transient EDA rows require time evidence."
        raise TypeError(message)
    regular = np.asarray(time.get("regular_state_hours"), dtype=float)
    tolerance = float(time.get("classification_tolerance_hours", 0.0))
    matches = np.flatnonzero(np.isclose(regular, requested, rtol=0.0, atol=tolerance))
    if matches.size == 1:
        return field_values(
            frame,
            row,
            field,
            transient_state_index=int(matches[0]),
        )
    if matches.size > 1:
        message = "Transient stored-time classification is ambiguous."
        raise ValueError(message)
    exact_stop = row.get("exact_stop")
    if isinstance(exact_stop, Mapping):
        exact_time = exact_stop.get("time_hours")
        state = exact_stop.get("state")
        if (
            isinstance(exact_time, (int, float))
            and isinstance(state, Mapping)
            and np.isclose(
                float(exact_time),
                requested,
                rtol=0.0,
                atol=tolerance,
            )
            and field in state
        ):
            return np.asarray(state[field], dtype=float)
    message = f"Physical time {requested:g} h is unavailable for field {field!r}."
    raise ValueError(message)


def compatible_frames(
    datasets: Mapping[str, pd.DataFrame],
    resolution: ResolvedFieldSelection,
    field: str,
) -> dict[str, pd.DataFrame]:
    """Return selected frames that authoritatively provide one resolved field."""
    if field not in resolution.fields:
        message = f"EDA field {field!r} is not part of the resolved selection."
        raise ValueError(message)
    compatible = set(resolution.datasets_by_field[field])
    return {label: frame for label, frame in datasets.items() if label in compatible}


def availability_note(resolution: ResolvedFieldSelection) -> str:
    """Return one concise note for omitted dataset/field combinations."""
    omitted = tuple(f"{field}: {', '.join(labels)}" for field in resolution.fields if (labels := resolution.omitted_by_field[field]))
    if not omitted:
        return ""
    return "Unavailable dataset/field combinations omitted — " + "; ".join(omitted) + "."


def field_unit(frame: pd.DataFrame, field: str) -> str:
    """Return one authoritative physical unit, deriving magnitude from velocity."""
    units = frame.attrs.get("field_units")
    if not isinstance(units, Mapping):
        message = "EDA field-unit metadata must be a mapping."
        raise TypeError(message)
    source = "u" if field == "U" else field
    unit = units.get(source)
    if not isinstance(unit, str) or not unit:
        message = f"EDA field {field!r} has no authoritative unit metadata."
        raise ValueError(message)
    return unit


def field_representation(frame: pd.DataFrame, field: str) -> str:
    """Return one stored or explicit derived field representation."""
    if field == "U":
        return "derived_speed_magnitude"
    representations = frame.attrs.get("field_representations")
    value = representations.get(field) if isinstance(representations, Mapping) else None
    if value is None and is_transient_frame(frame):
        return "identity"
    if not isinstance(value, str) or not value:
        message = f"EDA field {field!r} has no authoritative representation metadata."
        raise ValueError(message)
    return value


def field_display_unit(frame: pd.DataFrame, field: str) -> str:
    """Return the unit of the values actually displayed for one field."""
    if field_representation(frame, field) in _DIMENSIONLESS_REPRESENTATIONS:
        return "1"
    return field_unit(frame, field)


def field_display_values(
    frame: pd.DataFrame,
    field: str,
    values: Sequence[float] | np.ndarray,
    *,
    quantity_kind: field_labels.TemperatureQuantityKind = "absolute",
) -> np.ndarray:
    """Return a non-mutating display array on the field's presentation scale."""
    if field_representation(frame, field) in _DIMENSIONLESS_REPRESENTATIONS:
        return np.array(values, dtype=np.float64, copy=True)
    return field_labels.display_values(
        values,
        field_unit(frame, field),
        quantity_kind=quantity_kind,
    )


def field_quantity_label(
    frame: pd.DataFrame,
    field: str,
    *,
    mathtext: bool = False,
) -> str:
    """Return one shared formula-or-fallback label with display unit."""
    return field_labels.field_label_with_unit(
        field,
        field_display_unit(frame, field),
        mathtext=mathtext,
    )


def resolved_field_labels(
    datasets: Mapping[str, pd.DataFrame],
    resolution: ResolvedFieldSelection,
    *,
    mathtext: bool = False,
) -> dict[str, str]:
    """Return labels for resolved fields from their first compatible frame."""
    labels: dict[str, str] = {}
    for field in resolution.fields:
        compatible = set(resolution.datasets_by_field[field])
        reference = next(frame for label, frame in datasets.items() if label in compatible)
        labels[field] = field_quantity_label(
            reference,
            field,
            mathtext=mathtext,
        )
    return labels


def field_values(
    frame: pd.DataFrame,
    row: pd.Series,
    field: str,
    *,
    transient_state_index: int | None = None,
) -> np.ndarray:
    """
    Return one direct, static, derived, or stored transient field array.

    ``transient_state_index=None`` retains the complete stored trajectory for a
    dynamic field. Supplying an index selects that exact stored position only.
    No interpolation or resampling is performed.
    """
    if not is_transient_frame(frame):
        if field not in row:
            message = f"Direct EDA field {field!r} is unavailable in one case."
            raise ValueError(message)
        return np.asarray(row[field], dtype=float)
    states = row.get("state_trajectories")
    static = row.get("static_fields")
    if not isinstance(states, Mapping) or not isinstance(static, Mapping):
        message = "Transient EDA rows require state_trajectories and static_fields mappings."
        raise TypeError(message)
    if field in states:
        values = np.asarray(states[field], dtype=float)
        if transient_state_index is None:
            return values
        if isinstance(transient_state_index, bool) or not isinstance(transient_state_index, int):
            message = "Transient stored-state indices must be integers."
            raise TypeError(message)
        if not 0 <= transient_state_index < values.shape[0]:
            message = f"Transient stored-state index {transient_state_index} is unavailable for {field!r}."
            raise ValueError(message)
        return values[transient_state_index]
    if field == "U":
        if "u" not in static or "v" not in static:
            message = "Derived speed magnitude requires retained static u and v fields."
            raise ValueError(message)
        return np.hypot(
            np.asarray(static["u"], dtype=float),
            np.asarray(static["v"], dtype=float),
        )
    if field in static:
        return np.asarray(static[field], dtype=float)
    message = f"Transient EDA field {field!r} is unavailable in one case."
    raise ValueError(message)


def _validate_view(view: str) -> None:
    """Require one maintained field-view identifier."""
    if view not in _SUPPORTED_VIEWS:
        message = f"Unsupported EDA field view {view!r}."
        raise ValueError(message)


def _direct_fields(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return task-declared non-coordinate direct two-dimensional fields."""
    declared = frame.attrs.get("field_names")
    roles = frame.attrs.get("field_roles")
    if not isinstance(declared, (list, tuple)) or not declared:
        message = "Direct EDA frames require ordered field_names metadata."
        raise TypeError(message)
    if not isinstance(roles, Mapping):
        message = "Direct EDA frames require field_roles metadata."
        raise TypeError(message)
    sample = frame.iloc[0]
    fields = []
    for raw_name in declared:
        if not isinstance(raw_name, str) or not raw_name:
            message = "EDA field names must be non-empty text."
            raise ValueError(message)
        if roles.get(raw_name) == "coordinate" or raw_name not in frame.columns:
            continue
        if np.asarray(sample[raw_name]).ndim >= _SPATIAL_DIMENSIONS:
            fields.append(raw_name)
    return tuple(fields)


def _transient_fields(
    frame: pd.DataFrame,
    *,
    view: FieldView,
) -> tuple[str, ...]:
    """Return metadata-declared nested fields for one transient plot family."""
    categories = frame.attrs.get("field_categories")
    if not isinstance(categories, Mapping):
        message = "Transient EDA frames require field_categories metadata."
        raise TypeError(message)
    dynamic = _text_tuple(categories.get("dynamic_state"), label="dynamic_state")
    static_declared = _text_tuple(categories.get("static_spatial"), label="static_spatial")
    row = frame.iloc[0]
    states = row.get("state_trajectories")
    static = row.get("static_fields")
    if not isinstance(states, Mapping) or not isinstance(static, Mapping):
        message = "Transient EDA frames require nested state and static mappings."
        raise TypeError(message)
    if any(name not in states for name in dynamic):
        message = "Transient dynamic-state metadata disagrees with retained fields."
        raise ValueError(message)
    static_fields = tuple(name for name in static_declared if name not in {"x", "y"})
    if any(name not in static for name in static_fields):
        message = "Transient static-field metadata disagrees with retained fields."
        raise ValueError(message)
    derived = ("U",) if {"u", "v"}.issubset(static) else ()
    if view == "state_trajectory":
        return dynamic
    return (*static_fields, *derived, *dynamic)


def _text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    """Normalize one authoritative metadata field-name collection."""
    if not isinstance(value, (list, tuple)) or any(not isinstance(name, str) or not name for name in value):
        message = f"Transient EDA {label} metadata must contain field names."
        raise TypeError(message)
    return cast("tuple[str, ...]", tuple(value))


def _authoritative_field_metadata() -> dict[str, channel_semantics.ChannelPresentationMetadata]:
    """Return schema-derived four-group metadata for generated-output fields."""
    airflow_task = domain.tasks.registry.get_task("steady_flow")
    metadata: dict[str, channel_semantics.ChannelPresentationMetadata] = {}
    airflow_inputs = tuple(field.name for field in airflow_task.inputs if field.role != "coordinate")
    for order, name in enumerate(airflow_inputs):
        metadata[name] = channel_semantics.ChannelPresentationMetadata(
            "airflow_input",
            order,
        )
    airflow_outputs = tuple(field.name for field in airflow_task.outputs)
    for order, name in enumerate((*airflow_outputs, "U")):
        metadata[name] = channel_semantics.ChannelPresentationMetadata(
            "airflow_output",
            order,
        )

    profile = generation.contracts.get_profile_contract("transient_drying")
    reserved = set(metadata)
    transient_inputs = tuple(field.name for field in profile.static_fields if field.name not in reserved)
    for order, name in enumerate(transient_inputs):
        metadata[name] = channel_semantics.ChannelPresentationMetadata(
            "transient_input",
            order,
        )
    for order, field in enumerate(profile.transient_fields):
        metadata[field.name] = channel_semantics.ChannelPresentationMetadata(
            "transient_output",
            order,
        )
    for order, field in enumerate(profile.coordinate_fields):
        metadata[field.name] = channel_semantics.ChannelPresentationMetadata(
            "coordinate",
            order,
        )
    return metadata


_FIELD_METADATA = _authoritative_field_metadata()


def field_group(frame: pd.DataFrame, field: str) -> channel_semantics.ChannelCategory:
    """Return one authoritative four-group presentation category."""
    return _field_presentation_metadata(frame, field).category


def _field_presentation_metadata(
    frame: pd.DataFrame,
    field: str,
) -> channel_semantics.ChannelPresentationMetadata:
    """Translate task/schema declarations into shared four-group ordering."""
    declared = _FIELD_METADATA.get(field)
    if declared is not None:
        return declared
    roles = frame.attrs.get("field_roles")
    role = roles.get(field) if isinstance(roles, Mapping) else None
    declared_order = tuple(roles) if isinstance(roles, Mapping) else ()
    order = declared_order.index(field) if field in declared_order else len(declared_order)
    if role == "coordinate":
        category: channel_semantics.ChannelCategory = "coordinate"
    elif is_transient_frame(frame) and role == "dynamic_state":
        category = "transient_output"
    elif is_transient_frame(frame):
        category = "transient_input"
    elif role in {"state", "derived_speed"}:
        category = "airflow_output"
    else:
        category = "airflow_input"
    return channel_semantics.ChannelPresentationMetadata(category, order)


def _resolve_requested_fields(
    available: tuple[str, ...],
    *,
    requested: Sequence[str] | None,
    metadata: Mapping[str, channel_semantics.ChannelPresentationMetadata],
) -> tuple[str, ...]:
    """Validate and semantically order one optional explicit field selection."""
    if requested is None:
        return available
    if isinstance(requested, str):
        message = "EDA field selection must be a sequence, not one string."
        raise TypeError(message)
    selected = tuple(requested)
    if not selected or len(selected) != len(set(selected)) or any(not isinstance(field, str) or not field for field in selected):
        message = "EDA field selection must contain unique non-empty names."
        raise ValueError(message)
    unknown = tuple(field for field in selected if field not in available)
    if unknown:
        message = f"Selected EDA fields are unavailable in every selected dataset: {unknown!r}."
        raise ValueError(message)
    return channel_semantics.ordered_channels(selected, metadata=metadata)
