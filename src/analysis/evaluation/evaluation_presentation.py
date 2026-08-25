"""
evaluation_presentation.py

Expose small, current-native presentation data shared by evaluation views.

Responsibilities:
  - Map canonical TaskSpec fields and vector groups to concise display fields
  - Derive case-level display arrays without changing canonical case identity
  - Build regular plotting coordinates from admitted shape and physical extent
  - Select and label supported scientific metadata parameters consistently
  - Select a deterministic parameter-centre reference case

Design principles:
  - Canonical artifact fields, groups, units, and membership remain authoritative
  - Derived magnitudes are explicit TaskSpec group views, never stored aliases
  - Metadata filtering is shared and exact rather than plot-local and heuristic
  - Plotting coordinates affect display geometry only, never scientific values

This module does NOT:
  - Load models, infer, generate, repair, mutate, or admit artifacts
  - Create an alternate dataframe or compatibility API
  - Own public view composition or plot-specific widget behavior
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from src.analysis.presentation.analysis_field_labels import (
    VELOCITY_MAGNITUDE_LABEL,
    VELOCITY_MAGNITUDE_MATHTEXT,
    display_unit,
    field_label,
    has_declared_field_metadata,
)

from . import evaluation_dataframe as dataframe
from . import evaluation_transient_artifact as transient_artifact

if TYPE_CHECKING:
    from .evaluation_case import EvaluationCase

_FIELD_LABELS: Mapping[str, str] = {
    "p": "p",
    "u": "u",
    "v": "v",
    "pressure": "p",
    "velocity_x": "u",
    "velocity_y": "v",
}
_GROUP_LABELS: Mapping[str, str] = {"velocity": VELOCITY_MAGNITUDE_LABEL}
_METADATA_PREFIXES = ("geometry_", "parameters_")
_MINIMUM_MAGNITUDE_COMPONENTS = 2
PARAMETER_VARIATION_ATOL = 1e-12
PARAMETER_VARIATION_RTOL = 1e-9
_MINIMUM_VARIATION_VALUES = 2


@dataclass(frozen=True, slots=True)
class ParameterPresentation:
    """Describe one sampled input in canonical source-generation order."""

    column: str
    label: str
    group: str
    order: int
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class ParameterSelection:
    """Return included parameters and internally useful omission reasons."""

    included: tuple[str, ...]
    omitted: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TransientDisplayField:
    """Describe one stored or explicitly supported derived transient channel."""

    key: str
    label: str
    unit: str
    stored: bool
    dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransientChannelResolution:
    """Return comparable transient channels and concise omission evidence."""

    fields: tuple[TransientDisplayField, ...]
    omitted: Mapping[str, str]

    @property
    def keys(self) -> tuple[str, ...]:
        """Return canonical channel keys in visible order."""
        return tuple(field.key for field in self.fields)

    @property
    def labels(self) -> Mapping[str, str]:
        """Return key-to-label widget text with authoritative displayed units."""
        return {field.key: f"{field.label} [{display_unit(field.unit)}]" for field in self.fields}


_PARAMETER_PRESENTATION = (
    # gen_structure_field: correlated background, then multiscale combination.
    ParameterPresentation("parameters_bed.structure.coarse_len_rel", "Coarse correlation length", "Spatial background", 10),
    ParameterPresentation("parameters_bed.structure.fine_len_rel", "Fine correlation length", "Spatial background", 20),
    ParameterPresentation("parameters_bed.structure.coarse_weight", "Coarse-scale weight", "Spatial background", 30),
    ParameterPresentation("parameters_bed.structure.fine_weight", "Fine-scale weight", "Spatial background", 40),
    ParameterPresentation("parameters_bed.structure.fine_ani_x", "Fine anisotropy x", "Spatial background", 50),
    ParameterPresentation("parameters_bed.structure.fine_ani_y", "Fine anisotropy y", "Spatial background", 60),
    ParameterPresentation("parameters_bed.structure.cross_scale_corr", "Cross-scale correlation", "Spatial background", 70),
    # Localized structures are applied after the background field.
    ParameterPresentation("parameters_bed.perturbations.amplitude", "Perturbation amplitude", "Structures and texture", 110),
    ParameterPresentation("parameters_bed.perturbations.granularity", "Perturbation granularity", "Structures and texture", 120),
    ParameterPresentation("parameters_bed.perturbations.sign_bias", "Perturbation sign bias", "Structures and texture", 130),
    # Permeability mapping/tensor construction precedes porosity construction.
    ParameterPresentation("parameters_kappa_mean", "Mean permeability", "Material and properties", 210, "m²"),
    ParameterPresentation("parameters_kappa_cv", "Relative permeability variation", "Material and properties", 220),
    ParameterPresentation("parameters_permeability.anisotropy.max_ratio", "Maximum anisotropy ratio", "Material and properties", 230),
    ParameterPresentation("parameters_permeability.anisotropy.exponent", "Anisotropy exponent", "Material and properties", 240),
    ParameterPresentation("parameters_permeability.anisotropy.strength", "Tensor strength", "Material and properties", 250),
    ParameterPresentation("parameters_permeability.orientation.smooth_len_rel", "Orientation smoothing length", "Material and properties", 260),
    ParameterPresentation("parameters_permeability.orientation.jitter", "Orientation jitter", "Material and properties", 270),
    ParameterPresentation("parameters_porosity.smooth_len_rel", "Porosity smoothing length", "Material and properties", 280),
    ParameterPresentation("parameters_porosity.texture_amp", "Porosity texture amplitude", "Material and properties", 290),
    # gen_pressure_bc builds shape terms in this order, then applies pressure_bc.mean.
    ParameterPresentation("parameters_pressure_bc.sin_amp", "Inlet sinusoid amplitude", "Boundary conditions", 410),
    ParameterPresentation("parameters_pressure_bc.sin_freq", "Inlet sinusoid frequency", "Boundary conditions", 420),
    ParameterPresentation("parameters_pressure_bc.sin_phase", "Inlet sinusoid phase", "Boundary conditions", 430, "rad"),
    ParameterPresentation("parameters_pressure_bc.gauss_count", "Inlet Gaussian count", "Boundary conditions", 440),
    ParameterPresentation("parameters_pressure_bc.gauss_amp", "Inlet Gaussian amplitude", "Boundary conditions", 450),
    ParameterPresentation("parameters_pressure_bc.gauss_width", "Inlet Gaussian width", "Boundary conditions", 460),
    ParameterPresentation("parameters_pressure_bc.gauss_jitter", "Inlet Gaussian jitter", "Boundary conditions", 470),
    ParameterPresentation("parameters_pressure_bc.linear_amp", "Inlet linear-gradient amplitude", "Boundary conditions", 480),
    ParameterPresentation("parameters_pressure_bc.mean", "Mean inlet pressure", "Boundary conditions", 490, "Pa"),
)
_PARAMETER_BY_COLUMN = {spec.column: spec for spec in _PARAMETER_PRESENTATION}
_STRUCTURAL_METADATA_LEAVES = frozenset({"dx", "dy", "lx", "ly", "nx", "ny", "res"})


@dataclass(frozen=True, slots=True)
class DisplayField:
    """Describe one canonical learned field or TaskSpec vector magnitude."""

    key: str
    label: str
    unit: str
    component_fields: tuple[str, ...]
    metric_column: str

    @property
    def is_magnitude(self) -> bool:
        """Return whether this display field is a vector-group magnitude."""
        return len(self.component_fields) > 1

    @property
    def matplotlib_label(self) -> str:
        """Return the canonical Matplotlib label for this display field."""
        if self.is_magnitude and self.key == "velocity_magnitude":
            return VELOCITY_MAGNITUDE_MATHTEXT
        return self.label


def _field_units(frame: pd.DataFrame) -> dict[str, str]:
    """Return the validated canonical output-field unit mapping."""
    return dataframe.field_units(frame)


def display_fields(frame: pd.DataFrame) -> tuple[DisplayField, ...]:
    """Return learned outputs followed by supported TaskSpec group magnitudes."""
    units = _field_units(frame)
    fields = tuple(frame.attrs["output_fields"])
    result = [
        DisplayField(
            key=field,
            label=_FIELD_LABELS.get(field, field),
            unit=units[field],
            component_fields=(field,),
            metric_column=f"normalized_rmse_{field}",
        )
        for field in fields
    ]
    raw_groups = frame.attrs.get("output_groups", ())
    for group_id, group_fields_value in raw_groups:
        group_fields = tuple(group_fields_value)
        if group_id not in _GROUP_LABELS or len(group_fields) < _MINIMUM_MAGNITUDE_COMPONENTS:
            continue
        if any(field not in units for field in group_fields):
            message = f"Output group {group_id!r} contains an undeclared learned field."
            raise dataframe.ComparisonCompatibilityError(message)
        group_units = {units[field] for field in group_fields}
        if len(group_units) != 1:
            message = f"Output group {group_id!r} cannot form a magnitude across unlike physical units."
            raise dataframe.ComparisonCompatibilityError(message)
        metric_column = f"normalized_{group_id}_vector_rmse"
        if metric_column not in frame.columns:
            continue
        result.append(
            DisplayField(
                key=f"{group_id}_magnitude",
                label=_GROUP_LABELS[group_id],
                unit=next(iter(group_units)),
                component_fields=group_fields,
                metric_column=metric_column,
            )
        )
    return tuple(result)


def shared_display_fields(frames: Sequence[pd.DataFrame]) -> tuple[DisplayField, ...]:
    """Return identical display-field semantics shared by every supplied frame."""
    if not frames:
        msg = "At least one evaluation frame is required."
        raise ValueError(msg)
    first = display_fields(frames[0])
    signature = tuple((field.key, field.label, field.unit, field.component_fields) for field in first)
    for frame in frames[1:]:
        current = display_fields(frame)
        if tuple((field.key, field.label, field.unit, field.component_fields) for field in current) != signature:
            msg = "Compared artifacts expose different display-field semantics."
            raise dataframe.ComparisonCompatibilityError(msg)
    return first


def case_field(
    case: EvaluationCase,
    field: DisplayField,
    *,
    source: Literal["reference", "prediction"],
) -> np.ndarray:
    """Return one learned case field or its declared vector-group magnitude."""
    values = case.reference if source == "reference" else case.prediction
    try:
        indices = tuple(case.fields.index(component) for component in field.component_fields)
    except ValueError as error:
        message = f"Case does not expose every component of display field {field.key!r}."
        raise dataframe.ComparisonCompatibilityError(message) from error
    if len(indices) == 1:
        return values[indices[0]]
    return np.sqrt(np.sum(values[np.asarray(indices)] ** 2, axis=0))


def case_error(
    case: EvaluationCase,
    field: DisplayField,
) -> np.ndarray:
    """Return prediction minus reference for one canonical display field."""
    return case_field(case, field, source="prediction") - case_field(case, field, source="reference")


def display_grid(case: EvaluationCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equally spaced 1-D and mesh coordinates from admitted extent/shape."""
    height, width = case.shape
    x_values = np.asarray(case.coordinates[0], dtype=float)
    y_values = np.asarray(case.coordinates[1], dtype=float)
    x = np.linspace(float(np.min(x_values)), float(np.max(x_values)), width)
    y = np.linspace(float(np.min(y_values)), float(np.max(y_values)), height)
    x_grid, y_grid = np.meshgrid(x, y)
    return x, y, x_grid, y_grid


def effectively_constant(values: np.ndarray) -> tuple[bool, float]:
    """Return near-constant status and the shared scale-aware tolerance."""
    numeric = np.asarray(values, dtype=float)
    if numeric.size < _MINIMUM_VARIATION_VALUES:
        return True, PARAMETER_VARIATION_ATOL
    if not np.isfinite(numeric).all():
        msg = "Variation checks require finite numeric values."
        raise ValueError(msg)
    scale = max(float(np.max(np.abs(numeric))), 1.0)
    tolerance = PARAMETER_VARIATION_ATOL + PARAMETER_VARIATION_RTOL * scale
    return float(np.ptp(numeric)) <= tolerance, tolerance


def _finite_variable(frame: pd.DataFrame, column: str, *, max_cases: int | None) -> tuple[bool, str]:
    """Apply the shared finite and near-constant policy to one admitted prefix."""
    selected = frame if max_cases is None else frame.iloc[:max_cases]
    values = np.asarray(pd.to_numeric(selected[column], errors="raise"), dtype=float)
    if values.size < _MINIMUM_VARIATION_VALUES:
        return False, "fewer than two admitted finite values"
    if not np.isfinite(values).all():
        return False, "contains non-finite values"
    constant, tolerance = effectively_constant(values)
    if constant:
        return False, f"constant within tolerance {tolerance:.3g}"
    return True, ""


def metadata_parameter_selection(
    frames: Sequence[pd.DataFrame],
    *,
    max_cases: int | None = None,
) -> ParameterSelection:
    """Filter and order parameters through one source-generation contract."""
    if not frames:
        msg = "At least one evaluation frame is required."
        raise ValueError(msg)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0):
        msg = "max_cases must be a positive integer when supplied."
        raise ValueError(msg)
    candidates = tuple(dataframe.numeric_metadata_columns(frames[0]))
    shared = set(candidates)
    for frame in frames[1:]:
        shared.intersection_update(dataframe.numeric_metadata_columns(frame))
    omitted: dict[str, str] = {}
    included: list[str] = []
    for column in candidates:
        if column not in shared:
            omitted[column] = "not shared by every displayed dataset"
            continue
        lowered = column.lower()
        leaf = lowered.removeprefix("geometry_").removeprefix("parameters_")
        if lowered.startswith("geometry_") or leaf in _STRUCTURAL_METADATA_LEAVES:
            omitted[column] = "fixed structural/grid metadata"
            continue
        if column not in _PARAMETER_BY_COLUMN:
            omitted[column] = "not a registered sampled sensitivity parameter"
            continue
        reasons = []
        for frame in frames:
            variable, reason = _finite_variable(frame, column, max_cases=max_cases)
            if not variable:
                reasons.append(reason)
        if reasons:
            omitted[column] = "; ".join(dict.fromkeys(reasons))
            continue
        included.append(column)
    included.sort(key=lambda column: _PARAMETER_BY_COLUMN[column].order)
    return ParameterSelection(tuple(included), omitted)


def metadata_parameters(
    frames: Sequence[pd.DataFrame],
    *,
    max_cases: int | None = None,
) -> tuple[str, ...]:
    """Return shared variable inputs in canonical source-application order."""
    return metadata_parameter_selection(frames, max_cases=max_cases).included


def metadata_label(column: str) -> str:
    """Return the canonical concise label, including units when declared."""
    spec = _PARAMETER_BY_COLUMN.get(column)
    if spec is not None:
        return f"{spec.label} [{spec.unit}]" if spec.unit else spec.label
    for prefix in _METADATA_PREFIXES:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def metadata_group(column: str) -> str:
    """Return the canonical scientific generation group for one parameter."""
    try:
        return _PARAMETER_BY_COLUMN[column].group
    except KeyError as error:
        msg = f"Unsupported metadata parameter {column!r}."
        raise ValueError(msg) from error


def reference_case_position(frame: pd.DataFrame, parameters: Sequence[str] | None = None) -> int:
    """Return the saved row position nearest the standardized parameter median."""
    columns = tuple(parameters) if parameters is not None else metadata_parameters((frame,))
    if not columns:
        return len(frame) // 2
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        msg = "Reference-case metadata must be finite."
        raise ValueError(msg)
    center = np.median(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    distances = np.sum(((values - center) / scale) ** 2, axis=1)
    return int(np.argmin(distances))


def case_number(frame: pd.DataFrame, row_position: int) -> int:
    """Return the canonical persisted case number at one saved row position."""
    if isinstance(row_position, bool) or not isinstance(row_position, int) or not 0 <= row_position < len(frame):
        msg = "row_position is outside the artifact membership."
        raise IndexError(msg)
    value = frame.iloc[row_position]["case_index"]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        msg = "case_index must be an integer."
        raise TypeError(msg)
    return int(value)


def _transient_field_declarations(
    frame: pd.DataFrame,
) -> tuple[tuple[str, ...], Mapping[str, Mapping[str, object]], Mapping[str, object]]:
    """Read stored channel names and optional metadata from admitted artifact evidence."""
    provenance = frame.attrs.get("artifact_provenance")
    evaluation_value = provenance.get("evaluation") if isinstance(provenance, Mapping) else None
    evaluation = evaluation_value if isinstance(evaluation_value, Mapping) else {}
    objective_value = evaluation.get("objective")
    objective = objective_value if isinstance(objective_value, Mapping) else {}
    declared = frame.attrs.get("output_fields", objective.get("fields"))
    if declared is None:
        if frame.attrs.get("artifact_kind") != transient_artifact.TRANSIENT_ARTIFACT_KIND:
            msg = "Transient channel resolution requires an admitted sequence artifact."
            raise ValueError(msg)
        fields = transient_artifact.STATE_ORDER
    else:
        if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
            msg = "Transient artifact output fields must be an ordered sequence."
            raise TypeError(msg)
        fields = tuple(declared)
        if not fields or any(not isinstance(field, str) or not field for field in fields) or len(fields) != len(set(fields)):
            msg = "Transient artifact output fields must be unique non-empty text."
            raise ValueError(msg)

    metadata_value = frame.attrs.get("field_metadata", evaluation.get("field_metadata", {}))
    if not isinstance(metadata_value, Mapping):
        msg = "Transient field metadata must be a mapping when supplied."
        raise TypeError(msg)
    metadata = {key: value for key, value in metadata_value.items() if isinstance(key, str) and isinstance(value, Mapping)}

    units_value = frame.attrs.get("field_units", evaluation.get("field_units", {}))
    if not isinstance(units_value, Mapping):
        msg = "Transient field units must be a mapping when supplied."
        raise TypeError(msg)
    units = {key: value for key, value in units_value.items() if isinstance(key, str)}
    return fields, metadata, units


def _transient_frame_channels(frame: pd.DataFrame) -> TransientChannelResolution:
    """Resolve one frame without inventing a label, unit, or derived quantity."""
    fields, metadata, declared_units = _transient_field_declarations(frame)
    contract_units = dict(zip(transient_artifact.STATE_ORDER, transient_artifact.STATE_UNITS, strict=True))
    known_order = tuple(field for field in transient_artifact.STATE_ORDER if field in fields)
    future_order = tuple(sorted(set(fields).difference(known_order)))
    resolved: list[TransientDisplayField] = []
    omitted: dict[str, str] = {}
    for field in (*known_order, *future_order):
        field_metadata = metadata.get(field, {})
        label_value = field_metadata.get("label")
        label = label_value if isinstance(label_value, str) and label_value else (field_label(field) if has_declared_field_metadata(field) else None)
        contract_unit = contract_units.get(field)
        unit: str | None
        if contract_unit is not None:
            explicit_units: list[object] = []
            if "unit" in field_metadata:
                explicit_units.append(field_metadata["unit"])
            if field in declared_units:
                explicit_units.append(declared_units[field])
            if any(value != contract_unit for value in explicit_units):
                omitted[field] = f"declared unit conflicts with canonical unit {contract_unit!r}"
                continue
            unit = contract_unit
        else:
            unit_value = field_metadata.get("unit", declared_units.get(field))
            unit = unit_value if isinstance(unit_value, str) and unit_value else None
        if label is None or unit is None:
            missing = "label and unit" if label is None and unit is None else ("label" if label is None else "unit")
            omitted[field] = f"missing authoritative {missing} metadata"
            continue
        resolved.append(
            TransientDisplayField(
                key=field,
                label=label,
                unit=unit,
                stored=True,
                dependencies=(field,),
            )
        )

    provenance = frame.attrs.get("artifact_provenance")
    evaluation_value = provenance.get("evaluation") if isinstance(provenance, Mapping) else None
    evaluation = evaluation_value if isinstance(evaluation_value, Mapping) else {}
    policy_value = evaluation.get("process_diagnostic_policy")
    policy = policy_value if isinstance(policy_value, Mapping) else {}
    bulk_value = policy.get("bulk_moisture")
    bulk = bulk_value if isinstance(bulk_value, Mapping) else {}
    resolved_keys = {field.key for field in resolved}
    moisture_units = {field.unit for field in resolved if field.key in {"w_surf", "w_int"}}
    supports_bulk = (
        bulk.get("available") is True
        and {"w_surf", "w_int"}.issubset(resolved_keys)
        and len(moisture_units) == 1
        and "f_surf" in transient_artifact.SCALAR_ORDER
        and has_declared_field_metadata("w_gr")
    )
    if supports_bulk:
        derived = TransientDisplayField(
            key="w_gr",
            label=field_label("w_gr"),
            unit=next(iter(moisture_units)),
            stored=False,
            dependencies=("w_surf", "w_int", "f_surf"),
        )
        canonical_count = sum(field.key in transient_artifact.STATE_ORDER for field in resolved)
        resolved.insert(canonical_count, derived)
    else:
        omitted["w_gr"] = "bulk-moisture derivation lacks compatible w_surf, w_int, f_surf, unit, or policy evidence"
    return TransientChannelResolution(tuple(resolved), omitted)


def transient_channel_resolution(
    frames: Sequence[pd.DataFrame],
) -> TransientChannelResolution:
    """Return the scientifically compatible channel intersection for frames."""
    if not frames:
        msg = "At least one transient evaluation frame is required."
        raise ValueError(msg)
    per_frame = tuple(_transient_frame_channels(frame) for frame in frames)
    first_by_key = {field.key: field for field in per_frame[0].fields}
    shared_keys = set(first_by_key)
    for resolution in per_frame[1:]:
        shared_keys.intersection_update(field.key for field in resolution.fields)
    admitted: list[TransientDisplayField] = []
    omitted: dict[str, str] = {}
    for key, reference in first_by_key.items():
        if key not in shared_keys:
            omitted[key] = "not supplied by every compared artifact"
            continue
        matches = tuple(field for resolution in per_frame for field in resolution.fields if field.key == key)
        signatures = {(field.label, field.unit, field.stored, field.dependencies) for field in matches}
        if len(signatures) != 1:
            omitted[key] = "compared artifacts declare incompatible channel semantics"
            continue
        admitted.append(reference)
    for resolution in per_frame:
        for key, reason in resolution.omitted.items():
            omitted.setdefault(key, reason)
    if not admitted:
        msg = "Compared transient artifacts expose no compatible labelled channels with authoritative units."
        raise ValueError(msg)
    return TransientChannelResolution(tuple(admitted), omitted)
