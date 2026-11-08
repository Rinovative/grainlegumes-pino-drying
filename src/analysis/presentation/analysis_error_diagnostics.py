"""Select field-error maps using persisted physical-state scale evidence.

Responsibilities:
  - Admit saved steady or transient train-only physical-state reference scales
  - Build the selected signed, absolute, relative or normalized error map
  - Reject incompatible comparison scales

Design principles:
  - Transient normalization uses saved state_std independently of increment scale mode
  - Missing or invalid scale provenance yields explicit unavailability
  - All required support remains finite before an error map is admitted

This module does NOT:
  - Fit reference scales or infer scales for derived fields
  - Modify predictions, choose display grids or render figures
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np

SteadyErrorMode = Literal["MAE", "Relative [%]", "Normalized absolute error"]
TransientErrorMode = Literal["Signed error", "Absolute error", "Normalized absolute error"]
ErrorMode = SteadyErrorMode | TransientErrorMode

NORMALIZED_ABSOLUTE_ERROR: Literal["Normalized absolute error"] = "Normalized absolute error"
STEADY_ERROR_MODES: tuple[SteadyErrorMode, ...] = ("MAE", "Relative [%]", NORMALIZED_ABSOLUTE_ERROR)
TRANSIENT_ERROR_MODES: tuple[TransientErrorMode, ...] = ("Signed error", "Absolute error", NORMALIZED_ABSOLUTE_ERROR)
DEFAULT_ERROR_MODE = NORMALIZED_ABSOLUTE_ERROR
_RELATIVE_DENOMINATOR_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class ReferenceScale:
    """Fixed physical-difference scale with its train-only comparison context."""

    field: str
    unit: str
    value: float
    source: str
    fit_split: str
    dataset_identity: str
    scaler_identity: str

    def __post_init__(self) -> None:
        """Admit only an identifiable positive train-only physical scale."""
        labels = (self.field, self.unit, self.source, self.dataset_identity, self.scaler_identity)
        if (
            not all(isinstance(value, str) and value for value in labels)
            or self.fit_split != "train"
            or isinstance(self.value, bool)
            or not isinstance(self.value, Real)
            or not np.isfinite(float(self.value))
            or float(self.value) <= 0.0
        ):
            msg = "Reference scales require a positive finite train-only value and explicit field, unit, Dataset, and scaler provenance."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SelectedErrorMap:
    """One selected map or one truthful unavailability reason."""

    map_values: np.ndarray | None
    label: str
    unit: str
    reason: str | None
    scale: ReferenceScale | None

    @property
    def available(self) -> bool:
        """Return whether the selected representation has complete support."""
        return self.map_values is not None


def _positive_scale(
    field: str, unit: str, raw: Any, source: str, fit_split: str, dataset_identity: str, scaler_identity: str
) -> ReferenceScale | str:
    if isinstance(raw, bool) or not isinstance(raw, Real) or not np.isfinite(float(raw)) or float(raw) <= 0.0:
        return f"No strictly positive finite physical-state scale is persisted for {field}."
    return ReferenceScale(field, unit, float(raw), source, fit_split, dataset_identity, scaler_identity)


def steady_reference_scale(provenance: Mapping[str, Any], *, field: str, unit: str) -> ReferenceScale | str:
    """Read a saved train-only output scale; derived fields have no inferred scale."""
    normalizer = provenance.get("normalizer")
    dataset = provenance.get("dataset")
    if not isinstance(normalizer, Mapping) or not isinstance(dataset, Mapping):
        return "Admitted artifact has no normalizer or Dataset provenance."
    scales = normalizer.get("output_standard_deviations")
    if not isinstance(scales, Mapping) or field not in scales:
        return f"No persisted physical output scale exists for {field}."
    if normalizer.get("output_normalization") != "train_fitted_per_channel_standardization":
        return f"Saved normalization for {field} is not a physical-difference standard deviation."
    fit_split = normalizer.get("fit_split")
    identity = dataset.get("name")
    scaler_identity = normalizer.get("sha256")
    if fit_split != "train" or not isinstance(identity, str) or not identity:
        return "Train-only scale provenance or Dataset identity is missing."
    if not isinstance(scaler_identity, str) or not scaler_identity:
        return "Saved output scale has no scaler identity."
    return _positive_scale(field, unit, scales[field], "saved_run_normalizer.output_standard_deviations", fit_split, identity, scaler_identity)


def transient_reference_scale(
    scaling_state: Mapping[str, Any],
    *,
    field: str,
    unit: str,
    state_order: Sequence[str],
    dataset_identity: str,
    expected_scaler_identity: str | None = None,
) -> ReferenceScale | str:
    """Use persisted state_std, regardless of the increment training scale mode."""
    if field not in state_order:
        return f"No persisted physical-state scale exists for derived field {field}."
    raw = scaling_state.get("state_std")
    if raw is None:
        return "Saved transient scaling has no state_std."
    values = np.asarray(raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw)
    if values.shape != (len(state_order),):
        return "Saved transient state_std does not match the state field order."
    if not dataset_identity:
        return "Transient Dataset scale identity is missing."
    if scaling_state.get("train_membership_digest") != dataset_identity:
        return "Transient train membership does not match the declared Dataset scale identity."
    declared_order = scaling_state.get("state_names")
    if declared_order is not None and tuple(declared_order) != tuple(state_order):
        return "Saved transient state scale field order is incompatible."
    try:
        serialized = json.dumps(
            scaling_state,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
            default=lambda value: value.detach().cpu().tolist() if hasattr(value, "detach") else value.tolist(),
        )
    except (TypeError, ValueError, AttributeError):
        return "Saved transient scaler provenance cannot be identified exactly."
    scaler_identity = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if expected_scaler_identity is not None and scaler_identity != expected_scaler_identity:
        return "Saved transient state scale contradicts the evaluated artifact's scaler identity."
    return _positive_scale(
        field, unit, values[state_order.index(field)].item(), "saved_run_normalizer.state_std", "train", dataset_identity, scaler_identity
    )


def require_common_reference_scale(scales: Sequence[ReferenceScale]) -> ReferenceScale:
    """Require one declared fixed physical scale for a direct model comparison."""
    if not scales:
        msg = "A normalized comparison needs at least one reference scale."
        raise ValueError(msg)
    first = scales[0]
    for scale in scales[1:]:
        if scale != first:
            msg = "Normalized errors use incompatible field, unit, scale, or train-only Dataset provenance."
            raise ValueError(msg)
    return first


def selected_error_map(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    mode: ErrorMode,
    field: str,
    physical_unit: str,
    scale: ReferenceScale | str | None = None,
    support: np.ndarray | None = None,
) -> SelectedErrorMap:
    """Compute only the chosen physical error representation on full support."""
    if mode not in (*STEADY_ERROR_MODES, *TRANSIENT_ERROR_MODES):
        msg = f"Unknown field error mode {mode!r}."
        raise ValueError(msg)
    actual = np.asarray(reference)
    predicted = np.asarray(prediction)
    if (
        actual.shape != predicted.shape
        or actual.size == 0
        or not np.issubdtype(actual.dtype, np.number)
        or not np.issubdtype(predicted.dtype, np.number)
    ):
        msg = "Reference and prediction require matching non-empty numeric shapes."
        raise ValueError(msg)
    admitted = np.ones(actual.shape, dtype=bool) if support is None else np.asarray(support)
    if admitted.shape != actual.shape or admitted.dtype != np.bool_ or not admitted.any():
        msg = "Error support must be a non-empty aligned boolean mask."
        raise ValueError(msg)
    unit = "1" if mode == NORMALIZED_ABSOLUTE_ERROR else "%" if mode == "Relative [%]" else physical_unit
    if not np.isfinite(actual[admitted]).all() or not np.isfinite(predicted[admitted]).all():
        return SelectedErrorMap(None, mode, unit, "Non-finite reference or prediction on required support.", None)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = predicted - actual
    if not np.isfinite(difference[admitted]).all():
        return SelectedErrorMap(None, mode, unit, "Computed error is non-finite on required support.", None)
    if mode == NORMALIZED_ABSOLUTE_ERROR:
        if not isinstance(scale, ReferenceScale):
            reason = scale if isinstance(scale, str) else "No valid fixed train-only physical-state scale is available."
            return SelectedErrorMap(None, mode, unit, reason, None)
        if scale.unit != physical_unit or scale.field != field:
            return SelectedErrorMap(None, mode, unit, "Reference scale belongs to a different physical field or unit.", None)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.abs(difference) / scale.value
    elif mode == "Signed error":
        values = difference
    elif mode == "Relative [%]":
        required = np.asarray(actual[admitted], dtype=np.float64)
        maximum = float(np.max(np.abs(required)))
        denominator = (maximum * float(np.sqrt(np.mean(np.square(required / maximum)))) if maximum else 0.0) + _RELATIVE_DENOMINATOR_FLOOR
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.abs(difference) / denominator * 100.0
    else:
        values = np.abs(difference)
    if not np.isfinite(values[admitted]).all():
        return SelectedErrorMap(None, mode, unit, "Computed selected error is non-finite on required support.", None)
    if support is not None:
        values = np.where(admitted, values, np.nan)
    admitted_scale = scale if mode == NORMALIZED_ABSOLUTE_ERROR and isinstance(scale, ReferenceScale) else None
    return SelectedErrorMap(values, mode, unit, None, admitted_scale)
