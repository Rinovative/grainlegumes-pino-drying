"""
evaluation_transient_metrics.py

Aggregate strict transient drying evaluation statistics.

Responsibilities:
  - Accumulate complete-dataset normalized and physical error statistics
  - Delegate normalized drying macro RMSE to the central learning metric
  - Derive target, plausibility, and rollout-stability diagnostics

Design principles:
  - All reductions use float64 sufficient statistics and explicit valid masks
  - Endpoint and cumulative scopes remain explicit in every result
  - Physical granular water content requires an explicit surface fraction

This module does NOT:
  - Run inference, load artifacts, or select evaluation cases
  - Impose monotonicity on surface or internal moisture states
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Final

import numpy as np
import torch

from src import domain
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.learning.metrics import learning_metrics as metrics

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_STATE_FIELDS = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
_AIRFLOW_METRIC_ID_COUNT = 2
_MINIMUM_STATE_ARRAY_RANK = 2
_MINIMUM_STABILITY_ARRAY_RANK = 3
_SPATIAL_MASK_RANK = 2
NORMALIZED_AIRFLOW_GROUP_MACRO_RMSE = "normalized_airflow_group_macro_rmse"
HISTORICAL_AIRFLOW_GROUP_MACRO_RMSE_ALIAS = "normalized_group_macro_rmse"
NORMALIZED_DRYING_GROUP_MACRO_RMSE = "normalized_drying_group_macro_rmse"
TEMPERATURE_PLAUSIBILITY_RANGE_K: Final = (0.0, 2_000.0)
BULK_MOISTURE_CELL_WEIGHTING: Final = "canonical_trapezoidal_boundary_weights_over_spatial_mask"
BULK_MOISTURE_INVALID_POLICY: Final = "unavailable_without_imputation"
STABILITY_GROWTH_FACTOR: Final = 2.0


def resolve_airflow_metric_value(values: Mapping[str, object]) -> float:
    """Read the explicit Airflow metric ID while admitting its historical alias."""
    explicit = values.get(NORMALIZED_AIRFLOW_GROUP_MACRO_RMSE)
    historical = values.get(HISTORICAL_AIRFLOW_GROUP_MACRO_RMSE_ALIAS)
    if explicit is None and historical is None:
        msg = "Airflow Evaluation requires normalized_airflow_group_macro_rmse or its historical normalized_group_macro_rmse alias."
        raise KeyError(msg)
    candidates = [value for value in (explicit, historical) if value is not None]
    admitted: list[float] = []
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, Real):
            msg = "Airflow group macro RMSE values must be finite real scalars."
            raise TypeError(msg)
        admitted_value = float(value)
        if not np.isfinite(admitted_value):
            msg = "Airflow group macro RMSE values must be finite real scalars."
            raise ValueError(msg)
        admitted.append(admitted_value)
    if len(admitted) == _AIRFLOW_METRIC_ID_COUNT and admitted[0] != admitted[1]:
        msg = "Explicit and historical Airflow metric IDs disagree."
        raise ValueError(msg)
    return admitted[0]


@dataclass(frozen=True, slots=True)
class TransientMetricSummary:
    """Store complete-dataset transient error results for one explicit scope."""

    scope: str
    normalized_drying_group_macro_rmse: float
    normalized_rmse: Mapping[str, float]
    physical_rmse: Mapping[str, float]
    physical_mae: Mapping[str, float]
    relative_l2: Mapping[str, float]
    physical_w_gr_rmse: float
    physical_w_gr_mae: float
    bulk_dry_basis_rmse: float | None
    bulk_dry_basis_mae: float | None
    bulk_wet_basis_rmse: float | None
    bulk_wet_basis_mae: float | None
    predicted_bulk_dry_basis_mean: float | None
    reference_bulk_dry_basis_mean: float | None
    predicted_bulk_wet_basis_mean: float | None
    reference_bulk_wet_basis_mean: float | None
    bulk_moisture_valid_count: int
    bulk_moisture_unavailable_count: int
    valid_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class TargetCensoringDiagnostics:
    """Store target-attainment and censoring denominators without imputation."""

    total_count: int
    available_count: int
    predicted_reached_count: int
    reference_reached_count: int
    predicted_right_censored_count: int
    reference_right_censored_count: int
    agreement_count: int


@dataclass(frozen=True, slots=True)
class PlausibilityDiagnostics:
    """Store state plausibility counts with an explicit inspected denominator."""

    inspected_values: int
    nonfinite_values: int
    negative_moisture_values: int
    relative_humidity_bound_violations: int
    temperature_range_violations: int


@dataclass(frozen=True, slots=True)
class StabilityDiagnostics:
    """Store rollout increment diagnostics without a monotonic-moisture rule."""

    increment_count: int
    nonfinite_increment_count: int
    oscillatory_increment_count: int
    abnormal_growth_count: int


def _as_float64(
    value: np.ndarray | torch.Tensor,
    *,
    label: str,
    require_finite: bool = True,
) -> np.ndarray:
    """Return one float64 state array without mutating caller data."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        msg = f"{label} must be a numpy array or torch tensor."
        raise TypeError(msg)
    if array.ndim < _MINIMUM_STATE_ARRAY_RANK or array.shape[1] != len(_STATE_FIELDS):
        msg = f"{label} must have channel axis 1 in exact state order {_STATE_FIELDS}."
        raise ValueError(msg)
    result = np.asarray(array, dtype=np.float64)
    if require_finite and not np.isfinite(result).all():
        msg = f"{label} must contain only finite values for error aggregation."
        raise ValueError(msg)
    return result


def _mask_for(value: np.ndarray | torch.Tensor | None, *, shape: tuple[int, ...]) -> np.ndarray:
    """Return one broadcast boolean validity mask aligned to a state tensor."""
    if value is None:
        return np.ones(shape, dtype=bool)
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        raw = value
    else:
        msg = "valid_mask must be a numpy array, torch tensor, or None."
        raise TypeError(msg)
    if raw.dtype != np.bool_:
        msg = "valid_mask must have boolean dtype."
        raise TypeError(msg)
    try:
        result = np.broadcast_to(raw, shape)
    except ValueError as error:
        msg = "valid_mask must broadcast to prediction and reference shape."
        raise ValueError(msg) from error
    return np.array(result, dtype=bool, copy=True)


def trapezoidal_cell_weights(spatial_mask: np.ndarray) -> np.ndarray:
    """Return canonical structured-grid integration weights over valid cells."""
    mask = np.asarray(spatial_mask)
    if mask.dtype != np.bool_ or mask.ndim != _SPATIAL_MASK_RANK or not bool(mask.any()):
        msg = "spatial_mask must be one non-empty boolean [Y,X] array."
        raise ValueError(msg)
    weights = np.ones(mask.shape, dtype=np.float64)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    weights[~mask] = 0.0
    return weights


def _surface_fraction(value: np.ndarray | torch.Tensor, *, target_shape: tuple[int, ...]) -> np.ndarray:
    """Validate and broadcast the explicit physical surface fraction."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        raw = value
    else:
        msg = "f_surf must be a numpy array or torch tensor."
        raise TypeError(msg)
    fraction = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(fraction).all() or np.any((fraction <= 0.0) | (fraction >= 1.0)):
        msg = "f_surf must be finite and strictly between zero and one."
        raise ValueError(msg)
    try:
        return np.broadcast_to(fraction, target_shape)
    except ValueError as error:
        msg = "f_surf must broadcast to the moisture-state shape."
        raise ValueError(msg) from error


def _broadcast_spatial_field(
    value: np.ndarray | torch.Tensor,
    *,
    target_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """Return one finite spatial field broadcast across evaluated time states."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        raw = value
    else:
        msg = f"{label} must be a numpy array or torch tensor."
        raise TypeError(msg)
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        msg = f"{label} must contain only finite values."
        raise ValueError(msg)
    try:
        return np.broadcast_to(array, target_shape)
    except ValueError as error:
        msg = f"{label} must broadcast to the batch-spatial moisture shape."
        raise ValueError(msg) from error


def _bulk_moisture_values(
    water: np.ndarray,
    dry_density: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float] | None:
    """Return canonical bulk dry/wet moisture or explicit physical unavailability."""
    selected = weights > 0.0
    if not bool(selected.any()):
        return None
    selected_water = water[selected]
    selected_density = dry_density[selected]
    selected_weights = weights[selected]
    if np.any(selected_water < 0.0) or np.any(selected_density <= 0.0):
        return None
    wet_basis = domain.moisture.bulk_wet_basis_moisture(
        selected_water,
        selected_density,
        cell_weights=selected_weights,
    )
    dry_basis = float(domain.moisture.wet_basis_to_dry_basis(np.asarray(wet_basis)))
    return dry_basis, wet_basis


class TransientMetricAccumulator:
    """
    Accumulate one explicit endpoint or cumulative transient evaluation scope.

    Parameters
    ----------
    scope : {"endpoint", "cumulative"}
        Declared evaluation reduction scope. The accumulator never changes it.

    """

    def __init__(self, *, scope: str) -> None:
        """Initialize float64 sufficient statistics and the central macro metric."""
        if scope not in {"endpoint", "cumulative"}:
            msg = "scope must be either 'endpoint' or 'cumulative'."
            raise ValueError(msg)
        definition = metrics.ResolvedMetric(
            id=NORMALIZED_DRYING_GROUP_MACRO_RMSE,
            kind="drying_group_macro_rmse",
            space="normalized",
            fields=_STATE_FIELDS,
            field_indices=(0, 1, 2, 3),
            reduction="group_macro_element_mean",
            direction="minimize",
            unit="1",
            operator_dimensionality=2,
            groups=(),
            field_standard_deviations=(),
        )
        self._scope = scope
        self._normalized_macro = metrics.DryingGroupMacroRMSEMetric(definition, device=torch.device("cpu"))
        self._squared_sums = np.zeros(len(_STATE_FIELDS), dtype=np.float64)
        self._absolute_sums = np.zeros(len(_STATE_FIELDS), dtype=np.float64)
        self._reference_squared_sums = np.zeros(len(_STATE_FIELDS), dtype=np.float64)
        self._counts = np.zeros(len(_STATE_FIELDS), dtype=np.int64)
        self._w_gr_squared_sum = 0.0
        self._w_gr_absolute_sum = 0.0
        self._w_gr_count = 0
        self._bulk_dry_squared_sum = 0.0
        self._bulk_dry_absolute_sum = 0.0
        self._bulk_wet_squared_sum = 0.0
        self._bulk_wet_absolute_sum = 0.0
        self._predicted_bulk_dry_sum = 0.0
        self._reference_bulk_dry_sum = 0.0
        self._predicted_bulk_wet_sum = 0.0
        self._reference_bulk_wet_sum = 0.0
        self._bulk_moisture_valid_count = 0
        self._bulk_moisture_unavailable_count = 0
        self._batch_index = 0

    def update(
        self,
        *,
        normalized_prediction: np.ndarray | torch.Tensor,
        normalized_reference: np.ndarray | torch.Tensor,
        physical_prediction: np.ndarray | torch.Tensor,
        physical_reference: np.ndarray | torch.Tensor,
        f_surf: np.ndarray | torch.Tensor,
        rho_bu_dry: np.ndarray | torch.Tensor,
        cell_weights: np.ndarray | torch.Tensor | None = None,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """
        Add one batch of aligned normalized and physical reconstructed states.

        Parameters
        ----------
        normalized_prediction, normalized_reference, physical_prediction, physical_reference
            Arrays with matching shape ``(batch, 4, ...)`` in canonical state order.
        f_surf
            Explicit physical surface fraction broadcastable to one moisture channel.
        rho_bu_dry
            Canonical dry-bulk-density field broadcastable across evaluated states.
        cell_weights
            Optional non-negative structured-grid integration weights.
        valid_mask
            Optional boolean mask broadcastable to every state channel.

        """
        normalized_pred = _as_float64(normalized_prediction, label="normalized_prediction")
        normalized_ref = _as_float64(normalized_reference, label="normalized_reference")
        physical_pred = _as_float64(physical_prediction, label="physical_prediction")
        physical_ref = _as_float64(physical_reference, label="physical_reference")
        if normalized_pred.shape != normalized_ref.shape or physical_pred.shape != physical_ref.shape:
            msg = "Prediction and reference arrays must match within each representation."
            raise ValueError(msg)
        if normalized_pred.shape != physical_pred.shape:
            msg = "Normalized and physical arrays must have the same reconstructed-state shape."
            raise ValueError(msg)
        mask = _mask_for(valid_mask, shape=physical_pred.shape)
        if not bool(mask.any()):
            msg = "valid_mask must retain at least one state value."
            raise ValueError(msg)
        normalized_pred_tensor = torch.as_tensor(normalized_pred, dtype=torch.float64)
        normalized_ref_tensor = torch.as_tensor(normalized_ref, dtype=torch.float64)
        self._normalized_macro.update(
            normalized_pred_tensor,
            normalized_ref_tensor,
            space="normalized",
            batch_index=self._batch_index,
            mask=torch.as_tensor(mask, dtype=torch.bool),
        )
        difference = physical_pred - physical_ref
        for index in range(len(_STATE_FIELDS)):
            selected = difference[:, index][mask[:, index]]
            self._squared_sums[index] += float(np.square(selected, dtype=np.float64).sum(dtype=np.float64))
            self._absolute_sums[index] += float(np.abs(selected).sum(dtype=np.float64))
            selected_reference = physical_ref[:, index][mask[:, index]]
            self._reference_squared_sums[index] += float(np.square(selected_reference, dtype=np.float64).sum(dtype=np.float64))
            self._counts[index] += selected.size
        fraction = _surface_fraction(f_surf, target_shape=physical_pred[:, 2].shape)
        predicted_water = fraction * physical_pred[:, 2] + (1.0 - fraction) * physical_pred[:, 3]
        reference_water = fraction * physical_ref[:, 2] + (1.0 - fraction) * physical_ref[:, 3]
        water_mask = mask[:, 2] & mask[:, 3]
        water_difference = (predicted_water - reference_water)[water_mask]
        if water_difference.size == 0:
            msg = "valid_mask must retain paired surface and internal moisture values for w_gr."
            raise ValueError(msg)
        self._w_gr_squared_sum += float(np.square(water_difference, dtype=np.float64).sum(dtype=np.float64))
        self._w_gr_absolute_sum += float(np.abs(water_difference).sum(dtype=np.float64))
        self._w_gr_count += int(water_difference.size)

        spatial_shape = predicted_water.shape
        dry_density = _broadcast_spatial_field(
            rho_bu_dry,
            target_shape=spatial_shape,
            label="rho_bu_dry",
        )
        integration_weights = (
            np.ones(spatial_shape, dtype=np.float64)
            if cell_weights is None
            else _broadcast_spatial_field(
                cell_weights,
                target_shape=spatial_shape,
                label="cell_weights",
            )
        )
        if np.any(integration_weights < 0.0):
            msg = "cell_weights must be non-negative."
            raise ValueError(msg)
        for index in range(predicted_water.shape[0]):
            weights = np.where(water_mask[index], integration_weights[index], 0.0)
            predicted_bulk = _bulk_moisture_values(
                predicted_water[index],
                dry_density[index],
                weights,
            )
            reference_bulk = _bulk_moisture_values(
                reference_water[index],
                dry_density[index],
                weights,
            )
            if predicted_bulk is None or reference_bulk is None:
                self._bulk_moisture_unavailable_count += 1
                continue
            predicted_dry, predicted_wet = predicted_bulk
            reference_dry, reference_wet = reference_bulk
            dry_error = predicted_dry - reference_dry
            wet_error = predicted_wet - reference_wet
            self._bulk_dry_squared_sum += dry_error**2
            self._bulk_dry_absolute_sum += abs(dry_error)
            self._bulk_wet_squared_sum += wet_error**2
            self._bulk_wet_absolute_sum += abs(wet_error)
            self._predicted_bulk_dry_sum += predicted_dry
            self._reference_bulk_dry_sum += reference_dry
            self._predicted_bulk_wet_sum += predicted_wet
            self._reference_bulk_wet_sum += reference_wet
            self._bulk_moisture_valid_count += 1
        self._batch_index += 1

    def state_dict(self) -> dict[str, object]:
        """Return JSON-compatible float64 sufficient statistics for exact merging."""
        normalized_sums = self._normalized_macro._field_sums  # noqa: SLF001 -- canonical metric persistence
        return {
            "scope": self._scope,
            "normalized_squared_sums": [float(value) for value in normalized_sums],
            "squared_sums": self._squared_sums.tolist(),
            "absolute_sums": self._absolute_sums.tolist(),
            "reference_squared_sums": self._reference_squared_sums.tolist(),
            "counts": self._counts.tolist(),
            "w_gr_squared_sum": self._w_gr_squared_sum,
            "w_gr_absolute_sum": self._w_gr_absolute_sum,
            "w_gr_count": self._w_gr_count,
            "bulk_dry_squared_sum": self._bulk_dry_squared_sum,
            "bulk_dry_absolute_sum": self._bulk_dry_absolute_sum,
            "bulk_wet_squared_sum": self._bulk_wet_squared_sum,
            "bulk_wet_absolute_sum": self._bulk_wet_absolute_sum,
            "predicted_bulk_dry_sum": self._predicted_bulk_dry_sum,
            "reference_bulk_dry_sum": self._reference_bulk_dry_sum,
            "predicted_bulk_wet_sum": self._predicted_bulk_wet_sum,
            "reference_bulk_wet_sum": self._reference_bulk_wet_sum,
            "bulk_moisture_valid_count": self._bulk_moisture_valid_count,
            "bulk_moisture_unavailable_count": self._bulk_moisture_unavailable_count,
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping[str, object],
    ) -> TransientMetricAccumulator:
        """Recreate one accumulator from validated persisted sufficient statistics."""
        vector_names = (
            "normalized_squared_sums",
            "squared_sums",
            "absolute_sums",
            "reference_squared_sums",
            "counts",
        )
        scalar_names = (
            "w_gr_squared_sum",
            "w_gr_absolute_sum",
            "bulk_dry_squared_sum",
            "bulk_dry_absolute_sum",
            "bulk_wet_squared_sum",
            "bulk_wet_absolute_sum",
            "predicted_bulk_dry_sum",
            "reference_bulk_dry_sum",
            "predicted_bulk_wet_sum",
            "reference_bulk_wet_sum",
        )
        count_names = (
            "w_gr_count",
            "bulk_moisture_valid_count",
            "bulk_moisture_unavailable_count",
        )
        expected = {"scope", *vector_names, *scalar_names, *count_names}
        if set(value) != expected:
            msg = "Transient metric statistic fields do not match the persisted schema."
            raise ValueError(msg)
        scope = value["scope"]
        if scope not in {"endpoint", "cumulative"}:
            msg = "Transient metric statistics scope is invalid."
            raise ValueError(msg)
        vectors: dict[str, np.ndarray] = {}
        for name in vector_names:
            try:
                raw = np.asarray(value[name], dtype=np.float64)
            except (TypeError, ValueError) as error:
                msg = f"Transient metric statistics {name} is not a real vector."
                raise ValueError(msg) from error
            if raw.shape != (len(_STATE_FIELDS),) or not np.isfinite(raw).all() or np.any(raw < 0.0):
                msg = f"Transient metric statistics {name} is invalid."
                raise ValueError(msg)
            vectors[name] = raw
        raw_counts = np.asarray(value["counts"])
        if raw_counts.dtype.kind not in {"i", "u"} or np.any(raw_counts <= 0):
            msg = "Transient metric statistic counts must be positive integers."
            raise ValueError(msg)
        result = cls(scope=str(scope))
        result._normalized_macro._field_sums = [  # noqa: SLF001 -- canonical metric restoration
            float(item) for item in vectors["normalized_squared_sums"]
        ]
        result._normalized_macro._field_counts = [  # noqa: SLF001 -- canonical metric restoration
            int(item) for item in raw_counts
        ]
        result._squared_sums = vectors["squared_sums"].copy()
        result._absolute_sums = vectors["absolute_sums"].copy()
        result._reference_squared_sums = vectors["reference_squared_sums"].copy()
        result._counts = raw_counts.astype(np.int64, copy=True)
        scalar_values: dict[str, float] = {}
        for name in scalar_names:
            raw_scalar = value[name]
            if isinstance(raw_scalar, bool) or not isinstance(raw_scalar, Real) or not np.isfinite(float(raw_scalar)) or float(raw_scalar) < 0.0:
                msg = f"Transient metric statistics {name} is invalid."
                raise ValueError(msg)
            scalar_values[name] = float(raw_scalar)
        count_values: dict[str, int] = {}
        for name in count_names:
            raw_count = value[name]
            if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0 or (name == "w_gr_count" and raw_count == 0):
                msg = f"Transient metric statistics {name} is invalid."
                raise ValueError(msg)
            count_values[name] = raw_count
        result._w_gr_squared_sum = scalar_values["w_gr_squared_sum"]
        result._w_gr_absolute_sum = scalar_values["w_gr_absolute_sum"]
        result._bulk_dry_squared_sum = scalar_values["bulk_dry_squared_sum"]
        result._bulk_dry_absolute_sum = scalar_values["bulk_dry_absolute_sum"]
        result._bulk_wet_squared_sum = scalar_values["bulk_wet_squared_sum"]
        result._bulk_wet_absolute_sum = scalar_values["bulk_wet_absolute_sum"]
        result._predicted_bulk_dry_sum = scalar_values["predicted_bulk_dry_sum"]
        result._reference_bulk_dry_sum = scalar_values["reference_bulk_dry_sum"]
        result._predicted_bulk_wet_sum = scalar_values["predicted_bulk_wet_sum"]
        result._reference_bulk_wet_sum = scalar_values["reference_bulk_wet_sum"]
        result._w_gr_count = count_values["w_gr_count"]
        result._bulk_moisture_valid_count = count_values["bulk_moisture_valid_count"]
        result._bulk_moisture_unavailable_count = count_values["bulk_moisture_unavailable_count"]
        result._batch_index = 1
        return result

    @classmethod
    def merged(
        cls,
        states: Sequence[Mapping[str, object]],
    ) -> TransientMetricAccumulator:
        """Merge same-scope persisted sufficient statistics without RMSE averaging."""
        if not states:
            msg = "Transient metric statistics require at least one state."
            raise ValueError(msg)
        accumulators = [cls.from_state_dict(state) for state in states]
        scope = str(states[0]["scope"])
        if any(state["scope"] != scope for state in states):
            msg = "Transient metric statistic scopes disagree."
            raise ValueError(msg)
        result = cls(scope=scope)
        normalized_sums = np.sum(
            [item._normalized_macro._field_sums for item in accumulators],  # noqa: SLF001 -- canonical metric merge
            axis=0,
        )
        normalized_counts = np.sum(
            [item._normalized_macro._field_counts for item in accumulators],  # noqa: SLF001 -- canonical metric merge
            axis=0,
        )
        result._normalized_macro._field_sums = normalized_sums.tolist()  # noqa: SLF001 -- canonical metric merge
        result._normalized_macro._field_counts = (  # noqa: SLF001 -- canonical metric merge
            normalized_counts.astype(int).tolist()
        )
        for name in (
            "_squared_sums",
            "_absolute_sums",
            "_reference_squared_sums",
            "_counts",
        ):
            setattr(
                result,
                name,
                np.sum([getattr(item, name) for item in accumulators], axis=0),
            )
        for name in (
            "_w_gr_squared_sum",
            "_w_gr_absolute_sum",
            "_w_gr_count",
            "_bulk_dry_squared_sum",
            "_bulk_dry_absolute_sum",
            "_bulk_wet_squared_sum",
            "_bulk_wet_absolute_sum",
            "_predicted_bulk_dry_sum",
            "_reference_bulk_dry_sum",
            "_predicted_bulk_wet_sum",
            "_reference_bulk_wet_sum",
            "_bulk_moisture_valid_count",
            "_bulk_moisture_unavailable_count",
        ):
            setattr(
                result,
                name,
                sum(getattr(item, name) for item in accumulators),
            )
        result._batch_index = len(accumulators)
        return result

    def finalize(self) -> TransientMetricSummary:
        """Finalize complete-dataset metrics from the accumulated statistics."""
        if self._batch_index == 0 or np.any(self._counts == 0) or self._w_gr_count == 0:
            msg = "Transient metric accumulation requires valid complete-dataset samples for every state."
            raise RuntimeError(msg)
        normalized = self._normalized_macro.components()
        physical_rmse = {
            field: float(np.sqrt(total / count)) for field, total, count in zip(_STATE_FIELDS, self._squared_sums, self._counts, strict=True)
        }
        physical_mae = {field: float(total / count) for field, total, count in zip(_STATE_FIELDS, self._absolute_sums, self._counts, strict=True)}
        relative_l2 = {
            field: float(np.sqrt(squared_error / max(reference_squared, np.finfo(np.float64).eps)))
            for field, squared_error, reference_squared in zip(
                _STATE_FIELDS,
                self._squared_sums,
                self._reference_squared_sums,
                strict=True,
            )
        }
        bulk_count = self._bulk_moisture_valid_count
        bulk_dry_rmse = float(np.sqrt(self._bulk_dry_squared_sum / bulk_count)) if bulk_count else None
        bulk_dry_mae = float(self._bulk_dry_absolute_sum / bulk_count) if bulk_count else None
        bulk_wet_rmse = float(np.sqrt(self._bulk_wet_squared_sum / bulk_count)) if bulk_count else None
        bulk_wet_mae = float(self._bulk_wet_absolute_sum / bulk_count) if bulk_count else None
        return TransientMetricSummary(
            scope=self._scope,
            normalized_drying_group_macro_rmse=self._normalized_macro.compute(),
            normalized_rmse={field: float(normalized[field]) for field in _STATE_FIELDS},
            physical_rmse=physical_rmse,
            physical_mae=physical_mae,
            relative_l2=relative_l2,
            physical_w_gr_rmse=float(np.sqrt(self._w_gr_squared_sum / self._w_gr_count)),
            physical_w_gr_mae=float(self._w_gr_absolute_sum / self._w_gr_count),
            bulk_dry_basis_rmse=bulk_dry_rmse,
            bulk_dry_basis_mae=bulk_dry_mae,
            bulk_wet_basis_rmse=bulk_wet_rmse,
            bulk_wet_basis_mae=bulk_wet_mae,
            predicted_bulk_dry_basis_mean=(self._predicted_bulk_dry_sum / bulk_count if bulk_count else None),
            reference_bulk_dry_basis_mean=(self._reference_bulk_dry_sum / bulk_count if bulk_count else None),
            predicted_bulk_wet_basis_mean=(self._predicted_bulk_wet_sum / bulk_count if bulk_count else None),
            reference_bulk_wet_basis_mean=(self._reference_bulk_wet_sum / bulk_count if bulk_count else None),
            bulk_moisture_valid_count=bulk_count,
            bulk_moisture_unavailable_count=self._bulk_moisture_unavailable_count,
            valid_counts={field: int(count) for field, count in zip(_STATE_FIELDS, self._counts, strict=True)},
        )


def derive_target_censoring_diagnostics(
    *,
    predicted_reached: np.ndarray,
    reference_reached: np.ndarray,
) -> TargetCensoringDiagnostics:
    """Derive target/censoring agreement counts from aligned boolean availability."""
    predicted = np.asarray(predicted_reached)
    reference = np.asarray(reference_reached)
    if predicted.shape != reference.shape or predicted.ndim != 1:
        msg = "Target arrays must be aligned rank-one arrays."
        raise ValueError(msg)
    available = (predicted != None) & (reference != None)  # noqa: E711
    if any(value is not None and not isinstance(value, (bool, np.bool_)) for value in predicted.tolist() + reference.tolist()):
        msg = "Target arrays may contain only booleans or None."
        raise TypeError(msg)
    paired_prediction = predicted[available].astype(bool)
    paired_reference = reference[available].astype(bool)
    return TargetCensoringDiagnostics(
        total_count=int(predicted.size),
        available_count=int(available.sum()),
        predicted_reached_count=int(paired_prediction.sum()),
        reference_reached_count=int(paired_reference.sum()),
        predicted_right_censored_count=int((~paired_prediction).sum()),
        reference_right_censored_count=int((~paired_reference).sum()),
        agreement_count=int((paired_prediction == paired_reference).sum()),
    )


def derive_plausibility_diagnostics(
    states: np.ndarray | torch.Tensor,
    *,
    temperature_range: tuple[float, float],
) -> PlausibilityDiagnostics:
    """Count physical plausibility violations without assuming moisture monotonicity."""
    array = _as_float64(states, label="states", require_finite=False)
    lower, upper = temperature_range
    if not all(isinstance(value, Real) and np.isfinite(value) for value in (lower, upper)) or lower >= upper:
        msg = "temperature_range must contain finite ascending bounds."
        raise ValueError(msg)
    return PlausibilityDiagnostics(
        inspected_values=int(array.size),
        nonfinite_values=int((~np.isfinite(array)).sum()),
        negative_moisture_values=int((array[:, 2:] < 0.0).sum()),
        relative_humidity_bound_violations=int(((array[:, 1] < 0.0) | (array[:, 1] > 1.0)).sum()),
        temperature_range_violations=int(((array[:, 0] < lower) | (array[:, 0] > upper)).sum()),
    )


def derive_stability_diagnostics(
    states: np.ndarray | torch.Tensor,
    *,
    growth_factor: float = STABILITY_GROWTH_FACTOR,
) -> StabilityDiagnostics:
    """Count nonfinite, sign-oscillatory, and growing rollout increments."""
    array = _as_float64(states, label="states", require_finite=False)
    if array.ndim < _MINIMUM_STABILITY_ARRAY_RANK or array.shape[1] != len(_STATE_FIELDS):
        msg = "states must have shape (time, 4, ...) for stability diagnostics."
        raise ValueError(msg)
    if not isinstance(growth_factor, Real) or not np.isfinite(growth_factor) or growth_factor <= 1.0:
        msg = "growth_factor must be finite and greater than one."
        raise ValueError(msg)
    increments = np.diff(array, axis=0)
    if increments.shape[0] == 0:
        return StabilityDiagnostics(0, 0, 0, 0)
    finite = np.isfinite(increments)
    nonfinite = ~finite
    magnitudes = np.abs(increments)
    paired_finite = finite[1:] & finite[:-1]
    oscillations = paired_finite & ((increments[1:] * increments[:-1]) < 0.0)
    growth = paired_finite & (magnitudes[1:] > growth_factor * np.maximum(magnitudes[:-1], np.finfo(np.float64).eps))
    return StabilityDiagnostics(
        increment_count=int(increments.size),
        nonfinite_increment_count=int(nonfinite.sum()),
        oscillatory_increment_count=int(oscillations.sum()),
        abnormal_growth_count=int(growth.sum()),
    )
