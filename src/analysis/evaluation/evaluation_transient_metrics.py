"""Aggregate strict transient drying evaluation statistics.

Responsibilities:
  - Accumulate the physical field statistics of one case (learning.metrics.metrics.field_statistics) over its states
  - Accumulate per-case grain-water and bulk-moisture errors and reduce them over cases like every field metric
  - Derive target, plausibility, and rollout-stability diagnostics

Design principles:
  - All reductions use float64 sufficient statistics over the full Evaluation grid
  - Moisture errors weigh cases equally within a material and materials equally, root after the mean
  - Endpoint and cumulative scopes remain explicit in every result
  - Physical granular water content requires an explicit surface fraction

This module does NOT:
  - Run inference, load artifacts, or select evaluation cases
  - Impose monotonicity on surface or internal moisture states
  - Reduce field errors or the drying objective over cases; the session weighs cases and materials equally for them
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import torch

from src import domain, learning
from src.datasets.contracts import dataset_contracts_transient as transient_contract

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_STATE_FIELDS = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.dynamic_state)
_STATIC_FIELDS = tuple(field.name for field in transient_contract.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning)
_MINIMUM_STATE_ARRAY_RANK = 2
_MINIMUM_STABILITY_ARRAY_RANK = 3
_SPATIAL_RANK = 2
NORMALIZED_AIRFLOW_GROUP_MACRO_RMSE = "normalized_airflow_group_macro_rmse"
NORMALIZED_DRYING_GROUP_MACRO_RMSE = "normalized_drying_group_macro_rmse"
TEMPERATURE_PLAUSIBILITY_RANGE_K: Final = (0.0, 2_000.0)
BULK_MOISTURE_CELL_WEIGHTING: Final = "canonical_trapezoidal_boundary_weights_over_the_full_grid"
BULK_MOISTURE_INVALID_POLICY: Final = "unavailable_without_imputation"
STABILITY_GROWTH_FACTOR: Final = 2.0


@dataclass(frozen=True, slots=True)
class CaseMoisture:
    """
    Hold one case's grain-water and bulk-moisture error means for one scope.

    ``w_gr_*`` average over every grid value of every state; the bulk values average over
    the states with a valid bulk moisture and are ``None`` without one.
    """

    scope: str
    w_gr_squared: float
    w_gr_absolute: float
    bulk_dry_squared: float | None
    bulk_dry_absolute: float | None
    bulk_wet_squared: float | None
    bulk_wet_absolute: float | None
    predicted_bulk_dry: float | None
    reference_bulk_dry: float | None
    predicted_bulk_wet: float | None
    reference_bulk_wet: float | None
    bulk_valid_count: int
    bulk_unavailable_count: int


@dataclass(frozen=True, slots=True)
class TransientMetricSummary:
    """Store grain-water and bulk-moisture results of one case population and scope, each case weighing equally."""

    scope: str
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


def trapezoidal_cell_weights(shape: tuple[int, ...]) -> np.ndarray:
    """Return canonical structured-grid integration weights over one full ``(Y, X)`` grid."""
    if len(shape) != _SPATIAL_RANK or min(shape) < 1:
        msg = f"Cell weights require one non-empty (Y, X) grid shape, got {shape}."
        raise ValueError(msg)
    weights = np.ones(shape, dtype=np.float64)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    return weights


def grid_spacing(static_conditioning: np.ndarray) -> tuple[float, float]:
    """Return the uniform (rows, columns) spacing in metres of one record's static ``x`` and ``y`` coordinate fields."""
    static = np.asarray(static_conditioning, dtype=np.float64)
    x_coordinates, y_coordinates = static[_STATIC_FIELDS.index("x")], static[_STATIC_FIELDS.index("y")]
    return float(abs(y_coordinates[1, 0] - y_coordinates[0, 0])), float(abs(x_coordinates[0, 1] - x_coordinates[0, 0]))


def state_field_statistics(prediction: np.ndarray, reference: np.ndarray, *, spacing: tuple[float, float]) -> np.ndarray:
    """Return the physical field statistics of every state row, ``(rows, 4, len(CASE_STATISTICS))``."""
    return learning.metrics.metrics.field_statistics(
        torch.from_numpy(np.ascontiguousarray(prediction, dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(reference, dtype=np.float64)),
        grid_spacing=spacing,
    )


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
        """Initialize float64 field, grain-water and bulk-moisture sufficient statistics."""
        if scope not in {"endpoint", "cumulative"}:
            msg = "scope must be either 'endpoint' or 'cumulative'."
            raise ValueError(msg)
        self._scope = scope
        self._field_sums = np.zeros((len(_STATE_FIELDS), len(learning.metrics.metrics.CASE_STATISTICS)), dtype=np.float64)
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
        physical_prediction: np.ndarray | torch.Tensor,
        physical_reference: np.ndarray | torch.Tensor,
        f_surf: np.ndarray | torch.Tensor,
        rho_bu_dry: np.ndarray | torch.Tensor,
        spacing: tuple[float, float],
    ) -> None:
        """
        Add aligned physical reconstructed states of one case on its full Evaluation grid.

        Parameters
        ----------
        physical_prediction, physical_reference
            Arrays with matching shape ``(states, 4, Y, X)`` in canonical state order.
        f_surf
            Explicit physical surface fraction broadcastable to one moisture channel.
        rho_bu_dry
            Canonical dry-bulk-density field broadcastable across evaluated states.
        spacing
            Grid spacing in metres (rows, then columns) of the H1 statistics.

        """
        physical_pred = _as_float64(physical_prediction, label="physical_prediction")
        physical_ref = _as_float64(physical_reference, label="physical_reference")
        if physical_pred.shape != physical_ref.shape:
            msg = "Prediction and reference arrays must match."
            raise ValueError(msg)
        self._field_sums += state_field_statistics(physical_pred, physical_ref, spacing=spacing).sum(axis=0)
        fraction = _surface_fraction(f_surf, target_shape=physical_pred[:, 2].shape)
        predicted_water = fraction * physical_pred[:, 2] + (1.0 - fraction) * physical_pred[:, 3]
        reference_water = fraction * physical_ref[:, 2] + (1.0 - fraction) * physical_ref[:, 3]
        water_difference = predicted_water - reference_water
        self._w_gr_squared_sum += float(np.square(water_difference, dtype=np.float64).sum(dtype=np.float64))
        self._w_gr_absolute_sum += float(np.abs(water_difference).sum(dtype=np.float64))
        self._w_gr_count += int(water_difference.size)

        spatial_shape = predicted_water.shape
        dry_density = _broadcast_spatial_field(
            rho_bu_dry,
            target_shape=spatial_shape,
            label="rho_bu_dry",
        )
        weights = trapezoidal_cell_weights(spatial_shape[-2:])
        for index in range(predicted_water.shape[0]):
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
        return {
            "scope": self._scope,
            "field_statistics": self._field_sums.tolist(),
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
        expected = {"scope", "field_statistics", *scalar_names, *count_names}
        if set(value) != expected:
            msg = "Transient metric statistic fields do not match the persisted schema."
            raise ValueError(msg)
        scope = value["scope"]
        if scope not in {"endpoint", "cumulative"}:
            msg = "Transient metric statistics scope is invalid."
            raise ValueError(msg)
        result = cls(scope=str(scope))
        try:
            # CaseStatistics admits the sums: finite, non-negative, a positive count for every field.
            sums = np.asarray(value["field_statistics"], dtype=np.float64)
            result._field_sums = learning.metrics.metrics.CaseStatistics("persisted", "persisted", _STATE_FIELDS, sums).sums.copy()
        except (TypeError, ValueError) as error:
            msg = "Transient metric statistics field_statistics is invalid."
            raise ValueError(msg) from error
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
        result._field_sums = np.sum([np.asarray(state["field_statistics"], dtype=np.float64) for state in states], axis=0)
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

    def case_statistics(self, case_id: str, material: str) -> learning.metrics.metrics.CaseStatistics:
        """Return the accumulated field statistics as the statistics of one case."""
        return learning.metrics.metrics.CaseStatistics(case_id, material, _STATE_FIELDS, self._field_sums.copy())

    def case_moisture(self) -> CaseMoisture:
        """Return the accumulated grain-water and bulk-moisture means as those of one case."""
        if self._batch_index == 0 or self._w_gr_count == 0:
            msg = "Transient metric accumulation requires valid complete-dataset samples for every state."
            raise RuntimeError(msg)
        bulk_count = self._bulk_moisture_valid_count

        def bulk_mean(total: float) -> float | None:
            """Return a bulk-moisture total per valid state, or ``None`` without one."""
            return total / bulk_count if bulk_count else None

        return CaseMoisture(
            scope=self._scope,
            w_gr_squared=self._w_gr_squared_sum / self._w_gr_count,
            w_gr_absolute=self._w_gr_absolute_sum / self._w_gr_count,
            bulk_dry_squared=bulk_mean(self._bulk_dry_squared_sum),
            bulk_dry_absolute=bulk_mean(self._bulk_dry_absolute_sum),
            bulk_wet_squared=bulk_mean(self._bulk_wet_squared_sum),
            bulk_wet_absolute=bulk_mean(self._bulk_wet_absolute_sum),
            predicted_bulk_dry=bulk_mean(self._predicted_bulk_dry_sum),
            reference_bulk_dry=bulk_mean(self._reference_bulk_dry_sum),
            predicted_bulk_wet=bulk_mean(self._predicted_bulk_wet_sum),
            reference_bulk_wet=bulk_mean(self._reference_bulk_wet_sum),
            bulk_valid_count=bulk_count,
            bulk_unavailable_count=self._bulk_moisture_unavailable_count,
        )


def summarize_moisture(cases: Iterable[tuple[str, CaseMoisture]]) -> TransientMetricSummary:
    """
    Reduce the grain-water and bulk-moisture errors of one case population like every reported field metric.

    Parameters
    ----------
    cases : Iterable[tuple[str, CaseMoisture]]
        ``(material, case means)`` of every case of one scope.

    Returns
    -------
    TransientMetricSummary
        Each case's mean over its states (bulk moisture: over its valid states) averaged
        equally over the cases of a material and over the materials
        (:func:`learning.metrics.metrics.material_case_mean`); an RMSE takes its root after
        that mean. A bulk value is ``None`` when a material has no valid bulk state; the
        valid and unavailable bulk counts are state counts summed over the cases.

    Raises
    ------
    ValueError
        If no case is given or the cases mix scopes.

    """
    population = tuple(cases)
    scopes = {moisture.scope for _material, moisture in population}
    if len(scopes) != 1:
        msg = "A moisture summary requires at least one case and one shared scope."
        raise ValueError(msg)

    def reduced(name: str, *, root: bool = False) -> float | None:
        """Return the material-case mean of one case mean, its root for an RMSE."""
        value = learning.metrics.metrics.material_case_mean((material, getattr(moisture, name)) for material, moisture in population)
        return None if value is None else (math.sqrt(value) if root else value)

    return TransientMetricSummary(
        scope=scopes.pop(),
        physical_w_gr_rmse=cast("float", reduced("w_gr_squared", root=True)),
        physical_w_gr_mae=cast("float", reduced("w_gr_absolute")),
        bulk_dry_basis_rmse=reduced("bulk_dry_squared", root=True),
        bulk_dry_basis_mae=reduced("bulk_dry_absolute"),
        bulk_wet_basis_rmse=reduced("bulk_wet_squared", root=True),
        bulk_wet_basis_mae=reduced("bulk_wet_absolute"),
        predicted_bulk_dry_basis_mean=reduced("predicted_bulk_dry"),
        reference_bulk_dry_basis_mean=reduced("reference_bulk_dry"),
        predicted_bulk_wet_basis_mean=reduced("predicted_bulk_wet"),
        reference_bulk_wet_basis_mean=reduced("reference_bulk_wet"),
        bulk_moisture_valid_count=sum(moisture.bulk_valid_count for _material, moisture in population),
        bulk_moisture_unavailable_count=sum(moisture.bulk_unavailable_count for _material, moisture in population),
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
