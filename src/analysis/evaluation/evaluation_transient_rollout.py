"""
Orchestrate transient-drying Evaluation through the public inference service.

The module assembles complete case trajectories from Dataset-owned one-step
items and distinguishes teacher-forced, full-autonomous, and rolling-origin
requests. It does not load checkpoints, construct models, or fit scaling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import torch

from src import domain
from src.learning.inference import learning_inference_transient as inference

from . import evaluation_transient_metrics as transient_metrics
from .evaluation_transient_artifact import (
    BOUNDARY_ORDER,
    FIXED_HORIZONS,
    SCALAR_ORDER,
    STATE_ORDER,
    STATIC_ORDER,
    TARGET_CRITERION,
    DatasetRole,
    TransientSequenceRecord,
    UnavailableHorizon,
)

_MAX_BENCHMARK_REPETITIONS = 32
_SPATIAL_STATE_RANK = 3
_F_SURF_SCALAR_INDEX = SCALAR_ORDER.index("f_surf")
_RHO_BULK_DENSITY_STATIC_INDEX = STATIC_ORDER.index("rho_bu_dry")
_SURFACE_MOISTURE_STATE_INDEX = STATE_ORDER.index("w_surf")
_INTERNAL_MOISTURE_STATE_INDEX = STATE_ORDER.index("w_int")
_FRACTION_ROUNDOFF_TOLERANCE = 4.0 * math.ulp(1.0)


@dataclass(frozen=True, slots=True)
class TransientEvaluationCase:
    """Hold one complete physical regular-time case assembled from Dataset items."""

    case_id: str
    dataset_role: DatasetRole
    physical_times: np.ndarray
    reference_states: np.ndarray
    static_conditioning: np.ndarray
    boundary_conditioning: np.ndarray
    scalar_conditioning: np.ndarray
    spatial_mask: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def transition_count(self) -> int:
        """Return the number of consecutive regular transitions."""
        return int(self.reference_states.shape[0] - 1)


@dataclass(frozen=True, slots=True)
class PreparedTransientEvaluationCase:
    """Hold one case and its immutable inputs on the inference device."""

    case: TransientEvaluationCase
    request: inference.TransientPreparedRequest
    reference_states: torch.Tensor
    _reference_states_version: int
    airflow_source: Literal["comsol_reference", "external"]
    external_airflow: np.ndarray | None


@dataclass(frozen=True, slots=True)
class TransientRolloutEvaluation:
    """Return available sequence records and explicit unsupported horizons."""

    records: tuple[TransientSequenceRecord, ...]
    unavailable_horizons: tuple[UnavailableHorizon, ...]


@dataclass(frozen=True, slots=True)
class TransientRolloutBenchmark:
    """Store cold and warmed full-rollout timing from public inference requests."""

    case_id: str
    warmup_passes: int
    repetitions: int
    model_clock: str
    wall_clock: str
    model_calls_per_repetition: int
    cold_model_seconds: float
    cold_end_to_end_seconds: float
    warmed_model_seconds: tuple[float, ...]
    warmed_end_to_end_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CanonicalTargetCompletion:
    """Hold Generation-admitted completed-case target and censoring evidence."""

    physical_duration_hours: float
    time_to_target_hours: float | None
    target_reached: bool
    right_censored: bool
    final_wet_fraction: float
    target_fraction_limit: float


def _numpy(value: Any, *, label: str) -> np.ndarray:
    """Detach one tensor or admit one numeric array as owned CPU evidence."""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"} or not np.isfinite(array).all():
        message = f"Transient case {label} must contain finite real values."
        raise ValueError(message)
    return np.ascontiguousarray(array)


def _scalar_float(value: Any, *, label: str) -> float:
    """Return one finite scalar from a scalar tensor or numeric value."""
    array = _numpy(value, label=label)
    if array.size != 1:
        message = f"Transient case {label} must be scalar."
        raise ValueError(message)
    return float(array.reshape(-1)[0])


def assemble_transient_evaluation_case(  # noqa: C901, PLR0912
    items: Sequence[Mapping[str, Any]],
    *,
    dataset_role: DatasetRole,
    spatial_mask: np.ndarray | None = None,
) -> TransientEvaluationCase:
    """Assemble one exact contiguous case from ordered physical one-step items."""
    if not items or isinstance(items, (str, bytes)):
        message = "Transient Evaluation case assembly requires ordered one-step items."
        raise ValueError(message)
    if dataset_role not in {"id", "ood"}:
        message = "Transient Evaluation dataset_role must be 'id' or 'ood'."
        raise ValueError(message)
    states: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    times: list[float] = []
    first_static: np.ndarray | None = None
    first_scalars: np.ndarray | None = None
    first_metadata: dict[str, Any] | None = None
    previous_next: np.ndarray | None = None
    previous_index: int | None = None
    case_id: str | None = None
    final_time = math.nan
    for position, item in enumerate(items):
        if not isinstance(item, Mapping) or set(item) != {"state", "static", "boundary", "scalars", "time", "target", "metadata"}:
            message = f"Transient one-step item {position} does not match the Dataset item contract."
            raise ValueError(message)
        metadata = item["metadata"]
        time = item["time"]
        if not isinstance(metadata, Mapping) or not isinstance(time, Mapping):
            message = f"Transient one-step item {position} metadata/time must be mappings."
            raise TypeError(message)
        if metadata.get("sample_mode") != "one_step_transition" or metadata.get("rollout_length") != 1:
            message = "Transient Evaluation assembly accepts only Dataset one-step transition items."
            raise ValueError(message)
        current_case = metadata.get("simulation_case_id")
        if not isinstance(current_case, str) or not current_case:
            message = "Transient Dataset metadata lacks simulation_case_id."
            raise TypeError(message)
        if case_id is None:
            case_id = current_case
        elif current_case != case_id:
            message = "Transient Evaluation case assembly cannot cross simulation cases."
            raise ValueError(message)
        index_n = metadata.get("time_index_n")
        index_next = metadata.get("time_index_n_plus_1")
        if (
            isinstance(index_n, bool)
            or not isinstance(index_n, int)
            or isinstance(index_next, bool)
            or not isinstance(index_next, int)
            or index_next != index_n + 1
            or (previous_index is not None and index_n != previous_index + 1)
        ):
            message = "Transient Evaluation items must form one contiguous transition chain."
            raise ValueError(message)
        state = _numpy(item["state"], label=f"item[{position}].state").astype(np.float32, copy=False)
        target = _numpy(item["target"], label=f"item[{position}].target").astype(np.float32, copy=False)
        static = _numpy(item["static"], label=f"item[{position}].static").astype(np.float32, copy=False)
        boundary = _numpy(item["boundary"], label=f"item[{position}].boundary").astype(np.float32, copy=False)
        scalars = _numpy(item["scalars"], label=f"item[{position}].scalars").astype(np.float32, copy=False)
        if state.ndim != _SPATIAL_STATE_RANK or state.shape[0] != len(STATE_ORDER) or target.shape != state.shape:
            message = "Transient Evaluation state and increment must share [4,Y,X] shape."
            raise ValueError(message)
        if (
            static.shape != (len(STATIC_ORDER), *state.shape[-2:])
            or boundary.shape != (len(BOUNDARY_ORDER),)
            or scalars.shape != (len(SCALAR_ORDER),)
        ):
            message = "Transient Evaluation conditioning does not match the canonical Dataset field-group sizes."
            raise ValueError(message)
        next_state = np.ascontiguousarray(state + target, dtype=np.float32)
        if previous_next is not None and not np.allclose(state, previous_next, rtol=0.0, atol=2.0e-5):
            message = "Consecutive Dataset transitions disagree at their shared absolute state."
            raise ValueError(message)
        t_n = _scalar_float(time.get("t_n"), label=f"item[{position}].time.t_n")
        t_next = _scalar_float(time.get("t_n_plus_1"), label=f"item[{position}].time.t_n_plus_1")
        dt = _scalar_float(time.get("dt"), label=f"item[{position}].time.dt")
        if not t_next > t_n or not math.isclose(t_next - t_n, dt, rel_tol=0.0, abs_tol=1.0e-6):
            message = "Transient Dataset time endpoints and dt are inconsistent."
            raise ValueError(message)
        if position and not math.isclose(t_n, final_time, rel_tol=0.0, abs_tol=1.0e-6):
            message = "Transient Dataset physical times are not contiguous."
            raise ValueError(message)
        if first_static is None:
            first_static = static
            first_scalars = scalars
            first_metadata = dict(metadata)
            times.append(t_n)
        elif first_scalars is None:
            message = "Transient Evaluation scalar initialization is inconsistent."
            raise RuntimeError(message)
        elif not np.array_equal(static, first_static) or not np.array_equal(scalars, first_scalars):
            message = "Static and scalar conditioning changed within one transient case."
            raise ValueError(message)
        states.append(state)
        boundaries.append(boundary)
        times.append(t_next)
        previous_next = next_state
        previous_index = index_n
        final_time = t_next
    if case_id is None or first_static is None or first_scalars is None or first_metadata is None or previous_next is None:
        message = "Transient Evaluation case assembly produced no materialized evidence."
        raise RuntimeError(message)
    states.append(previous_next)
    state_array = np.stack(states, axis=0)
    expected_states = first_metadata.get("sequence_length")
    if isinstance(expected_states, bool) or not isinstance(expected_states, int) or expected_states != state_array.shape[0]:
        message = "Assembled trajectory length contradicts Dataset case metadata."
        raise ValueError(message)
    if spatial_mask is None:
        admitted_mask = np.ones(state_array.shape[-2:], dtype=bool)
    else:
        admitted_mask = np.asarray(spatial_mask)
        if admitted_mask.dtype != np.bool_ or admitted_mask.shape != state_array.shape[-2:] or not admitted_mask.any():
            message = "Transient Evaluation spatial mask must be non-empty boolean [Y,X]."
            raise ValueError(message)
        admitted_mask = np.ascontiguousarray(admitted_mask)
    return TransientEvaluationCase(
        case_id=case_id,
        dataset_role=dataset_role,
        physical_times=np.asarray(times, dtype=np.float64),
        reference_states=state_array,
        static_conditioning=np.ascontiguousarray(first_static),
        boundary_conditioning=np.stack(boundaries, axis=0),
        scalar_conditioning=np.ascontiguousarray(first_scalars),
        spatial_mask=admitted_mask,
        metadata=first_metadata,
    )


def default_rolling_origins(transition_count: int) -> tuple[int, ...]:
    """Return unique early, middle, and late origins with future support."""
    if isinstance(transition_count, bool) or not isinstance(transition_count, int) or transition_count < 1:
        message = "transition_count must be a positive integer."
        raise ValueError(message)
    return tuple(dict.fromkeys((0, (transition_count - 1) // 2, transition_count - 1)))


def _tensor(value: np.ndarray, *, add_batch: bool = True) -> torch.Tensor:
    """Create one contiguous float32 CPU request tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    return tensor.unsqueeze(0) if add_batch else tensor


def _request(
    case: TransientEvaluationCase,
    *,
    origin: int,
    length: int,
    state: np.ndarray,
    airflow_source: Literal["comsol_reference", "external"],
    external_airflow: np.ndarray | None,
) -> dict[str, torch.Tensor | str | None]:
    """Build one exact public-inference request over a case-local interval."""
    stop = origin + length
    airflow = None
    if external_airflow is not None:
        admitted_airflow = np.asarray(external_airflow, dtype=np.float32)
        if admitted_airflow.shape != (3, *case.reference_states.shape[-2:]) or not np.isfinite(admitted_airflow).all():
            message = "External airflow must be finite [3,Y,X] evidence on the identical case grid."
            raise ValueError(message)
        airflow = _tensor(admitted_airflow)
    return {
        "state": _tensor(state),
        "static": _tensor(case.static_conditioning),
        "boundary": _tensor(case.boundary_conditioning[origin:stop]),
        "scalars": _tensor(case.scalar_conditioning),
        "t_n": _tensor(case.physical_times[origin:stop], add_batch=True),
        "t_next": _tensor(case.physical_times[origin + 1 : stop + 1], add_batch=True),
        "dt": _tensor(np.diff(case.physical_times[origin : stop + 1]), add_batch=True),
        "airflow_source": airflow_source,
        "external_airflow": airflow,
    }


def prepare_transient_evaluation_case(
    context: inference.TransientInferenceContext,
    case: TransientEvaluationCase,
    *,
    airflow_source: Literal["comsol_reference", "external"] = "comsol_reference",
    external_airflow: np.ndarray | None = None,
) -> PreparedTransientEvaluationCase:
    """Validate and move one complete case once for benchmark and record inference."""
    if not isinstance(context, inference.TransientInferenceContext) or not isinstance(
        case,
        TransientEvaluationCase,
    ):
        message = "Transient preparation requires admitted inference context and case evidence."
        raise TypeError(message)
    request = _request(
        case,
        origin=0,
        length=case.transition_count,
        state=case.reference_states[0],
        airflow_source=airflow_source,
        external_airflow=external_airflow,
    )
    prepared = inference.prepare_transient_request(
        context,
        **cast("dict[str, Any]", request),
    )
    reference_states = _tensor(case.reference_states).to(
        device=context.device,
        dtype=torch.float32,
        copy=True,
    )
    admitted_airflow = None if external_airflow is None else np.array(external_airflow, dtype=np.float32, copy=True, order="C")
    return PreparedTransientEvaluationCase(
        case=case,
        request=prepared,
        reference_states=reference_states,
        _reference_states_version=reference_states._version,  # noqa: SLF001 -- PyTorch mutation evidence
        airflow_source=airflow_source,
        external_airflow=admitted_airflow,
    )


def _prepared_evaluation_case(
    context: inference.TransientInferenceContext,
    case: TransientEvaluationCase,
    *,
    prepared_case: PreparedTransientEvaluationCase | None,
    airflow_source: Literal["comsol_reference", "external"],
    external_airflow: np.ndarray | None,
) -> PreparedTransientEvaluationCase:
    """Create or admit one exact reusable case-runtime binding."""
    if prepared_case is None:
        return prepare_transient_evaluation_case(
            context,
            case,
            airflow_source=airflow_source,
            external_airflow=external_airflow,
        )
    if (
        not isinstance(prepared_case, PreparedTransientEvaluationCase)
        or prepared_case.case is not case
        or prepared_case.request.context is not context
        or prepared_case.airflow_source != airflow_source
    ):
        message = "Prepared transient case contradicts the requested context, case, or airflow policy."
        raise ValueError(message)
    if prepared_case.reference_states.device != context.device or prepared_case.reference_states.dtype != torch.float32:
        message = "Prepared transient reference-state placement drifted from its inference context."
        raise RuntimeError(message)
    if prepared_case.reference_states._version != prepared_case._reference_states_version:  # noqa: SLF001 -- owned mutation evidence
        message = "Prepared transient reference states were mutated after strict admission."
        raise RuntimeError(message)
    admitted_airflow = prepared_case.external_airflow
    if external_airflow is not None:
        candidate = np.asarray(external_airflow, dtype=np.float32)
        if admitted_airflow is None or not np.array_equal(candidate, admitted_airflow):
            message = "Prepared transient case contradicts external airflow evidence."
            raise ValueError(message)
    elif airflow_source == "external" and admitted_airflow is None:
        message = "Prepared external-airflow case lacks its airflow evidence."
        raise ValueError(message)
    return prepared_case


def _prepared_request(
    prepared_case: PreparedTransientEvaluationCase,
    *,
    origin: int,
    length: int,
) -> inference.TransientPreparedRequest:
    """Return one zero-copy request window with the exact reference current state."""
    return inference.window_transient_prepared_request(
        prepared_case.request,
        origin=origin,
        length=length,
        state=prepared_case.reference_states[:, origin],
    )


def benchmark_transient_full_rollout(
    context: inference.TransientInferenceContext,
    case: TransientEvaluationCase,
    *,
    airflow_source: Literal["comsol_reference", "external"] = "comsol_reference",
    external_airflow: np.ndarray | None = None,
    prepared_case: PreparedTransientEvaluationCase | None = None,
    warmup_passes: int = 1,
    repetitions: int = 3,
) -> TransientRolloutBenchmark:
    """Measure bounded independent full rollouts with model and monotonic wall clocks."""
    if not isinstance(context, inference.TransientInferenceContext) or not isinstance(case, TransientEvaluationCase):
        message = "Transient benchmark requires admitted inference context and case evidence."
        raise TypeError(message)
    if (
        isinstance(warmup_passes, bool)
        or not isinstance(warmup_passes, int)
        or warmup_passes < 0
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 1 <= repetitions <= _MAX_BENCHMARK_REPETITIONS
    ):
        message = "Transient benchmark requires non-negative warmup and 1..32 warmed repetitions."
        raise ValueError(message)
    runtime_case = _prepared_evaluation_case(
        context,
        case,
        prepared_case=prepared_case,
        airflow_source=airflow_source,
        external_airflow=external_airflow,
    )
    request = _prepared_request(
        runtime_case,
        origin=0,
        length=case.transition_count,
    )

    def invoke() -> tuple[float, float]:
        """Run one independent prepared request and retain factual model/wall clocks."""
        started = perf_counter()
        result = inference.rollout_prepared_transient_autonomous(
            context,
            request,
        )
        elapsed = perf_counter() - started
        model_seconds = float(result.timing.model_seconds)
        if not math.isfinite(model_seconds) or model_seconds < 0.0 or not math.isfinite(elapsed) or elapsed <= 0.0:
            message = "Transient benchmark produced invalid model or wall-clock duration."
            raise RuntimeError(message)
        return model_seconds, elapsed

    cold_model, cold_wall = invoke()
    for _ in range(warmup_passes):
        invoke()
    warmed_model: list[float] = []
    warmed_wall: list[float] = []
    for _ in range(repetitions):
        model_seconds, wall_seconds = invoke()
        warmed_model.append(model_seconds)
        warmed_wall.append(wall_seconds)
    return TransientRolloutBenchmark(
        case_id=case.case_id,
        warmup_passes=warmup_passes,
        repetitions=repetitions,
        model_clock=("torch.cuda.Event" if context.device.type == "cuda" else "time.perf_counter"),
        wall_clock="time.perf_counter",
        model_calls_per_repetition=case.transition_count,
        cold_model_seconds=cold_model,
        cold_end_to_end_seconds=cold_wall,
        warmed_model_seconds=tuple(warmed_model),
        warmed_end_to_end_seconds=tuple(warmed_wall),
    )


def _admit_canonical_target_completion(
    value: Mapping[str, Any],
    *,
    expected_limit: float,
) -> CanonicalTargetCompletion:
    """Admit exact completed-case target evidence from the Generation timing owner."""
    required = {
        "physical_duration_hours",
        "time_to_target_hours",
        "target_reached",
        "right_censored",
        "final_wet_fraction",
        "target_wet_fraction_limit",
        "physical_duration_availability",
        "target_wet_fraction_limit_availability",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        message = "Canonical reference completion fields do not match the Generation timing contract."
        raise ValueError(message)
    if value["physical_duration_availability"] != "available" or value["target_wet_fraction_limit_availability"] != "available":
        message = "Transient Evaluation requires available canonical duration and target-limit evidence."
        raise ValueError(message)

    def finite(value_0: Any, *, label: str, unit_interval: bool = False) -> float:
        """Return one finite non-negative completion scalar."""
        if isinstance(value_0, bool) or not isinstance(value_0, (int, float)):
            message_0 = f"Canonical reference completion {label} must be a real scalar."
            raise TypeError(message_0)
        result = float(value_0)
        above_unit_interval = (
            unit_interval
            and result > 1.0
            and not math.isclose(
                result,
                1.0,
                rel_tol=0.0,
                abs_tol=_FRACTION_ROUNDOFF_TOLERANCE,
            )
        )
        if not math.isfinite(result) or result < 0.0 or above_unit_interval:
            message_0 = f"Canonical reference completion {label} is outside its physical range."
            raise ValueError(message_0)
        return result

    duration = finite(value["physical_duration_hours"], label="physical_duration_hours")
    final_fraction = finite(value["final_wet_fraction"], label="final_wet_fraction", unit_interval=True)
    limit = finite(value["target_wet_fraction_limit"], label="target_wet_fraction_limit", unit_interval=True)
    if not 0.0 < limit < 1.0 or not math.isclose(limit, expected_limit, rel_tol=0.0, abs_tol=1.0e-12):
        message = "Canonical reference completion target limit contradicts the Evaluation request."
        raise ValueError(message)
    reached = value["target_reached"]
    censored = value["right_censored"]
    if not isinstance(reached, bool) or not isinstance(censored, bool) or reached is censored:
        message = "Canonical reference reached and right-censored states must be complementary booleans."
        raise TypeError(message)
    raw_time = value["time_to_target_hours"]
    if reached:
        target_time = finite(raw_time, label="time_to_target_hours")
        if not math.isclose(target_time, duration, rel_tol=0.0, abs_tol=1.0e-12):
            message = "Canonical reached-only target time must equal the completed physical duration."
            raise ValueError(message)
    elif raw_time is not None:
        message = "Canonical right-censored completion must not fabricate a target time."
        raise ValueError(message)
    else:
        target_time = None
    if reached != (final_fraction <= limit):
        message = "Canonical reference completion status contradicts its signed final target gap."
        raise ValueError(message)
    return CanonicalTargetCompletion(
        physical_duration_hours=duration,
        time_to_target_hours=target_time,
        target_reached=reached,
        right_censored=censored,
        final_wet_fraction=final_fraction,
        target_fraction_limit=limit,
    )


def _wet_fraction_series(
    states: np.ndarray,
    *,
    static: np.ndarray,
    scalars: np.ndarray,
    mask: np.ndarray,
    target_wet_basis: float,
) -> np.ndarray:
    """Return canonical dry-solid-mass fraction above the local wet-basis target."""
    if not 0.0 < target_wet_basis < 1.0:
        message = "target_wet_basis must lie strictly inside (0, 1)."
        raise ValueError(message)
    f_surf = float(scalars[_F_SURF_SCALAR_INDEX])
    if not 0.0 <= f_surf <= 1.0:
        message = "f_surf must lie in [0, 1]."
        raise ValueError(message)
    rho = np.asarray(static[_RHO_BULK_DENSITY_STATIC_INDEX], dtype=np.float64)
    if np.any(rho[mask] <= 0.0):
        message = "Packed-bed dry bulk density must be positive on the Evaluation mask."
        raise ValueError(message)
    cell_weights = transient_metrics.trapezoidal_cell_weights(mask)
    admitted_rho = rho[mask]
    admitted_weights = cell_weights[mask]
    values: list[float] = []
    dry_total = float(np.sum(admitted_rho * admitted_weights, dtype=np.float64))
    for state in states:
        w_gr = f_surf * np.asarray(state[_SURFACE_MOISTURE_STATE_INDEX], dtype=np.float64) + (1.0 - f_surf) * np.asarray(
            state[_INTERNAL_MOISTURE_STATE_INDEX],
            dtype=np.float64,
        )
        admitted_water = w_gr[mask]
        if np.any(admitted_water < 0.0):
            values.append(math.nan)
            continue
        wet_basis = domain.moisture.wet_basis_moisture(admitted_water, admitted_rho)
        wet = wet_basis > target_wet_basis
        wet_dry_mass = float(np.sum(admitted_rho[wet] * admitted_weights[wet], dtype=np.float64))
        values.append(wet_dry_mass / dry_total)
    return np.asarray(values, dtype=np.float64)


def _target_evidence(
    predicted: np.ndarray,
    times: np.ndarray,
    *,
    static: np.ndarray,
    scalars: np.ndarray,
    mask: np.ndarray,
    target_wet_basis: float,
    target_fraction_limit: float,
    reference_completion: CanonicalTargetCompletion | None,
) -> dict[str, Any]:
    """Build distinct canonical-reference and regular-grid prediction target evidence."""
    if not 0.0 < target_fraction_limit < 1.0:
        message = "target_fraction_limit must lie strictly inside (0, 1)."
        raise ValueError(message)
    predicted_fraction = _wet_fraction_series(
        predicted,
        static=static,
        scalars=scalars,
        mask=mask,
        target_wet_basis=target_wet_basis,
    )

    def classify(values: np.ndarray) -> tuple[bool, bool, str | None, float | None, float | None, float | None]:
        """Classify one prediction on its exact regular sequence grid."""
        if not bool(np.isfinite(values).all()):
            return False, False, "target_fraction_unavailable_for_nonphysical_state", None, None, None
        reached_positions = np.flatnonzero(values <= target_fraction_limit)
        reached = bool(reached_positions.size)
        target_time = float(times[int(reached_positions[0])]) if reached else None
        return True, reached, None, target_time, float(values[-1] - target_fraction_limit), float(times[-1])

    predicted_available, predicted_reached, predicted_reason, predicted_time, predicted_gap, predicted_final_time = classify(predicted_fraction)
    if reference_completion is None:
        reference = {
            "reference_evidence_scope": "unavailable_partial_interval",
            "reference_available": False,
            "reference_unavailable_reason": "canonical_completed_case_target_unavailable_for_partial_interval",
            "reference_reached": False,
            "reference_time_to_target": None,
            "reference_censored": False,
            "reference_final_gap": None,
            "reference_final_time": None,
        }
    else:
        reference = {
            "reference_evidence_scope": "canonical_completed_case",
            "reference_available": True,
            "reference_unavailable_reason": None,
            "reference_reached": reference_completion.target_reached,
            "reference_time_to_target": reference_completion.time_to_target_hours,
            "reference_censored": reference_completion.right_censored,
            "reference_final_gap": reference_completion.final_wet_fraction - reference_completion.target_fraction_limit,
            "reference_final_time": reference_completion.physical_duration_hours,
        }
    return {
        "criterion": TARGET_CRITERION,
        "limit": float(target_fraction_limit),
        **reference,
        "predicted_evidence_scope": "regular_sequence_grid",
        "predicted_available": predicted_available,
        "predicted_unavailable_reason": predicted_reason,
        "predicted_reached": predicted_reached,
        "predicted_time_to_target": predicted_time,
        "predicted_censored": predicted_available and not predicted_reached,
        "predicted_final_gap": predicted_gap,
        "predicted_final_time": predicted_final_time,
    }


def _record(
    *,
    case: TransientEvaluationCase,
    mode: Literal["teacher_forced_one_step", "autonomous_full", "rolling_origin"],
    origin: int,
    requested_horizon: int | Literal["full"],
    predicted: np.ndarray,
    identity: Mapping[str, Any],
    timing: Mapping[str, Any],
    target_wet_basis: float,
    target_fraction_limit: float,
    reference_completion: CanonicalTargetCompletion | None,
) -> TransientSequenceRecord:
    """Construct one validated artifact record for an available interval."""
    carries_completion = mode == "autonomous_full" and origin == 0 and requested_horizon == "full"
    if carries_completion is not (reference_completion is not None):
        message = "Only the full-autonomous record may carry canonical completed-case target evidence."
        raise ValueError(message)
    length = int(predicted.shape[0] - 1)
    stop = origin + length
    reference = np.ascontiguousarray(case.reference_states[origin : stop + 1])
    times = np.ascontiguousarray(case.physical_times[origin : stop + 1])
    target = _target_evidence(
        predicted,
        times,
        static=case.static_conditioning,
        scalars=case.scalar_conditioning,
        mask=case.spatial_mask,
        target_wet_basis=target_wet_basis,
        target_fraction_limit=target_fraction_limit,
        reference_completion=reference_completion,
    )
    return TransientSequenceRecord(
        mode=mode,
        case_id=case.case_id,
        dataset_role=case.dataset_role,
        origin_index=origin,
        requested_horizon=requested_horizon,
        available_horizon=length,
        trajectory_length=case.transition_count + 1,
        physical_times=times,
        transition_indices=np.arange(origin, stop, dtype=np.int64),
        reference_states=reference,
        predicted_states=np.ascontiguousarray(predicted, dtype=np.float32),
        reference_increments=np.diff(reference, axis=0),
        predicted_increments=np.diff(predicted, axis=0),
        spatial_mask=case.spatial_mask,
        temporal_mask=np.ones(length + 1, dtype=bool),
        static_conditioning=case.static_conditioning,
        boundary_conditioning=case.boundary_conditioning[origin:stop],
        scalar_conditioning=case.scalar_conditioning,
        identity=identity,
        target=target,
        timing=timing,
        exclusion={"excluded": False, "reason": None},
    ).validated()


def evaluate_transient_case(
    context: inference.TransientInferenceContext,
    case: TransientEvaluationCase,
    *,
    identity: Mapping[str, Any],
    reference_completion: Mapping[str, Any],
    target_wet_basis: float,
    target_fraction_limit: float,
    airflow_source: Literal["comsol_reference", "external"] = "comsol_reference",
    external_airflow: np.ndarray | None = None,
    prepared_case: PreparedTransientEvaluationCase | None = None,
    rolling_origins: Sequence[int] | None = None,
    fixed_horizons: Sequence[int] = FIXED_HORIZONS,
) -> TransientRolloutEvaluation:
    """Evaluate all required modes while resetting each public autonomous request."""
    if not isinstance(context, inference.TransientInferenceContext):
        message = "context must be one public TransientInferenceContext."
        raise TypeError(message)
    if context.model_kind not in {"fno", "uno", "rno"}:
        message = "Transient Evaluation supports only FNO, U-NO, and official RNO contexts."
        raise ValueError(message)
    if airflow_source == "comsol_reference" and external_airflow is not None:
        message = "COMSOL-airflow Evaluation cannot accept external airflow fields."
        raise ValueError(message)
    if airflow_source == "external" and external_airflow is None:
        message = "External-airflow Evaluation requires explicit u, v, p fields."
        raise ValueError(message)
    horizons = tuple(fixed_horizons)
    if (
        not horizons
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in horizons)
        or len(set(horizons)) != len(horizons)
        or tuple(sorted(horizons)) != horizons
    ):
        message = "fixed_horizons must be unique increasing positive integers."
        raise ValueError(message)
    canonical_completion = _admit_canonical_target_completion(
        reference_completion,
        expected_limit=target_fraction_limit,
    )
    regular_end = float(case.physical_times[-1])
    regular_step = float(case.physical_times[-1] - case.physical_times[-2])
    tolerance = 1.0e-6
    metadata_stop = case.metadata.get("t_stop_exact")
    metadata_has_exact = case.metadata.get("has_exact_stop_state")
    if (
        isinstance(metadata_stop, bool)
        or not isinstance(metadata_stop, (int, float))
        or not math.isclose(canonical_completion.physical_duration_hours, float(metadata_stop), rel_tol=0.0, abs_tol=tolerance)
        or not isinstance(metadata_has_exact, bool)
        or metadata_has_exact is not (canonical_completion.physical_duration_hours > regular_end + tolerance)
        or canonical_completion.physical_duration_hours < regular_end - tolerance
        or canonical_completion.physical_duration_hours > regular_end + regular_step + tolerance
    ):
        message = "Canonical target completion contradicts the Dataset regular/exact-stop time contract."
        raise ValueError(message)

    admitted_identity = dict(identity)
    if admitted_identity.get("case_id") != case.case_id or admitted_identity.get("dataset_role") != case.dataset_role:
        message = "Rollout identity must bind the exact case and Dataset role."
        raise ValueError(message)
    if admitted_identity.get("model_kind") != context.model_kind or admitted_identity.get("inference_airflow_source") != airflow_source:
        message = "Rollout identity contradicts the admitted model or airflow policy."
        raise ValueError(message)
    runtime_case = _prepared_evaluation_case(
        context,
        case,
        prepared_case=prepared_case,
        airflow_source=airflow_source,
        external_airflow=external_airflow,
    )

    records: list[TransientSequenceRecord] = []
    unavailable: list[UnavailableHorizon] = []
    for origin in range(case.transition_count):
        request = _prepared_request(
            runtime_case,
            origin=origin,
            length=1,
        )
        result = inference.predict_prepared_transient_step(context, request)
        predicted = np.concatenate(
            (
                case.reference_states[origin : origin + 1],
                result.next_state.detach().cpu().numpy(),
            ),
            axis=0,
        )
        records.append(
            _record(
                case=case,
                mode="teacher_forced_one_step",
                origin=origin,
                requested_horizon=1,
                predicted=predicted,
                identity=admitted_identity,
                timing={
                    "kind": "model_only_public_inference",
                    "seconds": result.timing.model_seconds,
                    "model_calls": result.timing.model_calls,
                    "device": result.timing.device,
                    "precision": result.timing.precision,
                },
                target_wet_basis=target_wet_basis,
                target_fraction_limit=target_fraction_limit,
                reference_completion=None,
            )
        )

    full_request = _prepared_request(
        runtime_case,
        origin=0,
        length=case.transition_count,
    )
    full_result = inference.rollout_prepared_transient_autonomous(
        context,
        full_request,
    )
    full_predicted = np.concatenate(
        (case.reference_states[0:1], full_result.states.detach().cpu().numpy()[0]),
        axis=0,
    )
    records.append(
        _record(
            case=case,
            mode="autonomous_full",
            origin=0,
            requested_horizon="full",
            predicted=full_predicted,
            identity=admitted_identity,
            timing={
                "kind": "model_only_public_inference",
                "seconds": full_result.timing.model_seconds,
                "model_calls": full_result.timing.model_calls,
                "device": full_result.timing.device,
                "precision": full_result.timing.precision,
            },
            target_wet_basis=target_wet_basis,
            target_fraction_limit=target_fraction_limit,
            reference_completion=canonical_completion,
        )
    )

    origins = default_rolling_origins(case.transition_count) if rolling_origins is None else tuple(rolling_origins)
    if (
        not origins
        or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < case.transition_count for value in origins)
        or len(set(origins)) != len(origins)
        or tuple(sorted(origins)) != origins
    ):
        message = "rolling_origins must be unique ordered case-local transition indices."
        raise ValueError(message)
    for origin in origins:
        remaining = case.transition_count - origin
        origin_request = _prepared_request(
            runtime_case,
            origin=origin,
            length=remaining,
        )
        origin_result = inference.rollout_prepared_transient_autonomous(
            context,
            origin_request,
        )
        origin_predicted = np.concatenate(
            (case.reference_states[origin : origin + 1], origin_result.states.detach().cpu().numpy()[0]),
            axis=0,
        )
        for horizon in horizons:
            if remaining < horizon:
                unavailable.append(
                    UnavailableHorizon(
                        case_id=case.case_id,
                        dataset_role=case.dataset_role,
                        origin_index=origin,
                        requested_horizon=horizon,
                        available_transitions=remaining,
                        reason="requested_fixed_horizon_exceeds_case_future_support",
                    )
                )
                continue
            records.append(
                _record(
                    case=case,
                    mode="rolling_origin",
                    origin=origin,
                    requested_horizon=horizon,
                    predicted=origin_predicted[: horizon + 1],
                    identity=admitted_identity,
                    timing={
                        "kind": "shared_origin_rollout_prefix",
                        "seconds": None,
                        "model_calls": horizon,
                        "device": origin_result.timing.device,
                        "precision": origin_result.timing.precision,
                        "source_full_request_seconds": origin_result.timing.model_seconds,
                        "source_full_request_model_calls": origin_result.timing.model_calls,
                    },
                    target_wet_basis=target_wet_basis,
                    target_fraction_limit=target_fraction_limit,
                    reference_completion=None,
                )
            )
        records.append(
            _record(
                case=case,
                mode="rolling_origin",
                origin=origin,
                requested_horizon="full",
                predicted=origin_predicted,
                identity=admitted_identity,
                timing={
                    "kind": "model_only_public_inference",
                    "seconds": origin_result.timing.model_seconds,
                    "model_calls": origin_result.timing.model_calls,
                    "device": origin_result.timing.device,
                    "precision": origin_result.timing.precision,
                },
                target_wet_basis=target_wet_basis,
                target_fraction_limit=target_fraction_limit,
                reference_completion=None,
            )
        )
    return TransientRolloutEvaluation(records=tuple(records), unavailable_horizons=tuple(unavailable))
