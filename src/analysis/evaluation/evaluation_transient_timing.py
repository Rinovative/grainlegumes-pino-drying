"""
evaluation_transient_timing.py

Build strict transient drying timing and speedup summaries.

Responsibilities:
  - Validate bounded raw timing repetitions and disclosure metadata
  - Derive per-case component-composed speedups from named timing components
  - Aggregate speedups as ratios of summed paired durations

Design principles:
  - Missing timing inputs remain unavailable with a precise reason
  - Model-only and wall-clock components remain distinct
  - Aggregates never substitute arithmetic mean ratios for ratio-of-sums

This module does NOT:
  - Measure timings, parse logs, or access persisted artifacts
  - Infer unavailable timing components from unrelated measurements
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import cast

import numpy as np

_MAX_RAW_REPETITIONS = 10_000
TIMING_COMPONENTS = frozenset(
    {
        "airflow_no_model_seconds",
        "airflow_no_preprocessing_seconds",
        "airflow_no_device_transfer_seconds",
        "airflow_no_postprocessing_seconds",
        "airflow_no_end_to_end_seconds",
        "drying_no_rollout_model_seconds",
        "drying_no_preprocessing_seconds",
        "drying_no_device_transfer_seconds",
        "drying_no_postprocessing_seconds",
        "drying_no_end_to_end_seconds",
        "surrogate_pipeline_model_seconds",
        "surrogate_pipeline_end_to_end_seconds",
        "comsol_transient_drying_seconds",
        "comsol_scientific_solver_seconds",
        "comsol_stationary_airflow_seconds",
        "comsol_process_seconds",
        "generation_compute_end_to_end_seconds",
    }
)
_SPEEDUP_FORMULAS = {
    "drying_only_solver_speedup": ("comsol_transient_drying_seconds", ("drying_no_rollout_model_seconds",)),
    "full_pipeline_solver_speedup": ("comsol_scientific_solver_seconds", ("airflow_no_model_seconds", "drying_no_rollout_model_seconds")),
    "hybrid_component_speedup": ("comsol_scientific_solver_seconds", ("comsol_stationary_airflow_seconds", "drying_no_rollout_model_seconds")),
    "comsol_process_speedup": ("comsol_process_seconds", ("surrogate_pipeline_end_to_end_seconds",)),
    "generation_compute_end_to_end_speedup": ("generation_compute_end_to_end_seconds", ("surrogate_pipeline_end_to_end_seconds",)),
}


@dataclass(frozen=True, slots=True)
class TransientTimingCase:
    """Store bounded raw timing repetitions and immutable disclosure for one case."""

    case_id: str
    repetitions: Mapping[str, tuple[float, ...]]
    device: str
    precision: str
    dataset_backend: str
    warmup_passes: int
    batch_size: int = 1
    cpu: str = "unknown"
    gpu: str | None = None
    software_versions: Mapping[str, str] = field(default_factory=dict)
    cold_timings: Mapping[str, float] = field(default_factory=dict)
    unavailable_reasons: Mapping[str, str] = field(default_factory=dict)
    pt_payload_identity: str | None = None


@dataclass(frozen=True, slots=True)
class SpeedupCaseValue:
    """Store one paired speedup value or its exact unavailable reason."""

    case_id: str
    reference_seconds: float | None
    surrogate_seconds: float | None
    speedup: float | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class SpeedupSummary:
    """Store distributions and the authoritative ratio-of-sums aggregate."""

    name: str
    total_count: int
    available_count: int
    unavailable_count: int
    minimum: float | None
    median: float | None
    mean: float | None
    q10: float | None
    q25: float | None
    q75: float | None
    q90: float | None
    q95: float | None
    maximum: float | None
    ratio_of_sums: float | None
    cases: tuple[SpeedupCaseValue, ...]


@dataclass(frozen=True, slots=True)
class TransientTimingReport:
    """Store all five required speedup summaries and component-composition disclosure."""

    cases: tuple[TransientTimingCase, ...]
    speedups: Mapping[str, SpeedupSummary]
    component_composed: bool


def _duration(value: object, *, label: str) -> float:
    """Validate one strictly positive finite duration."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real duration."
        raise TypeError(msg)
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        msg = f"{label} must be finite and positive."
        raise ValueError(msg)
    return result


def _case_component(case: TransientTimingCase, name: str) -> tuple[float | None, str | None]:
    """Return one warmed representative component duration or an unavailable reason."""
    raw = case.repetitions.get(name)
    if raw is None:
        reason = case.unavailable_reasons.get(name)
        return None, reason or f"not_recorded_in_timing_evidence:{name}"
    if not isinstance(raw, tuple) or not raw:
        msg = f"Timing repetitions for {case.case_id!r}/{name!r} must be a non-empty tuple."
        raise ValueError(msg)
    values = [_duration(value, label=f"{case.case_id}.{name}") for value in raw]
    return float(np.median(np.asarray(values, dtype=np.float64))), None


def component_case_medians(
    report: TransientTimingReport,
    component_name: str,
) -> Mapping[str, float]:
    """Return warmed per-case medians for one declared timing component."""
    if not isinstance(report, TransientTimingReport):
        msg = "Timing component selection requires one TransientTimingReport."
        raise TypeError(msg)
    if component_name not in TIMING_COMPONENTS:
        msg = f"Unknown transient timing component {component_name!r}."
        raise ValueError(msg)
    values: dict[str, float] = {}
    for case in report.cases:
        value, _reason = _case_component(case, component_name)
        if value is not None:
            values[case.case_id] = value
    return values


def _summary(name: str, cases: Sequence[TransientTimingCase]) -> SpeedupSummary:
    """Build one required speedup formula from paired case timing components."""
    reference_name, surrogate_names = _SPEEDUP_FORMULAS[name]
    values: list[SpeedupCaseValue] = []
    for case in cases:
        reference, reference_reason = _case_component(case, reference_name)
        surrogate_parts = [_case_component(case, component) for component in surrogate_names]
        reasons = [reason for _, reason in surrogate_parts if reason is not None]
        if reference_reason is not None:
            reasons.insert(0, reference_reason)
        if reasons:
            values.append(SpeedupCaseValue(case.case_id, reference, None, None, "; ".join(reasons)))
            continue
        if reference is None:
            msg = "Available timing formula unexpectedly lacks its reference component."
            raise RuntimeError(msg)
        admitted_surrogate_parts: list[float] = []
        for value, _reason in surrogate_parts:
            if value is None:
                msg = "Available timing formula unexpectedly lacks a surrogate component."
                raise RuntimeError(msg)
            admitted_surrogate_parts.append(value)
        surrogate = float(sum(admitted_surrogate_parts))
        values.append(SpeedupCaseValue(case.case_id, reference, surrogate, reference / surrogate, None))
    available = [item for item in values if item.speedup is not None]
    if not available:
        distribution: dict[str, float | None] = dict.fromkeys(("minimum", "median", "mean", "q10", "q25", "q75", "q90", "q95", "maximum"))
        ratio = None
    else:
        array = np.asarray([item.speedup for item in available], dtype=np.float64)
        distribution = {
            "minimum": float(np.min(array)),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "q10": float(np.quantile(array, 0.10)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "q90": float(np.quantile(array, 0.90)),
            "q95": float(np.quantile(array, 0.95)),
            "maximum": float(np.max(array)),
        }
        ratio = float(
            sum(item.reference_seconds for item in available if item.reference_seconds is not None)
            / sum(item.surrogate_seconds for item in available if item.surrogate_seconds is not None)
        )
    return SpeedupSummary(name, len(values), len(available), len(values) - len(available), ratio_of_sums=ratio, cases=tuple(values), **distribution)


def build_transient_timing_report(cases: Sequence[TransientTimingCase]) -> TransientTimingReport:
    """Validate cases and derive all required component-composed speedup reports."""
    if not cases:
        msg = "Transient timing requires at least one case."
        raise ValueError(msg)
    identifiers: set[str] = set()
    for case in cases:
        if not isinstance(case, TransientTimingCase) or not case.case_id or case.case_id in identifiers:
            msg = "Timing cases require unique non-empty identifiers."
            raise ValueError(msg)
        identifiers.add(case.case_id)
        if not case.device or not case.precision or case.dataset_backend not in {"canonical_hdf5", "pt_shards"}:
            msg = "Timing cases must disclose device, precision, and a canonical Dataset backend."
            raise ValueError(msg)
        if case.precision != "float32":
            msg = "Transient timing currently supports only float32 Evaluation."
            raise ValueError(msg)
        if isinstance(case.batch_size, bool) or not isinstance(case.batch_size, int) or case.batch_size < 1:
            msg = "batch_size must be a positive integer."
            raise ValueError(msg)
        if not case.cpu or (case.device.startswith("cuda") and not case.gpu):
            msg = "Timing cases must disclose CPU and any selected GPU."
            raise ValueError(msg)
        if isinstance(case.warmup_passes, bool) or not isinstance(case.warmup_passes, int) or case.warmup_passes < 0:
            msg = "warmup_passes must be a non-negative integer."
            raise ValueError(msg)
        if not isinstance(case.software_versions, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in case.software_versions.items()
        ):
            msg = "software_versions must map non-empty names to non-empty versions."
            raise ValueError(msg)
        repeated = set(case.repetitions)
        cold = set(case.cold_timings)
        unavailable = set(case.unavailable_reasons)
        if (
            not isinstance(case.repetitions, Mapping)
            or not isinstance(case.cold_timings, Mapping)
            or not isinstance(case.unavailable_reasons, Mapping)
            or not repeated.issubset(TIMING_COMPONENTS)
            or not cold.issubset(TIMING_COMPONENTS)
            or not unavailable.issubset(TIMING_COMPONENTS)
            or repeated.intersection(unavailable)
        ):
            msg = "Timing evidence may contain only declared components and cannot be both available and unavailable."
            raise ValueError(msg)
        if case.dataset_backend == "pt_shards" and not case.pt_payload_identity:
            msg = "PT-shard timing requires its payload identity."
            raise ValueError(msg)
        if case.dataset_backend == "canonical_hdf5" and case.pt_payload_identity is not None:
            msg = "Canonical-HDF5 timing must not claim a PT payload identity."
            raise ValueError(msg)
        for name, value in case.cold_timings.items():
            _duration(value, label=f"{case.case_id}.cold.{name}")
        for name, reason in case.unavailable_reasons.items():
            if not isinstance(reason, str) or not reason:
                msg = f"Unavailable timing reason for {case.case_id!r}/{name!r} must be non-empty."
                raise ValueError(msg)
        for name, values in case.repetitions.items():
            if not isinstance(values, tuple) or not values or len(values) > _MAX_RAW_REPETITIONS:
                msg = f"Timing repetitions for {case.case_id!r}/{name!r} must be bounded and non-empty."
                raise ValueError(msg)
            for value in values:
                _duration(value, label=f"{case.case_id}.{name}")
    ordered = tuple(cases)
    return TransientTimingReport(ordered, {name: _summary(name, ordered) for name in _SPEEDUP_FORMULAS}, True)


def admit_transient_timing_report(value: Mapping[str, object]) -> TransientTimingReport:
    """Recompute and admit one serialized timing report without trusting aggregates."""
    if not isinstance(value, Mapping) or set(value) != {"cases", "speedups", "component_composed"}:
        msg = "Serialized transient timing report fields are invalid."
        raise ValueError(msg)
    raw_cases = value["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        msg = "Serialized transient timing report requires non-empty cases."
        raise ValueError(msg)
    cases: list[TransientTimingCase] = []
    expected_fields = set(TransientTimingCase.__dataclass_fields__)
    for position, item in enumerate(raw_cases):
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            msg = f"Serialized transient timing case {position} fields are invalid."
            raise ValueError(msg)
        repetitions = item["repetitions"]
        software_versions = item["software_versions"]
        cold_timings = item["cold_timings"]
        unavailable_reasons = item["unavailable_reasons"]
        text_fields = ("case_id", "device", "precision", "dataset_backend", "cpu")
        if any(not isinstance(item[name], str) or not item[name] for name in text_fields):
            msg = f"Serialized transient timing case {position} text fields are invalid."
            raise TypeError(msg)
        if (
            not isinstance(repetitions, Mapping)
            or any(not isinstance(name, str) or not name or not isinstance(values, list) for name, values in repetitions.items())
            or not isinstance(software_versions, Mapping)
            or not isinstance(cold_timings, Mapping)
            or not isinstance(unavailable_reasons, Mapping)
        ):
            msg = f"Serialized transient timing case {position} mappings are invalid."
            raise TypeError(msg)
        warmup_passes = item["warmup_passes"]
        batch_size = item["batch_size"]
        gpu = item["gpu"]
        pt_payload_identity = item["pt_payload_identity"]
        if (
            isinstance(warmup_passes, bool)
            or not isinstance(warmup_passes, int)
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or (gpu is not None and (not isinstance(gpu, str) or not gpu))
            or (pt_payload_identity is not None and (not isinstance(pt_payload_identity, str) or not pt_payload_identity))
        ):
            msg = f"Serialized transient timing case {position} scalar fields are invalid."
            raise TypeError(msg)
        cases.append(
            TransientTimingCase(
                case_id=cast("str", item["case_id"]),
                repetitions=cast("Mapping[str, tuple[float, ...]]", {name: tuple(values) for name, values in repetitions.items()}),
                device=cast("str", item["device"]),
                precision=cast("str", item["precision"]),
                dataset_backend=cast("str", item["dataset_backend"]),
                warmup_passes=warmup_passes,
                batch_size=batch_size,
                cpu=cast("str", item["cpu"]),
                gpu=cast("str | None", gpu),
                software_versions=cast("Mapping[str, str]", dict(software_versions)),
                cold_timings=cast("Mapping[str, float]", dict(cold_timings)),
                unavailable_reasons=cast("Mapping[str, str]", dict(unavailable_reasons)),
                pt_payload_identity=cast("str | None", pt_payload_identity),
            )
        )
    report = build_transient_timing_report(cases)
    normalized_expected = json.loads(json.dumps(asdict(report), allow_nan=False, sort_keys=True))
    normalized_observed = json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True))
    if normalized_observed != normalized_expected:
        msg = "Serialized transient timing aggregates do not match raw repetitions."
        raise ValueError(msg)
    return report
