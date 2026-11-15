"""Measure and validate COMSOL and neural artifact runtime comparisons.

Responsibilities:
  - Measure synchronized neural forward passes after explicit warmup
  - Pair each steady case's batch-amortized forward with its COMSOL solver and process times and its solver hardware
  - Derive and validate the steady speed-ups as ratios of sums over the paired cases
  - Own the required non-identity timing block of both tasks' completion markers and write it into unpublished stages

Design principles:
  - Timing evidence is bound to run and membership but never part of scientific artifact identity
  - Timing is written into the stage before publication, so a timing failure publishes nothing
  - Device synchronization, inference dtype, batch size and solver hardware are recorded explicitly
  - A missing COMSOL time leaves its case unpaired and counted, never zero; speed-ups are rebuilt on admission

This module does NOT:
  - Select runs, datasets, checkpoints, or artifact memberships
  - Generate Parquet or NPZ scientific payloads
  - Own the transient timing report (``evaluation_transient_timing``)
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from src import common

from . import analysis_artifact_contracts as contracts
from . import analysis_artifact_performance as artifact_performance

if TYPE_CHECKING:
    from src.datasets.preprocessing.dataset_preprocessing_normalization import TrainScaling
    from src.generation.runtime.generation_runtime_timing import CaseTimingEvidence

TIMING_KEY = "timing"
RUNTIME_COMPARISON_SCHEMA_KIND = "comsol_neural_operator_runtime_comparison"
WARMUP_PASSES = 1
MEASUREMENT_CLOCK = "time.perf_counter_ns"
CASE_DURATION_ATTRIBUTION = "equal_share_of_observed_batch_forward_duration"
SURROGATE_BOUNDARY = "Synchronized warmed loader-batch forward wall time divided equally among batch cases"
# Each steady speed-up divides one COMSOL clock, summed over the paired cases, by the same cases' summed forward time.
STEADY_SPEEDUPS: Final = {
    "solver_speedup": ("comsol_solver_s", "COMSOL stationary study top-level Solution time from its solver log"),
    "process_speedup": ("comsol_process_s", "Measured COMSOL process wall time; includes process overhead beyond the solver study"),
}
_NEURAL_CASE_FIELDS = frozenset({"case_id", "source_index", "batch_amortized_neural_operator_forward_s"})
_SOLVER_CASE_FIELDS = frozenset({"comsol_solver_s", "comsol_process_s", "solver_hostname", "solver_requested_cores"})
_COMPARISON_FIELDS = frozenset(
    {
        "schema_kind",
        "split_role",
        "dataset_identity",
        "model_identity",
        "neural_runtime",
        "measurement",
        "surrogate_boundary",
        "cases",
        "speedups",
        "solver_hardware",
    }
)
_MEASUREMENT_FIELDS = frozenset({"clock", "batch_size", "case_duration_attribution", "warmup_passes", "cuda_synchronized"})
_SPLIT_ROLES = ("eval", "material_ood", "parameter_ood")


def _duration(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real scalar."
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        msg = f"{label} must be finite and positive."
        raise ValueError(msg)
    return result


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{label} must be non-empty text."
        raise TypeError(msg)
    return value


def _count(value: Any, *, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer."
        raise TypeError(msg)
    if value < minimum:
        msg = f"{label} must be at least {minimum}."
        raise ValueError(msg)
    return value


def _available_seconds(value: float | None) -> float | None:
    """Return one persisted COMSOL clock, or None when Generation did not record a positive duration."""
    return float(value) if value is not None and math.isfinite(value) and value > 0.0 else None


def _validate_case(value: Any, *, position: int) -> dict[str, Any]:
    """Validate one paired case record; an absent COMSOL clock stays None."""
    if not isinstance(value, Mapping) or set(value) != _NEURAL_CASE_FIELDS | _SOLVER_CASE_FIELDS:
        msg = f"Runtime comparison case {position} has invalid fields."
        raise ValueError(msg)
    label = f"cases[{position}]"
    clocks = {
        component: None if value[component] is None else _duration(value[component], label=f"{label}.{component}")
        for component, _boundary in STEADY_SPEEDUPS.values()
    }
    return {
        "case_id": _require_text(value["case_id"], label=f"{label}.case_id"),
        "source_index": _count(value["source_index"], label=f"{label}.source_index", minimum=0),
        "batch_amortized_neural_operator_forward_s": _duration(
            value["batch_amortized_neural_operator_forward_s"], label=f"{label}.batch_amortized_neural_operator_forward_s"
        ),
        **clocks,
        "solver_hostname": _require_text(value["solver_hostname"], label=f"{label}.solver_hostname"),
        "solver_requested_cores": _count(value["solver_requested_cores"], label=f"{label}.solver_requested_cores"),
    }


def _paired_speedup(cases: Sequence[Mapping[str, Any]], component: str, boundary: str) -> dict[str, Any]:
    """Return Σ COMSOL time / Σ surrogate time over the cases that have this COMSOL clock, with the unpaired count."""
    paired = [case for case in cases if case[component] is not None]
    reference = math.fsum(float(case[component]) for case in paired)
    surrogate = math.fsum(float(case["batch_amortized_neural_operator_forward_s"]) for case in paired)
    return {
        "reference_component": component,
        "reference_boundary": boundary,
        "paired_case_count": len(paired),
        "unavailable_case_count": len(cases) - len(paired),
        "reference_seconds_sum": reference if paired else None,
        "surrogate_seconds_sum": surrogate if paired else None,
        "ratio_of_sums": reference / surrogate if paired else None,
    }


def _derived_runtime(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive the speed-ups and the solver hardware census from validated case records."""
    hardware = Counter((str(case["solver_hostname"]), int(case["solver_requested_cores"])) for case in cases)
    return {
        "speedups": {name: _paired_speedup(cases, component, boundary) for name, (component, boundary) in STEADY_SPEEDUPS.items()},
        "solver_hardware": [{"hostname": host, "requested_cores": cores, "case_count": count} for (host, cores), count in sorted(hardware.items())],
    }


def synchronize_device(device: torch.device) -> None:
    """Synchronize one resolved CUDA device. CPU never accesses CUDA APIs."""
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        msg = f"Timing requires one concrete CPU or CUDA torch.device, got {device!r}."
        raise TypeError(msg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_forward(
    *,
    model: Any,
    normalized_inputs: torch.Tensor,
    device: torch.device,
) -> tuple[Any, float]:
    """Measure one direct non-empty loader-batch model call on the device."""
    if not isinstance(normalized_inputs, torch.Tensor) or normalized_inputs.ndim == 0 or normalized_inputs.shape[0] <= 0:
        msg = "Authoritative neural-operator timing requires a non-empty input batch."
        raise ValueError(msg)
    model.eval()
    with torch.inference_mode():
        synchronize_device(device)
        started_ns = time.perf_counter_ns()
        prediction = model(normalized_inputs)
        synchronize_device(device)
        elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    return prediction, _duration(elapsed_s, label="neural-operator batch forward duration")


def warm_up_forward(
    *,
    representative_batch: Mapping[str, Any],
    model: Any,
    scaling: TrainScaling,
    device: torch.device,
    passes: int = WARMUP_PASSES,
) -> None:
    """Warm up the complete online-equivalent batch path without publishing it."""
    if isinstance(passes, bool) or not isinstance(passes, int) or passes <= 0:
        msg = "Warmup passes must be a positive integer."
        raise ValueError(msg)
    raw_inputs = representative_batch.get("x")
    if not isinstance(raw_inputs, torch.Tensor) or raw_inputs.ndim == 0 or raw_inputs.shape[0] <= 0:
        msg = "Forward warmup requires a non-empty tensor batch under key 'x'."
        raise TypeError(msg)
    model.eval()
    with torch.inference_mode():
        normalized_inputs = scaling.encode_inputs(raw_inputs.to(device))
        for _ in range(passes):
            synchronize_device(device)
            model(normalized_inputs)
            synchronize_device(device)


def model_inference_dtype(model: Any) -> str:
    """Return the first model-parameter dtype without allocating device state."""
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "unknown"
    raw_parameters = parameters()
    if not isinstance(raw_parameters, Iterable):
        return "unknown"
    try:
        parameter = next(iter(raw_parameters))
    except StopIteration:
        return "unknown"
    return str(parameter.dtype).removeprefix("torch.")


def neural_runtime_metadata(
    *,
    device_metadata: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    """Extend the shared resolved-device record with compact benchmark context."""
    return {
        **dict(device_metadata),
        "hostname": platform.node() or "unknown",
        "platform": platform.platform() or "unknown",
        "processor": platform.processor() or platform.machine() or "unknown",
        "python_version": platform.python_version() or sys.version.split()[0],
        "inference_dtype": model_inference_dtype(model),
        "torch_num_threads": torch.get_num_threads(),
    }


def _validate_neural_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        msg = "neural_runtime must be a mapping."
        raise TypeError(msg)
    runtime = dict(value)
    base = {
        "requested_policy",
        "resolved_device",
        "device_type",
        "pytorch_version",
        "hostname",
        "platform",
        "processor",
        "python_version",
        "inference_dtype",
        "torch_num_threads",
    }
    cuda = {"cuda_index", "cuda_device_name", "cuda_build_version"}
    device_type = runtime.get("device_type")
    expected = base | (cuda if device_type == "cuda" else set())
    if device_type not in {"cpu", "cuda"} or set(runtime) != expected:
        msg = "neural_runtime has invalid fields for its resolved device type."
        raise ValueError(msg)
    if runtime.get("requested_policy") not in {"auto", "cpu", "cuda"}:
        msg = "neural_runtime.requested_policy is invalid."
        raise ValueError(msg)
    for field in (
        "resolved_device",
        "pytorch_version",
        "hostname",
        "platform",
        "processor",
        "python_version",
        "inference_dtype",
    ):
        _require_text(runtime.get(field), label=f"neural_runtime.{field}")
    threads = runtime.get("torch_num_threads")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        msg = "neural_runtime.torch_num_threads must be a positive integer."
        raise TypeError(msg)
    if device_type == "cuda":
        index = runtime.get("cuda_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            msg = "neural_runtime.cuda_index must be a non-negative integer."
            raise TypeError(msg)
        _require_text(runtime.get("cuda_device_name"), label="neural_runtime.cuda_device_name")
        _require_text(runtime.get("cuda_build_version"), label="neural_runtime.cuda_build_version")
    return runtime


def solver_case_timing(evidence: CaseTimingEvidence) -> dict[str, Any]:
    """Project one admitted Generation case timing onto the steady comparison's COMSOL clocks and solver hardware."""
    return {
        "comsol_solver_s": _available_seconds(evidence.stationary_airflow_solver_seconds),
        "comsol_process_s": _available_seconds(evidence.comsol_process_seconds),
        "solver_hostname": evidence.solver_hostname,
        "solver_requested_cores": evidence.solver_requested_cores,
    }


def runtime_dataset_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Dataset identity a runtime comparison binds: role name, every source part and the ordered membership."""
    dataset = provenance["dataset"]
    return {
        "name": dataset["name"],
        "dataset_provenance_sha256": common.serialization.canonical_json_sha256(dataset),
        "effective_ordered_source_indices_sha256": provenance["selection"]["effective_ordered_source_indices_sha256"],
    }


def _validate_identity(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        msg = f"{label} has invalid fields."
        raise ValueError(msg)
    identity = dict(value)
    for field in fields:
        _require_text(identity.get(field), label=f"{label}.{field}")
    return identity


def build_runtime_comparison(
    *,
    split_role: str,
    dataset_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    neural_runtime: Mapping[str, Any],
    batch_size: int,
    cases: Sequence[Mapping[str, Any]],
    solver_cases: Mapping[str, Mapping[str, Any]],
    warmup_passes: int = WARMUP_PASSES,
) -> dict[str, Any]:
    """
    Pair every measured steady case with its COMSOL clocks and solver hardware and derive the speed-ups.

    Parameters
    ----------
    split_role : str
        Artifact split the cases belong to.
    dataset_identity, model_identity : Mapping[str, Any]
        Exact Dataset and checkpoint identities the comparison binds.
    neural_runtime : Mapping[str, Any]
        Surrogate hardware and software record from :func:`neural_runtime_metadata`.
    batch_size : int
        Loader batch size whose synchronized forward is shared equally among its cases.
    cases : Sequence[Mapping[str, Any]]
        Measured neural cases (``case_id``, ``source_index``, ``batch_amortized_neural_operator_forward_s``) in split order.
    solver_cases : Mapping[str, Mapping[str, Any]]
        :func:`solver_case_timing` record of every measured case, keyed by package case ID.
    warmup_passes : int, optional
        Untimed forward passes before measurement.

    Returns
    -------
    dict[str, Any]
        Validated comparison: per-case records, each speed-up as Σ COMSOL time / Σ surrogate time over the cases
        that have that COMSOL clock (with the unpaired count), and the solver hardware census.

    Raises
    ------
    ValueError
        If a measured case lacks its solver record or any record is invalid.

    """
    records: list[dict[str, Any]] = []
    for position, case in enumerate(cases):
        if not isinstance(case, Mapping) or set(case) != _NEURAL_CASE_FIELDS:
            msg = f"Neural timing case {position} has invalid fields."
            raise ValueError(msg)
        solver = solver_cases.get(str(case["case_id"]))
        if solver is None:
            msg = f"Neural timing case {case['case_id']!r} has no Generation solver timing."
            raise ValueError(msg)
        records.append(_validate_case({**case, **solver}, position=position))
    payload = {
        "schema_kind": RUNTIME_COMPARISON_SCHEMA_KIND,
        "split_role": split_role,
        "dataset_identity": dict(dataset_identity),
        "model_identity": dict(model_identity),
        "neural_runtime": dict(neural_runtime),
        "measurement": {
            "clock": MEASUREMENT_CLOCK,
            "batch_size": batch_size,
            "case_duration_attribution": CASE_DURATION_ATTRIBUTION,
            "warmup_passes": warmup_passes,
            "cuda_synchronized": neural_runtime.get("device_type") == "cuda",
        },
        "surrogate_boundary": SURROGATE_BOUNDARY,
        "cases": records,
        **_derived_runtime(records),
    }
    return validate_runtime_comparison(payload)


def validate_runtime_comparison(value: Any) -> dict[str, Any]:
    """Validate identities, measurement, every paired case and the rebuilt speed-ups of one steady runtime comparison."""
    if not isinstance(value, Mapping):
        msg = "Runtime comparison must be a mapping."
        raise TypeError(msg)
    payload = dict(value)
    schema = (payload.get("schema_kind"), payload.get("surrogate_boundary"))
    if set(payload) != _COMPARISON_FIELDS or schema != (RUNTIME_COMPARISON_SCHEMA_KIND, SURROGATE_BOUNDARY):
        msg = "Runtime comparison has an invalid schema; rerun the exact target with --rebuild."
        raise ValueError(msg)
    if payload.get("split_role") not in _SPLIT_ROLES:
        msg = f"Runtime comparison split_role must be one of {list(_SPLIT_ROLES)}."
        raise ValueError(msg)
    _validate_identity(
        payload.get("dataset_identity"),
        fields={"name", "dataset_provenance_sha256", "effective_ordered_source_indices_sha256"},
        label="dataset_identity",
    )
    _validate_identity(
        payload.get("model_identity"),
        fields={"run_name", "effective_config_digest", "best_checkpoint_sha256"},
        label="model_identity",
    )
    runtime = _validate_neural_runtime(payload.get("neural_runtime"))
    measurement = payload.get("measurement")
    if (
        not isinstance(measurement, Mapping)
        or set(measurement) != _MEASUREMENT_FIELDS
        or measurement.get("clock") != MEASUREMENT_CLOCK
        or measurement.get("case_duration_attribution") != CASE_DURATION_ATTRIBUTION
        or measurement.get("cuda_synchronized") is not (runtime["device_type"] == "cuda")
    ):
        msg = "Runtime comparison must describe synchronized, amortized perf_counter_ns loader-batch timing."
        raise ValueError(msg)
    _count(measurement["batch_size"], label="measurement.batch_size")
    _count(measurement["warmup_passes"], label="measurement.warmup_passes")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        msg = "Runtime comparison cases must be a non-empty JSON array."
        raise ValueError(msg)
    cases = [_validate_case(item, position=position) for position, item in enumerate(raw_cases)]
    if len({case["case_id"] for case in cases}) != len(cases) or len({case["source_index"] for case in cases}) != len(cases):
        msg = "Runtime comparison case IDs and source indices must be unique."
        raise ValueError(msg)
    derived = _derived_runtime(cases)
    if payload["speedups"] != derived["speedups"] or payload["solver_hardware"] != derived["solver_hardware"]:
        msg = "Runtime comparison speed-ups and solver hardware must be the ratio of sums and census of its paired cases."
        raise ValueError(msg)
    return payload


def timing_block(provenance: Mapping[str, Any], *, records: frozenset[str]) -> dict[str, Any]:
    """
    Return the required non-identity timing block of one canonical artifact of either task.

    The block holds the task's timing records (``records``) and the operational
    performance snapshot; both stay outside the scientific artifact identity.
    """
    block = provenance.get(TIMING_KEY)
    expected = records | {"operational_performance"}
    if not isinstance(block, Mapping) or set(block) != expected:
        found = sorted(block) if isinstance(block, Mapping) else type(block).__name__
        msg = f"Artifact timing block must hold exactly {sorted(expected)}, found {found}; rerun the exact target with --rebuild."
        raise ValueError(msg)
    artifact_performance.validate_operational_performance(block["operational_performance"])
    return dict(block)


def write_stage_timing(stage_root: str | Path, entries: Mapping[str, Any]) -> None:
    """Merge operational timing into one unpublished stage marker; scientific provenance and payloads stay untouched."""
    marker = contracts.artifact_provenance_path(Path(stage_root))
    document = json.loads(marker.read_text(encoding="utf-8"))
    document[TIMING_KEY] = {**document.get(TIMING_KEY, {}), **entries}
    common.serialization.atomic_write_json(marker, document)
