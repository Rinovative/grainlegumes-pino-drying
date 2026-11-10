"""Load validated runtime and physical-duration evidence from completed cases.

Responsibilities:
  - Read hash-admitted timing and status sidecars from completed-case evidence
  - Bind runtime values to the admitted case and batch identities
  - Preserve physical duration, target attainment, and censoring separately from runtime
  - Expose persisted solver timing and explicit component availability without parsing logs
  - Expose when a case first reached the stall rule's minimum simulated time, with absence kept explicit

Design principles:
  - Completed-case evidence remains the sole authority for artifact admission
  - Current persisted schemas are validated fail-closed before values are exposed
  - Runtime evidence never becomes part of scientific case identity

This module does NOT:
  - Parse COMSOL logs, scheduler output, or sacct text
  - Derive component solver or queue timing that Generation did not persist
  - Calculate target-criterion gaps or downstream speedup metrics
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Protocol

from src import common

from . import generation_runtime_comsol_timing as comsol_timing

if TYPE_CHECKING:
    from .generation_runtime_batch import TerminalCaseEvidence

_TIMING_SCHEMA_KIND = "simulation_case_timing"
_STATUS_SCHEMA_KIND = "simulation_case_status"
_UNAVAILABLE_NOT_PERSISTED = "unavailable_not_persisted"
_UNAVAILABLE_NOT_APPLICABLE = "unavailable_not_applicable"
_UNAVAILABLE_MISSING = "unavailable_missing"
_UNAVAILABLE_AMBIGUOUS = "unavailable_ambiguous"
_AVAILABLE = "available"


class _CompletedTimingBatch(Protocol):
    """Describe the batch identity required to interpret completed-case timing."""

    @property
    def batch_id(self) -> str:
        """Return immutable batch identity."""
        ...

    @property
    def simulation_profile(self) -> str:
        """Return canonical simulation profile."""
        ...

    @property
    def git_commit(self) -> str:
        """Return persisted source commit."""
        ...

    def case(self, case_id: str) -> TerminalCaseEvidence:
        """Return one exact admitted completed-case member."""
        ...

    def scientific_config_payload(self) -> dict[str, Any]:
        """Return independent resolved scientific configuration."""
        ...


@dataclass(frozen=True, slots=True)
class CaseTimingEvidence:
    """Validated timing and physical-duration evidence for one completed case."""

    case_id: str
    batch_id: str
    simulation_profile: str
    git_commit: str
    physical_duration_hours: float | None
    time_to_target_hours: float | None
    target_reached: bool | None
    right_censored: bool | None
    final_wet_fraction: float | None
    target_wet_fraction_limit: float | None
    physical_duration_availability: str
    target_wet_fraction_limit_availability: str
    comsol_process_seconds: float
    runtime_seconds: float
    export_conversion_seconds: float
    complete_execution_seconds: float
    licence_wait_seconds: float
    stationary_airflow_solver_seconds: float | None
    transient_drying_solver_seconds: float | None
    scientific_solver_seconds: float | None
    stall_minimum_simulated_seconds: float | None
    stall_minimum_reached_after_seconds: float | None
    queue_wait_seconds: None
    generation_compute_end_to_end_seconds: None
    comsol_solver_timing: comsol_timing.ComsolSolverTiming
    component_timing_availability: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible representation."""
        return {
            "case_id": self.case_id,
            "batch_id": self.batch_id,
            "simulation_profile": self.simulation_profile,
            "git_commit": self.git_commit,
            "physical_duration_hours": self.physical_duration_hours,
            "time_to_target_hours": self.time_to_target_hours,
            "target_reached": self.target_reached,
            "right_censored": self.right_censored,
            "final_wet_fraction": self.final_wet_fraction,
            "target_wet_fraction_limit": self.target_wet_fraction_limit,
            "physical_duration_availability": self.physical_duration_availability,
            "target_wet_fraction_limit_availability": self.target_wet_fraction_limit_availability,
            "comsol_process_seconds": self.comsol_process_seconds,
            "runtime_seconds": self.runtime_seconds,
            "export_conversion_seconds": self.export_conversion_seconds,
            "complete_execution_seconds": self.complete_execution_seconds,
            "licence_wait_seconds": self.licence_wait_seconds,
            "stationary_airflow_solver_seconds": self.stationary_airflow_solver_seconds,
            "transient_drying_solver_seconds": self.transient_drying_solver_seconds,
            "scientific_solver_seconds": self.scientific_solver_seconds,
            "stall_minimum_simulated_seconds": self.stall_minimum_simulated_seconds,
            "stall_minimum_reached_after_seconds": self.stall_minimum_reached_after_seconds,
            "queue_wait_seconds": None,
            "generation_compute_end_to_end_seconds": None,
            "comsol_solver_timing": self.comsol_solver_timing.as_dict(),
            "component_timing_availability": dict(self.component_timing_availability),
        }


def _load_artifact_payload(case: TerminalCaseEvidence, relative_path: str) -> dict[str, Any]:
    """Read one fully hash-validated processed JSON artifact."""
    artifact = case.artifact("processed", relative_path)
    try:
        payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Admitted {relative_path} for case {case.case_id!r} is not a readable JSON object."
        raise RuntimeError(msg) from error
    if not isinstance(payload, dict):
        msg = f"Admitted {relative_path} for case {case.case_id!r} must contain one JSON object."
        raise TypeError(msg)
    return payload


def _finite_non_negative(value: Any, *, label: str) -> float:
    """Require one finite non-negative real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real scalar."
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        msg = f"{label} must be finite and non-negative."
        raise ValueError(msg)
    return result


def _require_unit(units: Any, *, field: str, expected: str) -> None:
    """Require one exact persisted unit."""
    if not isinstance(units, dict) or units.get(field) != expected:
        msg = f"Case status must declare {field!r} in {expected!r}."
        raise ValueError(msg)


def _target_limit(batch: _CompletedTimingBatch) -> tuple[float | None, str]:
    """Read the admitted target limit without reopening configuration sources."""
    fixed = batch.scientific_config_payload().get("scientific_fixed_values")
    value = fixed.get("f_wet_dm_max") if isinstance(fixed, dict) else None
    if value is None:
        return None, _UNAVAILABLE_NOT_APPLICABLE
    return _finite_non_negative(value, label="admitted f_wet_dm_max"), "available"


def _validate_batch_binding(case: TerminalCaseEvidence, batch: _CompletedTimingBatch) -> None:
    """Require the case to be one of the exact admitted batch members."""
    if batch.case(case.case_id) != case:
        msg = f"Case {case.case_id!r} is not the supplied batch's admitted case object."
        raise ValueError(msg)
    if case.hdf5_identity.simulation_profile != batch.simulation_profile:
        msg = f"Case {case.case_id!r} profile disagrees with admitted batch evidence."
        raise ValueError(msg)
    if case.hdf5_identity.git_commit not in {None, batch.git_commit}:
        msg = f"Case {case.case_id!r} HDF5 commit disagrees with admitted batch evidence."
        raise ValueError(msg)


def _physical_status(
    status: dict[str, Any], *, simulation_profile: str
) -> tuple[float | None, float | None, bool | None, bool | None, float | None, str, float]:
    """Validate physical-duration values while keeping non-transient evidence explicit."""
    if status.get("solver_success") is not True or status.get("case_state") != "successful":
        msg = "Completed case status does not represent a successful solver outcome."
        raise ValueError(msg)
    stages = status.get("stages")
    if not isinstance(stages, dict) or any(stages.get(stage) != "succeeded" for stage in ("solver", "exports", "conversion", "publication")):
        msg = "Completed case status lacks successful terminal stages."
        raise ValueError(msg)
    if status.get("contains_nan_or_inf") is not False or status.get("field_shape_valid") is not True:
        msg = "Completed case status does not admit finite validated output fields."
        raise ValueError(msg)
    units = status.get("units")
    _require_unit(units, field="runtime_s", expected="s")
    status_runtime = _finite_non_negative(status.get("runtime_s"), label="case status runtime_s")
    if simulation_profile != "transient_drying":
        if any(status.get(field) is not None for field in ("t_stop_exact", "f_wet_dm_final", "target_reached")):
            msg = "Non-transient case status unexpectedly persists transient physical-duration values."
            raise ValueError(msg)
        return None, None, None, None, None, _UNAVAILABLE_NOT_APPLICABLE, status_runtime
    _require_unit(units, field="t_stop_exact", expected="h")
    _require_unit(units, field="f_wet_dm_final", expected="1")
    duration = _finite_non_negative(status.get("t_stop_exact"), label="case status t_stop_exact")
    final_wet = _finite_non_negative(status.get("f_wet_dm_final"), label="case status f_wet_dm_final")
    target_reached = status.get("target_reached")
    if not isinstance(target_reached, bool):
        msg = "Transient case status target_reached must be boolean."
        raise TypeError(msg)
    return duration, duration if target_reached else None, target_reached, not target_reached, final_wet, "available", status_runtime


def _stall_progress(timing: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """Validate the persisted stall-rule progress record and keep its absence explicit."""
    if "stall_progress" not in timing:
        return None, None, _UNAVAILABLE_NOT_PERSISTED
    record = timing["stall_progress"]
    if record is None:
        return None, None, _UNAVAILABLE_NOT_APPLICABLE
    if not isinstance(record, dict):
        msg = "Case timing stall_progress must be null or one JSON object."
        raise TypeError(msg)
    if set(record) != {"minimum_simulated_seconds", "reached_after_seconds"}:
        msg = "Case timing stall_progress must contain exactly minimum_simulated_seconds and reached_after_seconds."
        raise ValueError(msg)
    minimum = _finite_non_negative(record["minimum_simulated_seconds"], label="case timing stall_progress minimum_simulated_seconds")
    if minimum <= 0.0:
        msg = "Case timing stall_progress minimum_simulated_seconds must be positive."
        raise ValueError(msg)
    reached_value = record["reached_after_seconds"]
    if reached_value is None:
        return minimum, None, _UNAVAILABLE_MISSING
    reached = _finite_non_negative(reached_value, label="case timing stall_progress reached_after_seconds")
    return minimum, reached, _AVAILABLE


def _phase_availability(status: comsol_timing.PhaseStatus) -> str:
    """Translate structural phase status to the downstream availability vocabulary."""
    if status == "complete":
        return _AVAILABLE
    if status == "not_applicable":
        return _UNAVAILABLE_NOT_APPLICABLE
    if status == "ambiguous":
        return _UNAVAILABLE_AMBIGUOUS
    return _UNAVAILABLE_MISSING


def load_case_timing(case: TerminalCaseEvidence, *, batch: _CompletedTimingBatch) -> CaseTimingEvidence:
    """
    Load current validated timing and status evidence for one completed case.

    Parameters
    ----------
    case : TerminalCaseEvidence
        Case already admitted by Generation's completed-case validator.
    batch : completed timing batch evidence
        Matching admitted batch view that provides profile, commit, and science identity.

    Returns
    -------
    CaseTimingEvidence
        Immutable evidence with explicit unavailable component timing.

    """
    _validate_batch_binding(case, batch)
    timing = _load_artifact_payload(case, f"{common.paths.GENERATION_EVIDENCE_DIRNAME}/timing.json")
    status = _load_artifact_payload(case, f"{common.paths.GENERATION_EVIDENCE_DIRNAME}/status.json")
    if timing.get("schema_kind") != _TIMING_SCHEMA_KIND:
        msg = "Case timing does not use the current simulation_case_timing schema."
        raise ValueError(msg)
    expected = {
        "batch_id": batch.batch_id,
        "case_id": case.case_id,
        "case_input_id": case.case_input_id,
        "simulation_case_id": case.simulation_case_id,
        "simulation_profile": batch.simulation_profile,
        "git_commit": batch.git_commit,
    }
    if any(timing.get(field) != value for field, value in expected.items()):
        msg = "Case timing identity disagrees with admitted completed-case or batch evidence."
        raise ValueError(msg)
    if timing.get("exit_code") != 0 or timing.get("timed_out") is not False:
        msg = "Case timing does not represent a successful completed COMSOL process."
        raise ValueError(msg)
    if status.get("schema_kind") != _STATUS_SCHEMA_KIND:
        msg = "Case status does not use the current simulation_case_status schema."
        raise ValueError(msg)
    physical_duration, time_to_target, target_reached, right_censored, final_wet, physical_availability, status_runtime = _physical_status(
        status, simulation_profile=batch.simulation_profile
    )
    target_limit, target_limit_availability = _target_limit(batch)
    if target_limit is not None and final_wet is not None and target_reached is not None and target_reached != (final_wet <= target_limit):
        msg = "Case status target_reached disagrees with the admitted wet-fraction limit."
        raise ValueError(msg)
    values = {
        "comsol_process_seconds": _finite_non_negative(timing.get("comsol_process_seconds"), label="case timing comsol_process_seconds"),
        "runtime_seconds": _finite_non_negative(timing.get("runtime_s"), label="case timing runtime_s"),
        "export_conversion_seconds": _finite_non_negative(timing.get("export_conversion_seconds"), label="case timing export_conversion_seconds"),
        "complete_execution_seconds": _finite_non_negative(timing.get("complete_execution_s"), label="case timing complete_execution_s"),
        "licence_wait_seconds": _finite_non_negative(timing.get("license_wait_seconds"), label="case timing license_wait_seconds"),
    }
    if not math.isclose(values["comsol_process_seconds"], values["runtime_seconds"], rel_tol=0.0, abs_tol=1e-12):
        msg = "Case timing COMSOL process and runtime values disagree."
        raise ValueError(msg)
    if not math.isclose(status_runtime, values["runtime_seconds"], rel_tol=0.0, abs_tol=1e-12):
        msg = "Case status and timing runtime values disagree."
        raise ValueError(msg)
    if values["complete_execution_seconds"] < values["comsol_process_seconds"]:
        msg = "Case complete execution time cannot be shorter than COMSOL process time."
        raise ValueError(msg)
    solver_timing = comsol_timing.admit_persisted_solver_timing(
        timing,
        simulation_profile=batch.simulation_profile,
    )
    stationary_seconds = solver_timing.stationary_airflow.seconds
    transient_seconds = solver_timing.transient_drying.seconds
    scientific_seconds = solver_timing.scientific_solver_seconds
    scientific_availability = (
        _AVAILABLE if solver_timing.status == "complete" else _UNAVAILABLE_AMBIGUOUS if solver_timing.status == "ambiguous" else _UNAVAILABLE_MISSING
    )
    solver_components = (
        ("stationary_airflow_solver_seconds", _phase_availability(solver_timing.stationary_airflow.status)),
        ("transient_drying_solver_seconds", _phase_availability(solver_timing.transient_drying.status)),
        ("scientific_solver_seconds", scientific_availability),
    )
    stall_minimum, stall_reached_after, stall_availability = _stall_progress(timing)
    components = (
        *solver_components,
        ("stall_progress", stall_availability),
        ("queue_wait_seconds", _UNAVAILABLE_NOT_PERSISTED),
        ("generation_compute_end_to_end_seconds", _UNAVAILABLE_NOT_PERSISTED),
    )
    return CaseTimingEvidence(
        case_id=case.case_id,
        batch_id=batch.batch_id,
        simulation_profile=batch.simulation_profile,
        git_commit=batch.git_commit,
        physical_duration_hours=physical_duration,
        time_to_target_hours=time_to_target,
        target_reached=target_reached,
        right_censored=right_censored,
        final_wet_fraction=final_wet,
        target_wet_fraction_limit=target_limit,
        physical_duration_availability=physical_availability,
        target_wet_fraction_limit_availability=target_limit_availability,
        stationary_airflow_solver_seconds=stationary_seconds,
        transient_drying_solver_seconds=transient_seconds,
        scientific_solver_seconds=scientific_seconds,
        stall_minimum_simulated_seconds=stall_minimum,
        stall_minimum_reached_after_seconds=stall_reached_after,
        queue_wait_seconds=None,
        generation_compute_end_to_end_seconds=None,
        comsol_solver_timing=solver_timing,
        component_timing_availability=components,
        **values,
    )
