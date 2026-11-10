"""Verify terminal timing evidence loading without runtime-text parsing.

Responsibilities:
  - Check terminal timing admission, source ownership, component sums, and record counts
  - Check the validated stall-progress record and its explicit availability states

Design principles:
  - Physical time and runtime evidence remain distinct; legacy missing components stay unavailable

This module does NOT:
  - Parse runtime log text or invent missing timing measurements
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.generation.runtime import generation_runtime_comsol_timing as comsol_timing
from src.generation.runtime import generation_runtime_timing as timing

if TYPE_CHECKING:
    from pathlib import Path

    from src.generation.runtime.generation_runtime_batch import (
        TerminalBatchEvidence,
        TerminalCaseEvidence,
    )


class _Case:
    def __init__(self, directory: Path, *, git_commit: str = "a" * 40) -> None:
        self.case_id = "case_0001"
        self.case_input_id = "b" * 64
        self.simulation_case_id = "c" * 64
        self.hdf5_identity = SimpleNamespace(simulation_profile="transient_drying", git_commit=git_commit)
        self.directory = directory
        self.requests: list[tuple[str, str]] = []

    def artifact(self, stage: str, relative_path: str) -> Any:
        self.requests.append((stage, relative_path))
        path = self.directory / relative_path
        return SimpleNamespace(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class _Batch:
    simulation_profile = "transient_drying"

    def __init__(
        self,
        case: _Case,
        *,
        batch_id: str = "batch-1",
        git_commit: str = "a" * 40,
    ) -> None:
        self._case = case
        self.batch_id = batch_id
        self.git_commit = git_commit

    def case(self, case_id: str) -> _Case:
        if case_id != self._case.case_id:
            msg = "unknown test case"
            raise ValueError(msg)
        return self._case

    @staticmethod
    def scientific_config_payload() -> dict[str, Any]:
        return {"scientific_fixed_values": {"f_wet_dm_max": 0.05}}


_NOT_PERSISTED = object()


def _write_sidecars(
    directory: Path,
    *,
    include_solver_timing: bool = True,
    batch_id: str = "batch-1",
    git_commit: str = "a" * 40,
    stall_progress: Any = _NOT_PERSISTED,
) -> None:
    timing_payload = {
        "schema_kind": "simulation_case_timing",
        "batch_id": batch_id,
        "case_id": "case_0001",
        "case_input_id": "b" * 64,
        "simulation_case_id": "c" * 64,
        "simulation_profile": "transient_drying",
        "git_commit": git_commit,
        "exit_code": 0,
        "timed_out": False,
        "runtime_s": 12.0,
        "comsol_process_seconds": 12.0,
        "export_conversion_seconds": 3.0,
        "complete_execution_s": 16.0,
        "license_wait_seconds": 2.0,
    }
    if stall_progress is not _NOT_PERSISTED:
        timing_payload["stall_progress"] = stall_progress
    if include_solver_timing:
        solver_timing = comsol_timing.parse_comsol_batch_log_text(
            (
                "<---- Stationary Solver 1 in Stationary Airflow/Stationary Airflow Solution (sol1) -\n"
                "Solution time: 3 s.\n"
                "----- Stationary Solver 1 in Stationary Airflow/Stationary Airflow Solution (sol1) >\n"
                "<---- Time-Dependent Solver 1 in Transient Drying/Transient Drying Solution (sol2) -\n"
                "Solution time: 4 s.\n"
                "----- Time-Dependent Solver 1 in Transient Drying/Transient Drying Solution (sol2) >"
            ),
            simulation_profile="transient_drying",
        )
        timing_payload.update(solver_timing.timing_fields())
    status_payload = {
        "schema_kind": "simulation_case_status",
        "solver_success": True,
        "target_reached": False,
        "t_stop_exact": 96.0,
        "f_wet_dm_final": 0.07,
        "runtime_s": 12.0,
        "units": {"runtime_s": "s", "t_stop_exact": "h", "f_wet_dm_final": "1"},
        "contains_nan_or_inf": False,
        "field_shape_valid": True,
        "case_state": "successful",
        "stages": {"solver": "succeeded", "exports": "succeeded", "conversion": "succeeded", "publication": "succeeded"},
    }
    evidence = directory / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "timing.json").write_text(json.dumps(timing_payload), encoding="utf-8")
    (evidence / "status.json").write_text(json.dumps(status_payload), encoding="utf-8")


def test_load_case_timing_separates_physical_and_runtime_evidence(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    case = _Case(tmp_path)
    result = timing.load_case_timing(
        cast("TerminalCaseEvidence", case),
        batch=cast("TerminalBatchEvidence", _Batch(case)),
    )
    assert case.requests == [("processed", "evidence/timing.json"), ("processed", "evidence/status.json")]
    assert (result.physical_duration_hours, result.time_to_target_hours, result.target_reached, result.right_censored) == (96.0, None, False, True)
    assert (result.final_wet_fraction, result.target_wet_fraction_limit) == (0.07, 0.05)
    assert (result.comsol_process_seconds, result.export_conversion_seconds, result.complete_execution_seconds) == (12.0, 3.0, 16.0)
    assert (result.stationary_airflow_solver_seconds, result.transient_drying_solver_seconds) == (3.0, 4.0)
    assert result.scientific_solver_seconds == 7.0
    assert result.comsol_process_seconds != result.scientific_solver_seconds
    payload = result.as_dict()
    assert payload["comsol_solver_timing"]["status"] == "complete"
    assert payload["component_timing_availability"]["scientific_solver_seconds"] == "available"
    assert payload["component_timing_availability"]["queue_wait_seconds"] == "unavailable_not_persisted"
    payload["component_timing_availability"]["queue_wait_seconds"] = "changed"
    assert result.as_dict()["component_timing_availability"]["queue_wait_seconds"] == "unavailable_not_persisted"


def test_load_case_timing_admits_replacement_with_its_actual_source_batch(tmp_path: Path) -> None:
    batch_id = "replacement-batch"
    git_commit = "d" * 40
    _write_sidecars(tmp_path, batch_id=batch_id, git_commit=git_commit)
    case = _Case(tmp_path, git_commit=git_commit)

    result = timing.load_case_timing(
        cast("TerminalCaseEvidence", case),
        batch=cast(
            "TerminalBatchEvidence",
            _Batch(case, batch_id=batch_id, git_commit=git_commit),
        ),
    )

    assert (result.batch_id, result.git_commit) == (batch_id, git_commit)


def test_load_case_timing_rejects_target_status_inconsistent_with_admitted_limit(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / "evidence" / "status.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["target_reached"] = True
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)
    with pytest.raises(ValueError, match="target_reached disagrees"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


def test_load_case_timing_rejects_status_runtime_inconsistent_with_timing(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / "evidence" / "status.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["runtime_s"] = 13.0
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)
    with pytest.raises(ValueError, match="status and timing runtime"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


def test_load_case_timing_rejects_timing_owned_by_a_different_source_batch(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / "evidence" / "timing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["batch_id"] = "replacement-batch"
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)

    with pytest.raises(ValueError, match="timing identity"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [("timing.json", "git_commit", "d" * 40), ("timing.json", "complete_execution_s", float("nan")), ("status.json", "schema_kind", "other")],
)
def test_load_case_timing_rejects_identity_schema_and_nonfinite_evidence(tmp_path: Path, path: str, field: str, value: Any) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / "evidence" / path
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload[field] = value
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


def test_load_case_timing_rejects_a_sidecar_without_solver_timing(tmp_path: Path) -> None:
    _write_sidecars(tmp_path, include_solver_timing=False)
    case = _Case(tmp_path)

    with pytest.raises(ValueError, match="incomplete COMSOL solver-timing fields"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


@pytest.mark.parametrize(
    ("stall_progress", "expected", "availability"),
    [
        ({"minimum_simulated_seconds": 3600.0, "reached_after_seconds": 412.5}, (3600.0, 412.5), "available"),
        ({"minimum_simulated_seconds": 3600.0, "reached_after_seconds": None}, (3600.0, None), "unavailable_missing"),
        (None, (None, None), "unavailable_not_applicable"),
        (_NOT_PERSISTED, (None, None), "unavailable_not_persisted"),
    ],
)
def test_load_case_timing_exposes_the_stall_progress_with_explicit_availability(
    tmp_path: Path, stall_progress: Any, expected: tuple[float | None, float | None], availability: str
) -> None:
    _write_sidecars(tmp_path, stall_progress=stall_progress)
    case = _Case(tmp_path)

    result = timing.load_case_timing(
        cast("TerminalCaseEvidence", case),
        batch=cast("TerminalBatchEvidence", _Batch(case)),
    )

    assert (result.stall_minimum_simulated_seconds, result.stall_minimum_reached_after_seconds) == expected
    payload = result.as_dict()
    assert (payload["stall_minimum_simulated_seconds"], payload["stall_minimum_reached_after_seconds"]) == expected
    assert payload["component_timing_availability"]["stall_progress"] == availability


@pytest.mark.parametrize(
    "stall_progress",
    [
        "3600",
        {"minimum_simulated_seconds": 3600.0},
        {"minimum_simulated_seconds": 3600.0, "reached_after_seconds": 5.0, "extra": 1},
        {"minimum_simulated_seconds": 0.0, "reached_after_seconds": None},
        {"minimum_simulated_seconds": float("nan"), "reached_after_seconds": None},
        {"minimum_simulated_seconds": True, "reached_after_seconds": None},
        {"minimum_simulated_seconds": 3600.0, "reached_after_seconds": -1.0},
        {"minimum_simulated_seconds": 3600.0, "reached_after_seconds": float("inf")},
        {"minimum_simulated_seconds": 3600.0, "reached_after_seconds": "soon"},
    ],
)
def test_load_case_timing_rejects_malformed_stall_progress(tmp_path: Path, stall_progress: Any) -> None:
    _write_sidecars(tmp_path, stall_progress=stall_progress)
    case = _Case(tmp_path)

    with pytest.raises((TypeError, ValueError), match="stall_progress"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


def test_load_case_timing_rejects_solver_sum_inconsistent_with_structural_evidence(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / "evidence" / "timing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["comsol_scientific_solver_seconds"] = 8.0
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)

    with pytest.raises(ValueError, match="disagree with structural evidence"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


def test_load_case_timing_rejects_total_record_count_smaller_than_phase_evidence(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / "evidence" / "timing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["comsol_solver_timing"]["solution_time_record_count"] = 1
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)

    with pytest.raises(ValueError, match="counts exceed the total record count"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )
