# ruff: noqa: S101, D103, EM101, TRY003, TC003
"""Verify terminal timing evidence loading without runtime-text parsing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.generation.runtime import generation_runtime_timing as timing

if TYPE_CHECKING:
    from src.generation.runtime.generation_runtime_batch import (
        TerminalBatchEvidence,
        TerminalCaseEvidence,
    )


class _Case:
    def __init__(self, directory: Path) -> None:
        self.case_id = "case_0001"
        self.case_input_id = "b" * 64
        self.simulation_case_id = "c" * 64
        self.hdf5_identity = SimpleNamespace(simulation_profile="transient_drying", git_commit="a" * 40)
        self.directory = directory
        self.requests: list[tuple[str, str]] = []

    def artifact(self, stage: str, relative_path: str) -> Any:
        self.requests.append((stage, relative_path))
        path = self.directory / relative_path
        return SimpleNamespace(path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class _Batch:
    batch_id = "batch-1"
    simulation_profile = "transient_drying"
    git_commit = "a" * 40

    def __init__(self, case: _Case) -> None:
        self._case = case

    def case(self, case_id: str) -> _Case:
        if case_id != self._case.case_id:
            raise ValueError("unknown test case")
        return self._case

    @staticmethod
    def scientific_config_payload() -> dict[str, Any]:
        return {"scientific_fixed_values": {"f_wet_dm_max": 0.05}}


def _write_sidecars(directory: Path) -> None:
    timing_payload = {
        "schema_kind": "simulation_case_timing",
        "schema_version": 1,
        "batch_id": "batch-1",
        "case_id": "case_0001",
        "case_input_id": "b" * 64,
        "simulation_case_id": "c" * 64,
        "simulation_profile": "transient_drying",
        "git_commit": "a" * 40,
        "exit_code": 0,
        "timed_out": False,
        "runtime_s": 12.0,
        "comsol_process_seconds": 12.0,
        "export_conversion_seconds": 3.0,
        "complete_execution_s": 16.0,
        "license_wait_seconds": 2.0,
    }
    status_payload = {
        "schema_kind": "simulation_case_status",
        "schema_version": 1,
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
    (directory / "timing.json").write_text(json.dumps(timing_payload), encoding="utf-8")
    (directory / "status.json").write_text(json.dumps(status_payload), encoding="utf-8")


def test_load_case_timing_separates_physical_and_runtime_evidence(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    case = _Case(tmp_path)
    result = timing.load_case_timing(
        cast("TerminalCaseEvidence", case),
        batch=cast("TerminalBatchEvidence", _Batch(case)),
    )
    assert case.requests == [("processed", "timing.json"), ("processed", "status.json")]
    assert (result.physical_duration_hours, result.time_to_target_hours, result.target_reached, result.right_censored) == (96.0, None, False, True)
    assert (result.final_wet_fraction, result.target_wet_fraction_limit) == (0.07, 0.05)
    assert (result.comsol_process_seconds, result.export_conversion_seconds, result.complete_execution_seconds) == (12.0, 3.0, 16.0)
    payload = result.as_dict()
    assert payload["component_timing_availability"]["queue_wait_seconds"] == "unavailable_not_persisted"
    payload["component_timing_availability"]["queue_wait_seconds"] = "changed"
    assert result.as_dict()["component_timing_availability"]["queue_wait_seconds"] == "unavailable_not_persisted"


def test_load_case_timing_rejects_target_status_inconsistent_with_admitted_limit(tmp_path: Path) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / "status.json"
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
    source = tmp_path / "status.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["runtime_s"] = 13.0
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)
    with pytest.raises(ValueError, match="status and timing runtime"):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )


@pytest.mark.parametrize(
    ("path", "field", "value"),
    [("timing.json", "git_commit", "d" * 40), ("timing.json", "complete_execution_s", float("nan")), ("status.json", "schema_version", 2)],
)
def test_load_case_timing_rejects_identity_schema_and_nonfinite_evidence(tmp_path: Path, path: str, field: str, value: Any) -> None:
    _write_sidecars(tmp_path)
    source = tmp_path / path
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload[field] = value
    source.write_text(json.dumps(payload), encoding="utf-8")
    case = _Case(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        timing.load_case_timing(
            cast("TerminalCaseEvidence", case),
            batch=cast("TerminalBatchEvidence", _Batch(case)),
        )
