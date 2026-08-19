"""Focused controlled-stop lifecycle and workspace-containment contracts."""

from __future__ import annotations

import json
import signal
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from src.generation.runtime import generation_runtime_stop as stop_service

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Process:
    def __init__(self, clock: _Clock, *, exit_at: float | None = None) -> None:
        self.pid = 4172
        self._clock = clock
        self._exit_at = exit_at
        self._returncode: int | None = None
        self.signals: list[signal.Signals] = []

    def poll(self) -> int | None:
        if self._returncode is None and self._exit_at is not None and self._clock() >= self._exit_at:
            self._returncode = 0
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        completed = self.poll()
        if completed is not None:
            return completed
        if timeout is None:
            pytest.fail("Fake process must only receive bounded waits.")
        self._clock.advance(timeout)
        completed = self.poll()
        if completed is not None:
            return completed
        command = "comsol"
        raise subprocess.TimeoutExpired(command, timeout)

    def finish(self, exit_code: int) -> None:
        self._returncode = exit_code


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _workspace(tmp_path: Path) -> Path:
    directory = tmp_path / "active-case"
    directory.mkdir()
    return directory


def test_timeout_requests_stop_at_reserve_then_classifies_clean_controlled_exit(tmp_path: Path) -> None:
    """A clean controlled exit remains classified as a timeout."""
    clean_exit_at = 3310.0
    clock = _Clock()
    process = _Process(clock, exit_at=clean_exit_at)
    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=3600.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        timestamp_clock=lambda: "2026-08-18T00:00:00+00:00",
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    _require(result.exit_code == 0, "Controlled fake process did not exit cleanly.")
    _require(result.timed_out, "Controlled timeout exit lost its timeout classification.")
    _require(result.graceful_stop_requested, "Reserve boundary did not request controlled stop.")
    _require(not result.force_escalated, "Clean controlled exit unexpectedly escalated.")
    _require(clock() == clean_exit_at, "Stop was not requested at the 3300-second reserve boundary.")
    work_directory = tmp_path / "active-case"
    _require(
        (work_directory / "solved.mph.status").read_text(encoding="utf-8") == "Stop 2\n",
        "COMSOL controlled-stop status content differs from the documented command.",
    )
    evidence = json.loads((work_directory / "runtime" / "stop.json").read_text(encoding="utf-8"))
    _require(
        evidence
        == {
            "command": "Stop 2",
            "reason": "timeout",
            "requested_at": "2026-08-18T00:00:00+00:00",
            "schema_kind": "generation_runtime_stop",
            "schema_version": 1,
        },
        "Controlled-stop evidence does not preserve the timeout request.",
    )


def test_hard_deadline_escalates_only_owned_group_until_no_process_remains(tmp_path: Path) -> None:
    """The hard deadline sends signals only to the owned process group."""
    expected_deadline_completion = 3610.0
    clock = _Clock()
    process = _Process(clock)
    signalled: list[tuple[int, signal.Signals]] = []

    def signal_group(process_group_id: int, value: signal.Signals) -> None:
        _require(process_group_id == process.pid, "Controller attempted to signal another process group.")
        signalled.append((process_group_id, value))
        if value is signal.SIGKILL:
            process.finish(-signal.SIGKILL)

    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=3600.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        signal_process_group=signal_group,
        escalation_wait_seconds=5.0,
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    _require(clock() == expected_deadline_completion, "Escalation did not begin at the 3600-second hard deadline.")
    _require(result.timed_out, "Hard deadline lost its timeout classification.")
    _require(result.force_escalated, "Unresponsive process was not force escalated.")
    _require(process.poll() == -signal.SIGKILL, "Owned process remained alive after hard-deadline escalation.")
    _require(
        signalled == [(4172, signal.SIGTERM), (4172, signal.SIGKILL)],
        "Escalation did not limit signals to TERM then KILL for the owned group.",
    )
    _require(
        (tmp_path / "active-case" / "solved.mph.status").read_text(encoding="utf-8") == "Cancel\n",
        "Hard-deadline escalation did not publish the COMSOL Cancel command.",
    )


def test_stop_paths_reject_symbolic_links_and_invalid_timing(tmp_path: Path) -> None:
    """Status derivation rejects both symbolic-link and timing-contract violations."""
    work_directory = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (work_directory / "solved.mph.status").symlink_to(outside / "status")

    with pytest.raises(RuntimeError, match="symbolic link"):
        stop_service.derive_stop_status_path(work_directory)
    with pytest.raises(ValueError, match="0 < graceful_stop_reserve_seconds < timeout_seconds"):
        stop_service.validate_stop_timing(timeout_seconds=3600.0, graceful_stop_reserve_seconds=3600.0)


def test_unexpected_status_content_is_preserved_with_bounded_diagnostics(tmp_path: Path) -> None:
    """Never replace unknown owned content and report only one bounded excerpt."""
    work_directory = _workspace(tmp_path)
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    original = "unexpected solver-owned marker " + "x" * 500
    status_path.write_text(original, encoding="utf-8")
    clock = _Clock()
    controller = stop_service.SolverStopController(
        _Process(clock),
        work_directory,
        timeout_seconds=10.0,
        graceful_stop_reserve_seconds=2.0,
        monotonic_clock=clock,
    )

    with pytest.raises(stop_service.UnexpectedStopStatusContentError) as caught:
        controller.request_graceful_stop("timeout")

    _require(status_path.read_text(encoding="utf-8") == original, "Unknown text status content was overwritten.")
    diagnostics = json.loads(str(caught.value).split(": ", maxsplit=1)[1])
    _require(diagnostics["status_path"] == str(status_path), "Status diagnostics lost the exact path.")
    _require(
        diagnostics["expected_content_class"] == "exact_command_or_admitted_predecessor",
        "Status diagnostics lost the accepted-content contract.",
    )
    _require(diagnostics["actual_content_class"] == "unexpected_utf8_text", "Unexpected text was misclassified.")
    _require(diagnostics["actual_content_excerpt"] == original[:160], "Status excerpt was not bounded exactly.")
    _require(diagnostics["actual_content_excerpt_truncated"] is True, "Truncated status content was not identified.")
    _require(diagnostics["file_size"] == len(original.encode("utf-8")), "Status byte size was not preserved.")
    _require(
        diagnostics["ownership_evidence"] == "derived_regular_file_below_owned_active_workspace",
        "Owned workspace evidence was not explicit.",
    )
    _require(diagnostics["solver_exit_code"] is None, "Pre-termination diagnostics invented a solver exit code.")
    _require(diagnostics["required_exports_present"] is None, "Pre-termination diagnostics invented export evidence.")
    _require(diagnostics["replay_available"] is None, "Pre-termination diagnostics invented replay evidence.")

    enriched = caught.value.with_runtime_evidence(
        exit_code=-signal.SIGTERM,
        required_exports_present=False,
        replay_available=False,
    )
    enriched_diagnostics = json.loads(str(enriched).split(": ", maxsplit=1)[1])
    _require(enriched_diagnostics["solver_exit_code"] == -signal.SIGTERM, "Solver exit evidence was not retained.")
    _require(enriched_diagnostics["required_exports_present"] is False, "Missing export evidence was not retained.")
    _require(enriched_diagnostics["replay_available"] is False, "Unavailable replay evidence was not retained.")


def test_non_utf8_status_content_is_preserved_and_classified(tmp_path: Path) -> None:
    """Classify a short binary marker without decoding or overwriting it."""
    work_directory = _workspace(tmp_path)
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    original = b"\xff\xfe"
    status_path.write_bytes(original)
    controller = stop_service.SolverStopController(
        _Process(_Clock()),
        work_directory,
        timeout_seconds=10.0,
        graceful_stop_reserve_seconds=2.0,
        monotonic_clock=_Clock(),
    )

    with pytest.raises(stop_service.UnexpectedStopStatusContentError) as caught:
        controller.request_graceful_stop("cancelled")

    _require(status_path.read_bytes() == original, "Unknown binary status content was overwritten.")
    diagnostics = json.loads(str(caught.value).split(": ", maxsplit=1)[1])
    _require(diagnostics["actual_content_class"] == "invalid_utf8_bytes", "Binary status content was misclassified.")
    _require(diagnostics["actual_content_excerpt"] == original.hex(), "Binary status excerpt was not safely encoded.")
