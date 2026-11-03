"""Focused controlled-stop lifecycle and workspace-containment contracts.

Responsibilities:
  - Check controlled stop, owned process escalation, status grammar, and capacity cleanup
  - Check that the stall rule's simulated-time minimum is recorded without changing stop control

Design principles:
  - Ambiguous or changed status bytes remain preserved within exact workspace bounds

This module does NOT:
  - Terminate unrelated processes or delete unknown artifacts
"""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.generation.runtime import generation_runtime_batch as batch_runtime
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


def test_checkout_termination_uses_only_bounded_term_and_kill_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed if an owned checkout survives both bounded signal waits."""
    wait_timeouts: list[float | None] = []
    signals: list[signal.Signals] = []

    class NeverExits:
        pid = 8842

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            if timeout is None:
                pytest.fail("Checkout termination must never use an unbounded wait.")
            wait_timeouts.append(timeout)
            command = "comsol"
            raise subprocess.TimeoutExpired(command, timeout)

    def signal_group(process_group_id: int, value: signal.Signals) -> None:
        assert process_group_id == NeverExits.pid
        signals.append(value)

    monkeypatch.setattr(batch_runtime.os, "killpg", signal_group)

    with pytest.raises(RuntimeError, match="bounded TERM and KILL waits"):
        batch_runtime._terminate_solver_and_wait(cast("subprocess.Popen[str]", NeverExits()))  # pyright: ignore[reportPrivateUsage]

    assert wait_timeouts == [5.0, 5.0]
    assert signals == [signal.SIGTERM, signal.SIGKILL]


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
    (tmp_path / "active-case" / "solved.mph.status").write_text("1790892938336\nRunning", encoding="utf-8")

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    _require(result.exit_code == 0, "Controlled fake process did not exit cleanly.")
    _require(result.timed_out, "Controlled timeout exit lost its timeout classification.")
    _require(result.minimum_simulated_reached_after_seconds is None, "A controller without a stall rule invented a minimum time.")
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


def test_a_stalled_solver_is_stopped_after_the_check_time_with_its_own_reserve(tmp_path: Path) -> None:
    """Below the simulated minimum after the check time, the solver is stopped as stalled and escalated after one reserve, not at the timeout."""
    clock = _Clock()
    process = _Process(clock)
    escalations: list[signal.Signals] = []
    probe_times: list[float] = []

    def signal_group(process_group_id: int, value: signal.Signals) -> None:
        assert process_group_id == process.pid
        escalations.append(value)
        process.finish(-int(value))

    def stuck_at_start() -> float:
        probe_times.append(clock())
        return 54.0

    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=2400.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        signal_process_group=signal_group,
        timestamp_clock=lambda: "2026-10-26T00:00:00+00:00",
        stall_rule=stop_service.StallRule(check_after_seconds=900.0, minimum_simulated_seconds=3600.0, simulated_seconds=stuck_at_start),
    )
    (tmp_path / "active-case" / "solved.mph.status").write_text("1790892938336\nRunning", encoding="utf-8")

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    assert result.stalled
    assert not result.timed_out
    assert result.force_escalated
    assert result.minimum_simulated_reached_after_seconds is None
    assert escalations == [signal.SIGTERM]
    assert 1200.0 <= clock() < 2100.0
    assert len(set(probe_times)) == len(probe_times)
    evidence = json.loads((tmp_path / "active-case" / "runtime" / "stop.json").read_text(encoding="utf-8"))
    assert evidence["reason"] == "stalled"


@pytest.mark.parametrize(("simulated", "reached_after"), [(None, None), (7200.0, 0.0)])
def test_an_unknown_or_progressing_simulated_time_never_triggers_the_stall_rule(
    tmp_path: Path, simulated: float | None, reached_after: float | None
) -> None:
    """Only a known simulated time below the minimum stops the solver; the timeout stays the backstop otherwise."""
    clock = _Clock()
    process = _Process(clock, exit_at=1500.0)
    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=2400.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        stall_rule=stop_service.StallRule(
            check_after_seconds=900.0,
            minimum_simulated_seconds=3600.0,
            simulated_seconds=lambda: simulated,
            observed_simulated_seconds=lambda: simulated,
        ),
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    assert result.reason is None
    assert result.exit_code == 0
    assert result.minimum_simulated_reached_after_seconds == reached_after


def test_recording_the_minimum_leaves_the_stall_decision_reads_unchanged(tmp_path: Path) -> None:
    """The minimum time comes from the observation reader alone; the decision reader is still read only from the check time on."""
    clock = _Clock()
    process = _Process(clock)
    decision_reads: list[float] = []

    def decision_reader() -> float:
        decision_reads.append(clock())
        return 54.0

    def signal_group(_process_group_id: int, value: signal.Signals) -> None:
        process.finish(-int(value))

    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=2400.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        signal_process_group=signal_group,
        poll_interval_seconds=10.0,
        stall_rule=stop_service.StallRule(
            check_after_seconds=900.0,
            minimum_simulated_seconds=3600.0,
            simulated_seconds=decision_reader,
            observed_simulated_seconds=lambda: 5.0 * clock(),
        ),
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    assert result.stalled
    assert decision_reads == [900.0]
    assert result.minimum_simulated_reached_after_seconds == 720.0


def test_a_failing_observation_reader_changes_no_stop_decision(tmp_path: Path) -> None:
    """An observation error is ignored: an operator force request still escalates and no minimum time is invented."""
    clock = _Clock()
    process = _Process(clock)

    def broken_reader() -> float:
        message = "stdout unreadable"
        raise OSError(message)

    def signal_group(_process_group_id: int, value: signal.Signals) -> None:
        process.finish(-int(value))

    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=2400.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        signal_process_group=signal_group,
        poll_interval_seconds=10.0,
        stall_rule=stop_service.StallRule(
            check_after_seconds=900.0,
            minimum_simulated_seconds=3600.0,
            simulated_seconds=lambda: 7200.0,
            observed_simulated_seconds=broken_reader,
        ),
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False, force_requested=lambda: clock() >= 50.0)

    assert result.reason == "cancelled"
    assert result.force_escalated
    assert result.minimum_simulated_reached_after_seconds is None


def test_the_stall_rule_records_the_minimum_with_its_own_uncached_reader(tmp_path: Path) -> None:
    """The decision reader keeps its refresh cache; the observation reader sees appended solver rows at once."""
    stdout = tmp_path / "stdout.log"
    header = "Step Time Stepsize Res Jac Sol Order Tfail NLfail LinErr LinRes\n"
    stdout.write_text(header + "1 10 1 2 1 2 1 0 0 1e-3 1e-4\n", encoding="utf-8")
    config = SimpleNamespace(execution_values={"runtime": {"stall_stop": {"check_after_seconds": 900.0, "minimum_simulated_seconds": 3600.0}}})
    rule = batch_runtime._stall_rule(cast("Any", config), cast("Any", SimpleNamespace(runtime_directory=tmp_path)))  # pyright: ignore[reportPrivateUsage]
    assert rule is not None
    assert rule.observed_simulated_seconds is not None
    assert rule.simulated_seconds() == 10.0
    assert rule.observed_simulated_seconds() == 10.0

    with stdout.open("a", encoding="utf-8") as stream:
        stream.write("2 4000 3990 4 2 4 1 0 0 1e-3 1e-4\n")

    assert rule.simulated_seconds() == 10.0
    assert rule.observed_simulated_seconds() == 4000.0


def test_the_first_simulated_minimum_time_is_recorded_before_the_stall_check_time(tmp_path: Path) -> None:
    """The observation reader is read on every iteration, so a progressing case records its first minimum time long before check_after_seconds."""
    clock = _Clock()
    process = _Process(clock, exit_at=1500.0)
    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=2400.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        poll_interval_seconds=10.0,
        stall_rule=stop_service.StallRule(
            check_after_seconds=900.0,
            minimum_simulated_seconds=3600.0,
            simulated_seconds=lambda: 5.0 * clock(),
            observed_simulated_seconds=lambda: 5.0 * clock(),
        ),
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    assert result.reason is None
    assert result.exit_code == 0
    assert result.minimum_simulated_reached_after_seconds == 720.0


def test_a_minimum_reached_only_at_process_exit_is_recorded_by_the_final_observation(tmp_path: Path) -> None:
    """The observation reader is read once more after the process has exited, because the last solver output may arrive with the exit."""
    clock = _Clock()
    process = _Process(clock, exit_at=100.0)
    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=2400.0,
        graceful_stop_reserve_seconds=300.0,
        monotonic_clock=clock,
        poll_interval_seconds=10.0,
        stall_rule=stop_service.StallRule(
            check_after_seconds=900.0,
            minimum_simulated_seconds=3600.0,
            simulated_seconds=lambda: 10.0,
            observed_simulated_seconds=lambda: 4000.0 if process.poll() is not None else 10.0,
        ),
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    assert result.reason is None
    assert result.minimum_simulated_reached_after_seconds == 100.0


def test_a_forced_stop_result_carries_the_recorded_minimum_time(tmp_path: Path) -> None:
    """Every result, including the one after hard-deadline escalation, keeps the time the minimum was first reached."""
    clock = _Clock()
    process = _Process(clock)

    def signal_group(_process_group_id: int, value: signal.Signals) -> None:
        process.finish(-int(value))

    controller = stop_service.SolverStopController(
        process,
        _workspace(tmp_path),
        timeout_seconds=1000.0,
        graceful_stop_reserve_seconds=100.0,
        monotonic_clock=clock,
        signal_process_group=signal_group,
        poll_interval_seconds=10.0,
        stall_rule=stop_service.StallRule(
            check_after_seconds=900.0,
            minimum_simulated_seconds=3600.0,
            simulated_seconds=lambda: 7200.0 if clock() >= 300.0 else 10.0,
            observed_simulated_seconds=lambda: 7200.0 if clock() >= 300.0 else 10.0,
        ),
    )

    result = controller.wait_for_exit(cancellation_requested=lambda: False)

    assert result.timed_out
    assert result.force_escalated
    assert result.minimum_simulated_reached_after_seconds == 300.0


def test_the_timing_stall_progress_binds_the_rule_minimum_to_the_observed_time() -> None:
    """The persisted record exists only with a stall rule and keeps an unobserved minimum explicit."""
    without_rule = SimpleNamespace(execution_values={"runtime": {}})
    with_rule = SimpleNamespace(execution_values={"runtime": {"stall_stop": {"check_after_seconds": 600, "minimum_simulated_seconds": 1800}}})
    reached = stop_service.StopResult(0, None, False, False, 412.5)

    def progress(config: object, result: stop_service.StopResult | None) -> dict[str, float | None] | None:
        return batch_runtime._stall_progress(cast("Any", config), result)  # pyright: ignore[reportPrivateUsage]

    assert progress(without_rule, reached) is None
    assert progress(with_rule, None) == {"minimum_simulated_seconds": 1800.0, "reached_after_seconds": None}
    assert progress(with_rule, stop_service.StopResult(0, None, False, False)) == {"minimum_simulated_seconds": 1800.0, "reached_after_seconds": None}
    assert progress(with_rule, reached) == {"minimum_simulated_seconds": 1800.0, "reached_after_seconds": 412.5}


@pytest.mark.parametrize("content", ["1790892938336\nDone", "1790892938336\nRunning extra", "Running"])
def test_stop_rejects_nonrunning_or_malformed_solver_status(tmp_path: Path, content: str) -> None:
    """Only an owned exact COMSOL Running record may precede Stop 2."""
    work_directory = _workspace(tmp_path)
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_text(content, encoding="utf-8")
    controller = stop_service.SolverStopController(
        _Process(_Clock()),
        work_directory,
        timeout_seconds=10.0,
        graceful_stop_reserve_seconds=2.0,
        monotonic_clock=_Clock(),
    )

    with pytest.raises(stop_service.UnexpectedStopStatusContentError):
        controller.request_graceful_stop("timeout")
    assert status_path.read_text(encoding="utf-8") == content


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

    enriched = caught.value.with_runtime_evidence(exit_code=-signal.SIGTERM, required_exports_present=False)
    enriched_diagnostics = json.loads(str(enriched).split(": ", maxsplit=1)[1])
    _require(enriched_diagnostics["solver_exit_code"] == -signal.SIGTERM, "Solver exit evidence was not retained.")
    _require(enriched_diagnostics["required_exports_present"] is False, "Missing export evidence was not retained.")


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


def test_capacity_status_created_by_one_exited_checkout_is_removed_exactly(tmp_path: Path) -> None:
    """Admit only the known small COMSOL status class and revalidate before unlink."""
    work_directory = _workspace(tmp_path)
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=2,
    )
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_text("1787215759123\nFailed\n", encoding="utf-8")

    artifact = stop_service.inspect_capacity_checkout_status(
        prelaunch,
        process_id=9812,
        process_exit_code=0,
        temporary_capacity_classified=True,
        solver_progress_started=False,
        required_exports_exist=False,
        scientific_result_exists=False,
    )

    assert artifact is not None
    assert artifact.prelaunch.checkout_index == 2
    assert artifact.status_state == "Failed"
    assert artifact.status_timestamp_milliseconds == 1787215759123
    assert artifact.file_size_bytes == status_path.stat().st_size
    assert len(artifact.content_sha256) == 64
    stop_service.remove_capacity_checkout_status(artifact)
    assert not status_path.exists()


@pytest.mark.parametrize(
    ("payload", "expected_state"),
    [
        (b"1787228251108\nRunning", "Running"),
        (b"1787228251108\r\nDone\r\n", "Done"),
        (b"1787228251108\nFailed\n", "Failed"),
        (b"1787228251108\nError", "Error"),
    ],
)
def test_capacity_status_byte_grammar_accepts_only_maintained_states(
    tmp_path: Path,
    payload: bytes,
    expected_state: str,
) -> None:
    """Accept exact bounded ASCII COMSOL status records after capacity proof."""
    work_directory = _workspace(tmp_path)
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=1,
    )
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_bytes(payload)

    artifact = stop_service.inspect_capacity_checkout_status(
        prelaunch,
        process_id=9812,
        process_exit_code=0,
        temporary_capacity_classified=True,
        solver_progress_started=False,
        required_exports_exist=False,
        scientific_result_exists=False,
    )

    assert artifact is not None
    assert artifact.status_state == expected_state
    assert artifact.content_sha256 == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        b"Error",
        b"1787228251108\nError\nforeign",
        b"1787228251108\nError suffix",
        b'{"state":"Error"}',
        b"1787228251108\nError\nstack trace",
        b"1787228251108\nError\xff",
        b"1" * 161,
        b"178722825108\nError",
        b"1787228251108\rError",
    ],
)
def test_capacity_status_byte_grammar_rejects_unknown_payloads(
    tmp_path: Path,
    payload: bytes,
) -> None:
    """Reject malformed, non-ASCII, oversized, and extended status payloads."""
    work_directory = _workspace(tmp_path)
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=1,
    )
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_bytes(payload)

    with pytest.raises(
        stop_service.UnsafeCapacityStatusArtifactError,
        match=r"unknown_capacity_status_content|status_content_exceeds_bound",
    ):
        stop_service.inspect_capacity_checkout_status(
            prelaunch,
            process_id=9812,
            process_exit_code=0,
            temporary_capacity_classified=True,
            solver_progress_started=False,
            required_exports_exist=False,
            scientific_result_exists=False,
        )

    assert status_path.read_bytes() == payload


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"process_exit_code": None}, "checkout_process_may_still_be_alive"),
        ({"temporary_capacity_classified": False}, "checkout_not_strongly_classified"),
        ({"solver_progress_started": True}, "solver_progress_started"),
        ({"required_exports_exist": True}, "required_exports_exist"),
        ({"scientific_result_exists": True}, "scientific_result_exists"),
    ],
)
def test_capacity_status_recovery_rejects_ambiguous_lifecycle(
    tmp_path: Path,
    overrides: dict[str, object],
    reason: str,
) -> None:
    """Do not clean a status artifact unless every capacity-only predicate holds."""
    work_directory = _workspace(tmp_path)
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=1,
    )
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_text("1787228251108\nError", encoding="utf-8")
    arguments: dict[str, Any] = {
        "process_id": 9812,
        "process_exit_code": 0,
        "temporary_capacity_classified": True,
        "solver_progress_started": False,
        "required_exports_exist": False,
        "scientific_result_exists": False,
    }
    arguments.update(overrides)

    with pytest.raises(stop_service.UnsafeCapacityStatusArtifactError, match=reason):
        stop_service.inspect_capacity_checkout_status(prelaunch, **arguments)

    assert status_path.exists()


def test_capacity_status_recovery_rejects_preexisting_unknown_and_changed_content(
    tmp_path: Path,
) -> None:
    """Preserve status files that are pre-existing, unknown, or changed after admission."""
    work_directory = _workspace(tmp_path)
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_text("1787215759123\nFailed\n", encoding="utf-8")
    with pytest.raises(
        stop_service.UnsafeCapacityStatusArtifactError,
        match="status_path_existed_before_checkout",
    ):
        stop_service.prepare_capacity_checkout_status(
            work_directory,
            checkout_index=1,
        )

    status_path.unlink()
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=1,
    )
    status_path.write_text("foreign marker\n", encoding="utf-8")
    with pytest.raises(
        stop_service.UnsafeCapacityStatusArtifactError,
        match="unknown_capacity_status_content",
    ):
        stop_service.inspect_capacity_checkout_status(
            prelaunch,
            process_id=9812,
            process_exit_code=0,
            temporary_capacity_classified=True,
            solver_progress_started=False,
            required_exports_exist=False,
            scientific_result_exists=False,
        )
    assert status_path.read_text(encoding="utf-8") == "foreign marker\n"

    status_path.write_text("1787215759123\nDone\n", encoding="utf-8")
    artifact = stop_service.inspect_capacity_checkout_status(
        prelaunch,
        process_id=9812,
        process_exit_code=0,
        temporary_capacity_classified=True,
        solver_progress_started=False,
        required_exports_exist=False,
        scientific_result_exists=False,
    )
    assert artifact is not None
    status_path.write_text("1787215759124\nDone\n", encoding="utf-8")
    with pytest.raises(
        stop_service.UnsafeCapacityStatusArtifactError,
        match=r"status_identity_changed_before_cleanup|status_content_changed_before_cleanup",
    ):
        stop_service.remove_capacity_checkout_status(artifact)
    assert status_path.exists()


def test_capacity_status_cleanup_failure_preserves_the_artifact(tmp_path: Path) -> None:
    """Fail closed when the exact admitted unlink cannot be completed."""
    work_directory = _workspace(tmp_path)
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=1,
    )
    status_path = work_directory / stop_service.STOP_STATUS_FILENAME
    status_path.write_text("1787215759123\nFailed\n", encoding="utf-8")
    artifact = stop_service.inspect_capacity_checkout_status(
        prelaunch,
        process_id=9812,
        process_exit_code=0,
        temporary_capacity_classified=True,
        solver_progress_started=False,
        required_exports_exist=False,
        scientific_result_exists=False,
    )
    assert artifact is not None

    def fail_unlink(_path: Path) -> None:
        message = "synthetic unlink denial"
        raise PermissionError(message)

    with pytest.raises(
        stop_service.UnsafeCapacityStatusArtifactError,
        match="status_cleanup_failed",
    ):
        stop_service.remove_capacity_checkout_status(
            artifact,
            unlink_status=fail_unlink,
        )
    assert status_path.exists()
