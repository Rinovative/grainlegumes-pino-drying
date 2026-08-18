"""
===============================================================================
generation_runtime_stop.py
===============================================================================
Control one owned COMSOL process through its documented status-file protocol.
Responsibilities:
  - Validate bounded graceful-stop and hard-timeout timing contracts
  - Derive and publish COMSOL status commands only inside one owned workspace
  - Persist concise stop-request evidence with an explicit reason and timestamp
  - Escalate an unresponsive owned process group through Cancel, TERM, and KILL
Design principles:
  - Status-file paths are fail-closed against escapes and symbolic links
  - Graceful stopping precedes process-group escalation at the hard deadline
  - Injected time and wait boundaries make lifecycle behavior directly testable
This module does NOT:
  - Start COMSOL processes or prepare case workspaces
  - Classify solver output or publish simulation results
  - Signal process groups that were not supplied as owned solver processes
===============================================================================
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from src import common

STOP_COMMAND = "Stop 2\n"
CANCEL_COMMAND = "Cancel\n"
STOP_STATUS_FILENAME = "solved.mph.status"
STOP_EVIDENCE_FILENAME = "stop.json"
STOP_EVIDENCE_SCHEMA_KIND = "generation_runtime_stop"
STOP_EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_POLL_INTERVAL_SECONDS = 0.25
DEFAULT_ESCALATION_WAIT_SECONDS = 5.0

StopReason = Literal["timeout", "cancelled"]


class OwnedSolverProcess(Protocol):
    """Describe the owned process boundary required for stop control."""

    pid: int

    def poll(self) -> int | None:
        """Return the process status when it has exited."""
        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process completion and return its status."""
        ...


WaitProcess = Callable[[OwnedSolverProcess, float | None], int]
SignalProcessGroup = Callable[[int, signal.Signals], None]
MonotonicClock = Callable[[], float]
TimestampClock = Callable[[], str]
CancellationRequested = Callable[[], bool]
ForceRequested = Callable[[], bool]
ProgressCallback = Callable[[], None]


@dataclass(frozen=True, slots=True)
class StopResult:
    """
    Describe one owned solver process after lifecycle stop control.

    Attributes
    ----------
    exit_code : int
        Final process exit status.
    reason : StopReason | None
        The accepted graceful-stop reason, if a stop was requested.
    graceful_stop_requested : bool
        Whether the COMSOL ``Stop 2`` command was published.
    force_escalated : bool
        Whether the hard deadline required ``Cancel`` and process-group signals.

    """

    exit_code: int
    reason: StopReason | None
    graceful_stop_requested: bool
    force_escalated: bool

    @property
    def timed_out(self) -> bool:
        """Return whether timeout control, rather than cancellation, ended the case."""
        return self.reason == "timeout"


def _utc_timestamp() -> str:
    """Return one timezone-aware UTC evidence timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _default_wait(process: OwnedSolverProcess, timeout: float | None) -> int:
    """Wait through the owned process interface."""
    return process.wait(timeout=timeout)


def _default_signal_process_group(process_group_id: int, value: signal.Signals) -> None:
    """Signal one caller-owned solver process group."""
    os.killpg(process_group_id, value)


def validate_stop_timing(*, timeout_seconds: float, graceful_stop_reserve_seconds: float) -> None:
    """
    Validate one hard timeout and its earlier graceful-stop reserve.

    Parameters
    ----------
    timeout_seconds : float
        Positive hard deadline measured from solver start.
    graceful_stop_reserve_seconds : float
        Positive time reserved before the hard deadline for COMSOL controlled stop.

    Raises
    ------
    ValueError
        If the reserve is not strictly inside the timeout interval.

    """
    if timeout_seconds <= 0.0:
        msg = f"timeout_seconds must be positive, got {timeout_seconds!r}."
        raise ValueError(msg)
    if not 0.0 < graceful_stop_reserve_seconds < timeout_seconds:
        msg = (
            "graceful_stop_reserve_seconds must satisfy "
            "0 < graceful_stop_reserve_seconds < timeout_seconds; "
            f"got {graceful_stop_reserve_seconds!r} and {timeout_seconds!r}."
        )
        raise ValueError(msg)


def _reject_symbolic_links(path: Path, *, require_directory: bool) -> Path:
    """Resolve one existing absolute path while rejecting every symbolic-link component."""
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            msg = f"Owned runtime path is missing: {current}."
            raise FileNotFoundError(msg) from error
        if stat.S_ISLNK(mode):
            msg = f"Owned runtime path must not contain symbolic links: {current}."
            raise RuntimeError(msg)
    if require_directory and not current.is_dir():
        msg = f"Owned active work directory is not a directory: {current}."
        raise NotADirectoryError(msg)
    return current


def _derive_owned_path(active_work_directory: Path | str, *parts: str, create_parent: bool = False) -> Path:
    """Derive one non-symlink path strictly below an owned active workspace."""
    work_directory = _reject_symbolic_links(Path(active_work_directory), require_directory=True)
    candidate = work_directory.joinpath(*parts)
    if candidate.parent != work_directory and create_parent:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    parent = _reject_symbolic_links(candidate.parent, require_directory=True)
    if parent != work_directory.joinpath(*parts[:-1]).absolute():
        msg = f"Owned runtime path escaped its active work directory: {candidate}."
        raise RuntimeError(msg)
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError:
        return candidate
    if stat.S_ISLNK(mode):
        msg = f"Owned runtime output must not be a symbolic link: {candidate}."
        raise RuntimeError(msg)
    if not stat.S_ISREG(mode):
        msg = f"Owned runtime output is not a regular file: {candidate}."
        raise RuntimeError(msg)
    return candidate


def derive_stop_status_path(active_work_directory: Path | str) -> Path:
    """Return the exact COMSOL status-file path below one owned active workspace."""
    return _derive_owned_path(active_work_directory, STOP_STATUS_FILENAME)


def _derive_stop_evidence_path(active_work_directory: Path | str) -> Path:
    """Return the owned stop-evidence path, creating only its runtime directory."""
    return _derive_owned_path(active_work_directory, "runtime", STOP_EVIDENCE_FILENAME, create_parent=True)


def _write_verified_command(path: Path, command: str, *, accepted_existing: frozenset[str]) -> None:
    """Atomically publish one exact COMSOL command without replacing unknown content."""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == command:
            return
        if existing not in accepted_existing:
            msg = f"Refusing to replace unexpected COMSOL stop status content at {path}."
            raise RuntimeError(msg)
    common.serialization.atomic_write_text(path, command)
    if path.read_text(encoding="utf-8") != command:
        msg = f"COMSOL stop status command was not durably verified at {path}."
        raise RuntimeError(msg)


class SolverStopController:
    """
    Control graceful stopping and hard-deadline escalation for one owned solver.

    Parameters
    ----------
    process : OwnedSolverProcess
        COMSOL process started in its own session, whose process group is its PID.
    active_work_directory : Path | str
        Exact prepared work directory owned by the active solver.
    timeout_seconds : float
        Hard solver deadline measured from controller construction.
    graceful_stop_reserve_seconds : float
        Reserved controlled-stop interval before the hard deadline.
    monotonic_clock : MonotonicClock, optional
        Monotonic clock for deadline control.
    wait_process : WaitProcess, optional
        Bounded process wait operation.
    signal_process_group : SignalProcessGroup, optional
        Owned process-group signaling operation.
    timestamp_clock : TimestampClock, optional
        UTC evidence timestamp source.
    poll_interval_seconds : float, optional
        Maximum regular wait interval.
    escalation_wait_seconds : float, optional
        Bounded wait between Cancel, TERM, and KILL escalation steps.

    """

    def __init__(
        self,
        process: OwnedSolverProcess,
        active_work_directory: Path | str,
        *,
        timeout_seconds: float,
        graceful_stop_reserve_seconds: float,
        monotonic_clock: MonotonicClock,
        wait_process: WaitProcess = _default_wait,
        signal_process_group: SignalProcessGroup = _default_signal_process_group,
        timestamp_clock: TimestampClock = _utc_timestamp,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        escalation_wait_seconds: float = DEFAULT_ESCALATION_WAIT_SECONDS,
    ) -> None:
        """Initialize one controller for a single owned solver process group."""
        validate_stop_timing(
            timeout_seconds=timeout_seconds,
            graceful_stop_reserve_seconds=graceful_stop_reserve_seconds,
        )
        if process.pid <= 0:
            msg = f"Owned solver process PID must be positive, got {process.pid!r}."
            raise ValueError(msg)
        if poll_interval_seconds <= 0.0 or escalation_wait_seconds <= 0.0:
            msg = "poll_interval_seconds and escalation_wait_seconds must be positive."
            raise ValueError(msg)
        self._process = process
        self._active_work_directory = Path(active_work_directory)
        self._monotonic_clock = monotonic_clock
        self._wait_process = wait_process
        self._signal_process_group = signal_process_group
        self._timestamp_clock = timestamp_clock
        self._poll_interval_seconds = poll_interval_seconds
        self._escalation_wait_seconds = escalation_wait_seconds
        self._started_at = monotonic_clock()
        self._graceful_deadline = self._started_at + timeout_seconds - graceful_stop_reserve_seconds
        self._hard_deadline = self._started_at + timeout_seconds
        self._reason: StopReason | None = None
        self._force_escalated = False

    def request_graceful_stop(self, reason: StopReason) -> None:
        """Atomically request a COMSOL controlled stop and persist its evidence."""
        if self._reason is not None:
            return
        status_path = derive_stop_status_path(self._active_work_directory)
        _write_verified_command(status_path, STOP_COMMAND, accepted_existing=frozenset())
        requested_at = self._timestamp_clock()
        evidence_path = _derive_stop_evidence_path(self._active_work_directory)
        common.serialization.atomic_write_json(
            evidence_path,
            {
                "schema_kind": STOP_EVIDENCE_SCHEMA_KIND,
                "schema_version": STOP_EVIDENCE_SCHEMA_VERSION,
                "reason": reason,
                "requested_at": requested_at,
                "command": STOP_COMMAND.rstrip(),
            },
        )
        self._reason = reason

    def wait_for_exit(
        self,
        *,
        cancellation_requested: CancellationRequested,
        force_requested: ForceRequested = lambda: False,
        progress_callback: ProgressCallback = lambda: None,
    ) -> StopResult:
        """
        Wait through graceful stop and bounded hard-deadline escalation.

        Parameters
        ----------
        cancellation_requested : CancellationRequested
            Callback that reports cooperative cancellation for this owned case.
        force_requested : ForceRequested, optional
            Callback that reports an operator-requested immediate force escalation.
        progress_callback : ProgressCallback, optional
            Best-effort observation callback invoked after bounded wait intervals.

        Returns
        -------
        StopResult
            Final process status and the accepted stop-control outcome.

        """
        while True:
            completed = self._process.poll()
            if completed is not None:
                return StopResult(completed, self._reason, self._reason is not None, self._force_escalated)
            now = self._monotonic_clock()
            if force_requested():
                self.request_graceful_stop("cancelled")
                return self._force_stop()
            if cancellation_requested():
                self.request_graceful_stop("cancelled")
            elif now >= self._graceful_deadline:
                self.request_graceful_stop("timeout")
            if now >= self._hard_deadline:
                return self._force_stop()
            next_deadline = self._hard_deadline if self._reason is not None else self._graceful_deadline
            wait_seconds = min(self._poll_interval_seconds, max(0.0, next_deadline - now))
            try:
                completed = self._wait_process(self._process, wait_seconds)
            except subprocess.TimeoutExpired:
                progress_callback()
                continue
            return StopResult(completed, self._reason, self._reason is not None, self._force_escalated)

    def _force_stop(self) -> StopResult:
        """Escalate the owned process group only after the hard deadline."""
        if self._process.poll() is not None:
            return StopResult(self._wait_process(self._process, 0.0), self._reason, self._reason is not None, self._force_escalated)
        status_path = derive_stop_status_path(self._active_work_directory)
        _write_verified_command(status_path, CANCEL_COMMAND, accepted_existing=frozenset({STOP_COMMAND}))
        self._force_escalated = True
        try:
            completed = self._wait_process(self._process, self._escalation_wait_seconds)
        except subprocess.TimeoutExpired:
            self._signal_owned_group(signal.SIGTERM)
        else:
            return StopResult(completed, self._reason, self._reason is not None, True)
        try:
            completed = self._wait_process(self._process, self._escalation_wait_seconds)
        except subprocess.TimeoutExpired:
            self._signal_owned_group(signal.SIGKILL)
            completed = self._wait_process(self._process, self._escalation_wait_seconds)
        return StopResult(completed, self._reason, self._reason is not None, True)

    def _signal_owned_group(self, value: signal.Signals) -> None:
        """Signal only the process group identified by the owned solver PID."""
        if self._process.poll() is not None:
            return
        try:
            self._signal_process_group(self._process.pid, value)
        except ProcessLookupError:
            return
