"""
generation_runtime_stop.py

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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
_MAX_STATUS_EXCERPT_BYTES = 160

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


_CAPACITY_STATUS_PATTERN = re.compile(rb"(?P<timestamp>[0-9]{13})\r?\n(?P<state>Running|Done|Failed|Error)(?:\r?\n)?")
_MAX_CAPACITY_STATUS_BYTES = 160
_WORKSPACE_NOT_OWNED = "workspace_not_owned_by_current_user"
_STATUS_PREEXISTED = "status_path_existed_before_checkout"
_PROCESS_MAY_BE_ALIVE = "checkout_process_may_still_be_alive"
_NOT_TEMPORARY_CAPACITY = "checkout_not_strongly_classified_as_temporary_capacity"
_SOLVER_PROGRESS_CONFLICT = "solver_progress_started_before_status_recovery"
_EXPORT_CONFLICT = "required_exports_exist_before_status_recovery"
_SCIENTIFIC_RESULT_CONFLICT = "scientific_result_exists_before_status_recovery"
_WORKSPACE_IDENTITY_CHANGED = "checkout_workspace_identity_changed"
_NOT_OWNED_REGULAR_FILE = "status_file_is_not_an_owned_regular_file"
_STATUS_TOO_LARGE = "status_content_exceeds_bound"
_UNKNOWN_STATUS_CONTENT = "unknown_capacity_status_content"
_STATUS_DISAPPEARED = "admitted_status_disappeared_before_cleanup"
_STATUS_IDENTITY_CHANGED = "status_identity_changed_before_cleanup"
_STATUS_CONTENT_CHANGED = "status_content_changed_before_cleanup"
_STATUS_CLEANUP_FAILED = "status_cleanup_failed"
_STATUS_CLEANUP_NOT_OBSERVED = "status_cleanup_was_not_observed"


@dataclass(frozen=True, slots=True)
class CapacityStatusPrelaunch:
    """Bind one absent canonical status path to one exact checkout workspace."""

    checkout_index: int
    path: Path
    checked_at: str
    workspace_device: int
    workspace_inode: int


@dataclass(frozen=True, slots=True)
class CapacityStatusArtifact:
    """Describe one small COMSOL status record created by a completed checkout."""

    prelaunch: CapacityStatusPrelaunch
    process_id: int
    process_exit_code: int
    file_device: int
    file_inode: int
    file_user_id: int
    file_size_bytes: int
    file_modified_nanoseconds: int
    content_sha256: str
    content_excerpt: str
    status_timestamp_milliseconds: int
    status_state: str


class UnsafeCapacityStatusArtifactError(RuntimeError):
    """Reject a status artifact whose checkout-local ownership is not exact."""

    def __init__(self, reason: str, *, path: Path, diagnostics: dict[str, object] | None = None) -> None:
        """Initialize one bounded fail-closed status-artifact diagnostic."""
        self.reason = reason
        self.path = path
        self.diagnostics = {"reason": reason, "status_path": str(path), **(diagnostics or {})}
        message = "Unsafe COMSOL capacity-checkout status artifact: " + json.dumps(
            self.diagnostics,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        super().__init__(message)


def prepare_capacity_checkout_status(
    active_work_directory: Path | str,
    *,
    checkout_index: int,
) -> CapacityStatusPrelaunch:
    """Require canonical status absence immediately before one checkout launch."""
    if isinstance(checkout_index, bool) or not isinstance(checkout_index, int) or checkout_index < 1:
        message = "Capacity-checkout status ownership requires a positive checkout index."
        raise ValueError(message)
    work_directory = _reject_symbolic_links(Path(active_work_directory), require_directory=True)
    workspace_identity = work_directory.stat()
    if workspace_identity.st_uid != os.geteuid():
        raise UnsafeCapacityStatusArtifactError(
            _WORKSPACE_NOT_OWNED,
            path=work_directory,
            diagnostics={
                "workspace_user_id": workspace_identity.st_uid,
                "effective_user_id": os.geteuid(),
            },
        )
    path = derive_stop_status_path(work_directory)
    try:
        identity = path.lstat()
    except FileNotFoundError:
        return CapacityStatusPrelaunch(
            checkout_index=checkout_index,
            path=path,
            checked_at=_utc_timestamp(),
            workspace_device=workspace_identity.st_dev,
            workspace_inode=workspace_identity.st_ino,
        )
    raise UnsafeCapacityStatusArtifactError(
        _STATUS_PREEXISTED,
        path=path,
        diagnostics={
            "file_type": stat.S_IFMT(identity.st_mode),
            "file_size_bytes": identity.st_size,
        },
    )


def _read_capacity_status_without_following(
    path: Path,
) -> tuple[os.stat_result, bytes]:
    """Read at most the bounded status payload through one no-follow file handle."""
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        identity = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = _MAX_CAPACITY_STATUS_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return identity, b"".join(chunks)
    finally:
        os.close(descriptor)


def inspect_capacity_checkout_status(
    prelaunch: CapacityStatusPrelaunch,
    *,
    process_id: int,
    process_exit_code: int | None,
    temporary_capacity_classified: bool,
    solver_progress_started: bool,
    required_exports_exist: bool,
    scientific_result_exists: bool,
) -> CapacityStatusArtifact | None:
    """Admit one newly created known COMSOL status record after a capacity-only exit."""
    if process_id <= 0:
        message = "Capacity-checkout status ownership requires a positive process ID."
        raise ValueError(message)
    if process_exit_code is None:
        raise UnsafeCapacityStatusArtifactError(
            _PROCESS_MAY_BE_ALIVE,
            path=prelaunch.path,
            diagnostics={"process_id": process_id},
        )
    if not temporary_capacity_classified:
        raise UnsafeCapacityStatusArtifactError(
            _NOT_TEMPORARY_CAPACITY,
            path=prelaunch.path,
        )
    if solver_progress_started:
        raise UnsafeCapacityStatusArtifactError(
            _SOLVER_PROGRESS_CONFLICT,
            path=prelaunch.path,
        )
    if required_exports_exist:
        raise UnsafeCapacityStatusArtifactError(
            _EXPORT_CONFLICT,
            path=prelaunch.path,
        )
    if scientific_result_exists:
        raise UnsafeCapacityStatusArtifactError(
            _SCIENTIFIC_RESULT_CONFLICT,
            path=prelaunch.path,
        )

    work_directory = _reject_symbolic_links(prelaunch.path.parent, require_directory=True)
    workspace_identity = work_directory.stat()
    if workspace_identity.st_uid != os.geteuid():
        raise UnsafeCapacityStatusArtifactError(
            _WORKSPACE_NOT_OWNED,
            path=prelaunch.path,
            diagnostics={
                "workspace_user_id": workspace_identity.st_uid,
                "effective_user_id": os.geteuid(),
            },
        )
    if workspace_identity.st_dev != prelaunch.workspace_device or workspace_identity.st_ino != prelaunch.workspace_inode:
        raise UnsafeCapacityStatusArtifactError(
            _WORKSPACE_IDENTITY_CHANGED,
            path=prelaunch.path,
        )
    path = derive_stop_status_path(work_directory)
    try:
        identity, raw = _read_capacity_status_without_following(path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise UnsafeCapacityStatusArtifactError(
            _NOT_OWNED_REGULAR_FILE,
            path=path,
            diagnostics={"error_type": type(error).__name__},
        ) from error
    if not stat.S_ISREG(identity.st_mode) or identity.st_uid != os.geteuid():
        raise UnsafeCapacityStatusArtifactError(
            _NOT_OWNED_REGULAR_FILE,
            path=path,
            diagnostics={
                "file_type": stat.S_IFMT(identity.st_mode),
                "file_user_id": identity.st_uid,
                "effective_user_id": os.geteuid(),
            },
        )
    if identity.st_size > _MAX_CAPACITY_STATUS_BYTES or len(raw) > _MAX_CAPACITY_STATUS_BYTES:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_TOO_LARGE,
            path=path,
            diagnostics={"file_size_bytes": identity.st_size},
        )
    if len(raw) != identity.st_size:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_IDENTITY_CHANGED,
            path=path,
        )
    match = _CAPACITY_STATUS_PATTERN.fullmatch(raw)
    if match is None:
        bounded = raw[:_MAX_CAPACITY_STATUS_BYTES]
        try:
            excerpt = bounded.decode("utf-8")
        except UnicodeDecodeError:
            excerpt = bounded.hex()
        raise UnsafeCapacityStatusArtifactError(
            _UNKNOWN_STATUS_CONTENT,
            path=path,
            diagnostics={
                "file_size_bytes": identity.st_size,
                "content_excerpt": excerpt,
                "content_sha256": hashlib.sha256(raw).hexdigest(),
            },
        )
    return CapacityStatusArtifact(
        prelaunch=prelaunch,
        process_id=process_id,
        process_exit_code=process_exit_code,
        file_device=identity.st_dev,
        file_inode=identity.st_ino,
        file_user_id=identity.st_uid,
        file_size_bytes=identity.st_size,
        file_modified_nanoseconds=identity.st_mtime_ns,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        content_excerpt=raw.decode("utf-8").rstrip("\r\n"),
        status_timestamp_milliseconds=int(match.group("timestamp")),
        status_state=match.group("state").decode("ascii"),
    )


def _unlink_status(path: Path) -> None:
    """Unlink one already-admitted ordinary status path."""
    path.unlink()


def remove_capacity_checkout_status(
    artifact: CapacityStatusArtifact,
    *,
    unlink_status: Callable[[Path], None] = _unlink_status,
) -> None:
    """Revalidate and atomically unlink only the exact admitted status artifact."""
    path = derive_stop_status_path(artifact.prelaunch.path.parent)
    try:
        identity, raw = _read_capacity_status_without_following(path)
    except FileNotFoundError as error:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_DISAPPEARED,
            path=path,
        ) from error
    except OSError as error:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_IDENTITY_CHANGED,
            path=path,
            diagnostics={"error_type": type(error).__name__},
        ) from error
    expected_identity = (
        artifact.file_device,
        artifact.file_inode,
        artifact.file_user_id,
        artifact.file_size_bytes,
        artifact.file_modified_nanoseconds,
    )
    actual_identity = (
        identity.st_dev,
        identity.st_ino,
        identity.st_uid,
        identity.st_size,
        identity.st_mtime_ns,
    )
    if not stat.S_ISREG(identity.st_mode) or actual_identity != expected_identity:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_IDENTITY_CHANGED,
            path=path,
        )
    if len(raw) != identity.st_size:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_IDENTITY_CHANGED,
            path=path,
        )
    if hashlib.sha256(raw).hexdigest() != artifact.content_sha256:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_CONTENT_CHANGED,
            path=path,
        )
    try:
        unlink_status(path)
    except OSError as error:
        raise UnsafeCapacityStatusArtifactError(
            _STATUS_CLEANUP_FAILED,
            path=path,
            diagnostics={"error_type": type(error).__name__},
        ) from error
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise UnsafeCapacityStatusArtifactError(
        _STATUS_CLEANUP_NOT_OBSERVED,
        path=path,
    )


class UnexpectedStopStatusContentError(RuntimeError):
    """Report bounded evidence for an owned status file that is unsafe to replace."""

    def __init__(
        self,
        diagnostics: dict[str, object],
        *,
        exit_code: int | None = None,
        required_exports_present: bool | None = None,
        replay_available: bool | None = None,
    ) -> None:
        """Initialize exact status-file and optional post-termination evidence."""
        self.diagnostics = {
            **diagnostics,
            "solver_exit_code": exit_code,
            "required_exports_present": required_exports_present,
            "replay_available": replay_available,
        }
        self.exit_code = exit_code
        self.timed_out = False
        message = "Refusing to replace unexpected COMSOL stop status content: " + json.dumps(
            self.diagnostics,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        super().__init__(message)

    def with_runtime_evidence(
        self,
        *,
        exit_code: int,
        required_exports_present: bool,
        replay_available: bool,
    ) -> UnexpectedStopStatusContentError:
        """Return the same rejection enriched after the owned solver has exited."""
        base = {
            key: value for key, value in self.diagnostics.items() if key not in {"solver_exit_code", "required_exports_present", "replay_available"}
        }
        return UnexpectedStopStatusContentError(
            base,
            exit_code=exit_code,
            required_exports_present=required_exports_present,
            replay_available=replay_available,
        )


def _unexpected_status_diagnostics(
    path: Path,
    command: str,
    accepted_existing: frozenset[str],
) -> dict[str, object]:
    """Return bounded content and ownership evidence for one rejected status file."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(_MAX_STATUS_EXCERPT_BYTES + 1)
    bounded = raw[:_MAX_STATUS_EXCERPT_BYTES]
    try:
        excerpt = bounded.decode("utf-8")
    except UnicodeDecodeError:
        actual_class = "invalid_utf8_bytes"
        excerpt = bounded.hex()
    else:
        if not raw:
            actual_class = "empty"
        elif excerpt == STOP_COMMAND:
            actual_class = "stop_command"
        elif excerpt == CANCEL_COMMAND:
            actual_class = "cancel_command"
        else:
            actual_class = "unexpected_utf8_text"
    expected = sorted({command, *accepted_existing})
    return {
        "status_path": str(path),
        "expected_content_class": "exact_command_or_admitted_predecessor",
        "expected_contents": [value.rstrip("\n") for value in expected],
        "actual_content_class": actual_class,
        "actual_content_excerpt": excerpt,
        "actual_content_excerpt_truncated": size > _MAX_STATUS_EXCERPT_BYTES,
        "file_size": size,
        "ownership_evidence": "derived_regular_file_below_owned_active_workspace",
    }


def _write_verified_command(path: Path, command: str, *, accepted_existing: frozenset[str]) -> None:
    """Atomically publish one exact COMSOL command without replacing unknown content."""
    if path.exists():
        admitted = {command, *accepted_existing}
        if path.stat().st_size > max(len(value.encode("utf-8")) for value in admitted):
            raise UnexpectedStopStatusContentError(_unexpected_status_diagnostics(path, command, accepted_existing))
        try:
            existing = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise UnexpectedStopStatusContentError(_unexpected_status_diagnostics(path, command, accepted_existing)) from error
        if existing == command:
            return
        if existing not in accepted_existing:
            raise UnexpectedStopStatusContentError(_unexpected_status_diagnostics(path, command, accepted_existing))
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
