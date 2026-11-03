"""Run generic tickets as Slurm job steps inside one Generation node job.

Responsibilities:
  - Define and validate the node ticket, receipt, node, and exit-record contracts
  - Publish tickets and the cancel marker for dispatchers, accepting an exact replay after a dispatcher crash
  - Start ticket steps with srun --exact inside the node allocation and refill free cores when a step ends
  - Enforce strict ticket order, group barriers, core accounting, and the remaining-time gate
  - Honour a group success target: run candidates only while successes plus running steps stay below it, then skip
    the group's remaining tickets with a skipped receipt (a failed step frees its place for the next candidate)
  - Stop launching on a cancel marker or termination signal, forward TERM to running steps, and always leave an exit record

Design principles:
  - The worker knows no campaign or benchmark: dispatchers write immutable tickets, the worker writes receipts
  - Every record is published once and never rewritten; an invalid ticket stops the worker explicitly
  - A step starts only when the remaining node time is known and sufficient; unknown time never starts a step
  - Cancellation signals the srun clients, which forward TERM to the step tasks; Slurm reclaims what survives the node job

This module does NOT:
  - Decide which cases run, interpret step outcomes beyond exit status zero, retry failed steps, or reconcile campaign evidence
  - Submit node jobs or choose nodes, partitions, or node-job slots
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from src import common

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import FrameType

TICKET_SCHEMA: Final = "generation_node_ticket"
STARTED_RECEIPT_SCHEMA: Final = "generation_node_started_receipt"
FINISHED_RECEIPT_SCHEMA: Final = "generation_node_finished_receipt"
NODE_RECORD_SCHEMA: Final = "generation_node"
NODE_EXIT_SCHEMA: Final = "generation_node_exit"
SCHEMA_VERSION: Final = 1
STEP_KIND: Final = "step"
STOP_KIND: Final = "stop"
TICKETS_DIRECTORY: Final = "tickets"
RECEIPTS_DIRECTORY: Final = "receipts"
CONTROL_DIRECTORY: Final = "control"
CANCEL_MARKER: Final = "cancel"
NODE_RECORD_FILE: Final = "node.json"
NODE_EXIT_FILE: Final = "node_exit.json"
LAUNCHER_NAME: Final = "generation_node.sh"
MAX_TICKET_SEQ: Final = 999_999
MAX_LABEL_LENGTH: Final = 48
STEP_START_WAIT_SECONDS: Final = 120

ReceiptState = Literal["started", "finished"]
NodeRecord = Literal["node", "exit"]

_TICKET_FILE = re.compile(r"(?P<seq>[0-9]{6})\.json")
_LABEL = re.compile(r"[A-Za-z0-9._-]+")
_MEMORY = re.compile(r"[1-9][0-9]*[MGT]")
_TIME_LIMIT = re.compile(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]")
_ENV_KEY = re.compile(r"GENERATION_[A-Z0-9_]+|STORAGE_ROOT")
_REMAINING = re.compile(r"(?:(?P<days>[0-9]+)-)?(?:(?P<hours>[0-9]+):)?(?P<minutes>[0-9]+):(?P<seconds>[0-9]{2})")
_STEP_KEYS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "seq",
        "kind",
        "group",
        "cores",
        "memory",
        "time_limit",
        "required_seconds",
        "label",
        "argv",
        "env",
        "success_target",
    }
)
_STOP_KEYS: Final = frozenset({"schema", "schema_version", "seq", "kind"})
_EXIT_CODES: Final = {
    "stop_ticket": 0,
    "cancelled": 0,
    "idle_timeout": 0,
    "time_exhausted": 0,
    "terminated": 143,
    "error": 1,
    "invalid_ticket": 3,
}
_UNKNOWN_TIME_POLL_LIMIT: Final = 3
_SIGNAL_SLICE_SECONDS: Final = 1.0
_KILL_WAIT_SECONDS: Final = 30.0
_SQUEUE_TIMEOUT_SECONDS: Final = 30
_STEP_ID_PATTERN: Final = re.compile(r"[0-9]+\.[0-9]+")
_FOREIGN_JOB_FIELDS: Final = 3
_NO_STEP_FIELDS: Final[tuple[object, ...]] = (0, 0, "", "", 0, "", (), {}, None)


class StepProcess(Protocol):
    """The part of a running srun client the worker controls."""

    def poll(self) -> int | None:
        """Return the exit status, or None while the step runs."""
        ...

    def send_signal(self, sig: int, /) -> None:
        """Deliver one signal to the srun client."""
        ...


def _require_int(value: object, *, label: str, minimum: int) -> int:
    """Return one non-boolean integer of at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        message = f"{label} must be an integer >= {minimum}, got {value!r}."
        raise ValueError(message)
    return value


def _require_text(value: object, pattern: re.Pattern[str], *, label: str) -> str:
    """Return one string that matches ``pattern`` completely."""
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        message = f"{label} must match {pattern.pattern!r}, got {value!r}."
        raise ValueError(message)
    return value


def require_slurm_memory(value: object, *, label: str) -> str:
    """Return one Slurm memory size of the form ``<int>M|G|T``."""
    return _require_text(value, _MEMORY, label=label)


def require_slurm_time(value: object, *, label: str) -> str:
    """Return one positive Slurm time limit of the form ``HH:MM:SS``."""
    text = _require_text(value, _TIME_LIMIT, label=label)
    if text == "00:00:00":
        message = f"{label} must be positive."
        raise ValueError(message)
    return text


def _validate_launcher(argv: tuple[str, ...]) -> None:
    """Require argv[0] to be the absolute path of a regular generation_node.sh file."""
    if not argv or any(not isinstance(item, str) or "\0" in item for item in argv):
        message = "Ticket argv must be a non-empty list of strings without NUL."
        raise ValueError(message)
    launcher = Path(argv[0])
    if not launcher.is_absolute() or launcher.name != LAUNCHER_NAME or launcher.is_symlink() or not launcher.is_file():
        message = f"Ticket argv[0] must be the absolute path of a regular, non-symlink {LAUNCHER_NAME} file, got {argv[0]!r}."
        raise ValueError(message)


def _validate_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return the sorted copy of a ticket environment after checking every key and value."""
    if not isinstance(environment, Mapping):
        message = "Ticket env must be a mapping."
        raise ValueError(message)  # noqa: TRY004 -- one contract error type for every ticket field
    checked: dict[str, str] = {}
    for key, value in sorted(environment.items()):
        _require_text(key, _ENV_KEY, label="Ticket env key")
        # srun --export splits on commas, so a comma in a value would silently create a second binding.
        if not isinstance(value, str) or any(character in value for character in ("\0", "\n", "\r", ",")):
            message = f"Ticket env value of {key} must be a string without NUL, newline, or comma."
            raise ValueError(message)
        checked[key] = value
    return checked


@dataclass(frozen=True, slots=True)
class NodeTicket:
    """One immutable unit of work: a Slurm job step to run, or the stop marker that ends the worker."""

    seq: int
    kind: str
    group: int = 0
    cores: int = 0
    memory: str = ""
    time_limit: str = ""
    required_seconds: int = 0
    label: str = ""
    argv: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    # Successful steps (exit 0) after which the remaining tickets of this group are skipped; None runs every ticket.
    success_target: int | None = None

    def __post_init__(self) -> None:
        """Validate every field fail-closed and freeze the collections."""
        seq = _require_int(self.seq, label="Ticket seq", minimum=1)
        if seq > MAX_TICKET_SEQ:
            message = f"Ticket seq must be at most {MAX_TICKET_SEQ}, got {seq}."
            raise ValueError(message)
        if self.kind == STOP_KIND:
            if (
                self.group,
                self.cores,
                self.memory,
                self.time_limit,
                self.required_seconds,
                self.label,
                tuple(self.argv),
                dict(self.env),
                self.success_target,
            ) != _NO_STEP_FIELDS:
                message = "A stop ticket carries no step fields."
                raise ValueError(message)
            return
        if self.kind != STEP_KIND:
            message = f"Ticket kind must be {STEP_KIND!r} or {STOP_KIND!r}, got {self.kind!r}."
            raise ValueError(message)
        _require_int(self.group, label="Ticket group", minimum=0)
        _require_int(self.cores, label="Ticket cores", minimum=1)
        _require_int(self.required_seconds, label="Ticket required_seconds", minimum=0)
        if self.success_target is not None:
            _require_int(self.success_target, label="Ticket success_target", minimum=1)
        require_slurm_memory(self.memory, label="Ticket memory")
        require_slurm_time(self.time_limit, label="Ticket time_limit")
        _require_text(self.label, _LABEL, label="Ticket label")
        if len(self.label) > MAX_LABEL_LENGTH:
            message = f"Ticket label must contain at most {MAX_LABEL_LENGTH} characters."
            raise ValueError(message)
        argv = tuple(self.argv)
        _validate_launcher(argv)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "env", _validate_environment(self.env))

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON object a dispatcher publishes for this ticket."""
        payload: dict[str, Any] = {"schema": TICKET_SCHEMA, "schema_version": SCHEMA_VERSION, "seq": self.seq, "kind": self.kind}
        if self.kind == STEP_KIND:
            payload.update(
                group=self.group,
                cores=self.cores,
                memory=self.memory,
                time_limit=self.time_limit,
                required_seconds=self.required_seconds,
                label=self.label,
                argv=list(self.argv),
                env=dict(self.env),
                success_target=self.success_target,
            )
        return payload

    @classmethod
    def from_payload(cls, payload: object, *, expected_seq: int | None = None) -> NodeTicket:
        """Parse one ticket document; unknown, missing, or malformed fields are errors."""
        if not isinstance(payload, Mapping):
            message = "A node ticket must be a JSON object."
            raise ValueError(message)  # noqa: TRY004 -- one contract error type for every ticket field
        version = payload.get("schema_version")
        if payload.get("schema") != TICKET_SCHEMA or type(version) is not int or version != SCHEMA_VERSION:
            message = f"A node ticket must declare schema {TICKET_SCHEMA!r} version {SCHEMA_VERSION}."
            raise ValueError(message)
        allowed = _STOP_KEYS if payload.get("kind") == STOP_KIND else _STEP_KEYS
        if set(payload) != allowed:
            message = (
                f"Node ticket fields differ from the contract: missing {sorted(allowed - set(payload))}, unknown {sorted(set(payload) - allowed)}."
            )
            raise ValueError(message)
        argv = payload.get("argv", [])
        if not isinstance(argv, list):
            message = "Ticket argv must be a list."
            raise ValueError(message)  # noqa: TRY004 -- one contract error type for every ticket field
        ticket = cls(
            seq=payload["seq"],
            kind=payload["kind"],
            group=payload.get("group", 0),
            cores=payload.get("cores", 0),
            memory=payload.get("memory", ""),
            time_limit=payload.get("time_limit", ""),
            required_seconds=payload.get("required_seconds", 0),
            label=payload.get("label", ""),
            argv=tuple(argv),
            env=payload.get("env", {}),
            success_target=payload.get("success_target"),
        )
        if expected_seq is not None and ticket.seq != expected_seq:
            message = f"Ticket seq {ticket.seq} differs from its file name sequence {expected_seq}."
            raise ValueError(message)
        return ticket


@dataclass(frozen=True, slots=True)
class NodeWorkerResult:
    """Terminal state of one node worker run."""

    reason: str
    detail: str
    started: int
    finished: int

    @property
    def exit_code(self) -> int:
        """Return the process exit status: zero only for orderly endings."""
        return _EXIT_CODES[self.reason]


def ticket_path(node_directory: Path | str, seq: int) -> Path:
    """Return the file of one ticket inside a node directory."""
    return Path(node_directory) / TICKETS_DIRECTORY / f"{_require_int(seq, label='seq', minimum=1):06d}.json"


def receipt_path(node_directory: Path | str, seq: int, state: ReceiptState) -> Path:
    """Return the started or finished receipt file of one ticket."""
    if state not in ("started", "finished"):
        message = f"Receipt state must be 'started' or 'finished', got {state!r}."
        raise ValueError(message)
    return Path(node_directory) / RECEIPTS_DIRECTORY / f"{_require_int(seq, label='seq', minimum=1):06d}.{state}.json"


def _utc_now() -> str:
    """Return the current UTC time in ISO form."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _publish_once(destination: Path, payload: Mapping[str, Any]) -> None:
    """Publish a JSON object atomically and refuse to replace an existing file."""
    staging = destination.with_name(f".{destination.name}.once-{os.getpid()}")
    common.serialization.atomic_write_json(staging, payload)
    try:
        os.link(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def write_node_ticket(node_directory: Path | str, ticket: NodeTicket) -> Path:
    """Publish one ticket for the worker; an existing ticket of that sequence is never replaced."""
    destination = ticket_path(node_directory, ticket.seq)
    _publish_once(destination, ticket.to_payload())
    return destination


def ensure_node_ticket(node_directory: Path | str, ticket: NodeTicket) -> Path:
    """
    Publish one ticket exactly once, accepting a replay of the identical ticket.

    A dispatcher that crashed after publishing a ticket repeats this call on recovery: the ticket file that is already
    there is the result when its JSON equals the payload of ``ticket``, so no ticket is ever replaced or duplicated.

    Parameters
    ----------
    node_directory : Path | str
        Node directory whose ``tickets/`` directory exists.
    ticket : NodeTicket
        The ticket to publish.

    Returns
    -------
    Path
        The ticket file of ``ticket.seq``.

    Raises
    ------
    FileExistsError
        If that sequence already holds a ticket whose content differs from ``ticket`` or cannot be read as JSON.

    """
    destination = ticket_path(node_directory, ticket.seq)
    payload = ticket.to_payload()
    try:
        _publish_once(destination, payload)
    except FileExistsError as error:
        try:
            existing = _read_json_object(destination)
        except (ValueError, TypeError) as defect:
            message = f"Node ticket {destination} exists and is not a JSON object."
            raise FileExistsError(message) from defect
        if existing != payload:
            message = f"Node ticket {destination} exists with different content."
            raise FileExistsError(message) from error
    return destination


def publish_node_cancel(node_directory: Path | str) -> None:
    """
    Create the ``control/cancel`` marker of a node directory; an existing marker is accepted unchanged.

    Raises
    ------
    FileExistsError
        If ``control/cancel`` exists but is not a regular file.

    """
    control = Path(node_directory) / CONTROL_DIRECTORY
    control.mkdir(parents=True, exist_ok=True)
    marker = control / CANCEL_MARKER
    try:
        with marker.open("x", encoding="utf-8"):
            pass
    except FileExistsError:
        if not marker.is_file():
            raise


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """Return one JSON object, None while the file does not exist, and raise for any other defect."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        message = f"Node file is not valid JSON: {path}"
        raise ValueError(message) from error
    if not isinstance(payload, dict):
        message = f"Node file must hold a JSON object: {path}"
        raise TypeError(message)
    return payload


def _read_record(path: Path, *, schema: str, seq: int | None = None) -> dict[str, Any] | None:
    """Return one validated worker-written record, or None while it is absent."""
    payload = _read_json_object(path)
    if payload is None:
        return None
    version = payload.get("schema_version")
    if payload.get("schema") != schema or type(version) is not int or version != SCHEMA_VERSION or (seq is not None and payload.get("seq") != seq):
        message = f"Node record {path} does not match schema {schema!r} version {SCHEMA_VERSION}" + ("." if seq is None else f" and seq {seq}.")
        raise ValueError(message)
    return payload


def read_node_receipt(node_directory: Path | str, seq: int, state: ReceiptState) -> dict[str, Any] | None:
    """Return one receipt of the worker, or None when it has not been written yet."""
    schema = STARTED_RECEIPT_SCHEMA if state == "started" else FINISHED_RECEIPT_SCHEMA
    return _read_record(receipt_path(node_directory, seq, state), schema=schema, seq=seq)


def read_node_record(node_directory: Path | str, which: NodeRecord) -> dict[str, Any] | None:
    """Return node.json (``node``) or node_exit.json (``exit``), or None when it has not been written yet."""
    if which == "node":
        return _read_record(Path(node_directory) / NODE_RECORD_FILE, schema=NODE_RECORD_SCHEMA)
    if which == "exit":
        return _read_record(Path(node_directory) / NODE_EXIT_FILE, schema=NODE_EXIT_SCHEMA)
    message = f"Node record must be 'node' or 'exit', got {which!r}."
    raise ValueError(message)


def build_srun_step_command(ticket: NodeTicket) -> list[str]:
    """
    Build the srun argument vector that runs one step ticket exactly inside the node allocation.

    The ticket label is the unchanged step name (``--job-name``), by which sacct identifies the step; ``--immediate``
    bounds the wait for step resources, so a step that cannot start exits non-zero without ever running its launcher.
    """
    if ticket.kind != STEP_KIND:
        message = "Only a step ticket has a srun command."
        raise ValueError(message)
    exports = ",".join(["ALL", *(f"{key}={value}" for key, value in sorted(ticket.env.items()))])
    return [
        "srun",
        "--exact",
        f"--immediate={STEP_START_WAIT_SECONDS}",
        "--ntasks=1",
        "--nodes=1",
        f"--cpus-per-task={ticket.cores}",
        f"--mem={ticket.memory}",
        f"--time={ticket.time_limit}",
        f"--job-name={ticket.label}",
        f"--export={exports}",
        *ticket.argv,
    ]


def parse_remaining_seconds(text: str) -> float | None:
    """Parse a squeue ``%L`` value (``[D-]HH:MM:SS``, ``MM:SS``, ``UNLIMITED``); None when it is not a time."""
    value = text.strip()
    if value == "UNLIMITED":
        return math.inf
    match = _REMAINING.fullmatch(value)
    if match is None or (match["days"] is not None and match["hours"] is None):
        return None
    return float(
        int(match["days"] or 0) * 86400 + int(match["hours"] or 0) * 3600 + int(match["minutes"]) * 60 + int(match["seconds"]),
    )


def _run_squeue(arguments: list[str]) -> str | None:
    """Return squeue's stdout, or None when the queue cannot be read."""
    try:
        result = subprocess.run(  # noqa: S603 -- fixed executable name and controlled argument vector; no shell interpolation
            ["squeue", *arguments],  # noqa: S607 -- the node's Slurm client is resolved through PATH like every other launcher command
            capture_output=True,
            text=True,
            check=False,
            timeout=_SQUEUE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _squeue_remaining_seconds(job_id: str) -> float | None:
    """Return the remaining wall time of this node job, or None when it cannot be determined."""
    output = _run_squeue(["-h", "-j", job_id, "-o", "%L"])
    lines = [] if output is None else [line.strip() for line in output.splitlines() if line.strip()]
    return parse_remaining_seconds(lines[0]) if len(lines) == 1 else None


def _foreign_jobs(host: str, job_id: str) -> list[dict[str, Any]] | None:
    """Return the other jobs on this node, or None when the queue cannot be read (unknown is not empty)."""
    output = _run_squeue(["-h", "-w", host, "-o", "%u|%i|%C"])
    if output is None:
        return None
    jobs: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("|")
        if len(parts) != _FOREIGN_JOB_FIELDS or not parts[2].isdigit():
            return None
        if parts[1] != job_id:
            jobs.append({"user": parts[0], "job_id": parts[1], "cpus": int(parts[2])})
    return jobs


def read_cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | None:
    """Return the first ``model name`` of /proc/cpuinfo, or None when it is not available."""
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip() or None
    except OSError:
        return None
    return None


def _node_name() -> str:
    """Return the Slurm node name of this host."""
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname()


def _launch_srun_step(ticket: NodeTicket) -> StepProcess:
    """Start one step as an srun client whose output joins the node job log."""
    return subprocess.Popen(build_srun_step_command(ticket), stdin=subprocess.DEVNULL)  # noqa: S603 -- executable name and vector are contract-validated


def _scancel_steps_term(job_id: str, labels: Sequence[str]) -> None:
    """Send TERM through Slurm to the running steps of this node job whose names are ``labels``.

    Steps that are not (yet or any more) visible are skipped; the caller's KILL escalation covers them.
    """
    output = _run_squeue(["-h", "--steps", "-j", job_id, "-o", "%i|%j"])
    if output is None:
        return
    wanted = set(labels)
    for line in output.splitlines():
        step_id, _, name = line.strip().partition("|")
        if name in wanted and _STEP_ID_PATTERN.fullmatch(step_id) is not None and step_id.split(".", maxsplit=1)[0] == job_id:
            with suppress(OSError, subprocess.SubprocessError):
                subprocess.run(  # noqa: S603 -- fixed scancel vector with a validated step id
                    ["scancel", "--signal=TERM", step_id],  # noqa: S607 -- site PATH owns Slurm
                    check=False,
                    timeout=_SQUEUE_TIMEOUT_SECONDS,
                )


@dataclass(slots=True)
class _RunningStep:
    """One launched step and the facts its finished receipt needs."""

    ticket: NodeTicket
    process: StepProcess
    started_at: str


class _InvalidTicketError(Exception):
    """A visible ticket violates the contract; the worker stops with reason invalid_ticket."""


class _NodeWorker:
    """Single-threaded poll loop that owns running steps, receipts, and the exit record of one node job."""

    def __init__(
        self,
        node_directory: Path,
        *,
        cores: int,
        job_id: str,
        idle_timeout_seconds: float,
        poll_seconds: float,
        cancel_grace_seconds: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        remaining_seconds: Callable[[], float | None],
        launch_step: Callable[[NodeTicket], StepProcess],
        signal_steps: Callable[[str, Sequence[str]], None],
    ) -> None:
        self.node_directory = node_directory
        self.signal_steps = signal_steps
        self.cores = cores
        self.job_id = job_id
        self.idle_timeout_seconds = idle_timeout_seconds
        self.poll_seconds = poll_seconds
        self.cancel_grace_seconds = cancel_grace_seconds
        self.clock = clock
        self.sleep = sleep
        self.remaining_seconds = remaining_seconds
        self.launch_step = launch_step
        self.host = _node_name()
        self.running: list[_RunningStep] = []
        self.tickets: dict[int, NodeTicket] = {}
        self.next_seq = 1
        self.last_group: int | None = None
        self.started_count = 0
        self.finished_count = 0
        self.skipped_count = 0
        self.group_targets: dict[int, int | None] = {}
        self.group_successes: dict[int, int] = {}
        self.reason: str | None = None
        self.detail = ""
        self.cancel_requested = False
        self.term_sent_at: float | None = None
        self.signal_number: int | None = None
        self.idle_since: float | None = None
        self.unknown_time_polls = 0

    # -- records --

    def _write_node_record(self) -> None:
        """Publish node.json once; a node directory serves exactly one worker."""
        destination = self.node_directory / NODE_RECORD_FILE
        if destination.exists() or (self.node_directory / NODE_EXIT_FILE).exists():
            message = f"Node directory already holds a worker record and is not reused: {self.node_directory}"
            raise FileExistsError(message)
        (self.node_directory / RECEIPTS_DIRECTORY).mkdir(exist_ok=True)
        _publish_once(
            destination,
            {
                "schema": NODE_RECORD_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "job_id": self.job_id,
                "host": self.host,
                "cpu_model": read_cpu_model(),
                "cores": self.cores,
                "started_at": _utc_now(),
                "foreign_jobs_at_start": _foreign_jobs(self.host, self.job_id),
                "seat_name": os.environ.get("SLURM_JOB_NAME"),
            },
        )

    def _write_exit_record(self) -> None:
        """Publish node_exit.json best effort; a failure never masks the original outcome."""
        try:
            _publish_once(
                self.node_directory / NODE_EXIT_FILE,
                {
                    "schema": NODE_EXIT_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "job_id": self.job_id,
                    "ended_at": _utc_now(),
                    "reason": self.reason,
                    "detail": self.detail,
                    "started": self.started_count,
                    "finished": self.finished_count,
                    "skipped": self.skipped_count,
                    "foreign_jobs_at_end": _foreign_jobs(self.host, self.job_id),
                },
            )
        except Exception:  # noqa: BLE001 -- the exit record is best effort evidence
            traceback.print_exc()

    def _record_finished(
        self,
        ticket: NodeTicket,
        started_at: str | None,
        exit_code: int | None,
        *,
        outcome: str = "exited",
        strict: bool = True,
    ) -> None:
        """Publish the finished receipt of one step; a skipped ticket has neither a start nor an exit code."""
        if outcome == "skipped":
            self.skipped_count += 1
        else:
            self.finished_count += 1
        try:
            _publish_once(
                receipt_path(self.node_directory, ticket.seq, "finished"),
                {
                    "schema": FINISHED_RECEIPT_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "seq": ticket.seq,
                    "started_at": started_at,
                    "ended_at": _utc_now(),
                    "exit_code": exit_code,
                    "host": self.host,
                    "outcome": outcome,
                },
            )
        except Exception:
            if strict:
                raise
            traceback.print_exc()

    # -- signals, cancellation, termination --

    def _on_signal(self, signum: int, _frame: FrameType | None) -> None:
        """Remember the first termination signal; the poll loop acts on it."""
        if self.signal_number is None:
            self.signal_number = signum

    def _install_signal_handlers(self) -> dict[int, Any]:
        """Install the TERM and INT handlers when running in the main thread and return the previous ones."""
        if threading.current_thread() is not threading.main_thread():
            return {}
        return {number: signal.signal(number, self._on_signal) for number in (signal.SIGTERM, signal.SIGINT)}

    def _request_cancel(self, reason: str, detail: str) -> None:
        """Stop launching; the first reason of the run is kept."""
        self._set_reason(reason, detail)
        self.cancel_requested = True

    def _set_reason(self, reason: str, detail: str) -> None:
        """Record why the worker ends; only the first reason counts."""
        if self.reason is None:
            self.reason = reason
            self.detail = detail

    def _check_cancel(self) -> None:
        """Turn a termination signal or the control/cancel marker into a cancel request."""
        if self.signal_number is not None:
            self._request_cancel("terminated", f"received {signal.Signals(self.signal_number).name}")
        elif (self.node_directory / CONTROL_DIRECTORY / CANCEL_MARKER).exists():
            self._request_cancel("cancelled", "control/cancel is present")

    @staticmethod
    def _signal_step(step: _RunningStep, number: int) -> None:
        """Deliver a signal to one srun client; a client that already ended is not an error."""
        with suppress(OSError):
            step.process.send_signal(number)

    def _advance_cancel(self) -> None:
        """Send TERM once to every running step through Slurm, then SIGKILL the srun clients still alive after the grace period.

        TERM goes to the step tasks by step name (``scancel --signal=TERM <job>.<step>``), whose launcher traps it and
        stops the case in an orderly way. The srun clients themselves never receive TERM: on ICE (probe P1, 2026-10-27)
        an srun client that gets TERM aborts its step and the task is killed without seeing TERM.
        """
        if self.term_sent_at is None:
            self.term_sent_at = self.clock()
            if self.running:
                try:
                    self.signal_steps(self.job_id, [step.ticket.label for step in self.running])
                except Exception:  # noqa: BLE001 -- the KILL escalation after the grace period still ends the steps
                    traceback.print_exc()
        elif self.running and self.clock() - self.term_sent_at >= self.cancel_grace_seconds:
            self._kill_running()

    def _kill_running(self) -> None:
        """SIGKILL every remaining srun client and record what cannot be reaped in time."""
        for step in self.running:
            self._signal_step(step, signal.SIGKILL)
        deadline = self.clock() + _KILL_WAIT_SECONDS
        self._reap()
        while self.running and self.clock() < deadline:
            self.sleep(min(self.poll_seconds, _SIGNAL_SLICE_SECONDS))
            self._reap()
        for step in list(self.running):
            self.running.remove(step)
            self._record_finished(step.ticket, step.started_at, -int(signal.SIGKILL))

    def _abort_running(self) -> None:
        """After an internal error: terminate every step without trusting further receipt writes."""
        self.cancel_requested = True
        while self.running:
            self._reap(strict=False)
            if not self.running:
                break
            try:
                self._advance_cancel()
            except Exception:  # noqa: BLE001 -- keep terminating whatever still runs
                traceback.print_exc()
                break
            self.sleep(min(self.poll_seconds, _SIGNAL_SLICE_SECONDS))

    # -- tickets --

    def _scan_tickets(self) -> None:
        """Load and validate every newly visible ticket; any contract violation is an invalid ticket."""
        directory = self.node_directory / TICKETS_DIRECTORY
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries if not entry.name.startswith("."))
        for name in names:
            match = _TICKET_FILE.fullmatch(name)
            if match is None:
                message = f"Unexpected entry in the ticket directory: {name!r}"
                raise _InvalidTicketError(message)
            seq = int(match["seq"])
            if seq in self.tickets or seq < self.next_seq:
                continue
            try:
                payload = _read_json_object(directory / name)
                if payload is None:
                    continue  # renamed away between listing and reading; look again on the next poll
                ticket = NodeTicket.from_payload(payload, expected_seq=seq)
            except (TypeError, ValueError) as error:
                message = f"Ticket {name} is invalid: {error}"
                raise _InvalidTicketError(message) from error
            if ticket.kind == STEP_KIND and ticket.cores > self.cores:
                message = f"Ticket {name} needs {ticket.cores} cores but the node job has {self.cores}."
                raise _InvalidTicketError(message)
            if ticket.kind == STEP_KIND:
                if ticket.group in self.group_targets and self.group_targets[ticket.group] != ticket.success_target:
                    message = f"Ticket {name} changes the success target of group {ticket.group}."
                    raise _InvalidTicketError(message)
                self.group_targets[ticket.group] = ticket.success_target
            self.tickets[seq] = ticket
        stops = [seq for seq, ticket in self.tickets.items() if ticket.kind == STOP_KIND]
        if stops and any(seq > min(stops) for seq in self.tickets):
            message = f"A ticket follows the stop ticket {min(stops):06d}."
            raise _InvalidTicketError(message)

    # -- scheduling --

    def _reap(self, *, strict: bool = True) -> None:
        """Publish a finished receipt for every step whose srun client has ended."""
        for step in list(self.running):
            code = step.process.poll()
            if code is not None:
                self.running.remove(step)
                if code == 0:
                    self.group_successes[step.ticket.group] = self.group_successes.get(step.ticket.group, 0) + 1
                self._record_finished(step.ticket, step.started_at, code, strict=strict)

    def _note_idle(self) -> None:
        """End the worker after idle_timeout_seconds without any step and without any unstarted ticket."""
        now = self.clock()
        if self.idle_since is None:
            self.idle_since = now
        elif now - self.idle_since >= self.idle_timeout_seconds:
            self._set_reason("idle_timeout", f"no running step and no unstarted ticket for {self.idle_timeout_seconds:g} s")

    def _time_gate(self, ticket: NodeTicket) -> bool:
        """Return whether the node job has enough known time left to start ``ticket``."""
        remaining = self.remaining_seconds()
        if remaining is not None and remaining >= ticket.required_seconds:
            self.unknown_time_polls = 0
            return True
        if self.running:
            return False  # a running step may still end; decide on a free node
        if remaining is None:
            self.unknown_time_polls += 1
            if self.unknown_time_polls >= _UNKNOWN_TIME_POLL_LIMIT:
                self._set_reason("time_exhausted", f"remaining node time stayed unknown for {self.unknown_time_polls} polls")
        else:
            self._set_reason(
                "time_exhausted", f"remaining node time {remaining:.0f} s is below the {ticket.required_seconds} s ticket {ticket.seq} requires"
            )
        return False

    def _start_next(self) -> bool:
        """Start the next ticket in sequence when every gate allows it; return whether a step was started."""
        ticket = self.tickets.get(self.next_seq)
        if ticket is None:
            if not self.running:
                self._note_idle()
            return False
        self.idle_since = None
        if ticket.kind == STOP_KIND:
            if not self.running:
                self._set_reason("stop_ticket", f"stop ticket {ticket.seq:06d}")
            return False
        if self.last_group is not None and ticket.group < self.last_group:
            message = f"Ticket {ticket.seq:06d} lowers the group from {self.last_group} to {ticket.group}."
            raise _InvalidTicketError(message)
        if ticket.success_target is not None and self.group_successes.get(ticket.group, 0) >= ticket.success_target:
            self._skip(ticket)
            return True
        if any(step.ticket.group != ticket.group for step in self.running):
            return False
        if ticket.success_target is not None and self.group_successes.get(ticket.group, 0) + len(self.running) >= ticket.success_target:
            return False  # enough candidates run to reach the target; a failing one frees its place for the next

        if sum(step.ticket.cores for step in self.running) + ticket.cores > self.cores or not self._time_gate(ticket):
            return False
        self._launch(ticket)
        return True

    def _skip(self, ticket: NodeTicket) -> None:
        """Mark a ticket of a group that already reached its success target as skipped; it never starts."""
        self._record_finished(ticket, None, None, outcome="skipped")
        self.next_seq = ticket.seq + 1
        self.last_group = ticket.group

    def _launch(self, ticket: NodeTicket) -> None:
        """Publish the started receipt, then start the step; a launch failure becomes a finished receipt."""
        started_at = _utc_now()
        _publish_once(
            receipt_path(self.node_directory, ticket.seq, "started"),
            {
                "schema": STARTED_RECEIPT_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "seq": ticket.seq,
                "started_at": started_at,
                "host": self.host,
                "worker_pid": os.getpid(),
            },
        )
        self.next_seq = ticket.seq + 1
        self.last_group = ticket.group
        self.started_count += 1
        try:
            process = self.launch_step(ticket)
        except OSError as error:
            print(f"Could not launch ticket {ticket.seq:06d}: {error}", file=sys.stderr)
            self._record_finished(ticket, started_at, -1, outcome="launch_failed")
            return
        self.running.append(_RunningStep(ticket, process, started_at))

    def _schedule(self) -> None:
        """Start tickets until a gate stops the head of the sequence."""
        try:
            self._scan_tickets()
            while self.reason is None and self._start_next():
                pass
        except _InvalidTicketError as error:
            self._set_reason("invalid_ticket", str(error))

    def _wait_for_next_poll(self) -> None:
        """Sleep until the next poll in short slices, so a termination signal is acted on within a second."""
        deadline = self.clock() + self.poll_seconds
        while self.clock() < deadline:
            if self.signal_number is not None and not self.cancel_requested:
                return  # act on the new signal now; an already handled one must not turn the wait into a busy loop
            self.sleep(min(deadline - self.clock(), _SIGNAL_SLICE_SECONDS))

    def _loop(self) -> None:
        """Run until the worker has a reason to end and no step is left."""
        while True:
            self._reap()
            self._check_cancel()
            if self.cancel_requested:
                self._advance_cancel()
            elif self.reason is None:
                self._schedule()
            if self.reason is not None and not self.running:
                return
            self._wait_for_next_poll()

    def run(self) -> NodeWorkerResult:
        """Run the worker to its end and return the reason; every path leaves node_exit.json."""
        previous_handlers = self._install_signal_handlers()
        try:
            try:
                self._write_node_record()
                self._loop()
            except Exception as error:  # noqa: BLE001 -- every failure must stop the steps and leave an exit record
                traceback.print_exc()
                self._set_reason("error", f"{type(error).__name__}: {error}")
                self._abort_running()
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, signal.SIG_DFL if handler is None else handler)
            self._write_exit_record()
        return NodeWorkerResult(self.reason or "error", self.detail, self.started_count, self.finished_count)


def run_node_worker(
    node_directory: Path | str,
    *,
    cores: int,
    idle_timeout_seconds: float = 900.0,
    poll_seconds: float = 5.0,
    cancel_grace_seconds: float = 600.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    remaining_seconds: Callable[[], float | None] | None = None,
    launch_step: Callable[[NodeTicket], StepProcess] = _launch_srun_step,
    signal_steps: Callable[[str, Sequence[str]], None] | None = None,
) -> NodeWorkerResult:
    """
    Run the generic node worker of the current Slurm node job until it has nothing left to do.

    Parameters
    ----------
    node_directory : Path | str
        Absolute node directory prepared by the dispatcher; it must contain ``tickets/``.
    cores : int
        Cores of the node job; the cores of all running steps never exceed this.
    idle_timeout_seconds : float, optional
        Exit with ``idle_timeout`` after this long without a running step and without an unstarted ticket.
    poll_seconds : float, optional
        Interval between ticket scans, receipt checks, and cancel checks.
    cancel_grace_seconds : float, optional
        Time between forwarding TERM to the running steps and killing their srun clients.
    clock, sleep : Callable, optional
        Monotonic clock and sleep function; replaced by tests.
    remaining_seconds : Callable[[], float | None], optional
        Remaining node time probe; defaults to ``squeue -h -j $SLURM_JOB_ID -o %L``.
    launch_step : Callable[[NodeTicket], StepProcess], optional
        Starts one step; defaults to ``subprocess.Popen`` of the srun command.
    signal_steps : Callable[[str, Sequence[str]], None], optional
        Sends TERM to the running steps of the node job by step name; defaults to ``squeue --steps`` plus
        ``scancel --signal=TERM <job>.<step>``.

    Returns
    -------
    NodeWorkerResult
        Why the worker ended; ``exit_code`` is the process status to return.

    Raises
    ------
    ValueError
        If the arguments, the Slurm environment, or the node directory layout are invalid; nothing is written then.

    Notes
    -----
    Cancel and TERM (also INT) stop launching and send TERM once to every running step through Slurm by step name;
    after ``cancel_grace_seconds`` the srun clients still alive are killed. The clients never receive TERM, because an
    srun client that gets TERM aborts its step with KILL (probe P1). Slurm's ``--signal=B:TERM@120`` reaches the batch
    shell only, whose launcher forwards it to this process, so a time-limit stop is a termination.

    """
    directory = Path(node_directory)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not directory.is_absolute() or not (directory / TICKETS_DIRECTORY).is_dir():
        message = f"The node directory must be absolute and contain {TICKETS_DIRECTORY}/: {directory}"
        raise ValueError(message)
    if not job_id.isdigit():
        message = "The node worker requires SLURM_JOB_ID of its node job."
        raise ValueError(message)
    _require_int(cores, label="cores", minimum=1)
    if idle_timeout_seconds <= 0 or poll_seconds <= 0 or cancel_grace_seconds < 0:
        message = "idle_timeout_seconds and poll_seconds must be positive and cancel_grace_seconds non-negative."
        raise ValueError(message)
    worker = _NodeWorker(
        directory,
        cores=cores,
        job_id=job_id,
        idle_timeout_seconds=idle_timeout_seconds,
        poll_seconds=poll_seconds,
        cancel_grace_seconds=cancel_grace_seconds,
        clock=clock,
        sleep=sleep,
        remaining_seconds=remaining_seconds or (lambda: _squeue_remaining_seconds(job_id)),
        launch_step=launch_step,
        signal_steps=signal_steps or _scancel_steps_term,
    )
    return worker.run()
