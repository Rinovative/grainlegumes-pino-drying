"""
generation_runtime_progress.py

Persist non-authoritative incremental COMSOL runtime progress for campaign cases.
Responsibilities:
  - Parse supported English COMSOL 6.4 stdout progress incrementally
  - Bind progress records to exact campaign submission identities
  - Atomically persist bounded operational progress receipts
Design principles:
  - Observability never controls solver, publication, or admission behavior
  - Persisted identities fail closed against unsafe or conflicting records
  - Parser evidence reports only values explicitly emitted by COMSOL
This module does NOT:
  - Classify solver convergence, health, or failure
  - Modify campaign manifests, case data, or publication state
"""

from __future__ import annotations

import json
import math
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping

PROGRESS_POLL_INTERVAL_SECONDS: Final = 20.0
PROGRESS_HEARTBEAT_SECONDS: Final = 300.0
PROGRESS_STALE_AFTER_SECONDS: Final = 600.0
_MAX_COMSOL_PROGRESS_PERCENT: Final = 100
_PROGRESS_SCHEMA_KIND: Final = "generation_campaign_runtime_progress"
_PROGRESS_SCHEMA_VERSION: Final = 1
_TERMINAL_PHASES: Final = frozenset({"completed", "failed"})
_PHASES: Final = frozenset(
    {
        "preparing",
        "starting_solver",
        "stationary_airflow",
        "transient_drying",
        "collecting_exports",
        "canonicalizing",
        "publishing",
        "completed",
        "failed",
    }
)
_OPTIONAL_KEYS: Final = (
    "parser_state",
    "comsol_section",
    "comsol_progress_percent",
    "step_index",
    "simulated_time_seconds",
    "step_size_seconds",
    "residual_evaluations",
    "jacobian_evaluations",
    "linear_solves",
    "order",
    "time_failures",
    "nonlinear_failures",
    "nonlinear_iteration",
    "last_solver_log_update_at",
)
_RECORD_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "campaign_run_id",
        "batch_name",
        "batch_id",
        "case_index",
        "case_id",
        "slurm_job_id",
        "hostname",
        "started_at",
        "updated_at",
        "elapsed_seconds",
        "phase",
        "terminal",
        *_OPTIONAL_KEYS,
    }
)
_TRANSIENT_COLUMNS: Final = (
    "Step",
    "Time",
    "Stepsize",
    "Res",
    "Jac",
    "Sol",
    "Order",
    "Tfail",
    "NLfail",
    "LinErr",
    "LinRes",
)
_STATIONARY_COLUMNS: Final = (
    "Iter",
    "SolEst",
    "ResEst",
    "Damping",
    "Stepsize",
    "#Res",
    "#Jac",
    "#Sol",
    "LinErr",
    "LinRes",
)
_PROGRESS_PATTERN: Final = re.compile(r"Current Progress:\s*([0-9]+)\s*%")
_INTEGER_PATTERN: Final = re.compile(r"[+-]?[0-9]+")


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    """Serialize one UTC timestamp with portable second precision."""
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime | None:
    """Parse one persisted timezone-aware timestamp without raising."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _numeric_job_id(value: str | int) -> str:
    """Normalize one numeric Slurm job identifier."""
    text = str(value)
    if not text or not text.isascii() or not text.isdigit():
        message = "Runtime progress requires a numeric Slurm job ID."
        raise ValueError(message)
    return text


def _path_uses_symbolic_link(path: Path) -> bool:
    """Return whether resolving a path changes any existing path component."""
    return path.resolve(strict=False) != path.absolute()


def _progress_receipt_path(
    run_id: str,
    job_id: str,
    *,
    storage_root: Path | str | None,
    create_directory: bool,
) -> Path:
    """Resolve one symlink-free operational progress receipt path."""
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage_root)
    if run_directory.is_symlink() or not run_directory.is_dir() or _path_uses_symbolic_link(run_directory):
        message = f"Campaign run directory is missing or unsafe: {run_directory}"
        raise FileNotFoundError(message)
    progress_directory = run_directory / campaign_evidence.RUNTIME_PROGRESS_DIRECTORY_NAME
    if (
        progress_directory.is_symlink()
        or (progress_directory.exists() and not progress_directory.is_dir())
        or _path_uses_symbolic_link(progress_directory)
    ):
        message = f"Campaign progress directory is unsafe: {progress_directory}"
        raise ValueError(message)
    if create_directory:
        progress_directory.mkdir(mode=0o700, exist_ok=True)
        if progress_directory.is_symlink() or not progress_directory.is_dir() or _path_uses_symbolic_link(progress_directory):
            message = f"Campaign progress directory is unsafe: {progress_directory}"
            raise ValueError(message)
    path = progress_directory / f"{job_id}.json"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        message = f"Campaign progress receipt is unsafe: {path}"
        raise ValueError(message)
    return path


def _bound_identity(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    batch_name: str,
    batch_id: str,
    case_index: int,
    case_id: str,
    job_id: str,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Validate and return the immutable identity for one job-bound case."""
    batches = manifest.get("batches")
    batch_matches = (
        [record for record in batches if isinstance(record, dict) and record.get("batch_name") == batch_name and record.get("batch_id") == batch_id]
        if isinstance(batches, list)
        else []
    )
    if len(batch_matches) != 1:
        message = "Runtime progress batch identity is not bound to the campaign run."
        raise ValueError(message)
    expected_case = {
        "batch_name": batch_name,
        "batch_id": batch_id,
        "case_index": case_index,
        "case_id": case_id,
    }
    submissions = manifest.get("submissions")
    matches = (
        [record for record in submissions if isinstance(record, dict) and record.get("job_id") == job_id and record.get("case") == expected_case]
        if isinstance(submissions, list)
        else []
    )
    if len(matches) != 1:
        message = "Runtime progress job is not bound to this exact campaign case."
        raise ValueError(message)
    identity: dict[str, Any] = {
        "schema_kind": _PROGRESS_SCHEMA_KIND,
        "schema_version": _PROGRESS_SCHEMA_VERSION,
        "campaign_run_id": run_id,
        **expected_case,
        "slurm_job_id": job_id,
    }
    if hostname is not None:
        identity["hostname"] = hostname
    return identity


def _unavailable(reason: str) -> dict[str, Any]:
    """Return one explicit unavailable operational-progress view."""
    return {
        "availability": "unavailable",
        "reason": reason,
        "age_seconds": None,
        "stale": None,
    }


def _parse_integer(token: str) -> int | None:
    """Return one exact integer token."""
    return int(token) if _INTEGER_PATTERN.fullmatch(token) is not None else None


def _parse_float(token: str) -> float | None:
    """Return one finite floating-point token."""
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


class ComsolProgressParser:
    """Incrementally parse complete supported English COMSOL stdout lines."""

    def __init__(self) -> None:
        """Initialize unavailable evidence with no active solver table."""
        self._state: dict[str, Any] = {"parser_state": "unavailable"}
        self._table: str | None = None

    @property
    def state(self) -> dict[str, Any]:
        """Return a defensive copy of parsed raw evidence."""
        return dict(self._state)

    def mark_log_update(self, updated_at: datetime) -> None:
        """Record when at least one complete solver-log line arrived."""
        self._state["last_solver_log_update_at"] = _timestamp(updated_at)

    def _consume_transient_row(self, tokens: list[str]) -> None:
        """Consume one complete supported time-stepping row when numeric."""
        if len(tokens) != len(_TRANSIENT_COLUMNS):
            return
        integers = [_parse_integer(tokens[index]) for index in (0, 3, 4, 5, 6, 7, 8)]
        floating = [_parse_float(tokens[index]) for index in (1, 2, 9, 10)]
        if any(value is None for value in (*integers, *floating)):
            return
        self._state.update(
            {
                "parser_state": "available",
                "comsol_section": "transient_drying",
                "step_index": integers[0],
                "simulated_time_seconds": floating[0],
                "step_size_seconds": floating[1],
                "residual_evaluations": integers[1],
                "jacobian_evaluations": integers[2],
                "linear_solves": integers[3],
                "order": integers[4],
                "time_failures": integers[5],
                "nonlinear_failures": integers[6],
            }
        )

    def _consume_stationary_row(self, tokens: list[str]) -> None:
        """Consume one complete supported stationary nonlinear row when numeric."""
        if len(tokens) != len(_STATIONARY_COLUMNS):
            return
        iteration = _parse_integer(tokens[0])
        remaining = [_parse_float(token) for token in tokens[1:]]
        if iteration is None or any(value is None for value in remaining):
            return
        self._state.update(
            {
                "parser_state": "available",
                "comsol_section": "stationary_airflow",
                "nonlinear_iteration": iteration,
            }
        )

    def consume(self, lines: list[str]) -> dict[str, Any]:
        """Consume complete lines and return the updated raw evidence."""
        for line in lines:
            stripped = line.strip()
            progress = _PROGRESS_PATTERN.search(stripped)
            if progress is not None:
                self._state["parser_state"] = "available"
                self._state["comsol_progress_percent"] = int(progress.group(1))
            if "Stationary Solver" in stripped:
                self._state.update({"parser_state": "available", "comsol_section": "stationary_airflow"})
            if "Time-Dependent Solver" in stripped and "Transient Drying" in stripped:
                self._state.update({"parser_state": "available", "comsol_section": "transient_drying"})
            tokens = stripped.split()
            if tuple(tokens) == _TRANSIENT_COLUMNS:
                self._table = "transient"
                self._state.update({"parser_state": "available", "comsol_section": "transient_drying"})
                continue
            if tuple(tokens) == _STATIONARY_COLUMNS:
                self._table = "stationary"
                self._state.update({"parser_state": "available", "comsol_section": "stationary_airflow"})
                continue
            if self._table == "transient":
                self._consume_transient_row(tokens)
            elif self._table == "stationary":
                self._consume_stationary_row(tokens)
        return self.state


class RuntimeProgressReporter:
    """Observe one solver stdout file and persist operational evidence."""

    def __init__(
        self,
        identity: Mapping[str, Any],
        path: Path,
        *,
        stdout_path: Path | str | None = None,
        started_at: datetime | None = None,
    ) -> None:
        """Initialize one exact reporter without writing a receipt."""
        self._identity = dict(identity)
        self._path = path
        self._stdout_path = Path(stdout_path) if stdout_path is not None else None
        self._offset = 0
        self._trailing = b""
        self._parser = ComsolProgressParser()
        self._started_at = started_at or _utc_now()
        self._last_write_at: datetime | None = None
        self._last_signature: str | None = None

    @property
    def path(self) -> Path:
        """Return the operational receipt path."""
        return self._path

    def bind_stdout(self, path: Path | str) -> None:
        """Bind the single case-local stdout source observed incrementally."""
        candidate = Path(path)
        if self._stdout_path is not None and self._stdout_path != candidate:
            message = "Runtime progress reporter stdout source is already bound."
            raise ValueError(message)
        self._stdout_path = candidate

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        batch_name: str,
        batch_id: str,
        case_index: int,
        case_id: str,
        slurm_job_id: str | int,
        storage_root: Path | str | None = None,
        stdout_path: Path | str | None = None,
        hostname: str | None = None,
    ) -> RuntimeProgressReporter:
        """Create one strict job-bound runtime progress reporter."""
        return create_runtime_progress_reporter(
            run_id,
            batch_name=batch_name,
            batch_id=batch_id,
            case_index=case_index,
            case_id=case_id,
            slurm_job_id=slurm_job_id,
            storage_root=storage_root,
            stdout_path=stdout_path,
            hostname=hostname,
        )

    def _consume_stdout(self, now: datetime) -> None:
        """Read and parse only newly appended complete stdout lines."""
        if self._stdout_path is None:
            return
        with self._stdout_path.open("rb") as stream:
            stream.seek(self._offset)
            appended = stream.read()
        self._offset += len(appended)
        combined = self._trailing + appended
        complete, separator, trailing = combined.rpartition(b"\n")
        if not separator:
            self._trailing = combined
            return
        self._trailing = trailing
        lines = complete.decode("utf-8", errors="replace").splitlines()
        self._parser.consume(lines)
        if lines:
            self._parser.mark_log_update(now)

    def _payload(self, *, phase: str, terminal: bool, now: datetime) -> dict[str, Any]:
        """Return one complete persisted progress payload."""
        if phase not in _PHASES:
            message = f"Unsupported runtime progress phase: {phase!r}."
            raise ValueError(message)
        if terminal != (phase in _TERMINAL_PHASES):
            message = "Runtime progress terminal state must agree with its phase."
            raise ValueError(message)
        parsed = self._parser.state
        effective_phase = phase
        if phase == "starting_solver" and parsed.get("comsol_section") in {"stationary_airflow", "transient_drying"}:
            effective_phase = str(parsed["comsol_section"])
        payload = {
            **self._identity,
            "started_at": _timestamp(self._started_at),
            "updated_at": _timestamp(now),
            "elapsed_seconds": max(0.0, (now - self._started_at).total_seconds()),
            "phase": effective_phase,
            "terminal": terminal,
        }
        for key in _OPTIONAL_KEYS:
            payload[key] = parsed.get(key)
        return payload

    def _write(self, payload: Mapping[str, Any], *, force: bool, now: datetime) -> bool:
        """Best-effort write meaningful changes or a bounded heartbeat."""
        meaningful = {
            key: value for key, value in payload.items() if key not in {"started_at", "updated_at", "elapsed_seconds", "last_solver_log_update_at"}
        }
        signature = common.serialization.canonical_json_sha256(meaningful)
        if (
            not force
            and signature == self._last_signature
            and self._last_write_at is not None
            and (now - self._last_write_at).total_seconds() < PROGRESS_HEARTBEAT_SECONDS
        ):
            return False
        try:
            common.serialization.atomic_write_json(self._path, payload)
        except Exception:  # noqa: BLE001 -- monitoring persistence cannot terminate a solver
            return False
        self._last_signature = signature
        self._last_write_at = now
        return True

    def update(self, *, phase: str, terminal: bool = False, force: bool = False) -> bool:
        """Consume appended stdout and best-effort persist current progress."""
        now = _utc_now()
        try:
            self._consume_stdout(now)
            payload = self._payload(phase=phase, terminal=terminal, now=now)
        except Exception:  # noqa: BLE001 -- monitoring parsing cannot terminate a solver
            return False
        return self._write(payload, force=force or terminal, now=now)


def create_runtime_progress_reporter(
    run_id: str,
    *,
    batch_name: str,
    batch_id: str,
    case_index: int,
    case_id: str,
    slurm_job_id: str | int,
    storage_root: Path | str | None = None,
    stdout_path: Path | str | None = None,
    hostname: str | None = None,
) -> RuntimeProgressReporter:
    """
    Create a strict job-bound runtime progress reporter.

    Raises
    ------
    FileNotFoundError
        If the campaign-run metadata directory is absent or unsafe.
    ValueError
        If the job/case identity, hostname, or existing receipt conflicts.

    """
    job_id = _numeric_job_id(slurm_job_id)
    manifest = campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    resolved_hostname = hostname or socket.gethostname()
    if not resolved_hostname or any(character in resolved_hostname for character in "\r\n\t"):
        message = "Runtime progress hostname must be non-empty text without control characters."
        raise ValueError(message)
    identity = _bound_identity(
        manifest,
        run_id=run_id,
        batch_name=batch_name,
        batch_id=batch_id,
        case_index=case_index,
        case_id=case_id,
        job_id=job_id,
        hostname=resolved_hostname,
    )
    path = _progress_receipt_path(
        run_id,
        job_id,
        storage_root=storage_root,
        create_directory=True,
    )
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            message = f"Runtime progress receipt is malformed: {path}"
            raise ValueError(message) from error
        if not isinstance(existing, dict) or any(existing.get(key) != value for key, value in identity.items()):
            message = f"Runtime progress receipt identity conflicts: {path}"
            raise ValueError(message)
        if existing.get("terminal") is True:
            message = f"Runtime progress receipt is already terminal: {path}"
            raise FileExistsError(message)
    return RuntimeProgressReporter(identity, path, stdout_path=stdout_path)


def _is_nonnegative_integer(value: object) -> bool:
    """Return whether a value is one nonnegative integer but not a boolean."""
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_nonnegative_number(value: object) -> bool:
    """Return whether a value is one finite nonnegative number."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def _payload_is_supported(payload: object, expected: Mapping[str, Any]) -> bool:
    """Return whether one loaded payload satisfies the progress schema."""
    if not isinstance(payload, dict) or set(payload) != _RECORD_KEYS:
        return False
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    hostname = payload.get("hostname")
    elapsed = payload.get("elapsed_seconds")
    phase = payload.get("phase")
    terminal = payload.get("terminal")
    parser_state = payload.get("parser_state")
    started_at = _parse_timestamp(payload.get("started_at"))
    updated_at = _parse_timestamp(payload.get("updated_at"))
    integer_keys = (
        "step_index",
        "residual_evaluations",
        "jacobian_evaluations",
        "linear_solves",
        "order",
        "time_failures",
        "nonlinear_failures",
        "nonlinear_iteration",
    )
    number_keys = ("simulated_time_seconds", "step_size_seconds")
    stage = payload.get("comsol_progress_percent")
    section = payload.get("comsol_section")
    log_updated_at = payload.get("last_solver_log_update_at")
    return (
        isinstance(hostname, str)
        and bool(hostname)
        and not any(character in hostname for character in "\r\n\t")
        and _is_nonnegative_number(elapsed)
        and phase in _PHASES
        and isinstance(terminal, bool)
        and terminal == (phase in _TERMINAL_PHASES)
        and parser_state in {"available", "unavailable"}
        and started_at is not None
        and updated_at is not None
        and updated_at >= started_at
        and section in {None, "stationary_airflow", "transient_drying"}
        and (stage is None or (_is_nonnegative_integer(stage) and stage <= _MAX_COMSOL_PROGRESS_PERCENT))
        and all(payload[key] is None or _is_nonnegative_integer(payload[key]) for key in integer_keys)
        and all(payload[key] is None or _is_nonnegative_number(payload[key]) for key in number_keys)
        and (log_updated_at is None or _parse_timestamp(log_updated_at) is not None)
    )


def load_runtime_progress(
    run_id: str,
    job_id: str | int,
    expected_identity: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
    manifest: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Load one exact job-bound progress receipt without operational failures.

    Parameters
    ----------
    run_id : str
        Exact campaign-run identifier.
    job_id : str | int
        Exact numeric Slurm job identifier.
    expected_identity : Mapping[str, Any]
        Canonical batch and case identity for the job.
    storage_root : Path | str | None, optional
        Generation storage root.
    manifest : Mapping[str, Any] | None, optional
        Already validated campaign manifest, avoiding repeated loads.
    now : datetime | None, optional
        UTC clock override for deterministic freshness inspection.

    Returns
    -------
    dict[str, Any]
        Available progress with age/staleness or an explicit unavailable view.

    """
    try:
        numeric_job_id = _numeric_job_id(job_id)
        required = ("batch_name", "batch_id", "case_index", "case_id")
        if any(key not in expected_identity for key in required):
            return _unavailable("invalid_expected_identity")
        source_manifest = manifest or campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
        expected = _bound_identity(
            source_manifest,
            run_id=run_id,
            batch_name=str(expected_identity["batch_name"]),
            batch_id=str(expected_identity["batch_id"]),
            case_index=expected_identity["case_index"],
            case_id=str(expected_identity["case_id"]),
            job_id=numeric_job_id,
        )
        path = _progress_receipt_path(
            run_id,
            numeric_job_id,
            storage_root=storage_root,
            create_directory=False,
        )
        if not path.exists():
            return _unavailable("not_reported")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _payload_is_supported(payload, expected):
            return _unavailable("invalid_or_unsupported")
        updated_at = _parse_timestamp(payload["updated_at"])
        if updated_at is None:
            return _unavailable("invalid_or_unsupported")
        age = max(0.0, ((now or _utc_now()) - updated_at).total_seconds())
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return _unavailable("invalid_or_unavailable")
    else:
        return {
            **payload,
            "availability": "available",
            "reason": None,
            "age_seconds": age,
            "stale": age > PROGRESS_STALE_AFTER_SECONDS,
        }
