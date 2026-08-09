"""
===============================================================================
generation_runtime.py
===============================================================================
Run, validate, and atomically publish isolated profile-qualified COMSOL cases.
Responsibilities:
  - Execute safe one-node COMSOL commands and retain complete runtime evidence
  - Collect explicit raw adapters and convert them to canonical case.h5
  - Atomically publish resume-safe cases and terminal batch manifests
Design principles:
  - Scientific configuration and execution provenance are physically separate
  - Successful CSV and solved-model retention is explicit and off by default
  - Failed scratch is removed only after compact persistent evidence is complete
This module does NOT:
  - Modify COMSOL templates or infer internal tags, expressions, or signs
  - Publish a parallel canonical CSV learning view
===============================================================================
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import common

from . import generation_case as case_service
from . import generation_config as config_contract
from . import generation_source as source_service
from . import generation_storage as storage_service
from . import generation_workspace as workspace_service

PUBLICATION_SCHEMA_VERSION = 1
CASE_FAILURE_SCHEMA_KIND = "simulation_case_failure"
CASE_FAILURE_SCHEMA_VERSION = 1
_CASE_FAILURE_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "state",
        "simulation_profile",
        "batch_id",
        "batch_identity",
        "scientific_config_digest",
        "case_id",
        "case_index",
        "git_commit",
        "recorded_at",
        "execution",
        "error",
        "work_directory",
        "input_files",
        "template_sha256",
        "missing_or_invalid_artifacts",
        "log_tail",
        "scratch_cleanup",
    }
)
_FAILURE_EXECUTION_KEYS = frozenset(
    {
        "worker_slot",
        "scheduler_kind",
        "allocated_node",
        "hostname",
        "scheduler_job_id",
        "scheduler_array_job_id",
        "scheduler_array_task_id",
        "scheduler_step_id",
        "command",
        "cwd",
        "exit_code",
        "timed_out",
        "configured_modules",
        "loaded_modules",
    }
)


class CaseExecutionError(RuntimeError):
    """Report one failed case with structured execution evidence."""

    def __init__(
        self,
        message: str,
        *,
        work_directory: Path,
        command: tuple[str, ...] = (),
        exit_code: int | None = None,
        timed_out: bool = False,
        missing_or_invalid_artifacts: tuple[str, ...] = (),
    ) -> None:
        """Initialize one structured case-execution failure."""
        super().__init__(message)
        self.work_directory = work_directory
        self.command = command
        self.cwd = work_directory
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.missing_or_invalid_artifacts = missing_or_invalid_artifacts


class CaseCleanupError(RuntimeError):
    """Report cleanup failure after persistent outcome evidence exists."""


class CaseInterruptedError(InterruptedError):
    """Report cooperative campaign cancellation with solver evidence."""

    def __init__(
        self,
        message: str,
        *,
        work_directory: Path,
        command: tuple[str, ...],
        exit_code: int | None,
    ) -> None:
        """Initialize one structured cancellation error."""
        super().__init__(message)
        self.work_directory = work_directory
        self.cwd = work_directory
        self.command = command
        self.exit_code = exit_code
        self.timed_out = False
        self.missing_or_invalid_artifacts: tuple[str, ...] = ()


_ACTIVE_SOLVER_LOCK = threading.Lock()
_ACTIVE_SOLVERS: dict[int, subprocess.Popen[str]] = {}
_RUNTIME_CANCELLATION = threading.Event()


def reset_runtime_cancellation() -> None:
    """Clear cooperative cancellation before one campaign worker starts."""
    with _ACTIVE_SOLVER_LOCK:
        if _ACTIVE_SOLVERS:
            message = "Cannot reset runtime cancellation while solvers remain active."
            raise RuntimeError(message)
        _RUNTIME_CANCELLATION.clear()


def runtime_cancellation_requested() -> bool:
    """Return whether the current campaign worker received cancellation."""
    return _RUNTIME_CANCELLATION.is_set()


def _signal_solver_termination(process: subprocess.Popen[str]) -> None:
    """Best-effort TERM one solver-owned process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def request_runtime_cancellation() -> None:
    """Request cancellation and TERM every currently registered solver group."""
    _RUNTIME_CANCELLATION.set()
    with _ACTIVE_SOLVER_LOCK:
        processes = tuple(_ACTIVE_SOLVERS.values())
    for process in processes:
        _signal_solver_termination(process)


def _register_solver(process: subprocess.Popen[str]) -> None:
    """Register one solver and close the signal-before-registration race."""
    with _ACTIVE_SOLVER_LOCK:
        _ACTIVE_SOLVERS[process.pid] = process
    if runtime_cancellation_requested():
        _signal_solver_termination(process)


def _unregister_solver(process: subprocess.Popen[str]) -> None:
    """Remove one completed or terminated solver from cancellation tracking."""
    with _ACTIVE_SOLVER_LOCK:
        _ACTIVE_SOLVERS.pop(process.pid, None)


def _terminate_solver_and_wait(process: subprocess.Popen[str]) -> int:
    """TERM then KILL one solver group and return its final status."""
    _signal_solver_termination(process)
    try:
        return process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return process.wait()


@dataclass(frozen=True, slots=True)
class CollectedExport:
    """One validated raw export adapter and its exact-byte identity."""

    source_path: Path
    relative_path: Path
    role: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """One successful COMSOL process and validated canonical conversion."""

    prepared: case_service.PreparedCase
    command: tuple[str, ...]
    timing: dict[str, Any]
    exports: tuple[CollectedExport, ...]
    canonical_case: storage_service.CanonicalCase
    solver_log: Path
    solved_model: Path
    execution_provenance: Path


@dataclass(frozen=True, slots=True)
class CaseRunOutcome:
    """One skipped or newly published completed case."""

    status: str
    case_id: str
    processed_directory: Path
    work_directory: Path | None


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _state_batch_root(config: config_contract.GenerationConfig, *, storage_root: Path | str | None) -> Path:
    """Return the private state root for one profile-qualified batch."""
    return common.paths.get_generation_state_root(storage_root=storage_root) / config.profile.id / config.batch_id


def case_lock_path(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one persistent case-level advisory-lock anchor."""
    return _state_batch_root(config, storage_root=storage_root) / "locks" / f"{config.case_id(case_index)}.lock"


def raw_case_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one permanent case-input provenance directory."""
    return common.paths.resolve_generated_batch_dir(config.batch_id, stage="raw", storage_root=storage_root) / config.case_id(case_index)


def processed_case_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one permanent canonical completed-case directory."""
    return common.paths.resolve_generated_batch_dir(config.batch_id, stage="processed", storage_root=storage_root) / config.case_id(case_index)


def batch_meta_directory(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one batch-owned metadata directory."""
    return common.paths.get_generation_meta_root(storage_root=storage_root) / config.batch_id


def _immutable_json(path: Path, payload: dict[str, Any], *, label: str) -> Path:
    """Write or validate one immutable deterministic JSON object."""
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != serialized:
            msg = f"Existing {label} disagrees with the resolved configuration: {path}"
            raise RuntimeError(msg)
        return path
    return common.serialization.atomic_write_text(path, serialized)


def initialize_batch_metadata(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Publish the immutable scientific config and separate execution provenance."""
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    directory = batch_meta_directory(config, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    scientific_path = _immutable_json(
        directory / "resolved_generation_config.json",
        config.scientific_values,
        label="resolved scientific generation configuration",
    )
    persisted_scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
    if common.serialization.canonical_json_sha256(persisted_scientific) != config.scientific_config_digest:
        msg = "Persisted resolved_generation_config.json digest disagrees with scientific identity."
        raise RuntimeError(msg)
    execution_digest = common.serialization.canonical_json_sha256(config.execution_values)
    execution_directory = directory / "execution_configs"
    execution_directory.mkdir(exist_ok=True)
    _immutable_json(
        execution_directory / f"{execution_digest}.json",
        config.execution_values,
        label="resolved execution provenance",
    )
    return scientific_path


def resolve_comsol_executable(config: config_contract.GenerationConfig) -> str:
    """Return the configured COMSOL executable without shell parsing."""
    executable = config.execution_values["runtime"].get("executable") or os.environ.get("COMSOL_EXECUTABLE")
    if not executable:
        msg = "COMSOL executable is unresolved; configure runtime.executable or COMSOL_EXECUTABLE."
        raise FileNotFoundError(msg)
    return str(executable)


def build_comsol_command(
    config: config_contract.GenerationConfig,
    *,
    cores_per_case: int,
    scheduler_kind: str = "local",
    node_hostname: str | None = None,
) -> list[str]:
    """Build one safe single-node COMSOL batch argument vector."""
    if isinstance(cores_per_case, bool) or not isinstance(cores_per_case, int) or cores_per_case < 1:
        msg = f"cores_per_case must be a positive integer, got {cores_per_case!r}."
        raise ValueError(msg)
    command = [
        resolve_comsol_executable(config),
        "batch",
        "-inputfile",
        "model.mph",
        "-outputfile",
        "solved.mph",
        "-np",
        str(cores_per_case),
        *config.execution_values["runtime"]["extra_arguments"],
    ]
    if scheduler_kind == "local":
        return command
    if scheduler_kind != "slurm":
        msg = f"Unsupported scheduler kind for case execution: {scheduler_kind!r}."
        raise ValueError(msg)
    hostname = node_hostname or socket.gethostname()
    if not hostname or any(character.isspace() for character in hostname):
        msg = f"Cannot create a node-confined Slurm substep for hostname {hostname!r}."
        raise ValueError(msg)
    return [
        "srun",
        "--exclusive",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cores_per_case}",
        "--cpu-bind=cores",
        f"--nodelist={hostname}",
        *command,
    ]


def _require_executable(command: list[str], *, comsol_executable: str) -> None:
    """Require the launcher and COMSOL executable before process creation."""
    for name in dict.fromkeys((command[0], comsol_executable)):
        candidate = Path(name).expanduser()
        if candidate.parent != Path():
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                msg = f"Required executable is missing or not executable: {name}"
                raise FileNotFoundError(msg)
        elif shutil.which(name) is None:
            msg = f"Required executable is not available on PATH: {name}"
            raise FileNotFoundError(msg)


def collect_exports(
    config: config_contract.GenerationConfig,
    prepared: case_service.PreparedCase,
) -> tuple[CollectedExport, ...]:
    """Collect every explicit profile mapping without interpreting binary template data."""
    root = prepared.exports_directory.resolve()
    collected: dict[Path, CollectedExport] = {}
    for contract in config.scientific_values["output_contract"]["exports"]:
        pattern = contract["pattern"]
        if not isinstance(pattern, str):
            msg = f"Profile mapping for {contract['role']!r} remains unresolved."
            raise TypeError(msg)
        matches = sorted(root.glob(pattern))
        files = [path for path in matches if path.is_file() and not path.is_symlink()]
        if contract["required"] and not files:
            msg = f"Required COMSOL export pattern produced no files: {pattern!r} under {root}"
            raise FileNotFoundError(msg)
        if not contract["allow_multiple"] and len(files) > 1:
            msg = f"COMSOL export pattern must match at most one file: {pattern!r}"
            raise ValueError(msg)
        for path in files:
            canonical = path.resolve()
            if not canonical.is_relative_to(root):
                msg = f"Configured export escapes its case-owned root: {path}"
                raise ValueError(msg)
            size = canonical.stat().st_size
            if size <= 0:
                msg = f"Configured COMSOL export is empty: {canonical}"
                raise ValueError(msg)
            relative = canonical.relative_to(root)
            if relative in collected:
                msg = f"COMSOL export {relative} matches more than one configured role."
                raise ValueError(msg)
            collected[relative] = CollectedExport(
                source_path=canonical,
                relative_path=relative,
                role=contract["role"],
                sha256=common.serialization.file_sha256(canonical),
                size_bytes=size,
            )
    if not collected:
        msg = f"No configured COMSOL exports were collected from {root}."
        raise FileNotFoundError(msg)
    return tuple(collected[path] for path in sorted(collected, key=lambda item: item.as_posix()))


def _write_solver_log(prepared: case_service.PreparedCase) -> Path:
    """Combine case-owned stdout and stderr into one retained solver log."""
    stdout_path = prepared.runtime_directory / "stdout.log"
    stderr_path = prepared.runtime_directory / "stderr.log"
    payload = "===== stdout =====\n" + stdout_path.read_text(encoding="utf-8", errors="replace")
    if not payload.endswith("\n"):
        payload += "\n"
    payload += "===== stderr =====\n" + stderr_path.read_text(encoding="utf-8", errors="replace")
    if not payload.endswith("\n"):
        payload += "\n"
    path = prepared.runtime_directory / "solver.log"
    common.serialization.atomic_write_text(path, payload)
    return path


def _execution_provenance(
    config: config_contract.GenerationConfig,
    prepared: case_service.PreparedCase,
    *,
    command: list[str],
    cores_per_case: int,
    worker_slot: int,
    scheduler_kind: str,
    allocated_node: str | None,
) -> Path:
    """Persist resolved execution settings separately from scientific provenance."""
    execution_digest = common.serialization.canonical_json_sha256(config.execution_values)
    payload = {
        "schema_kind": "simulation_execution_provenance",
        "schema_version": 1,
        "case_id": prepared.bundle.case_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "execution_config_digest": execution_digest,
        "git_commit": prepared.bundle.case_payload["git_commit"],
        "execution_config": config.execution_values,
        "invocation": {
            "arguments": command,
            "working_directory": str(prepared.work_directory),
            "executable_path": shutil.which(resolve_comsol_executable(config)),
            "requested_cores": cores_per_case,
            "worker_slot": worker_slot,
            "scheduler_kind": scheduler_kind,
            "allocated_node": allocated_node,
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "configured_modules": list(config.execution_values["runtime"]["module_initialization"]),
            "loaded_modules": os.environ.get("LOADEDMODULES"),
            "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
            "scheduler_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "scheduler_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "scheduler_step_id": os.environ.get("SLURM_STEP_ID"),
            "scheduler_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
            "scheduler_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "result": {
            "state": "starting",
            "exit_code": None,
            "timed_out": False,
            "started_at": None,
            "ended_at": None,
            "runtime_s": None,
        },
    }
    return common.serialization.atomic_write_json(prepared.runtime_directory / "execution_provenance.json", payload)


def _complete_execution_provenance(
    path: Path,
    *,
    state: str,
    exit_code: int | None,
    timed_out: bool,
    started_at: str | None,
    ended_at: str,
    runtime_seconds: float | None,
) -> None:
    """Complete one execution record with the observed process outcome."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"Execution provenance became malformed before completion: {path}"
        raise TypeError(message)
    payload["result"] = {
        "state": state,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_s": runtime_seconds,
    }
    common.serialization.atomic_write_json(path, payload)


def _solver_environment(prepared: case_service.PreparedCase) -> dict[str, str]:
    """Return the inherited solver environment plus exact case identities."""
    environment = os.environ.copy()
    environment.update(
        {
            "GENERATION_CASE_ID": prepared.bundle.case_id,
            "GENERATION_SIMULATION_PROFILE": prepared.bundle.case_payload["simulation_profile"],
            "GENERATION_WORK_DIRECTORY": str(prepared.work_directory),
        }
    )
    return environment


def execute_prepared_case(
    config: config_contract.GenerationConfig,
    prepared: case_service.PreparedCase,
    *,
    cores_per_case: int,
    worker_slot: int,
    scheduler_kind: str = "local",
    allocated_node: str | None = None,
) -> ExecutionResult:
    """Run one isolated COMSOL process and create its validated canonical HDF5."""
    command = build_comsol_command(
        config,
        cores_per_case=cores_per_case,
        scheduler_kind=scheduler_kind,
        node_hostname=allocated_node,
    )
    try:
        _require_executable(command, comsol_executable=resolve_comsol_executable(config))
    except FileNotFoundError as error:
        raise CaseExecutionError(
            str(error),
            work_directory=prepared.work_directory,
            command=tuple(command),
        ) from error
    execution_provenance = _execution_provenance(
        config,
        prepared,
        command=command,
        cores_per_case=cores_per_case,
        worker_slot=worker_slot,
        scheduler_kind=scheduler_kind,
        allocated_node=allocated_node,
    )
    hostname = socket.gethostname()
    started_at = _utc_now()
    monotonic_start = time.monotonic()
    stdout_path = prepared.runtime_directory / "stdout.log"
    stderr_path = prepared.runtime_directory / "stderr.log"
    process: subprocess.Popen[str] | None = None
    timed_out = False
    exit_code: int | None = None
    if runtime_cancellation_requested():
        message = "Campaign cancellation was requested before COMSOL launch."
        raise CaseInterruptedError(
            message,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=None,
        )
    with (
        stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_stream,
        stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_stream,
    ):
        try:
            process = subprocess.Popen(  # noqa: S603 -- validated argument vector without a shell
                command,
                cwd=prepared.work_directory,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                start_new_session=True,
                env=_solver_environment(prepared),
            )
            _register_solver(process)
            try:
                try:
                    exit_code = process.wait(timeout=float(config.execution_values["runtime"]["timeout_seconds"]))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    exit_code = _terminate_solver_and_wait(process)
                except BaseException:
                    _terminate_solver_and_wait(process)
                    raise
            finally:
                _unregister_solver(process)
        except OSError as error:
            ended_at = _utc_now()
            _complete_execution_provenance(
                execution_provenance,
                state="start_failed",
                exit_code=None,
                timed_out=False,
                started_at=started_at,
                ended_at=ended_at,
                runtime_seconds=time.monotonic() - monotonic_start,
            )
            msg = f"Could not start COMSOL command {command!r}: {error}"
            raise CaseExecutionError(
                msg,
                work_directory=prepared.work_directory,
                command=tuple(command),
            ) from error
    elapsed = time.monotonic() - monotonic_start
    timing = {
        "schema_kind": "simulation_case_timing",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "batch_id": config.batch_id,
        "case_id": prepared.bundle.case_id,
        "case_input_id": prepared.bundle.case_input_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "simulation_profile": config.profile.id,
        "git_commit": prepared.bundle.case_payload["git_commit"],
        "executable": resolve_comsol_executable(config),
        "arguments": command,
        "working_directory": str(prepared.work_directory),
        "hostname": hostname,
        "process_id": None if process is None else process.pid,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "runtime_s": elapsed,
        "requested_cores": cores_per_case,
        "worker_slot": worker_slot,
        "allocated_node": allocated_node,
        "scheduler_kind": scheduler_kind,
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "scheduler_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "scheduler_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "scheduler_step_id": os.environ.get("SLURM_STEP_ID"),
        "configured_modules": list(config.execution_values["runtime"]["module_initialization"]),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
        "python_version": sys.version,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "template_sha256": config.template_sha256,
    }
    common.serialization.atomic_write_json(prepared.runtime_directory / "timing.json", timing)
    cancelled = runtime_cancellation_requested()
    _complete_execution_provenance(
        execution_provenance,
        state=("cancelled" if cancelled else "timed_out" if timed_out else "succeeded" if exit_code == 0 else "failed"),
        exit_code=exit_code,
        timed_out=timed_out,
        started_at=started_at,
        ended_at=timing["ended_at"],
        runtime_seconds=elapsed,
    )
    solver_log = _write_solver_log(prepared)
    if cancelled:
        message = "Campaign cancellation terminated the COMSOL process."
        raise CaseInterruptedError(
            message,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
        )
    if timed_out:
        msg = "COMSOL case exceeded its configured timeout."
        raise CaseExecutionError(
            msg,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            timed_out=True,
        )
    if exit_code != 0:
        msg = f"COMSOL case exited with status {exit_code}."
        raise CaseExecutionError(
            msg,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
        )
    solved_model = prepared.work_directory / "solved.mph"
    if not solved_model.is_file() or solved_model.stat().st_size <= 0:
        msg = "COMSOL completed without a non-empty solved.mph output."
        raise CaseExecutionError(
            msg,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            missing_or_invalid_artifacts=("solved.mph",),
        )
    try:
        exports = collect_exports(config, prepared)
        canonical_case = storage_service.convert_exports_to_hdf5(
            config,
            prepared.bundle.case_payload,
            exports,
            work_directory=prepared.work_directory,
            runtime_directory=prepared.runtime_directory,
            runtime_seconds=elapsed,
        )
    except Exception as error:
        raise CaseExecutionError(
            str(error),
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            missing_or_invalid_artifacts=(str(error),),
        ) from error
    return ExecutionResult(
        prepared=prepared,
        command=tuple(command),
        timing=timing,
        exports=exports,
        canonical_case=canonical_case,
        solver_log=solver_log,
        solved_model=solved_model,
        execution_provenance=execution_provenance,
    )


def _artifact_map(directory: Path) -> dict[str, dict[str, Any]]:
    """Return exact-byte identities for all staged payload artifacts."""
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            msg = f"Published case staging cannot contain symbolic links: {path}"
            raise ValueError(msg)
        if not path.is_file() or path.name in {"provenance.json", "_SUCCESS"}:
            continue
        artifacts[path.relative_to(directory).as_posix()] = {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return artifacts


def _complete_stage(
    directory: Path,
    *,
    config: config_contract.GenerationConfig,
    case_payload: dict[str, Any],
    stage: str,
) -> None:
    """Write digest-bound publication provenance and final success evidence."""
    artifacts = _artifact_map(directory)
    provenance = {
        "schema_kind": "simulation_case_publication",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "stage": stage,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "case_id": case_payload["case_id"],
        "case_input_id": case_payload["case_input_id"],
        "simulation_case_id": case_payload["simulation_case_id"],
        "material_family": case_payload["material_family"],
        "git_commit": case_payload["git_commit"],
        "template_sha256": config.template_sha256,
        "scientific_config_digest": config.scientific_config_digest,
        "export_contract_sha256": case_payload["export_contract_sha256"],
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "artifacts": artifacts,
    }
    provenance_path = common.serialization.atomic_write_json(directory / "provenance.json", provenance)
    success = {
        "schema_kind": "simulation_case_success",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "stage": stage,
        "batch_id": config.batch_id,
        "case_id": case_payload["case_id"],
        "case_input_id": case_payload["case_input_id"],
        "simulation_case_id": case_payload["simulation_case_id"],
        "provenance_sha256": common.serialization.file_sha256(provenance_path),
    }
    common.serialization.atomic_write_json(directory / "_SUCCESS", success)


def _copy_retained_csv(config: config_contract.GenerationConfig, result: ExecutionResult, destination: Path) -> None:
    """Copy input and raw-export CSV only under the explicit retention policy."""
    if not config.execution_values["retention"]["retain_raw_csv"]:
        return
    adapter_root = destination / "raw_csv" / "inputs"
    adapter_root.mkdir(parents=True)
    for path in result.prepared.bundle.input_paths:
        shutil.copy2(path, adapter_root / path.name)
    export_root = destination / "raw_csv" / "exports"
    for export in result.exports:
        target = export_root / export.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export.source_path, target)
        if common.serialization.file_sha256(target) != export.sha256:
            msg = f"Retained raw export digest changed during copy: {target}"
            raise RuntimeError(msg)


def _stage_raw_case(config: config_contract.GenerationConfig, result: ExecutionResult, destination: Path) -> None:
    """Stage compact case-input provenance and optional adapter CSV."""
    destination.mkdir(parents=True)
    shutil.copy2(result.prepared.work_directory / "case.json", destination / "case.json")
    _copy_retained_csv(config, result, destination)
    _complete_stage(destination, config=config, case_payload=result.prepared.bundle.case_payload, stage="raw")


def _stage_processed_case(config: config_contract.GenerationConfig, result: ExecutionResult, destination: Path) -> None:
    """Stage the sole canonical payload and retained runtime evidence."""
    destination.mkdir(parents=True)
    shutil.copy2(result.prepared.work_directory / "case.json", destination / "case.json")
    shutil.copy2(result.canonical_case.path, destination / "case.h5")
    shutil.copy2(result.solver_log, destination / "solver.log")
    shutil.copy2(result.prepared.runtime_directory / "timing.json", destination / "timing.json")
    shutil.copy2(result.canonical_case.status_path, destination / "status.json")
    shutil.copy2(result.execution_provenance, destination / "execution_provenance.json")
    if config.execution_values["retention"]["retain_solved_model"]:
        shutil.copy2(result.solved_model, destination / "solved.mph")
    _complete_stage(destination, config=config, case_payload=result.prepared.bundle.case_payload, stage="processed")


def case_failure_path(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the persistent private failure-evidence path for one case."""
    return _state_batch_root(config, storage_root=storage_root) / "failures" / f"{config.case_id(case_index)}.json"


def _optional_json_object(path: Path) -> dict[str, Any] | None:
    """Load one optional non-symlink JSON object for failure evidence."""
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _failure_input_evidence(
    config: config_contract.GenerationConfig,
    work_directory: Path | None,
) -> dict[str, Any]:
    """Return declared and observed exact input identities."""
    if work_directory is None:
        return {"declared": {}, "observed": {}}
    case_payload = _optional_json_object(work_directory / "case.json")
    declared = case_payload.get("input_files", {}) if isinstance(case_payload, dict) else {}
    if not isinstance(declared, dict):
        declared = {}
    observed: dict[str, dict[str, Any]] = {}
    expected_names = {"fields.csv"}
    if config.profile.id == "transient_drying":
        expected_names.update({"scalars.csv", "schedule.csv"})
    for name in sorted(expected_names | set(declared)):
        path = work_directory / name
        if path.is_file() and not path.is_symlink():
            observed[name] = {
                "sha256": common.serialization.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return {"declared": declared, "observed": observed}


def _failure_log_tail(work_directory: Path | None) -> dict[str, str] | None:
    """Return a compact UTF-8 tail from the best available case-local log."""
    if work_directory is None:
        return None
    candidates = (
        work_directory / "runtime" / "solver.log",
        work_directory / "runtime" / "stderr.log",
        work_directory / "runtime" / "stdout.log",
    )
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-80:])
        return {"source": path.name, "text": tail[-16384:]}
    return None


def _failure_missing_artifacts(
    config: config_contract.GenerationConfig,
    error: BaseException,
    work_directory: Path | None,
) -> list[str]:
    """Return exact missing or invalid runtime artifacts known at failure."""
    declared = getattr(error, "missing_or_invalid_artifacts", ())
    missing = {str(item) for item in declared if str(item)}
    if work_directory is None:
        missing.add("case_workspace")
        return sorted(missing)
    required = {"model.mph", "case.json", "fields.csv"}
    if config.profile.id == "transient_drying":
        required.update({"scalars.csv", "schedule.csv"})
    for name in required:
        path = work_directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            missing.add(name)
    solved = work_directory / "solved.mph"
    if not solved.is_file() or solved.is_symlink() or solved.stat().st_size <= 0:
        missing.add("solved.mph")
    exports_root = work_directory / config.scientific_values["output_contract"]["exports_root"]
    for contract in config.scientific_values["output_contract"]["exports"]:
        pattern = contract.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            missing.add(f"export:{contract['role']}:unresolved_mapping")
            continue
        matches = [path for path in exports_root.glob(pattern) if path.is_file() and not path.is_symlink() and path.stat().st_size > 0]
        if contract["required"] and not matches:
            missing.add(f"export:{contract['role']}")
    return sorted(missing)


def _failure_execution_evidence(
    config: config_contract.GenerationConfig,
    error: BaseException,
    *,
    worker_slot: int,
    scheduler_kind: str,
    allocated_node: str | None,
    work_directory: Path | None,
) -> dict[str, Any]:
    """Return command, cwd, scheduler, module, and exit evidence."""
    runtime = None
    if work_directory is not None:
        runtime = _optional_json_object(work_directory / "runtime" / "execution_provenance.json")
    invocation = runtime.get("invocation", {}) if isinstance(runtime, dict) else {}
    result = runtime.get("result", {}) if isinstance(runtime, dict) else {}
    command = getattr(error, "command", ())
    if not command and isinstance(invocation, dict):
        command = invocation.get("arguments", ())
    cwd = getattr(error, "cwd", None)
    if cwd is None and isinstance(invocation, dict):
        cwd = invocation.get("working_directory")
    exit_code = getattr(error, "exit_code", None)
    if exit_code is None and isinstance(result, dict):
        exit_code = result.get("exit_code")
    timed_out = bool(getattr(error, "timed_out", False))
    if not timed_out and isinstance(result, dict):
        timed_out = result.get("timed_out") is True
    return {
        "worker_slot": worker_slot,
        "scheduler_kind": scheduler_kind,
        "allocated_node": allocated_node,
        "hostname": socket.gethostname(),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "scheduler_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "scheduler_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "scheduler_step_id": os.environ.get("SLURM_STEP_ID"),
        "command": list(command) if isinstance(command, (list, tuple)) else [],
        "cwd": None if cwd is None else str(cwd),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "configured_modules": list(config.execution_values["runtime"]["module_initialization"]),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
    }


def record_case_failure(
    config: config_contract.GenerationConfig,
    case_index: int,
    error: BaseException,
    *,
    worker_slot: int,
    scheduler_kind: str,
    allocated_node: str | None,
    work_directory: Path | None,
    storage_root: Path | str | None = None,
    scratch_cleanup_status: str = "pending",
) -> Path:
    """Persist compact failure evidence before node-local scratch cleanup."""
    if scratch_cleanup_status not in {"pending", "not_created"}:
        message = f"Initial scratch cleanup status is invalid: {scratch_cleanup_status!r}"
        raise ValueError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    path = case_failure_path(config, case_index, storage_root=storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = "cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed"
    payload = {
        "schema_kind": CASE_FAILURE_SCHEMA_KIND,
        "schema_version": CASE_FAILURE_SCHEMA_VERSION,
        "state": state,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "scientific_config_digest": config.scientific_config_digest,
        "case_id": config.case_id(case_index),
        "case_index": case_index,
        "git_commit": source_service.required_git_commit(),
        "recorded_at": _utc_now(),
        "execution": _failure_execution_evidence(
            config,
            error,
            worker_slot=worker_slot,
            scheduler_kind=scheduler_kind,
            allocated_node=allocated_node,
            work_directory=work_directory,
        ),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "work_directory": None if work_directory is None else str(work_directory),
        "input_files": _failure_input_evidence(config, work_directory),
        "template_sha256": config.template_sha256,
        "missing_or_invalid_artifacts": _failure_missing_artifacts(
            config,
            error,
            work_directory,
        ),
        "log_tail": _failure_log_tail(work_directory),
        "scratch_cleanup": {
            "status": scratch_cleanup_status,
            "reclaimed_bytes": 0,
            "error": None,
        },
    }
    common.serialization.atomic_write_json(path, payload)
    return path


def _complete_failure_cleanup(
    path: Path,
    *,
    status: str,
    reclaimed_bytes: int,
    error: str | None,
) -> None:
    """Atomically complete one previously persisted scratch-cleanup receipt."""
    if status not in {"complete", "failed", "not_created"}:
        message = f"Completed scratch cleanup status is invalid: {status!r}"
        raise ValueError(message)
    payload = _load_json_object(path, label="case failure evidence")
    payload["scratch_cleanup"] = {
        "status": status,
        "reclaimed_bytes": reclaimed_bytes,
        "error": error,
    }
    common.serialization.atomic_write_json(path, payload)


def clear_case_failure(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> None:
    """Clear stale failure evidence after exact completion or integrity reuse."""
    case_failure_path(config, case_index, storage_root=storage_root).unlink(missing_ok=True)


def case_failure_is_recorded(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> bool:
    """Validate and report durable failure evidence for one incomplete case."""
    path = case_failure_path(config, case_index, storage_root=storage_root)
    if not path.exists():
        return False
    if not path.is_file() or path.is_symlink():
        msg = f"Case failure evidence is unsafe: {path}"
        raise ValueError(msg)
    payload = _load_json_object(path, label="case failure evidence")
    if (
        set(payload) != _CASE_FAILURE_KEYS
        or payload.get("schema_kind") != CASE_FAILURE_SCHEMA_KIND
        or payload.get("schema_version") != CASE_FAILURE_SCHEMA_VERSION
        or payload.get("state") not in {"failed", "cancelled"}
        or payload.get("simulation_profile") != config.profile.id
        or payload.get("batch_id") != config.batch_id
        or payload.get("batch_identity") != config.batch_identity
        or payload.get("scientific_config_digest") != config.scientific_config_digest
        or payload.get("case_id") != config.case_id(case_index)
        or payload.get("case_index") != case_index
        or payload.get("template_sha256") != config.template_sha256
    ):
        msg = f"Case failure evidence identity is invalid: {path}"
        raise ValueError(msg)
    source_service.validate_git_commit(payload.get("git_commit"))
    execution = payload.get("execution")
    error = payload.get("error")
    if (
        not isinstance(execution, dict)
        or set(execution) != _FAILURE_EXECUTION_KEYS
        or not isinstance(execution.get("worker_slot"), int)
        or isinstance(execution.get("worker_slot"), bool)
        or not isinstance(execution.get("scheduler_kind"), str)
        or not isinstance(execution.get("hostname"), str)
        or not isinstance(execution.get("command"), list)
        or not all(isinstance(argument, str) for argument in execution.get("command", []))
        or not isinstance(execution.get("timed_out"), bool)
        or not isinstance(execution.get("configured_modules"), list)
    ):
        msg = f"Case failure execution evidence is invalid: {path}"
        raise ValueError(msg)
    if (
        not isinstance(error, dict)
        or set(error) != {"type", "message"}
        or not isinstance(error["type"], str)
        or not isinstance(error["message"], str)
    ):
        msg = f"Case failure error evidence is invalid: {path}"
        raise ValueError(msg)
    work_directory = payload.get("work_directory")
    if work_directory is not None and (not isinstance(work_directory, str) or not work_directory or not Path(work_directory).is_absolute()):
        msg = f"Case failure work-directory evidence is invalid: {path}"
        raise ValueError(msg)
    inputs = payload.get("input_files")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"declared", "observed"}
        or not isinstance(inputs["declared"], dict)
        or not isinstance(inputs["observed"], dict)
    ):
        msg = f"Case failure input identity evidence is invalid: {path}"
        raise ValueError(msg)
    missing = payload.get("missing_or_invalid_artifacts")
    if not isinstance(missing, list) or not all(isinstance(item, str) and item for item in missing):
        msg = f"Case failure artifact evidence is invalid: {path}"
        raise ValueError(msg)
    log_tail = payload.get("log_tail")
    if log_tail is not None and (
        not isinstance(log_tail, dict) or set(log_tail) != {"source", "text"} or not all(isinstance(value, str) for value in log_tail.values())
    ):
        msg = f"Case failure log-tail evidence is invalid: {path}"
        raise ValueError(msg)
    cleanup = payload.get("scratch_cleanup")
    if (
        not isinstance(cleanup, dict)
        or set(cleanup) != {"status", "reclaimed_bytes", "error"}
        or cleanup.get("status") not in {"pending", "complete", "failed", "not_created"}
        or not isinstance(cleanup.get("reclaimed_bytes"), int)
        or isinstance(cleanup.get("reclaimed_bytes"), bool)
        or cleanup.get("reclaimed_bytes", -1) < 0
        or (cleanup.get("error") is not None and not isinstance(cleanup.get("error"), str))
    ):
        msg = f"Case failure scratch-cleanup evidence is invalid: {path}"
        raise ValueError(msg)
    return True


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required JSON object with contextual errors."""
    if not path.is_file() or path.is_symlink():
        msg = f"Missing required {label}: {path}"
        raise FileNotFoundError(msg)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Could not read {label}: {path}"
        raise ValueError(msg) from error
    if not isinstance(value, dict):
        msg = f"{label} must contain a JSON object: {path}"
        raise TypeError(msg)
    return value


def _validate_artifacts(directory: Path, provenance: dict[str, Any]) -> None:
    """Validate exact publication membership and all declared hashes."""
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        msg = f"Case publication has no artifact identity map: {directory}"
        raise RuntimeError(msg)
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path.name not in {"provenance.json", "_SUCCESS"}
    }
    if actual != set(artifacts):
        msg = f"Case publication membership mismatch in {directory}."
        raise RuntimeError(msg)
    for relative, identity in artifacts.items():
        path = directory / relative
        if not isinstance(identity, dict) or set(identity) != {"sha256", "size_bytes"}:
            msg = f"Malformed artifact identity for {relative!r} in {directory}."
            raise RuntimeError(msg)
        if path.stat().st_size != identity["size_bytes"] or common.serialization.file_sha256(path) != identity["sha256"]:
            msg = f"Case artifact integrity failure for {path}."
            raise RuntimeError(msg)


def _validate_publication_directory(
    directory: Path,
    *,
    config: config_contract.GenerationConfig,
    case_index: int,
    stage: str,
) -> dict[str, Any]:
    """Validate one exact raw or processed case publication."""
    success = _load_json_object(directory / "_SUCCESS", label="case success marker")
    provenance_path = directory / "provenance.json"
    provenance = _load_json_object(provenance_path, label="case publication provenance")
    case_payload = _load_json_object(directory / "case.json", label="canonical case provenance")
    expected = config.batch_id, config.case_id(case_index), stage
    if (success.get("batch_id"), success.get("case_id"), success.get("stage")) != expected:
        msg = f"Case success identity mismatch in {directory}."
        raise RuntimeError(msg)
    if (provenance.get("batch_id"), provenance.get("case_id"), provenance.get("stage")) != expected:
        msg = f"Case publication identity mismatch in {directory}."
        raise RuntimeError(msg)
    if success.get("provenance_sha256") != common.serialization.file_sha256(provenance_path):
        msg = f"Case provenance digest mismatch in {directory}."
        raise RuntimeError(msg)
    if case_service.compute_case_input_id(case_payload) != case_payload.get("case_input_id"):
        msg = f"Canonical case-input identity mismatch in {directory}."
        raise RuntimeError(msg)
    if case_service.compute_simulation_case_id(case_payload) != case_payload.get("simulation_case_id"):
        msg = f"Canonical simulation-case identity mismatch in {directory}."
        raise RuntimeError(msg)
    for key in ("case_input_id", "simulation_case_id"):
        if success.get(key) != case_payload.get(key) or provenance.get(key) != case_payload.get(key):
            msg = f"Case {key} evidence disagrees in {directory}."
            raise RuntimeError(msg)
    git_commit = source_service.validate_git_commit(case_payload.get("git_commit"))
    if provenance.get("git_commit") != git_commit:
        msg = f"Case Git-commit evidence disagrees in {directory}."
        raise RuntimeError(msg)
    if (
        case_payload.get("batch_identity") != config.batch_identity
        or case_payload.get("scientific_config_digest") != config.scientific_config_digest
        or case_payload.get("case_input_config_digest") != config.case_input_config_digest
        or case_payload.get("simulation_profile") != config.profile.id
        or case_payload.get("template", {}).get("sha256") != config.template_sha256
        or case_payload.get("available_learning_views") != list(config.profile.available_learning_views)
        or provenance.get("export_contract_sha256") != common.serialization.canonical_json_sha256(config.scientific_values["output_contract"])
    ):
        msg = f"Case scientific, profile, template, or export identity mismatch in {directory}."
        raise RuntimeError(msg)
    _validate_artifacts(directory, provenance)
    if stage == "processed":
        required = {"case.h5", "solver.log", "timing.json", "status.json", "execution_provenance.json", "case.json"}
        if not required.issubset(provenance["artifacts"]):
            msg = f"Processed publication lacks canonical payload or runtime evidence: {directory}"
            raise RuntimeError(msg)
        timing = _load_json_object(directory / "timing.json", label="case timing")
        execution = _load_json_object(directory / "execution_provenance.json", label="case execution provenance")
        if timing.get("git_commit") != git_commit or execution.get("git_commit") != git_commit:
            msg = f"Processed runtime Git-commit evidence disagrees in {directory}."
            raise RuntimeError(msg)
        hdf5_identity = storage_service.validate_case_hdf5(directory / "case.h5", expected_profile=config.profile.id)
        if (
            hdf5_identity["case_input_id"] != case_payload["case_input_id"]
            or hdf5_identity["simulation_case_id"] != case_payload["simulation_case_id"]
            or hdf5_identity["git_commit"] != git_commit
        ):
            msg = f"Canonical HDF5 identities disagree with case.json in {directory}."
            raise RuntimeError(msg)
    return provenance


def validate_completed_case(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate compact input and canonical processed publications."""
    raw = _validate_publication_directory(
        raw_case_directory(config, case_index, storage_root=storage_root),
        config=config,
        case_index=case_index,
        stage="raw",
    )
    processed = _validate_publication_directory(
        processed_case_directory(config, case_index, storage_root=storage_root),
        config=config,
        case_index=case_index,
        stage="processed",
    )
    if (raw["case_input_id"], raw["simulation_case_id"]) != (
        processed["case_input_id"],
        processed["simulation_case_id"],
    ):
        msg = f"Raw and processed identities disagree for {config.case_id(case_index)}."
        raise RuntimeError(msg)
    return processed


def completed_case_is_valid(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> bool:
    """Return false only when completion is absent; corruption fails closed."""
    raw = raw_case_directory(config, case_index, storage_root=storage_root)
    processed = processed_case_directory(config, case_index, storage_root=storage_root)
    raw_success = (raw / "_SUCCESS").exists()
    processed_success = (processed / "_SUCCESS").exists()
    if not processed_success:
        if raw_success:
            _validate_publication_directory(raw, config=config, case_index=case_index, stage="raw")
        return False
    if not raw_success:
        msg = f"Processed completion exists without input provenance: {processed}"
        raise RuntimeError(msg)
    validate_completed_case(config, case_index, storage_root=storage_root)
    return True


def _quarantine_incomplete(path: Path, *, state_root: Path) -> None:
    """Move an incomplete final directory into recoverable private state."""
    if not path.exists():
        return
    if (path / "_SUCCESS").exists():
        msg = f"Refusing to replace an existing completed case: {path}"
        raise FileExistsError(msg)
    quarantine = state_root / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix=f"{path.name}_incomplete_", dir=quarantine))
    target.rmdir()
    path.replace(target)


def publish_completed_case(
    config: config_contract.GenerationConfig,
    result: ExecutionResult,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Atomically publish raw and processed directories without overwriting completion."""
    case_index = int(result.prepared.bundle.case_payload["case_index"])
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    state_root = _state_batch_root(config, storage_root=storage)
    publication_root = state_root / "publications"
    publication_root.mkdir(parents=True, exist_ok=True)
    staging = workspace_service.create_publication_staging(
        storage_root=storage,
        publication_root=publication_root,
        run_id=result.prepared.workspace_run_id,
        case_id=result.prepared.bundle.case_id,
    )
    raw_stage = staging / "raw"
    processed_stage = staging / "processed"
    raw_destination = raw_case_directory(config, case_index, storage_root=storage)
    processed_destination = processed_case_directory(
        config,
        case_index,
        storage_root=storage,
    )
    try:
        _stage_raw_case(config, result, raw_stage)
        _stage_processed_case(config, result, processed_stage)
        if raw_destination.exists() and (raw_destination / "_SUCCESS").exists():
            existing = _validate_publication_directory(raw_destination, config=config, case_index=case_index, stage="raw")
            if existing["simulation_case_id"] != result.prepared.bundle.simulation_case_id:
                msg = f"Existing raw case belongs to another simulation identity: {raw_destination}"
                raise RuntimeError(msg)
            # The marked publication root owns this redundant stage until cleanup.
        else:
            _quarantine_incomplete(raw_destination, state_root=state_root)
            raw_destination.parent.mkdir(parents=True, exist_ok=True)
            raw_stage.replace(raw_destination)
        if processed_destination.exists() and (processed_destination / "_SUCCESS").exists():
            msg = f"Refusing to overwrite existing completed case: {processed_destination}"
            raise FileExistsError(msg)
        _quarantine_incomplete(processed_destination, state_root=state_root)
        processed_destination.parent.mkdir(parents=True, exist_ok=True)
        processed_stage.replace(processed_destination)
        validate_completed_case(config, case_index, storage_root=storage)
    finally:
        workspace_service.cleanup_publication_staging(
            staging,
            storage_root=storage,
            publication_root=publication_root,
            run_id=result.prepared.workspace_run_id,
            case_id=result.prepared.bundle.case_id,
            allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
        )
    return processed_destination


def _case_cleanup_failure_path(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path,
) -> Path:
    """Return persistent evidence for cleanup after completed publication."""
    root = _state_batch_root(config, storage_root=storage_root)
    return root / "cleanup_failures" / f"{config.case_id(case_index)}.json"


def _record_case_cleanup_failure(
    config: config_contract.GenerationConfig,
    case_index: int,
    error: BaseException,
    *,
    work_directory: Path | None,
    storage_root: Path,
) -> Path:
    """Persist a cleanup error without reclassifying valid publication."""
    path = _case_cleanup_failure_path(
        config,
        case_index,
        storage_root=storage_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_kind": "simulation_case_cleanup_failure",
        "schema_version": 1,
        "publication_complete": completed_case_is_valid(
            config,
            case_index,
            storage_root=storage_root,
        ),
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "case_id": config.case_id(case_index),
        "case_index": case_index,
        "recorded_at": _utc_now(),
        "work_directory": (None if work_directory is None else str(work_directory)),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    return common.serialization.atomic_write_json(path, payload)


def _workspace_from_attempt(
    prepared: case_service.PreparedCase | None,
    error: BaseException,
) -> tuple[Path | None, Path | None, str | None]:
    """Return cleanup boundaries from a prepared case or preparation failure."""
    if prepared is not None:
        return (
            prepared.work_directory,
            prepared.work_root,
            prepared.workspace_run_id,
        )
    work_directory = getattr(error, "work_directory", None)
    work_root = getattr(error, "work_root", None)
    run_id = getattr(error, "workspace_run_id", None)
    return (
        work_directory if isinstance(work_directory, Path) else None,
        work_root if isinstance(work_root, Path) else None,
        run_id if isinstance(run_id, str) else None,
    )


def _cleanup_case_attempt(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    work_directory: Path,
    work_root: Path,
    run_id: str,
    storage_root: Path,
) -> int:
    """Remove one marked case attempt under the exact current identity."""
    return workspace_service.cleanup_case_workspace(
        work_directory,
        allowed_root=work_root,
        storage_root=storage_root,
        expected_run_id=run_id,
        expected_case_id=config.case_id(case_index),
        allow_active_job_id=os.environ.get("SLURM_JOB_ID"),
    )


def run_case(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    cores_per_case: int,
    worker_slot: int = 0,
    scheduler_kind: str = "local",
    allocated_node: str | None = None,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
    blocking_lock: bool = True,
) -> CaseRunOutcome:
    """Run or integrity-skip one case and always close marked scratch."""
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    initialize_batch_metadata(config, storage_root=storage)
    lock_path = case_lock_path(config, case_index, storage_root=storage)
    with common.locking.exclusive_file_lock(lock_path, blocking=blocking_lock):
        if completed_case_is_valid(config, case_index, storage_root=storage):
            clear_case_failure(config, case_index, storage_root=storage)
            return CaseRunOutcome(
                status="skipped",
                case_id=config.case_id(case_index),
                processed_directory=processed_case_directory(
                    config,
                    case_index,
                    storage_root=storage,
                ),
                work_directory=None,
            )
        prepared: case_service.PreparedCase | None = None
        try:
            prepared = case_service.prepare_case_work_directory(
                config,
                case_index,
                storage_root=storage,
                work_root=work_root,
            )
            result = execute_prepared_case(
                config,
                prepared,
                cores_per_case=cores_per_case,
                worker_slot=worker_slot,
                scheduler_kind=scheduler_kind,
                allocated_node=allocated_node,
            )
            destination = publish_completed_case(
                config,
                result,
                storage_root=storage,
            )
        except BaseException as error:
            attempt_directory, attempt_root, attempt_run_id = _workspace_from_attempt(prepared, error)
            try:
                publication_complete = completed_case_is_valid(
                    config,
                    case_index,
                    storage_root=storage,
                )
            except Exception:  # noqa: BLE001 -- corruption remains failed, but scratch still closes
                publication_complete = False
            failure_path: Path | None = None
            if publication_complete:
                _record_case_cleanup_failure(
                    config,
                    case_index,
                    error,
                    work_directory=attempt_directory,
                    storage_root=storage,
                )
            else:
                failure_path = record_case_failure(
                    config,
                    case_index,
                    error,
                    worker_slot=worker_slot,
                    scheduler_kind=scheduler_kind,
                    allocated_node=allocated_node,
                    work_directory=attempt_directory,
                    storage_root=storage,
                    scratch_cleanup_status=("pending" if attempt_directory is not None else "not_created"),
                )
            if attempt_directory is not None and attempt_root is not None and attempt_run_id is not None:
                try:
                    reclaimed = _cleanup_case_attempt(
                        config,
                        case_index,
                        work_directory=attempt_directory,
                        work_root=attempt_root,
                        run_id=attempt_run_id,
                        storage_root=storage,
                    )
                except BaseException as cleanup_error:  # noqa: BLE001 -- cleanup evidence must survive interruption
                    if failure_path is not None:
                        _complete_failure_cleanup(
                            failure_path,
                            status="failed",
                            reclaimed_bytes=0,
                            error=str(cleanup_error),
                        )
                    else:
                        _record_case_cleanup_failure(
                            config,
                            case_index,
                            cleanup_error,
                            work_directory=attempt_directory,
                            storage_root=storage,
                        )
                    message = f"Persistent outcome evidence exists, but marked case scratch cleanup failed: {cleanup_error}"
                    raise CaseCleanupError(message) from error
                if failure_path is not None:
                    _complete_failure_cleanup(
                        failure_path,
                        status="complete",
                        reclaimed_bytes=reclaimed,
                        error=None,
                    )
            raise
        clear_case_failure(config, case_index, storage_root=storage)
        try:
            _cleanup_case_attempt(
                config,
                case_index,
                work_directory=prepared.work_directory,
                work_root=prepared.work_root,
                run_id=prepared.workspace_run_id,
                storage_root=storage,
            )
        except BaseException as cleanup_error:
            _record_case_cleanup_failure(
                config,
                case_index,
                cleanup_error,
                work_directory=prepared.work_directory,
                storage_root=storage,
            )
            message = f"Case publication is valid, but marked scratch cleanup failed: {cleanup_error}"
            raise CaseCleanupError(message) from cleanup_error
        return CaseRunOutcome(
            status="completed",
            case_id=config.case_id(case_index),
            processed_directory=destination,
            work_directory=prepared.work_directory,
        )


def _validate_exact_batch_directory_membership(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None,
) -> None:
    """Require raw and processed roots to contain exactly intended cases."""
    expected = {config.case_id(case_index) for case_index in config.case_indices}
    for stage in ("raw", "processed"):
        root = common.paths.resolve_generated_batch_dir(config.batch_id, stage=stage, storage_root=storage_root)
        entries = tuple(root.iterdir()) if root.is_dir() else ()
        actual = {entry.name for entry in entries}
        unsafe = sorted(entry.name for entry in entries if not entry.is_dir() or entry.is_symlink())
        if actual != expected or unsafe:
            msg = (
                f"Terminal {stage} batch membership mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}, unsafe={unsafe}."
            )
            raise RuntimeError(msg)


def finalize_batch(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Validate exact membership and atomically publish the terminal manifest."""
    initialize_batch_metadata(config, storage_root=storage_root)
    records: list[dict[str, Any]] = []
    git_commits: set[str] = set()
    for case_index in config.case_indices:
        provenance = validate_completed_case(config, case_index, storage_root=storage_root)
        git_commits.add(source_service.validate_git_commit(provenance.get("git_commit")))
        success_path = processed_case_directory(config, case_index, storage_root=storage_root) / "_SUCCESS"
        records.append(
            {
                "case_index": case_index,
                "case_id": config.case_id(case_index),
                "material_family": provenance["material_family"],
                "case_input_id": provenance["case_input_id"],
                "simulation_case_id": provenance["simulation_case_id"],
                "success_sha256": common.serialization.file_sha256(success_path),
                "provenance_sha256": common.serialization.file_sha256(success_path.parent / "provenance.json"),
                "case_hdf5_sha256": common.serialization.file_sha256(success_path.parent / "case.h5"),
            }
        )
    _validate_exact_batch_directory_membership(config, storage_root=storage_root)
    if len(git_commits) != 1:
        msg = f"Completed batch contains multiple source commits: {sorted(git_commits)}."
        raise RuntimeError(msg)
    git_commit = next(iter(git_commits))
    manifest = {
        "schema_kind": "simulation_batch_manifest",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "simulation_profile": config.profile.id,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "batch_name": config.batch_name,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "git_commit": git_commit,
        "scientific_config_digest": config.scientific_config_digest,
        "template": {"relative_path": config.profile.template_relative_path, "sha256": config.template_sha256},
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
        "intended_case_indices": list(config.case_indices),
        "cases": records,
    }
    meta_directory = batch_meta_directory(config, storage_root=storage_root)
    manifest_path = _immutable_json(meta_directory / "batch_manifest.json", manifest, label="terminal batch manifest")
    success = {
        "schema_kind": "simulation_batch_success",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "manifest_sha256": common.serialization.file_sha256(manifest_path),
    }
    success_path = meta_directory / "_SUCCESS"
    if success_path.exists():
        if _load_json_object(success_path, label="batch success marker") != success:
            msg = f"Existing batch success marker disagrees with terminal manifest: {success_path}"
            raise RuntimeError(msg)
    else:
        common.serialization.atomic_write_json(success_path, success)
    return manifest_path


def validate_terminal_batch(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate a transferred or local terminal batch without mutation."""
    meta_directory = batch_meta_directory(config, storage_root=storage_root)
    manifest_path = meta_directory / "batch_manifest.json"
    success = _load_json_object(meta_directory / "_SUCCESS", label="batch success marker")
    manifest = _load_json_object(manifest_path, label="terminal batch manifest")
    if success.get("manifest_sha256") != common.serialization.file_sha256(manifest_path):
        msg = f"Terminal batch manifest digest mismatch: {manifest_path}"
        raise RuntimeError(msg)
    expected = {
        "simulation_profile": config.profile.id,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "template": {"relative_path": config.profile.template_relative_path, "sha256": config.template_sha256},
        "batch_name": config.batch_name,
        "batch_identity": config.batch_identity,
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "scientific_config_digest": config.scientific_config_digest,
        "intended_case_indices": list(config.case_indices),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        msg = f"Terminal batch scientific identity or exact membership mismatch: {manifest_path}"
        raise RuntimeError(msg)
    manifest_git_commit = source_service.validate_git_commit(manifest.get("git_commit"))
    records = manifest.get("cases")
    if not isinstance(records, list) or [record.get("case_index") for record in records if isinstance(record, dict)] != list(config.case_indices):
        msg = f"Terminal batch case records do not match intended order: {manifest_path}"
        raise RuntimeError(msg)
    _validate_exact_batch_directory_membership(config, storage_root=storage_root)
    for case_index, record in zip(config.case_indices, records, strict=True):
        provenance = validate_completed_case(config, case_index, storage_root=storage_root)
        if (
            record.get("case_input_id") != provenance["case_input_id"]
            or record.get("simulation_case_id") != provenance["simulation_case_id"]
            or provenance.get("git_commit") != manifest_git_commit
        ):
            msg = f"Terminal manifest case identities mismatch for {config.case_id(case_index)}."
            raise RuntimeError(msg)
    return manifest
