"""
===============================================================================
generation_runtime.py
===============================================================================
Run, validate, and atomically publish isolated profile-qualified COMSOL cases.
Responsibilities:
  - Build safe COMSOL argument vectors and record complete process evidence
  - Validate and collect every configured export without scientific reduction
  - Publish input snapshots and self-contained completed cases atomically
  - Validate resume integrity and terminal batch membership
Design principles:
  - Case locks and digest-bound success evidence define completion
  - Failed work directories remain inspectable unless cleanup is explicit
  - Completed destinations are never overwritten or silently repaired
This module does NOT:
  - Modify COMSOL templates or define internal model node names
  - Select training fields, normalize exports, or build datasets
===============================================================================
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src import common

from . import generation_case as case_service
from . import generation_config as config_contract
from . import generation_profiles as profiles

PUBLICATION_SCHEMA_VERSION = 1
_MIN_EXPORT_TABLE_ROWS = 2


class CaseExecutionError(RuntimeError):
    """Report one failed case while preserving its work-directory location."""

    def __init__(self, message: str, *, work_directory: Path) -> None:
        """Initialize one retained-work failure."""
        super().__init__(f"{message} Work directory retained at: {work_directory}")
        self.work_directory = work_directory


@dataclass(frozen=True, slots=True)
class CollectedExport:
    """One validated raw export and its exact-byte identity."""

    source_path: Path
    relative_path: Path
    role: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """One successful COMSOL process and validated export set."""

    prepared: case_service.PreparedCase
    command: tuple[str, ...]
    timing: dict[str, Any]
    exports: tuple[CollectedExport, ...]
    steady_flow_view: Path
    solver_log: Path
    solved_model: Path


@dataclass(frozen=True, slots=True)
class CaseRunOutcome:
    """One skipped or newly published completed case."""

    status: str
    case_id: str
    processed_directory: Path
    work_directory: Path | None


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


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
    """Return one permanent generated-input snapshot directory."""
    return common.paths.resolve_generated_batch_dir(config.batch_id, stage="raw", storage_root=storage_root) / config.case_id(case_index)


def processed_case_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one permanent self-contained completed-case directory."""
    return common.paths.resolve_generated_batch_dir(config.batch_id, stage="processed", storage_root=storage_root) / config.case_id(case_index)


def batch_meta_directory(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one batch-owned metadata directory."""
    return common.paths.get_generation_meta_root(storage_root=storage_root) / config.batch_id


def initialize_batch_metadata(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Publish or validate the immutable resolved batch configuration."""
    directory = batch_meta_directory(config, storage_root=storage_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "resolved_config.json"
    payload = {
        "schema_kind": "resolved_generation_config",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "template": {
            "relative_path": config.profile.template_relative_path,
            "sha256": config.template_sha256,
        },
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "intended_case_indices": list(config.case_indices),
        "configuration": config.values,
    }
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != serialized:
            message = f"Existing resolved batch configuration disagrees with {config.batch_id}: {path}"
            raise RuntimeError(message)
        return path
    common.serialization.atomic_write_text(path, serialized)
    return path


def resolve_comsol_executable(config: config_contract.GenerationConfig) -> str:
    """Return the configured COMSOL executable without shell parsing."""
    executable = config.values["execution"].get("executable") or os.environ.get("COMSOL_EXECUTABLE")
    if not executable:
        message = "COMSOL executable is unresolved; set execution.executable or COMSOL_EXECUTABLE."
        raise FileNotFoundError(message)
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
        message = f"cores_per_case must be a positive integer, got {cores_per_case!r}."
        raise ValueError(message)
    executable = resolve_comsol_executable(config)
    command = [
        executable,
        "batch",
        "-inputfile",
        "model.mph",
        "-outputfile",
        "solved.mph",
        "-np",
        str(cores_per_case),
        *config.values["execution"]["extra_arguments"],
    ]
    if scheduler_kind == "local":
        return command
    if scheduler_kind != "slurm":
        message = f"Unsupported scheduler kind for case execution: {scheduler_kind!r}."
        raise ValueError(message)
    hostname = node_hostname or socket.gethostname()
    if not hostname or any(character.isspace() for character in hostname):
        message = f"Cannot create a node-confined Slurm substep for hostname {hostname!r}."
        raise ValueError(message)
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
    """Require the local launcher and COMSOL executable before process creation."""
    names = list(dict.fromkeys((command[0], comsol_executable)))
    for name in names:
        candidate = Path(name).expanduser()
        if candidate.parent != Path():
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                message = f"Required executable is missing or not executable: {name}"
                raise FileNotFoundError(message)
        elif shutil.which(name) is None:
            message = f"Required executable is not available on PATH: {name}"
            raise FileNotFoundError(message)


def _parse_numeric_table(path: Path, *, delimiter: str) -> None:
    """Require one readable, rectangular table with at least one finite numeric row."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith(("%", "#"))]
    except (OSError, UnicodeDecodeError) as error:
        message = f"Configured numeric export is not readable text: {path}"
        raise ValueError(message) from error
    if not lines:
        message = f"Configured numeric export has no table rows: {path}"
        raise ValueError(message)
    rows = list(csv.reader(lines, delimiter=delimiter))
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        message = f"Configured numeric export has inconsistent row widths: {path}"
        raise ValueError(message)
    numeric_rows = 0
    for row_index, row in enumerate(rows):
        try:
            values = [float(item.strip()) for item in row]
        except ValueError:
            if row_index == 0:
                continue
            message = f"Configured numeric export contains malformed row {row_index + 1}: {path}"
            raise ValueError(message) from None
        if not all(math.isfinite(value) for value in values):
            message = f"Configured numeric export contains non-finite values: {path}"
            raise ValueError(message)
        numeric_rows += 1
    if numeric_rows == 0:
        message = f"Configured numeric export contains no numeric data rows: {path}"
        raise ValueError(message)


def collect_exports(
    config: config_contract.GenerationConfig,
    prepared: case_service.PreparedCase,
) -> tuple[CollectedExport, ...]:
    """Validate and identity-bind all configured exports without changing their bytes."""
    root = prepared.exports_directory.resolve()
    collected: dict[Path, CollectedExport] = {}
    for contract in config.values["exports"]["contracts"]:
        matches = sorted(root.glob(contract["pattern"]))
        files = [path for path in matches if path.is_file() and not path.is_symlink()]
        if contract["required"] and not files:
            message = f"Required COMSOL export pattern produced no files: {contract['pattern']!r} under {root}"
            raise FileNotFoundError(message)
        if not contract["allow_multiple"] and len(files) > 1:
            message = f"COMSOL export pattern must match at most one file: {contract['pattern']!r}"
            raise ValueError(message)
        for path in files:
            canonical = path.resolve()
            if not canonical.is_relative_to(root):
                message = f"Configured export escapes its case-owned export root: {path}"
                raise ValueError(message)
            size = path.stat().st_size
            if size <= 0:
                message = f"Configured COMSOL export is empty: {path}"
                raise ValueError(message)
            if contract["format"] == "numeric_table":
                _parse_numeric_table(path, delimiter=contract["delimiter"])
            elif contract["format"] == "text":
                try:
                    path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as error:
                    message = f"Configured COMSOL text export is unreadable: {path}"
                    raise ValueError(message) from error
            relative = canonical.relative_to(root)
            if relative in collected:
                message = f"COMSOL export {relative} matches more than one profile role."
                raise ValueError(message)
            collected[relative] = CollectedExport(
                source_path=canonical,
                relative_path=relative,
                role=contract["role"],
                sha256=common.serialization.file_sha256(canonical),
                size_bytes=size,
            )
    if not collected:
        message = f"No configured COMSOL exports were collected from {root}."
        raise FileNotFoundError(message)
    return tuple(collected[path] for path in sorted(collected, key=lambda item: item.as_posix()))


def _steady_flow_contract(config: config_contract.GenerationConfig) -> dict[str, Any]:
    """Return the configured export contract for the profile-owned airflow role."""
    role = profiles.STEADY_FLOW_EXPORT_ROLE
    for contract in config.values["exports"]["contracts"]:
        if contract["role"] == role:
            return contract
    message = f"Profile {config.profile.id!r} has no configured steady-flow export mapping."
    raise RuntimeError(message)


def _canonicalize_steady_flow_view(
    config: config_contract.GenerationConfig,
    prepared: case_service.PreparedCase,
    exports: tuple[CollectedExport, ...],
) -> Path:
    """Validate airflow fields and write one canonical static learning-view table."""
    contract = _steady_flow_contract(config)
    matches = [export for export in exports if export.role == contract["role"]]
    if len(matches) != 1:
        message = f"Profile {config.profile.id!r} requires exactly one steady-flow export, found {len(matches)}."
        raise ValueError(message)
    path = matches[0].source_path
    try:
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith(("%", "#"))]
    except (OSError, UnicodeDecodeError) as error:
        message = f"Steady-flow export is not readable text: {path}"
        raise ValueError(message) from error
    rows = list(csv.reader(lines, delimiter=contract["delimiter"]))
    if len(rows) < _MIN_EXPORT_TABLE_ROWS:
        message = f"Steady-flow export must contain a header and numeric rows: {path}"
        raise ValueError(message)
    header = [item.strip() for item in rows[0]]
    if len(header) != len(set(header)):
        message = f"Steady-flow export contains duplicate headers: {path}"
        raise ValueError(message)
    role_spec = config.profile.export_role(contract["role"])
    columns = contract["columns"]
    required_headers = [columns[name] for name in role_spec.canonical_fields]
    time_column = contract.get("time_column")
    if time_column is not None:
        required_headers.append(time_column)
    missing = [name for name in required_headers if name not in header]
    if missing:
        message = f"Steady-flow export is missing configured header(s) {missing}: {path}"
        raise ValueError(message)
    positions = {name: header.index(source_name) for name, source_name in columns.items()}
    time_position = None if time_column is None else header.index(time_column)
    numeric_rows: list[dict[str, float]] = []
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            message = f"Steady-flow export row {row_index} has the wrong width: {path}"
            raise ValueError(message)
        try:
            values = {name: float(row[position].strip()) for name, position in positions.items()}
            if time_position is not None:
                float(row[time_position].strip())
        except ValueError:
            message = f"Steady-flow export row {row_index} contains non-numeric mapped fields: {path}"
            raise ValueError(message) from None
        if not all(math.isfinite(value) for value in values.values()):
            message = f"Steady-flow export row {row_index} contains non-finite mapped fields: {path}"
            raise ValueError(message)
        numeric_rows.append(values)
    by_coordinate: dict[tuple[float, float], dict[str, float]] = {}
    tolerance = profiles.STATIONARITY_TOLERANCE
    state_fields = role_spec.canonical_fields[2:]
    for values in numeric_rows:
        coordinate = values["x"], values["y"]
        first = by_coordinate.setdefault(coordinate, values)
        if time_position is None and first is not values:
            message = f"Static steady-flow export repeats coordinate {coordinate}: {path}"
            raise ValueError(message)
        for field in state_fields:
            if not math.isclose(values[field], first[field], rel_tol=tolerance, abs_tol=tolerance):
                message = f"Supposedly stationary airflow field {field!r} varies beyond tolerance {tolerance} at coordinate {coordinate}: {path}"
                raise ValueError(message)
    if not by_coordinate:
        message = f"Steady-flow export contains no mapped numeric rows: {path}"
        raise ValueError(message)
    table = [list(role_spec.canonical_fields)]
    table.extend([format(values[field], ".17g") for field in role_spec.canonical_fields] for values in by_coordinate.values())
    destination = prepared.runtime_directory / "steady_flow_view.csv"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";", lineterminator="\n")
    writer.writerows(table)
    common.serialization.atomic_write_text(destination, stream.getvalue())
    return destination


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


def execute_prepared_case(
    config: config_contract.GenerationConfig,
    prepared: case_service.PreparedCase,
    *,
    cores_per_case: int,
    worker_slot: int,
    scheduler_kind: str = "local",
    allocated_node: str | None = None,
) -> ExecutionResult:
    """Run one isolated COMSOL process and validate its complete configured output."""
    command = build_comsol_command(
        config,
        cores_per_case=cores_per_case,
        scheduler_kind=scheduler_kind,
        node_hostname=allocated_node,
    )
    try:
        _require_executable(command, comsol_executable=resolve_comsol_executable(config))
    except FileNotFoundError as error:
        raise CaseExecutionError(str(error), work_directory=prepared.work_directory) from error
    hostname = socket.gethostname()
    started_at = _utc_now()
    monotonic_start = time.monotonic()
    stdout_path = prepared.runtime_directory / "stdout.log"
    stderr_path = prepared.runtime_directory / "stderr.log"
    process: subprocess.Popen[str] | None = None
    timed_out = False
    exit_code: int | None = None
    with (
        stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_stream,
        stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_stream,
    ):
        try:
            process = subprocess.Popen(  # noqa: S603 -- validated argument vector; no shell
                command,
                cwd=prepared.work_directory,
                stdin=subprocess.DEVNULL,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=config.values["execution"]["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    exit_code = process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    exit_code = process.wait()
        except OSError as error:
            message = f"Could not start COMSOL command {command!r}: {error}"
            raise CaseExecutionError(message, work_directory=prepared.work_directory) from error
    ended_at = _utc_now()
    elapsed = time.monotonic() - monotonic_start
    timing = {
        "schema_kind": "simulation_case_timing",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "batch_id": config.batch_id,
        "case_id": prepared.bundle.case_id,
        "case_identity": prepared.bundle.case_identity,
        "executable": resolve_comsol_executable(config),
        "arguments": command,
        "hostname": hostname,
        "process_id": None if process is None else process.pid,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "requested_cores": cores_per_case,
        "worker_slot": worker_slot,
        "allocated_node": allocated_node,
        "scheduler_kind": scheduler_kind,
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "work_directory": str(prepared.work_directory),
        "simulation_profile": config.profile.id,
        "template_sha256": config.template_sha256,
        "airflow_source": config.profile.airflow_source,
    }
    common.serialization.atomic_write_json(prepared.runtime_directory / "timing.json", timing)
    solver_log = _write_solver_log(prepared)
    if timed_out:
        message = "COMSOL case exceeded its configured timeout."
        raise CaseExecutionError(message, work_directory=prepared.work_directory)
    if exit_code != 0:
        message = f"COMSOL case exited with status {exit_code}."
        raise CaseExecutionError(message, work_directory=prepared.work_directory)
    solved_model = prepared.work_directory / "solved.mph"
    if not solved_model.is_file() or solved_model.stat().st_size <= 0:
        message = "COMSOL completed without a non-empty solved.mph output."
        raise CaseExecutionError(message, work_directory=prepared.work_directory)
    try:
        exports = collect_exports(config, prepared)
        steady_flow_view = _canonicalize_steady_flow_view(config, prepared, exports)
    except Exception as error:
        raise CaseExecutionError(str(error), work_directory=prepared.work_directory) from error
    return ExecutionResult(
        prepared=prepared,
        command=tuple(command),
        timing=timing,
        exports=exports,
        steady_flow_view=steady_flow_view,
        solver_log=solver_log,
        solved_model=solved_model,
    )


def _artifact_map(directory: Path) -> dict[str, dict[str, Any]]:
    """Return deterministic exact-byte identities for all staged payload files."""
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            message = f"Published case staging cannot contain symbolic links: {path}"
            raise ValueError(message)
        if not path.is_file() or path.name in {"provenance.json", "_SUCCESS"}:
            continue
        relative = path.relative_to(directory).as_posix()
        artifacts[relative] = {
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
    """Write digest-bound provenance and the final stage success marker."""
    artifacts = _artifact_map(directory)
    provenance = {
        "schema_kind": "simulation_case_publication",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "stage": stage,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "case_id": case_payload["case_id"],
        "case_identity": case_payload["case_identity"],
        "template_sha256": config.template_sha256,
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
        "case_identity": case_payload["case_identity"],
        "provenance_sha256": common.serialization.file_sha256(provenance_path),
    }
    common.serialization.atomic_write_json(directory / "_SUCCESS", success)


def _stage_raw_case(config: config_contract.GenerationConfig, result: ExecutionResult, destination: Path) -> None:
    """Stage one generated-input snapshot."""
    destination.mkdir(parents=True)
    shutil.copy2(result.prepared.work_directory / "case.json", destination / "case.json")
    for path in result.prepared.bundle.input_paths:
        shutil.copy2(path, destination / path.name)
    _complete_stage(destination, config=config, case_payload=result.prepared.bundle.case_payload, stage="raw")


def _stage_processed_case(config: config_contract.GenerationConfig, result: ExecutionResult, destination: Path) -> None:
    """Stage one self-contained completed case with inputs, logs, timing, and exports."""
    destination.mkdir(parents=True)
    shutil.copy2(result.prepared.work_directory / "case.json", destination / "case.json")
    for path in result.prepared.bundle.input_paths:
        shutil.copy2(path, destination / path.name)
    export_root = destination / "exports"
    for export in result.exports:
        target = export_root / export.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export.source_path, target)
        if common.serialization.file_sha256(target) != export.sha256:
            message = f"Staged export digest changed during copy: {target}"
            raise RuntimeError(message)
    view_path = destination / "learning_views" / "steady_flow" / "fields.csv"
    view_path.parent.mkdir(parents=True)
    shutil.copy2(result.steady_flow_view, view_path)
    shutil.copy2(result.solver_log, destination / "solver.log")
    shutil.copy2(result.prepared.runtime_directory / "timing.json", destination / "timing.json")
    if config.values["execution"]["retain_solved_model"]:
        shutil.copy2(result.solved_model, destination / "solved.mph")
    _complete_stage(destination, config=config, case_payload=result.prepared.bundle.case_payload, stage="processed")


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required JSON object with a contextual error."""
    if not path.is_file() or path.is_symlink():
        message = f"Missing required {label}: {path}"
        raise FileNotFoundError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not read {label}: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain a JSON object: {path}"
        raise TypeError(message)
    return value


def _validate_publication_directory(
    directory: Path,
    *,
    config: config_contract.GenerationConfig,
    case_index: int,
    stage: str,
) -> dict[str, Any]:
    """Validate one exact completed raw or processed case publication."""
    success = _load_json_object(directory / "_SUCCESS", label="case success marker")
    provenance_path = directory / "provenance.json"
    provenance = _load_json_object(provenance_path, label="case provenance")
    case_payload = _load_json_object(directory / "case.json", label="canonical case provenance")
    case_id = config.case_id(case_index)
    expected = (config.batch_id, case_id, stage)
    if (success.get("batch_id"), success.get("case_id"), success.get("stage")) != expected:
        message = f"Case success identity mismatch in {directory}."
        raise RuntimeError(message)
    if (provenance.get("batch_id"), provenance.get("case_id"), provenance.get("stage")) != expected:
        message = f"Case publication identity mismatch in {directory}."
        raise RuntimeError(message)
    if success.get("provenance_sha256") != common.serialization.file_sha256(provenance_path):
        message = f"Case provenance digest mismatch in {directory}."
        raise RuntimeError(message)
    if case_payload.get("case_identity") != success.get("case_identity") or provenance.get("case_identity") != success.get("case_identity"):
        message = f"Case identity evidence disagrees in {directory}."
        raise RuntimeError(message)
    try:
        computed_case_identity = case_service.compute_case_identity(case_payload)
    except (TypeError, ValueError) as error:
        message = f"Case identity payload is malformed in {directory}."
        raise RuntimeError(message) from error
    if computed_case_identity != case_payload["case_identity"]:
        message = f"Canonical case identity mismatch in {directory}."
        raise RuntimeError(message)
    if (
        case_payload.get("batch_identity") != config.batch_identity
        or case_payload.get("simulation_profile") != config.profile.id
        or case_payload.get("template", {}).get("sha256") != config.template_sha256
        or case_payload.get("available_learning_views") != list(config.profile.available_learning_views)
        or case_payload.get("airflow_source") != config.profile.airflow_source
    ):
        message = f"Case profile, batch, template, learning-view, or airflow identity mismatch in {directory}."
        raise RuntimeError(message)
    if (
        provenance.get("simulation_profile") != config.profile.id
        or provenance.get("available_learning_views") != list(config.profile.available_learning_views)
        or provenance.get("airflow_source") != config.profile.airflow_source
    ):
        message = f"Case publication profile metadata mismatch in {directory}."
        raise RuntimeError(message)
    if provenance.get("export_contract_sha256") != common.serialization.canonical_json_sha256(config.values["exports"]):
        message = f"Case export contract mismatch in {directory}."
        raise RuntimeError(message)
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        message = f"Case publication has no artifact identity map: {directory}"
        raise RuntimeError(message)
    actual_files = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path.name not in {"provenance.json", "_SUCCESS"}
    }
    if actual_files != set(artifacts):
        message = f"Case publication membership mismatch in {directory}."
        raise RuntimeError(message)
    for relative, identity in artifacts.items():
        path = directory / relative
        if not isinstance(identity, dict) or set(identity) != {"sha256", "size_bytes"}:
            message = f"Malformed artifact identity for {relative!r} in {directory}."
            raise RuntimeError(message)
        if path.stat().st_size != identity["size_bytes"] or common.serialization.file_sha256(path) != identity["sha256"]:
            message = f"Case artifact integrity failure for {path}."
            raise RuntimeError(message)
    input_files = case_payload.get("input_files")
    if not isinstance(input_files, dict) or not input_files:
        message = f"Canonical case provenance has no input digest map in {directory}."
        raise RuntimeError(message)
    for filename, identity in input_files.items():
        if not isinstance(filename, str) or Path(filename).name != filename or artifacts.get(filename) != identity:
            message = f"Published input digest binding mismatch for {filename!r} in {directory}."
            raise RuntimeError(message)
    return provenance


def validate_completed_case(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate raw input and processed output publications for one completed case."""
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
    if raw["case_identity"] != processed["case_identity"]:
        message = f"Raw and processed case identities disagree for {config.case_id(case_index)}."
        raise RuntimeError(message)
    return processed


def completed_case_is_valid(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> bool:
    """Return false only when a complete case is absent; corruption still fails closed."""
    raw = raw_case_directory(config, case_index, storage_root=storage_root)
    processed = processed_case_directory(config, case_index, storage_root=storage_root)
    raw_success = (raw / "_SUCCESS").exists()
    processed_success = (processed / "_SUCCESS").exists()
    if not processed_success:
        if raw_success:
            _validate_publication_directory(raw, config=config, case_index=case_index, stage="raw")
        return False
    if not raw_success:
        message = f"Processed completion exists without its raw input snapshot: {processed}"
        raise RuntimeError(message)
    validate_completed_case(config, case_index, storage_root=storage_root)
    return True


def _quarantine_incomplete(path: Path, *, state_root: Path) -> None:
    """Move one incomplete final directory into recoverable private state."""
    if not path.exists():
        return
    if (path / "_SUCCESS").exists():
        message = f"Refusing to replace an existing completed case: {path}"
        raise FileExistsError(message)
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
    """Atomically publish raw and processed case directories without overwriting completion."""
    case_index = int(result.prepared.bundle.case_payload["case_index"])
    state_root = _state_batch_root(config, storage_root=storage_root)
    publication_root = state_root / "publications"
    publication_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{result.prepared.bundle.case_id}_", dir=publication_root))
    raw_stage = staging / "raw"
    processed_stage = staging / "processed"
    raw_destination = raw_case_directory(config, case_index, storage_root=storage_root)
    processed_destination = processed_case_directory(config, case_index, storage_root=storage_root)
    try:
        _stage_raw_case(config, result, raw_stage)
        _stage_processed_case(config, result, processed_stage)
        if raw_destination.exists() and (raw_destination / "_SUCCESS").exists():
            _validate_publication_directory(raw_destination, config=config, case_index=case_index, stage="raw")
            existing_case = _load_json_object(raw_destination / "case.json", label="existing raw case")
            if existing_case.get("case_identity") != result.prepared.bundle.case_identity:
                message = f"Existing raw case belongs to another identity: {raw_destination}"
                raise RuntimeError(message)
            shutil.rmtree(raw_stage)
        else:
            _quarantine_incomplete(raw_destination, state_root=state_root)
            raw_destination.parent.mkdir(parents=True, exist_ok=True)
            raw_stage.replace(raw_destination)
        if processed_destination.exists() and (processed_destination / "_SUCCESS").exists():
            message = f"Refusing to overwrite existing completed case: {processed_destination}"
            raise FileExistsError(message)
        _quarantine_incomplete(processed_destination, state_root=state_root)
        processed_destination.parent.mkdir(parents=True, exist_ok=True)
        processed_stage.replace(processed_destination)
        validate_completed_case(config, case_index, storage_root=storage_root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return processed_destination


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
    cleanup_failed: bool = False,
    blocking_lock: bool = True,
) -> CaseRunOutcome:
    """Run or integrity-skip one case under its authoritative filesystem lock."""
    initialize_batch_metadata(config, storage_root=storage_root)
    lock_path = case_lock_path(config, case_index, storage_root=storage_root)
    with common.locking.exclusive_file_lock(lock_path, blocking=blocking_lock):
        if completed_case_is_valid(config, case_index, storage_root=storage_root):
            return CaseRunOutcome(
                status="skipped",
                case_id=config.case_id(case_index),
                processed_directory=processed_case_directory(config, case_index, storage_root=storage_root),
                work_directory=None,
            )
        prepared = case_service.prepare_case_work_directory(
            config,
            case_index,
            storage_root=storage_root,
            work_root=work_root,
        )
        try:
            result = execute_prepared_case(
                config,
                prepared,
                cores_per_case=cores_per_case,
                worker_slot=worker_slot,
                scheduler_kind=scheduler_kind,
                allocated_node=allocated_node,
            )
            destination = publish_completed_case(config, result, storage_root=storage_root)
        except BaseException:
            if cleanup_failed:
                shutil.rmtree(prepared.work_directory, ignore_errors=True)
            raise
        shutil.rmtree(prepared.work_directory)
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
    """Require raw and processed batch directories to contain exactly intended cases."""
    expected = {config.case_id(case_index) for case_index in config.case_indices}
    for stage in ("raw", "processed"):
        root = common.paths.resolve_generated_batch_dir(config.batch_id, stage=stage, storage_root=storage_root)
        entries = tuple(root.iterdir()) if root.is_dir() else ()
        actual = {entry.name for entry in entries}
        unsafe = sorted(entry.name for entry in entries if not entry.is_dir() or entry.is_symlink())
        if actual != expected or unsafe:
            message = (
                f"Terminal {stage} batch membership mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}, unsafe={unsafe}."
            )
            raise RuntimeError(message)


def finalize_batch(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Validate exact batch membership and atomically publish its terminal manifest."""
    initialize_batch_metadata(config, storage_root=storage_root)
    records: list[dict[str, Any]] = []
    for case_index in config.case_indices:
        provenance = validate_completed_case(config, case_index, storage_root=storage_root)
        success_path = processed_case_directory(config, case_index, storage_root=storage_root) / "_SUCCESS"
        records.append(
            {
                "case_index": case_index,
                "case_id": config.case_id(case_index),
                "case_identity": provenance["case_identity"],
                "success_sha256": common.serialization.file_sha256(success_path),
                "provenance_sha256": common.serialization.file_sha256(success_path.parent / "provenance.json"),
            }
        )
    _validate_exact_batch_directory_membership(config, storage_root=storage_root)
    manifest = {
        "schema_kind": "simulation_batch_manifest",
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "simulation_profile": config.profile.id,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "template": {
            "relative_path": config.profile.template_relative_path,
            "sha256": config.template_sha256,
        },
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.values["exports"]),
        "intended_case_indices": list(config.case_indices),
        "cases": records,
    }
    meta_directory = batch_meta_directory(config, storage_root=storage_root)
    manifest_path = meta_directory / "batch_manifest.json"
    serialized = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != serialized:
        message = f"Existing terminal batch manifest disagrees with validated membership: {manifest_path}"
        raise RuntimeError(message)
    if not manifest_path.exists():
        common.serialization.atomic_write_text(manifest_path, serialized)
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
        existing = _load_json_object(success_path, label="batch success marker")
        if existing != success:
            message = f"Existing batch success marker disagrees with terminal manifest: {success_path}"
            raise RuntimeError(message)
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
        message = f"Terminal batch manifest digest mismatch: {manifest_path}"
        raise RuntimeError(message)
    if (
        manifest.get("simulation_profile") != config.profile.id
        or manifest.get("available_learning_views") != list(config.profile.available_learning_views)
        or manifest.get("airflow_source") != config.profile.airflow_source
        or manifest.get("template")
        != {
            "relative_path": config.profile.template_relative_path,
            "sha256": config.template_sha256,
        }
        or manifest.get("batch_identity") != config.batch_identity
        or manifest.get("intended_case_indices") != list(config.case_indices)
    ):
        message = f"Terminal batch profile identity or exact membership mismatch: {manifest_path}"
        raise RuntimeError(message)
    records = manifest.get("cases")
    if not isinstance(records, list) or [record.get("case_index") for record in records if isinstance(record, dict)] != list(config.case_indices):
        message = f"Terminal batch case records do not match intended order: {manifest_path}"
        raise RuntimeError(message)
    _validate_exact_batch_directory_membership(config, storage_root=storage_root)
    for case_index, record in zip(config.case_indices, records, strict=True):
        provenance = validate_completed_case(config, case_index, storage_root=storage_root)
        if record.get("case_identity") != provenance["case_identity"]:
            message = f"Terminal manifest case identity mismatch for {config.case_id(case_index)}."
            raise RuntimeError(message)
    return manifest
