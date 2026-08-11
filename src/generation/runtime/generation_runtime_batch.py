"""
===============================================================================
generation_runtime_batch.py
===============================================================================
Run, admit, and atomically publish isolated profile-qualified COMSOL cases.
Responsibilities:
  - Execute safe one-node COMSOL commands and retain complete runtime evidence
  - Collect explicit raw adapters and convert them to canonical case.h5
  - Publish cases and admit terminal batches through immutable typed evidence
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
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from src import common
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_config as config_contract
from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_storage as storage_service

from . import generation_runtime_workspace as workspace_service
from .generation_runtime_preparation import PreparedCase, prepare_case_work_directory

PUBLICATION_SCHEMA_VERSION = 1
_BATCH_MANIFEST_SCHEMA_KIND: Final = "simulation_batch_manifest"
_BATCH_SUCCESS_SCHEMA_KIND: Final = "simulation_batch_success"
_CASE_PUBLICATION_SCHEMA_KIND: Final = "simulation_case_publication"
_CASE_SUCCESS_SCHEMA_KIND: Final = "simulation_case_success"
_CASE_ID_PATTERN: Final = re.compile(r"case_[0-9]{4,}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_BATCH_MANIFEST_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "status",
        "simulation_profile",
        "available_learning_views",
        "airflow_source",
        "batch_name",
        "batch_id",
        "batch_identity",
        "material_family",
        "sampling_regime",
        "git_commit",
        "scientific_config_digest",
        "template",
        "export_contract_sha256",
        "intended_case_indices",
        "cases",
    }
)
_BATCH_SUCCESS_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "simulation_profile",
        "batch_id",
        "batch_identity",
        "manifest_sha256",
    }
)
_CASE_RECORD_KEYS: Final = frozenset(
    {
        "case_index",
        "case_id",
        "material_family",
        "case_input_id",
        "simulation_case_id",
        "success_sha256",
        "provenance_sha256",
        "case_hdf5_sha256",
    }
)
_CASE_PUBLICATION_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "stage",
        "simulation_profile",
        "batch_id",
        "batch_identity",
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "material_family",
        "git_commit",
        "template_sha256",
        "scientific_config_digest",
        "export_contract_sha256",
        "available_learning_views",
        "airflow_source",
        "artifacts",
    }
)
_CASE_SUCCESS_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "stage",
        "batch_id",
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "provenance_sha256",
    }
)
CASE_FAILURE_SCHEMA_KIND = "simulation_case_failure"
CASE_FAILURE_SCHEMA_VERSION = 3
_CASE_FAILURE_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "state",
        "failure_stage",
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
        "retained_artifacts",
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
_FAILURE_STAGES = frozenset({"input", "solver", "export", "conversion", "invalid_result"})


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
        failure_stage: str = "solver",
    ) -> None:
        """Initialize one structured case-execution failure."""
        super().__init__(message)
        self.work_directory = work_directory
        self.command = command
        self.cwd = work_directory
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.missing_or_invalid_artifacts = missing_or_invalid_artifacts
        if failure_stage not in _FAILURE_STAGES:
            message = f"Unsupported case failure stage: {failure_stage!r}"
            raise ValueError(message)
        self.failure_stage = failure_stage


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
        self.failure_stage = "solver"


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


class _SolvedModelOutputError(RuntimeError):
    """Report an unsafe or ambiguous solver-produced model output."""

    def __init__(self, message: str, *, artifacts: tuple[str, ...]) -> None:
        """Initialize one solved-model output failure."""
        super().__init__(message)
        self.artifacts = artifacts


@dataclass(frozen=True, slots=True)
class _SolvedModelInventoryEntry:
    """Identify one immediate solved-model candidate without following links."""

    relative_path: str
    mode: int
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    sha256: str | None

    @property
    def is_valid(self) -> bool:
        """Return whether the candidate is a non-empty regular file."""
        return stat.S_ISREG(self.mode) and self.size_bytes > 0 and self.sha256 is not None


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

    prepared: PreparedCase
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


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """Describe one hash-validated file in a terminal case publication."""

    relative_path: str
    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return the persisted artifact-identity representation."""
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class HDF5IdentityEvidence:
    """Describe identities admitted from one canonical case HDF5 payload."""

    simulation_profile: str
    git_commit: str
    template_relative_path: str
    template_sha256: str
    case_input_id: str
    simulation_case_id: str
    scientific_config_digest: str
    export_contract_sha256: str
    available_learning_views: tuple[str, ...]
    airflow_source: str


@dataclass(frozen=True, slots=True)
class TerminalCaseEvidence:
    """Describe one completely admitted raw and processed terminal case."""

    case_index: int
    case_id: str
    material_family: str
    case_input_id: str
    simulation_case_id: str
    success_sha256: str
    provenance_sha256: str
    case_hdf5_sha256: str
    raw_directory: Path
    processed_directory: Path
    hdf5_path: Path
    raw_artifacts: tuple[ArtifactEvidence, ...]
    processed_artifacts: tuple[ArtifactEvidence, ...]
    hdf5_identity: HDF5IdentityEvidence
    _case_metadata_json: str

    def metadata_payload(self) -> dict[str, Any]:
        """Return an independent mutable copy of canonical case metadata."""
        value = json.loads(self._case_metadata_json)
        if not isinstance(value, dict):
            msg = f"Admitted metadata for {self.case_id!r} is no longer an object."
            raise TypeError(msg)
        return value

    def record_payload(self) -> dict[str, Any]:
        """Return this case's terminal-manifest record."""
        return {
            "case_index": self.case_index,
            "case_id": self.case_id,
            "material_family": self.material_family,
            "case_input_id": self.case_input_id,
            "simulation_case_id": self.simulation_case_id,
            "success_sha256": self.success_sha256,
            "provenance_sha256": self.provenance_sha256,
            "case_hdf5_sha256": self.case_hdf5_sha256,
        }

    def artifact(self, stage: str, relative_path: str) -> ArtifactEvidence:
        """Return one admitted artifact by publication stage and relative path."""
        if stage == "raw":
            artifacts = self.raw_artifacts
        elif stage == "processed":
            artifacts = self.processed_artifacts
        else:
            msg = f"Unsupported terminal publication stage: {stage!r}."
            raise ValueError(msg)
        matches = tuple(item for item in artifacts if item.relative_path == relative_path)
        if len(matches) != 1:
            msg = f"Terminal case {self.case_id!r} has no unique {stage} artifact {relative_path!r}."
            raise ValueError(msg)
        artifact = matches[0]
        if (
            not artifact.path.is_file()
            or artifact.path.is_symlink()
            or artifact.path.stat().st_size != artifact.size_bytes
            or common.serialization.file_sha256(artifact.path) != artifact.sha256
        ):
            msg = f"Admitted terminal artifact changed after admission: {artifact.path}"
            raise RuntimeError(msg)
        return artifact


@dataclass(frozen=True, slots=True)
class TerminalBatchEvidence:
    """Describe one config-independent, completely admitted terminal batch."""

    generation_root: Path
    meta_directory: Path
    raw_directory: Path
    processed_directory: Path
    manifest_path: Path
    manifest_sha256: str
    simulation_profile: str
    available_learning_views: tuple[str, ...]
    airflow_source: str
    batch_name: str
    batch_id: str
    batch_identity: str
    material_family: str
    sampling_regime: str
    git_commit: str
    scientific_config_digest: str
    template_relative_path: str
    template_sha256: str
    export_contract_sha256: str
    cases: tuple[TerminalCaseEvidence, ...]
    _scientific_config_json: str

    def case(self, case_id: str) -> TerminalCaseEvidence:
        """Return one admitted case by its terminal identifier."""
        matches = tuple(item for item in self.cases if item.case_id == case_id)
        if len(matches) != 1:
            msg = f"Terminal batch {self.batch_id!r} has no unique case {case_id!r}."
            raise ValueError(msg)
        return matches[0]

    def scientific_config_payload(self) -> dict[str, Any]:
        """Return an independent mutable copy of persisted resolved science."""
        value = json.loads(self._scientific_config_json)
        if not isinstance(value, dict):
            msg = f"Admitted scientific configuration for {self.batch_id!r} is no longer an object."
            raise TypeError(msg)
        return value

    def manifest_payload(self) -> dict[str, Any]:
        """Return the exact terminal-manifest payload represented by this evidence."""
        return {
            "schema_kind": _BATCH_MANIFEST_SCHEMA_KIND,
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "status": "complete",
            "simulation_profile": self.simulation_profile,
            "available_learning_views": list(self.available_learning_views),
            "airflow_source": self.airflow_source,
            "batch_name": self.batch_name,
            "batch_id": self.batch_id,
            "batch_identity": self.batch_identity,
            "material_family": self.material_family,
            "sampling_regime": self.sampling_regime,
            "git_commit": self.git_commit,
            "scientific_config_digest": self.scientific_config_digest,
            "template": {
                "relative_path": self.template_relative_path,
                "sha256": self.template_sha256,
            },
            "export_contract_sha256": self.export_contract_sha256,
            "intended_case_indices": [case.case_index for case in self.cases],
            "cases": [case.record_payload() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class _PublicationEvidence:
    """Hold internally validated evidence for one publication stage."""

    directory: Path
    stage: str
    case_payload: dict[str, Any]
    provenance: dict[str, Any]
    artifacts: tuple[ArtifactEvidence, ...]
    hdf5_identity: HDF5IdentityEvidence | None


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


def _comsol_parameter_arguments(
    config: config_contract.GenerationConfig,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None,
) -> list[str]:
    """Return the exact admitted transient runtime vector or no steady flags."""
    if config.profile.id == profiles.STEADY_FLOW_PROFILE:
        if scalar_handoff is not None:
            message = "Steady-flow COMSOL commands cannot receive a transient scalar handoff."
            raise ValueError(message)
        return []
    if config.profile.id != profiles.TRANSIENT_DRYING_PROFILE:
        message = f"Unsupported simulation profile for COMSOL execution: {config.profile.id!r}."
        raise ValueError(message)
    if scalar_handoff is None:
        message = "Transient COMSOL commands require one admitted scalar handoff."
        raise ValueError(message)
    if scalar_handoff.profile_id != config.profile.id:
        message = "Scalar-handoff profile identity disagrees with the generation configuration."
        raise ValueError(message)
    entries = scalar_handoff.entries
    names = tuple(entry.name for entry in entries)
    if names != profiles.TRANSIENT_SCALAR_INPUT_FIELDS:
        message = "Transient COMSOL runtime overrides do not match the canonical case-dependent contract."
        raise ValueError(message)
    values = tuple(scalar_handoff_contract.format_comsol_parameter(entry) for entry in entries)
    indices = tuple(str(index) for index in range(1, len(entries) + 1))
    return [
        "-pname",
        ",".join(names),
        "-plist",
        ",".join(values),
        "-pindex",
        ",".join(indices),
    ]


def build_comsol_command(
    config: config_contract.GenerationConfig,
    *,
    cores_per_case: int,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None = None,
    scheduler_kind: str = "local",
    node_hostname: str | None = None,
) -> list[str]:
    """Build one safe single-node COMSOL batch argument vector."""
    if isinstance(cores_per_case, bool) or not isinstance(cores_per_case, int) or cores_per_case < 1:
        msg = f"cores_per_case must be a positive integer, got {cores_per_case!r}."
        raise ValueError(msg)
    parameter_arguments = _comsol_parameter_arguments(config, scalar_handoff)
    command = [
        resolve_comsol_executable(config),
        "batch",
        "-inputfile",
        "model.mph",
        "-outputfile",
        "solved.mph",
        *parameter_arguments,
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


def _solved_model_entry(path: Path) -> _SolvedModelInventoryEntry:
    """Return a stable no-follow identity for one solved-model candidate."""
    try:
        before = path.lstat()
        digest = common.serialization.file_sha256(path) if stat.S_ISREG(before.st_mode) and before.st_size > 0 else None
        after = path.lstat()
    except OSError as error:
        message = f"Could not inspect solver-produced model candidate {path.name!r}: {error}"
        raise _SolvedModelOutputError(message, artifacts=(path.name,)) from error
    before_identity = (
        before.st_mode,
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_mode,
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        message = f"Solver-produced model candidate changed during inspection: {path.name!r}."
        raise _SolvedModelOutputError(message, artifacts=(path.name,))
    return _SolvedModelInventoryEntry(
        relative_path=path.name,
        mode=after.st_mode,
        device=after.st_dev,
        inode=after.st_ino,
        size_bytes=after.st_size,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
        sha256=digest,
    )


def _solved_model_inventory(work_directory: Path) -> dict[str, _SolvedModelInventoryEntry]:
    """Inventory immediate solved*.mph candidates without trusting stale files."""
    inventory: dict[str, _SolvedModelInventoryEntry] = {}
    for candidate in sorted(work_directory.glob("solved*.mph"), key=lambda item: item.name):
        entry = _solved_model_entry(candidate)
        inventory[entry.relative_path] = entry
    return inventory


def _canonicalize_solved_model(
    work_directory: Path,
    before: Mapping[str, _SolvedModelInventoryEntry],
) -> tuple[Path, dict[str, Any]]:
    """Admit exactly one new or replaced solved output and canonicalize its name."""
    after = _solved_model_inventory(work_directory)
    changed = tuple(entry for name, entry in sorted(after.items()) if before.get(name) != entry)
    invalid = tuple(entry.relative_path for entry in changed if not entry.is_valid)
    if invalid:
        message = "COMSOL produced unsafe or empty solved-model candidate(s): " + ", ".join(invalid)
        raise _SolvedModelOutputError(message, artifacts=invalid)
    if len(changed) != 1:
        names = tuple(entry.relative_path for entry in changed)
        observed = "none" if not names else ", ".join(names)
        message = f"COMSOL must produce exactly one new or replaced non-empty solved*.mph candidate; observed {observed}."
        raise _SolvedModelOutputError(message, artifacts=names or ("solved*.mph",))

    admitted = changed[0]
    observed_path = work_directory / admitted.relative_path
    canonical_path = work_directory / "solved.mph"
    canonicalized = admitted.relative_path != canonical_path.name
    if canonicalized:
        try:
            observed_path.replace(canonical_path)
        except OSError as error:
            message = f"Could not atomically canonicalize {admitted.relative_path!r} as {canonical_path.name!r}: {error}"
            raise _SolvedModelOutputError(
                message,
                artifacts=(admitted.relative_path, canonical_path.name),
            ) from error
    try:
        canonical_stat = canonical_path.lstat()
        canonical_parent = canonical_path.resolve(strict=True).parent
    except OSError as error:
        message = f"Could not revalidate canonical solved.mph: {error}"
        raise _SolvedModelOutputError(message, artifacts=(canonical_path.name,)) from error
    expected_identity = (
        admitted.mode,
        admitted.device,
        admitted.inode,
        admitted.size_bytes,
        admitted.modified_ns,
    )
    canonical_identity = (
        canonical_stat.st_mode,
        canonical_stat.st_dev,
        canonical_stat.st_ino,
        canonical_stat.st_size,
        canonical_stat.st_mtime_ns,
    )
    if (
        not stat.S_ISREG(canonical_stat.st_mode)
        or canonical_stat.st_size <= 0
        or canonical_parent != work_directory.resolve()
        or canonical_identity != expected_identity
    ):
        message = "Canonical solved.mph does not preserve the admitted in-workspace solver output identity."
        raise _SolvedModelOutputError(message, artifacts=(canonical_path.name,))
    disposition = "new" if admitted.relative_path not in before else "replaced"
    evidence = {
        "requested_relative_path": "solved.mph",
        "observed_relative_path": admitted.relative_path,
        "canonical_relative_path": canonical_path.name,
        "disposition": disposition,
        "canonicalized": canonicalized,
        "size_bytes": admitted.size_bytes,
        "sha256": admitted.sha256,
    }
    return canonical_path, evidence


def _record_solved_model_provenance(path: Path, evidence: Mapping[str, Any]) -> None:
    """Attach canonical solved-model identity to completed execution evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    scalar_handoff = payload.get("scalar_handoff") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict) or not isinstance(scalar_handoff, dict):
        message = f"Execution provenance became malformed before output admission: {path}"
        raise TypeError(message)
    payload["result"]["solved_model"] = dict(evidence)
    scalar_handoff["original_comsol_output_filename"] = evidence["observed_relative_path"]
    scalar_handoff["canonical_solved_model_filename"] = evidence["canonical_relative_path"]
    common.serialization.atomic_write_json(path, payload)


def collect_exports(
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
) -> tuple[CollectedExport, ...]:
    """Collect explicit mappings and reject unresolved required roles."""
    root = prepared.exports_directory.resolve()
    collected: dict[Path, CollectedExport] = {}
    for contract in config.scientific_values["output_contract"]["exports"]:
        role = str(contract["role"])
        if contract["mapping_state"] == "mapping_probe_required":
            if contract["required"]:
                message = f"Required profile mapping for {role!r} still requires a mapping probe."
                raise RuntimeError(message)
            continue
        pattern = contract.get("pattern")
        if not isinstance(pattern, str):
            message = f"Profile mapping for {role!r} has no declared source filename."
            raise TypeError(message)
        matches = sorted(root.glob(pattern))
        files = [candidate for candidate in matches if candidate.is_file() and not candidate.is_symlink()]
        if contract["required"] and not files:
            message = f"Required COMSOL export pattern produced no files: {pattern!r} under {root}"
            raise FileNotFoundError(message)
        if not contract["allow_multiple"] and len(files) > 1:
            message = f"COMSOL export pattern must match at most one file: {pattern!r}"
            raise ValueError(message)
        for candidate in files:
            canonical = candidate.resolve()
            if not canonical.is_relative_to(root):
                message = f"Configured export escapes its case-owned root: {candidate}"
                raise ValueError(message)
            size = canonical.stat().st_size
            if size <= 0:
                message = f"Configured COMSOL export is empty: {canonical}"
                raise ValueError(message)
            relative = canonical.relative_to(root)
            if relative in collected:
                message = f"COMSOL export {relative} matches more than one configured role."
                raise ValueError(message)
            collected[relative] = CollectedExport(
                source_path=canonical,
                relative_path=relative,
                role=role,
                sha256=common.serialization.file_sha256(canonical),
                size_bytes=size,
            )
    if not collected:
        message = f"No configured COMSOL exports were collected from {root}."
        raise FileNotFoundError(message)
    return tuple(
        collected[candidate]
        for candidate in sorted(
            collected,
            key=lambda item: item.as_posix(),
        )
    )


def _write_solver_log(prepared: PreparedCase) -> Path:
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


def _scalar_handoff_execution_provenance(
    config: config_contract.GenerationConfig,
    scalar_handoff: scalar_handoff_contract.ScalarHandoffAdmission | None,
) -> dict[str, Any]:
    """Return explicit runtime-binding evidence for steady or transient execution."""
    output_names = {
        "original_comsol_output_filename": None,
        "canonical_solved_model_filename": None,
    }
    if config.profile.id == profiles.STEADY_FLOW_PROFILE:
        if scalar_handoff is not None:
            message = "Steady execution provenance cannot receive a transient scalar handoff."
            raise ValueError(message)
        return {
            "state": "not_applicable",
            "mechanism": "parameter_free",
            "reason": "steady_flow_has_no_transient_scalar_runtime_overrides",
            **output_names,
        }
    if scalar_handoff is None:
        message = "Transient execution provenance requires an admitted scalar handoff."
        raise ValueError(message)
    runtime_entries = scalar_handoff.entries
    payload = scalar_handoff.provenance_payload(include_source_path=True)
    payload.update(
        {
            "state": "applied",
            "mechanism": "comsol_cli_pname_plist",
            "runtime_override_names": [entry.name for entry in runtime_entries],
            "runtime_override_values": [entry.value for entry in runtime_entries],
            "formatted_plist_expressions": [scalar_handoff_contract.format_comsol_parameter(entry) for entry in runtime_entries],
            "pindex_values": list(range(1, len(runtime_entries) + 1)),
            **output_names,
        }
    )
    return payload


def _execution_provenance(
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
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
        "schema_version": 2,
        "case_id": prepared.bundle.case_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "execution_config_digest": execution_digest,
        "git_commit": prepared.bundle.case_payload["git_commit"],
        "execution_config": config.execution_values,
        "scalar_handoff": _scalar_handoff_execution_provenance(
            config,
            prepared.bundle.scalar_handoff,
        ),
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
            "solved_model": None,
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
        "solved_model": None,
    }
    common.serialization.atomic_write_json(path, payload)


def _solver_environment(prepared: PreparedCase) -> dict[str, str]:
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
    prepared: PreparedCase,
    *,
    cores_per_case: int,
    worker_slot: int,
    scheduler_kind: str = "local",
    allocated_node: str | None = None,
) -> ExecutionResult:
    """Run one isolated COMSOL process and create its validated canonical HDF5."""
    scalar_handoff = prepared.bundle.scalar_handoff
    if scalar_handoff is not None:
        scalar_handoff_contract.validate_transient_scalar_source(scalar_handoff)
    command = build_comsol_command(
        config,
        cores_per_case=cores_per_case,
        scalar_handoff=scalar_handoff,
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
    if runtime_cancellation_requested():
        message = "Campaign cancellation was requested before COMSOL launch."
        raise CaseInterruptedError(
            message,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=None,
        )
    try:
        solved_models_before = _solved_model_inventory(prepared.work_directory)
    except _SolvedModelOutputError as error:
        raise CaseExecutionError(
            str(error),
            work_directory=prepared.work_directory,
            command=tuple(command),
            missing_or_invalid_artifacts=error.artifacts,
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
    try:
        solved_model, solved_model_evidence = _canonicalize_solved_model(
            prepared.work_directory,
            solved_models_before,
        )
        _record_solved_model_provenance(execution_provenance, solved_model_evidence)
    except (_SolvedModelOutputError, OSError, TypeError, ValueError) as error:
        artifacts = error.artifacts if isinstance(error, _SolvedModelOutputError) else ("solved.mph",)
        raise CaseExecutionError(
            str(error),
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            missing_or_invalid_artifacts=artifacts,
        ) from error
    try:
        exports = collect_exports(config, prepared)
    except Exception as error:
        raise CaseExecutionError(
            str(error),
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            missing_or_invalid_artifacts=(str(error),),
            failure_stage="export",
        ) from error
    try:
        canonical_case = storage_service.convert_exports_to_hdf5(
            config,
            prepared.bundle.case_payload,
            exports,
            scalar_handoff=prepared.bundle.scalar_handoff,
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
            failure_stage="conversion",
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


def case_failure_artifacts_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the private compact failure-artifact directory for one case."""
    return _state_batch_root(config, storage_root=storage_root) / "failure_artifacts" / config.case_id(case_index)


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
        role = str(contract["role"])
        pattern = contract.get("pattern")
        if contract["mapping_state"] == "mapping_probe_required":
            if contract["required"]:
                missing.add(f"export:{role}:mapping_probe_required")
            continue
        if not isinstance(pattern, str) or not pattern:
            missing.add(f"export:{role}:unresolved_mapping")
            continue
        matches = [
            candidate
            for candidate in exports_root.glob(pattern)
            if candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_size > 0
        ]
        if contract["required"] and not matches:
            missing.add(f"export:{role}")
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


def _failure_export_inventory(
    config: config_contract.GenerationConfig,
    work_directory: Path,
) -> list[dict[str, Any]]:
    """Return hashes and sizes for available exports without copying large payloads."""
    exports_root = work_directory / config.scientific_values["output_contract"]["exports_root"]
    if not exports_root.is_dir() or exports_root.is_symlink():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(exports_root.rglob("*")):
        if path.is_symlink():
            message = f"Failure export inventory contains a symbolic link: {path}"
            raise ValueError(message)
        if path.is_file():
            records.append(
                {
                    "relative_path": path.relative_to(work_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": common.serialization.file_sha256(path),
                }
            )
    return records


def _retain_failure_artifacts(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    work_directory: Path | None,
    storage_root: Path,
) -> dict[str, dict[str, Any]]:
    """Retain compact pilot diagnostics before marked scratch is removed."""
    if config.scientific_values.get("campaign_purpose") != config_contract.PILOT_CAMPAIGN_PURPOSE or work_directory is None:
        return {}
    target = case_failure_artifacts_directory(
        config,
        case_index,
        storage_root=storage_root,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_dir() or target.is_symlink() or not target.resolve().is_relative_to(target.parent.resolve()):
            message = f"Pilot failure artifact target is unsafe: {target}"
            raise ValueError(message)
        shutil.rmtree(target)
    staging = Path(tempfile.mkdtemp(prefix=f"{config.case_id(case_index)}.", dir=target.parent))
    retained_paths = (
        Path("case.json"),
        Path("fields.csv"),
        Path("scalars.csv"),
        Path("schedule.csv"),
        Path("runtime/solver.log"),
        Path("runtime/stdout.log"),
        Path("runtime/stderr.log"),
        Path("runtime/execution_provenance.json"),
        Path("runtime/timing.json"),
        Path("runtime/status.json"),
    )
    try:
        for relative in retained_paths:
            source = work_directory / relative
            if not source.is_file() or source.is_symlink():
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        export_inventory = _failure_export_inventory(config, work_directory)
        common.serialization.atomic_write_json(
            staging / "export_inventory.json",
            {
                "schema_kind": "simulation_case_failure_export_inventory",
                "schema_version": 1,
                "exports": export_inventory,
            },
        )
        records = {
            path.relative_to(staging).as_posix(): {
                "sha256": common.serialization.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        staging.replace(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return records


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
    failure_stage: str,
) -> Path:
    """Persist compact failure evidence before node-local scratch cleanup."""
    if failure_stage not in _FAILURE_STAGES:
        message = f"Unsupported case failure stage: {failure_stage!r}"
        raise ValueError(message)
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
        "failure_stage": failure_stage,
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
        "retained_artifacts": _retain_failure_artifacts(
            config,
            case_index,
            work_directory=work_directory,
            storage_root=storage,
        ),
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
    artifacts = case_failure_artifacts_directory(
        config,
        case_index,
        storage_root=storage_root,
    )
    if artifacts.exists():
        if not artifacts.is_dir() or artifacts.is_symlink():
            message = f"Case failure artifact path is unsafe: {artifacts}"
            raise ValueError(message)
        shutil.rmtree(artifacts)


def _validate_failure_retained_artifacts(
    config: config_contract.GenerationConfig,
    case_index: int,
    retained: Any,
    *,
    storage_root: Path | str | None,
    failure_path: Path,
) -> None:
    """Validate exact compact artifact membership for one failed case."""
    artifacts_root = case_failure_artifacts_directory(
        config,
        case_index,
        storage_root=storage_root,
    )
    if not isinstance(retained, dict):
        message = f"Case failure retained-artifact evidence is invalid: {failure_path}"
        raise TypeError(message)
    if not retained:
        if artifacts_root.exists():
            message = f"Undeclared case failure artifacts exist: {artifacts_root}"
            raise ValueError(message)
        return
    if not artifacts_root.is_dir() or artifacts_root.is_symlink():
        message = f"Case failure retained artifacts are missing or unsafe: {artifacts_root}"
        raise ValueError(message)
    actual_paths: set[str] = set()
    for artifact in artifacts_root.rglob("*"):
        if artifact.is_symlink():
            message = f"Case failure retained artifacts contain a symbolic link: {artifact}"
            raise ValueError(message)
        if artifact.is_file():
            actual_paths.add(artifact.relative_to(artifacts_root).as_posix())
    if actual_paths != set(retained):
        message = f"Case failure retained-artifact membership changed: {artifacts_root}"
        raise ValueError(message)
    for relative_value, identity in retained.items():
        relative = Path(relative_value)
        artifact = (artifacts_root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not artifact.is_relative_to(artifacts_root.resolve())
            or not isinstance(identity, dict)
            or set(identity) != {"sha256", "size_bytes"}
            or artifact.stat().st_size != identity["size_bytes"]
            or common.serialization.file_sha256(artifact) != identity["sha256"]
        ):
            message = f"Case failure retained-artifact identity is invalid: {artifact}"
            raise ValueError(message)


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
        or payload.get("failure_stage") not in _FAILURE_STAGES
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
    _validate_failure_retained_artifacts(
        config,
        case_index,
        payload.get("retained_artifacts"),
        storage_root=storage_root,
        failure_path=path,
    )
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


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    """Return one deterministic compact JSON representation."""
    return json.dumps(dict(payload), ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _require_sha256(value: object, *, label: str) -> str:
    """Return one lowercase SHA-256 digest or fail closed."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} must be one lowercase SHA-256 digest."
        raise ValueError(msg)
    return value


def _safe_file_sha256(path: Path, *, label: str) -> str:
    """Return the digest of one required non-symlink regular file."""
    if not path.is_file() or path.is_symlink():
        msg = f"Missing or unsafe {label}: {path}"
        raise FileNotFoundError(msg)
    return common.serialization.file_sha256(path)


def _admit_artifacts(directory: Path, provenance: Mapping[str, Any]) -> tuple[ArtifactEvidence, ...]:
    """Admit exact publication membership and every declared artifact hash."""
    raw_artifacts = provenance.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        msg = f"Case publication has no artifact identity map: {directory}"
        raise RuntimeError(msg)
    artifacts: list[ArtifactEvidence] = []
    declared: set[str] = set()
    for relative, raw_identity in raw_artifacts.items():
        relative_path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or not isinstance(raw_identity, dict)
            or set(raw_identity) != {"sha256", "size_bytes"}
        ):
            msg = f"Malformed artifact identity for {relative!r} in {directory}."
            raise RuntimeError(msg)
        digest = _require_sha256(raw_identity["sha256"], label=f"artifact {relative!r} sha256")
        size_bytes = raw_identity["size_bytes"]
        artifact_path = directory / relative_path
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            msg = f"Artifact {relative!r} has an invalid byte count in {directory}."
            raise RuntimeError(msg)
        if (
            not artifact_path.is_file()
            or artifact_path.is_symlink()
            or artifact_path.stat().st_size != size_bytes
            or common.serialization.file_sha256(artifact_path) != digest
        ):
            msg = f"Case artifact integrity failure for {artifact_path}."
            raise RuntimeError(msg)
        declared.add(relative)
        artifacts.append(
            ArtifactEvidence(
                relative_path=relative,
                path=artifact_path.resolve(),
                sha256=digest,
                size_bytes=size_bytes,
            )
        )
    unsafe = tuple(path for path in directory.rglob("*") if path.is_symlink())
    if unsafe:
        msg = f"Case publication contains symbolic links: {[str(path) for path in unsafe]}"
        raise RuntimeError(msg)
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path.name not in {"provenance.json", "_SUCCESS"}
    }
    if actual != declared:
        msg = f"Case publication membership mismatch in {directory}: missing={sorted(declared - actual)}, extra={sorted(actual - declared)}."
        raise RuntimeError(msg)
    return tuple(sorted(artifacts, key=lambda artifact: artifact.relative_path))


def _hdf5_evidence(identity: Mapping[str, Any]) -> HDF5IdentityEvidence:
    """Normalize canonical storage validation into immutable typed evidence."""
    views = identity.get("available_learning_views")
    if not isinstance(views, tuple) or not all(isinstance(item, str) for item in views):
        msg = "Canonical HDF5 validation returned malformed learning-view evidence."
        raise RuntimeError(msg)
    return HDF5IdentityEvidence(
        simulation_profile=str(identity["simulation_profile"]),
        git_commit=str(identity["git_commit"]),
        template_relative_path=str(identity["template_relative_path"]),
        template_sha256=_require_sha256(identity["template_sha256"], label="HDF5 template_sha256"),
        case_input_id=_require_sha256(identity["case_input_id"], label="HDF5 case_input_id"),
        simulation_case_id=_require_sha256(identity["simulation_case_id"], label="HDF5 simulation_case_id"),
        scientific_config_digest=_require_sha256(
            identity["scientific_config_digest"],
            label="HDF5 scientific_config_digest",
        ),
        export_contract_sha256=_require_sha256(
            identity["export_contract_sha256"],
            label="HDF5 export_contract_sha256",
        ),
        available_learning_views=views,
        airflow_source=str(identity["airflow_source"]),
    )


def _admit_publication_directory(directory: Path, *, stage: str) -> _PublicationEvidence:
    """Admit one raw or processed case publication by producer-owned contracts."""
    if stage not in {"raw", "processed"}:
        msg = f"Unsupported case publication stage: {stage!r}."
        raise ValueError(msg)
    if not directory.is_dir() or directory.is_symlink():
        msg = f"Case publication directory is missing or unsafe: {directory}"
        raise FileNotFoundError(msg)
    success_path = directory / "_SUCCESS"
    provenance_path = directory / "provenance.json"
    success = _load_json_object(success_path, label=f"{stage} case success marker")
    provenance = _load_json_object(provenance_path, label=f"{stage} case publication provenance")
    case_payload = _load_json_object(directory / "case.json", label=f"{stage} canonical case provenance")
    if (
        set(success) != _CASE_SUCCESS_KEYS
        or success.get("schema_kind") != _CASE_SUCCESS_SCHEMA_KIND
        or success.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        or set(provenance) != _CASE_PUBLICATION_KEYS
        or provenance.get("schema_kind") != _CASE_PUBLICATION_SCHEMA_KIND
        or provenance.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        or case_payload.get("schema_kind") != case_service.CASE_SCHEMA_KIND
        or case_payload.get("schema_version") != case_service.CASE_SCHEMA_VERSION
    ):
        msg = f"Case publication schema is not current: {directory}"
        raise RuntimeError(msg)
    case_id = case_payload.get("case_id")
    case_index = case_payload.get("case_index")
    if (
        not isinstance(case_id, str)
        or _CASE_ID_PATTERN.fullmatch(case_id) is None
        or isinstance(case_index, bool)
        or not isinstance(case_index, int)
        or case_index < 1
        or case_id != f"case_{case_index:04d}"
    ):
        msg = f"Canonical case identifier or index is malformed: {directory}"
        raise RuntimeError(msg)
    batch_id = common.paths.validate_logical_name(case_payload.get("batch_id"), label="batch_id")
    batch_identity = _require_sha256(case_payload.get("batch_identity"), label="case batch_identity")
    scientific_digest = _require_sha256(
        case_payload.get("scientific_config_digest"),
        label="case scientific_config_digest",
    )
    if batch_identity != scientific_digest:
        msg = f"Canonical case batch and scientific identities disagree: {directory}"
        raise RuntimeError(msg)
    _require_sha256(case_payload.get("case_input_config_digest"), label="case_input_config_digest")
    case_input_id = _require_sha256(case_payload.get("case_input_id"), label="case_input_id")
    simulation_case_id = _require_sha256(case_payload.get("simulation_case_id"), label="simulation_case_id")
    export_contract_sha256 = _require_sha256(
        case_payload.get("export_contract_sha256"),
        label="case export_contract_sha256",
    )
    git_commit = source_service.validate_git_commit(case_payload.get("git_commit"))
    profile_id = case_payload.get("simulation_profile")
    if not isinstance(profile_id, str):
        msg = f"Canonical case simulation_profile is malformed: {directory}"
        raise TypeError(msg)
    profile = profiles.resolve_profile(profile_id)
    views = case_payload.get("available_learning_views")
    template = case_payload.get("template")
    if not isinstance(template, dict) or set(template) != {"relative_path", "filename", "sha256"}:
        msg = f"Canonical case template descriptor is malformed: {directory}"
        raise RuntimeError(msg)
    template_sha256 = _require_sha256(template["sha256"], label="case template sha256")
    if (
        views != list(profile.available_learning_views)
        or case_payload.get("airflow_source") != profile.airflow_source
        or template["relative_path"] != profile.template_relative_path
        or template["filename"] != Path(profile.template_relative_path).name
    ):
        msg = f"Canonical case profile or template descriptor is invalid: {directory}"
        raise RuntimeError(msg)
    material_family = materials.validate_material_family(case_payload.get("material_family"))
    if case_service.compute_case_input_id(case_payload) != case_input_id:
        msg = f"Canonical case-input identity mismatch in {directory}."
        raise RuntimeError(msg)
    if case_service.compute_simulation_case_id(case_payload) != simulation_case_id:
        msg = f"Canonical simulation-case identity mismatch in {directory}."
        raise RuntimeError(msg)
    expected_publication = {
        "stage": stage,
        "simulation_profile": profile.id,
        "batch_id": batch_id,
        "batch_identity": batch_identity,
        "case_id": case_id,
        "case_input_id": case_input_id,
        "simulation_case_id": simulation_case_id,
        "material_family": material_family,
        "git_commit": git_commit,
        "template_sha256": template_sha256,
        "scientific_config_digest": scientific_digest,
        "export_contract_sha256": export_contract_sha256,
        "available_learning_views": list(profile.available_learning_views),
        "airflow_source": profile.airflow_source,
    }
    if any(provenance.get(key) != value for key, value in expected_publication.items()):
        msg = f"Case publication identity mismatch in {directory}."
        raise RuntimeError(msg)
    expected_success = {
        "stage": stage,
        "batch_id": batch_id,
        "case_id": case_id,
        "case_input_id": case_input_id,
        "simulation_case_id": simulation_case_id,
    }
    if any(success.get(key) != value for key, value in expected_success.items()):
        msg = f"Case success identity mismatch in {directory}."
        raise RuntimeError(msg)
    provenance_sha256 = _safe_file_sha256(provenance_path, label="case publication provenance")
    if success.get("provenance_sha256") != provenance_sha256:
        msg = f"Case provenance digest mismatch in {directory}."
        raise RuntimeError(msg)
    artifacts = _admit_artifacts(directory, provenance)
    hdf5_identity: HDF5IdentityEvidence | None = None
    if stage == "processed":
        required = {"case.h5", "solver.log", "timing.json", "status.json", "execution_provenance.json", "case.json"}
        artifact_names = {artifact.relative_path for artifact in artifacts}
        if not required.issubset(artifact_names):
            msg = f"Processed publication lacks canonical payload or runtime evidence: {directory}"
            raise RuntimeError(msg)
        timing = _load_json_object(directory / "timing.json", label="case timing")
        execution = _load_json_object(directory / "execution_provenance.json", label="case execution provenance")
        if timing.get("git_commit") != git_commit or execution.get("git_commit") != git_commit:
            msg = f"Processed runtime Git-commit evidence disagrees in {directory}."
            raise RuntimeError(msg)
        hdf5_identity = _hdf5_evidence(
            storage_service.validate_case_hdf5(
                directory / "case.h5",
                expected_profile=profile.id,
            )
        )
        expected_hdf5 = HDF5IdentityEvidence(
            simulation_profile=profile.id,
            git_commit=git_commit,
            template_relative_path=profile.template_relative_path,
            template_sha256=template_sha256,
            case_input_id=case_input_id,
            simulation_case_id=simulation_case_id,
            scientific_config_digest=scientific_digest,
            export_contract_sha256=export_contract_sha256,
            available_learning_views=profile.available_learning_views,
            airflow_source=profile.airflow_source,
        )
        if hdf5_identity != expected_hdf5:
            msg = f"Canonical HDF5 identities disagree with case publication in {directory}."
            raise RuntimeError(msg)
    return _PublicationEvidence(
        directory=directory.resolve(),
        stage=stage,
        case_payload=case_payload,
        provenance=provenance,
        artifacts=artifacts,
        hdf5_identity=hdf5_identity,
    )


def _require_case_payload_matches_config(
    payload: Mapping[str, Any],
    *,
    directory: Path,
    config: config_contract.GenerationConfig,
    case_index: int,
) -> None:
    """Require admitted case metadata to match one authored configuration."""
    expected = {
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "scientific_config_digest": config.scientific_config_digest,
        "case_input_config_digest": config.case_input_config_digest,
        "case_id": config.case_id(case_index),
        "case_index": case_index,
        "material_family": config.material_family,
        "material_role": config.material_role,
        "evaluation_regime": config.evaluation_regime,
        "sampling_regime": config.sampling_regime,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "template": {
            "relative_path": config.profile.template_relative_path,
            "filename": config.template_path.name,
            "sha256": config.template_sha256,
        },
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        msg = f"Case scientific identity disagrees with authored configuration: {directory}"
        raise RuntimeError(msg)


def _require_publication_matches_config(
    evidence: _PublicationEvidence,
    *,
    config: config_contract.GenerationConfig,
    case_index: int,
) -> None:
    """Layer exact authored-configuration expectations over admitted evidence."""
    _require_case_payload_matches_config(
        evidence.case_payload,
        directory=evidence.directory,
        config=config,
        case_index=case_index,
    )


def validate_completed_case(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate one completed case and layer exact authored-config comparison."""
    raw = _admit_publication_directory(
        raw_case_directory(config, case_index, storage_root=storage_root),
        stage="raw",
    )
    processed = _admit_publication_directory(
        processed_case_directory(config, case_index, storage_root=storage_root),
        stage="processed",
    )
    _require_publication_matches_config(raw, config=config, case_index=case_index)
    _require_publication_matches_config(processed, config=config, case_index=case_index)
    if raw.case_payload != processed.case_payload:
        msg = f"Raw and processed canonical metadata disagree for {config.case_id(case_index)}."
        raise RuntimeError(msg)
    return dict(processed.provenance)


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
            evidence = _admit_publication_directory(raw, stage="raw")
            _require_publication_matches_config(evidence, config=config, case_index=case_index)
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
            existing = _admit_publication_directory(raw_destination, stage="raw")
            _require_publication_matches_config(existing, config=config, case_index=case_index)
            if existing.provenance["simulation_case_id"] != result.prepared.bundle.simulation_case_id:
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
    prepared: PreparedCase | None,
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
        prepared: PreparedCase | None = None
        failure_stage = "input"
        try:
            prepared = prepare_case_work_directory(
                config,
                case_index,
                storage_root=storage,
                work_root=work_root,
            )
            failure_stage = "solver"
            result = execute_prepared_case(
                config,
                prepared,
                cores_per_case=cores_per_case,
                worker_slot=worker_slot,
                scheduler_kind=scheduler_kind,
                allocated_node=allocated_node,
            )
            failure_stage = "invalid_result"
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
                    failure_stage=str(getattr(error, "failure_stage", failure_stage)),
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
    batch_id: str,
    case_ids: tuple[str, ...],
    *,
    storage_root: Path | str | None,
) -> tuple[Path, Path]:
    """Require raw and processed roots to contain exactly intended cases."""
    expected = set(case_ids)
    roots: list[Path] = []
    for stage in ("raw", "processed"):
        root = common.paths.resolve_generated_batch_dir(batch_id, stage=stage, storage_root=storage_root)
        entries = tuple(root.iterdir()) if root.is_dir() and not root.is_symlink() else ()
        actual = {entry.name for entry in entries}
        unsafe = sorted(entry.name for entry in entries if not entry.is_dir() or entry.is_symlink())
        if actual != expected or unsafe:
            msg = (
                f"Terminal {stage} batch membership mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}, unsafe={unsafe}."
            )
            raise RuntimeError(msg)
        roots.append(root.resolve())
    return roots[0], roots[1]


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
    _validate_exact_batch_directory_membership(
        config.batch_id,
        tuple(config.case_id(case_index) for case_index in config.case_indices),
        storage_root=storage_root,
    )
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
    validate_terminal_batch(config, storage_root=storage_root)
    return manifest_path


def _validate_terminal_scientific_config(
    scientific: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    case_count: int,
) -> None:
    """Bind persisted resolved science to terminal profile and batch descriptors."""
    material = scientific.get("material")
    output_contract = scientific.get("output_contract")
    assignments = scientific.get("assignments")
    if (
        scientific.get("schema_kind") != "resolved_generation_batch"
        or scientific.get("schema_version") != config_contract.CONFIG_SCHEMA_VERSION
        or scientific.get("simulation_profile") != manifest["simulation_profile"]
        or scientific.get("sampling_regime") != manifest["sampling_regime"]
        or scientific.get("case_count") != case_count
        or not isinstance(material, Mapping)
        or material.get("material_family") != manifest["material_family"]
        or not isinstance(output_contract, Mapping)
        or common.serialization.canonical_json_sha256(output_contract) != manifest["export_contract_sha256"]
        or scientific.get("available_learning_views") != manifest["available_learning_views"]
        or scientific.get("airflow_source") != manifest["airflow_source"]
        or not isinstance(assignments, list)
        or len(assignments) != case_count
    ):
        msg = f"Persisted resolved science disagrees with terminal batch descriptors: {manifest_path}"
        raise RuntimeError(msg)


def _require_case_matches_terminal(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    scientific: Mapping[str, Any],
    record: Mapping[str, Any],
    directory: Path,
) -> None:
    """Bind one internally valid case to terminal and persisted-science evidence."""
    assignments = scientific.get("assignments")
    matches = (
        [assignment for assignment in assignments if isinstance(assignment, Mapping) and assignment.get("case_index") == record["case_index"]]
        if isinstance(assignments, list)
        else []
    )
    assignment = matches[0] if len(matches) == 1 else {}
    natural_support_state = (
        "nominal_reference" if assignment.get("pilot_case_kind") == "nominal_reference" else scientific.get("natural_support_state")
    )
    expected = {
        "simulation_profile": manifest["simulation_profile"],
        "batch_id": manifest["batch_id"],
        "batch_identity": manifest["batch_identity"],
        "scientific_config_digest": manifest["scientific_config_digest"],
        "case_input_config_digest": config_contract.compute_case_input_config_digest(scientific),
        "case_id": record["case_id"],
        "case_index": record["case_index"],
        "case_input_id": record["case_input_id"],
        "simulation_case_id": record["simulation_case_id"],
        "generator_version": scientific.get("generator_version"),
        "git_commit": manifest["git_commit"],
        "material_family": record["material_family"],
        "material_role": scientific.get("material_role"),
        "evaluation_regime": scientific.get("evaluation_regime"),
        "sampling_regime": manifest["sampling_regime"],
        "natural_support_state": natural_support_state,
        "available_learning_views": manifest["available_learning_views"],
        "airflow_source": manifest["airflow_source"],
        "stationary_fixed_ownership": scientific.get("stationary_fixed_ownership"),
        "template": {
            "relative_path": manifest["template"]["relative_path"],
            "filename": Path(manifest["template"]["relative_path"]).name,
            "sha256": manifest["template"]["sha256"],
        },
        "export_contract_sha256": manifest["export_contract_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        msg = f"Canonical case metadata disagrees with terminal batch evidence: {directory}"
        raise RuntimeError(msg)
    ood = payload.get("ood")
    parameter_ood = scientific.get("parameter_ood")
    allowed_groups = parameter_ood.get("groups") if isinstance(parameter_ood, Mapping) else None
    if (
        len(matches) != 1
        or not isinstance(ood, Mapping)
        or not isinstance(allowed_groups, list)
        or ood.get("group") != matches[0].get("ood_group")
        or ood.get("natural_support_state") != natural_support_state
        or (ood.get("group") is not None and ood.get("group") not in allowed_groups)
    ):
        msg = f"Canonical case OOD group or support state disagrees with its persisted Generation assignment: {directory}"
        raise RuntimeError(msg)
    if "steady_flow_conditioning" in scientific and payload.get("steady_flow_conditioning") != scientific["steady_flow_conditioning"]:
        msg = f"Canonical case conditioning disagrees with persisted resolved science: {directory}"
        raise RuntimeError(msg)


def admit_terminal_batch(
    batch_id: str,
    *,
    storage_root: Path | str | None = None,
) -> TerminalBatchEvidence:
    """
    Admit one terminal batch without requiring its authored configuration.

    Parameters
    ----------
    batch_id : str
        Immutable generated-batch identifier.
    storage_root : Path | str | None, optional
        Storage root containing the Generation publication.

    Returns
    -------
    TerminalBatchEvidence
        Immutable evidence for the manifest, exact membership, case
        publications, artifacts, identities, and canonical HDF5 payloads.

    Raises
    ------
    FileNotFoundError
        If required terminal evidence is missing or unsafe.
    ValueError
        If an identifier, digest, profile, or persisted value is malformed.
    RuntimeError
        If independently valid evidence disagrees across publication layers.

    """
    safe_batch_id = common.paths.validate_logical_name(batch_id, label="batch_id")
    generation_root = common.paths.get_generation_root(storage_root=storage_root).expanduser().resolve()
    meta_candidate = common.paths.get_generation_meta_root(storage_root=storage_root) / safe_batch_id
    if not meta_candidate.is_dir() or meta_candidate.is_symlink():
        msg = f"Terminal batch metadata directory is missing or unsafe: {meta_candidate}"
        raise FileNotFoundError(msg)
    meta_directory = meta_candidate.resolve()
    manifest_path = meta_directory / "batch_manifest.json"
    success_path = meta_directory / "_SUCCESS"
    scientific_path = meta_directory / "resolved_generation_config.json"
    manifest = _load_json_object(manifest_path, label="terminal batch manifest")
    success = _load_json_object(success_path, label="terminal batch success marker")
    scientific = _load_json_object(scientific_path, label="resolved scientific generation configuration")
    manifest_sha256 = _safe_file_sha256(manifest_path, label="terminal batch manifest")
    if (
        set(manifest) != _BATCH_MANIFEST_KEYS
        or manifest.get("schema_kind") != _BATCH_MANIFEST_SCHEMA_KIND
        or manifest.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("batch_id") != safe_batch_id
    ):
        msg = f"Terminal batch manifest schema or completion state is invalid: {manifest_path}"
        raise RuntimeError(msg)
    batch_identity = _require_sha256(manifest.get("batch_identity"), label="terminal batch_identity")
    scientific_digest = _require_sha256(
        manifest.get("scientific_config_digest"),
        label="terminal scientific_config_digest",
    )
    export_contract_sha256 = _require_sha256(
        manifest.get("export_contract_sha256"),
        label="terminal export_contract_sha256",
    )
    if batch_identity != scientific_digest or common.serialization.canonical_json_sha256(scientific) != scientific_digest:
        msg = f"Terminal batch identity is not bound to persisted resolved science: {manifest_path}"
        raise RuntimeError(msg)
    profile_id = manifest.get("simulation_profile")
    if not isinstance(profile_id, str):
        msg = f"Terminal simulation_profile is malformed: {manifest_path}"
        raise TypeError(msg)
    profile = profiles.resolve_profile(profile_id)
    template = manifest.get("template")
    if not isinstance(template, dict) or set(template) != {"relative_path", "sha256"}:
        msg = f"Terminal template descriptor is malformed: {manifest_path}"
        raise RuntimeError(msg)
    template_sha256 = _require_sha256(template["sha256"], label="terminal template sha256")
    if (
        manifest.get("available_learning_views") != list(profile.available_learning_views)
        or manifest.get("airflow_source") != profile.airflow_source
        or template["relative_path"] != profile.template_relative_path
    ):
        msg = f"Terminal profile or template descriptor is invalid: {manifest_path}"
        raise RuntimeError(msg)
    material_family = materials.validate_material_family(manifest.get("material_family"))
    sampling_regime = manifest.get("sampling_regime")
    batch_name = manifest.get("batch_name")
    batch_kind = (
        config_contract.PILOT_CAMPAIGN_PURPOSE if scientific.get("campaign_purpose") == config_contract.PILOT_CAMPAIGN_PURPOSE else sampling_regime
    )
    identity_is_valid = False
    if isinstance(material_family, str) and isinstance(sampling_regime, str) and isinstance(batch_name, str) and isinstance(batch_kind, str):
        expected_name = config_contract.build_batch_name(profile.id, material_family, batch_kind)
        identity_is_valid = batch_name == expected_name and safe_batch_id == config_contract.build_batch_id(expected_name, scientific_digest)
    if not sampling_regime or not identity_is_valid:
        msg = f"Terminal batch name, material, sampling regime, or immutable identifier is invalid: {manifest_path}"
        raise RuntimeError(msg)
    git_commit = source_service.validate_git_commit(manifest.get("git_commit"))
    if set(success) != _BATCH_SUCCESS_KEYS or success != {
        "schema_kind": _BATCH_SUCCESS_SCHEMA_KIND,
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "simulation_profile": profile.id,
        "batch_id": safe_batch_id,
        "batch_identity": batch_identity,
        "manifest_sha256": manifest_sha256,
    }:
        msg = f"Terminal success marker does not bind the manifest: {success_path}"
        raise RuntimeError(msg)
    indices = manifest.get("intended_case_indices")
    records = manifest.get("cases")
    if (
        not isinstance(indices, list)
        or not indices
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in indices)
        or indices != sorted(set(indices))
        or not isinstance(records, list)
        or len(records) != len(indices)
    ):
        msg = f"Terminal batch membership is malformed: {manifest_path}"
        raise RuntimeError(msg)
    _validate_terminal_scientific_config(
        scientific,
        manifest=manifest,
        manifest_path=manifest_path,
        case_count=len(indices),
    )
    case_ids = tuple(f"case_{index:04d}" for index in indices)
    raw_root, processed_root = _validate_exact_batch_directory_membership(
        safe_batch_id,
        case_ids,
        storage_root=storage_root,
    )
    admitted_cases: list[TerminalCaseEvidence] = []
    for expected_index, expected_case_id, raw_record in zip(indices, case_ids, records, strict=True):
        if (
            not isinstance(raw_record, dict)
            or set(raw_record) != _CASE_RECORD_KEYS
            or raw_record.get("case_index") != expected_index
            or raw_record.get("case_id") != expected_case_id
            or raw_record.get("material_family") != material_family
        ):
            msg = f"Terminal case record is malformed for {expected_case_id}: {manifest_path}"
            raise RuntimeError(msg)
        record = raw_record
        for key in (
            "case_input_id",
            "simulation_case_id",
            "success_sha256",
            "provenance_sha256",
            "case_hdf5_sha256",
        ):
            _require_sha256(record.get(key), label=f"{expected_case_id}.{key}")
        raw = _admit_publication_directory(raw_root / expected_case_id, stage="raw")
        processed = _admit_publication_directory(processed_root / expected_case_id, stage="processed")
        if raw.case_payload != processed.case_payload:
            msg = f"Raw and processed canonical metadata disagree for {expected_case_id}."
            raise RuntimeError(msg)
        _require_case_matches_terminal(
            raw.case_payload,
            manifest=manifest,
            scientific=scientific,
            record=record,
            directory=raw.directory,
        )
        processed_success_sha256 = _safe_file_sha256(
            processed.directory / "_SUCCESS",
            label=f"{expected_case_id} success marker",
        )
        processed_provenance_sha256 = _safe_file_sha256(
            processed.directory / "provenance.json",
            label=f"{expected_case_id} publication provenance",
        )
        processed_hdf5_sha256 = _safe_file_sha256(
            processed.directory / "case.h5",
            label=f"{expected_case_id} canonical HDF5",
        )
        if (
            processed_success_sha256 != record["success_sha256"]
            or processed_provenance_sha256 != record["provenance_sha256"]
            or processed_hdf5_sha256 != record["case_hdf5_sha256"]
        ):
            msg = f"Terminal manifest artifact digests disagree for {expected_case_id}."
            raise RuntimeError(msg)
        if processed.hdf5_identity is None:
            msg = f"Processed case admission lost canonical HDF5 evidence for {expected_case_id}."
            raise RuntimeError(msg)
        admitted_cases.append(
            TerminalCaseEvidence(
                case_index=expected_index,
                case_id=expected_case_id,
                material_family=str(record["material_family"]),
                case_input_id=str(record["case_input_id"]),
                simulation_case_id=str(record["simulation_case_id"]),
                success_sha256=str(record["success_sha256"]),
                provenance_sha256=str(record["provenance_sha256"]),
                case_hdf5_sha256=str(record["case_hdf5_sha256"]),
                raw_directory=raw.directory,
                processed_directory=processed.directory,
                hdf5_path=(processed.directory / "case.h5").resolve(),
                raw_artifacts=raw.artifacts,
                processed_artifacts=processed.artifacts,
                hdf5_identity=processed.hdf5_identity,
                _case_metadata_json=_canonical_json_text(raw.case_payload),
            )
        )
    evidence = TerminalBatchEvidence(
        generation_root=generation_root,
        meta_directory=meta_directory,
        raw_directory=raw_root,
        processed_directory=processed_root,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=manifest_sha256,
        simulation_profile=profile.id,
        available_learning_views=profile.available_learning_views,
        airflow_source=profile.airflow_source,
        batch_name=str(batch_name),
        batch_id=safe_batch_id,
        batch_identity=batch_identity,
        material_family=str(material_family),
        sampling_regime=str(sampling_regime),
        git_commit=git_commit,
        scientific_config_digest=scientific_digest,
        template_relative_path=profile.template_relative_path,
        template_sha256=template_sha256,
        export_contract_sha256=export_contract_sha256,
        cases=tuple(admitted_cases),
        _scientific_config_json=_canonical_json_text(scientific),
    )
    if evidence.manifest_payload() != manifest:
        msg = f"Typed terminal evidence does not exactly represent its manifest: {manifest_path}"
        raise RuntimeError(msg)
    return evidence


def validate_terminal_batch(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Admit a terminal batch and require exact authored-config agreement."""
    evidence = admit_terminal_batch(config.batch_id, storage_root=storage_root)
    expected = {
        "simulation_profile": config.profile.id,
        "available_learning_views": config.profile.available_learning_views,
        "airflow_source": config.profile.airflow_source,
        "batch_name": config.batch_name,
        "batch_identity": config.batch_identity,
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "scientific_config_digest": config.scientific_config_digest,
        "template_relative_path": config.profile.template_relative_path,
        "template_sha256": config.template_sha256,
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
        "cases": tuple(config.case_indices),
    }
    actual = {
        "simulation_profile": evidence.simulation_profile,
        "available_learning_views": evidence.available_learning_views,
        "airflow_source": evidence.airflow_source,
        "batch_name": evidence.batch_name,
        "batch_identity": evidence.batch_identity,
        "material_family": evidence.material_family,
        "sampling_regime": evidence.sampling_regime,
        "scientific_config_digest": evidence.scientific_config_digest,
        "template_relative_path": evidence.template_relative_path,
        "template_sha256": evidence.template_sha256,
        "export_contract_sha256": evidence.export_contract_sha256,
        "cases": tuple(case.case_index for case in evidence.cases),
    }
    if actual != expected or evidence.scientific_config_payload() != config.scientific_values:
        msg = f"Terminal batch scientific identity disagrees with authored configuration: {evidence.manifest_path}"
        raise RuntimeError(msg)
    for case in evidence.cases:
        _require_case_payload_matches_config(
            case.metadata_payload(),
            directory=case.processed_directory,
            config=config,
            case_index=case.case_index,
        )
    return evidence.manifest_payload()
