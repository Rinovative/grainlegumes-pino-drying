"""
generation_runtime_batch.py

Run, admit, and atomically publish isolated profile-qualified COMSOL cases.
Responsibilities:
  - Execute safe one-node COMSOL commands and retain complete runtime evidence
  - Collect explicit raw adapters and convert them to canonical case.h5
  - Publish non-authoritative case phases without changing solver outcomes
  - Publish cases, recover hash-bound Full-Retention HDF5, and admit terminal batches
Design principles:
  - Scientific configuration and execution provenance are physically separate
  - Successful CSV and solved-model retention is explicit and off by default
  - Failed scratch is removed only after policy-bound durable evidence is complete
This module does NOT:
  - Modify COMSOL templates or infer internal tags, expressions, or signs
  - Publish a parallel canonical CSV learning view
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
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from src import common
from src.generation.cases import generation_cases_admission as admission_service
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_config as config_contract
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_scalar_handoff as scalar_handoff_contract
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.contracts import generation_contracts_templates as templates
from src.generation.publication import generation_publication_attempt as attempt_service
from src.generation.publication import (
    generation_publication_campaign_evidence as campaign_evidence,
)
from src.generation.publication import generation_publication_storage as storage_service

from . import generation_runtime_comsol as comsol_service
from . import generation_runtime_license as license_service
from . import generation_runtime_progress as progress_service
from . import generation_runtime_stop as stop_service
from . import generation_runtime_workspace as workspace_service
from .generation_runtime_preparation import PreparedCase, prepare_case_work_directory

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.generation.validation.generation_validation_policy import DiagnosticRecord

PUBLICATION_SCHEMA_VERSION = 1
BATCH_MANIFEST_SCHEMA_VERSION = 1
_SMOKE_HDF5_RECONSTRUCTION_SCHEMA_KIND: Final = "generation_smoke_hdf5_reconstruction"
_SMOKE_HDF5_RECONSTRUCTION_SCHEMA_VERSION: Final = 1
_BATCH_MANIFEST_SCHEMA_KIND: Final = "simulation_batch_manifest"
_BATCH_SUCCESS_SCHEMA_KIND: Final = "simulation_batch_success"
_CASE_PUBLICATION_SCHEMA_KIND: Final = "simulation_case_publication"
_CASE_SUCCESS_SCHEMA_KIND: Final = "simulation_case_success"
_CASE_ID_PATTERN: Final = re.compile(r"case_[0-9]{4,}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_MAX_RETAINED_SOLVER_LOG_BYTES: Final = 1024 * 1024
_CHECKOUT_TERMINATION_WAIT_SECONDS: Final = 5.0
ValidationDepth = Literal["routine", "full", "deep"]
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
        "batch_storage_name",
        "batch_identity",
        "campaign_purpose",
        "material_family",
        "sampling_regime",
        "git_commit",
        "input_generation_id",
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
        "input_generation_id",
        "template_sha256",
        "scientific_config_digest",
        "export_contract_sha256",
        "available_learning_views",
        "airflow_source",
        "retention_policy",
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
CASE_FAILURE_SCHEMA_VERSION = 1
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
        "execution_run_id",
        "recorded_at",
        "execution",
        "error",
        "work_directory",
        "input_files",
        "template_sha256",
        "missing_or_invalid_artifacts",
        "export_diagnostics",
        "log_tail",
        "retained_artifacts",
        "retention_error",
        "failure_diagnostics",
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
_FAILURE_STAGES = frozenset({"input", "solver", "exports", "conversion", "publication"})


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


class CaseLocalReplayError(RuntimeError):
    """Report a replay defect after durable case-local evidence is complete."""


class ReplayIntegrityError(RuntimeError):
    """Report a replay path or identity violation that remains campaign-global."""


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
_RUNTIME_FORCE_CANCELLATION = threading.Event()


def reset_runtime_cancellation() -> None:
    """Clear cooperative cancellation before one campaign worker starts."""
    with _ACTIVE_SOLVER_LOCK:
        if _ACTIVE_SOLVERS:
            message = "Cannot reset runtime cancellation while solvers remain active."
            raise RuntimeError(message)
        _RUNTIME_CANCELLATION.clear()
        _RUNTIME_FORCE_CANCELLATION.clear()


def runtime_cancellation_requested() -> bool:
    """Return whether the current campaign worker received cancellation."""
    return _RUNTIME_CANCELLATION.is_set()


def runtime_force_cancellation_requested() -> bool:
    """Return whether the current campaign worker received force cancellation."""
    return _RUNTIME_FORCE_CANCELLATION.is_set()


def _signal_solver_termination(process: subprocess.Popen[str]) -> None:
    """Best-effort TERM one solver-owned process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def request_runtime_cancellation() -> None:
    """Request a controlled stop for each current or subsequently launched solver."""
    _RUNTIME_CANCELLATION.set()


def request_runtime_force_cancellation() -> None:
    """Request immediate bounded force escalation for owned solver groups."""
    _RUNTIME_CANCELLATION.set()
    _RUNTIME_FORCE_CANCELLATION.set()


def _register_solver(process: subprocess.Popen[str]) -> None:
    """Register one solver before its stop controller begins polling."""
    with _ACTIVE_SOLVER_LOCK:
        _ACTIVE_SOLVERS[process.pid] = process


def _unregister_solver(process: subprocess.Popen[str]) -> None:
    """Remove one completed or terminated solver from cancellation tracking."""
    with _ACTIVE_SOLVER_LOCK:
        _ACTIVE_SOLVERS.pop(process.pid, None)


def _terminate_solver_and_wait(process: subprocess.Popen[str]) -> int:
    """TERM then KILL one solver group within two bounded waits."""
    _signal_solver_termination(process)
    try:
        return process.wait(timeout=_CHECKOUT_TERMINATION_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    try:
        return process.wait(timeout=_CHECKOUT_TERMINATION_WAIT_SECONDS)
    except subprocess.TimeoutExpired as error:
        message = f"Owned COMSOL process did not exit after bounded TERM and KILL waits: pid={process.pid}"
        raise RuntimeError(message) from error


def _create_runtime_progress_reporter(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    scheduler_kind: str,
    storage_root: Path,
) -> progress_service.RuntimeProgressReporter | None:
    """Create optional exact campaign progress without affecting execution."""
    if scheduler_kind != "slurm":
        return None
    run_id = os.environ.get("GENERATION_CAMPAIGN_RUN_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not run_id or not job_id:
        return None
    try:
        return progress_service.RuntimeProgressReporter.create(
            run_id,
            batch_name=config.batch_name,
            batch_id=config.batch_id,
            case_index=case_index,
            case_id=config.case_id(case_index),
            slurm_job_id=job_id,
            storage_root=storage_root,
        )
    except Exception:  # noqa: BLE001 -- monitoring setup cannot terminate a case
        return None


def _bind_runtime_progress_stdout(
    reporter: progress_service.RuntimeProgressReporter | None,
    stdout_path: Path,
) -> None:
    """Best-effort bind the case-local solver stdout source."""
    if reporter is None:
        return
    try:
        reporter.bind_stdout(stdout_path)
    except Exception:  # noqa: BLE001 -- monitoring setup cannot terminate a case
        return


def _update_runtime_progress(
    reporter: progress_service.RuntimeProgressReporter | None,
    *,
    phase: str,
    terminal: bool = False,
    force: bool = False,
) -> None:
    """Best-effort publish one observational phase or solver snapshot."""
    if reporter is None:
        return
    try:
        reporter.update(phase=phase, terminal=terminal, force=force)
    except Exception:  # noqa: BLE001 -- monitoring cannot terminate a case
        return


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
    solved_model: Path | None
    execution_provenance: Path
    processing_provenance: Path


@dataclass(frozen=True, slots=True)
class CaseRunOutcome:
    """One skipped or newly published completed case."""

    status: str
    case_id: str
    processed_directory: Path
    work_directory: Path | None
    message: str | None = None


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
    git_commit: str | None
    template_relative_path: str | None
    template_sha256: str
    case_input_id: str
    simulation_case_id: str
    scientific_config_digest: str
    export_contract_sha256: str
    available_learning_views: tuple[str, ...]
    airflow_source: str
    retention_policy: str


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

    def artifact_evidence(
        self,
        stage: str,
        relative_path: str,
    ) -> ArtifactEvidence:
        """Return identity-bound artifact evidence with cheap size admission."""
        if stage == "raw":
            artifacts = self.raw_artifacts
        elif stage == "processed":
            artifacts = self.processed_artifacts
        else:
            message = f"Unsupported terminal publication stage: {stage!r}."
            raise ValueError(message)
        matches = tuple(item for item in artifacts if item.relative_path == relative_path)
        if len(matches) != 1:
            message = f"Terminal case {self.case_id!r} has no unique {stage} artifact {relative_path!r}."
            raise ValueError(message)
        artifact = matches[0]
        if not artifact.path.is_file() or artifact.path.is_symlink() or artifact.path.stat().st_size != artifact.size_bytes:
            message = f"Admitted terminal artifact is missing, unsafe, or changed size: {artifact.path}"
            raise RuntimeError(message)
        return artifact

    def artifact(self, stage: str, relative_path: str) -> ArtifactEvidence:
        """Fully rehash and return one admitted artifact."""
        artifact = self.artifact_evidence(stage, relative_path)
        if common.serialization.file_sha256(artifact.path) != artifact.sha256:
            message = f"Admitted terminal artifact changed after admission: {artifact.path}"
            raise RuntimeError(message)
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
    batch_storage_name: str
    batch_identity: str
    campaign_purpose: str
    material_family: str
    sampling_regime: str
    git_commit: str
    input_generation_id: str
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
            "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "simulation_profile": self.simulation_profile,
            "available_learning_views": list(self.available_learning_views),
            "airflow_source": self.airflow_source,
            "batch_name": self.batch_name,
            "batch_id": self.batch_id,
            "batch_storage_name": self.batch_storage_name,
            "batch_identity": self.batch_identity,
            "campaign_purpose": self.campaign_purpose,
            "material_family": self.material_family,
            "sampling_regime": self.sampling_regime,
            "git_commit": self.git_commit,
            "input_generation_id": self.input_generation_id,
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


def _prior_license_wait_seconds(
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
) -> float:
    """Return controller-managed backoff accumulated before this real attempt."""
    campaign_run_id = os.environ.get("GENERATION_CAMPAIGN_RUN_ID")
    if campaign_run_id is None:
        return 0.0
    wait = license_service.load_temporary_license_wait(
        config,
        int(prepared.bundle.case_payload["case_index"]),
        campaign_run_id=campaign_run_id,
        storage_root=prepared.storage_root,
    )
    return 0.0 if wait is None else float(wait["cumulative_wait_seconds"])


def _state_batch_root(config: config_contract.GenerationConfig, *, storage_root: Path | str | None) -> Path:
    """Return the private state root for one profile-qualified batch."""
    return common.paths.resolve_generation_state_batch_directory(
        config.batch_storage_name,
        storage_root=storage_root,
    )


def case_lock_path(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one persistent case-level advisory-lock anchor."""
    return common.paths.resolve_generation_case_lock_path(
        config.batch_storage_name,
        config.case_id(case_index),
        storage_root=storage_root,
    )


def raw_case_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one exact input-generation case provenance directory."""
    input_generation_id = input_service.configured_input_generation_id(config)
    return common.paths.resolve_generation_input_generation_raw_directory(
        config.batch_storage_name,
        input_generation_id,
        storage_root=storage_root,
    ) / config.case_id(case_index)


def processed_case_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one permanent canonical completed-case directory."""
    return common.paths.resolve_generation_processed_case_directory(
        config.batch_storage_name,
        config.case_id(case_index),
        storage_root=storage_root,
    )


def batch_meta_directory(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one batch-owned metadata directory."""
    return common.paths.resolve_generation_batch_metadata_directory(
        config.batch_storage_name,
        storage_root=storage_root,
    )


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
    """Initialize immutable batch science and execution metadata."""
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    directory = batch_meta_directory(config, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    scientific_path = _immutable_json(
        directory / "resolved_generation_config.json",
        config.scientific_values,
        label="resolved scientific generation configuration",
    )
    execution_digest = common.serialization.canonical_json_sha256(
        config.execution_values,
    )
    execution_directory = directory / "execution_configs"
    execution_directory.mkdir(exist_ok=True)
    _immutable_json(
        execution_directory / f"{execution_digest}.json",
        config.execution_values,
        label="resolved execution provenance",
    )
    persisted_scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
    if config_contract.compute_scientific_config_digest(persisted_scientific) != config.scientific_config_digest:
        message = "Persisted resolved_generation_config.json digest disagrees with scientific identity."
        raise RuntimeError(message)
    return scientific_path


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
    canonical_path = work_directory / comsol_service.RETAINED_MODEL_FILENAME
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
        "requested_relative_path": comsol_service.RETAINED_MODEL_FILENAME,
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


def _bounded_log_text(path: Path, *, maximum_bytes: int) -> str:
    """Return bounded UTF-8 head-and-tail evidence from one runtime log."""
    size = path.stat().st_size
    if size <= maximum_bytes:
        payload = path.read_bytes()
    else:
        half = maximum_bytes // 2
        with path.open("rb") as stream:
            head = stream.read(half)
            stream.seek(-half, os.SEEK_END)
            tail = stream.read(half)
        payload = head + b"\n... retained log middle omitted ...\n" + tail
    return payload.decode("utf-8", errors="replace")


def _write_solver_log(prepared: PreparedCase) -> Path:
    """Combine bounded head-and-tail evidence from case-owned solver logs."""
    stdout_path = prepared.runtime_directory / "stdout.log"
    stderr_path = prepared.runtime_directory / "stderr.log"
    per_stream = _MAX_RETAINED_SOLVER_LOG_BYTES // 2
    payload = "===== stdout =====\n" + _bounded_log_text(stdout_path, maximum_bytes=per_stream)
    if not payload.endswith("\n"):
        payload += "\n"
    payload += "===== stderr =====\n" + _bounded_log_text(stderr_path, maximum_bytes=per_stream)
    if not payload.endswith("\n"):
        payload += "\n"
    path = prepared.runtime_directory / "solver.log"
    common.serialization.atomic_write_text(path, payload)
    return path


def _expected_exports_exist(
    config: config_contract.GenerationConfig,
    work_directory: Path,
) -> bool:
    """Return whether every required configured export already exists."""
    export_root = work_directory / str(config.scientific_values["output_contract"]["exports_root"])
    if not export_root.is_dir() or export_root.is_symlink():
        return False
    for contract in config.scientific_values["output_contract"]["exports"]:
        if contract["required"] is not True:
            continue
        pattern = contract.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return False
        matches = tuple(candidate for candidate in export_root.glob(pattern) if candidate.is_file() and not candidate.is_symlink())
        if not matches or (contract["allow_multiple"] is not True and len(matches) != 1):
            return False
    return True


def _workspace_regular_file_bytes(directory: Path) -> int:
    """Return current regular-file bytes without following symbolic links."""
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file() and not path.is_symlink())


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
        "schema_version": 1,
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
            "executable_path": shutil.which(comsol_service.resolve_comsol_executable(config)),
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


def _write_processing_provenance(
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
    *,
    mode: str,
    solver_git_commit: str,
    source_attempt: attempt_service.AttemptEvidence | None = None,
) -> Path:
    """Persist distinct solver and postprocessing source provenance."""
    if mode not in {"initial", "replay_conversion", "replay_publication"}:
        message = f"Unsupported processing provenance mode: {mode!r}."
        raise ValueError(message)
    solver_commit = source_service.validate_git_commit(solver_git_commit)
    processing_commit = source_service.required_git_commit()
    payload = {
        "schema_kind": "generation_processing_provenance",
        "schema_version": 1,
        "case_id": prepared.bundle.case_id,
        "case_input_id": prepared.bundle.case_input_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "mode": mode,
        "solver_git_commit": solver_commit,
        "processing_git_commit": processing_commit,
        "source_attempt": (
            None
            if source_attempt is None
            else {
                "campaign_run_id": source_attempt.payload["campaign_run_id"],
                "attempt_index": source_attempt.payload["attempt_index"],
                "attempt_receipt_sha256": common.serialization.file_sha256(source_attempt.receipt_path),
            }
        ),
        "scientific_config_digest": config.scientific_config_digest,
        "template_sha256": config.template_sha256,
        "recorded_at": _utc_now(),
    }
    return common.serialization.atomic_write_json(
        prepared.runtime_directory / "processing_provenance.json",
        payload,
    )


def _final_solver_metrics_from_directory(work_directory: Path) -> dict[str, Any]:
    """Parse final supported finite solver evidence from an owned directory."""
    stdout_path = work_directory / "runtime" / "stdout.log"
    if not stdout_path.is_file() or stdout_path.is_symlink():
        return {}
    parser = progress_service.ComsolProgressParser()
    state = parser.consume(stdout_path.read_text(encoding="utf-8", errors="replace").splitlines())
    return state if state.get("parser_state") == "available" else {}


def _final_solver_metrics(prepared: PreparedCase) -> dict[str, Any]:
    """Parse final supported finite solver evidence from one prepared case."""
    return _final_solver_metrics_from_directory(prepared.work_directory)


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


def _reset_runtime_progress_stdout(
    reporter: progress_service.RuntimeProgressReporter | None,
    stdout_path: Path,
) -> None:
    """Best-effort reset parsing after one owned checkout log is truncated."""
    if reporter is None:
        return
    try:
        reporter.reset_stdout(stdout_path)
    except Exception:  # noqa: BLE001 -- monitoring cannot terminate a case
        return


def _update_license_acquisition_progress(
    reporter: progress_service.RuntimeProgressReporter | None,
    *,
    window_started_monotonic: float,
    window_limit_seconds: float,
    checkout_attempt_count: int,
    last_result: str | None,
    force: bool = False,
) -> None:
    """Best-effort publish compact current allocation-window progress."""
    if reporter is None:
        return
    try:
        reporter.update_license_acquisition(
            window_seconds=max(0.0, time.monotonic() - window_started_monotonic),
            window_limit_seconds=window_limit_seconds,
            checkout_attempt_count=checkout_attempt_count,
            last_result=last_result,
            force=force,
        )
    except Exception:  # noqa: BLE001 -- monitoring cannot terminate a case
        return


def _captured_startup_text(prepared: PreparedCase) -> str:
    """Return bounded current stdout and stderr for one owned startup attempt."""
    chunks = []
    for name in ("stdout.log", "stderr.log"):
        candidate = prepared.runtime_directory / name
        if candidate.is_file() and not candidate.is_symlink():
            chunks.append(_bounded_log_text(candidate, maximum_bytes=32_768))
    return "\n".join(chunks)


def _wait_for_solver_start_or_exit(
    process: subprocess.Popen[str],
    prepared: PreparedCase,
    *,
    deadline: float | None,
    window_started_monotonic: float,
    window_limit_seconds: float,
    checkout_attempt_count: int,
    last_result: str | None,
    progress_reporter: progress_service.RuntimeProgressReporter | None,
) -> tuple[str, int | None]:
    """Wait until solver evidence, process exit, cancellation, or window deadline."""
    poll_seconds = 0.25
    while True:
        captured = _captured_startup_text(prepared)
        if license_service.solver_progress_started(captured):
            _update_runtime_progress(progress_reporter, phase="starting_solver", force=True)
            return "solver_progress_started", None
        completed = process.poll()
        if completed is not None:
            return "process_exited", process.wait()
        if runtime_cancellation_requested():
            return "cancelled", _terminate_solver_and_wait(process)
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            return "window_deadline", _terminate_solver_and_wait(process)
        wait_seconds = poll_seconds if deadline is None else min(poll_seconds, max(0.0, deadline - now))
        try:
            completed = process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            _update_license_acquisition_progress(
                progress_reporter,
                window_started_monotonic=window_started_monotonic,
                window_limit_seconds=window_limit_seconds,
                checkout_attempt_count=checkout_attempt_count,
                last_result=last_result,
            )
            continue
        return "process_exited", completed


def _assert_capacity_retry_workspace_is_clean(
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
    solved_models_before: Mapping[str, _SolvedModelInventoryEntry],
) -> None:
    """Fail closed unless a capacity-only checkout changed no scientific artifacts."""
    if _solved_model_inventory(prepared.work_directory) != solved_models_before:
        message = "Temporary-capacity checkout changed solved-model artifacts before retry."
        raise CaseExecutionError(message, work_directory=prepared.work_directory)
    export_root = prepared.work_directory / str(config.scientific_values["output_contract"]["exports_root"])
    if export_root.exists() and (export_root.is_symlink() or not export_root.is_dir() or any(export_root.iterdir())):
        message = "Temporary-capacity checkout left unexpected export artifacts before retry."
        raise CaseExecutionError(message, work_directory=prepared.work_directory)


def _persist_in_allocation_window(
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
    summaries: tuple[license_service.InAllocationLicenseCheckoutSummary, ...],
    *,
    window_started_at: datetime,
    window_started_monotonic: float,
    outcome: str,
) -> Path | None:
    """Persist one final campaign allocation-window receipt when identity is available."""
    campaign_run_id = os.environ.get("GENERATION_CAMPAIGN_RUN_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    if campaign_run_id is None or job_id is None:
        return None
    result = license_service.in_allocation_license_window_result(
        config,
        int(prepared.bundle.case_payload["case_index"]),
        campaign_run_id=campaign_run_id,
        job_id=job_id,
        hostname=socket.gethostname(),
        window_started_at=window_started_at,
        window_ended_at=datetime.now(timezone.utc),
        window_started_monotonic_seconds=window_started_monotonic,
        window_ended_monotonic_seconds=time.monotonic(),
        checkout_summaries=summaries,
        solver_progress_started=outcome == "solver_progress_started",
        outcome=outcome,
    )
    return license_service.record_in_allocation_license_window(
        config,
        int(prepared.bundle.case_payload["case_index"]),
        result,
        storage_root=prepared.storage_root,
    )


def execute_prepared_case(  # noqa: C901, PLR0912, PLR0915 -- centralized COMSOL process lifecycle
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
    *,
    cores_per_case: int,
    worker_slot: int,
    scheduler_kind: str = "local",
    allocated_node: str | None = None,
    progress_reporter: progress_service.RuntimeProgressReporter | None = None,
    diagnostic_observer: Callable[[str, PreparedCase, Mapping[str, Any]], None] | None = None,
) -> ExecutionResult:
    """Run one isolated COMSOL process and create its validated canonical HDF5."""
    scalar_handoff = prepared.bundle.scalar_handoff
    if scalar_handoff is not None:
        scalar_handoff_contract.validate_transient_scalar_source(scalar_handoff)
    batch_log = prepared.runtime_directory / "comsol_batch.log" if diagnostic_observer is not None else None
    command = comsol_service.build_comsol_command(
        config,
        cores_per_case=cores_per_case,
        scalar_handoff=scalar_handoff,
        scheduler_kind=scheduler_kind,
        diagnostic_batchlog=(None if batch_log is None else str(batch_log)),
    )
    if diagnostic_observer is not None:
        diagnostic_observer(
            "prepared",
            prepared,
            {
                "command": tuple(command),
                "batch_log_path": (None if batch_log is None else str(batch_log)),
                "cores_per_case": cores_per_case,
            },
        )
    try:
        _require_executable(command, comsol_executable=comsol_service.resolve_comsol_executable(config))
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
    retain_solved_model = config.execution_values["retention_policy"] == "full"
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
    stdout_path = prepared.runtime_directory / "stdout.log"
    stderr_path = prepared.runtime_directory / "stderr.log"
    _bind_runtime_progress_stdout(progress_reporter, stdout_path)
    retry_policy = config.execution_values["runtime"]["temporary_license_retry"]
    in_allocation_policy = retry_policy["in_allocation_retry"]
    in_allocation_enabled = bool(scheduler_kind == "slurm" and retry_policy["enabled"] and in_allocation_policy["enabled"])
    window_started_at = datetime.now(timezone.utc)
    window_started_monotonic = time.monotonic()
    window_deadline = window_started_monotonic + float(in_allocation_policy["maximum_window_seconds"]) if in_allocation_enabled else None
    checkout_summaries: list[license_service.InAllocationLicenseCheckoutSummary] = []
    in_allocation_license_window_seconds = 0.0
    in_allocation_pause_seconds = 0.0
    status_artifact_recovery_count = 0
    process: subprocess.Popen[str] | None = None
    stop_result: stop_service.StopResult | None = None
    timed_out = False
    exit_code: int | None = None
    started_at = _utc_now()
    monotonic_start = window_started_monotonic
    while True:
        if in_allocation_enabled and runtime_cancellation_requested():
            message = "Campaign cancellation was requested between COMSOL license checkouts."
            raise CaseInterruptedError(
                message,
                work_directory=prepared.work_directory,
                command=tuple(command),
                exit_code=None,
            )
        if window_deadline is not None and time.monotonic() >= window_deadline:
            if not checkout_summaries or checkout_summaries[-1].classification is None:
                message = "In-allocation COMSOL startup deadline elapsed without strong temporary-capacity evidence."
                raise CaseExecutionError(message, work_directory=prepared.work_directory, command=tuple(command))
            _persist_in_allocation_window(
                config,
                prepared,
                tuple(checkout_summaries),
                window_started_at=window_started_at,
                window_started_monotonic=window_started_monotonic,
                outcome="window_exhausted",
            )
            last = checkout_summaries[-1]
            evidence = last.classification
            if evidence is None:
                message = "Exhausted allocation window lacks temporary-capacity evidence."
                raise RuntimeError(message)
            message = "In-allocation COMSOL license-acquisition window exhausted."
            raise license_service.TemporaryLicenseCapacityError(
                message,
                work_directory=prepared.work_directory,
                command=tuple(command),
                exit_code=last.process_exit_code,
                evidence=evidence,
            )
        attempt_started_at = datetime.now(timezone.utc)
        attempt_started_monotonic = time.monotonic()
        started_at = attempt_started_at.isoformat()
        monotonic_start = attempt_started_monotonic
        status_prelaunch = (
            stop_service.prepare_capacity_checkout_status(
                prepared.work_directory,
                checkout_index=len(checkout_summaries) + 1,
            )
            if in_allocation_enabled
            else None
        )
        with (
            stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_stream,
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_stream,
        ):
            _reset_runtime_progress_stdout(progress_reporter, stdout_path)
            if in_allocation_enabled:
                _update_license_acquisition_progress(
                    progress_reporter,
                    window_started_monotonic=window_started_monotonic,
                    window_limit_seconds=float(in_allocation_policy["maximum_window_seconds"]),
                    checkout_attempt_count=len(checkout_summaries) + 1,
                    last_result=(None if not checkout_summaries else license_service.TEMPORARY_LICENSE_CAPACITY),
                    force=True,
                )
            else:
                _update_runtime_progress(progress_reporter, phase="starting_solver", force=True)
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
                raise CaseExecutionError(msg, work_directory=prepared.work_directory, command=tuple(command)) from error
            _register_solver(process)
            try:
                if not in_allocation_enabled:
                    stop_controller = stop_service.SolverStopController(
                        process,
                        prepared.work_directory,
                        timeout_seconds=float(config.execution_values["runtime"]["timeout_seconds"]),
                        graceful_stop_reserve_seconds=float(config.execution_values["runtime"]["graceful_stop_reserve_seconds"]),
                        monotonic_clock=time.monotonic,
                    )
                    stop_result = stop_controller.wait_for_exit(
                        cancellation_requested=runtime_cancellation_requested,
                        force_requested=runtime_force_cancellation_requested,
                        progress_callback=lambda: _update_runtime_progress(
                            progress_reporter,
                            phase="starting_solver",
                        ),
                    )
                    exit_code = stop_result.exit_code
                    timed_out = stop_result.timed_out
                    break
                startup_outcome, startup_exit_code = _wait_for_solver_start_or_exit(
                    process,
                    prepared,
                    deadline=window_deadline,
                    window_started_monotonic=window_started_monotonic,
                    window_limit_seconds=float(in_allocation_policy["maximum_window_seconds"]),
                    checkout_attempt_count=len(checkout_summaries) + 1,
                    last_result=(None if not checkout_summaries else license_service.TEMPORARY_LICENSE_CAPACITY),
                    progress_reporter=progress_reporter,
                )
                if startup_outcome == "cancelled":
                    message = "Campaign cancellation terminated the COMSOL process during license acquisition."
                    raise CaseInterruptedError(  # noqa: TRY301 -- translated at the owned process boundary
                        message,
                        work_directory=prepared.work_directory,
                        command=tuple(command),
                        exit_code=startup_exit_code,
                    )
                if startup_outcome == "window_deadline":
                    if runtime_cancellation_requested():
                        message = "Campaign cancellation coincided with the COMSOL license-acquisition deadline."
                        raise CaseInterruptedError(  # noqa: TRY301 -- cancellation retains lifecycle priority
                            message,
                            work_directory=prepared.work_directory,
                            command=tuple(command),
                            exit_code=startup_exit_code,
                        )
                    captured = _captured_startup_text(prepared)
                    if license_service.solver_progress_started(captured):
                        message = "COMSOL solver progress appeared while resolving the license-acquisition deadline."
                        raise CaseExecutionError(  # noqa: TRY301 -- ambiguous scientific progress fails closed
                            message,
                            work_directory=prepared.work_directory,
                            command=tuple(command),
                            exit_code=startup_exit_code,
                        )
                    _assert_capacity_retry_workspace_is_clean(config, prepared, solved_models_before)
                    case_index = int(prepared.bundle.case_payload["case_index"])
                    if completed_case_is_valid(
                        config,
                        case_index,
                        storage_root=prepared.storage_root,
                    ):
                        message = "Canonical scientific completion exists at the license-acquisition deadline."
                        raise CaseExecutionError(  # noqa: TRY301 -- canonical success cannot become operational retry
                            message,
                            work_directory=prepared.work_directory,
                            command=tuple(command),
                            exit_code=startup_exit_code,
                        )
                    deadline_evidence = license_service.controller_owned_window_deadline_evidence()
                    checkout_summaries.append(
                        license_service.InAllocationLicenseCheckoutSummary(
                            checkout_index=len(checkout_summaries) + 1,
                            started_at=attempt_started_at,
                            ended_at=datetime.now(timezone.utc),
                            started_monotonic_seconds=attempt_started_monotonic,
                            ended_monotonic_seconds=time.monotonic(),
                            process_exit_code=startup_exit_code,
                            classification=deadline_evidence,
                            solver_progress_started=False,
                        )
                    )
                    _persist_in_allocation_window(
                        config,
                        prepared,
                        tuple(checkout_summaries),
                        window_started_at=window_started_at,
                        window_started_monotonic=window_started_monotonic,
                        outcome="window_exhausted",
                    )
                    message = "In-allocation COMSOL license-acquisition window exhausted."
                    raise license_service.TemporaryLicenseCapacityError(  # noqa: TRY301 -- existing operational retry owner
                        message,
                        work_directory=prepared.work_directory,
                        command=tuple(command),
                        exit_code=startup_exit_code,
                        evidence=deadline_evidence,
                    )
                if startup_outcome == "solver_progress_started":
                    checkout_summaries.append(
                        license_service.InAllocationLicenseCheckoutSummary(
                            checkout_index=len(checkout_summaries) + 1,
                            started_at=attempt_started_at,
                            ended_at=datetime.now(timezone.utc),
                            started_monotonic_seconds=attempt_started_monotonic,
                            ended_monotonic_seconds=time.monotonic(),
                            process_exit_code=None,
                            classification=None,
                            solver_progress_started=True,
                        )
                    )
                    if in_allocation_enabled:
                        in_allocation_license_window_seconds = time.monotonic() - window_started_monotonic
                        _persist_in_allocation_window(
                            config,
                            prepared,
                            tuple(checkout_summaries),
                            window_started_at=window_started_at,
                            window_started_monotonic=window_started_monotonic,
                            outcome="solver_progress_started",
                        )
                    elapsed_before_solver_wait = time.monotonic() - attempt_started_monotonic
                    timeout_seconds = float(config.execution_values["runtime"]["timeout_seconds"])
                    reserve_seconds = float(config.execution_values["runtime"]["graceful_stop_reserve_seconds"])
                    remaining_timeout = timeout_seconds - elapsed_before_solver_wait
                    if remaining_timeout <= reserve_seconds:
                        terminated_exit_code = _terminate_solver_and_wait(process)
                        message = "COMSOL case exhausted its runtime timeout during startup resolution."
                        raise CaseExecutionError(  # noqa: TRY301 -- translated at the owned process boundary
                            message,
                            work_directory=prepared.work_directory,
                            command=tuple(command),
                            exit_code=terminated_exit_code,
                            timed_out=True,
                        )
                    stop_controller = stop_service.SolverStopController(
                        process,
                        prepared.work_directory,
                        timeout_seconds=remaining_timeout,
                        graceful_stop_reserve_seconds=reserve_seconds,
                        monotonic_clock=time.monotonic,
                    )
                    stop_result = stop_controller.wait_for_exit(
                        cancellation_requested=runtime_cancellation_requested,
                        force_requested=runtime_force_cancellation_requested,
                        progress_callback=lambda: _update_runtime_progress(progress_reporter, phase="starting_solver"),
                    )
                    exit_code = stop_result.exit_code
                    timed_out = stop_result.timed_out
                    break
                exit_code = startup_exit_code
            except stop_service.UnexpectedStopStatusContentError as error:
                terminated_exit_code = _terminate_solver_and_wait(process)
                raise error.with_runtime_evidence(
                    exit_code=terminated_exit_code,
                    required_exports_present=_expected_exports_exist(config, prepared.work_directory),
                    replay_available=False,
                ) from error
            except BaseException:
                if process.poll() is None:
                    _terminate_solver_and_wait(process)
                raise
            finally:
                _unregister_solver(process)
        solver_log = _write_solver_log(prepared)
        captured_solver_text = solver_log.read_text(encoding="utf-8", errors="replace")
        license_evidence = license_service.classify_temporary_license_capacity(captured_solver_text)
        solver_started = license_service.solver_progress_started(captured_solver_text)
        expected_exports_exist = _expected_exports_exist(config, prepared.work_directory)
        if license_evidence is not None and not solver_started and not expected_exports_exist:
            attempt_ended_at = datetime.now(timezone.utc)
            attempt_ended_monotonic = time.monotonic()
            checkout_index = len(checkout_summaries) + 1
            if not in_allocation_enabled:
                message = f"COMSOL could not obtain temporary floating-license capacity for {license_evidence.feature!r}."
                raise license_service.TemporaryLicenseCapacityError(
                    message,
                    work_directory=prepared.work_directory,
                    command=tuple(command),
                    exit_code=exit_code,
                    evidence=license_evidence,
                )
            _assert_capacity_retry_workspace_is_clean(config, prepared, solved_models_before)
            if process is None or status_prelaunch is None:
                message = "Capacity-checkout status ownership lacks its exact process or prelaunch evidence."
                raise RuntimeError(message)
            completed_exit_code = process.poll()
            artifact = stop_service.inspect_capacity_checkout_status(
                status_prelaunch,
                process_id=process.pid,
                process_exit_code=completed_exit_code,
                temporary_capacity_classified=True,
                solver_progress_started=False,
                required_exports_exist=False,
                scientific_result_exists=False,
            )
            if artifact is not None:
                campaign_run_id = os.environ.get("GENERATION_CAMPAIGN_RUN_ID")
                job_id = os.environ.get("SLURM_JOB_ID")
                if campaign_run_id is None or job_id is None:
                    message = "Capacity status-artifact recovery requires campaign and Slurm identity."
                    raise RuntimeError(message)
                license_service.record_in_allocation_status_artifact_recovery(
                    config,
                    int(prepared.bundle.case_payload["case_index"]),
                    campaign_run_id=campaign_run_id,
                    job_id=job_id,
                    checkout_started_at=attempt_started_at,
                    checkout_ended_at=attempt_ended_at,
                    hostname=hostname,
                    artifact=artifact,
                    classification=license_evidence,
                    cleanup_state="pending",
                    storage_root=prepared.storage_root,
                )
                stop_service.remove_capacity_checkout_status(artifact)
                license_service.record_in_allocation_status_artifact_recovery(
                    config,
                    int(prepared.bundle.case_payload["case_index"]),
                    campaign_run_id=campaign_run_id,
                    job_id=job_id,
                    checkout_started_at=attempt_started_at,
                    checkout_ended_at=attempt_ended_at,
                    hostname=hostname,
                    artifact=artifact,
                    classification=license_evidence,
                    cleanup_state="complete",
                    storage_root=prepared.storage_root,
                )
                status_artifact_recovery_count += 1
            checkout_summaries.append(
                license_service.InAllocationLicenseCheckoutSummary(
                    checkout_index=checkout_index,
                    started_at=attempt_started_at,
                    ended_at=attempt_ended_at,
                    started_monotonic_seconds=attempt_started_monotonic,
                    ended_monotonic_seconds=attempt_ended_monotonic,
                    process_exit_code=exit_code,
                    classification=license_evidence,
                    solver_progress_started=False,
                )
            )
            if window_deadline is None:
                message = "Enabled in-allocation retry requires a monotonic deadline."
                raise RuntimeError(message)
            remaining = max(0.0, window_deadline - time.monotonic())
            pause_seconds = min(float(in_allocation_policy["pause_after_capacity_failure_seconds"]), remaining)
            if pause_seconds > 0.0:
                time.sleep(pause_seconds)
                in_allocation_pause_seconds += pause_seconds
            continue
        if in_allocation_enabled and (solver_started or expected_exports_exist):
            in_allocation_license_window_seconds = time.monotonic() - window_started_monotonic
            checkout_summaries.append(
                license_service.InAllocationLicenseCheckoutSummary(
                    checkout_index=len(checkout_summaries) + 1,
                    started_at=attempt_started_at,
                    ended_at=datetime.now(timezone.utc),
                    started_monotonic_seconds=attempt_started_monotonic,
                    ended_monotonic_seconds=time.monotonic(),
                    process_exit_code=exit_code,
                    classification=None,
                    solver_progress_started=True,
                )
            )
            _persist_in_allocation_window(
                config,
                prepared,
                tuple(checkout_summaries),
                window_started_at=window_started_at,
                window_started_monotonic=window_started_monotonic,
                outcome="solver_progress_started",
            )
        break
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
        "executable": comsol_service.resolve_comsol_executable(config),
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
    cancelled = stop_result is not None and stop_result.reason == "cancelled"
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
    captured_solver_text = solver_log.read_text(encoding="utf-8", errors="replace")
    license_evidence = license_service.classify_temporary_license_capacity(
        captured_solver_text,
    )
    solver_progress_started = license_service.solver_progress_started(
        captured_solver_text,
    )
    expected_exports_exist = _expected_exports_exist(
        config,
        prepared.work_directory,
    )
    if license_evidence is not None and not solver_progress_started and not expected_exports_exist:
        message = f"COMSOL could not obtain temporary floating-license capacity for {license_evidence.feature!r}."
        raise license_service.TemporaryLicenseCapacityError(
            message,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            evidence=license_evidence,
            solver_progress_started=solver_progress_started,
            expected_exports_exist=expected_exports_exist,
        )
    if exit_code != 0:
        msg = f"COMSOL case exited with status {exit_code}."
        raise CaseExecutionError(
            msg,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
        )
    export_conversion_start = time.monotonic()
    _update_runtime_progress(progress_reporter, phase="collecting_exports", force=True)
    try:
        solved_model, solved_model_evidence = _canonicalize_solved_model(
            prepared.work_directory,
            solved_models_before,
        )
        _record_solved_model_provenance(execution_provenance, solved_model_evidence)
    except (_SolvedModelOutputError, OSError, TypeError, ValueError) as error:
        artifacts = error.artifacts if isinstance(error, _SolvedModelOutputError) else (comsol_service.RETAINED_MODEL_FILENAME,)
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
            failure_stage="exports",
        ) from error
    _update_runtime_progress(progress_reporter, phase="canonicalizing", force=True)
    processing_provenance = _write_processing_provenance(
        config,
        prepared,
        mode="initial",
        solver_git_commit=str(prepared.bundle.case_payload["git_commit"]),
    )
    solver_metrics = _final_solver_metrics(prepared)
    try:
        canonical_case = storage_service.convert_exports_to_hdf5(
            config,
            prepared.bundle.case_payload,
            exports,
            scalar_handoff=prepared.bundle.scalar_handoff,
            work_directory=prepared.work_directory,
            runtime_directory=prepared.runtime_directory,
            runtime_seconds=elapsed,
            solver_metrics=solver_metrics,
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
    export_conversion_seconds = time.monotonic() - export_conversion_start
    source_export_bytes = sum(export.size_bytes for export in exports)
    timing.update(
        {
            "comsol_process_seconds": elapsed,
            "in_allocation_license_window_seconds": in_allocation_license_window_seconds,
            "in_allocation_capacity_pause_seconds": in_allocation_pause_seconds,
            "in_allocation_checkout_attempt_count": len(checkout_summaries),
            "status_artifact_recovery_count": status_artifact_recovery_count,
            "export_conversion_s": export_conversion_seconds,
            "export_conversion_seconds": export_conversion_seconds,
            "complete_execution_s": time.monotonic() - monotonic_start,
            "license_wait_seconds": _prior_license_wait_seconds(config, prepared),
            "source_export_bytes": source_export_bytes,
            "case_hdf5_bytes": canonical_case.path.stat().st_size,
            "solved_model_scratch_bytes": solved_model.stat().st_size,
            "scratch_peak_bytes": _workspace_regular_file_bytes(prepared.work_directory),
            "bytes_hashed_during_conversion": source_export_bytes,
        }
    )
    common.serialization.atomic_write_json(
        prepared.runtime_directory / "timing.json",
        timing,
    )
    return ExecutionResult(
        prepared=prepared,
        command=tuple(command),
        timing=timing,
        exports=exports,
        canonical_case=canonical_case,
        solver_log=solver_log,
        solved_model=(solved_model if retain_solved_model else None),
        execution_provenance=execution_provenance,
        processing_provenance=processing_provenance,
    )


def _artifact_map(
    directory: Path,
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return exact-byte identities and verify declared copied artifacts once."""
    expected = {} if expected_sha256 is None else dict(expected_sha256)
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            msg = f"Published case staging cannot contain symbolic links: {path}"
            raise ValueError(msg)
        if not path.is_file() or path.name in {"provenance.json", "_SUCCESS"}:
            continue
        relative = path.relative_to(directory).as_posix()
        digest = common.serialization.file_sha256(path)
        expected_digest = expected.get(relative)
        if expected_digest is not None and digest != expected_digest:
            msg = f"Copied artifact digest changed during publication: {path}"
            raise RuntimeError(msg)
        artifacts[relative] = {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    missing = set(expected) - set(artifacts)
    if missing:
        msg = f"Expected copied artifacts are missing during publication: {sorted(missing)}"
        raise FileNotFoundError(msg)
    return artifacts


def _complete_stage(
    directory: Path,
    *,
    config: config_contract.GenerationConfig,
    case_payload: dict[str, Any],
    input_generation_id: str,
    stage: str,
    expected_artifact_sha256: Mapping[str, str] | None = None,
) -> None:
    """Write digest-bound publication provenance and final success evidence."""
    artifacts = _artifact_map(
        directory,
        expected_sha256=expected_artifact_sha256,
    )
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
        "input_generation_id": input_generation_id,
        "template_sha256": config.template_sha256,
        "scientific_config_digest": config.scientific_config_digest,
        "export_contract_sha256": case_payload["export_contract_sha256"],
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "retention_policy": config.execution_values["retention_policy"],
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


def _stable_timing_byte_accounting(
    timing: Mapping[str, Any],
    *,
    fixed_payload_bytes: int,
) -> dict[str, Any]:
    """Resolve timing.json's self-referential size without filesystem churn."""
    resolved = dict(timing)
    for _ in range(8):
        serialized = json.dumps(
            resolved,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        persistent_bytes = fixed_payload_bytes + len(f"{serialized}\n".encode())
        if resolved["persistent_case_bytes"] == persistent_bytes and resolved["bytes_hashed_during_publication"] == persistent_bytes:
            return resolved
        resolved["persistent_case_bytes"] = persistent_bytes
        resolved["bytes_hashed_during_publication"] = persistent_bytes
    message = "Case timing byte accounting did not reach a deterministic fixed point."
    raise RuntimeError(message)


def _stage_processed_case(config: config_contract.GenerationConfig, result: ExecutionResult, destination: Path) -> None:
    """Stage one retention-exact canonical post-COMSOL payload with I/O evidence."""
    publication_start = time.monotonic()
    destination.mkdir(parents=True)
    copied_bytes = 0
    fixed_payload_bytes = 0
    retained_export_bytes = 0
    expected_artifact_sha256: dict[str, str] = {}
    if config.execution_values["retention_policy"] == "full":
        export_root = destination / "comsol_exports"
        for export in result.exports:
            target = export_root / export.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(export.source_path, target)
            observed_size = target.stat().st_size
            if observed_size != export.size_bytes:
                msg = f"COMSOL export size changed during publication: {target}"
                raise RuntimeError(msg)
            copied_bytes += export.size_bytes
            fixed_payload_bytes += observed_size
            retained_export_bytes += export.size_bytes
            relative = target.relative_to(destination).as_posix()
            if relative in expected_artifact_sha256:
                msg = f"COMSOL export destination is duplicated during publication: {relative}"
                raise RuntimeError(msg)
            expected_artifact_sha256[relative] = export.sha256
    timing_path = destination / "timing.json"
    copied_sources = (
        (result.canonical_case.path, destination / "case.h5"),
        (result.solver_log, destination / "solver.log"),
        (result.prepared.runtime_directory / "timing.json", timing_path),
        (result.execution_provenance, destination / "execution_provenance.json"),
        (result.processing_provenance, destination / "processing_provenance.json"),
    )
    for source, target in copied_sources:
        source_bytes = source.stat().st_size
        shutil.copy2(source, target)
        observed_size = target.stat().st_size
        if observed_size != source_bytes:
            message = f"Copied case artifact size changed during publication: {target}"
            raise RuntimeError(message)
        copied_bytes += source_bytes
        if target != timing_path:
            fixed_payload_bytes += observed_size
    status = _load_json_object(
        result.canonical_case.status_path,
        label="converted case status",
    )
    stages = status.get("stages")
    if not isinstance(stages, dict) or stages.get("publication") != "pending":
        message = "Converted case status lacks the pending publication stage."
        raise ValueError(message)
    stages["publication"] = "succeeded"
    status_path = common.serialization.atomic_write_json(destination / "status.json", status)
    fixed_payload_bytes += status_path.stat().st_size
    retained_model_bytes = 0
    if config.execution_values["retention_policy"] == "full":
        if result.solved_model is None:
            message = "Full retention completed without an admitted solved model."
            raise FileNotFoundError(message)
        retained_model_bytes = result.solved_model.stat().st_size
        retained_model_path = destination / comsol_service.RETAINED_MODEL_FILENAME
        shutil.copy2(result.solved_model, retained_model_path)
        observed_size = retained_model_path.stat().st_size
        if observed_size != retained_model_bytes:
            message = f"Solved model size changed during publication: {retained_model_path}"
            raise RuntimeError(message)
        copied_bytes += retained_model_bytes
        fixed_payload_bytes += observed_size
    validation_start = time.monotonic()
    storage_service.validate_case_hdf5(
        destination / "case.h5",
        expected_profile=config.profile.id,
    )
    post_validation_seconds = time.monotonic() - validation_start
    timing = _stable_timing_byte_accounting(
        {
            **result.timing,
            "publication_seconds": time.monotonic() - publication_start,
            "post_publication_validation_seconds": post_validation_seconds,
            "persistent_case_bytes": 0,
            "direct_exports_retained_bytes": retained_export_bytes,
            "solved_model_retained_bytes": retained_model_bytes,
            "recovery_payload_bytes": 0,
            "bytes_hashed_during_publication": 0,
            "bytes_hashed_during_post_publication_validation": 0,
            "bytes_copied_during_publication": copied_bytes,
        },
        fixed_payload_bytes=fixed_payload_bytes,
    )
    common.serialization.atomic_write_json(timing_path, timing)
    if fixed_payload_bytes + timing_path.stat().st_size != timing["persistent_case_bytes"]:
        message = f"Case timing byte accounting changed during publication: {timing_path}"
        raise RuntimeError(message)
    _complete_stage(
        destination,
        config=config,
        case_payload=result.prepared.bundle.case_payload,
        input_generation_id=result.prepared.input_generation_id,
        stage="processed",
        expected_artifact_sha256=expected_artifact_sha256,
    )
    if not status_path.is_file():
        message = f"Case status disappeared during staging: {status_path}"
        raise RuntimeError(message)


def _optional_json_object(path: Path) -> dict[str, Any] | None:
    """Load one optional non-symlink JSON object for failure evidence."""
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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


def _attempt_case_state(
    error: BaseException,
    *,
    failure_stage: str,
) -> attempt_service.AttemptCaseState:
    """Classify one precise unsuccessful attempt state without collapsing stages."""
    if isinstance(error, license_service.TemporaryLicenseCapacityError):
        return "license_blocked"
    if bool(getattr(error, "timed_out", False)):
        return "timed_out"
    if isinstance(error, CaseInterruptedError):
        return "cancelled"
    if isinstance(error, (KeyboardInterrupt, InterruptedError)):
        return "interrupted"
    if failure_stage == "exports":
        return "exports_failed"
    if failure_stage == "conversion":
        return "conversion_failed"
    if failure_stage == "publication":
        return "publication_failed"
    return "failed"


def _attempt_quality_flags(
    work_directory: Path | None,
) -> tuple[DiagnosticRecord, ...]:
    """Return any already-persisted advisory quality flags for attempt evidence."""
    if work_directory is None:
        return ()
    status = _optional_json_object(work_directory / "runtime" / "status.json")
    records = status.get("quality_flags") if isinstance(status, dict) else None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        return ()
    return tuple(cast("DiagnosticRecord", dict(record)) for record in records)


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
    """Publish authoritative append-only attempt evidence before scratch cleanup."""
    if failure_stage not in _FAILURE_STAGES:
        message = f"Unsupported case failure stage: {failure_stage!r}"
        raise ValueError(message)
    if scratch_cleanup_status not in {"pending", "not_created"}:
        message = f"Initial scratch cleanup status is invalid: {scratch_cleanup_status!r}"
        raise ValueError(message)
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    run_id = workspace_service.workspace_run_id(config)
    execution = _failure_execution_evidence(
        config,
        error,
        worker_slot=worker_slot,
        scheduler_kind=scheduler_kind,
        allocated_node=allocated_node,
        work_directory=work_directory,
    )
    solver_metrics = {} if work_directory is None else _final_solver_metrics_from_directory(work_directory)
    evidence = attempt_service.publish_case_attempt(
        config,
        case_index,
        campaign_run_id=run_id,
        case_state=_attempt_case_state(error, failure_stage=failure_stage),
        failure_stage=failure_stage,
        reason=str(error) or type(error).__name__,
        solver_git_commit=source_service.required_git_commit(),
        processing_git_commit=source_service.required_git_commit(),
        work_directory=work_directory,
        storage_root=storage,
        worker_slot=worker_slot,
        scheduler_kind=scheduler_kind,
        allocated_node=allocated_node,
        exit_code=(
            execution["exit_code"] if isinstance(execution.get("exit_code"), int) and not isinstance(execution.get("exit_code"), bool) else None
        ),
        timed_out=execution["timed_out"] is True,
        solver_metrics=solver_metrics,
        quality_flags=_attempt_quality_flags(work_directory),
    )
    if scratch_cleanup_status == "not_created":
        attempt_service.record_attempt_cleanup(
            evidence,
            status="not_created",
            reclaimed_bytes=0,
            error=None,
        )
    return evidence.receipt_path


def _report_technical_smoke_failure_artifacts(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    failure_path: Path,
    storage_root: Path,
) -> None:
    """Print the visible bounded attempt path for one failed Technical Smoke."""
    del case_index, storage_root
    if config.scientific_values.get("campaign_purpose") != "technical_runtime_smoke":
        return
    evidence = attempt_service.load_attempt(failure_path.parent)
    print("Retained attempt evidence:", evidence.directory, file=sys.stderr)


def _complete_failure_cleanup(
    path: Path,
    *,
    status: str,
    reclaimed_bytes: int,
    error: str | None,
) -> None:
    """Atomically complete one persisted scratch-cleanup receipt."""
    if status not in {"complete", "failed", "not_created"}:
        message = f"Completed scratch cleanup status is invalid: {status!r}"
        raise ValueError(message)
    if path.name != "attempt.json":
        message = f"Case failure cleanup requires an attempt receipt: {path}"
        raise ValueError(message)
    attempt_service.record_attempt_cleanup(
        attempt_service.load_attempt(path.parent),
        status=status,
        reclaimed_bytes=reclaimed_bytes,
        error=error,
    )


def case_failure_is_recorded(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    execution_run_id: str | None = None,
    git_commit: str | None = None,
) -> bool:
    """Validate authoritative append-only attempt evidence for one case."""
    run_id = (
        workspace_service.workspace_run_id(config)
        if execution_run_id is None
        else common.paths.validate_logical_name(
            execution_run_id,
            label="failure execution_run_id",
        )
    )
    commit = source_service.required_git_commit() if git_commit is None else source_service.validate_git_commit(git_commit)
    attempt = attempt_service.latest_case_attempt(
        config,
        case_index,
        run_id,
        storage_root=workspace_service.resolve_storage_root(
            storage_root,
            create=False,
        ),
    )
    if attempt is None:
        return False
    cleanup = attempt_service.attempt_cleanup_evidence(attempt)
    return (
        (attempt.payload["case_state"] != "license_blocked" or (cleanup is not None and cleanup["status"] == "failed"))
        and attempt.payload["campaign_run_id"] == run_id
        and attempt.payload["solver_git_commit"] == commit
    )


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


def _admit_artifacts(
    directory: Path,
    provenance: Mapping[str, Any],
    *,
    validation_depth: ValidationDepth,
) -> tuple[ArtifactEvidence, ...]:
    """Admit exact membership and hash content at full or deep boundaries."""
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
            or (validation_depth != "routine" and common.serialization.file_sha256(artifact_path) != digest)
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
        git_commit=(None if identity["git_commit"] is None else str(identity["git_commit"])),
        template_relative_path=(None if identity["template_relative_path"] is None else str(identity["template_relative_path"])),
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
        retention_policy=str(identity["retention_policy"]),
    )


def _raw_publication_from_reference(
    reference: admission_service.InputCaseReference,
) -> _PublicationEvidence:
    """Build raw publication evidence from one batch-admitted case reference."""
    directory = reference.case_directory.resolve()
    case_path = directory / "case.json"
    if (
        not case_path.is_file()
        or case_path.is_symlink()
        or case_path.stat().st_size != reference.case_json_size_bytes
        or common.serialization.file_sha256(case_path) != reference.case_json_sha256
    ):
        message = f"Batch-admitted raw case definition is missing, unsafe, or changed: {case_path}"
        raise RuntimeError(message)
    case_payload = reference.case_payload()
    case_service.validate_case_payload_schema(case_payload)
    expected = {
        "case_id": reference.case_id,
        "case_index": reference.case_index,
        "case_input_id": reference.case_input_id,
        "simulation_case_id": reference.simulation_case_id,
        "batch_id": reference.batch_id,
        "batch_identity": reference.batch_identity,
        "simulation_profile": reference.profile_id,
        "material_family": reference.material_family,
        "sampling_regime": reference.sampling_regime,
    }
    if any(case_payload.get(key) != value for key, value in expected.items()):
        message = f"Batch-admitted raw reference disagrees with canonical case evidence: {directory}"
        raise RuntimeError(message)
    raw_artifacts = [
        ArtifactEvidence(
            relative_path="case.json",
            path=case_path.resolve(),
            sha256=reference.case_json_sha256,
            size_bytes=reference.case_json_size_bytes,
        )
    ]
    for filename, identity in sorted(case_payload["input_files"].items()):
        input_path = reference.input_directory / filename
        raw_artifacts.append(
            ArtifactEvidence(
                relative_path=f"inputs/{filename}",
                path=input_path.resolve(),
                sha256=str(identity["sha256"]),
                size_bytes=int(identity["size_bytes"]),
            )
        )
    return _PublicationEvidence(
        directory,
        "raw",
        case_payload,
        {},
        tuple(raw_artifacts),
        None,
    )


def _admit_raw_publication_directory(
    directory: Path,
    *,
    validation_depth: ValidationDepth,
) -> _PublicationEvidence:
    """Admit one raw case through its authoritative immutable batch evidence."""
    if directory.parent.parent.name != "input_generations":
        message = f"Canonical raw case is not scoped by input-generation identity: {directory}"
        raise RuntimeError(message)
    input_generation_id = common.paths.validate_logical_name(
        directory.parent.name,
        label="input_generation_id",
    )
    batch_storage_name = common.paths.validate_logical_name(
        directory.parents[2].name,
        label="batch_storage_name",
    )
    generation_root = directory.parents[4]
    metadata_directory = generation_root / "meta" / batch_storage_name / "input_generations" / input_generation_id
    source = admission_service.admit_input_batch_source(
        metadata_directory,
        raw_directory=directory.parent,
        expected_input_generation_id=input_generation_id,
        validation_depth=("evidence" if validation_depth == "routine" else "full"),
    )
    matches = tuple(reference for reference in source.cases if reference.case_id == directory.name)
    if len(matches) != 1:
        message = f"Input manifest does not declare exactly one raw case {directory.name!r}."
        raise RuntimeError(message)
    return _raw_publication_from_reference(matches[0])


def _require_processed_publication_layout(
    directory: Path,
    *,
    artifact_names: set[str],
    required: set[str],
    retention_policy: str,
) -> None:
    """Require the explicit full or compact processed ownership boundary."""
    expected_top_level = {
        "_SUCCESS",
        "provenance.json",
        *required,
    }
    retained_exports = {name for name in artifact_names if name.startswith("comsol_exports/")}
    if retention_policy == "full":
        expected_top_level.update({"comsol_exports", "solved.mph"})
        exports = directory / "comsol_exports"
        valid = bool(retained_exports) and "solved.mph" in artifact_names and exports.is_dir() and not exports.is_symlink()
    elif retention_policy == "compact":
        valid = not retained_exports and "solved.mph" not in artifact_names and not (directory / "comsol_exports").exists()
    else:
        message = f"Processed publication has an invalid retention policy: {retention_policy!r}."
        raise RuntimeError(message)
    if {entry.name for entry in directory.iterdir()} != expected_top_level or not valid:
        message = f"Processed publication top-level membership is not canonical for {retention_policy}: {directory}"
        raise RuntimeError(message)


def _admit_processed_raw_publication(
    directory: Path,
    provenance: Mapping[str, Any],
    *,
    raw_evidence: _PublicationEvidence | None,
    validation_depth: ValidationDepth,
) -> tuple[str, _PublicationEvidence]:
    """Resolve and admit the exact raw input named by processed evidence."""
    if (
        set(provenance) != _CASE_PUBLICATION_KEYS
        or provenance.get("schema_kind") != _CASE_PUBLICATION_SCHEMA_KIND
        or provenance.get("schema_version") != PUBLICATION_SCHEMA_VERSION
    ):
        message = f"Case publication schema is not current: {directory}"
        raise RuntimeError(message)
    input_generation_id = common.paths.validate_logical_name(
        provenance.get("input_generation_id"),
        label="input_generation_id",
    )
    raw_case = directory.parents[2] / "raw" / directory.parent.name / "input_generations" / input_generation_id / directory.name
    if raw_evidence is not None:
        if raw_evidence.stage != "raw" or raw_evidence.directory != raw_case.resolve():
            message = f"Processed publication raw evidence is bound to another case: {directory}"
            raise RuntimeError(message)
        return input_generation_id, raw_evidence
    return input_generation_id, _admit_raw_publication_directory(
        raw_case,
        validation_depth=validation_depth,
    )


def _require_validation_depth(value: ValidationDepth) -> ValidationDepth:
    """Return one supported explicit publication-validation depth."""
    if value not in {"routine", "full", "deep"}:
        message = f"Unsupported publication validation depth: {value!r}."
        raise ValueError(message)
    return value


def _admit_raw_publication(
    directory: Path,
    *,
    validation_depth: ValidationDepth,
    prior_evidence: _PublicationEvidence | None,
) -> _PublicationEvidence:
    """Dispatch raw admission while rejecting an ambiguous prior binding."""
    if prior_evidence is not None:
        message = "Raw publication admission cannot receive prior raw evidence."
        raise TypeError(message)
    return _admit_raw_publication_directory(
        directory,
        validation_depth=validation_depth,
    )


def _admit_publication_directory(
    directory: Path,
    *,
    stage: str,
    validation_depth: ValidationDepth = "full",
    raw_evidence: _PublicationEvidence | None = None,
) -> _PublicationEvidence:
    """Admit one publication at an explicit integrity-validation depth."""
    validation_depth = _require_validation_depth(validation_depth)
    if stage not in {"raw", "processed"}:
        msg = f"Unsupported case publication stage: {stage!r}."
        raise ValueError(msg)
    if not directory.is_dir() or directory.is_symlink():
        msg = f"Case publication directory is missing or unsafe: {directory}"
        raise FileNotFoundError(msg)
    if stage == "raw":
        return _admit_raw_publication(
            directory,
            validation_depth=validation_depth,
            prior_evidence=raw_evidence,
        )
    success_path = directory / "_SUCCESS"
    provenance_path = directory / "provenance.json"
    success = _load_json_object(success_path, label=f"{stage} case success marker")
    provenance = _load_json_object(provenance_path, label=f"{stage} case publication provenance")
    input_generation_id, admitted_raw = _admit_processed_raw_publication(
        directory,
        provenance,
        raw_evidence=raw_evidence,
        validation_depth=validation_depth,
    )
    case_payload = admitted_raw.case_payload
    try:
        case_service.validate_case_payload_schema(case_payload)
    except (KeyError, TypeError, ValueError) as error:
        msg = f"Canonical case provenance does not match the active exact schema: {directory}"
        raise RuntimeError(msg) from error
    if (
        set(success) != _CASE_SUCCESS_KEYS
        or success.get("schema_kind") != _CASE_SUCCESS_SCHEMA_KIND
        or success.get("schema_version") != PUBLICATION_SCHEMA_VERSION
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
    template_relative_path = templates.validate_template_relative_path(
        template["relative_path"],
        label="case template relative_path",
    )
    template_sha256 = _require_sha256(template["sha256"], label="case template sha256")
    if (
        views != list(profile.available_learning_views)
        or case_payload.get("airflow_source") != profile.airflow_source
        or template["filename"] != Path(template_relative_path).name
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
        "input_generation_id": input_generation_id,
        "template_sha256": template_sha256,
        "scientific_config_digest": scientific_digest,
        "export_contract_sha256": export_contract_sha256,
        "available_learning_views": list(profile.available_learning_views),
        "airflow_source": profile.airflow_source,
        "retention_policy": provenance.get("retention_policy"),
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
    artifacts = _admit_artifacts(
        directory,
        provenance,
        validation_depth=validation_depth,
    )
    required = {
        "case.h5",
        "solver.log",
        "timing.json",
        "status.json",
        "execution_provenance.json",
        "processing_provenance.json",
    }
    artifact_names = {artifact.relative_path for artifact in artifacts}
    retention_policy = provenance.get("retention_policy")
    _require_processed_publication_layout(
        directory,
        artifact_names=artifact_names,
        required=required,
        retention_policy=str(retention_policy),
    )
    if not required.issubset(artifact_names):
        msg = f"Processed publication lacks canonical payload or runtime evidence: {directory}"
        raise RuntimeError(msg)
    timing = _load_json_object(directory / "timing.json", label="case timing")
    execution = _load_json_object(
        directory / "execution_provenance.json",
        label="case execution provenance",
    )
    processing = _load_json_object(
        directory / "processing_provenance.json",
        label="case processing provenance",
    )
    processing_keys = {
        "schema_kind",
        "schema_version",
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "mode",
        "solver_git_commit",
        "processing_git_commit",
        "source_attempt",
        "scientific_config_digest",
        "template_sha256",
        "recorded_at",
    }
    if (
        timing.get("git_commit") != git_commit
        or execution.get("git_commit") != git_commit
        or set(processing) != processing_keys
        or processing.get("schema_kind") != "generation_processing_provenance"
        or processing.get("schema_version") != 1
        or processing.get("case_id") != case_id
        or processing.get("case_input_id") != case_input_id
        or processing.get("simulation_case_id") != simulation_case_id
        or processing.get("mode") not in {"initial", "replay_conversion", "replay_publication"}
        or processing.get("solver_git_commit") != git_commit
        or processing.get("scientific_config_digest") != scientific_digest
        or processing.get("template_sha256") != template_sha256
    ):
        msg = f"Processed solver/processing provenance disagrees in {directory}."
        raise RuntimeError(msg)
    source_service.validate_git_commit(processing.get("processing_git_commit"))
    if validation_depth == "routine":
        hdf5_identity = HDF5IdentityEvidence(
            simulation_profile=profile.id,
            git_commit=None,
            template_relative_path=None,
            template_sha256=template_sha256,
            case_input_id=case_input_id,
            simulation_case_id=simulation_case_id,
            scientific_config_digest=scientific_digest,
            export_contract_sha256=export_contract_sha256,
            available_learning_views=profile.available_learning_views,
            airflow_source=profile.airflow_source,
            retention_policy=str(retention_policy),
        )
    else:
        hdf5_identity = _hdf5_evidence(
            storage_service.validate_case_hdf5(
                directory / "case.h5",
                expected_profile=profile.id,
            )
        )
    expected_hdf5 = HDF5IdentityEvidence(
        simulation_profile=profile.id,
        git_commit=(None if hdf5_identity.git_commit is None else git_commit),
        template_relative_path=(None if hdf5_identity.template_relative_path is None else template_relative_path),
        template_sha256=template_sha256,
        case_input_id=case_input_id,
        simulation_case_id=simulation_case_id,
        scientific_config_digest=scientific_digest,
        export_contract_sha256=export_contract_sha256,
        available_learning_views=profile.available_learning_views,
        airflow_source=profile.airflow_source,
        retention_policy=str(retention_policy),
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
            "relative_path": config.template_relative_path,
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
    validation_depth: ValidationDepth = "full",
    input_reference: admission_service.InputCaseReference | None = None,
) -> dict[str, Any]:
    """Validate one completed case at an explicit integrity depth."""
    raw_evidence = None if input_reference is None else _raw_publication_from_reference(input_reference)
    processed = _admit_publication_directory(
        processed_case_directory(config, case_index, storage_root=storage_root),
        stage="processed",
        validation_depth=validation_depth,
        raw_evidence=raw_evidence,
    )
    _require_publication_matches_config(
        processed,
        config=config,
        case_index=case_index,
    )
    return dict(processed.provenance)


def admit_completed_case(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    validation_depth: ValidationDepth = "full",
    input_reference: admission_service.InputCaseReference | None = None,
    git_commit: str | None = None,
) -> TerminalCaseEvidence:
    """
    Admit one independently published completed case without a terminal batch manifest.

    Parameters
    ----------
    config : GenerationConfig
        Exact resolved batch configuration that owns the case.
    case_index : int
        Configured case index to admit.
    storage_root : Path | str | None, optional
        Storage root containing the raw and processed case publications.
    validation_depth : {"routine", "full", "deep"}, optional
        Integrity depth used for raw and processed publication admission.
    input_reference : InputCaseReference | None, optional
        Exact raw-input evidence already admitted by the caller.
    git_commit : str | None, optional
        Persisted source commit used to bind configured raw-input identity when
        the caller has not already admitted an input reference.

    Returns
    -------
    TerminalCaseEvidence
        Immutable case-local publication, artifact, HDF5, and metadata evidence.

    Raises
    ------
    FileNotFoundError
        If required raw or processed case evidence is absent or unsafe.
    ValueError
        If the configured case index or persisted evidence is malformed.
    RuntimeError
        If raw, processed, configuration, or HDF5 identities disagree.

    Notes
    -----
    This admits one terminal case publication only. It does not create or imply
    terminal batch membership, a batch manifest, or Dataset publication eligibility.

    """
    validation_depth = _require_validation_depth(validation_depth)
    config.case_id(case_index)
    if input_reference is None:
        input_reference = input_service.admit_configured_input_case(
            config,
            case_index,
            storage_root=storage_root,
            git_commit=git_commit,
        )
    raw, processed = _admit_terminal_case_publications(
        input_reference,
        processed_directory=processed_case_directory(
            config,
            case_index,
            storage_root=storage_root,
        ),
        validation_depth=validation_depth,
    )
    _require_publication_matches_config(raw, config=config, case_index=case_index)
    _require_publication_matches_config(processed, config=config, case_index=case_index)
    if raw.case_payload != processed.case_payload:
        message = f"Raw and processed canonical metadata disagree for {config.case_id(case_index)!r}."
        raise RuntimeError(message)
    hdf5_artifacts = tuple(item for item in processed.artifacts if item.relative_path == "case.h5")
    if len(hdf5_artifacts) != 1 or processed.hdf5_identity is None:
        message = f"Completed case lacks one admitted canonical HDF5: {config.case_id(case_index)!r}."
        raise RuntimeError(message)
    return TerminalCaseEvidence(
        case_index=case_index,
        case_id=config.case_id(case_index),
        material_family=config.material_family,
        case_input_id=str(raw.case_payload["case_input_id"]),
        simulation_case_id=str(raw.case_payload["simulation_case_id"]),
        success_sha256=_safe_file_sha256(
            processed.directory / "_SUCCESS",
            label=f"{config.case_id(case_index)} success marker",
        ),
        provenance_sha256=_safe_file_sha256(
            processed.directory / "provenance.json",
            label=f"{config.case_id(case_index)} publication provenance",
        ),
        case_hdf5_sha256=hdf5_artifacts[0].sha256,
        raw_directory=raw.directory,
        processed_directory=processed.directory,
        hdf5_path=(processed.directory / "case.h5").resolve(),
        raw_artifacts=raw.artifacts,
        processed_artifacts=processed.artifacts,
        hdf5_identity=processed.hdf5_identity,
        _case_metadata_json=_canonical_json_text(raw.case_payload),
    )


def _retained_export_hdf5_repair_evidence(
    config: config_contract.GenerationConfig,
    case_index: int,
    processed: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], float]:
    """Admit every completed-publication byte except the invalid HDF5 payload."""
    if config.execution_values["retention_policy"] != "full":
        message = "Completed HDF5 reconstruction requires Full-Retention source exports."
        raise RuntimeError(message)
    if not processed.is_dir() or processed.is_symlink():
        message = f"Completed HDF5 reconstruction source is missing or unsafe: {processed}"
        raise FileNotFoundError(message)
    success_path = processed / "_SUCCESS"
    provenance_path = processed / "provenance.json"
    success = _load_json_object(success_path, label="completed case success marker")
    provenance = _load_json_object(provenance_path, label="completed case publication provenance")
    input_generation_id, raw_evidence = _admit_processed_raw_publication(
        processed,
        provenance,
        raw_evidence=None,
        validation_depth="full",
    )
    case_payload = raw_evidence.case_payload
    _require_case_payload_matches_config(
        case_payload,
        directory=raw_evidence.directory,
        config=config,
        case_index=case_index,
    )
    case_id = config.case_id(case_index)
    expected_success = {
        "stage": "processed",
        "batch_id": config.batch_id,
        "case_id": case_id,
        "case_input_id": case_payload["case_input_id"],
        "simulation_case_id": case_payload["simulation_case_id"],
    }
    if (
        set(success) != _CASE_SUCCESS_KEYS
        or success.get("schema_kind") != _CASE_SUCCESS_SCHEMA_KIND
        or success.get("schema_version") != PUBLICATION_SCHEMA_VERSION
        or any(success.get(key) != value for key, value in expected_success.items())
        or success.get("provenance_sha256") != _safe_file_sha256(provenance_path, label="completed case publication provenance")
    ):
        message = f"Completed HDF5 reconstruction success identity is invalid: {processed}"
        raise RuntimeError(message)
    expected_publication = {
        "stage": "processed",
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "case_id": case_id,
        "case_input_id": case_payload["case_input_id"],
        "simulation_case_id": case_payload["simulation_case_id"],
        "material_family": config.material_family,
        "git_commit": case_payload["git_commit"],
        "input_generation_id": input_generation_id,
        "template_sha256": config.template_sha256,
        "scientific_config_digest": config.scientific_config_digest,
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "retention_policy": "full",
    }
    if any(provenance.get(key) != value for key, value in expected_publication.items()):
        message = f"Completed HDF5 reconstruction publication identity is invalid: {processed}"
        raise RuntimeError(message)
    raw_artifacts = provenance.get("artifacts")
    if not isinstance(raw_artifacts, dict) or "case.h5" not in raw_artifacts:
        message = f"Completed HDF5 reconstruction lacks a bound case.h5 identity: {processed}"
        raise RuntimeError(message)
    artifacts: dict[str, dict[str, Any]] = {}
    for relative, identity in raw_artifacts.items():
        relative_path = Path(relative) if isinstance(relative, str) else Path()
        if (
            not isinstance(relative, str)
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or not isinstance(identity, dict)
            or set(identity) != {"sha256", "size_bytes"}
        ):
            message = f"Completed HDF5 reconstruction artifact identity is malformed: {relative!r}."
            raise RuntimeError(message)
        digest = _require_sha256(identity["sha256"], label=f"reconstruction artifact {relative!r} sha256")
        size_bytes = identity["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            message = f"Completed HDF5 reconstruction artifact size is malformed: {relative!r}."
            raise RuntimeError(message)
        artifacts[relative] = {"sha256": digest, "size_bytes": size_bytes}
    if any(candidate.is_symlink() for candidate in processed.rglob("*")):
        message = f"Completed HDF5 reconstruction source contains symbolic links: {processed}"
        raise RuntimeError(message)
    actual = {
        candidate.relative_to(processed).as_posix()
        for candidate in processed.rglob("*")
        if candidate.is_file() and candidate.name not in {"provenance.json", "_SUCCESS"}
    }
    if actual != set(artifacts):
        message = (
            "Completed HDF5 reconstruction source membership differs from immutable provenance: "
            f"missing={sorted(set(artifacts) - actual)}, extra={sorted(actual - set(artifacts))}."
        )
        raise RuntimeError(message)
    required = {
        "case.h5",
        "solver.log",
        "timing.json",
        "status.json",
        "execution_provenance.json",
        "processing_provenance.json",
    }
    _require_processed_publication_layout(
        processed,
        artifact_names=set(artifacts),
        required=required,
        retention_policy="full",
    )
    if not required.issubset(artifacts):
        message = f"Completed HDF5 reconstruction source lacks required runtime evidence: {processed}"
        raise RuntimeError(message)
    for relative, identity in artifacts.items():
        candidate = processed / relative
        if relative == "case.h5":
            if not candidate.is_file() or candidate.is_symlink():
                message = f"Invalid completed HDF5 is unavailable for atomic repair: {candidate}"
                raise FileNotFoundError(message)
            continue
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size != identity["size_bytes"]
            or common.serialization.file_sha256(candidate) != identity["sha256"]
        ):
            message = f"Non-HDF5 reconstruction source artifact changed: {candidate}"
            raise RuntimeError(message)
    timing = _load_json_object(processed / "timing.json", label="completed case timing")
    execution = _load_json_object(
        processed / "execution_provenance.json",
        label="completed case execution provenance",
    )
    status = _load_json_object(processed / "status.json", label="completed case status")
    runtime_seconds = timing.get("runtime_s")
    execution_result = execution.get("result")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not 0.0 <= float(runtime_seconds) < float("inf")
        or timing.get("git_commit") != case_payload["git_commit"]
        or execution.get("git_commit") != case_payload["git_commit"]
        or not isinstance(execution_result, dict)
        or execution_result.get("state") != "succeeded"
        or status.get("solver_success") is not True
    ):
        message = f"Completed HDF5 reconstruction runtime evidence is not successful: {processed}"
        raise RuntimeError(message)
    export_artifacts = {
        relative.removeprefix("comsol_exports/"): identity for relative, identity in artifacts.items() if relative.startswith("comsol_exports/")
    }
    if not export_artifacts:
        message = f"Completed HDF5 reconstruction has no retained source exports: {processed}"
        raise RuntimeError(message)
    return case_payload, input_generation_id, artifacts["case.h5"], export_artifacts, float(runtime_seconds)


def _completed_hdf5_reconstruction_receipt_path(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    storage_root: Path,
) -> Path:
    """Bind one repair to its exact completed campaign and safe receipt path."""
    manifest = campaign_evidence.load_campaign_run(
        campaign_run_id,
        storage_root=storage_root,
    )
    campaign = campaign_evidence.campaign_from_manifest(manifest)
    if (
        manifest.get("campaign_run_id") != campaign_run_id
        or manifest.get("state") != "complete"
        or campaign.campaign_purpose != "technical_runtime_smoke"
        or config.scientific_values["campaign_purpose"] != "technical_runtime_smoke"
    ):
        message = "Completed HDF5 reconstruction requires the exact terminal Technical Smoke campaign owner."
        raise RuntimeError(message)
    try:
        owned = campaign.batch(config.batch_name)
    except ValueError as error:
        message = "Completed HDF5 reconstruction batch is not owned by the campaign run."
        raise RuntimeError(message) from error
    if (
        owned.batch_id != config.batch_id
        or owned.batch_identity != config.batch_identity
        or owned.scientific_config_digest != config.scientific_config_digest
        or case_index not in owned.case_indices
    ):
        message = "Completed HDF5 reconstruction config differs from its campaign owner."
        raise RuntimeError(message)
    run_directory = campaign_evidence.campaign_run_directory(
        campaign_run_id,
        storage_root=storage_root,
    )
    if not run_directory.is_dir() or run_directory.is_symlink():
        message = f"Completed HDF5 reconstruction campaign evidence is missing or unsafe: {run_directory}"
        raise FileNotFoundError(message)
    receipt_root = run_directory / "hdf5_reconstructions"
    if receipt_root.exists() and (not receipt_root.is_dir() or receipt_root.is_symlink()):
        message = f"Completed HDF5 reconstruction receipt root is unsafe: {receipt_root}"
        raise ValueError(message)
    receipt_directory = receipt_root / config.batch_id
    if receipt_directory.exists() and (not receipt_directory.is_dir() or receipt_directory.is_symlink()):
        message = f"Completed HDF5 reconstruction receipt directory is unsafe: {receipt_directory}"
        raise ValueError(message)
    receipt = receipt_directory / f"{config.case_id(case_index)}.json"
    if receipt.exists() and (not receipt.is_file() or receipt.is_symlink()):
        message = f"Completed HDF5 reconstruction receipt is unsafe: {receipt}"
        raise ValueError(message)
    return receipt


def _repair_completed_case_hdf5_from_retained_exports_locked(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> dict[str, Any]:
    """Reconstruct one invalid completed Full-Retention HDF5 without COMSOL."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    run_id = common.paths.validate_logical_name(campaign_run_id, label="campaign_run_id")
    processed = processed_case_directory(config, case_index, storage_root=storage)
    receipt = _completed_hdf5_reconstruction_receipt_path(
        config,
        case_index,
        campaign_run_id=run_id,
        storage_root=storage,
    )
    try:
        admitted = validate_completed_case(
            config,
            case_index,
            storage_root=storage,
            validation_depth="deep",
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as validation_error:
        initial_error = str(validation_error)
    else:
        return {
            "schema_kind": _SMOKE_HDF5_RECONSTRUCTION_SCHEMA_KIND,
            "schema_version": _SMOKE_HDF5_RECONSTRUCTION_SCHEMA_VERSION,
            "status": "already_valid",
            "campaign_run_id": run_id,
            "batch_id": config.batch_id,
            "case_id": config.case_id(case_index),
            "case_hdf5_sha256": admitted["artifacts"]["case.h5"]["sha256"],
            "receipt": None,
        }
    case_payload, input_generation_id, expected_hdf5, export_artifacts, runtime_seconds = _retained_export_hdf5_repair_evidence(
        config,
        case_index,
        processed,
    )
    prepared: PreparedCase | None = None

    def complete_repair(active: PreparedCase) -> dict[str, Any]:
        """Install one exact reconstruction and record no-solver provenance."""
        for relative, identity in export_artifacts.items():
            source = processed / "comsol_exports" / relative
            destination = active.exports_directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if (
                destination.is_symlink()
                or destination.stat().st_size != identity["size_bytes"]
                or common.serialization.file_sha256(destination) != identity["sha256"]
            ):
                message = f"Retained export changed during HDF5 reconstruction: {source}"
                raise RuntimeError(message)
        exports = collect_exports(config, active)
        observed_exports = {export.relative_path.as_posix() for export in exports}
        if observed_exports != set(export_artifacts):
            message = (
                "Retained exports do not exactly satisfy the current conversion contract: "
                f"missing={sorted(set(export_artifacts) - observed_exports)}, "
                f"extra={sorted(observed_exports - set(export_artifacts))}."
            )
            raise RuntimeError(message)
        reconstructed = storage_service.convert_exports_to_hdf5(
            config,
            case_payload,
            exports,
            scalar_handoff=active.bundle.scalar_handoff,
            work_directory=active.work_directory,
            runtime_directory=active.runtime_directory,
            runtime_seconds=runtime_seconds,
        )
        reconstructed_sha256 = common.serialization.file_sha256(reconstructed.path)
        if reconstructed.path.stat().st_size != expected_hdf5["size_bytes"] or reconstructed_sha256 != expected_hdf5["sha256"]:
            message = (
                "Retained exports reconstructed a valid HDF5, but its bytes do not match "
                "the immutable completed-publication identity; safe in-place repair is impossible."
            )
            raise RuntimeError(message)

        def install_reconstructed_hdf5(temporary: Path) -> None:
            """Copy exact reconstructed bytes into an atomic publication path."""
            shutil.copy2(reconstructed.path, temporary)

        common.serialization.atomic_path_write(
            processed / "case.h5",
            install_reconstructed_hdf5,
        )
        validate_completed_case(
            config,
            case_index,
            storage_root=storage,
            validation_depth="deep",
        )
        recovery = {
            "schema_kind": _SMOKE_HDF5_RECONSTRUCTION_SCHEMA_KIND,
            "schema_version": _SMOKE_HDF5_RECONSTRUCTION_SCHEMA_VERSION,
            "status": "complete",
            "campaign_run_id": run_id,
            "batch_id": config.batch_id,
            "case_id": config.case_id(case_index),
            "case_input_id": case_payload["case_input_id"],
            "simulation_case_id": case_payload["simulation_case_id"],
            "scientific_config_digest": config.scientific_config_digest,
            "original_validation_error": initial_error,
            "source_exports": {relative: dict(identity) for relative, identity in sorted(export_artifacts.items())},
            "reconstructed_hdf5_sha256": reconstructed_sha256,
            "comsol_executed": False,
            "processing_git_commit": source_service.required_git_commit(),
            "recorded_at": _utc_now(),
        }
        if receipt.exists():
            existing = _load_json_object(receipt, label="Smoke HDF5 reconstruction receipt")
            comparable = {key: value for key, value in existing.items() if key not in {"original_validation_error", "recorded_at"}}
            expected = {key: value for key, value in recovery.items() if key not in {"original_validation_error", "recorded_at"}}
            if comparable != expected:
                message = f"Existing Smoke HDF5 reconstruction receipt conflicts: {receipt}"
                raise FileExistsError(message)
        else:
            common.serialization.atomic_write_json(receipt, recovery)
        return {
            **recovery,
            "receipt": receipt.relative_to(storage).as_posix(),
        }

    try:
        prepared = prepare_case_work_directory(
            config,
            case_index,
            storage_root=storage,
            work_root=work_root,
            input_generation_id=input_generation_id,
        )
        return complete_repair(prepared)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as recovery_error:
        message = (
            f"Completed case {config.batch_id}/{config.case_id(case_index)} has invalid HDF5 evidence "
            f"and retained-export reconstruction failed without COMSOL; initial validation: {initial_error}; "
            f"recovery: {recovery_error}"
        )
        raise RuntimeError(message) from recovery_error
    finally:
        if prepared is not None and prepared.work_directory.exists():
            _cleanup_case_attempt(
                config,
                case_index,
                work_directory=prepared.work_directory,
                work_root=prepared.work_root,
                run_id=prepared.workspace_run_id,
                storage_root=storage,
            )


def repair_completed_case_hdf5_from_retained_exports(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
    blocking_lock: bool = True,
) -> dict[str, Any]:
    """Reconstruct one run-bound Full-Retention HDF5 under its case lock."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    with common.locking.exclusive_file_lock(
        case_lock_path(config, case_index, storage_root=storage),
        blocking=blocking_lock,
    ):
        return _repair_completed_case_hdf5_from_retained_exports_locked(
            config,
            case_index,
            campaign_run_id=campaign_run_id,
            storage_root=storage,
            work_root=work_root,
        )


def completed_case_is_valid(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    input_reference: admission_service.InputCaseReference | None = None,
    git_commit: str | None = None,
) -> bool:
    """Return false only when processed completion is absent; corruption fails closed."""
    raw = raw_case_directory(config, case_index, storage_root=storage_root)
    processed = processed_case_directory(config, case_index, storage_root=storage_root)
    if not (processed / "_SUCCESS").exists():
        if raw.exists():
            reference = input_reference or input_service.admit_configured_input_case(
                config,
                case_index,
                storage_root=storage_root,
                git_commit=git_commit,
            )
            _require_publication_matches_config(
                _raw_publication_from_reference(reference),
                config=config,
                case_index=case_index,
            )
        return False
    if not raw.exists():
        message = f"Processed completion exists without canonical raw inputs: {processed}"
        raise RuntimeError(message)
    reference = input_reference or input_service.admit_configured_input_case(
        config,
        case_index,
        storage_root=storage_root,
        git_commit=git_commit,
    )
    validate_completed_case(
        config,
        case_index,
        storage_root=storage_root,
        validation_depth="routine",
        input_reference=reference,
    )
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
    publication_root = common.paths.resolve_generation_case_publications_directory(
        config.batch_storage_name,
        storage_root=storage,
    )
    publication_root.mkdir(parents=True, exist_ok=True)
    staging = workspace_service.create_publication_staging(
        storage_root=storage,
        publication_root=publication_root,
        run_id=result.prepared.workspace_run_id,
        case_id=result.prepared.bundle.case_id,
    )
    processed_stage = staging / "processed"
    raw_destination = result.prepared.canonical_raw_directory
    processed_destination = processed_case_directory(
        config,
        case_index,
        storage_root=storage,
    )
    try:
        _stage_processed_case(config, result, processed_stage)
        existing = _admit_publication_directory(raw_destination, stage="raw")
        _require_publication_matches_config(existing, config=config, case_index=case_index)
        if existing.case_payload["simulation_case_id"] != result.prepared.bundle.simulation_case_id:
            message = f"Canonical raw case belongs to another simulation identity: {raw_destination}"
            raise RuntimeError(message)
        if processed_destination.exists() and (processed_destination / "_SUCCESS").exists():
            msg = f"Refusing to overwrite existing completed case: {processed_destination}"
            raise FileExistsError(msg)
        _quarantine_incomplete(processed_destination, state_root=state_root)
        processed_destination.parent.mkdir(parents=True, exist_ok=True)
        processed_stage.replace(processed_destination)
        validate_completed_case(
            config,
            case_index,
            storage_root=storage,
            validation_depth="routine",
        )
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


def _current_replay_identities(
    config: config_contract.GenerationConfig,
) -> dict[str, str]:
    """Return the narrow conversion and configured replay contract identities."""
    converter_path = Path(storage_service.__file__).resolve()
    if not converter_path.is_file() or converter_path.is_symlink():
        message = f"Replay converter dependency is missing or unsafe: {converter_path}"
        raise RuntimeError(message)
    return {
        "converter_dependency_sha256": common.serialization.file_sha256(converter_path),
        "output_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
        "time_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["time"]),
    }


def _replay_failure_attempt_count(attempt: attempt_service.AttemptEvidence) -> int:
    """Return the validated number of replay-failure receipts in one case history."""
    count = 0
    for directory in attempt.directory.parent.iterdir():
        if (
            directory.is_dir()
            and not directory.is_symlink()
            and directory.name.startswith("attempt_")
            and directory.name[8:].isdigit()
            and (directory / "replay_failure.json").exists()
        ):
            attempt_service.load_attempt(directory)
            count += 1
    return count


def replay_case_postprocessing_status(
    config: config_contract.GenerationConfig,
    attempt: attempt_service.AttemptEvidence | None,
) -> dict[str, Any]:
    """Return read-only replay eligibility without constructing a solver workspace."""
    identities = _current_replay_identities(config)
    if attempt is None:
        return {
            "eligible": False,
            "blocked": False,
            "reason": "no_attempt",
            "evidence_path": None,
            "attempt_count": 0,
            "identities": identities,
        }
    failure = attempt_service.replay_failure_evidence(attempt)
    evidence_path = attempt.directory / "replay_failure.json" if failure is not None else None
    count = _replay_failure_attempt_count(attempt)
    if not attempt.replay_available:
        return {
            "eligible": False,
            "blocked": False,
            "reason": "payload_unavailable",
            "evidence_path": None if evidence_path is None else str(evidence_path),
            "attempt_count": count,
            "identities": identities,
        }
    if failure is not None and all(failure[key] == value for key, value in identities.items()):
        return {
            "eligible": False,
            "blocked": True,
            "reason": "unchanged_replay_identity",
            "evidence_path": str(evidence_path),
            "attempt_count": count,
            "identities": identities,
        }
    return {
        "eligible": True,
        "blocked": False,
        "reason": "identity_changed" if failure is not None else "eligible",
        "evidence_path": None if evidence_path is None else str(evidence_path),
        "attempt_count": count,
        "identities": identities,
    }


def _require_replay_attempt_identity(
    config: config_contract.GenerationConfig,
    case_index: int,
    attempt: attempt_service.AttemptEvidence,
    *,
    campaign_run_id: str,
    storage_root: Path,
) -> str:
    """Require one attempt to remain bound to exact canonical case science."""
    expected = {
        "campaign_run_id": common.paths.validate_logical_name(
            campaign_run_id,
            label="replay campaign_run_id",
        ),
        "batch_storage_name": config.batch_storage_name,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "case_id": config.case_id(case_index),
        "case_index": case_index,
        "scientific_config_digest": config.scientific_config_digest,
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
    }
    if any(attempt.payload.get(key) != value for key, value in expected.items()):
        message = f"Replay attempt identity disagrees with the configured case: {attempt.receipt_path}"
        raise RuntimeError(message)
    input_generation_id = common.paths.validate_logical_name(
        attempt.payload.get("input_generation_id"),
        label="replay input_generation_id",
    )
    reference = input_service.admit_persisted_input_case(
        config,
        case_index,
        input_generation_id,
        storage_root=storage_root,
    )
    if attempt.payload.get("case_input_id") != reference.case_input_id or attempt.payload.get("simulation_case_id") != reference.simulation_case_id:
        message = f"Replay attempt case identity changed: {attempt.receipt_path}"
        raise RuntimeError(message)
    template = attempt.payload.get("template")
    if template != {
        "relative_path": config.template_relative_path,
        "sha256": config.template_sha256,
    }:
        message = f"Replay attempt template identity changed: {attempt.receipt_path}"
        raise RuntimeError(message)
    expected_raw = reference.case_directory.resolve()
    raw_reference = attempt.payload.get("canonical_raw_case")
    if not isinstance(raw_reference, str) or (storage_root / raw_reference).resolve() != expected_raw:
        message = f"Replay attempt canonical raw reference changed: {attempt.receipt_path}"
        raise RuntimeError(message)
    return input_generation_id


def _copy_replay_payload(
    attempt: attempt_service.AttemptEvidence,
    prepared: PreparedCase,
) -> None:
    """Restore only digest-admitted replay inputs into one fresh workspace."""
    required = list(attempt.payload["replay_required_payload"])
    if attempt.payload["retention_policy"] == "full":
        solved_relative = f"payload/{comsol_service.RETAINED_MODEL_FILENAME}"
        if solved_relative in attempt.payload["retained_inventory"]:
            required.append(solved_relative)
    inventory = attempt.payload["retained_inventory"]
    for retained_relative in dict.fromkeys(required):
        if retained_relative not in inventory:
            message = f"Replay-required artifact is undeclared: {retained_relative}"
            raise FileNotFoundError(message)
        relative = Path(retained_relative)
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "payload" or ".." in relative.parts:
            message = f"Replay artifact path is unsafe: {retained_relative}"
            raise ValueError(message)
        source = attempt.directory / relative
        if source.is_symlink():
            message = f"Replay artifact source is a symbolic link: {source}"
            raise ReplayIntegrityError(message)
        if not source.is_file():
            message = f"Replay-required artifact is missing: {source}"
            raise FileNotFoundError(message)
        destination_relative = Path(*relative.parts[1:])
        destination = prepared.work_directory / destination_relative
        if not destination.absolute().is_relative_to(prepared.work_directory.absolute()):
            message = f"Replay artifact escaped the prepared workspace: {destination}"
            raise ReplayIntegrityError(message)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        identity = inventory[retained_relative]
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size != identity["size_bytes"]
            or common.serialization.file_sha256(destination) != identity["sha256"]
        ):
            message = f"Replay artifact changed during restoration: {destination}"
            raise ReplayIntegrityError(message)


def _replayed_canonical_case(
    config: config_contract.GenerationConfig,
    prepared: PreparedCase,
    exports: tuple[CollectedExport, ...],
    attempt: attempt_service.AttemptEvidence,
) -> storage_service.CanonicalCase:
    """Replay conversion or admit a retained converted payload without COMSOL."""
    failure_stage = str(attempt.payload["failure_stage"])
    timing = _load_json_object(
        prepared.runtime_directory / "timing.json",
        label="replay timing",
    )
    runtime_seconds = timing.get("runtime_s")
    if isinstance(runtime_seconds, bool) or not isinstance(runtime_seconds, (int, float)) or float(runtime_seconds) < 0.0:
        message = "Replay timing lacks one non-negative runtime_s value."
        raise ValueError(message)
    if failure_stage == "conversion":
        return storage_service.convert_exports_to_hdf5(
            config,
            prepared.bundle.case_payload,
            exports,
            scalar_handoff=prepared.bundle.scalar_handoff,
            work_directory=prepared.work_directory,
            runtime_directory=prepared.runtime_directory,
            runtime_seconds=float(runtime_seconds),
            solver_metrics=_final_solver_metrics(prepared),
        )
    if failure_stage != "publication":
        message = f"Attempt stage is not replayable: {failure_stage!r}."
        raise ValueError(message)
    case_path = prepared.runtime_directory / "case.h5"
    status_path = prepared.runtime_directory / "status.json"
    storage_service.validate_case_hdf5(
        case_path,
        expected_profile=config.profile.id,
    )
    status = _load_json_object(status_path, label="retained converted status")
    if (
        status.get("schema_kind") != "simulation_case_status"
        or status.get("schema_version") != storage_service.STATUS_SCHEMA_VERSION
        or status.get("case_state") != "successful"
    ):
        message = f"Retained converted status is unsupported: {status_path}"
        raise ValueError(message)
    return storage_service.CanonicalCase(
        path=case_path,
        status_path=status_path,
        status=status,
        source_export_hashes={
            export.relative_path.as_posix(): {
                "role": export.role,
                "sha256": export.sha256,
                "size_bytes": export.size_bytes,
            }
            for export in exports
        },
    )


def _record_replay_failure(
    config: config_contract.GenerationConfig,
    case_index: int,
    source_attempt: attempt_service.AttemptEvidence,
    error: BaseException,
    prepared: PreparedCase,
    *,
    storage_root: Path,
) -> attempt_service.AttemptEvidence:
    """Append one precise replay failure before its fresh scratch is removed."""
    failure_stage = str(source_attempt.payload["failure_stage"])
    case_state: attempt_service.AttemptCaseState = "conversion_failed" if failure_stage == "conversion" else "publication_failed"
    return attempt_service.publish_case_attempt(
        config,
        case_index,
        campaign_run_id=str(source_attempt.payload["campaign_run_id"]),
        case_state=case_state,
        failure_stage=failure_stage,
        reason=f"Postprocessing replay failed: {error}",
        solver_git_commit=str(source_attempt.payload["solver_git_commit"]),
        processing_git_commit=source_service.required_git_commit(),
        work_directory=prepared.work_directory,
        storage_root=storage_root,
        worker_slot=0,
        scheduler_kind="postprocessing_replay",
        allocated_node=socket.gethostname(),
        exit_code=(
            int(source_attempt.payload["process_exit_code"])
            if isinstance(source_attempt.payload.get("process_exit_code"), int)
            and not isinstance(source_attempt.payload.get("process_exit_code"), bool)
            else None
        ),
        timed_out=source_attempt.payload.get("timed_out") is True,
        solver_metrics=_final_solver_metrics(prepared),
        quality_flags=_attempt_quality_flags(prepared.work_directory),
        input_generation_id=str(source_attempt.payload["input_generation_id"]),
    )


def replay_case_postprocessing(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    source_campaign_run_id: str | None = None,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
    blocking_lock: bool = True,
) -> CaseRunOutcome:
    """Replay one admitted conversion/publication attempt without constructing COMSOL."""
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    processed = processed_case_directory(config, case_index, storage_root=storage)
    with common.locking.exclusive_file_lock(
        case_lock_path(config, case_index, storage_root=storage),
        blocking=blocking_lock,
    ):
        if completed_case_is_valid(config, case_index, storage_root=storage):
            return CaseRunOutcome(
                status="skipped",
                case_id=config.case_id(case_index),
                processed_directory=processed,
                work_directory=None,
                message="Case is already successfully processed.",
            )
        run_id = (
            workspace_service.workspace_run_id(config)
            if source_campaign_run_id is None
            else common.paths.validate_logical_name(
                source_campaign_run_id,
                label="source_campaign_run_id",
            )
        )
        attempt = attempt_service.latest_case_attempt(
            config,
            case_index,
            run_id,
            storage_root=storage,
        )
        if attempt is None:
            return CaseRunOutcome(
                status="replay_unavailable",
                case_id=config.case_id(case_index),
                processed_directory=processed,
                work_directory=None,
                message="No unsuccessful attempt evidence is available.",
            )
        input_generation_id = _require_replay_attempt_identity(
            config,
            case_index,
            attempt,
            campaign_run_id=run_id,
            storage_root=storage,
        )
        replay_status = replay_case_postprocessing_status(config, attempt)
        if replay_status["blocked"]:
            return CaseRunOutcome(
                status="replay_blocked",
                case_id=config.case_id(case_index),
                processed_directory=processed,
                work_directory=None,
                message="Replay is blocked until its converter or output/time contract changes.",
            )
        if not replay_status["eligible"]:
            return CaseRunOutcome(
                status="replay_unavailable",
                case_id=config.case_id(case_index),
                processed_directory=processed,
                work_directory=None,
                message="Required postprocessing replay artifacts are unavailable or already consumed.",
            )
        prepared = prepare_case_work_directory(
            config,
            case_index,
            storage_root=storage,
            work_root=work_root,
            input_generation_id=input_generation_id,
        )
        recorded_failure: attempt_service.AttemptEvidence | None = None
        publication_complete = False
        destination = processed
        try:
            _copy_replay_payload(attempt, prepared)
            exports = collect_exports(config, prepared)
            processing_provenance = _write_processing_provenance(
                config,
                prepared,
                mode=("replay_conversion" if attempt.payload["failure_stage"] == "conversion" else "replay_publication"),
                solver_git_commit=str(attempt.payload["solver_git_commit"]),
                source_attempt=attempt,
            )
            canonical_case = _replayed_canonical_case(
                config,
                prepared,
                exports,
                attempt,
            )
            solved_model = prepared.work_directory / comsol_service.RETAINED_MODEL_FILENAME
            result = ExecutionResult(
                prepared=prepared,
                command=(),
                timing=_load_json_object(
                    prepared.runtime_directory / "timing.json",
                    label="replay timing",
                ),
                exports=exports,
                canonical_case=canonical_case,
                solver_log=prepared.runtime_directory / "solver.log",
                solved_model=(
                    solved_model
                    if config.execution_values["retention_policy"] == "full" and solved_model.is_file() and not solved_model.is_symlink()
                    else None
                ),
                execution_provenance=(prepared.runtime_directory / "execution_provenance.json"),
                processing_provenance=processing_provenance,
            )
            destination = publish_completed_case(
                config,
                result,
                storage_root=storage,
            )
            validate_completed_case(
                config,
                case_index,
                storage_root=storage,
                validation_depth="routine",
            )
            publication_complete = True
        except BaseException as error:
            if isinstance(error, (ReplayIntegrityError, FileExistsError)):
                raise
            recorded_failure = _record_replay_failure(
                config,
                case_index,
                attempt,
                error,
                prepared,
                storage_root=storage,
            )
            attempt_service.record_replay_failure(
                recorded_failure,
                attempt,
                converter_dependency_sha256=replay_status["identities"]["converter_dependency_sha256"],
                output_contract_sha256=replay_status["identities"]["output_contract_sha256"],
                time_contract_sha256=replay_status["identities"]["time_contract_sha256"],
                error=error,
            )
            if not isinstance(error, Exception):
                raise
            message = f"Case-local postprocessing replay failed: {error}"
            raise CaseLocalReplayError(message) from error
        else:
            try:
                attempt_service.record_replay_success(
                    attempt,
                    processed_directory=destination,
                    processing_git_commit=source_service.required_git_commit(),
                )
            except BaseException as audit_error:
                _record_case_cleanup_failure(
                    config,
                    case_index,
                    audit_error,
                    work_directory=prepared.work_directory,
                    storage_root=storage,
                )
                message = f"Replay publication is valid, but recovery-payload audit cleanup failed: {audit_error}"
                raise CaseCleanupError(message) from audit_error
        finally:
            if recorded_failure is not None or publication_complete:
                try:
                    reclaimed = _cleanup_case_attempt(
                        config,
                        case_index,
                        work_directory=prepared.work_directory,
                        work_root=prepared.work_root,
                        run_id=prepared.workspace_run_id,
                        storage_root=storage,
                    )
                except BaseException as cleanup_error:
                    if recorded_failure is not None:
                        attempt_service.record_attempt_cleanup(
                            recorded_failure,
                            status="failed",
                            reclaimed_bytes=0,
                            error=str(cleanup_error),
                        )
                    else:
                        _record_case_cleanup_failure(
                            config,
                            case_index,
                            cleanup_error,
                            work_directory=prepared.work_directory,
                            storage_root=storage,
                        )
                    message = f"Persistent replay outcome exists, but marked scratch cleanup failed: {cleanup_error}"
                    raise CaseCleanupError(message) from cleanup_error
                if recorded_failure is not None:
                    attempt_service.record_attempt_cleanup(
                        recorded_failure,
                        status="complete",
                        reclaimed_bytes=reclaimed,
                        error=None,
                    )
        return CaseRunOutcome(
            status="replayed",
            case_id=config.case_id(case_index),
            processed_directory=destination,
            work_directory=prepared.work_directory,
            message=(
                "Conversion replay completed without COMSOL."
                if attempt.payload["failure_stage"] == "conversion"
                else "Publication replay completed without COMSOL or reconversion."
            ),
        )


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
    diagnostic_observer: Callable[[str, PreparedCase, Mapping[str, Any]], None] | None = None,
) -> CaseRunOutcome:
    """Run or integrity-skip one case and always close marked scratch."""
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    progress_reporter = _create_runtime_progress_reporter(
        config,
        case_index,
        scheduler_kind=scheduler_kind,
        storage_root=storage,
    )
    _update_runtime_progress(progress_reporter, phase="preparing", force=True)
    try:
        initialize_batch_metadata(config, storage_root=storage)
    except BaseException:
        _update_runtime_progress(progress_reporter, phase="failed", terminal=True)
        raise
    lock_path = case_lock_path(config, case_index, storage_root=storage)
    with common.locking.exclusive_file_lock(lock_path, blocking=blocking_lock):
        if completed_case_is_valid(config, case_index, storage_root=storage):
            _update_runtime_progress(progress_reporter, phase="completed", terminal=True)
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
                progress_reporter=progress_reporter,
                diagnostic_observer=diagnostic_observer,
            )
            failure_stage = "publication"
            _update_runtime_progress(progress_reporter, phase="publishing", force=True)
            destination = publish_completed_case(
                config,
                result,
                storage_root=storage,
            )
            if diagnostic_observer is not None:
                diagnostic_observer("finished", prepared, {"exit_code": 0, "error": None})
            _update_runtime_progress(progress_reporter, phase="completed", terminal=True)
        except BaseException as error:
            _update_runtime_progress(progress_reporter, phase="failed", terminal=True)
            if diagnostic_observer is not None and prepared is not None:
                diagnostic_observer(
                    "finished",
                    prepared,
                    {"exit_code": getattr(error, "exit_code", None), "error": f"{type(error).__name__}: {error}"},
                )
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
            retry_path: Path | None = None
            temporary_license_error = (
                error
                if isinstance(
                    error,
                    license_service.TemporaryLicenseCapacityError,
                )
                else None
            )
            retry_enabled = scheduler_kind == "slurm" and bool(config.execution_values["runtime"]["temporary_license_retry"]["enabled"])
            if publication_complete:
                _record_case_cleanup_failure(
                    config,
                    case_index,
                    error,
                    work_directory=attempt_directory,
                    storage_root=storage,
                )
            elif temporary_license_error is not None and retry_enabled:
                try:
                    retry_path = license_service.record_temporary_license_wait(
                        config,
                        case_index,
                        temporary_license_error,
                        storage_root=storage,
                    )
                except Exception as retry_error:  # noqa: BLE001 -- failed provenance is terminal
                    failure_path = record_case_failure(
                        config,
                        case_index,
                        retry_error,
                        worker_slot=worker_slot,
                        scheduler_kind=scheduler_kind,
                        allocated_node=allocated_node,
                        work_directory=None,
                        storage_root=storage,
                        scratch_cleanup_status=("pending" if attempt_directory is not None else "not_created"),
                        failure_stage="solver",
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
                _report_technical_smoke_failure_artifacts(
                    config,
                    case_index,
                    failure_path=failure_path,
                    storage_root=storage,
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
                    if failure_path is None and retry_path is not None:
                        failure_path = record_case_failure(
                            config,
                            case_index,
                            cleanup_error,
                            worker_slot=worker_slot,
                            scheduler_kind=scheduler_kind,
                            allocated_node=allocated_node,
                            work_directory=None,
                            storage_root=storage,
                            scratch_cleanup_status="pending",
                            failure_stage="solver",
                        )
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
            if retry_path is not None:
                print(
                    f"Temporary COMSOL license capacity recorded; Slurm allocation will be released: {retry_path}",
                    flush=True,
                )
                return CaseRunOutcome(
                    status="license_blocked",
                    case_id=config.case_id(case_index),
                    processed_directory=processed_case_directory(config, case_index, storage_root=storage),
                    work_directory=None,
                    message="in_allocation_license_window_exhausted",
                )
            raise
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
    batch_storage_name: str,
    input_generation_id: str,
    case_ids: tuple[str, ...],
    *,
    storage_root: Path | str | None,
) -> tuple[Path, Path]:
    """Require the exact input generation and processed batch membership."""
    expected = set(case_ids)
    raw_root = common.paths.resolve_generation_input_generation_raw_directory(
        batch_storage_name,
        input_generation_id,
        storage_root=storage_root,
    )
    processed_root = common.paths.resolve_generated_batch_dir(
        batch_storage_name,
        stage="processed",
        storage_root=storage_root,
    )
    for stage, root in (("raw", raw_root), ("processed", processed_root)):
        entries = tuple(root.iterdir()) if root.is_dir() and not root.is_symlink() else ()
        actual = {entry.name for entry in entries}
        unsafe = sorted(entry.name for entry in entries if not entry.is_dir() or entry.is_symlink())
        if actual != expected or unsafe:
            msg = (
                f"Terminal {stage} batch membership mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}, unsafe={unsafe}."
            )
            raise RuntimeError(msg)
    return raw_root.resolve(), processed_root.resolve()


def finalize_batch(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Validate exact membership and atomically publish the terminal manifest."""
    initialize_batch_metadata(config, storage_root=storage_root)
    records: list[dict[str, Any]] = []
    git_commits: set[str] = set()
    input_generation_ids: set[str] = set()
    input_references = input_service.admit_configured_input_references(
        config,
        storage_root=storage_root,
        validation_depth="evidence",
    )
    for case_index in config.case_indices:
        provenance = validate_completed_case(
            config,
            case_index,
            storage_root=storage_root,
            validation_depth="routine",
            input_reference=input_references[case_index],
        )
        git_commits.add(source_service.validate_git_commit(provenance.get("git_commit")))
        input_generation_ids.add(
            common.paths.validate_logical_name(
                provenance.get("input_generation_id"),
                label="input_generation_id",
            )
        )
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
                "case_hdf5_sha256": provenance["artifacts"]["case.h5"]["sha256"],
            }
        )
    if len(input_generation_ids) != 1:
        msg = f"Completed batch contains multiple input-generation identities: {sorted(input_generation_ids)}."
        raise RuntimeError(msg)
    input_generation_id = next(iter(input_generation_ids))
    _validate_exact_batch_directory_membership(
        config.batch_storage_name,
        input_generation_id,
        tuple(config.case_id(case_index) for case_index in config.case_indices),
        storage_root=storage_root,
    )
    if len(git_commits) != 1:
        msg = f"Completed batch contains multiple source commits: {sorted(git_commits)}."
        raise RuntimeError(msg)
    git_commit = next(iter(git_commits))
    manifest = {
        "schema_kind": "simulation_batch_manifest",
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "simulation_profile": config.profile.id,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "batch_name": config.batch_name,
        "batch_id": config.batch_id,
        "batch_storage_name": config.batch_storage_name,
        "batch_identity": config.batch_identity,
        "campaign_purpose": config.scientific_values["campaign_purpose"],
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "git_commit": git_commit,
        "input_generation_id": input_generation_id,
        "scientific_config_digest": config.scientific_config_digest,
        "template": {"relative_path": config.template_relative_path, "sha256": config.template_sha256},
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
        "intended_case_indices": list(config.case_indices),
        "cases": records,
    }
    meta_directory = batch_meta_directory(config, storage_root=storage_root)
    manifest_path = _immutable_json(meta_directory / "batch_manifest.json", manifest, label="terminal batch manifest")
    success = {
        "schema_kind": "simulation_batch_success",
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
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
    validate_terminal_batch(
        config,
        storage_root=storage_root,
        input_references=input_references,
    )
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
    reference_template = scientific.get("reference_template")
    reference_template_matches = (
        isinstance(reference_template, Mapping)
        and set(reference_template) == {"sha256"}
        and reference_template.get("sha256") == manifest["template"]["sha256"]
    )
    if (
        not reference_template_matches
        or scientific.get("schema_kind") != "resolved_generation_batch"
        or scientific.get("schema_version") != config_contract.CONFIG_SCHEMA_VERSION
        or scientific.get("simulation_profile") != manifest["simulation_profile"]
        or scientific.get("campaign_purpose") != manifest["campaign_purpose"]
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


def _admit_terminal_case_publications(
    reference: admission_service.InputCaseReference,
    *,
    processed_directory: Path,
    validation_depth: ValidationDepth,
) -> tuple[_PublicationEvidence, _PublicationEvidence]:
    """Reuse one admitted raw case while validating its processed publication."""
    raw = _raw_publication_from_reference(reference)
    processed = _admit_publication_directory(
        processed_directory,
        stage="processed",
        validation_depth=validation_depth,
        raw_evidence=raw,
    )
    return raw, processed


def _terminal_input_references(
    *,
    storage_root: Path | str | None,
    batch_storage_name: str,
    input_generation_id: str,
    raw_root: Path,
    case_ids: tuple[str, ...],
    validation_depth: ValidationDepth,
    manifest_path: Path,
    input_references: Mapping[int, admission_service.InputCaseReference] | None,
) -> dict[str, admission_service.InputCaseReference]:
    """Admit or reuse one exact batch input-reference index."""
    if input_references is None:
        metadata_directory = (
            common.paths.get_generation_meta_root(storage_root=storage_root) / batch_storage_name / "input_generations" / input_generation_id
        )
        source = admission_service.admit_input_batch_source(
            metadata_directory,
            raw_directory=raw_root,
            expected_input_generation_id=input_generation_id,
            validation_depth=("evidence" if validation_depth == "routine" else "full"),
        )
        candidates = source.cases
    else:
        candidates = tuple(input_references.values())
    references = {reference.case_id: reference for reference in candidates}
    expected_root = raw_root.resolve()
    invalid = any(
        reference.source_kind != "input_generated"
        or reference.source_id != input_generation_id
        or reference.batch_storage_name != batch_storage_name
        or reference.case_directory.parent.resolve() != expected_root
        for reference in references.values()
    )
    if invalid or set(references) != set(case_ids) or len(references) != len(candidates):
        message = f"Terminal batch input evidence does not cover exact membership: {manifest_path}"
        raise RuntimeError(message)
    return references


def admit_terminal_batch(
    batch_storage_name: str,
    *,
    storage_root: Path | str | None = None,
    validation_depth: ValidationDepth = "full",
    input_references: Mapping[int, admission_service.InputCaseReference] | None = None,
) -> TerminalBatchEvidence:
    """
    Admit one terminal batch without requiring its authored configuration.

    Parameters
    ----------
    batch_storage_name : str
        Flat semantic locator for the generated batch publication.
    storage_root : Path | str | None, optional
        Storage root containing the Generation publication.
    validation_depth : {"routine", "full", "deep"}, optional
        Integrity depth; full and deep rehash every retained artifact.
    input_references : Mapping[int, InputCaseReference] | None, optional
        Exact operation-local raw-input evidence already admitted by the caller.

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
    validation_depth = _require_validation_depth(validation_depth)
    safe_storage_name = common.paths.validate_logical_name(
        batch_storage_name,
        label="batch_storage_name",
    )
    generation_root = (
        common.paths.get_generation_root(
            storage_root=storage_root,
        )
        .expanduser()
        .resolve()
    )
    meta_candidate = (
        common.paths.get_generation_meta_root(
            storage_root=storage_root,
        )
        / safe_storage_name
    )
    if not meta_candidate.is_dir() or meta_candidate.is_symlink():
        msg = f"Terminal batch metadata directory is missing or unsafe: {meta_candidate}"
        raise FileNotFoundError(msg)
    meta_directory = meta_candidate.resolve()
    manifest_path = meta_directory / "batch_manifest.json"
    success_path = meta_directory / "_SUCCESS"
    scientific_path = meta_directory / "resolved_generation_config.json"
    manifest = _load_json_object(manifest_path, label="terminal batch manifest")
    success = _load_json_object(success_path, label="terminal batch success marker")
    scientific = _load_json_object(
        scientific_path,
        label="resolved scientific generation configuration",
    )
    manifest_sha256 = _safe_file_sha256(
        manifest_path,
        label="terminal batch manifest",
    )
    if (
        set(manifest) != _BATCH_MANIFEST_KEYS
        or manifest.get("schema_kind") != _BATCH_MANIFEST_SCHEMA_KIND
        or manifest.get("schema_version") != BATCH_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("batch_storage_name") != safe_storage_name
    ):
        msg = f"Terminal batch manifest schema, locator, or completion state is invalid: {manifest_path}"
        raise RuntimeError(msg)
    batch_id = manifest.get("batch_id")
    if not isinstance(batch_id, str):
        msg = f"Terminal batch_id is malformed: {manifest_path}"
        raise TypeError(msg)
    safe_batch_id = common.paths.validate_logical_name(batch_id, label="batch_id")
    batch_identity = _require_sha256(manifest.get("batch_identity"), label="terminal batch_identity")
    scientific_digest = _require_sha256(
        manifest.get("scientific_config_digest"),
        label="terminal scientific_config_digest",
    )
    export_contract_sha256 = _require_sha256(
        manifest.get("export_contract_sha256"),
        label="terminal export_contract_sha256",
    )
    if batch_identity != scientific_digest or config_contract.compute_scientific_config_digest(scientific) != scientific_digest:
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
    template_relative_path = templates.validate_template_relative_path(
        template["relative_path"],
        label="terminal template relative_path",
    )
    template_sha256 = _require_sha256(template["sha256"], label="terminal template sha256")
    if manifest.get("available_learning_views") != list(profile.available_learning_views) or manifest.get("airflow_source") != profile.airflow_source:
        msg = f"Terminal profile or template descriptor is invalid: {manifest_path}"
        raise RuntimeError(msg)
    material_family = materials.validate_material_family(manifest.get("material_family"))
    sampling_regime = manifest.get("sampling_regime")
    campaign_purpose = manifest.get("campaign_purpose")
    batch_name = manifest.get("batch_name")
    batch_kind = config_contract.PILOT_CAMPAIGN_PURPOSE if campaign_purpose == config_contract.PILOT_CAMPAIGN_PURPOSE else sampling_regime
    identity_is_valid = False
    if (
        isinstance(material_family, str)
        and isinstance(sampling_regime, str)
        and isinstance(campaign_purpose, str)
        and isinstance(batch_name, str)
        and isinstance(batch_kind, str)
    ):
        expected_name = config_contract.build_batch_name(
            profile.id,
            material_family,
            batch_kind,
        )
        expected_storage_name = config_contract.build_batch_storage_name(
            profile.id,
            material_family,
            sampling_regime,
            campaign_purpose,
            scientific_digest,
        )
        identity_is_valid = (
            batch_name == expected_name
            and safe_batch_id == config_contract.build_batch_id(expected_name, scientific_digest)
            and safe_storage_name == expected_storage_name
            and scientific.get("campaign_purpose") == campaign_purpose
        )
    if not sampling_regime or not identity_is_valid:
        msg = f"Terminal batch name, purpose, storage locator, or immutable identifier is invalid: {manifest_path}"
        raise RuntimeError(msg)
    git_commit = source_service.validate_git_commit(manifest.get("git_commit"))
    input_generation_id = common.paths.validate_logical_name(
        manifest.get("input_generation_id"),
        label="input_generation_id",
    )
    if set(success) != _BATCH_SUCCESS_KEYS or success != {
        "schema_kind": _BATCH_SUCCESS_SCHEMA_KIND,
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
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
        safe_storage_name,
        input_generation_id,
        case_ids,
        storage_root=storage_root,
    )
    references = _terminal_input_references(
        storage_root=storage_root,
        batch_storage_name=safe_storage_name,
        input_generation_id=input_generation_id,
        raw_root=raw_root,
        case_ids=case_ids,
        validation_depth=validation_depth,
        manifest_path=manifest_path,
        input_references=input_references,
    )
    admitted_cases: list[TerminalCaseEvidence] = []
    for expected_index, expected_case_id, raw_record in zip(
        indices,
        case_ids,
        records,
        strict=True,
    ):
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
        raw, processed = _admit_terminal_case_publications(
            references[expected_case_id],
            processed_directory=processed_root / expected_case_id,
            validation_depth=validation_depth,
        )
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
        hdf5_artifacts = tuple(artifact for artifact in processed.artifacts if artifact.relative_path == "case.h5")
        if len(hdf5_artifacts) != 1:
            message = f"Terminal case lacks one declared canonical HDF5: {expected_case_id}."
            raise RuntimeError(message)
        processed_hdf5_sha256 = hdf5_artifacts[0].sha256
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
        batch_storage_name=safe_storage_name,
        batch_identity=batch_identity,
        campaign_purpose=str(campaign_purpose),
        material_family=str(material_family),
        sampling_regime=str(sampling_regime),
        git_commit=git_commit,
        input_generation_id=input_generation_id,
        scientific_config_digest=scientific_digest,
        template_relative_path=template_relative_path,
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
    validation_depth: ValidationDepth = "full",
    input_references: Mapping[int, admission_service.InputCaseReference] | None = None,
) -> dict[str, Any]:
    """Admit a terminal batch and require exact authored-config agreement."""
    evidence = admit_terminal_batch(
        config.batch_storage_name,
        storage_root=storage_root,
        validation_depth=validation_depth,
        input_references=input_references,
    )
    expected = {
        "simulation_profile": config.profile.id,
        "available_learning_views": config.profile.available_learning_views,
        "airflow_source": config.profile.airflow_source,
        "batch_name": config.batch_name,
        "batch_id": config.batch_id,
        "batch_storage_name": config.batch_storage_name,
        "batch_identity": config.batch_identity,
        "campaign_purpose": config.scientific_values["campaign_purpose"],
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "scientific_config_digest": config.scientific_config_digest,
        "template_relative_path": config.template_relative_path,
        "template_sha256": config.template_sha256,
        "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
        "cases": tuple(config.case_indices),
    }
    actual = {
        "simulation_profile": evidence.simulation_profile,
        "available_learning_views": evidence.available_learning_views,
        "airflow_source": evidence.airflow_source,
        "batch_name": evidence.batch_name,
        "batch_id": evidence.batch_id,
        "batch_storage_name": evidence.batch_storage_name,
        "batch_identity": evidence.batch_identity,
        "campaign_purpose": evidence.campaign_purpose,
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
