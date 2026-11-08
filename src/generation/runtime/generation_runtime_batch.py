"""Run, admit, and atomically publish isolated profile-qualified COMSOL cases.

Responsibilities:
  - Execute safe one-node COMSOL commands and retain complete runtime evidence
  - Classify a solve the runtime stopped as stalled, timed_out, or solver_error, with a human detail of its progress
  - Collect explicit raw adapters and convert them to canonical case.h5
  - Publish non-authoritative case phases without changing solver outcomes
  - Publish cases, recover hash-bound Full-Retention HDF5, and admit terminal batches

Design principles:
  - Scientific configuration and execution provenance are physically separate
  - Successful CSV and solved-model retention is explicit and off by default
  - Failed scratch is removed only after policy-bound durable evidence is complete
  - The stall rule applies only to a profile with a time horizon: a stationary solve reports no simulated time, so only the timeout bounds it

This module does NOT:
  - Modify COMSOL templates or infer internal tags, expressions, or signs
  - Publish a parallel canonical CSV learning view

Notes:
  Case completion:
    - A staged raw or processed case writes hash-bound `provenance.json` before `_SUCCESS`; the marker binds the provenance digest, whose document
      holds the artifact inventory used for admission.
    - `run_case` integrity-skips an already valid completion; otherwise it prepares, executes, publishes and re-admits the processed case before
      removing marked scratch. Failure evidence or an eligible license-wait receipt is recorded before scratch cleanup.

"""

from __future__ import annotations

import hashlib
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
from datetime import UTC, datetime
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
from src.generation.publication import generation_publication_storage as storage_service

from . import generation_runtime_comsol as comsol_service
from . import generation_runtime_comsol_timing as comsol_timing
from . import generation_runtime_dispatch_identity as dispatch_identity
from . import generation_runtime_license as license_service
from . import generation_runtime_progress as progress_service
from . import generation_runtime_stop as stop_service
from . import generation_runtime_workspace as workspace_service
from .generation_runtime_preparation import PreparedCase, prepare_case_work_directory

if TYPE_CHECKING:
    from src.generation.validation.generation_validation_policy import DiagnosticRecord

BATCH_MANIFEST_SCHEMA_VERSION = 1
_EXECUTION_PROVENANCE_SCHEMA_VERSION: Final = 1
_PROCESSING_PROVENANCE_SCHEMA_VERSION: Final = 1
# The stall rule needs minutes-scale freshness; rereading the solver stdout every poll would only cost I/O.
_STALL_PROBE_REFRESH_SECONDS: Final = 15.0
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
_FAILURE_STAGES = frozenset({"input", "solver", "exports", "conversion", "publication"})
# How the runtime itself classified a solve it ended: its stall rule, its timeout, or a non-zero COMSOL exit.
RuntimeFailureReason = Literal["stalled", "timed_out", "solver_error"]
# A launcher shell reports death by signal N as exit status 128 + N; a direct child reports -N.
_SHELL_SIGNAL_EXIT_BASE = 128  # a launcher shell reports a death by signal n as 128 + n


class CaseExecutionError(RuntimeError):
    """Report one failed case with structured execution evidence and, when the runtime classified the stop itself, its reason."""

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
        runtime_reason: RuntimeFailureReason | None = None,
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
        self.runtime_reason = runtime_reason


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
    """Create optional exact campaign progress; only I/O failures of the monitor are tolerated, never identity defects."""
    if scheduler_kind != "slurm":
        return None
    dispatch = dispatch_identity.current_case_dispatch()
    if dispatch is None:
        return None
    run_id = common.paths.validate_logical_name(os.environ.get(dispatch_identity.CAMPAIGN_RUN_ID_ENVIRONMENT), label="campaign_run_id")
    try:
        return progress_service.RuntimeProgressReporter.create(
            run_id,
            batch_name=config.batch_name,
            batch_id=config.batch_id,
            case_index=case_index,
            case_id=config.case_id(case_index),
            dispatch=dispatch,
            storage_root=storage_root,
        )
    except progress_service.RuntimeProgressIdentityError:
        raise
    except Exception as error:  # noqa: BLE001 -- monitoring setup I/O cannot terminate a case
        print(f"Runtime progress is unavailable for dispatch {dispatch.dispatch_key}: {error}", file=sys.stderr)
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
    """Describe one admitted terminal case with optional retained raw evidence."""

    case_index: int
    case_id: str
    material_family: str
    case_input_id: str
    simulation_case_id: str
    success_sha256: str
    provenance_sha256: str
    case_hdf5_sha256: str
    raw_directory: Path | None
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
    return datetime.now(UTC).isoformat()


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
    """Return one case's immutable input directory, the raw publication stage."""
    return common.paths.resolve_generation_inputs_directory(
        config.batch_storage_name,
        config.case_id(case_index),
        storage_root=storage_root,
    )


def processed_case_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return one case's permanent result directory, the processed publication stage."""
    return common.paths.resolve_generation_result_directory(
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
    """Initialize immutable Source batch science; campaign records own execution."""
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    directory = batch_meta_directory(config, storage_root=storage)
    directory.mkdir(parents=True, exist_ok=True)
    scientific_path = _immutable_json(
        directory / "resolved_generation_config.json",
        config.scientific_values,
        label="resolved scientific generation configuration",
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


def _solver_log_text(prepared: PreparedCase) -> str:
    """Combine bounded head-and-tail evidence from case-owned solver logs; licence detection reads it, a success retains it."""
    stdout_path = prepared.runtime_directory / "stdout.log"
    stderr_path = prepared.runtime_directory / "stderr.log"
    per_stream = _MAX_RETAINED_SOLVER_LOG_BYTES // 2
    payload = "===== stdout =====\n" + _bounded_log_text(stdout_path, maximum_bytes=per_stream)
    if not payload.endswith("\n"):
        payload += "\n"
    payload += "===== stderr =====\n" + _bounded_log_text(stderr_path, maximum_bytes=per_stream)
    if not payload.endswith("\n"):
        payload += "\n"
    return payload


class _SimulatedTimeProbe:
    """Read one solver stdout incrementally and report its latest simulated time in seconds, at most every few seconds."""

    def __init__(self, stdout_path: Path, *, refresh_seconds: float = _STALL_PROBE_REFRESH_SECONDS) -> None:
        """Bind the probe to one case-owned stdout file without reading it yet."""
        self._path = stdout_path
        self._refresh_seconds = refresh_seconds
        self._offset = 0
        self._trailing = ""
        self._parser = progress_service.ComsolProgressParser()
        self._read_at: float | None = None
        self._latest: float | None = None

    def __call__(self) -> float | None:
        """Return the latest simulated time, reading only the bytes appended since the previous read."""
        now = time.monotonic()
        if self._read_at is not None and now - self._read_at < self._refresh_seconds:
            return self._latest
        self._read_at = now
        try:
            with self._path.open("rb") as stream:
                stream.seek(self._offset)
                appended = stream.read()
        except OSError:
            return self._latest
        self._offset += len(appended)
        lines = (self._trailing + appended.decode("utf-8", errors="replace")).split("\n")
        self._trailing = lines.pop()
        value = self._parser.consume(lines).get("simulated_time_seconds")
        if isinstance(value, (int, float)):
            self._latest = float(value)
        return self._latest


def _horizon_hours(config: config_contract.GenerationConfig) -> float | None:
    """Return the physical horizon in hours of a profile that integrates over time, or ``None`` for a stationary profile (it has no time section)."""
    time_section = config.scientific_values.get("time")
    return None if time_section is None else config_contract.physical_horizon_hours(time_section)


def _stall_stop(config: config_contract.GenerationConfig) -> dict[str, Any] | None:
    """Return the execution's stall rule for a profile with a time horizon; a stationary solve never reports simulated time, so none applies to it."""
    return None if _horizon_hours(config) is None else config.execution_values["runtime"].get("stall_stop")


def _stall_rule(config: config_contract.GenerationConfig, prepared: PreparedCase) -> stop_service.StallRule | None:
    """Build the stall rule over the solver stdout of one prepared case, or ``None`` when the profile or the execution sets none."""
    rule = _stall_stop(config)
    if rule is None:
        return None
    return stop_service.StallRule(
        check_after_seconds=float(rule["check_after_seconds"]),
        minimum_simulated_seconds=float(rule["minimum_simulated_seconds"]),
        simulated_seconds=_SimulatedTimeProbe(prepared.runtime_directory / "stdout.log"),
        # A separate uncached reader, so recording the first minimum time never changes what the stall decision reads.
        observed_simulated_seconds=_SimulatedTimeProbe(prepared.runtime_directory / "stdout.log", refresh_seconds=0.0),
    )


def _stall_progress(config: config_contract.GenerationConfig, stop_result: stop_service.StopResult | None) -> dict[str, float | None] | None:
    """Return the timing record of when the case first reached the stall rule's minimum, or ``None`` when no rule applies to the case."""
    rule = _stall_stop(config)
    if rule is None:
        return None
    return {
        "minimum_simulated_seconds": float(rule["minimum_simulated_seconds"]),
        "reached_after_seconds": None if stop_result is None else stop_result.minimum_simulated_reached_after_seconds,
    }


def _write_solver_step_history(prepared: PreparedCase) -> Path | None:
    """Persist the complete transient time-stepping history of the solver stdout as CSV, if it has one."""
    stdout_path = prepared.runtime_directory / "stdout.log"
    if not stdout_path.is_file() or stdout_path.is_symlink():
        return None
    with stdout_path.open(encoding="utf-8", errors="replace") as stream:
        rows = progress_service.transient_step_history(stream)
    if not rows:
        return None
    lines = [",".join(progress_service.TRANSIENT_STEP_HISTORY_COLUMNS), *(",".join(str(value) for value in row) for row in rows)]
    path = prepared.runtime_directory / "solver_steps.csv"
    common.serialization.atomic_write_text(path, "\n".join(lines) + "\n")
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
        "schema_version": _EXECUTION_PROVENANCE_SCHEMA_VERSION,
        "case_id": prepared.bundle.case_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "execution_config_digest": execution_digest,
        "git_commit": source_service.required_git_commit(),
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
    solver_git_commit: str,
) -> Path:
    """Persist distinct solver and postprocessing source provenance."""
    solver_commit = source_service.validate_git_commit(solver_git_commit)
    processing_commit = source_service.required_git_commit()
    payload = {
        "schema_kind": "generation_processing_provenance",
        "schema_version": _PROCESSING_PROVENANCE_SCHEMA_VERSION,
        "case_id": prepared.bundle.case_id,
        "case_input_id": prepared.bundle.case_input_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "mode": "initial",
        "solver_git_commit": solver_commit,
        "processing_git_commit": processing_commit,
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


def _runtime_failure_detail(
    config: config_contract.GenerationConfig, prepared: PreparedCase, reason: RuntimeFailureReason | None, solving_seconds: float, *, cause: str = ""
) -> str:
    """Return the human detail of a solve the runtime ended: its reason (none when unclassified), the solver progress it reported, and after how long.

    The progress is the simulated time of the horizon for a case that integrates over time, else the last stationary nonlinear iteration.
    """
    metrics = _final_solver_metrics(prepared)
    simulated = metrics.get("simulated_time_seconds")
    iteration = metrics.get("nonlinear_iteration")
    horizon = _horizon_hours(config)
    if isinstance(simulated, (int, float)):
        reached = f"{simulated / 3600:.2f} h{'' if horizon is None else f' of {horizon:g} h'} simulated"
    elif isinstance(iteration, int):
        reached = f"nonlinear iteration {iteration}"
    else:
        reached = "no progress reported"
    detail = f"{'' if reason is None else f'{reason}: '}{cause}{reached} after {solving_seconds:.0f} s solving"
    if reason != "stalled":
        return detail
    rule = config.execution_values["runtime"]["stall_stop"]
    return f"{detail} (threshold: {float(rule['minimum_simulated_seconds']) / 3600:.3g} h simulated after {float(rule['check_after_seconds']):g} s)"


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
    dispatch = dispatch_identity.current_case_dispatch()
    if campaign_run_id is None or dispatch is None:
        return None
    result = license_service.in_allocation_license_window_result(
        config,
        int(prepared.bundle.case_payload["case_index"]),
        campaign_run_id=campaign_run_id,
        dispatch_key=dispatch.dispatch_key,
        hostname=socket.gethostname(),
        window_started_at=window_started_at,
        window_ended_at=datetime.now(UTC),
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
) -> ExecutionResult:
    """Run one isolated COMSOL process and create its validated canonical HDF5."""
    scalar_handoff = prepared.bundle.scalar_handoff
    if scalar_handoff is not None:
        scalar_handoff_contract.validate_transient_scalar_source(scalar_handoff)
    batch_log = prepared.runtime_directory / comsol_timing.COMSOL_BATCH_LOG_FILENAME
    command = comsol_service.build_comsol_command(
        config,
        cores_per_case=cores_per_case,
        scalar_handoff=scalar_handoff,
        scheduler_kind=scheduler_kind,
        batch_log_path=str(batch_log),
        scratch_directory=str(prepared.comsol_directory),
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
    retry_policy = config.execution_values["runtime"]["license_wait"]
    in_allocation_policy = retry_policy["in_allocation_retry"]
    in_allocation_enabled = bool(scheduler_kind == "slurm" and retry_policy["enabled"] and in_allocation_policy["enabled"])
    window_started_at = datetime.now(UTC)
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
        attempt_started_at = datetime.now(UTC)
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
            batch_log.unlink(missing_ok=True)
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
                        stall_rule=_stall_rule(config, prepared),
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
                            ended_at=datetime.now(UTC),
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
                            ended_at=datetime.now(UTC),
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
                        raise CaseExecutionError(  # noqa: TRY301 -- translated at the owned process boundary
                            _runtime_failure_detail(config, prepared, "timed_out", elapsed_before_solver_wait),
                            work_directory=prepared.work_directory,
                            command=tuple(command),
                            exit_code=terminated_exit_code,
                            timed_out=True,
                            runtime_reason="timed_out",
                        )
                    stop_controller = stop_service.SolverStopController(
                        process,
                        prepared.work_directory,
                        timeout_seconds=remaining_timeout,
                        graceful_stop_reserve_seconds=reserve_seconds,
                        monotonic_clock=time.monotonic,
                        stall_rule=_stall_rule(config, prepared),
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
                ) from error
            except BaseException:
                if process.poll() is None:
                    _terminate_solver_and_wait(process)
                raise
            finally:
                _unregister_solver(process)
        captured_solver_text = _solver_log_text(prepared)
        license_evidence = license_service.classify_temporary_license_capacity(captured_solver_text)
        solver_started = license_service.solver_progress_started(captured_solver_text)
        expected_exports_exist = _expected_exports_exist(config, prepared.work_directory)
        if license_evidence is not None and not solver_started and not expected_exports_exist:
            attempt_ended_at = datetime.now(UTC)
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
                dispatch = dispatch_identity.current_case_dispatch()
                if campaign_run_id is None or dispatch is None:
                    message = "Capacity status-artifact recovery requires campaign dispatch identity."
                    raise RuntimeError(message)
                license_service.record_in_allocation_status_artifact_recovery(
                    config,
                    int(prepared.bundle.case_payload["case_index"]),
                    campaign_run_id=campaign_run_id,
                    dispatch_key=dispatch.dispatch_key,
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
                    dispatch_key=dispatch.dispatch_key,
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
                    ended_at=datetime.now(UTC),
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
    solver_timing = comsol_timing.parse_comsol_batch_log(
        batch_log,
        simulation_profile=config.profile.id,
    )
    if solver_timing.status != "complete":
        detail = "; ".join(solver_timing.diagnostics) or "required top-level solver timing was not found"
        print(
            f"Warning: COMSOL solver timing is {solver_timing.status} for {prepared.bundle.case_id}: {detail}",
            file=sys.stderr,
        )
    timing = {
        "schema_kind": "simulation_case_timing",
        "schema_version": storage_service.PUBLICATION_SCHEMA_VERSION,
        "batch_id": config.batch_id,
        "case_id": prepared.bundle.case_id,
        "case_input_id": prepared.bundle.case_input_id,
        "simulation_case_id": prepared.bundle.simulation_case_id,
        "simulation_profile": config.profile.id,
        "git_commit": source_service.required_git_commit(),
        "executable": comsol_service.resolve_comsol_executable(config),
        "arguments": command,
        "working_directory": str(prepared.work_directory),
        "hostname": hostname,
        "process_id": None if process is None else process.pid,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "runtime_s": elapsed,
        "comsol_process_seconds": elapsed,
        **solver_timing.timing_fields(),
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
        "stall_progress": _stall_progress(config, stop_result),
        "template_sha256": config.template_sha256,
    }
    common.serialization.atomic_write_json(prepared.runtime_directory / "timing.json", timing)
    cancelled = stop_result is not None and stop_result.reason == "cancelled"
    stalled = stop_result is not None and stop_result.stalled
    _complete_execution_provenance(
        execution_provenance,
        state=("cancelled" if cancelled else "timed_out" if timed_out else "stalled" if stalled else "succeeded" if exit_code == 0 else "failed"),
        exit_code=exit_code,
        timed_out=timed_out,
        started_at=started_at,
        ended_at=timing["ended_at"],
        runtime_seconds=elapsed,
    )
    captured_solver_text = _solver_log_text(prepared)
    _write_solver_step_history(prepared)
    if cancelled:
        message = "Campaign cancellation terminated the COMSOL process."
        raise CaseInterruptedError(
            message,
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
        )
    if timed_out:
        raise CaseExecutionError(
            _runtime_failure_detail(config, prepared, "timed_out", elapsed),
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            timed_out=True,
            runtime_reason="timed_out",
        )
    if stalled:
        raise CaseExecutionError(
            _runtime_failure_detail(config, prepared, "stalled", elapsed),
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            runtime_reason="stalled",
        )
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
        # A signal exit (negative status, or 128 + signal from the launcher shell) is typically a memory-limit kill: it carries no
        # runtime reason, so the campaign leaves the case unclassified for an explicit decision instead of replacing it silently.
        if exit_code is not None and (exit_code < 0 or 0 < exit_code - _SHELL_SIGNAL_EXIT_BASE < signal.NSIG):
            signal_number = -exit_code if exit_code < 0 else exit_code - _SHELL_SIGNAL_EXIT_BASE
            cause = f"COMSOL was terminated by signal {signal_number} (exit status {exit_code}) at "
            detail = f"{_runtime_failure_detail(config, prepared, None, elapsed, cause=cause)}; a memory-limit kill ends like this"
            raise CaseExecutionError(
                detail,
                work_directory=prepared.work_directory,
                command=tuple(command),
                exit_code=exit_code,
            )
        raise CaseExecutionError(
            _runtime_failure_detail(config, prepared, "solver_error", elapsed, cause=f"COMSOL exited with status {exit_code} at "),
            work_directory=prepared.work_directory,
            command=tuple(command),
            exit_code=exit_code,
            runtime_reason="solver_error",
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
        solver_git_commit=source_service.required_git_commit(),
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
    solver_log = prepared.runtime_directory / "solver.log"
    common.serialization.atomic_write_text(solver_log, captured_solver_text)
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
        "schema_version": storage_service.PUBLICATION_SCHEMA_VERSION,
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
        "schema_version": storage_service.PUBLICATION_SCHEMA_VERSION,
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
    evidence_directory = destination / common.paths.GENERATION_EVIDENCE_DIRNAME
    evidence_directory.mkdir()
    copied_bytes = 0
    fixed_payload_bytes = 0
    retained_export_bytes = 0
    expected_artifact_sha256: dict[str, str] = {}
    if config.execution_values["retention_policy"] == "full":
        export_root = evidence_directory / "comsol_exports"
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
    timing_path = evidence_directory / "timing.json"
    copied_sources = (
        (result.canonical_case.path, destination / common.paths.GENERATION_RESULT_HDF5_FILENAME),
        (result.solver_log, evidence_directory / "solver.log"),
        (result.prepared.runtime_directory / "timing.json", timing_path),
        (result.execution_provenance, evidence_directory / "execution_provenance.json"),
        (result.processing_provenance, evidence_directory / "processing_provenance.json"),
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
    status_path = common.serialization.atomic_write_json(evidence_directory / "status.json", status)
    fixed_payload_bytes += status_path.stat().st_size
    retained_model_bytes = 0
    if config.execution_values["retention_policy"] == "full":
        if result.solved_model is None:
            message = "Full retention completed without an admitted solved model."
            raise FileNotFoundError(message)
        retained_model_bytes = result.solved_model.stat().st_size
        retained_model_path = evidence_directory / comsol_service.RETAINED_MODEL_FILENAME
        shutil.copy2(result.solved_model, retained_model_path)
        observed_size = retained_model_path.stat().st_size
        if observed_size != retained_model_bytes:
            message = f"Solved model size changed during publication: {retained_model_path}"
            raise RuntimeError(message)
        copied_bytes += retained_model_bytes
        fixed_payload_bytes += observed_size
    validation_start = time.monotonic()
    storage_service.validate_case_hdf5(
        destination / common.paths.GENERATION_RESULT_HDF5_FILENAME,
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


def _load_json_object_at_sha256(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load the exact JSON object bound to one already admitted digest."""
    if not path.is_file() or path.is_symlink():
        message = f"Missing required {label}: {path}"
        raise FileNotFoundError(message)
    try:
        payload = path.read_bytes()
    except OSError as error:
        message = f"Could not read {label}: {path}"
        raise ValueError(message) from error
    if hashlib.sha256(payload).hexdigest() != _require_sha256(expected_sha256, label=f"{label} sha256"):
        message = f"{label} changed after admission: {path}"
        raise RuntimeError(message)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not decode {label}: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain a JSON object: {path}"
        raise TypeError(message)
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
    case_path = directory / common.paths.GENERATION_CASE_PAYLOAD_FILENAME
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
            relative_path=common.paths.GENERATION_CASE_PAYLOAD_FILENAME,
            path=case_path.resolve(),
            sha256=reference.case_json_sha256,
            size_bytes=reference.case_json_size_bytes,
        )
    ]
    for filename, identity in sorted(case_payload["input_files"].items()):
        input_path = reference.case_directory / filename
        raw_artifacts.append(
            ArtifactEvidence(
                relative_path=filename,
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
    """Admit one case's inputs through their authoritative immutable batch evidence."""
    if directory.name != common.paths.GENERATION_INPUTS_DIRNAME:
        message = f"Canonical raw case is not an inputs directory: {directory}"
        raise RuntimeError(message)
    location = common.paths.locate_generation_case_directory(directory.parent)
    reference = _load_json_object(
        directory / common.paths.GENERATION_INPUT_REFERENCE_FILENAME,
        label="canonical campaign input reference",
    )
    input_generation_id = common.paths.validate_logical_name(reference.get("input_generation_id"), label="input_generation_id")
    batch_storage_name = common.paths.validate_logical_name(reference.get("batch_storage_name"), label="batch_storage_name")
    parts = common.paths.campaign_batch_parts(batch_storage_name)
    if parts is None or parts[:3] != (location.task, location.campaign_name, location.material):
        message = "Canonical campaign raw case reference disagrees with its material namespace."
        raise RuntimeError(message)
    expected_raw_root = common.paths.resolve_generation_input_generation_cases_directory(
        batch_storage_name, input_generation_id, storage_root=location.storage_root
    )
    if (
        directory.parent.parent != expected_raw_root
        or reference.get("case_id") != location.case_id
        or reference.get("case_json_sha256") != common.serialization.file_sha256(directory / common.paths.GENERATION_CASE_PAYLOAD_FILENAME)
    ):
        message = "Canonical campaign raw case reference disagrees with its exact case payload."
        raise RuntimeError(message)
    metadata_directory = common.paths.resolve_generation_input_generation_metadata_directory(
        batch_storage_name, input_generation_id, storage_root=location.storage_root
    )
    source = admission_service.admit_input_batch_source(
        metadata_directory,
        raw_directory=expected_raw_root,
        expected_input_generation_id=input_generation_id,
        validation_depth=("evidence" if validation_depth == "routine" else "full"),
    )
    matches = tuple(item for item in source.cases if item.case_id == location.case_id)
    if len(matches) != 1:
        message = "Canonical campaign input manifest does not declare exactly one raw case."
        raise RuntimeError(message)
    return _raw_publication_from_reference(matches[0])


def _require_processed_publication_layout(
    directory: Path,
    *,
    artifact_names: set[str],
    required: set[str],
    retention_policy: str,
) -> None:
    """Require the explicit full or compact result and evidence ownership boundary."""
    evidence = common.paths.GENERATION_EVIDENCE_DIRNAME
    expected_top_level = {"_SUCCESS", "provenance.json", common.paths.GENERATION_RESULT_HDF5_FILENAME, evidence}
    expected_evidence = {Path(name).name for name in required if name.startswith(f"{evidence}/")}
    retained_exports = {name for name in artifact_names if name.startswith(f"{evidence}/comsol_exports/")}
    if retention_policy == "full":
        expected_evidence.update({"comsol_exports", "solved.mph"})
        exports = directory / evidence / "comsol_exports"
        valid = bool(retained_exports) and f"{evidence}/solved.mph" in artifact_names and exports.is_dir() and not exports.is_symlink()
    elif retention_policy == "compact":
        valid = not retained_exports and f"{evidence}/solved.mph" not in artifact_names and not (directory / evidence / "comsol_exports").exists()
    else:
        message = f"Processed publication has an invalid retention policy: {retention_policy!r}."
        raise RuntimeError(message)
    evidence_directory = directory / evidence
    observed_evidence = (
        {entry.name for entry in evidence_directory.iterdir()} if evidence_directory.is_dir() and not evidence_directory.is_symlink() else None
    )
    if {entry.name for entry in directory.iterdir()} != expected_top_level or observed_evidence != expected_evidence or not valid:
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
        or not common.serialization.is_schema_version(provenance.get("schema_version"), storage_service.PUBLICATION_SCHEMA_VERSION)
    ):
        message = f"Case publication schema is not current: {directory}"
        raise RuntimeError(message)
    input_generation_id = common.paths.validate_logical_name(
        provenance.get("input_generation_id"),
        label="input_generation_id",
    )
    if directory.name != common.paths.GENERATION_RESULT_DIRNAME:
        message = f"Canonical processed case is not a result directory: {directory}"
        raise RuntimeError(message)
    raw_case = common.paths.resolve_generation_case_inputs_directory(directory.parent)
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


def _admit_processed_case_payload(
    directory: Path,
    provenance: Mapping[str, Any],
    *,
    raw_evidence: _PublicationEvidence | None,
    validation_depth: ValidationDepth,
    case_payload_evidence: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Admit raw-backed or explicitly reconstructed processed case metadata."""
    if case_payload_evidence is None:
        input_generation_id, admitted_raw = _admit_processed_raw_publication(
            directory,
            provenance,
            raw_evidence=raw_evidence,
            validation_depth=validation_depth,
        )
        case_payload = dict(admitted_raw.case_payload)
    else:
        if raw_evidence is not None:
            message = "Processed-only admission cannot also receive raw publication evidence."
            raise TypeError(message)
        if (
            set(provenance) != _CASE_PUBLICATION_KEYS
            or provenance.get("schema_kind") != _CASE_PUBLICATION_SCHEMA_KIND
            or not common.serialization.is_schema_version(provenance.get("schema_version"), storage_service.PUBLICATION_SCHEMA_VERSION)
        ):
            message = f"Case publication schema is not current: {directory}"
            raise RuntimeError(message)
        input_generation_id = common.paths.validate_logical_name(
            provenance.get("input_generation_id"),
            label="input_generation_id",
        )
        case_payload = dict(case_payload_evidence)
    try:
        case_service.validate_case_payload_schema(case_payload)
    except (KeyError, TypeError, ValueError) as error:
        message = f"Canonical case provenance does not match the active exact schema: {directory}"
        raise RuntimeError(message) from error
    return input_generation_id, case_payload


def _require_processed_success_evidence(
    status: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    case_id: str,
    simulation_case_id: str,
    directory: Path,
) -> None:
    """Require positive scientific and COMSOL success evidence."""
    stages = status.get("stages")
    result = execution.get("result")
    successful_stages = ("solver", "exports", "conversion", "publication")
    if (
        status.get("schema_kind") != "simulation_case_status"
        or not common.serialization.is_schema_version(status.get("schema_version"), storage_service.STATUS_SCHEMA_VERSION)
        or status.get("case_state") != "successful"
        or status.get("solver_success") is not True
        or status.get("contains_nan_or_inf") is not False
        or status.get("field_shape_valid") is not True
        or not isinstance(stages, dict)
        or any(stages.get(name) != "succeeded" for name in successful_stages)
        or execution.get("schema_kind") != "simulation_execution_provenance"
        or not common.serialization.is_schema_version(execution.get("schema_version"), _EXECUTION_PROVENANCE_SCHEMA_VERSION)
        or execution.get("case_id") != case_id
        or execution.get("simulation_case_id") != simulation_case_id
        or not isinstance(result, dict)
        or result.get("state") != "succeeded"
        or result.get("exit_code") != 0
        or result.get("timed_out") is not False
    ):
        message = f"Processed scientific or COMSOL success evidence is invalid in {directory}."
        raise RuntimeError(message)


def _admit_publication_directory(
    directory: Path,
    *,
    stage: str,
    validation_depth: ValidationDepth = "full",
    raw_evidence: _PublicationEvidence | None = None,
    case_payload_evidence: Mapping[str, Any] | None = None,
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
    input_generation_id, case_payload = _admit_processed_case_payload(
        directory,
        provenance,
        raw_evidence=raw_evidence,
        validation_depth=validation_depth,
        case_payload_evidence=case_payload_evidence,
    )
    if (
        set(success) != _CASE_SUCCESS_KEYS
        or success.get("schema_kind") != _CASE_SUCCESS_SCHEMA_KIND
        or not common.serialization.is_schema_version(success.get("schema_version"), storage_service.PUBLICATION_SCHEMA_VERSION)
        or case_payload.get("schema_kind") != case_service.CASE_SCHEMA_KIND
        or not common.serialization.is_schema_version(case_payload.get("schema_version"), case_service.CASE_SCHEMA_VERSION)
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
    evidence_name = common.paths.GENERATION_EVIDENCE_DIRNAME
    required = {
        common.paths.GENERATION_RESULT_HDF5_FILENAME,
        f"{evidence_name}/solver.log",
        f"{evidence_name}/timing.json",
        f"{evidence_name}/status.json",
        f"{evidence_name}/execution_provenance.json",
        f"{evidence_name}/processing_provenance.json",
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
    evidence_directory = directory / evidence_name
    timing = _load_json_object(evidence_directory / "timing.json", label="case timing")
    status = _load_json_object(evidence_directory / "status.json", label="case status")
    execution = _load_json_object(
        evidence_directory / "execution_provenance.json",
        label="case execution provenance",
    )
    processing = _load_json_object(
        evidence_directory / "processing_provenance.json",
        label="case processing provenance",
    )
    _require_processed_success_evidence(
        status,
        execution,
        case_id=case_id,
        simulation_case_id=simulation_case_id,
        directory=directory,
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
        "scientific_config_digest",
        "template_sha256",
        "recorded_at",
    }
    if (
        timing.get("git_commit") != processing.get("solver_git_commit")
        or execution.get("git_commit") != processing.get("solver_git_commit")
        or set(processing) != processing_keys
        or processing.get("schema_kind") != "generation_processing_provenance"
        or not common.serialization.is_schema_version(processing.get("schema_version"), _PROCESSING_PROVENANCE_SCHEMA_VERSION)
        or processing.get("case_id") != case_id
        or processing.get("case_input_id") != case_input_id
        or processing.get("simulation_case_id") != simulation_case_id
        or processing.get("mode") != "initial"
        or not isinstance(processing.get("solver_git_commit"), str)
        or processing.get("scientific_config_digest") != scientific_digest
        or processing.get("template_sha256") != template_sha256
    ):
        msg = f"Processed solver/processing provenance disagrees in {directory}."
        raise RuntimeError(msg)
    source_service.validate_git_commit(processing.get("processing_git_commit"))
    source_service.validate_git_commit(processing.get("solver_git_commit"))
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
                directory / common.paths.GENERATION_RESULT_HDF5_FILENAME,
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
        "domain_profile": config.scientific_values["domain_profile"],
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
    """Validate one completed case through the authoritative terminal admission."""
    admitted = admit_completed_case(
        config,
        case_index,
        storage_root=storage_root,
        validation_depth=validation_depth,
        input_reference=input_reference,
    )
    return _load_json_object_at_sha256(
        admitted.processed_directory / "provenance.json",
        label="processed case publication provenance",
        expected_sha256=admitted.provenance_sha256,
    )


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
        Storage root containing the processed case and any retained raw publication.
    validation_depth : {"routine", "full", "deep"}, optional
        Integrity depth used for raw and processed publication admission.
    input_reference : InputCaseReference | None, optional
        Exact raw-input evidence already admitted by the caller.
    git_commit : str | None, optional
        Persisted source commit required to match the processed publication.

    Returns
    -------
    TerminalCaseEvidence
        Immutable case-local publication, artifact, HDF5, and metadata evidence.

    Raises
    ------
    FileNotFoundError
        If required processed or durable identity evidence is absent or unsafe.
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
    processed_directory = processed_case_directory(config, case_index, storage_root=storage_root)
    provenance = _load_json_object(processed_directory / "provenance.json", label="processed case publication provenance")
    input_generation_id = common.paths.validate_logical_name(
        provenance.get("input_generation_id"),
        label="processed input_generation_id",
    )
    persisted_raw = raw_case_directory(config, case_index, storage_root=storage_root)
    if persisted_raw.is_symlink() or persisted_raw.parent.is_symlink() or (persisted_raw.exists() and not persisted_raw.is_dir()):
        message = f"Processed publication references unsafe contradictory raw evidence: {persisted_raw}"
        raise RuntimeError(message)
    if git_commit is not None and provenance.get("git_commit") != source_service.validate_git_commit(git_commit):
        message = "Processed publication source commit disagrees with the requested completion source commit."
        raise RuntimeError(message)
    if not persisted_raw.is_dir():
        message = f"Completed case lacks its canonical raw input evidence: {persisted_raw}"
        raise FileNotFoundError(message)
    if input_reference is None:
        raw = _admit_raw_publication_directory(
            persisted_raw,
            validation_depth=validation_depth,
        )
    else:
        raw = _raw_publication_from_reference(input_reference)
        if input_reference.source_id != input_generation_id or raw.directory != persisted_raw.resolve():
            message = "Provided raw input evidence disagrees with the processed publication input-generation identity."
            raise RuntimeError(message)
    processed = _admit_publication_directory(
        processed_directory,
        stage="processed",
        validation_depth=validation_depth,
        raw_evidence=raw,
    )
    _require_publication_matches_config(raw, config=config, case_index=case_index)
    _require_publication_matches_config(processed, config=config, case_index=case_index)
    if raw.case_payload != processed.case_payload:
        message = f"Raw and processed canonical metadata disagree for {config.case_id(case_index)!r}."
        raise RuntimeError(message)
    hdf5_artifacts = tuple(item for item in processed.artifacts if item.relative_path == common.paths.GENERATION_RESULT_HDF5_FILENAME)
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
        hdf5_path=(processed.directory / common.paths.GENERATION_RESULT_HDF5_FILENAME).resolve(),
        raw_artifacts=raw.artifacts,
        processed_artifacts=processed.artifacts,
        hdf5_identity=processed.hdf5_identity,
        _case_metadata_json=_canonical_json_text(raw.case_payload),
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
    processed = processed_case_directory(config, case_index, storage_root=storage_root)
    if not (processed / "_SUCCESS").exists():
        raw = raw_case_directory(config, case_index, storage_root=storage_root)
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
    admit_completed_case(
        config,
        case_index,
        storage_root=storage_root,
        validation_depth="routine",
        input_reference=input_reference,
        git_commit=git_commit,
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
    processed_stage = staging / common.paths.GENERATION_RESULT_DIRNAME
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
        source_service.validate_admitted_source_before_publication()
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
    input_reference: admission_service.InputCaseReference | None = None,
) -> CaseRunOutcome:
    """Run or integrity-skip one case, closing scratch after durable outcome evidence."""
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
        if completed_case_is_valid(config, case_index, storage_root=storage, input_reference=input_reference):
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
                input_generation_id=None if input_reference is None else input_reference.source_id,
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
            )
            failure_stage = "publication"
            _update_runtime_progress(progress_reporter, phase="publishing", force=True)
            destination = publish_completed_case(
                config,
                result,
                storage_root=storage,
            )
            _update_runtime_progress(progress_reporter, phase="completed", terminal=True)
        except BaseException as error:
            _update_runtime_progress(progress_reporter, phase="failed", terminal=True)
            attempt_directory, attempt_root, attempt_run_id = _workspace_from_attempt(prepared, error)
            try:
                publication_complete = completed_case_is_valid(
                    config,
                    case_index,
                    storage_root=storage,
                    input_reference=input_reference,
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
            retry_enabled = scheduler_kind == "slurm" and bool(config.execution_values["runtime"]["license_wait"]["enabled"])
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
    cases_root = common.paths.resolve_generation_input_generation_cases_directory(
        batch_storage_name,
        input_generation_id,
        storage_root=storage_root,
    )
    if not cases_root.is_dir() or cases_root.is_symlink():
        message = "Canonical campaign material-local case namespace is missing or unsafe."
        raise RuntimeError(message)
    for case_id in sorted(expected):
        case_directory = cases_root / case_id
        stages = (
            common.paths.resolve_generation_case_inputs_directory(case_directory),
            common.paths.resolve_generation_case_result_directory(case_directory),
        )
        if case_directory.is_symlink() or any(not stage.is_dir() or stage.is_symlink() for stage in stages):
            message = f"Canonical campaign case lacks safe raw and processed evidence: {case_directory}"
            raise RuntimeError(message)
    return cases_root.resolve(), cases_root.resolve()


def finalize_batch(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
    input_references: Mapping[int, admission_service.InputCaseReference] | None = None,
) -> Path:
    """Validate exact membership and atomically publish the terminal manifest."""
    initialize_batch_metadata(config, storage_root=storage_root)
    records: list[dict[str, Any]] = []
    git_commits: set[str] = set()
    input_generation_ids: set[str] = set()
    if input_references is None:
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
                "case_hdf5_sha256": provenance["artifacts"][common.paths.GENERATION_RESULT_HDF5_FILENAME]["sha256"],
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
        or not common.serialization.is_schema_version(scientific.get("schema_version"), config_contract.CONFIG_SCHEMA_VERSION)
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
    natural_support_state = scientific.get("natural_support_state")
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
        "domain_profile": scientific.get("domain_profile"),
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
        metadata_directory = common.paths.resolve_generation_input_generation_metadata_directory(
            batch_storage_name, input_generation_id, storage_root=storage_root
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
        or reference.case_directory.parent.parent.resolve() != expected_root
        for reference in references.values()
    )
    if invalid or set(references) != set(case_ids) or len(references) != len(candidates):
        message = f"Terminal batch input evidence does not cover exact membership: {manifest_path}"
        raise RuntimeError(message)
    return references


def _terminal_identity_is_valid(
    profile_id: str,
    material_family: str,
    sampling_regime: str,
    campaign_purpose: str,
    batch_name: str,
    *,
    batch_id: str,
    storage_name: str,
    scientific: Mapping[str, Any],
    scientific_digest: str,
) -> bool:
    """Bind a terminal batch's name, storage locator, and immutable identifier to its persisted science."""
    expected_name, locator_regime = config_contract.resolve_batch_locators(profile_id, material_family, sampling_regime, scientific)
    storage_parts = common.paths.campaign_batch_parts(storage_name)
    if storage_parts is None:
        return False
    expected_storage_name = f"source~{profile_id}~{storage_parts[1]}~{material_family}~{locator_regime}"
    return (
        batch_name == expected_name
        and batch_id == config_contract.build_batch_id(expected_name, scientific_digest)
        and storage_name == expected_storage_name
        and scientific.get("campaign_purpose") == campaign_purpose
    )


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
    meta_candidate = common.paths.resolve_generation_batch_metadata_directory(safe_storage_name, storage_root=storage_root)
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
        or not common.serialization.is_schema_version(manifest.get("schema_version"), BATCH_MANIFEST_SCHEMA_VERSION)
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
    identity_is_valid = (
        isinstance(material_family, str)
        and isinstance(sampling_regime, str)
        and isinstance(campaign_purpose, str)
        and isinstance(batch_name, str)
        and _terminal_identity_is_valid(
            profile.id,
            material_family,
            sampling_regime,
            campaign_purpose,
            batch_name,
            batch_id=safe_batch_id,
            storage_name=safe_storage_name,
            scientific=scientific,
            scientific_digest=scientific_digest,
        )
    )
    if not sampling_regime or not identity_is_valid:
        msg = f"Terminal batch name, purpose, storage locator, or immutable identifier is invalid: {manifest_path}"
        raise RuntimeError(msg)
    git_commit = source_service.validate_git_commit(manifest.get("git_commit"))
    input_generation_id = common.paths.validate_logical_name(
        manifest.get("input_generation_id"),
        label="input_generation_id",
    )
    expected_success = {
        "schema_kind": _BATCH_SUCCESS_SCHEMA_KIND,
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "simulation_profile": profile.id,
        "batch_id": safe_batch_id,
        "batch_identity": batch_identity,
        "manifest_sha256": manifest_sha256,
    }
    if (
        set(success) != _BATCH_SUCCESS_KEYS
        or not common.serialization.is_schema_version(success["schema_version"], BATCH_MANIFEST_SCHEMA_VERSION)
        or success != expected_success
    ):
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
            processed_directory=common.paths.resolve_generation_case_result_directory(processed_root / expected_case_id),
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
        hdf5_artifacts = tuple(artifact for artifact in processed.artifacts if artifact.relative_path == common.paths.GENERATION_RESULT_HDF5_FILENAME)
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
                hdf5_path=(processed.directory / common.paths.GENERATION_RESULT_HDF5_FILENAME).resolve(),
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
