"""
generation_runtime_workspace.py

Validate persistent roots and own disposable generation workspaces.
Responsibilities:
  - Resolve user-owned persistent storage and explicitly bounded scratch roots
  - Create collision-safe case, worker, and transfer staging directories
  - Guard every recursive workspace cleanup with identity markers and containment
Design principles:
  - Persistent and disposable paths have separate ownership rules
  - Symlink resolution and exact marker identity precede every destructive action
  - Cleanup never targets repositories, storage roots, publications, or templates
This module does NOT:
  - Generate scientific inputs, run COMSOL, submit Slurm jobs, or publish cases
  - Delete canonical data or infer ownership from a directory name
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.contracts import generation_contracts_paths as path_contract

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.generation.cases.generation_cases_config import GenerationConfig

CASE_WORKSPACE_MARKER: Final = path_contract.CASE_WORKSPACE_MARKER
WORKER_WORKSPACE_MARKER: Final = ".generation-worker-workspace.json"
TRANSFER_STAGING_MARKER: Final = ".generation-transfer-staging.json"
PUBLICATION_STAGING_MARKER: Final = ".generation-publication-staging.json"
WORKSPACE_SCHEMA_VERSION: Final = 1
_SLURM_JOB_ID_PATTERN: Final = re.compile(r"[0-9]+")
_CASE_MARKER_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "run_id",
        "case_id",
        "simulation_profile",
        "work_root",
        "work_directory",
        "created_at",
        "hostname",
        "creator_pid",
        "slurm_job_id",
        "slurm_array_task_id",
        "slurm_step_id",
    }
)
_WORKER_MARKER_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "run_id",
        "work_directory",
        "created_at",
        "hostname",
        "creator_pid",
        "slurm_job_id",
        "slurm_array_task_id",
    }
)
_TRANSFER_MARKER_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "run_id",
        "storage_root",
        "work_directory",
        "created_at",
        "hostname",
        "creator_pid",
    }
)
_PUBLICATION_MARKER_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "run_id",
        "case_id",
        "storage_root",
        "work_directory",
        "created_at",
        "hostname",
        "creator_pid",
        "slurm_job_id",
    }
)


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


resolve_storage_root = path_contract.resolve_storage_root


def resolve_work_root(
    *,
    storage_root: Path,
    work_root: Path | str | None,
    create: bool,
) -> Path:
    """Resolve one disposable work root without falling back to persistent state."""
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if work_root is None:
        if not slurm_job_id:
            message = "Implicit TMPDIR scratch is permitted only inside an active Slurm allocation."
            raise ValueError(message)
        configured = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        resolved = path_contract.absolute_path(configured, label="TMPDIR scratch root")
        if not resolved.is_dir() or not os.access(resolved, os.W_OK | os.X_OK):
            message = f"TMPDIR scratch root is unavailable: {resolved}"
            raise PermissionError(message)
    else:
        resolved = path_contract.absolute_path(work_root, label="work_root")
        path_contract.require_user_owned_writable(resolved, label="work_root")
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        path_contract.require_user_owned_writable(resolved, label="work_root")
    repository = common.paths.get_project_root().resolve()
    home = Path.home().resolve()
    forbidden = {Path("/"), home, repository, storage_root}
    if resolved in forbidden or path_contract.is_relative_to(resolved, repository) or path_contract.is_relative_to(resolved, storage_root):
        message = f"work_root targets a forbidden repository or persistent boundary: {resolved}"
        raise ValueError(message)
    if create and not resolved.exists():
        resolved.mkdir(parents=True)
    if not resolved.is_dir() or resolved.is_symlink():
        message = f"work_root must be one safe directory: {resolved}"
        raise ValueError(message)
    return resolved


def workspace_run_id(config: GenerationConfig) -> str:
    """Return the campaign, benchmark, or local batch identity for work."""
    run_id = os.environ.get(
        "GENERATION_BENCHMARK_RUN_ID",
        os.environ.get("GENERATION_CAMPAIGN_RUN_ID", config.batch_id),
    )
    return common.paths.validate_logical_name(run_id, label="workspace run_id")


def create_case_workspace(
    config: GenerationConfig,
    *,
    case_id: str,
    storage_root: Path,
    work_root: Path | str | None,
) -> tuple[Path, Path, Path]:
    """Create one collision-safe marked case directory."""
    root = resolve_work_root(
        storage_root=storage_root,
        work_root=work_root,
        create=True,
    )
    directory = Path(tempfile.mkdtemp(prefix="vp2-case-", dir=root)).resolve()
    if not path_contract.is_relative_to(directory, root) or directory.parent != root:
        message = f"Created case workspace escaped its intended root: {directory}"
        raise RuntimeError(message)
    run_id = workspace_run_id(config)
    marker = {
        "schema_kind": "generation_case_workspace",
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": common.paths.validate_logical_name(case_id, label="case_id"),
        "simulation_profile": config.profile.id,
        "work_root": str(root),
        "work_directory": str(directory),
        "created_at": _utc_now(),
        "hostname": socket.gethostname(),
        "creator_pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }
    if set(marker) != _CASE_MARKER_KEYS:
        message = "Case workspace marker construction is incomplete."
        raise RuntimeError(message)
    marker_path = common.serialization.atomic_write_json(
        directory / CASE_WORKSPACE_MARKER,
        marker,
    )
    return directory, root, marker_path


def _load_marker(path: Path, *, expected_keys: frozenset[str], label: str) -> dict[str, Any]:
    """Load one exact workspace marker without following a marker symlink."""
    if not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise ValueError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"{label} is not valid JSON: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict) or set(value) != expected_keys:
        message = f"{label} has an unsupported structure: {path}"
        raise ValueError(message)
    return value


def _directory_size(path: Path) -> int:
    """Return the regular-file byte count below one non-symlink directory."""
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            message = f"Workspace contains an unsafe symbolic link: {item}"
            raise ValueError(message)
        if item.is_file():
            total += item.stat().st_size
    return total


def _slurm_job_is_active(job_id: str) -> bool:
    """Return whether Slurm still reports one exact root job identifier."""
    if _SLURM_JOB_ID_PATTERN.fullmatch(job_id) is None:
        message = f"Workspace Slurm job ID is malformed: {job_id!r}"
        raise ValueError(message)
    executable = shutil.which("squeue")
    if executable is None:
        message = f"Cannot prove Slurm job {job_id} is inactive because squeue is unavailable."
        raise RuntimeError(message)
    result = subprocess.run(  # noqa: S603 -- executable and numeric job ID are validated above
        [executable, "--noheader", f"--jobs={job_id}", "--format=%A"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        message = f"Could not query Slurm job {job_id}: {detail}"
        raise RuntimeError(message)
    return any(line.strip() == job_id for line in result.stdout.splitlines())


def _guard_cleanup_target(
    target: Path | str,
    *,
    allowed_root: Path,
    storage_root: Path,
    marker_filename: str,
    marker_keys: frozenset[str],
    marker_kind: str,
    expected_run_id: str,
    expected_case_id: str | None,
    allow_active_job_id: str | None,
    extra_forbidden: Iterable[Path] = (),
) -> tuple[Path, dict[str, Any]]:
    """Validate containment, forbidden roots, marker identity, and job state."""
    resolved_target = path_contract.absolute_path(target, label="cleanup target")
    resolved_root = path_contract.absolute_path(allowed_root, label="allowed cleanup root")
    if not resolved_target.exists() or not resolved_target.is_dir() or resolved_target.is_symlink():
        message = f"Cleanup target must be one existing non-symlink directory: {resolved_target}"
        raise ValueError(message)
    if resolved_target == resolved_root or not path_contract.is_relative_to(resolved_target, resolved_root):
        message = f"Cleanup target escapes or equals its allowed work root: {resolved_target}"
        raise ValueError(message)
    home = Path.home().resolve()
    repository = common.paths.get_project_root().resolve()
    forbidden = {
        Path("/"),
        home,
        repository,
        storage_root.resolve(),
        common.paths.get_generation_meta_root(storage_root=storage_root).resolve(),
        common.paths.get_generation_raw_root(storage_root=storage_root).resolve(),
        common.paths.get_generation_processed_root(storage_root=storage_root).resolve(),
        common.paths.get_datasets_root(storage_root=storage_root).resolve(),
        *(path.resolve() for path in extra_forbidden),
    }
    if resolved_target in forbidden or any(
        path_contract.is_relative_to(forbidden_path, resolved_target) for forbidden_path in forbidden if forbidden_path != Path("/")
    ):
        message = f"Cleanup target contains or equals a protected persistent boundary: {resolved_target}"
        raise ValueError(message)
    marker = _load_marker(
        resolved_target / marker_filename,
        expected_keys=marker_keys,
        label=f"{marker_kind} marker",
    )
    if (
        marker.get("schema_kind") != marker_kind
        or marker.get("schema_version") != WORKSPACE_SCHEMA_VERSION
        or marker.get("run_id") != expected_run_id
        or marker.get("work_directory") != str(resolved_target)
    ):
        message = f"Cleanup marker identity does not match its directory: {resolved_target}"
        raise ValueError(message)
    if expected_case_id is not None and marker.get("case_id") != expected_case_id:
        message = f"Cleanup marker case identity does not match {expected_case_id!r}."
        raise ValueError(message)
    job_id = marker.get("slurm_job_id")
    if job_id is not None:
        if not isinstance(job_id, str):
            message = "Workspace marker Slurm job ID must be null or text."
            raise ValueError(message)
        if job_id != allow_active_job_id and _slurm_job_is_active(job_id):
            message = f"Refusing cleanup while Slurm job {job_id} remains active."
            raise RuntimeError(message)
    return resolved_target, marker


def cleanup_case_workspace(
    target: Path | str,
    *,
    allowed_root: Path,
    storage_root: Path,
    expected_run_id: str,
    expected_case_id: str,
    allow_active_job_id: str | None = None,
) -> int:
    """Remove one exact marked case workspace and return reclaimed bytes."""
    resolved, _marker = _guard_cleanup_target(
        target,
        allowed_root=allowed_root,
        storage_root=storage_root,
        marker_filename=CASE_WORKSPACE_MARKER,
        marker_keys=_CASE_MARKER_KEYS,
        marker_kind="generation_case_workspace",
        expected_run_id=expected_run_id,
        expected_case_id=expected_case_id,
        allow_active_job_id=allow_active_job_id,
    )
    reclaimed = _directory_size(resolved)
    shutil.rmtree(resolved)
    return reclaimed


def create_cleanup_probe(
    *,
    storage_root: Path,
    work_root: Path | str,
) -> tuple[Path, Path]:
    """Create one marked non-production directory for cleanup preflight."""
    storage = resolve_storage_root(storage_root, create=False)
    root = resolve_work_root(
        storage_root=storage,
        work_root=work_root,
        create=False,
    )
    directory = Path(tempfile.mkdtemp(prefix="vp2-preflight-", dir=root)).resolve()
    marker = {
        "schema_kind": "generation_case_workspace",
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "run_id": "preflight",
        "case_id": "preflight-probe",
        "simulation_profile": "preflight",
        "work_root": str(root),
        "work_directory": str(directory),
        "created_at": _utc_now(),
        "hostname": socket.gethostname(),
        "creator_pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    }
    marker_path = common.serialization.atomic_write_json(
        directory / CASE_WORKSPACE_MARKER,
        marker,
    )
    return directory, marker_path


def initialize_worker_workspace(
    directory: Path | str,
    *,
    run_id: str,
    storage_root: Path,
) -> Path:
    """Mark one shell-created collision-safe Slurm worker root."""
    resolved = path_contract.absolute_path(directory, label="worker workspace")
    storage = resolve_storage_root(storage_root, create=False)
    if not resolved.is_dir() or resolved.is_symlink() or resolved.stat().st_uid != os.getuid():
        message = f"Worker workspace is missing, unsafe, or not user-owned: {resolved}"
        raise ValueError(message)
    resolve_work_root(storage_root=storage, work_root=resolved, create=False)
    safe_run_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    marker = {
        "schema_kind": "generation_worker_workspace",
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "run_id": safe_run_id,
        "work_directory": str(resolved),
        "created_at": _utc_now(),
        "hostname": socket.gethostname(),
        "creator_pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    return common.serialization.atomic_write_json(resolved / WORKER_WORKSPACE_MARKER, marker)


def cleanup_worker_workspace(
    directory: Path | str,
    *,
    run_id: str,
    storage_root: Path,
    allow_active_job_id: str | None,
) -> int:
    """Remove every marked case child and then its exact worker root."""
    resolved = path_contract.absolute_path(directory, label="worker workspace")
    root = resolved.parent.resolve()
    guarded, marker = _guard_cleanup_target(
        resolved,
        allowed_root=root,
        storage_root=storage_root,
        marker_filename=WORKER_WORKSPACE_MARKER,
        marker_keys=_WORKER_MARKER_KEYS,
        marker_kind="generation_worker_workspace",
        expected_run_id=run_id,
        expected_case_id=None,
        allow_active_job_id=allow_active_job_id,
    )
    reclaimed = 0
    for child in sorted(guarded.iterdir()):
        if child.name == WORKER_WORKSPACE_MARKER:
            continue
        if not child.is_dir() or child.is_symlink():
            message = f"Worker workspace contains an unowned entry: {child}"
            raise ValueError(message)
        case_marker = _load_marker(
            child / CASE_WORKSPACE_MARKER,
            expected_keys=_CASE_MARKER_KEYS,
            label="case workspace marker",
        )
        case_id = case_marker.get("case_id")
        if not isinstance(case_id, str):
            message = f"Case workspace marker has no valid case identity: {child}"
            raise TypeError(message)
        reclaimed += cleanup_case_workspace(
            child,
            allowed_root=guarded,
            storage_root=storage_root,
            expected_run_id=run_id,
            expected_case_id=case_id,
            allow_active_job_id=allow_active_job_id,
        )
    reclaimed += (guarded / WORKER_WORKSPACE_MARKER).stat().st_size
    (guarded / WORKER_WORKSPACE_MARKER).unlink()
    guarded.rmdir()
    if marker.get("run_id") != run_id:
        message = "Worker marker changed during cleanup."
        raise RuntimeError(message)
    return reclaimed


def create_publication_staging(
    *,
    storage_root: Path,
    publication_root: Path,
    run_id: str,
    case_id: str,
) -> Path:
    """Create one marked collision-safe persistent publication staging root."""
    storage = resolve_storage_root(storage_root, create=True)
    root = path_contract.absolute_path(publication_root, label="publication staging root")
    if not path_contract.is_relative_to(root, storage):
        message = f"Publication staging root must remain below storage_root: {root}"
        raise ValueError(message)
    root.mkdir(parents=True, exist_ok=True)
    safe_run_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    safe_case_id = common.paths.validate_logical_name(case_id, label="case_id")
    directory = Path(tempfile.mkdtemp(prefix="publication.", dir=root)).resolve()
    marker = {
        "schema_kind": "generation_publication_staging",
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "run_id": safe_run_id,
        "case_id": safe_case_id,
        "storage_root": str(storage),
        "work_directory": str(directory),
        "created_at": _utc_now(),
        "hostname": socket.gethostname(),
        "creator_pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    common.serialization.atomic_write_json(
        directory / PUBLICATION_STAGING_MARKER,
        marker,
    )
    return directory


def cleanup_publication_staging(
    directory: Path | str,
    *,
    storage_root: Path,
    publication_root: Path,
    run_id: str,
    case_id: str,
    allow_active_job_id: str | None,
) -> int:
    """Remove one exact marked publication staging directory."""
    storage = resolve_storage_root(storage_root, create=False)
    root = path_contract.absolute_path(publication_root, label="publication staging root")
    resolved, marker = _guard_cleanup_target(
        directory,
        allowed_root=root,
        storage_root=storage,
        marker_filename=PUBLICATION_STAGING_MARKER,
        marker_keys=_PUBLICATION_MARKER_KEYS,
        marker_kind="generation_publication_staging",
        expected_run_id=run_id,
        expected_case_id=case_id,
        allow_active_job_id=allow_active_job_id,
    )
    if marker.get("storage_root") != str(storage):
        message = f"Publication staging storage identity is invalid: {resolved}"
        raise ValueError(message)
    reclaimed = _directory_size(resolved)
    shutil.rmtree(resolved)
    return reclaimed


def create_transfer_staging(
    *,
    storage_root: Path | str,
    run_id: str,
) -> Path:
    """Create marked incoming staging on the destination filesystem."""
    storage = resolve_storage_root(storage_root, create=True)
    safe_run_id = common.paths.validate_logical_name(run_id, label="campaign_run_id")
    root = storage / ".incoming"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{safe_run_id}.", dir=root)).resolve()
    marker = {
        "schema_kind": "generation_transfer_staging",
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "run_id": safe_run_id,
        "storage_root": str(storage),
        "work_directory": str(directory),
        "created_at": _utc_now(),
        "hostname": socket.gethostname(),
        "creator_pid": os.getpid(),
    }
    common.serialization.atomic_write_json(directory / TRANSFER_STAGING_MARKER, marker)
    return directory


def validate_transfer_staging(
    directory: Path | str,
    *,
    run_id: str,
) -> Path:
    """Validate one exact marked transfer staging directory without removing it."""
    resolved = path_contract.absolute_path(directory, label="transfer staging")
    marker = _load_marker(
        resolved / TRANSFER_STAGING_MARKER,
        expected_keys=_TRANSFER_MARKER_KEYS,
        label="generation_transfer_staging marker",
    )
    storage_value = marker.get("storage_root")
    if not isinstance(storage_value, str):
        message = f"Transfer staging marker has no storage identity: {resolved}"
        raise TypeError(message)
    storage = resolve_storage_root(storage_value, create=False)
    root = storage / ".incoming"
    guarded, validated_marker = _guard_cleanup_target(
        resolved,
        allowed_root=root,
        storage_root=storage,
        marker_filename=TRANSFER_STAGING_MARKER,
        marker_keys=_TRANSFER_MARKER_KEYS,
        marker_kind="generation_transfer_staging",
        expected_run_id=run_id,
        expected_case_id=None,
        allow_active_job_id=None,
    )
    if validated_marker.get("storage_root") != str(storage):
        message = f"Transfer staging marker storage identity is invalid: {guarded}"
        raise ValueError(message)
    return guarded


def cleanup_transfer_staging(
    directory: Path | str,
    *,
    storage_root: Path | str,
    run_id: str,
) -> int:
    """Remove one exact marked transfer staging directory."""
    storage = resolve_storage_root(storage_root, create=False)
    root = storage / ".incoming"
    resolved, marker = _guard_cleanup_target(
        directory,
        allowed_root=root,
        storage_root=storage,
        marker_filename=TRANSFER_STAGING_MARKER,
        marker_keys=_TRANSFER_MARKER_KEYS,
        marker_kind="generation_transfer_staging",
        expected_run_id=run_id,
        expected_case_id=None,
        allow_active_job_id=None,
    )
    if marker.get("storage_root") != str(storage):
        message = f"Transfer staging marker storage identity is invalid: {resolved}"
        raise ValueError(message)
    reclaimed = _directory_size(resolved)
    shutil.rmtree(resolved)
    return reclaimed


def transfer_staging_candidates(
    *,
    storage_root: Path | str,
    run_id: str | None,
) -> tuple[dict[str, Any], ...]:
    """Return only valid marked transfer staging cleanup candidates."""
    storage = resolve_storage_root(storage_root, create=False)
    root = storage / ".incoming"
    if not root.exists():
        return ()
    if not root.is_dir() or root.is_symlink():
        message = f"Transfer staging root is unsafe: {root}"
        raise ValueError(message)
    safe_run_id = None if run_id is None else common.paths.validate_logical_name(run_id, label="campaign_run_id")
    candidates: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.is_symlink():
            message = f"Transfer staging root contains an unsafe entry: {directory}"
            raise ValueError(message)
        marker = _load_marker(
            directory / TRANSFER_STAGING_MARKER,
            expected_keys=_TRANSFER_MARKER_KEYS,
            label="transfer staging marker",
        )
        marker_run_id = marker.get("run_id")
        if safe_run_id is not None and marker_run_id != safe_run_id:
            continue
        if marker.get("storage_root") != str(storage) or marker.get("work_directory") != str(directory.resolve()):
            message = f"Transfer staging marker identity is invalid: {directory}"
            raise ValueError(message)
        candidates.append(
            {
                "path": str(directory.resolve()),
                "run_id": marker_run_id,
                "size_bytes": _directory_size(directory),
            }
        )
    return tuple(candidates)
