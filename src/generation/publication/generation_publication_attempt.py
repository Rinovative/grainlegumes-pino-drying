"""Publish and admit durable unsuccessful Generation attempt evidence.

Responsibilities:
  - Resolve collision-safe case and attempt storage paths
  - Derive failure-stage-specific retained artifact inventories
  - Atomically publish hash-inventoried attempt receipts and their retained payload
  - Record the Slurm allocation, step, and campaign submission index that produced each attempt
  - Admit current attempt evidence and its scratch-cleanup and license-compaction audits

Design principles:
  - Attempt paths are operational locators and never scientific identities
  - Failed work follows the resolved full or compact retention contract, whatever its failure stage
  - Compact retention bounds long logs but keeps each one complete as a gzip sibling
  - Audits are appended beside the original attempt receipt, which is never rewritten

This module does NOT:
  - Execute COMSOL, derive campaign terminality, or publish processed cases
  - Duplicate canonical sampled parameters from raw case evidence
"""

from __future__ import annotations

import copy
import gzip
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from src import common
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.runtime import generation_runtime_dispatch_identity as dispatch_identity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.generation.cases.generation_cases_config import GenerationConfig
    from src.generation.validation.generation_validation_policy import DiagnosticRecord

ATTEMPT_SCHEMA_KIND: Final = "generation_case_attempt"
CLEANUP_SCHEMA_KIND: Final = "generation_case_attempt_cleanup"
LICENSE_COMPACTION_SCHEMA_KIND: Final = "generation_license_attempt_compaction"
_CASE_STATES: Final = frozenset(
    {
        "failed",
        "timed_out",
        "cancelled",
        "interrupted",
        "exports_failed",
        "conversion_failed",
        "publication_failed",
        "license_blocked",
    }
)
_FAILURE_STAGES: Final = frozenset({"input", "solver", "exports", "conversion", "publication"})
_RUN_PREFIX: Final = "run_"
_RUN_DIRECTORY: Final = re.compile(r"run_([0-9]{4,})\Z")
_SMALL_RUNTIME_PATHS: Final = (
    Path("runtime/stdout.log"),
    Path("runtime/stderr.log"),
    Path("runtime/timing.json"),
    Path("runtime/execution_provenance.json"),
    Path("runtime/processing_provenance.json"),
    Path("runtime/status.json"),
    Path("runtime/stop.json"),
    Path("runtime/solver_steps.csv"),
)
_MAX_RETAINED_RUNTIME_LOG_BYTES: Final = 1024 * 1024
_BOUNDED_RUNTIME_LOG_PATHS: Final = frozenset(
    {
        Path("runtime/stdout.log"),
        Path("runtime/stderr.log"),
    }
)
_LARGE_MODEL_PATHS: Final = (Path("model.mph"), Path("solved.mph"))
_STAGE_STATE_KEYS: Final = (
    "solver_state",
    "exports_state",
    "conversion_state",
    "diagnostics_state",
    "publication_state",
)
_ATTEMPT_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "campaign_run_id",
        "campaign_purpose",
        "batch_storage_name",
        "batch_id",
        "batch_identity",
        "input_generation_id",
        "case_id",
        "case_index",
        "case_input_id",
        "simulation_case_id",
        "attempt_index",
        "previous_attempt",
        "case_state",
        *_STAGE_STATE_KEYS,
        "failure_stage",
        "reason",
        "solver_git_commit",
        "processing_git_commit",
        "solver_execution_provenance",
        "processing_provenance",
        "template",
        "scientific_config_digest",
        "export_contract_sha256",
        "canonical_raw_case",
        "input_files",
        "job_id",
        "slurm_step_id",
        "submission_index",
        "job_name",
        "node",
        "allocated_cores",
        "worker_slot",
        "scheduler_kind",
        "started_at",
        "ended_at",
        "elapsed_seconds",
        "final_simulation_time",
        "last_accepted_step_size",
        "Tfail",
        "NLfail",
        "process_exit_code",
        "timed_out",
        "retention_policy",
        "retained_inventory",
        "intentionally_omitted_artifacts",
        "unexpectedly_missing_artifacts",
        "quality_flags",
        "recorded_at",
    }
)
_CLEANUP_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "campaign_run_id",
        "case_id",
        "attempt_index",
        "status",
        "reclaimed_bytes",
        "error",
        "recorded_at",
    }
)
_LICENSE_COMPACTION_KEYS: Final = frozenset(
    {
        "schema_kind",
        "status",
        "campaign_run_id",
        "batch_id",
        "case_id",
        "attempt_index",
        "attempt_receipt_sha256",
        "license_wait_identity",
        "removed_inventory",
        "reclaimed_bytes",
        "recorded_at",
    }
)
_DIAGNOSTIC_KEYS: Final = frozenset(
    {
        "code",
        "severity",
        "stage",
        "message",
        "metrics",
        "thresholds",
        "source_artifacts",
        "recorded_at",
        "quality_flag",
    }
)
_RETENTION_POLICIES: Final = frozenset({"full", "compact"})
AttemptCaseState = Literal[
    "failed",
    "timed_out",
    "cancelled",
    "interrupted",
    "exports_failed",
    "conversion_failed",
    "publication_failed",
    "license_blocked",
]
_PREVIOUS_ATTEMPT_KEYS: Final = frozenset({"attempt_index", "receipt_sha256"})
_ATTEMPT_CHAIN_IDENTITY_KEYS: Final = (
    "campaign_run_id",
    "batch_storage_name",
    "batch_id",
    "batch_identity",
    "input_generation_id",
    "case_id",
    "case_index",
    "case_input_id",
    "simulation_case_id",
    "solver_git_commit",
    "scientific_config_digest",
    "export_contract_sha256",
)


@dataclass(frozen=True, slots=True)
class AttemptEvidence:
    """One admitted attempt receipt and its durable directory."""

    directory: Path
    receipt_path: Path
    payload: dict[str, Any]


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Load one ordinary JSON object with a path-rich error."""
    if not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"{label} is unreadable: {path}: {error}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain a JSON object: {path}"
        raise TypeError(message)
    return value


def _case_payload(
    config: GenerationConfig,
    case_index: int,
    *,
    work_directory: Path | None,
    storage_root: Path,
    input_generation_id: str | None,
) -> tuple[dict[str, Any], Path, str]:
    """Load exact case identity and return its canonical raw reference."""
    reference = (
        input_service.admit_configured_input_case(
            config,
            case_index,
            storage_root=storage_root,
        )
        if input_generation_id is None
        else input_service.admit_persisted_input_case(
            config,
            case_index,
            input_generation_id,
            storage_root=storage_root,
        )
    )
    canonical_path = reference.case_directory / "case.json"
    canonical = _load_json(
        canonical_path,
        label="canonical attempt case identity",
    )
    if work_directory is None:
        return canonical, canonical_path, reference.source_id
    scratch_path = work_directory / "case.json"
    if not scratch_path.is_file() or scratch_path.is_symlink():
        return canonical, canonical_path, reference.source_id
    scratch = _load_json(scratch_path, label="attempt scratch case identity")
    if scratch != canonical:
        message = f"Attempt scratch case identity differs from canonical raw: {scratch_path}"
        raise ValueError(message)
    return scratch, canonical_path, reference.source_id


def attempt_run_directories(attempts_directory: Path) -> tuple[Path, ...]:
    """List the retained failure bundles ``run_NNNN`` of one case in attempt order."""
    if not attempts_directory.is_dir() or attempts_directory.is_symlink():
        return ()
    runs = [
        (int(match.group(1)), entry)
        for entry in attempts_directory.iterdir()
        if entry.is_dir() and not entry.is_symlink() and (match := _RUN_DIRECTORY.fullmatch(entry.name)) is not None
    ]
    return tuple(entry for _index, entry in sorted(runs, key=lambda item: item[0]))


def _attempt_index(
    config: GenerationConfig,
    case_index: int,
    *,
    storage_root: Path,
    requested: int | None,
) -> int:
    """Return one explicit or next append-only attempt index."""
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            message = f"attempt_index must be a positive integer, got {requested!r}."
            raise ValueError(message)
        return requested
    root = common.paths.resolve_generation_attempts_directory(
        config.batch_storage_name,
        config.case_id(case_index),
        storage_root=storage_root,
    )
    _safe_attempt_history_root(root, storage_root=storage_root)
    runs = attempt_run_directories(root)
    return (int(runs[-1].name[len(_RUN_PREFIX) :]) if runs else 0) + 1


def _previous_attempt_evidence(
    target: Path,
    attempt_index: int,
) -> AttemptEvidence | None:
    """Admit the exact contiguous predecessor for one new attempt."""
    parent = target.parent
    existing_indices = [int(entry.name[len(_RUN_PREFIX) :]) for entry in attempt_run_directories(parent)]
    expected_indices = list(range(1, attempt_index))
    if existing_indices != expected_indices:
        message = f"Attempt history must be contiguous before append; expected={expected_indices}, observed={existing_indices}: {parent}"
        raise ValueError(message)
    if attempt_index == 1:
        return None
    case_directory = common.paths.locate_generation_attempt_case_directory(target)
    return load_attempt(common.paths.resolve_generation_case_attempt_run_directory(case_directory, attempt_index - 1))


def _path_has_symlink(path: Path, *, stop: Path) -> bool:
    """Return whether an existing lexical component is a symbolic link."""
    current = path
    while current != stop:
        if current.is_symlink():
            return True
        current = current.parent
    return stop.is_symlink()


def _safe_attempt_history_root(path: Path, *, storage_root: Path | str) -> None:
    """Reject relocated attempt histories before listing or reading them."""
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    root = path.absolute()
    if not root.is_relative_to(storage) or _path_has_symlink(root, stop=storage):
        message = f"Attempt history escapes storage or contains a symbolic link: {root}"
        raise ValueError(message)


def _safe_attempt_target(
    path: Path,
    *,
    storage_root: Path,
    batch_storage_name: str,
    case_id: str,
    attempt_index: int,
) -> None:
    """Admit only the exact resolved canonical attempt target."""
    storage = storage_root.absolute()
    target = path.absolute()
    expected = common.paths.resolve_generation_attempt_run_directory(
        batch_storage_name,
        case_id,
        attempt_index,
        storage_root=storage,
    ).absolute()
    if target != expected or target == storage or not target.is_relative_to(storage):
        message = f"Attempt target disagrees with its authoritative storage locator: {target}"
        raise ValueError(message)
    if _path_has_symlink(target.parent, stop=storage):
        message = f"Attempt target contains a symbolic-link component: {target}"
        raise ValueError(message)
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        message = f"Attempt target is not one ordinary directory: {target}"
        raise ValueError(message)


def _ordinary_source(root: Path, relative: Path) -> Path | None:
    """Return one contained ordinary source file or None when absent."""
    if relative.is_absolute() or ".." in relative.parts:
        message = f"Attempt source path is unsafe: {relative}"
        raise ValueError(message)
    candidate = root / relative
    if not candidate.exists():
        return None
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root) or candidate.is_symlink():
        message = f"Attempt source escapes its owned work directory: {candidate}"
        raise ValueError(message)
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        return None
    return candidate


def _export_paths(
    config: GenerationConfig,
    work_directory: Path,
) -> tuple[Path, ...]:
    """Return the configured ordinary exports present in one case workspace."""
    export_root = Path(config.scientific_values["output_contract"]["exports_root"])
    paths: list[Path] = []
    for contract in config.scientific_values["output_contract"]["exports"]:
        pattern = contract.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            continue
        for source in sorted((work_directory / export_root).glob(pattern)):
            relative = source.relative_to(work_directory)
            if _ordinary_source(work_directory, relative) is not None:
                paths.append(relative)
    return tuple(dict.fromkeys(paths))


def _retention_paths(
    config: GenerationConfig,
    *,
    work_directory: Path | None,
) -> tuple[str, tuple[Path, ...], list[str]]:
    """Derive the retained and the omitted artifact paths of one failed attempt, whatever its failure stage."""
    policy = str(config.execution_values["retention_policy"])
    if policy not in {"full", "compact"}:
        message = f"Failed-attempt retention policy is not resolved: {policy!r}."
        raise ValueError(message)
    if work_directory is None:
        return policy, (), ["case_workspace"]
    export_paths = _export_paths(config, work_directory)
    case_hdf5 = Path("runtime/case.h5")
    retained: list[Path] = [relative for relative in _SMALL_RUNTIME_PATHS if _ordinary_source(work_directory, relative) is not None]
    if policy == "full":
        retained.extend(relative for relative in _LARGE_MODEL_PATHS if _ordinary_source(work_directory, relative) is not None)
        retained.extend(export_paths)
        if _ordinary_source(work_directory, case_hdf5) is not None:
            retained.append(case_hdf5)
    retained = list(dict.fromkeys(retained))
    omitted = [
        relative.as_posix()
        for relative in (*_LARGE_MODEL_PATHS, *export_paths, case_hdf5)
        if _ordinary_source(work_directory, relative) is not None and relative not in retained
    ]
    return policy, tuple(retained), sorted(omitted)


def derive_stage_states(
    case_state: AttemptCaseState,
    failure_stage: str,
) -> dict[str, str]:
    """Return precise persisted pipeline states for one unsuccessful attempt."""
    if case_state not in _CASE_STATES or failure_stage not in _FAILURE_STAGES:
        message = f"Unsupported attempt outcome: case_state={case_state!r}, failure_stage={failure_stage!r}."
        raise ValueError(message)
    states = {
        "solver_state": "not_started",
        "exports_state": "not_started",
        "conversion_state": "not_started",
        "diagnostics_state": "not_started",
        "publication_state": "not_started",
    }
    if failure_stage == "input":
        return states
    if failure_stage == "solver":
        states["solver_state"] = case_state
        return states
    states["solver_state"] = "succeeded"
    if failure_stage == "exports":
        states["exports_state"] = "failed"
        return states
    states["exports_state"] = "succeeded"
    if failure_stage == "conversion":
        states["conversion_state"] = "failed"
        return states
    states["conversion_state"] = "succeeded"
    states["diagnostics_state"] = "complete"
    states["publication_state"] = "failed"
    return states


def _copy_retained_file(
    source: Path,
    destination: Path,
    *,
    relative: Path,
    full_retention: bool,
) -> tuple[Path, ...]:
    """Copy an artifact; under compact retention a long log keeps a bounded copy plus its complete gzip copy.

    The bounded copy keeps the retained log path; the ``.gz`` sibling retains the
    complete log as diagnostic evidence.
    """
    maximum = _MAX_RETAINED_RUNTIME_LOG_BYTES
    if full_retention or relative not in _BOUNDED_RUNTIME_LOG_PATHS or source.stat().st_size <= maximum:
        shutil.copy2(source, destination)
        return (destination,)
    marker = b"\n... retained log middle omitted ...\n"
    if maximum <= len(marker):
        message = "Retained runtime-log byte bound is too small for its omission marker."
        raise RuntimeError(message)
    payload_bytes = maximum - len(marker)
    head_bytes = payload_bytes // 2
    tail_bytes = payload_bytes - head_bytes
    with source.open("rb") as stream:
        head = stream.read(head_bytes)
        stream.seek(-tail_bytes, os.SEEK_END)
        tail = stream.read(tail_bytes)
    destination.write_bytes(head + marker + tail)
    shutil.copystat(source, destination, follow_symlinks=False)
    complete = destination.with_name(f"{destination.name}.gz")
    with source.open("rb") as stream, gzip.open(complete, "wb") as compressed:
        shutil.copyfileobj(stream, compressed)
    return (destination, complete)


def _copy_retained(
    work_directory: Path,
    staging: Path,
    paths: Sequence[Path],
    *,
    full_retention: bool,
) -> dict[str, dict[str, Any]]:
    """Copy and hash one exact retained inventory beneath staging."""
    inventory: dict[str, dict[str, Any]] = {}
    for relative in paths:
        source = _ordinary_source(work_directory, relative)
        if source is None:
            continue
        destination = staging / "payload" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for produced in _copy_retained_file(source, destination, relative=relative, full_retention=full_retention):
            if produced.is_symlink() or not produced.is_file():
                message = f"Attempt retention did not produce an ordinary file: {produced}"
                raise RuntimeError(message)
            inventory[produced.relative_to(staging).as_posix()] = {
                "sha256": common.serialization.file_sha256(produced),
                "size_bytes": produced.stat().st_size,
            }
    return inventory


def _timing_evidence(work_directory: Path | None) -> dict[str, Any]:
    """Return available timing evidence without fabricating missing fields."""
    if work_directory is None:
        return {}
    path = work_directory / "runtime/timing.json"
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        return _load_json(path, label="attempt timing evidence")
    except (FileNotFoundError, TypeError, ValueError):
        return {}


def _raw_reference(path: Path, *, storage_root: Path) -> str:
    """Return one storage-relative canonical raw case locator."""
    resolved = path.resolve()
    storage = storage_root.resolve()
    if not resolved.is_relative_to(storage):
        message = f"Canonical raw case reference escapes storage: {resolved}"
        raise ValueError(message)
    return resolved.relative_to(storage).as_posix()


def _reject_success_marker(root: Path) -> None:
    """Reject a forbidden success marker anywhere in an attempt bundle."""
    if any(path.name == "_SUCCESS" for path in root.rglob("*")):
        message = "Attempt bundles must never contain _SUCCESS."
        raise ValueError(message)


def _require_previous_attempt_identity(
    previous: AttemptEvidence | None,
    receipt: Mapping[str, Any],
) -> None:
    """Require one appended attempt to preserve its predecessor identity."""
    if previous is not None and any(previous.payload.get(key) != receipt.get(key) for key in _ATTEMPT_CHAIN_IDENTITY_KEYS):
        message = f"Previous attempt identity conflicts with the new append: {previous.receipt_path}"
        raise ValueError(message)


def publish_case_attempt(
    config: GenerationConfig,
    case_index: int,
    *,
    campaign_run_id: str,
    case_state: AttemptCaseState,
    failure_stage: str,
    reason: str,
    solver_git_commit: str,
    processing_git_commit: str,
    work_directory: Path | None,
    storage_root: Path | str,
    worker_slot: int,
    scheduler_kind: str,
    allocated_node: str | None,
    exit_code: int | None,
    timed_out: bool,
    attempt_index: int | None = None,
    solver_metrics: Mapping[str, Any] | None = None,
    quality_flags: Sequence[DiagnosticRecord] = (),
    input_generation_id: str | None = None,
) -> AttemptEvidence:
    """Atomically publish one stage-derived unsuccessful attempt bundle."""
    if not reason:
        message = "Attempt failure reason must be non-empty."
        raise ValueError(message)
    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    purpose = str(config.scientific_values["campaign_purpose"])
    case_id = config.case_id(case_index)
    index = _attempt_index(
        config,
        case_index,
        storage_root=storage,
        requested=attempt_index,
    )
    target = common.paths.resolve_generation_attempt_run_directory(
        config.batch_storage_name,
        case_id,
        index,
        storage_root=storage,
    )
    _safe_attempt_target(
        target,
        storage_root=storage,
        batch_storage_name=config.batch_storage_name,
        case_id=case_id,
        attempt_index=index,
    )
    if target.exists():
        existing = load_attempt(target)
        identity = existing.payload
        if (
            identity.get("case_state") == case_state
            and identity.get("failure_stage") == failure_stage
            and identity.get("reason") == reason
            and identity.get("solver_git_commit") == solver_git_commit
            and identity.get("processing_git_commit") == processing_git_commit
        ):
            return existing
        message = f"Attempt index already contains conflicting evidence: {target}"
        raise FileExistsError(message)

    (
        case_payload,
        case_payload_path,
        admitted_input_generation_id,
    ) = _case_payload(
        config,
        case_index,
        work_directory=work_directory,
        storage_root=storage,
        input_generation_id=input_generation_id,
    )
    dispatch = dispatch_identity.current_case_dispatch()
    policy, retained_paths, omitted = _retention_paths(config, work_directory=work_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    _safe_attempt_target(
        target,
        storage_root=storage,
        batch_storage_name=config.batch_storage_name,
        case_id=case_id,
        attempt_index=index,
    )
    previous = _previous_attempt_evidence(target, index)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        inventory = (
            {}
            if work_directory is None
            else _copy_retained(
                work_directory,
                staging,
                retained_paths,
                full_retention=policy == "full",
            )
        )
        timing = _timing_evidence(work_directory)
        metrics = dict(solver_metrics or {})
        unexpectedly_missing = [f"payload/{relative.as_posix()}" for relative in retained_paths if f"payload/{relative.as_posix()}" not in inventory]
        stage_states = derive_stage_states(case_state, failure_stage)
        receipt = {
            "schema_kind": ATTEMPT_SCHEMA_KIND,
            "campaign_run_id": campaign_run_id,
            "campaign_purpose": purpose,
            "batch_storage_name": config.batch_storage_name,
            "batch_id": config.batch_id,
            "batch_identity": config.batch_identity,
            "input_generation_id": admitted_input_generation_id,
            "case_id": config.case_id(case_index),
            "case_index": case_index,
            "case_input_id": case_payload["case_input_id"],
            "simulation_case_id": case_payload["simulation_case_id"],
            "attempt_index": index,
            "previous_attempt": (
                None
                if previous is None
                else {
                    "attempt_index": int(previous.payload["attempt_index"]),
                    "receipt_sha256": common.serialization.file_sha256(previous.receipt_path),
                }
            ),
            "case_state": case_state,
            **stage_states,
            "failure_stage": failure_stage,
            "reason": reason,
            "solver_git_commit": solver_git_commit,
            "processing_git_commit": processing_git_commit,
            "solver_execution_provenance": "payload/runtime/execution_provenance.json",
            "processing_provenance": "payload/runtime/processing_provenance.json",
            "template": {
                "relative_path": config.template_relative_path,
                "sha256": config.template_sha256,
            },
            "scientific_config_digest": config.scientific_config_digest,
            "export_contract_sha256": common.serialization.canonical_json_sha256(config.scientific_values["output_contract"]),
            "canonical_raw_case": _raw_reference(case_payload_path.parent, storage_root=storage),
            "input_files": case_payload["input_files"],
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": None if dispatch is None else dispatch.step_id,
            "submission_index": None if dispatch is None else dispatch.submission_index,
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": allocated_node,
            "allocated_cores": os.environ.get("SLURM_CPUS_PER_TASK"),
            "worker_slot": worker_slot,
            "scheduler_kind": scheduler_kind,
            "started_at": timing.get("started_at"),
            "ended_at": timing.get("ended_at") or _utc_now(),
            "elapsed_seconds": timing.get("runtime_s"),
            "final_simulation_time": metrics.get("simulated_time_seconds"),
            "last_accepted_step_size": metrics.get("step_size_seconds"),
            "Tfail": metrics.get("time_failures"),
            "NLfail": metrics.get("nonlinear_failures"),
            "process_exit_code": exit_code,
            "timed_out": timed_out,
            "retention_policy": policy,
            "retained_inventory": inventory,
            "intentionally_omitted_artifacts": omitted,
            "unexpectedly_missing_artifacts": unexpectedly_missing,
            "quality_flags": [dict(record) for record in quality_flags],
            "recorded_at": _utc_now(),
        }
        _require_previous_attempt_identity(previous, receipt)
        common.serialization.atomic_write_json(staging / "attempt.json", receipt)
        _reject_success_marker(staging)
        source_service.validate_admitted_source_before_publication()
        staging.replace(target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_attempt(target)


def _safe_relative_text(value: Any, *, prefix: str | None = None) -> bool:
    """Return whether one receipt path is relative, normalized, and contained."""
    if not isinstance(value, str) or not value:
        return False
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        return False
    return prefix is None or value.startswith(prefix)


def _nonempty_text(value: Any) -> bool:
    """Return whether one receipt value is non-empty text without controls."""
    return isinstance(value, str) and bool(value) and not any(character in value for character in "\r\n\t")


def _optional_nonnegative_number(value: Any) -> bool:
    """Return whether one optional metric is a finite non-negative number."""
    return value is None or (not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0 and value < float("inf"))


def _validate_quality_flags(value: Any, *, path: Path) -> None:
    """Admit complete advisory diagnostic records."""
    if not isinstance(value, list):
        message = f"Attempt quality flags are invalid: {path}"
        raise TypeError(message)
    for record in value:
        if (
            not isinstance(record, dict)
            or set(record) != _DIAGNOSTIC_KEYS
            or not all(_nonempty_text(record.get(key)) for key in ("code", "stage", "message", "recorded_at"))
            or record.get("severity") not in {"info", "warning"}
            or not isinstance(record.get("metrics"), dict)
            or not isinstance(record.get("thresholds"), dict)
            or not isinstance(record.get("source_artifacts"), list)
            or not all(_safe_relative_text(source) for source in record.get("source_artifacts", []))
            or not isinstance(record.get("quality_flag"), bool)
        ):
            message = f"Attempt quality flag is malformed: {path}"
            raise ValueError(message)


def _valid_previous_attempt_reference(value: Any, *, attempt_index: int) -> bool:
    """Return whether one receipt has the exact immediate-predecessor reference."""
    if attempt_index == 1:
        return value is None
    return (
        isinstance(value, dict)
        and set(value) == _PREVIOUS_ATTEMPT_KEYS
        and value.get("attempt_index") == attempt_index - 1
        and isinstance(value.get("receipt_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["receipt_sha256"]) is not None
    )


def _validate_attempt_receipt(payload: dict[str, Any], *, path: Path) -> None:
    """Admit the complete exact attempt receipt header."""
    case_state = payload.get("case_state")
    failure_stage = payload.get("failure_stage")
    if (
        set(payload) != _ATTEMPT_RECEIPT_KEYS
        or payload.get("schema_kind") != ATTEMPT_SCHEMA_KIND
        or case_state not in _CASE_STATES
        or failure_stage not in _FAILURE_STAGES
    ):
        message = f"Unsupported or malformed attempt receipt schema: {path}"
        raise ValueError(message)
    expected_stages = derive_stage_states(case_state, str(failure_stage))
    if any(payload.get(key) != value for key, value in expected_stages.items()):
        message = f"Attempt stage states are inconsistent: {path}"
        raise ValueError(message)
    positive_integers = ("case_index", "attempt_index")
    optional_integers = ("Tfail", "NLfail")
    if (
        not all(
            _nonempty_text(payload.get(key))
            for key in (
                "campaign_run_id",
                "campaign_purpose",
                "batch_storage_name",
                "batch_id",
                "batch_identity",
                "input_generation_id",
                "case_id",
                "case_input_id",
                "simulation_case_id",
                "reason",
                "scheduler_kind",
                "ended_at",
                "recorded_at",
            )
        )
        or not all(
            isinstance(payload.get(key), int) and not isinstance(payload.get(key), bool) and int(payload[key]) >= 1 for key in positive_integers
        )
        or not isinstance(payload.get("worker_slot"), int)
        or isinstance(payload.get("worker_slot"), bool)
        or int(payload["worker_slot"]) < 0
        or not all(
            value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)
            for value in (payload.get(key) for key in optional_integers)
        )
        or not all(
            _optional_nonnegative_number(payload.get(key))
            for key in (
                "elapsed_seconds",
                "final_simulation_time",
                "last_accepted_step_size",
            )
        )
        or not _valid_previous_attempt_reference(
            payload.get("previous_attempt"),
            attempt_index=int(payload.get("attempt_index", 0)),
        )
        or not isinstance(payload.get("timed_out"), bool)
        or payload["timed_out"] is not (case_state == "timed_out")
        or payload.get("retention_policy") not in _RETENTION_POLICIES
    ):
        message = f"Attempt receipt values are malformed: {path}"
        raise ValueError(message)
    for key in ("solver_git_commit", "processing_git_commit"):
        if not isinstance(payload.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", payload[key]) is None:
            message = f"Attempt source commit is malformed: {path}"
            raise ValueError(message)
    for key in (
        "scientific_config_digest",
        "export_contract_sha256",
    ):
        if not isinstance(payload.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None:
            message = f"Attempt digest is malformed: {path}"
            raise ValueError(message)
    template = payload.get("template")
    if (
        not isinstance(template, dict)
        or set(template) != {"relative_path", "sha256"}
        or not _safe_relative_text(template.get("relative_path"))
        or not isinstance(template.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", template["sha256"]) is None
        or not _safe_relative_text(payload.get("canonical_raw_case"))
        or payload.get("solver_execution_provenance") != "payload/runtime/execution_provenance.json"
        or payload.get("processing_provenance") != "payload/runtime/processing_provenance.json"
        or not isinstance(payload.get("input_files"), dict)
    ):
        message = f"Attempt identity references are malformed: {path}"
        raise ValueError(message)
    for key in ("job_id", "job_name", "node", "started_at"):
        if payload.get(key) is not None and not _nonempty_text(payload.get(key)):
            message = f"Attempt scheduler or timing evidence is malformed: {path}"
            raise ValueError(message)
    step_id = payload.get("slurm_step_id")
    submission_index = payload.get("submission_index")
    if (step_id is None) != (submission_index is None) or (
        step_id is not None
        and (
            not isinstance(step_id, str)
            or not step_id.isascii()
            or not step_id.isdigit()
            or isinstance(submission_index, bool)
            or not isinstance(submission_index, int)
            or submission_index < 1
        )
    ):
        message = f"Attempt dispatch identity is malformed: {path}"
        raise ValueError(message)
    allocated = payload.get("allocated_cores")
    exit_code = payload.get("process_exit_code")
    if (allocated is not None and (not isinstance(allocated, str) or not allocated.isdigit())) or (
        exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool))
    ):
        message = f"Attempt process evidence is malformed: {path}"
        raise ValueError(message)
    for key in (
        "intentionally_omitted_artifacts",
        "unexpectedly_missing_artifacts",
    ):
        values = payload.get(key)
        if not isinstance(values, list) or not all(_safe_relative_text(value) for value in values):
            message = f"Attempt artifact declaration is malformed: {path}"
            raise ValueError(message)
    _validate_quality_flags(payload.get("quality_flags"), path=path)


def _validate_cleanup_audit(
    directory: Path,
    *,
    attempt_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Admit an optional exact scratch-cleanup audit."""
    path = directory / "cleanup.json"
    if not path.exists():
        return None
    payload = _load_json(path, label="attempt cleanup audit")
    if (
        set(payload) != _CLEANUP_RECEIPT_KEYS
        or payload.get("schema_kind") != CLEANUP_SCHEMA_KIND
        or payload.get("campaign_run_id") != attempt_payload["campaign_run_id"]
        or payload.get("case_id") != attempt_payload["case_id"]
        or payload.get("attempt_index") != attempt_payload["attempt_index"]
        or payload.get("status") not in {"complete", "failed", "not_created"}
        or not isinstance(payload.get("reclaimed_bytes"), int)
        or isinstance(payload.get("reclaimed_bytes"), bool)
        or payload["reclaimed_bytes"] < 0
        or (payload.get("error") is not None and not _nonempty_text(payload["error"]))
        or not _nonempty_text(payload.get("recorded_at"))
    ):
        message = f"Attempt cleanup audit is invalid: {path}"
        raise ValueError(message)
    return payload


def _license_compaction_audit(
    directory: Path,
    *,
    attempt_payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, frozenset[str]]:
    """Admit optional crash-safe cleanup evidence for a license-only payload."""
    path = directory / "license_compaction.json"
    if not path.exists():
        return None, frozenset()
    payload = _load_json(path, label="license-attempt compaction audit")
    removed = payload.get("removed_inventory")
    if (
        set(payload) != _LICENSE_COMPACTION_KEYS
        or payload.get("schema_kind") != LICENSE_COMPACTION_SCHEMA_KIND
        or payload.get("status") not in {"pending", "complete"}
        or payload.get("campaign_run_id") != attempt_payload["campaign_run_id"]
        or payload.get("batch_id") != attempt_payload["batch_id"]
        or payload.get("case_id") != attempt_payload["case_id"]
        or payload.get("attempt_index") != attempt_payload["attempt_index"]
        or payload.get("attempt_receipt_sha256") != common.serialization.file_sha256(directory / "attempt.json")
        or not isinstance(payload.get("license_wait_identity"), str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["license_wait_identity"]) is None
        or not isinstance(removed, dict)
        or set(removed) != set(attempt_payload["retained_inventory"])
        or any(removed[key] != attempt_payload["retained_inventory"][key] for key in removed)
        or not isinstance(payload.get("reclaimed_bytes"), int)
        or isinstance(payload.get("reclaimed_bytes"), bool)
        or payload["reclaimed_bytes"] < 0
        or not _nonempty_text(payload.get("recorded_at"))
    ):
        message = f"License-attempt compaction audit is invalid: {path}"
        raise ValueError(message)
    expected_bytes = sum(int(identity["size_bytes"]) for identity in removed.values())
    if payload["status"] == "complete" and payload["reclaimed_bytes"] != expected_bytes:
        message = f"License-attempt reclaimed bytes disagree with its inventory: {path}"
        raise ValueError(message)
    return payload, frozenset(str(value) for value in removed)


def _validate_attempt_chain(
    directory: Path,
    payload: Mapping[str, Any],
) -> None:
    """Validate every immediate receipt link without rereading retained payload bytes."""
    current_directory = directory
    current_payload = dict(payload)
    case_directory = common.paths.locate_generation_attempt_case_directory(directory)
    while True:
        index = int(current_payload["attempt_index"])
        if current_directory != common.paths.resolve_generation_case_attempt_run_directory(case_directory, index):
            message = f"Attempt directory name disagrees with its receipt index: {current_directory}"
            raise ValueError(message)
        previous = current_payload["previous_attempt"]
        if index == 1:
            return
        if not isinstance(previous, dict):
            message = f"Attempt predecessor reference is malformed: {current_directory / 'attempt.json'}"
            raise TypeError(message)
        previous_directory = common.paths.resolve_generation_case_attempt_run_directory(case_directory, index - 1)
        previous_path = previous_directory / "attempt.json"
        previous_payload = _load_json(previous_path, label="previous attempt receipt")
        _validate_attempt_receipt(previous_payload, path=previous_path)
        if (
            previous["attempt_index"] != previous_payload["attempt_index"]
            or previous["receipt_sha256"] != common.serialization.file_sha256(previous_path)
            or any(previous_payload.get(key) != current_payload.get(key) for key in _ATTEMPT_CHAIN_IDENTITY_KEYS)
        ):
            message = f"Attempt predecessor chain is inconsistent: {current_directory / 'attempt.json'}"
            raise ValueError(message)
        current_directory = previous_directory
        current_payload = previous_payload


def load_attempt(directory: Path | str) -> AttemptEvidence:
    """Admit one attempt receipt, inventory, and optional cleanup and license-compaction audits."""
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        message = f"Attempt directory is missing or unsafe: {root}"
        raise FileNotFoundError(message)
    if any(path.name == "_SUCCESS" for path in root.rglob("*")):
        message = f"Attempt directory must not contain _SUCCESS: {root}"
        raise ValueError(message)
    receipt_path = root / "attempt.json"
    payload = _load_json(receipt_path, label="attempt receipt")
    _validate_attempt_receipt(payload, path=receipt_path)
    _validate_attempt_chain(root, payload)
    inventory = payload.get("retained_inventory")
    if not isinstance(inventory, dict):
        message = f"Attempt retained inventory is invalid: {receipt_path}"
        raise TypeError(message)
    _validate_cleanup_audit(root, attempt_payload=payload)
    compaction, compacted = _license_compaction_audit(
        root,
        attempt_payload=payload,
    )
    for relative, identity in inventory.items():
        if relative in compacted:
            compacted_path = root / relative
            if compaction is not None and compaction["status"] == "complete" and compacted_path.exists():
                message = f"License compaction claims a payload was removed but it still exists: {relative}"
                raise ValueError(message)
            if not compacted_path.exists():
                continue
        path = root / relative
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(identity, dict)
            or set(identity) != {"sha256", "size_bytes"}
            or not isinstance(identity.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
            or not isinstance(identity.get("size_bytes"), int)
            or isinstance(identity.get("size_bytes"), bool)
            or identity["size_bytes"] < 0
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != identity.get("size_bytes")
            or common.serialization.file_sha256(path) != identity.get("sha256")
        ):
            message = f"Attempt retained artifact identity is invalid: {path}"
            raise ValueError(message)
    return AttemptEvidence(directory=root, receipt_path=receipt_path, payload=payload)


def attempt_cleanup_evidence(
    attempt: AttemptEvidence,
) -> dict[str, Any] | None:
    """Return admitted terminal scratch-cleanup evidence when recorded."""
    return _validate_cleanup_audit(
        attempt.directory,
        attempt_payload=attempt.payload,
    )


def compact_license_only_attempt_payload(
    config: GenerationConfig,
    case_index: int,
    campaign_run_id: str,
    license_wait: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> dict[str, Any] | None:
    """Remove only hash-verified duplicate payload from one license-only attempt."""
    from src.generation.runtime import generation_runtime_license as license_service  # noqa: PLC0415

    storage = common.paths.get_storage_root(storage_root=storage_root).resolve()
    root = common.paths.resolve_generation_attempts_directory(
        config.batch_storage_name,
        config.case_id(case_index),
        storage_root=storage,
    )
    if not root.is_dir() or root.is_symlink():
        return None
    candidates = attempt_run_directories(root)
    if not candidates:
        return None
    attempt = load_attempt(candidates[-1])
    existing, _removed = _license_compaction_audit(
        attempt.directory,
        attempt_payload=attempt.payload,
    )
    if existing is not None and existing["status"] == "complete":
        return existing
    payload = attempt.payload
    expected_stage_states = {
        "solver_state": "license_blocked",
        "exports_state": "not_started",
        "conversion_state": "not_started",
        "diagnostics_state": "not_started",
        "publication_state": "not_started",
    }
    if (
        payload.get("case_state") != "license_blocked"
        or payload.get("failure_stage") != "solver"
        or any(payload.get(key) != value for key, value in expected_stage_states.items())
        or license_wait.get("classification") != license_service.TEMPORARY_LICENSE_CAPACITY
        or license_wait.get("campaign_run_id") != campaign_run_id
        or license_wait.get("batch_id") != config.batch_id
        or license_wait.get("case_id") != config.case_id(case_index)
        or license_wait.get("scientific_config_digest") != config.scientific_config_digest
        or license_wait.get("solver_progress_started", False) is not False
        or license_wait.get("expected_exports_exist", False) is not False
    ):
        return None
    reference = input_service.admit_persisted_input_case(
        config,
        case_index,
        str(payload["input_generation_id"]),
        storage_root=storage,
    )
    if reference.case_directory.relative_to(storage).as_posix() != payload["canonical_raw_case"]:
        message = f"License-only attempt canonical input is not independently admitted: {attempt.directory}"
        raise ValueError(message)
    inventory = payload["retained_inventory"]
    if not isinstance(inventory, dict):
        message = f"License-only attempt inventory is malformed: {attempt.receipt_path}"
        raise TypeError(message)
    export_root = Path(config.scientific_values["output_contract"]["exports_root"])
    forbidden_prefix = f"payload/{export_root.as_posix()}/"
    forbidden = [
        relative
        for relative in inventory
        if relative == "payload/runtime/case.h5" or relative.startswith(forbidden_prefix) or Path(relative).name == "_SUCCESS"
    ]
    if forbidden:
        return None
    raw_excerpt = license_wait.get("raw_excerpt")
    if not isinstance(raw_excerpt, str) or not raw_excerpt:
        evidence_parts: list[str] = []
        for relative in (
            "payload/runtime/stdout.log",
            "payload/runtime/stderr.log",
        ):
            path = attempt.directory / relative
            if path.is_file() and not path.is_symlink():
                evidence_parts.append(path.read_text(encoding="utf-8", errors="replace"))
        raw_excerpt = "\n".join(evidence_parts)
    classification = license_service.classify_temporary_license_capacity(raw_excerpt)
    if classification is None or license_service.solver_progress_started(raw_excerpt):
        return None
    feature = license_wait.get("feature")
    error_code = license_wait.get("error_code")
    if classification.feature != feature or classification.license_code != error_code:
        return None
    wait_identity = common.serialization.canonical_json_sha256(dict(license_wait))
    removed_inventory = copy.deepcopy(inventory)
    reclaimed_bytes = sum(int(identity["size_bytes"]) for identity in removed_inventory.values())
    audit = {
        "schema_kind": LICENSE_COMPACTION_SCHEMA_KIND,
        "status": "pending",
        "campaign_run_id": campaign_run_id,
        "batch_id": config.batch_id,
        "case_id": config.case_id(case_index),
        "attempt_index": payload["attempt_index"],
        "attempt_receipt_sha256": common.serialization.file_sha256(attempt.receipt_path),
        "license_wait_identity": wait_identity,
        "removed_inventory": removed_inventory,
        "reclaimed_bytes": 0,
        "recorded_at": _utc_now(),
    }
    audit_path = attempt.directory / "license_compaction.json"
    if existing is None:
        common.serialization.atomic_write_json(audit_path, audit)
    elif existing["license_wait_identity"] != wait_identity or existing["removed_inventory"] != removed_inventory:
        message = f"License-only compaction continuation conflicts with its pending audit: {audit_path}"
        raise ValueError(message)
    for relative, identity in removed_inventory.items():
        path = attempt.directory / relative
        if not path.exists():
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(attempt.directory.resolve())
            or path.stat().st_size != identity["size_bytes"]
            or common.serialization.file_sha256(path) != identity["sha256"]
        ):
            message = f"License-only cleanup payload changed after admission: {path}"
            raise ValueError(message)
        path.unlink()
    payload_root = attempt.directory / "payload"
    if payload_root.is_dir() and not payload_root.is_symlink():
        for directory in sorted((path for path in payload_root.rglob("*") if path.is_dir()), reverse=True):
            with suppress(OSError):
                directory.rmdir()
        with suppress(OSError):
            payload_root.rmdir()
    complete = {**audit, "status": "complete", "reclaimed_bytes": reclaimed_bytes, "recorded_at": _utc_now()}
    common.serialization.atomic_write_json(audit_path, complete)
    load_attempt(attempt.directory)
    return complete


def latest_case_attempt(
    config: GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str,
) -> AttemptEvidence | None:
    """Return the newest admitted failure bundle of one exact campaign case."""
    root = common.paths.resolve_generation_attempts_directory(
        config.batch_storage_name,
        config.case_id(case_index),
        storage_root=storage_root,
    )
    _safe_attempt_history_root(root, storage_root=storage_root)
    if not root.exists():
        return None
    if not root.is_dir() or root.is_symlink():
        message = f"Attempt history path is unsafe: {root}"
        raise ValueError(message)
    candidates = attempt_run_directories(root)
    return None if not candidates else load_attempt(candidates[-1])


def record_attempt_cleanup(
    attempt: AttemptEvidence,
    *,
    status: str,
    reclaimed_bytes: int,
    error: str | None,
) -> Path:
    """Append one immutable scratch-cleanup audit beside an attempt receipt."""
    if status not in {"complete", "failed", "not_created"}:
        message = f"Unsupported attempt cleanup status: {status!r}."
        raise ValueError(message)
    if isinstance(reclaimed_bytes, bool) or not isinstance(reclaimed_bytes, int) or reclaimed_bytes < 0:
        message = "Attempt cleanup reclaimed_bytes must be a non-negative integer."
        raise ValueError(message)
    target = attempt.directory / "cleanup.json"
    receipt = {
        "schema_kind": CLEANUP_SCHEMA_KIND,
        "campaign_run_id": attempt.payload["campaign_run_id"],
        "case_id": attempt.payload["case_id"],
        "attempt_index": attempt.payload["attempt_index"],
        "status": status,
        "reclaimed_bytes": reclaimed_bytes,
        "error": error,
        "recorded_at": _utc_now(),
    }
    if target.exists():
        existing = _load_json(target, label="attempt cleanup audit")
        comparable = {key: value for key, value in existing.items() if key != "recorded_at"}
        expected = {key: value for key, value in receipt.items() if key != "recorded_at"}
        if comparable != expected:
            message = f"Existing attempt cleanup audit conflicts: {target}"
            raise FileExistsError(message)
        return target
    return common.serialization.atomic_write_json(target, receipt)
