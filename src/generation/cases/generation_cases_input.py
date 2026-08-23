"""
generation_cases_input.py

Publish and plan canonical pre-execution inputs for maintained generation batches.
Responsibilities:
  - Initialize shared batch metadata without duplicating runtime ownership
  - Atomically publish bounded deterministic cases into canonical raw storage
  - Validate configured persisted cases before runtime materialization
  - Plan campaign selections without mutating persistent storage
Design principles:
  - One batch metadata and raw location serves EDA and later COMSOL execution
  - Existing cases are reused only after complete manifest and byte validation
  - Conflicting or incomplete canonical evidence always fails closed
This module does NOT:
  - Execute COMSOL, publish processed results, or admit alternate storage layouts
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

from src import common
from src.generation.contracts import generation_contracts_source as source_service

from . import generation_cases_admission as admission_service
from . import generation_cases_case as case_service
from . import generation_cases_config as config_service

INPUT_TRANSACTION_SCHEMA_VERSION = 1
INPUT_GENERATION_ACTIONS: Final = ("dry_run", "execute")


@dataclass(frozen=True, slots=True)
class CampaignInputGenerationRequest:
    """Describe one thin campaign-level canonical input-generation request."""

    campaign_config: Path | str
    storage_root: Path | str
    action: str = "dry_run"
    only_batch: str | None = None
    all_batches: bool = False
    only_regime: str | None = None
    case_start: int | None = None
    case_count: int | None = None
    all_cases: bool = False
    git_commit: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratedInputBatch:
    """Reference one canonical raw-input batch after a bounded request."""

    input_generation_id: str
    metadata_directory: Path
    raw_directory: Path
    requested_case_indices: tuple[int, ...]
    case_indices: tuple[int, ...]
    generated_case_count: int
    reused_case_count: int


@dataclass(frozen=True, slots=True)
class InputGenerationPlan:
    """Describe one read-only bounded canonical input-generation plan."""

    batch_id: str
    batch_storage_name: str
    batch_name: str
    raw_directory: Path
    metadata_directory: Path
    requested_case_indices: tuple[int, ...]
    reusable_case_indices: tuple[int, ...]
    missing_case_indices: tuple[int, ...]
    estimated_storage_bytes: int


def _resolved_config(config: config_service.GenerationConfig) -> dict[str, Any]:
    """Return the shared immutable resolved scientific batch configuration."""
    return copy.deepcopy(config.scientific_values)


def _manifest_base(
    config: config_service.GenerationConfig,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build source identity that remains stable while case membership grows."""
    scientific = config.scientific_values
    base = {
        "schema_kind": admission_service.INPUT_BATCH_SCHEMA_KIND,
        "schema_version": admission_service.INPUT_BATCH_SCHEMA_VERSION,
        "campaign_id": scientific["campaign_id"],
        "campaign_purpose": scientific["campaign_purpose"],
        "batch_name": config.batch_name,
        "batch_id": config.batch_id,
        "batch_storage_name": config.batch_storage_name,
        "batch_identity": config.batch_identity,
        "simulation_profile": config.profile.id,
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "case_input_config_digest": config.case_input_config_digest,
        "scientific_config_digest": config.scientific_config_digest,
        "git_commit": case_service.source_service.required_git_commit(),
        "generator_version": scientific["generator_version"],
        "case_schema_version": case_service.CASE_SCHEMA_VERSION,
        "case_contract_digest": case_service.CASE_CONTRACT_DIGEST,
        "template_relative_path": config.template_relative_path,
        "template_sha256": config.template_sha256,
        "resolved_config_sha256": common.serialization.canonical_json_sha256(resolved_config),
    }
    return {
        **base,
        "input_generation_id": admission_service.compute_input_generation_id(base),
        "status": "ready",
    }


def configured_input_generation_id(
    config: config_service.GenerationConfig,
) -> str:
    """
    Return the exact current input-generation identity for one batch.

    Parameters
    ----------
    config : GenerationConfig
        Fully resolved batch configuration bound to the active source commit.

    Returns
    -------
    str
        Logical identity of the sole current schema-v1 input generation.

    """
    return str(_manifest_base(config, _resolved_config(config))["input_generation_id"])


def admit_persisted_input_case(
    config: config_service.GenerationConfig,
    case_index: int,
    input_generation_id: str,
    *,
    storage_root: Path | str | None = None,
    validation_depth: Literal["evidence", "full"] = "full",
) -> admission_service.InputCaseReference:
    """
    Admit one persisted input case against its immutable scientific identity.

    Parameters
    ----------
    config : GenerationConfig
        Active scientific configuration used to validate stable case identity.
    case_index : int
        Configured case member to admit.
    input_generation_id : str
        Persisted generation identity, including its original source commit.
    storage_root : Path | str | None, optional
        Canonical storage root override.
    validation_depth : {"evidence", "full"}, optional
        Metadata evidence or selected-case content validation depth.

    Returns
    -------
    InputCaseReference
        Exact persisted case reference admitted through its batch manifest.

    Raises
    ------
    RuntimeError
        If the persisted source disagrees with the active scientific identity.

    Notes
    -----
    Source commit is intentionally owned by the persisted input generation.
    This permits postprocessing replay under newer code without rebinding the
    solver inputs to that newer processing commit.

    """
    case_id = config.case_id(case_index)
    generation_id = common.paths.validate_logical_name(
        input_generation_id,
        label="input_generation_id",
    )
    metadata_directory = common.paths.resolve_generation_input_generation_metadata_directory(
        config.batch_storage_name,
        generation_id,
        storage_root=storage_root,
    )
    raw_directory = common.paths.resolve_generation_input_generation_raw_directory(
        config.batch_storage_name,
        generation_id,
        storage_root=storage_root,
    )
    reference = admission_service.admit_input_case_evidence(
        metadata_directory,
        case_index,
        raw_directory=raw_directory,
        expected_input_generation_id=generation_id,
        validation_depth=validation_depth,
    )
    expected = {
        "source_id": generation_id,
        "profile_id": config.profile.id,
        "batch_id": config.batch_id,
        "batch_storage_name": config.batch_storage_name,
        "batch_identity": config.batch_identity,
        "material_family": config.material_family,
        "sampling_regime": config.sampling_regime,
        "campaign_purpose": config.scientific_values["campaign_purpose"],
        "case_id": case_id,
        "case_index": case_index,
    }
    if any(getattr(reference, key) != value for key, value in expected.items()):
        message = f"Persisted input generation disagrees with the active scientific configuration: {metadata_directory}"
        raise RuntimeError(message)
    return reference


def select_case_indices(
    config: config_service.GenerationConfig,
    case_count: int,
    case_start: int | None,
) -> tuple[int, ...]:
    """Resolve one contiguous bounded request within configured membership."""
    if isinstance(case_count, bool) or not isinstance(case_count, int) or not 0 < case_count <= len(config.case_indices):
        message = "case_count must be positive and no greater than configured batch membership."
        raise ValueError(message)
    if case_start is not None and (isinstance(case_start, bool) or not isinstance(case_start, int)):
        message = "case_start must be an integer when supplied."
        raise ValueError(message)
    start = config.case_indices[0] if case_start is None else case_start
    if start not in config.case_indices:
        message = "case_start must be a configured batch member."
        raise ValueError(message)
    position = config.case_indices.index(start)
    selected = tuple(config.case_indices[position : position + case_count])
    if len(selected) != case_count:
        message = "case_start plus case_count exceeds configured batch membership."
        raise ValueError(message)
    return selected


def _case_record(directory: Path) -> dict[str, Any]:
    """Build manifest identity and hash evidence for one generated case."""
    case_json = directory / "case.json"
    serialized = case_json.read_bytes()
    payload = json.loads(serialized)
    if not isinstance(payload, dict):
        message = f"Generated case payload must be a JSON object: {case_json}"
        raise TypeError(message)
    case_service.validate_case_payload_schema(payload)
    return {
        "case_index": payload["case_index"],
        "case_id": payload["case_id"],
        "case_input_id": payload["case_input_id"],
        "simulation_case_id": payload["simulation_case_id"],
        "case_json_sha256": hashlib.sha256(serialized).hexdigest(),
        "seed_evidence_sha256": common.serialization.canonical_json_sha256(payload["seed_evidence"]),
        "input_files": payload["input_files"],
    }


def _mutable_json_evidence(value: Any) -> Any:
    """Copy validated frozen JSON evidence without serializing it again."""
    if isinstance(value, Mapping):
        return {str(key): _mutable_json_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json_evidence(item) for item in value]
    return value


def _complete_manifest(
    base: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach sorted canonical membership to stable input-generation identity."""
    indices = sorted(records)
    return {
        **base,
        "case_indices": indices,
        "cases": [_mutable_json_evidence(records[index]) for index in indices],
    }


def _immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> Path:
    """Write or validate one immutable deterministic JSON object."""
    serialized = json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_text(encoding="utf-8") != serialized:
            message = f"Existing {label} disagrees with the resolved configuration: {path}"
            raise RuntimeError(message)
        return path
    return common.serialization.atomic_write_text(path, serialized)


def _initialize_input_generation_metadata(
    config: config_service.GenerationConfig,
    resolved_config: Mapping[str, Any],
    input_generation_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Initialize immutable metadata for one exact input generation."""
    directory = common.paths.resolve_generation_input_generation_metadata_directory(
        config.batch_storage_name,
        input_generation_id,
        storage_root=storage_root,
    )
    directory.mkdir(parents=True, exist_ok=True)
    return _immutable_json(
        directory / "resolved_generation_config.json",
        resolved_config,
        label="resolved scientific input-generation configuration",
    )


def _load_existing(
    metadata_directory: Path,
    raw_directory: Path,
    *,
    base: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    validation_depth: Literal["evidence", "full"] = "full",
) -> tuple[dict[int, Mapping[str, Any]], admission_service.InputSource]:
    """Admit existing canonical evidence and require the same source request."""
    try:
        source = admission_service.admit_input_batch_source(
            metadata_directory,
            raw_directory=raw_directory,
            expected_input_generation_id=str(base["input_generation_id"]),
            validation_depth=validation_depth,
        )
        manifest = source.manifest_payload()
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        message = f"Canonical input-generation evidence is incomplete or invalid: {metadata_directory}: {error}"
        raise FileExistsError(message) from error
    if not source.resolved_config_matches(resolved_config) or any(manifest.get(key) != value for key, value in base.items()):
        message = "Canonical input batch belongs to different source evidence."
        raise FileExistsError(message)
    records = {int(record["case_index"]): record for record in manifest["cases"]}
    return records, source


def _compatible_source(
    config: config_service.GenerationConfig,
    base: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    *,
    storage_root: Path | str | None,
) -> admission_service.InputSource | None:
    """Select one exact compatible immutable source when current evidence is absent."""
    discovery = admission_service.discover_input_batches(storage_root=storage_root)
    relevant_issues = [issue for issue in discovery.issues if issue.directory.parent.parent.name == config.batch_storage_name]
    if relevant_issues:
        message = f"Compatible input-source discovery found corrupt evidence for {config.batch_storage_name}: {relevant_issues[0].directory}"
        raise FileExistsError(message)
    expected_indices = tuple(config.case_indices)
    candidates = [
        source
        for source in discovery.sources
        if source.batch_storage_name == config.batch_storage_name
        and source.batch_id == config.batch_id
        and source.resolved_config_matches(resolved_config)
        and all(source.manifest_payload().get(field) == base[field] for field in admission_service.INPUT_GENERATION_COMPATIBILITY_FIELDS)
        and tuple(reference.case_index for reference in source.cases) == expected_indices
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        message = f"Compatible input-source selection is ambiguous for {config.batch_name!r}."
        raise FileExistsError(message)
    return candidates[0]


def _admit_selected_input_references(
    config: config_service.GenerationConfig,
    input_generation_id: str,
    *,
    input_source_git_commit: str,
    storage_root: Path | str | None,
    validation_depth: Literal["evidence", "full"],
) -> dict[int, admission_service.InputCaseReference]:
    """Admit one persisted selected source against active scientific evidence."""
    base, resolved, _metadata, _raw = _configured_input_locations(config, storage_root=storage_root)
    metadata = common.paths.resolve_generation_input_generation_metadata_directory(
        config.batch_storage_name, input_generation_id, storage_root=storage_root
    )
    raw = common.paths.resolve_generation_input_generation_raw_directory(config.batch_storage_name, input_generation_id, storage_root=storage_root)
    try:
        source = admission_service.admit_input_batch_source(
            metadata, raw_directory=raw, expected_input_generation_id=input_generation_id, validation_depth=validation_depth
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        message = f"Selected input source is incomplete or invalid: {metadata}: {error}"
        raise FileExistsError(message) from error
    manifest = source.manifest_payload()
    if manifest.get("git_commit") != source_service.validate_git_commit(input_source_git_commit):
        message = "Selected input source disagrees with its persisted source commit."
        raise RuntimeError(message)
    if (
        not source.resolved_config_matches(resolved)
        or any(manifest.get(field) != base[field] for field in admission_service.INPUT_GENERATION_COMPATIBILITY_FIELDS)
        or tuple(reference.case_index for reference in source.cases) != tuple(config.case_indices)
    ):
        message = "Selected input source disagrees with active scientific configuration or ordered membership."
        raise RuntimeError(message)
    return {reference.case_index: reference for reference in source.cases}


def _configured_input_locations(
    config: config_service.GenerationConfig,
    *,
    storage_root: Path | str | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    """Resolve exact maintained input evidence locations for one batch."""
    resolved_config = _resolved_config(config)
    base = _manifest_base(config, resolved_config)
    input_generation_id = str(base["input_generation_id"])
    metadata_directory = common.paths.resolve_generation_input_generation_metadata_directory(
        config.batch_storage_name,
        input_generation_id,
        storage_root=storage_root,
    )
    raw_directory = common.paths.resolve_generation_input_generation_raw_directory(
        config.batch_storage_name,
        input_generation_id,
        storage_root=storage_root,
    )
    return base, resolved_config, metadata_directory, raw_directory


def admit_configured_input_case(
    config: config_service.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    git_commit: str | None = None,
) -> admission_service.InputCaseReference:
    """Fully admit one configured raw case without rebuilding its batch."""
    commit_context = nullcontext() if git_commit is None else _generation_git_commit(source_service.validate_git_commit(git_commit))
    with commit_context:
        config.case_id(case_index)
        base, _resolved, metadata, raw = _configured_input_locations(
            config,
            storage_root=storage_root,
        )
        return admission_service.admit_input_case_evidence(
            metadata,
            case_index,
            raw_directory=raw,
            expected_input_generation_id=str(base["input_generation_id"]),
            validation_depth="full",
        )


def admit_configured_input_references(
    config: config_service.GenerationConfig,
    *,
    storage_root: Path | str | None = None,
    validation_depth: Literal["evidence", "full"] = "evidence",
    git_commit: str | None = None,
    input_generation_id: str | None = None,
    input_source_git_commit: str | None = None,
) -> dict[int, admission_service.InputCaseReference]:
    """Admit one configured batch once and index its exact case references."""
    commit_context = nullcontext() if git_commit is None else _generation_git_commit(source_service.validate_git_commit(git_commit))
    with commit_context:
        if (input_generation_id is None) != (input_source_git_commit is None):
            message = "Selected input generation and source commit must be supplied together."
            raise ValueError(message)
        if input_generation_id is not None and input_source_git_commit is not None:
            return _admit_selected_input_references(
                config,
                input_generation_id,
                input_source_git_commit=input_source_git_commit,
                storage_root=storage_root,
                validation_depth=validation_depth,
            )
        base, resolved, metadata, raw = _configured_input_locations(config, storage_root=storage_root)
        records, source = _load_existing(metadata, raw, base=base, resolved_config=resolved, validation_depth=validation_depth)
    expected = set(config.case_indices)
    if not expected.issubset(records):
        missing = tuple(sorted(expected.difference(records)))
        message = f"Canonical raw input manifest is missing configured cases: {missing}."
        raise FileNotFoundError(message)
    references = {reference.case_index: reference for reference in source.cases if reference.case_index in expected}
    if set(references) != expected:
        message = "Canonical raw input references do not cover configured membership exactly."
        raise RuntimeError(message)
    return references


def _estimate_case_bytes(config: config_service.GenerationConfig, case_index: int) -> int:
    """Estimate one missing case from an exact temporary canonical generation."""
    with tempfile.TemporaryDirectory(prefix="generation-input-plan-") as temporary:
        directory = Path(temporary) / config.case_id(case_index)
        case_service.generate_case_input_bundle(config, case_index, directory)
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def plan_input_cases(
    config: config_service.GenerationConfig,
    case_count: int,
    *,
    case_start: int | None = None,
    storage_root: Path | str | None = None,
) -> InputGenerationPlan:
    """Plan reusable and missing raw cases without mutating persistent storage."""
    requested = select_case_indices(config, case_count, case_start)
    resolved_config = _resolved_config(config)
    base = _manifest_base(config, resolved_config)
    input_generation_id = str(base["input_generation_id"])
    metadata_directory = common.paths.resolve_generation_input_generation_metadata_directory(
        config.batch_storage_name, input_generation_id, storage_root=storage_root
    )
    raw_directory = common.paths.resolve_generation_input_generation_raw_directory(
        config.batch_storage_name, input_generation_id, storage_root=storage_root
    )
    manifest_exists = (metadata_directory / "input_generation_manifest.json").is_file()
    raw_exists = raw_directory.is_dir()
    records: dict[int, Mapping[str, Any]] = {}
    if manifest_exists != raw_exists:
        message = "Canonical input manifest and raw batch must exist together."
        raise FileExistsError(message)
    if manifest_exists:
        records, _source = _load_existing(
            metadata_directory,
            raw_directory,
            base=base,
            resolved_config=resolved_config,
        )
    elif raw_directory.exists():
        message = f"Unmanifested canonical raw batch blocks input planning: {raw_directory}"
        raise FileExistsError(message)
    elif (metadata_directory / "resolved_generation_config.json").exists():
        _immutable_json(
            metadata_directory / "resolved_generation_config.json",
            resolved_config,
            label="resolved scientific generation configuration",
        )
    reusable = tuple(index for index in requested if index in records)
    missing = tuple(index for index in requested if index not in records)
    reusable_bytes = sum(
        sum(path.stat().st_size for path in (raw_directory / config.case_id(index)).rglob("*") if path.is_file()) for index in reusable
    )
    estimated_missing_bytes = 0
    if missing:
        estimated_missing_bytes = _estimate_case_bytes(config, missing[0]) * len(missing)
    return InputGenerationPlan(
        batch_id=config.batch_id,
        batch_storage_name=config.batch_storage_name,
        batch_name=config.batch_name,
        raw_directory=raw_directory,
        metadata_directory=metadata_directory,
        requested_case_indices=requested,
        reusable_case_indices=reusable,
        missing_case_indices=missing,
        estimated_storage_bytes=reusable_bytes + estimated_missing_bytes,
    )


def _write_staged_metadata(
    directory: Path,
    *,
    resolved_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Write one complete metadata candidate for fail-closed admission."""
    common.serialization.atomic_write_json(directory / "resolved_generation_config.json", resolved_config)
    common.serialization.atomic_write_json(directory / "input_generation_manifest.json", manifest)


def _require_case_publication_target(
    target: Path,
    *,
    declared: bool,
) -> None:
    """Require one manifest-declared target to exist, or a new target to be absent."""
    if declared and not target.is_dir():
        message = f"Input manifest declares a missing canonical case: {target}"
        raise FileExistsError(message)
    if not declared and target.exists():
        message = f"Unmanifested canonical input case blocks publication: {target}"
        raise FileExistsError(message)


def _recover_input_transaction(
    transaction: Path,
    *,
    metadata_directory: Path,
    raw_directory: Path,
    batch_id: str,
    batch_storage_name: str,
    base: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> None:
    """Complete a journaled case publication interrupted before manifest commit."""
    journal_path = transaction / "transaction.json"
    if not journal_path.is_file():
        # The journal is written before any final path changes, so this is only
        # abandoned pre-publication staging.
        shutil.rmtree(transaction)
        return
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    staged_metadata = transaction / "meta" / batch_storage_name / "input_generations" / str(base["input_generation_id"])
    staged_raw = transaction / "raw" / batch_storage_name / "input_generations" / str(base["input_generation_id"])
    manifest_path = staged_metadata / "input_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_case_ids = journal.get("new_case_ids") if isinstance(journal, dict) else None
    if (
        not isinstance(journal, dict)
        or set(journal)
        != {
            "schema_kind",
            "schema_version",
            "batch_id",
            "batch_storage_name",
            "input_generation_id",
            "manifest_sha256",
            "new_case_ids",
        }
        or journal.get("schema_kind") != "generation_input_transaction"
        or journal.get("schema_version") != INPUT_TRANSACTION_SCHEMA_VERSION
        or journal.get("batch_id") != batch_id
        or journal.get("batch_storage_name") != batch_storage_name
        or journal.get("input_generation_id") != base["input_generation_id"]
        or journal.get("manifest_sha256") != common.serialization.file_sha256(manifest_path)
        or not isinstance(new_case_ids, list)
        or any(not isinstance(case_id, str) for case_id in new_case_ids)
        or not isinstance(manifest, dict)
        or any(manifest.get(key) != value for key, value in base.items())
        or json.loads((staged_metadata / "resolved_generation_config.json").read_text(encoding="utf-8")) != resolved_config
    ):
        message = f"Input-generation transaction evidence is invalid: {transaction}"
        raise RuntimeError(message)
    raw_directory.mkdir(parents=True, exist_ok=True)
    for case_id in new_case_ids:
        target = raw_directory / case_id
        staged = staged_raw / case_id
        if target.exists() and staged.exists():
            message = f"Input transaction has conflicting staged and final cases: {case_id}"
            raise RuntimeError(message)
        if not target.exists():
            if not staged.is_dir():
                message = f"Input transaction lost unpublished case evidence: {case_id}"
                raise RuntimeError(message)
            staged.replace(target)
    admission_service.admit_input_batch_source(
        staged_metadata,
        raw_directory=raw_directory,
        expected_input_generation_id=str(base["input_generation_id"]),
    )
    common.serialization.atomic_write_json(
        metadata_directory / "input_generation_manifest.json",
        manifest,
    )
    shutil.rmtree(transaction, ignore_errors=True)


def _write_input_transaction_journal(
    transaction: Path,
    *,
    batch_id: str,
    batch_storage_name: str,
    input_generation_id: str,
    manifest_path: Path,
    new_case_ids: tuple[str, ...],
) -> None:
    """Commit recovery evidence before any canonical raw path changes."""
    common.serialization.atomic_write_json(
        transaction / "transaction.json",
        {
            "schema_kind": "generation_input_transaction",
            "schema_version": INPUT_TRANSACTION_SCHEMA_VERSION,
            "batch_id": batch_id,
            "batch_storage_name": batch_storage_name,
            "input_generation_id": input_generation_id,
            "manifest_sha256": common.serialization.file_sha256(manifest_path),
            "new_case_ids": list(new_case_ids),
        },
    )


def _emit_case_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    *,
    operation: str,
    cases_completed: int,
    cases_total: int,
    generated_cases: int,
    reused_cases: int,
) -> None:
    """Emit at most about twenty factual checkpoints for one input batch."""
    if callback is None:
        return
    interval = max(1, (cases_total + 19) // 20)
    if cases_completed not in {0, cases_total} and cases_completed % interval != 0:
        return
    callback(
        {
            "operation": operation,
            "cases_completed": cases_completed,
            "cases_total": cases_total,
            "generated_cases": generated_cases,
            "reused_cases": reused_cases,
            "eta": "unavailable",
        }
    )


def generate_input_cases(
    config: config_service.GenerationConfig,
    case_count: int,
    *,
    case_start: int | None = None,
    storage_root: Path | str | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    existing_validation_depth: Literal["evidence", "full"] = "full",
) -> GeneratedInputBatch:
    """Generate or exactly reuse bounded cases in one canonical raw batch."""
    requested = select_case_indices(config, case_count, case_start)
    resolved_config = _resolved_config(config)
    base = _manifest_base(config, resolved_config)
    input_generation_id = str(base["input_generation_id"])
    metadata_directory = common.paths.resolve_generation_input_generation_metadata_directory(
        config.batch_storage_name, input_generation_id, storage_root=storage_root
    )
    raw_directory = common.paths.resolve_generation_input_generation_raw_directory(
        config.batch_storage_name, input_generation_id, storage_root=storage_root
    )
    lock_path = common.paths.resolve_generation_input_lock_path(config.batch_storage_name, storage_root=storage_root)
    transaction = common.paths.resolve_generation_input_transaction_directory(input_generation_id, storage_root=storage_root)
    with common.locking.exclusive_file_lock(lock_path, blocking=True):
        _initialize_input_generation_metadata(
            config,
            resolved_config,
            input_generation_id,
            storage_root=storage_root,
        )
        if transaction.exists():
            _recover_input_transaction(
                transaction,
                metadata_directory=metadata_directory,
                raw_directory=raw_directory,
                batch_id=config.batch_id,
                batch_storage_name=config.batch_storage_name,
                base=base,
                resolved_config=resolved_config,
            )
        manifest_exists = (metadata_directory / "input_generation_manifest.json").is_file()
        raw_exists = raw_directory.is_dir()
        if manifest_exists != raw_exists:
            message = "Canonical input manifest and raw batch must exist together; found incomplete evidence."
            raise FileExistsError(message)
        records: dict[int, Mapping[str, Any]] = {}
        if manifest_exists:
            _emit_case_progress(
                progress,
                operation="canonical_input_validation",
                cases_completed=0,
                cases_total=len(requested),
                generated_cases=0,
                reused_cases=0,
            )
            records, _source = _load_existing(
                metadata_directory,
                raw_directory,
                base=base,
                resolved_config=resolved_config,
                validation_depth=existing_validation_depth,
            )
        elif raw_directory.exists():
            message = f"Unmanifested canonical raw batch blocks publication: {raw_directory}"
            raise FileExistsError(message)
        generated_indices = [index for index in requested if index not in records]
        reused_count = len(requested) - len(generated_indices)
        _emit_case_progress(
            progress,
            operation="canonical_input_generation",
            cases_completed=reused_count,
            cases_total=len(requested),
            generated_cases=0,
            reused_cases=reused_count,
        )
        if generated_indices:
            transaction.parent.mkdir(parents=True, exist_ok=True)
            transaction.mkdir()
            staged_raw = transaction / "raw" / config.batch_storage_name / "input_generations" / input_generation_id
            staged_metadata = transaction / "meta" / config.batch_storage_name / "input_generations" / input_generation_id
            try:
                for generated_count, case_index in enumerate(
                    generated_indices,
                    start=1,
                ):
                    case_id = config.case_id(case_index)
                    target_case = raw_directory / case_id
                    _require_case_publication_target(
                        target_case,
                        declared=False,
                    )
                    staged_case = staged_raw / case_id
                    case_service.generate_case_input_bundle(
                        config,
                        case_index,
                        staged_case,
                    )
                    records[case_index] = _case_record(staged_case)
                    _emit_case_progress(
                        progress,
                        operation="canonical_input_generation",
                        cases_completed=reused_count + generated_count,
                        cases_total=len(requested),
                        generated_cases=generated_count,
                        reused_cases=reused_count,
                    )
                manifest = _complete_manifest(base, records)
                _write_staged_metadata(
                    staged_metadata,
                    resolved_config=resolved_config,
                    manifest=manifest,
                )
                _write_input_transaction_journal(
                    transaction,
                    batch_id=config.batch_id,
                    batch_storage_name=config.batch_storage_name,
                    input_generation_id=input_generation_id,
                    manifest_path=staged_metadata / "input_generation_manifest.json",
                    new_case_ids=tuple(config.case_id(index) for index in generated_indices),
                )
                raw_directory.mkdir(parents=True, exist_ok=True)
                for case_index in generated_indices:
                    target_case = raw_directory / config.case_id(case_index)
                    (staged_raw / config.case_id(case_index)).replace(target_case)
                admission_service.admit_input_batch_source(
                    staged_metadata,
                    raw_directory=raw_directory,
                    expected_input_generation_id=input_generation_id,
                    validation_depth="evidence",
                )
                common.serialization.atomic_write_json(
                    metadata_directory / "input_generation_manifest.json",
                    manifest,
                )
            except BaseException:
                if not (transaction / "transaction.json").is_file():
                    shutil.rmtree(transaction)
                raise
            else:
                shutil.rmtree(transaction, ignore_errors=True)
    return GeneratedInputBatch(
        input_generation_id=input_generation_id,
        metadata_directory=metadata_directory,
        raw_directory=raw_directory,
        requested_case_indices=requested,
        case_indices=tuple(sorted(records)),
        generated_case_count=len(generated_indices),
        reused_case_count=reused_count,
    )


@contextmanager
def _generation_git_commit(git_commit: str) -> Iterator[None]:
    """Bind one exact source commit for generation and restore prior state."""
    previous = os.environ.get(source_service.GIT_COMMIT_ENVIRONMENT_VARIABLE)
    os.environ[source_service.GIT_COMMIT_ENVIRONMENT_VARIABLE] = git_commit
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(source_service.GIT_COMMIT_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[source_service.GIT_COMMIT_ENVIRONMENT_VARIABLE] = previous


def _resolve_campaign_request(
    request: CampaignInputGenerationRequest,
) -> tuple[config_service.CampaignConfig, tuple[tuple[config_service.GenerationConfig, tuple[int, ...]], ...]]:
    """Validate one request and resolve exact campaign batch membership."""
    if not isinstance(request.all_batches, bool) or not isinstance(request.all_cases, bool):
        message = "all_batches and all_cases must be booleans."
        raise TypeError(message)
    if request.all_batches == (request.only_batch is not None):
        message = "Select exactly one of only_batch or all_batches."
        raise ValueError(message)
    if request.all_cases == (request.case_count is not None):
        message = "Select exactly one of case_count or all_cases."
        raise ValueError(message)
    if request.all_cases and request.case_start is not None:
        message = "case_start cannot be combined with all_cases."
        raise ValueError(message)
    campaign = config_service.load_campaign_config(request.campaign_config)
    if request.all_batches:
        batches = campaign.batches
    else:
        only_batch = request.only_batch
        if only_batch is None:
            message = "A single-batch input request requires only_batch."
            raise RuntimeError(message)
        batches = (campaign.batch(only_batch),)
    if request.only_regime is not None:
        batches = tuple(batch for batch in batches if batch.sampling_regime == request.only_regime)
    if not batches:
        message = "Input-generation selection resolved no canonical batches."
        raise ValueError(message)
    selections = []
    for batch in batches:
        case_count = len(batch.case_indices) if request.all_cases else request.case_count
        if case_count is None:
            message = "Input generation requires case_count or all_cases."
            raise ValueError(message)
        selections.append(
            (
                batch,
                select_case_indices(
                    batch,
                    case_count,
                    request.case_start,
                ),
            )
        )
    return campaign, tuple(selections)


def _resolve_request_git_commit(
    request: CampaignInputGenerationRequest,
) -> tuple[str | None, str, str | None]:
    """Resolve explicit launcher evidence or clean interactive source identity."""
    if request.git_commit is not None:
        return source_service.validate_git_commit(request.git_commit), "provided", None
    try:
        return source_service.clean_repository_git_commit(), "clean_repository", None
    except RuntimeError as error:
        if request.action == "dry_run":
            return None, "blocked", str(error)
        message = "Input generation cannot execute without truthful source identity."
        raise RuntimeError(message) from error


def _equivalent_cli_command(
    request: CampaignInputGenerationRequest,
    campaign: config_service.CampaignConfig,
    git_commit: str,
) -> str:
    """Return the equivalent copyable canonical CLI command."""
    repository = common.paths.get_project_root().resolve()
    campaign_path = campaign.source_path.resolve()
    try:
        campaign_argument = campaign_path.relative_to(repository).as_posix()
    except ValueError:
        campaign_argument = str(campaign_path)
    command = [
        "./scripts/docker_python.sh",
        "-m",
        "src.generation.cli.cli_generation",
        "generate-input-cases",
        campaign_argument,
    ]
    if request.all_batches:
        command.append("--all-batches")
    else:
        command.extend(("--only-batch", str(request.only_batch)))
    if request.only_regime is not None:
        command.extend(("--only-regime", request.only_regime))
    if request.all_cases:
        command.append("--all-cases")
    else:
        if request.case_start is not None:
            command.extend(("--case-start", str(request.case_start)))
        command.extend(("--case-count", str(request.case_count)))
    if request.action == "dry_run":
        command.append("--dry-run")
    command.extend(("--git-commit", git_commit, "--storage-root", str(Path(request.storage_root).expanduser())))
    return shlex.join(command)


def prepare_campaign_inputs(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    existing_validation_depth: Literal["evidence", "full"] = "evidence",
) -> dict[str, Any]:
    """
    Materialize or admit every selected campaign case before submission.

    Parameters
    ----------
    campaign : CampaignConfig
        Exact campaign selection whose configured case membership is required.
    git_commit : str
        Lowercase source commit used in every input-generation identity.
    storage_root : Path | str | None, optional
        Canonical storage root, or the configured default when omitted.
    progress : Callable[[Mapping[str, Any]], None] | None, optional
        Receives bounded progress snapshots from the active generation loop.
    existing_validation_depth : {"evidence", "full"}, optional
        Integrity depth for already published immutable inputs.

    Returns
    -------
    dict[str, Any]
        Per-batch identities and aggregate generated and reused case counts.

    Raises
    ------
    ValueError
        If the source commit or selected case membership is invalid.
    FileExistsError
        If current identity evidence exists but is incomplete or invalid.

    """
    commit = source_service.validate_git_commit(git_commit)
    storage = Path(storage_root).expanduser() if storage_root is not None else None
    batches: list[dict[str, Any]] = []
    total_cases = sum(len(batch.case_indices) for batch in campaign.batches)
    completed_offset = 0
    with _generation_git_commit(commit):
        for batch_index, batch in enumerate(campaign.batches):

            def report_batch_progress(
                event: Mapping[str, Any],
                *,
                offset: int = completed_offset,
                ordinal: int = batch_index,
            ) -> None:
                if progress is None:
                    return
                progress(
                    {
                        **dict(event),
                        "cases_completed": (offset + int(event["cases_completed"])),
                        "cases_total": total_cases,
                        "batches_completed": ordinal,
                        "batches_total": len(campaign.batches),
                    }
                )

            base, resolved, metadata, raw = _configured_input_locations(batch, storage_root=storage)
            current: admission_service.InputSource | None
            try:
                _records, existing_source = _load_existing(
                    metadata,
                    raw,
                    base=base,
                    resolved_config=resolved,
                    validation_depth=existing_validation_depth,
                )
            except FileExistsError:
                if metadata.exists() or raw.exists():
                    raise
                current = _compatible_source(batch, base, resolved, storage_root=storage)
            else:
                current = existing_source
            if current is None:
                generated = generate_input_cases(
                    batch,
                    len(batch.case_indices),
                    case_start=batch.case_indices[0],
                    storage_root=storage,
                    progress=report_batch_progress,
                    existing_validation_depth=existing_validation_depth,
                )
                source_id = generated.input_generation_id
                generated_count, reused_count = generated.generated_case_count, generated.reused_case_count
                source_commit = commit
            else:
                source_id = current.source_id
                generated_count, reused_count = 0, len(batch.case_indices)
                source_commit = str(current.manifest_payload()["git_commit"])
            completed_offset += len(batch.case_indices)
            batches.append(
                {
                    "batch_id": batch.batch_id,
                    "batch_storage_name": batch.batch_storage_name,
                    "input_generation_id": source_id,
                    "source_git_commit": source_commit,
                    "case_indices": list(batch.case_indices),
                    "generated_case_count": generated_count,
                    "reused_case_count": reused_count,
                    "requested_case_count": len(batch.case_indices),
                }
            )
    return {
        "git_commit": commit,
        "selected_batch_count": len(batches),
        "generated_case_count": sum(item["generated_case_count"] for item in batches),
        "reused_case_count": sum(item["reused_case_count"] for item in batches),
        "batches": batches,
    }


def admit_campaign_inputs_with_references(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None = None,
    validation_depth: Literal["evidence", "full"] = "evidence",
) -> tuple[dict[str, Any], dict[str, dict[int, admission_service.InputCaseReference]]]:
    """Admit campaign inputs once and return provenance with indexed references."""
    commit = source_service.validate_git_commit(git_commit)
    storage = Path(storage_root).expanduser() if storage_root is not None else None
    batches: list[dict[str, Any]] = []
    references: dict[str, dict[int, admission_service.InputCaseReference]] = {}
    with _generation_git_commit(commit):
        for batch in campaign.batches:
            base, resolved, metadata, raw = _configured_input_locations(batch, storage_root=storage)
            try:
                records, source = _load_existing(
                    metadata,
                    raw,
                    base=base,
                    resolved_config=resolved,
                    validation_depth=validation_depth,
                )
                record_indices = set(records)
            except FileExistsError as error:
                if metadata.exists() or raw.exists():
                    raise
                compatible_source = _compatible_source(batch, base, resolved, storage_root=storage)
                if compatible_source is None:
                    message = f"Prepared canonical input batch {batch.batch_name!r} is unavailable."
                    raise FileNotFoundError(message) from error
                source = compatible_source
                record_indices = {reference.case_index for reference in source.cases}
            expected_indices = set(batch.case_indices)
            missing = tuple(case_index for case_index in batch.case_indices if case_index not in record_indices)
            if missing:
                message = f"Prepared canonical input batch {batch.batch_name!r} is missing configured cases: {missing}."
                raise FileNotFoundError(message)
            batch_references = {reference.case_index: reference for reference in source.cases if reference.case_index in expected_indices}
            if set(batch_references) != expected_indices:
                message = f"Prepared canonical input batch {batch.batch_name!r} has incomplete admitted references."
                raise RuntimeError(message)
            references[batch.batch_name] = batch_references
            source_manifest = source.manifest_payload()
            batches.append(
                {
                    "batch_id": batch.batch_id,
                    "batch_storage_name": batch.batch_storage_name,
                    "input_generation_id": source.source_id,
                    "source_git_commit": source_manifest["git_commit"],
                    "case_indices": list(batch.case_indices),
                    "admitted_case_count": len(batch.case_indices),
                }
            )
    evidence = {
        "git_commit": commit,
        "validation_depth": validation_depth,
        "selected_batch_count": len(batches),
        "admitted_case_count": sum(item["admitted_case_count"] for item in batches),
        "batches": batches,
    }
    return evidence, references


def admit_campaign_inputs(
    campaign: config_service.CampaignConfig,
    *,
    git_commit: str,
    storage_root: Path | str | None = None,
    validation_depth: Literal["evidence", "full"] = "evidence",
) -> dict[str, Any]:
    """Admit every configured input from immutable batch evidence without generation."""
    evidence, _references = admit_campaign_inputs_with_references(
        campaign,
        git_commit=git_commit,
        storage_root=storage_root,
        validation_depth=validation_depth,
    )
    return evidence


def run_campaign_input_generation(
    request: CampaignInputGenerationRequest,
) -> dict[str, Any]:
    """Plan, generate, or reuse canonical inputs for one campaign selection."""
    if not isinstance(request, CampaignInputGenerationRequest):
        message = "Campaign input generation requires CampaignInputGenerationRequest."
        raise TypeError(message)
    if request.action not in INPUT_GENERATION_ACTIONS:
        message = f"Unsupported input-generation action {request.action!r}; expected one of {INPUT_GENERATION_ACTIONS}."
        raise ValueError(message)
    campaign, selections = _resolve_campaign_request(request)
    git_commit, source_identity_status, source_identity_blocker = _resolve_request_git_commit(request)
    storage_root = Path(request.storage_root).expanduser()
    response: dict[str, Any] = {
        "action": request.action,
        "campaign_config": str(campaign.source_path),
        "campaign_name": campaign.campaign_name,
        "campaign_id": campaign.campaign_id,
        "campaign_purpose": campaign.campaign_purpose,
        "simulation_profile": campaign.profile.id,
        "input_only": True,
        "execution_status": "not_executed",
        "dry_run": request.action == "dry_run",
        "source_identity_status": source_identity_status,
        "git_commit": git_commit,
        "selected_batch_count": len(selections),
        "requested_case_count": sum(len(indices) for _batch, indices in selections),
    }
    if source_identity_blocker is not None:
        response.update(
            {
                "source_identity_blocker": source_identity_blocker,
                "estimated_storage_bytes": None,
                "generated_case_count": 0,
                "reused_case_count": 0,
                "batches": [
                    {
                        "batch_name": batch.batch_name,
                        "batch_id": batch.batch_id,
                        "batch_storage_name": batch.batch_storage_name,
                        "batch_identity": batch.batch_identity,
                        "simulation_profile": batch.profile.id,
                        "campaign_purpose": str(batch.scientific_values["campaign_purpose"]),
                        "requested_case_indices": list(indices),
                        "requested_case_range": [indices[0], indices[-1]],
                        "raw_batch_root": str(
                            common.paths.resolve_generation_raw_batch_directory(
                                batch.batch_storage_name,
                                storage_root=storage_root,
                            )
                        ),
                    }
                    for batch, indices in selections
                ],
                "equivalent_cli_command": None,
            }
        )
        if len(selections) == 1:
            response.update(response["batches"][0])
        return response
    if git_commit is None:
        message = "Resolved input-generation source identity is unexpectedly absent."
        raise RuntimeError(message)
    batch_results: list[dict[str, Any]] = []
    with _generation_git_commit(git_commit):
        for batch, requested in selections:
            input_plan = plan_input_cases(
                batch,
                len(requested),
                case_start=requested[0],
                storage_root=storage_root,
            )
            generated_case_count = 0
            reused_case_count = len(input_plan.reusable_case_indices)
            complete_case_indices = input_plan.reusable_case_indices
            input_generation_id: str | None = None
            if request.action == "execute":
                generated = generate_input_cases(
                    batch,
                    len(requested),
                    case_start=requested[0],
                    storage_root=storage_root,
                )
                generated_case_count = generated.generated_case_count
                reused_case_count = generated.reused_case_count
                complete_case_indices = generated.case_indices
                input_generation_id = generated.input_generation_id
            batch_results.append(
                {
                    "batch_name": batch.batch_name,
                    "batch_id": batch.batch_id,
                    "batch_storage_name": batch.batch_storage_name,
                    "batch_identity": batch.batch_identity,
                    "simulation_profile": batch.profile.id,
                    "campaign_purpose": str(batch.scientific_values["campaign_purpose"]),
                    "material_family": batch.material_family,
                    "sampling_regime": batch.sampling_regime,
                    "requested_case_indices": list(requested),
                    "requested_case_range": [requested[0], requested[-1]],
                    "requested_case_count": len(requested),
                    "reusable_case_indices": list(input_plan.reusable_case_indices),
                    "reused_case_count": reused_case_count,
                    "missing_case_indices": list(input_plan.missing_case_indices),
                    "generated_case_count": generated_case_count,
                    "would_generate_case_count": len(input_plan.missing_case_indices),
                    "complete_manifest_case_indices": list(complete_case_indices),
                    "estimated_storage_bytes": input_plan.estimated_storage_bytes,
                    "raw_batch_root": str(input_plan.raw_directory),
                    "raw_case_paths": [str(input_plan.raw_directory / batch.case_id(index)) for index in requested],
                    "manifest": str(input_plan.metadata_directory / "input_generation_manifest.json"),
                    "input_generation_id": input_generation_id,
                }
            )
    response.update(
        {
            "estimated_storage_bytes": sum(int(item["estimated_storage_bytes"]) for item in batch_results),
            "generated_case_count": sum(int(item["generated_case_count"]) for item in batch_results),
            "reused_case_count": sum(int(item["reused_case_count"]) for item in batch_results),
            "batches": batch_results,
            "equivalent_cli_command": _equivalent_cli_command(request, campaign, git_commit),
        }
    )
    if len(batch_results) == 1:
        response.update(batch_results[0])
    return response
