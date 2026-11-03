"""Persist one campaign's immutable plan and append-only material candidates.

Responsibilities:
  - Bind a task/name to one fully resolved scientific plan and its name-independent seed key
  - Reserve material-local case IDs durably before case execution
  - Record distinct operational attempts and terminal candidate decisions
  - Require a solver attempt's reason to be the runtime's typed stop reason (stalled, timed_out, solver_error) or unclassified
  - Derive deterministic selected-success membership from canonical evidence
  - Pre-register the authored replacement capacity at creation and append later explicit amendments with their reason
  - Resolve the execution batch that carries each reserved candidate's scheduler and input identity

Design principles:
  - Candidate IDs are never recycled after a failure
  - Immutable records live in their scientific case's _record/ folder, outside controller state
  - A reservation survives interruption and is reconciled before new allocation

This module does NOT:
  - Sample fields, submit Slurm jobs, convert COMSOL output, or package Datasets
  - Rewrite historical campaign runs or delete unsuccessful case evidence
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.cases import generation_cases_seeding as seeding
from src.generation.cases import generation_cases_selection as selection_service
from src.generation.contracts import generation_contracts_campaign_identity as identity
from src.generation.contracts import generation_contracts_material_registry as material_registry

_SHA256: Final = re.compile(r"[0-9a-f]{64}\Z")
_ORIGINS: Final = frozenset({"primary", "replacement"})
_ATTEMPT_CLASSES: Final = frozenset({"solver", "validation", "submission", "license", "filesystem", "publication", "controller"})
_ATTEMPT_SCHEMA: Final = ("generation_candidate_attempt", 2)
# A solver attempt's reason is the runtime's own classification of the stop (its RuntimeFailureReason) or ``unclassified``.
RUNTIME_SOLVER_REASONS: Final = frozenset({"stalled", "timed_out", "solver_error"})
UNCLASSIFIED_SOLVER_REASON: Final = "unclassified"
_CAMPAIGN_PLAN_SCHEMA_VERSION: Final = 1
_AMENDMENT_SCHEMA_VERSION: Final = 2
_AMENDMENT_SOURCES: Final = frozenset({"plan", "explicit"})
_EPOCH_OWNED_FIELDS: Final = frozenset(
    {"schema_kind", "schema_version", "campaign_run_id", "campaign_digest", "epoch_index", "first_submission_index", "previous_sha256"}
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.generation.cases import generation_cases_config as config_service


def _read_json(path: Path) -> dict[str, Any]:
    """Read one required immutable object without following a final symlink."""
    if path.is_symlink() or not path.is_file():
        msg = f"Required campaign evidence is missing or unsafe: {path}."
        raise FileNotFoundError(msg)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"Campaign evidence must contain one JSON object: {path}."
        raise ValueError(msg)  # noqa: TRY004 -- invalid persisted JSON evidence is reported by the established ValueError contract
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one immutable JSON record and reject conflicting prior bytes."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != encoded:
            msg = f"Immutable campaign evidence conflicts at {path}."
            raise FileExistsError(msg)
        return
    common.serialization.atomic_write_bytes(path, encoded)


def _require_attempt_contract(attempt: Mapping[str, Any], *, label: object) -> None:
    """Reject an attempt record of another schema or a solver attempt without a typed runtime reason."""
    if (attempt.get("schema_kind"), attempt.get("schema_version")) != _ATTEMPT_SCHEMA or (
        attempt.get("class") == "solver" and attempt.get("reason_code") not in {*RUNTIME_SOLVER_REASONS, UNCLASSIFIED_SOLVER_REASON}
    ):
        msg = f"Candidate attempt is not a schema-{_ATTEMPT_SCHEMA[1]} record with a typed solver reason: {label}."
        raise ValueError(msg)


def _root(task: str, campaign_name: str, storage_root: Path | str | None) -> Path:
    """Resolve and contain one campaign root without aliasing an existing symlink, and admit its ``_record/`` before any use."""
    root = common.paths.resolve_generation_campaign_directory(task, campaign_name, storage_root=storage_root)
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    if not root.resolve(strict=False).is_relative_to(storage) or any(
        path.is_symlink() for path in (root, *root.parents) if path.is_relative_to(storage)
    ):
        msg = f"Campaign path is outside its storage root or is a symlink: {root}."
        raise ValueError(msg)
    common.paths.admit_generation_campaign_record_directory(root)
    return root


def _controller_lock(root: Path) -> Path:
    """Return the lock anchor that serializes writers of one campaign's plan and candidates."""
    return common.paths.resolve_generation_controller_lock_path(root)


def _epochs_directory(root: Path) -> Path:
    """Return the execution-epoch directory of one campaign."""
    return common.paths.resolve_generation_epochs_directory(common.paths.resolve_generation_execution_directory(root))


def _source_records(campaign: config_service.CampaignConfig) -> list[dict[str, str]]:
    """Bind exact submitted YAML bytes separately from resolved science."""
    root = common.paths.get_project_root().resolve()
    records: list[dict[str, str]] = []
    for authored_path in campaign.source_files:
        if authored_path.is_symlink():
            msg = f"Campaign source is a symlink: {authored_path}."
            raise ValueError(msg)
        source_path = authored_path.resolve(strict=True)
        if not source_path.is_relative_to(root):
            msg = f"Campaign source is outside the repository: {source_path}."
            raise ValueError(msg)
        source_bytes = source_path.read_bytes()
        records.append(
            {
                "relative_path": source_path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "content": source_bytes.decode("utf-8"),
            }
        )
    return records


def _plan_seed_key(plan: Mapping[str, Any]) -> str:
    """Recompute the scientific seed key from the plan fields it is derived from."""
    return seeding.campaign_seed_key(
        task=str(plan["task"]),
        design_sizes=plan["original_design_sizes"],
        domain_profile=str(plan["domain_profile"]),
        sampling_method=str(plan["sampling_mode"]),
        seed_base=int(plan["root_seed"]),
        purpose=str(plan["campaign_purpose"]),
    )


def _require_unique_execution_keys(batches: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed unless every frozen batch owns a distinct ``<material>-<regime>`` execution key."""
    keys = [common.paths.generation_execution_key(str(batch["batch_storage_name"])) for batch in batches]
    if len(keys) != len(set(keys)):
        msg = "Campaign batches must have unique material-regime execution keys."
        raise ValueError(msg)


def _resolved_plan(campaign: config_service.CampaignConfig) -> dict[str, Any]:
    """Capture every case-generating value required for independent inspection."""
    plan: dict[str, Any] = {
        "schema_kind": "generation_source_campaign_plan",
        "schema_version": 1,
        "task": campaign.profile.id,
        "campaign_name": campaign.campaign_name,
        "source_name": campaign.source_name,
        "source_label": campaign.source_label,
        "authored_value": None if campaign.authored_value is None else copy.deepcopy(dict(campaign.authored_value)),
        "campaign_digest": campaign.campaign_digest,
        "campaign_purpose": campaign.campaign_purpose,
        "domain_profile": campaign.domain_profile,
        "campaign_id": campaign.campaign_id,
        "root_seed": campaign.root_seed,
        "seed_derivation_version": 1,
        "seed_key": "",
        "sampling_mode": campaign.sampling_mode,
        "selected_targets": dict(campaign.selected_targets),
        "original_design_sizes": dict(campaign.selected_targets),
        "materials": list(campaign.material_inventory),
        "source_registry": copy.deepcopy(campaign.source_registry),
        "scientific_overrides": dict(campaign.scientific_overrides or {}),
        "producer_identity": dict(campaign.producer_identity or {}),
        "execution_overrides": dict(campaign.execution_overrides or {}),
        "total_case_count": campaign.total_case_count,
        "duplicate_case_input_policy": campaign.duplicate_case_input_policy,
        "batches": [
            {
                "batch_name": batch.batch_name,
                "batch_id": batch.batch_id,
                "scientific_config_digest": batch.scientific_config_digest,
                "case_input_config_digest": batch.case_input_config_digest,
                "material_family": batch.material_family,
                "sampling_regime": batch.sampling_regime,
                "case_indices": list(batch.case_indices),
                "seed_base": batch.seed_base,
                "batch_identity": batch.batch_identity,
                "batch_storage_name": batch.batch_storage_name,
                "assignments": {str(index): assignment for index, assignment in batch.assignments.items()},
                "scientific_values": batch.scientific_values,
                "execution_values": batch.execution_values,
            }
            for batch in campaign.batches
        ],
        "execution_values": copy.deepcopy(campaign.execution_values),
        "source_files": _source_records(campaign),
        "template": {"relative_path": campaign.template_relative_path, "sha256": campaign.template_sha256},
    }
    plan["seed_key"] = _plan_seed_key(plan)
    _require_unique_execution_keys(plan["batches"])
    return plan


def create_campaign(
    campaign: config_service.CampaignConfig,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Create or reattach to a same-name, same-science campaign.

    The authored ``replacement_capacity`` is pre-registered only when the campaign
    is created, once per original batch; reattaching must repeat it exactly.
    """
    if campaign.schema_version != _CAMPAIGN_PLAN_SCHEMA_VERSION:
        msg = "The campaign-local lifecycle requires a canonical schema-v2 configuration."
        raise ValueError(msg)
    root = _root(campaign.profile.id, campaign.source_name, storage_root)
    plan_path = common.paths.resolve_generation_campaign_plan_path(root)
    expected = _resolved_plan(campaign)
    with common.locking.exclusive_file_lock(_controller_lock(root), blocking=True):
        if plan_path.exists():
            observed = _read_json(plan_path)
            identity.assert_campaign_semantics(campaign.source_name, campaign.campaign_digest, str(observed.get("campaign_digest")))
            if (
                observed.get("task") != campaign.profile.id
                or observed.get("selected_targets") != expected["selected_targets"]
                or observed.get("seed_key") != expected["seed_key"]
            ):
                msg = f"Campaign {campaign.source_name!r} conflicts with its frozen task or success targets."
                raise ValueError(msg)
            observed_batches = [(item.get("batch_name"), item.get("scientific_config_digest")) for item in observed.get("batches", [])]
            expected_batches = [(item["batch_name"], item["scientific_config_digest"]) for item in expected["batches"]]
            if observed_batches != expected_batches:
                msg = f"Campaign {campaign.source_name!r} conflicts with its frozen case design."
                raise ValueError(msg)
            _register_authored_capacity(root, expected, _authored_capacity(campaign), creating=False, storage_root=storage_root)
        else:
            # Register before the plan so an interrupted creation retries into the same state.
            _register_authored_capacity(root, expected, _authored_capacity(campaign), creating=True, storage_root=storage_root)
            _write_new_json(plan_path, expected)
    return root


def load_campaign_plan(task: str, campaign_name: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Read the frozen resolved plan without reopening mutable YAML."""
    root = _root(task, campaign_name, storage_root)
    plan = _read_json(common.paths.resolve_generation_campaign_plan_path(root))
    if plan.get("schema_kind") != "generation_source_campaign_plan" or plan.get("schema_version") != _CAMPAIGN_PLAN_SCHEMA_VERSION:
        msg = f"Unsupported frozen campaign plan: {root}."
        raise ValueError(msg)
    if (
        plan.get("task") != task
        or plan.get("source_name") != campaign_name
        or identity.source_name(str(plan.get("source_label") or plan.get("campaign_name")), str(plan.get("campaign_digest"))) != campaign_name
    ):
        msg = f"Frozen campaign plan disagrees with its directory: {root}."
        raise ValueError(msg)
    if plan.get("seed_key") != _plan_seed_key(plan):
        msg = f"Frozen campaign plan seed key disagrees with its scientific design: {root}."
        raise ValueError(msg)
    _require_unique_execution_keys(plan["batches"])
    return plan


def _validate_execution_epoch(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    index: int,
    previous_sha: str | None,
    previous_start: int | None,
) -> None:
    """Check one epoch before either admitting or publishing its immutable bytes."""
    if (
        record.get("schema_kind") != "generation_campaign_execution_epoch"
        or record.get("schema_version") != 1
        or record.get("campaign_run_id") != manifest["campaign_run_id"]
        or record.get("campaign_digest") != manifest["campaign_digest"]
        or record.get("epoch_index") != index
        or record.get("previous_sha256") != previous_sha
    ):
        msg = "Campaign execution epoch disagrees with its immutable lineage."
        raise ValueError(msg)
    if record.get("execution_config_digest") != common.serialization.canonical_json_sha256(record.get("execution_values")):
        msg = "Campaign execution epoch has a changed execution contract."
        raise ValueError(msg)
    start = record.get("first_submission_index")
    if not isinstance(record.get("submission_config"), dict) or type(start) is not int:
        msg = "Campaign execution epoch lacks its submission contract."
        raise ValueError(msg)
    if start < 1 or start > len(manifest["submissions"]) + 1:
        msg = "Campaign execution epoch points outside persisted submissions."
        raise ValueError(msg)
    if previous_start is not None and start < previous_start:
        msg = "Campaign execution epoch submission bounds overlap."
        raise ValueError(msg)


def execution_epochs(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Admit epochs with nondecreasing starts; later equal-start epochs supersede idle ones."""
    parts = common.paths.campaign_run_parts(str(manifest["campaign_run_id"]))
    if parts is None:
        return ()
    storage = Path(str(manifest["remote_storage_root"]))
    directory = _epochs_directory(_root(*parts, storage))
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        msg = "Campaign execution-epoch directory is unsafe."
        raise ValueError(msg)
    paths = sorted(directory.iterdir())
    records: list[dict[str, Any]] = []
    previous_sha: str | None = None
    for index, path in enumerate(paths, start=1):
        if path.name != f"epoch_{index:04d}.json":
            msg = "Campaign execution epochs must be contiguous immutable records."
            raise ValueError(msg)
        record = _read_json(path)
        _validate_execution_epoch(
            record,
            manifest,
            index=index,
            previous_sha=previous_sha,
            previous_start=None if not records else int(records[-1]["first_submission_index"]),
        )
        previous_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(record)
    return tuple(records)


def append_execution_epoch(manifest: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a source/resource binding only for the next unsent submission."""
    if _EPOCH_OWNED_FIELDS.intersection(record):
        msg = "Execution epoch caller cannot override generated identity or submission boundaries."
        raise ValueError(msg)
    parts = common.paths.campaign_run_parts(str(manifest["campaign_run_id"]))
    if parts is None:
        msg = "Execution epochs are reserved for canonical campaigns."
        raise ValueError(msg)
    root = _root(*parts, Path(str(manifest["remote_storage_root"])))
    previous = execution_epochs(manifest)
    previous_path = _epochs_directory(root) / f"epoch_{len(previous):04d}.json"
    previous_sha = None if not previous else hashlib.sha256(previous_path.read_bytes()).hexdigest()
    next_submission_index = len(manifest["submissions"]) + 1
    payload = {
        **record,
        "schema_kind": "generation_campaign_execution_epoch",
        "schema_version": 1,
        "campaign_run_id": manifest["campaign_run_id"],
        "campaign_digest": manifest["campaign_digest"],
        "epoch_index": len(previous) + 1,
        "first_submission_index": next_submission_index,
        "previous_sha256": previous_sha,
    }
    _validate_execution_epoch(
        payload,
        manifest,
        index=len(previous) + 1,
        previous_sha=previous_sha,
        previous_start=None if not previous else int(previous[-1]["first_submission_index"]),
    )
    path = _epochs_directory(root) / f"epoch_{len(previous) + 1:04d}.json"
    _write_new_json(path, payload)
    return payload


def campaign_with_execution(campaign: config_service.CampaignConfig, execution_values: Mapping[str, Any]) -> config_service.CampaignConfig:
    """Apply an admitted execution binding without changing frozen science."""
    values = copy.deepcopy(dict(execution_values))
    return replace(
        campaign,
        execution_values=values,
        batches=tuple(replace(batch, execution_values=copy.deepcopy(values)) for batch in campaign.batches),
    )


def _cases_directory(root: Path, material: str) -> Path:
    """Return the material directory of one campaign, its case namespace beside the reserved ``_record/``."""
    return root / common.paths.validate_logical_name(material, label="material")


def _reservations(root: Path, material: str) -> list[dict[str, Any]]:
    """Read every reservation of a material from its case directories; gaps, strays, and conflicts fail closed."""
    directory = _cases_directory(root, material)
    if not directory.exists():
        return []
    if directory.is_symlink() or directory.parent.is_symlink() or not directory.is_dir():
        msg = f"Material case namespace is unsafe: {directory}."
        raise ValueError(msg)
    entries = common.paths.generation_material_case_directories(directory)
    if [entry.name for entry in entries] != [f"case_{index:04d}" for index in range(1, len(entries) + 1)]:
        msg = f"Material {material!r} has malformed or missing append-only candidate IDs."
        raise ValueError(msg)
    records: list[dict[str, Any]] = []
    for entry in entries:
        reservation_path = common.paths.resolve_generation_case_reservation_path(entry)
        if reservation_path.parent.is_symlink() or reservation_path.is_symlink() or not reservation_path.is_file():
            msg = f"Case directory has no published reservation: {entry}."
            raise ValueError(msg)
        record = _read_json(reservation_path)
        if record.get("case_id") != entry.name:
            msg = f"Reservation conflicts with its case directory: {entry}."
            raise ValueError(msg)
        records.append(record)
    keys = [(record.get("origin"), record.get("design_key")) for record in records]
    if len(keys) != len(set(keys)):
        msg = f"Material {material!r} has duplicated durable reservations."
        raise ValueError(msg)
    return records


def _case_namespace_has_symlink(case_directory: Path) -> bool:
    """Return whether the case, its material, its campaign, or its ``_record/`` is a symlink."""
    record = common.paths.resolve_generation_case_record_directory(case_directory)
    return any(path.is_symlink() for path in (case_directory, case_directory.parent, case_directory.parent.parent, record))


def _publish_reserved_case(root: Path, material: str, reservation: Mapping[str, Any]) -> Path:
    """Publish a new case directory together with its reservation in one rename under the controller lock."""
    case_id = str(reservation["case_id"])
    staging = common.paths.resolve_generation_reservation_staging_directory(root, material) / case_id
    if staging.exists() or staging.is_symlink():
        # A leftover was never published: no reader sees staging, and the controller lock excludes other writers.
        shutil.rmtree(staging)
    _write_new_json(common.paths.resolve_generation_case_reservation_path(staging), reservation)
    final = _cases_directory(root, material) / case_id
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    return final


def reserve_candidate(
    task: str,
    campaign_name: str,
    material: str,
    *,
    origin: str,
    design_key: str,
    candidate_position: int,
    target_slot: int | None,
    sample: Mapping[str, Any],
    derived_seeds: Mapping[str, int],
    batch_id: str | None = None,
    design_evidence: Mapping[str, Any] | None = None,
    expected_case_id: str | None = None,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Atomically bind one candidate ID to an immutable design position."""
    plan = load_campaign_plan(task, campaign_name, storage_root=storage_root)
    if material not in plan["selected_targets"] or origin not in _ORIGINS:
        msg = "Reservation material or origin is outside the frozen campaign plan."
        raise ValueError(msg)
    if not isinstance(design_key, str) or not design_key or "/" in design_key or ".." in design_key:
        msg = "Reservation design_key must be one safe nonempty label."
        raise ValueError(msg)
    if isinstance(candidate_position, bool) or not isinstance(candidate_position, int) or candidate_position < 1:
        msg = "Reservation candidate_position must be positive."
        raise ValueError(msg)
    if target_slot is not None and (isinstance(target_slot, bool) or not isinstance(target_slot, int) or target_slot < 1):
        msg = "Reservation target_slot must be positive when supplied."
        raise ValueError(msg)
    target = int(plan["selected_targets"][material])
    matching_batches = [
        batch for batch in plan["batches"] if batch["material_family"] == material and (batch_id is None or batch["batch_id"] == batch_id)
    ]
    if len(matching_batches) != 1:
        msg = "Reservation requires one exact material/regime batch owner."
        raise ValueError(msg)
    batch = matching_batches[0]
    resolved_batch_id = str(batch["batch_id"])
    batch_target = len(batch["case_indices"])
    if target_slot is not None and target_slot > target:
        msg = "Reservation target_slot exceeds the frozen material success target."
        raise ValueError(msg)
    if origin == "primary" and (candidate_position > target or target_slot != candidate_position):
        msg = "Primary reservation must occupy its own frozen target slot."
        raise ValueError(msg)
    if origin == "primary" and candidate_position not in batch["case_indices"]:
        msg = "Primary reservation is outside its exact frozen batch membership."
        raise ValueError(msg)
    root = _root(task, campaign_name, storage_root)
    with common.locking.exclusive_file_lock(_controller_lock(root), blocking=True):
        records = _reservations(root, material)
        existing = [record for record in records if record["origin"] == origin and record["design_key"] == design_key]
        if existing:
            reservation = existing[0]
            if (
                reservation["candidate_position"] != candidate_position
                or reservation["target_slot"] != target_slot
                or reservation["sample"] != dict(sample)
                or reservation["derived_seeds"] != dict(derived_seeds)
                or reservation["batch_id"] != resolved_batch_id
                or reservation.get("design_evidence") != (None if design_evidence is None else dict(design_evidence))
            ):
                msg = "Existing candidate reservation conflicts with the requested design evidence."
                raise ValueError(msg)
        else:
            if common.paths.resolve_generation_selected_pool_path(root).exists():
                msg = "A sealed selected-success pool cannot admit another candidate."
                raise RuntimeError(msg)
            if origin == "replacement":
                material_records = [
                    row
                    for row in candidate_records(task, campaign_name, storage_root=storage_root)
                    if row["reservation"]["batch_id"] == resolved_batch_id
                ]
                if selection_service.replacement_deficit(material_records, target=batch_target) < 1:
                    msg = "Replacement reservation has no terminal-failure deficit after active work is counted."
                    raise ValueError(msg)
            case_id = f"case_{len(records) + 1:04d}"
            if expected_case_id is not None and case_id != expected_case_id:
                msg = "Candidate reservation would violate the frozen material-local case index order."
                raise ValueError(msg)
            reservation = {
                "schema_kind": "generation_candidate_reservation",
                "schema_version": 1,
                "task": task,
                "campaign_name": campaign_name,
                "campaign_digest": plan["campaign_digest"],
                "material_family": material,
                "batch_id": resolved_batch_id,
                "sampling_regime": batch["sampling_regime"],
                "case_id": case_id,
                "origin": origin,
                "design_key": design_key,
                "candidate_position": candidate_position,
                "target_slot": target_slot,
                "root_seed": plan["root_seed"],
                "seed_derivation_version": plan["seed_derivation_version"],
                "candidate_seed": seeding.derive_seed(plan["root_seed"], "campaign_candidate", task, plan["seed_key"], material, origin, design_key),
                "derived_seeds": dict(derived_seeds),
                "design_evidence": None if design_evidence is None else copy.deepcopy(dict(design_evidence)),
                "sample": dict(sample),
            }
            _publish_reserved_case(root, material, reservation)
        if expected_case_id is not None and reservation["case_id"] != expected_case_id:
            msg = "Existing candidate reservation disagrees with the frozen material-local case index."
            raise ValueError(msg)
    return reservation


def record_candidate_attempt(
    task: str,
    campaign_name: str,
    material: str,
    case_id: str,
    *,
    attempt_index: int,
    stage: str,
    failure_class: str,
    reason_code: str,
    retryable: bool,
    detail: str,
    evidence: Mapping[str, Any],
    storage_root: Path | str | None = None,
) -> Path:
    """Append a classified operational attempt without changing candidate fate; a solver attempt carries a typed runtime reason."""
    if failure_class not in _ATTEMPT_CLASSES or not stage or not reason_code or not detail:
        msg = "Attempt requires a stable stage/class/reason and human detail."
        raise ValueError(msg)
    record = {
        "schema_kind": _ATTEMPT_SCHEMA[0],
        "schema_version": _ATTEMPT_SCHEMA[1],
        "case_id": case_id,
        "attempt_index": attempt_index,
        "stage": stage,
        "class": failure_class,
        "reason_code": reason_code,
        "retryable": retryable,
        "terminal": False,
        "detail": detail,
        "evidence": dict(evidence),
    }
    _require_attempt_contract(record, label=case_id)
    if not isinstance(retryable, bool) or isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
        msg = "Attempt retryability and index are malformed."
        raise ValueError(msg)
    root = _root(task, campaign_name, storage_root)
    case_dir = common.paths.resolve_generation_campaign_case_directory(task, campaign_name, material, case_id, storage_root=storage_root)
    if _case_namespace_has_symlink(case_dir):
        msg = "Campaign case namespace contains a symlink."
        raise ValueError(msg)
    reservation = _read_json(common.paths.resolve_generation_case_reservation_path(case_dir))
    if reservation.get("case_id") != case_id:
        msg = "Attempt candidate identity conflicts with reservation."
        raise ValueError(msg)
    path = common.paths.resolve_generation_case_attempt_record_path(case_dir, attempt_index)
    attempts_dir = common.paths.resolve_generation_case_attempts_directory(case_dir)
    with common.locking.exclusive_file_lock(_controller_lock(root), blocking=True):
        existing = sorted(attempts_dir.glob("attempt_*.json")) if attempts_dir.exists() else []
        if not path.exists() and len(existing) + 1 != attempt_index:
            msg = "Attempt index must extend the append-only attempt chain."
            raise ValueError(msg)
        _write_new_json(path, record)
    return path


def decide_candidate(
    task: str,
    campaign_name: str,
    material: str,
    case_id: str,
    *,
    outcome: str,
    stage: str,
    failure_class: str | None,
    reason_code: str | None,
    detail: str,
    case_input_id: str,
    simulation_case_id: str,
    case_hdf5_relative: str | None,
    case_hdf5_sha256: str | None,
    evidence: Mapping[str, Any],
    storage_root: Path | str | None = None,
) -> Path:
    """Commit one terminal success or science-level failure decision."""
    if outcome not in {"successful", "failed"} or not stage or not detail:
        msg = "Terminal candidate decision is malformed."
        raise ValueError(msg)
    if outcome == "failed" and (failure_class not in {"solver", "validation"} or not reason_code):
        msg = "A terminal failure requires an explicit solver/validation decision."
        raise ValueError(msg)
    if outcome == "successful" and (failure_class is not None or reason_code is not None):
        msg = "A successful candidate cannot carry a failure decision."
        raise ValueError(msg)
    if not case_input_id or not simulation_case_id:
        msg = "Terminal candidate decision requires scientific case identities."
        raise ValueError(msg)
    if outcome == "successful" and (not case_hdf5_relative or not case_hdf5_sha256 or _SHA256.fullmatch(case_hdf5_sha256) is None):
        msg = "Successful candidate requires an exact canonical HDF5 artifact hash."
        raise ValueError(msg)
    if case_hdf5_relative is not None and (Path(case_hdf5_relative).is_absolute() or ".." in Path(case_hdf5_relative).parts):
        msg = "Candidate HDF5 reference must remain storage-relative and contained."
        raise ValueError(msg)
    case_dir = common.paths.resolve_generation_campaign_case_directory(task, campaign_name, material, case_id, storage_root=storage_root)
    if _case_namespace_has_symlink(case_dir):
        msg = "Campaign case namespace contains a symlink."
        raise ValueError(msg)
    reservation = _read_json(common.paths.resolve_generation_case_reservation_path(case_dir))
    if reservation.get("case_id") != case_id:
        msg = "Candidate decision conflicts with its reservation."
        raise ValueError(msg)
    path = common.paths.resolve_generation_case_decision_path(case_dir)
    root = _root(task, campaign_name, storage_root)
    if outcome == "successful":
        storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
        artifact = storage / str(case_hdf5_relative)
        if artifact.is_symlink() or not artifact.resolve(strict=False).is_relative_to(storage) or not artifact.is_file():
            msg = "Successful candidate HDF5 is missing, unsafe, or outside storage."
            raise ValueError(msg)
        if (
            reservation["origin"] in {"primary", "replacement"}
            and artifact != common.paths.resolve_generation_case_result_directory(case_dir) / common.paths.GENERATION_RESULT_HDF5_FILENAME
        ):
            msg = "Generated success must reference its own canonical material-local case artifact."
            raise ValueError(msg)
        if common.serialization.file_sha256(artifact) != case_hdf5_sha256:
            msg = "Successful candidate HDF5 hash disagrees with the canonical artifact."
            raise ValueError(msg)
    with common.locking.exclusive_file_lock(_controller_lock(root), blocking=True):
        _write_new_json(
            path,
            {
                "schema_kind": "generation_candidate_decision",
                "schema_version": 1,
                "case_id": case_id,
                "outcome": outcome,
                "stage": stage,
                "class": failure_class,
                "reason_code": reason_code,
                "retryable": False,
                "terminal": True,
                "detail": detail,
                "case_input_id": case_input_id,
                "simulation_case_id": simulation_case_id,
                "case_hdf5_relative": case_hdf5_relative,
                "case_hdf5_sha256": case_hdf5_sha256,
                "evidence": dict(evidence),
            },
        )
    return path


def candidate_records(task: str, campaign_name: str, *, storage_root: Path | str | None = None) -> tuple[dict[str, Any], ...]:
    """Read every reserved candidate through one bounded frozen campaign plan."""
    plan = load_campaign_plan(task, campaign_name, storage_root=storage_root)
    root = _root(task, campaign_name, storage_root)
    rows: list[dict[str, Any]] = []
    for material in material_registry.ordered_materials(list(plan["selected_targets"])):
        for reservation in sorted(_reservations(root, material), key=lambda item: int(str(item["case_id"]).removeprefix("case_"))):
            case_dir = _cases_directory(root, material) / reservation["case_id"]
            decision_path = common.paths.resolve_generation_case_decision_path(case_dir)
            decision = _read_json(decision_path) if decision_path.exists() else None
            attempts_dir = common.paths.resolve_generation_case_attempts_directory(case_dir)
            attempts = [_read_json(path) for path in sorted(attempts_dir.glob("attempt_*.json"))] if attempts_dir.exists() else []
            for attempt in attempts:
                _require_attempt_contract(attempt, label=case_dir)
            rows.append({"reservation": reservation, "decision": decision, "attempts": attempts, "case_directory": case_dir})
    return tuple(rows)


def selected_successes(task: str, campaign_name: str, *, storage_root: Path | str | None = None) -> tuple[dict[str, Any], ...]:
    """Select exact successful target membership by immutable design order."""
    plan = load_campaign_plan(task, campaign_name, storage_root=storage_root)
    records = candidate_records(task, campaign_name, storage_root=storage_root)
    selected: list[dict[str, Any]] = []
    for batch in plan["batches"]:
        target = len(batch["case_indices"])
        batch_records = [row for row in records if row["reservation"]["batch_id"] == batch["batch_id"]]
        if sum(row["reservation"]["origin"] == "primary" for row in batch_records) < target:
            msg = f"Campaign {campaign_name!r} still has unreserved primary target slots for {batch['batch_name']!r}."
            raise RuntimeError(msg)
        selected.extend(dict(row) for row in selection_service.select_successes(batch_records, target=target))
    return tuple(selected)


def _selected_pool_receipt(task: str, campaign_name: str, *, storage_root: Path | str | None) -> dict[str, Any]:
    """Derive exact selected membership from frozen decisions without writing."""
    root = _root(task, campaign_name, storage_root)
    selected = selected_successes(task, campaign_name, storage_root=storage_root)
    plan = load_campaign_plan(task, campaign_name, storage_root=storage_root)
    return {
        "schema_kind": "generation_campaign_selected_pool",
        "schema_version": 1,
        "task": task,
        "campaign_name": campaign_name,
        "campaign_digest": plan["campaign_digest"],
        "campaign_plan_sha256": common.serialization.file_sha256(common.paths.resolve_generation_campaign_plan_path(root)),
        "selected": [
            {
                "material_family": row["reservation"]["material_family"],
                "batch_id": row["reservation"]["batch_id"],
                "case_id": row["reservation"]["case_id"],
                "case_input_id": row["decision"]["case_input_id"],
                "simulation_case_id": row["decision"]["simulation_case_id"],
                "case_hdf5_relative": row["decision"]["case_hdf5_relative"],
                "case_hdf5_sha256": row["decision"]["case_hdf5_sha256"],
                "reservation_sha256": common.serialization.file_sha256(common.paths.resolve_generation_case_reservation_path(row["case_directory"])),
                "decision_sha256": common.serialization.file_sha256(common.paths.resolve_generation_case_decision_path(row["case_directory"])),
            }
            for row in selected
        ],
    }


def load_selected_pool(task: str, campaign_name: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Read a sealed pool and rederive its immutable decision and plan binding."""
    path = common.paths.resolve_generation_selected_pool_path(_root(task, campaign_name, storage_root))
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        msg = f"Canonical selected pool is missing or unsafe: {path}."
        raise FileNotFoundError(msg)
    recorded = _read_json(path)
    if recorded != _selected_pool_receipt(task, campaign_name, storage_root=storage_root):
        msg = "Canonical selected pool differs from its frozen plan and selected decisions."
        raise ValueError(msg)
    return recorded


def seal_selected_pool(task: str, campaign_name: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Publish one immutable exact success membership with decision byte hashes."""
    root = _root(task, campaign_name, storage_root)
    with common.locking.exclusive_file_lock(_controller_lock(root), blocking=True):
        receipt = _selected_pool_receipt(task, campaign_name, storage_root=storage_root)
        _write_new_json(common.paths.resolve_generation_selected_pool_path(root), receipt)
    return receipt


def _replacement_amendments(directory: Path, batch_id: str) -> tuple[dict[str, Any], ...]:
    """Read one batch's explicit, append-only replacement capacity history from its design directory."""
    if directory.is_symlink() or directory.parent.is_symlink():
        msg = "Replacement capacity evidence contains a symlink."
        raise ValueError(msg)
    paths = sorted(directory.glob("amendment_*.json")) if directory.is_dir() else ()
    records = tuple(_read_json(path) for path in paths)
    if any(
        record.get("schema_kind") != "generation_replacement_pool_amendment"
        or record.get("schema_version") != _AMENDMENT_SCHEMA_VERSION
        or record.get("batch_id") != batch_id
        or record.get("amendment_index") != index
        or record.get("registered_by") not in _AMENDMENT_SOURCES
        or (record["registered_by"] == "plan" and index != 1)
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
        or record.get("high_water") != int(record.get("previous_high_water", -1)) + int(record.get("added_capacity", -1))
        or int(record["added_capacity"]) < 1
        or (index > 1 and record["previous_high_water"] != records[index - 2]["high_water"])
        for index, record in enumerate(records, start=1)
    ):
        msg = "Replacement capacity amendments are malformed or discontinuous."
        raise ValueError(msg)
    return records


def _frozen_batch(plan: Mapping[str, Any], batch_id: str) -> Mapping[str, Any]:
    """Return the one original batch of a frozen plan that owns a replacement capacity."""
    matches = [batch for batch in plan["batches"] if batch["batch_id"] == batch_id]
    if len(matches) != 1:
        msg = "Replacement capacity requires one frozen batch identity."
        raise ValueError(msg)
    return matches[0]


def _batch_amendments(batch: Mapping[str, Any], *, storage_root: Path | str | None) -> tuple[dict[str, Any], ...]:
    """Read the validated capacity history of one frozen original batch."""
    directory = common.paths.resolve_generation_replacements_directory(batch["batch_storage_name"], storage_root=storage_root)
    return _replacement_amendments(directory, batch["batch_id"])


def replacement_pool_high_water(
    task: str,
    campaign_name: str,
    batch_id: str,
    *,
    storage_root: Path | str | None = None,
) -> int:
    """Read the last explicit replacement capacity amendment for one batch."""
    plan = load_campaign_plan(task, campaign_name, storage_root=storage_root)
    amendments = _batch_amendments(_frozen_batch(plan, batch_id), storage_root=storage_root)
    return int(amendments[-1]["high_water"]) if amendments else 0


def replacement_policy(
    task: str,
    campaign_name: str,
    *,
    storage_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Report every original batch's pre-registered capacity and its complete amendment history."""
    plan = load_campaign_plan(task, campaign_name, storage_root=storage_root)
    policy: list[dict[str, Any]] = []
    for batch in plan["batches"]:
        amendments = _batch_amendments(batch, storage_root=storage_root)
        policy.append(
            {
                "material_family": batch["material_family"],
                "sampling_regime": batch["sampling_regime"],
                "batch_id": batch["batch_id"],
                "sampling_algorithm": "sobol",
                "planned_capacity": int(amendments[0]["high_water"]) if amendments and amendments[0]["registered_by"] == "plan" else 0,
                "high_water": int(amendments[-1]["high_water"]) if amendments else 0,
                "amendments": [
                    {
                        key: amendment[key]
                        for key in ("amendment_index", "previous_high_water", "added_capacity", "high_water", "registered_by", "reason")
                    }
                    for amendment in amendments
                ],
            }
        )
    return policy


def _append_amendment(
    root: Path,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    high_water: int,
    reason: str,
    registered_by: str,
    storage_root: Path | str | None,
) -> int:
    """Append one capacity amendment while the caller holds the controller lock."""
    if common.paths.resolve_generation_selected_pool_path(root).exists():
        msg = "A sealed selected-success pool cannot extend its replacement capacity."
        raise RuntimeError(msg)
    amendments = _batch_amendments(batch, storage_root=storage_root)
    previous = int(amendments[-1]["high_water"]) if amendments else 0
    if high_water < previous:
        msg = "Replacement high-water cannot decrease or rewrite an earlier design prefix."
        raise ValueError(msg)
    if high_water == previous:
        return previous
    if registered_by == "plan" and amendments:
        msg = "A pre-registered plan capacity must be the first amendment of its batch."
        raise ValueError(msg)
    index = len(amendments) + 1
    directory = common.paths.resolve_generation_replacements_directory(batch["batch_storage_name"], storage_root=storage_root)
    _write_new_json(
        directory / f"amendment_{index:04d}.json",
        {
            "schema_kind": "generation_replacement_pool_amendment",
            "schema_version": _AMENDMENT_SCHEMA_VERSION,
            "campaign_digest": plan["campaign_digest"],
            "batch_id": batch["batch_id"],
            "amendment_index": index,
            "previous_high_water": previous,
            "added_capacity": high_water - previous,
            "high_water": high_water,
            "sampling_algorithm": "sobol",
            "registered_by": registered_by,
            "reason": reason.strip(),
        },
    )
    return high_water


def amend_replacement_pool(
    task: str,
    campaign_name: str,
    batch_id: str,
    *,
    high_water: int,
    reason: str,
    storage_root: Path | str | None = None,
) -> int:
    """Explicitly raise one batch's prefix-stable replacement candidate limit and record why."""
    if isinstance(high_water, bool) or not isinstance(high_water, int) or high_water < 1:
        msg = "Replacement high-water must be a positive integer."
        raise ValueError(msg)
    if not isinstance(reason, str) or not reason.strip():
        msg = "A replacement capacity amendment requires a non-empty reason."
        raise ValueError(msg)
    plan = load_campaign_plan(task, campaign_name, storage_root=storage_root)
    batch = _frozen_batch(plan, batch_id)
    root = _root(task, campaign_name, storage_root)
    with common.locking.exclusive_file_lock(_controller_lock(root), blocking=True):
        return _append_amendment(root, plan, batch, high_water=high_water, reason=reason, registered_by="explicit", storage_root=storage_root)


def _authored_capacity(campaign: config_service.CampaignConfig) -> dict[str, int]:
    """Resolve the replacement capacity pre-registered by the campaign's authored plan."""
    from src.generation.cases import generation_cases_config as config_service  # noqa: PLC0415 -- runtime-only config materialization

    return config_service.resolve_replacement_capacity(campaign.authored_value, materials=list(campaign.selected_targets))


def _register_authored_capacity(
    root: Path,
    plan: Mapping[str, Any],
    capacity: Mapping[str, int],
    *,
    creating: bool,
    storage_root: Path | str | None,
) -> None:
    """Pre-register the authored capacity of each original batch at creation and refuse any later difference."""
    for batch in plan["batches"]:
        material = str(batch["material_family"])
        wanted = int(capacity.get(material, 0))
        amendments = _batch_amendments(batch, storage_root=storage_root)
        recorded = int(amendments[0]["high_water"]) if amendments and amendments[0]["registered_by"] == "plan" else 0
        if recorded == wanted:
            continue
        if not creating or amendments:
            msg = (
                f"Replacement capacity of {material!r} ({batch['sampling_regime']}) is pre-registered per Source: "
                f"{plan['source_name']!r} records {recorded} from its authored plan, but the plan now requests {wanted}. "
                "Restore the recorded value, add capacity explicitly with amend-replacements, or create a new Source."
            )
            raise ValueError(msg)
        _append_amendment(
            root,
            plan,
            batch,
            high_water=wanted,
            reason="authored plan generation.replacement_capacity",
            registered_by="plan",
            storage_root=storage_root,
        )


def execution_batch(campaign: config_service.CampaignConfig, reservation: Mapping[str, Any]) -> config_service.GenerationConfig:
    """Resolve the frozen batch that executes one reserved candidate.

    A replacement reservation keeps its original parent batch for target-slot
    lineage, while its own one-case batch carries the scheduler and input identity.
    """
    parent_id = reservation.get("batch_id")
    if reservation.get("origin") == "replacement":
        matches = [
            batch
            for batch in campaign.batches
            if isinstance(batch.scientific_values.get("supplemental_design"), dict)
            and batch.scientific_values["supplemental_design"].get("parent_batch_id") == parent_id
            and batch.scientific_values["supplemental_design"].get("candidate_id") == reservation.get("design_key")
        ]
    else:
        matches = [batch for batch in campaign.batches if batch.batch_id == parent_id]
    if len(matches) != 1:
        msg = "Candidate reservation is not bound to one frozen execution batch."
        raise ValueError(msg)
    return matches[0]


def _replacement_locators(parent: config_service.GenerationConfig, design: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve the canonical batch name and locator regime of one replacement design under its parent batch."""
    from src.generation.cases import generation_cases_config as config_service  # noqa: PLC0415 -- runtime-only config materialization

    return config_service.resolve_batch_locators(parent.profile.id, parent.material_family, parent.sampling_regime, {"supplemental_design": design})


def _resolved_replacement_batch(
    campaign: config_service.CampaignConfig,
    parent: config_service.GenerationConfig,
    design: Mapping[str, Any],
) -> config_service.GenerationConfig:
    """Rebuild one one-case batch from exact frozen supplemental evidence."""
    from src.generation.cases import generation_cases_config as config_service  # noqa: PLC0415 -- runtime-only config materialization

    case_index = int(design["material_case_index"])
    ordinal = int(design["candidate_ordinal"])
    target_slot = int(design["target_slot"])
    assignment = parent.case_assignment(target_slot)
    assignment["case_index"] = case_index
    assignment["regime_index"] = max(int(value["regime_index"]) for value in parent.assignments.values()) + ordinal
    scientific = copy.deepcopy(parent.scientific_values)
    scientific.update(
        {"case_count": 1, "case_index_start": case_index, "assignments": [assignment], "supplemental_design": copy.deepcopy(dict(design))}
    )
    batch_name, _ = _replacement_locators(parent, design)
    storage_name = f"source~{campaign.profile.id}~{campaign.source_name}~{parent.material_family}~{design['batch_locator_regime']}"
    scientific_digest = config_service.compute_scientific_config_digest(scientific)
    return replace(
        parent,
        batch_name=batch_name,
        batch_storage_name=storage_name,
        scientific_values=scientific,
        case_indices=(case_index,),
        assignments={case_index: assignment},
        scientific_config_digest=scientific_digest,
        case_input_config_digest=config_service.compute_case_input_config_digest(scientific),
        batch_identity=scientific_digest,
        batch_id=config_service.build_batch_id(batch_name, scientific_digest),
    )


def replacement_batch_from_reservation(
    campaign: config_service.CampaignConfig,
    reservation: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> config_service.GenerationConfig:
    """Reconstruct a published replacement without mutable YAML or a new campaign."""
    from src.generation.cases import generation_cases_sampling as sampling  # noqa: PLC0415 -- validate frozen scientific sample

    design = reservation.get("design_evidence")
    if reservation.get("origin") != "replacement" or not isinstance(design, dict) or design.get("schema_version") != _CAMPAIGN_PLAN_SCHEMA_VERSION:
        msg = "Reservation is not one canonical replacement design."
        raise ValueError(msg)
    ordinal = design.get("candidate_ordinal")
    case_index = design.get("material_case_index")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (ordinal, case_index)):
        msg = "Replacement reservation has an invalid candidate ordinal or case index."
        raise ValueError(msg)
    matches = [batch for batch in campaign.batches if batch.batch_id == reservation.get("batch_id")]
    if len(matches) != 1:
        msg = "Replacement reservation lacks its frozen parent batch."
        raise ValueError(msg)
    parent = matches[0]
    plan = load_campaign_plan(campaign.profile.id, campaign.source_name, storage_root=storage_root)
    plan_sha = common.serialization.file_sha256(
        common.paths.resolve_generation_campaign_plan_path(_root(campaign.profile.id, campaign.source_name, storage_root))
    )
    if (
        plan["campaign_digest"] != campaign.campaign_digest
        or design.get("campaign_plan_sha256") != plan_sha
        or design.get("parent_campaign_digest") != campaign.campaign_digest
        or design.get("parent_batch_id") != parent.batch_id
        or design.get("parent_scientific_config_digest") != parent.scientific_config_digest
        or design.get("root_seed") != campaign.root_seed
        or reservation.get("material_family") != parent.material_family
        or reservation.get("case_id") != f"case_{case_index:04d}"
        or design.get("batch_locator_regime") != _replacement_locators(parent, design)[1]
    ):
        msg = "Replacement reservation conflicts with its frozen campaign science or material-local case ID."
        raise ValueError(msg)
    unbound = {**design, "candidate_id": "", "physical_values": None, "physical_values_sha256": None}
    if design.get("candidate_id") != f"replacement__{common.serialization.canonical_json_sha256(unbound)[:24]}":
        msg = "Replacement candidate identity disagrees with its deterministic design."
        raise ValueError(msg)
    batch = _resolved_replacement_batch(campaign, parent, design)
    sampling.sample_case(batch, batch.case_indices[0])
    return batch


def materialize_replacement_batch(
    campaign: config_service.CampaignConfig,
    batch_id: str,
    *,
    storage_root: Path | str | None = None,
) -> config_service.GenerationConfig:
    """Derive the next one-case Sobol design inside the original campaign."""
    from src.generation.cases import generation_cases_sampling as sampling  # noqa: PLC0415 -- runtime-only scientific sampling

    if campaign.schema_version != _CAMPAIGN_PLAN_SCHEMA_VERSION or campaign.sampling_mode not in {"lhs", "sobol"}:
        msg = "Generated replacements require one canonical sampled campaign."
        raise ValueError(msg)
    plan = load_campaign_plan(campaign.profile.id, campaign.source_name, storage_root=storage_root)
    if plan["campaign_digest"] != campaign.campaign_digest:
        msg = "Replacement campaign differs from its frozen scientific plan."
        raise ValueError(msg)
    matches = [batch for batch in campaign.batches if batch.batch_id == batch_id]
    if len(matches) != 1:
        msg = "Replacement requires one original frozen batch."
        raise ValueError(msg)
    parent = matches[0]
    material = parent.material_family
    root = _root(campaign.profile.id, campaign.source_name, storage_root)
    records = candidate_records(campaign.profile.id, campaign.source_name, storage_root=storage_root)
    material_records = [row for row in records if row["reservation"]["material_family"] == material]
    batch_records = [row for row in material_records if row["reservation"]["batch_id"] == batch_id]
    target = len(parent.case_indices)
    primaries = [row for row in batch_records if row["reservation"]["origin"] == "primary"]
    if len(primaries) != target or any(row["decision"] is None for row in primaries):
        msg = "All primary candidates in the batch must settle before deterministic replacement planning."
        raise RuntimeError(msg)
    if sum(row["reservation"]["origin"] == "primary" for row in material_records) != plan["selected_targets"][material]:
        msg = "All material primary slots must be reserved before allocating a replacement case ID."
        raise RuntimeError(msg)
    if selection_service.replacement_deficit(batch_records, target=target) < 1:
        msg = "No terminal-failure success deficit remains after active replacement work."
        raise RuntimeError(msg)
    failed = sorted((row for row in primaries if row["decision"]["outcome"] == "failed"), key=lambda row: row["reservation"]["case_id"])
    occupied_slots = {
        row["reservation"]["target_slot"]
        for row in batch_records
        if row["reservation"]["origin"] == "replacement" and (row["decision"] is None or row["decision"]["outcome"] == "successful")
    }
    available = [row for row in failed if row["reservation"]["target_slot"] not in occupied_slots]
    if not available:
        msg = "Replacement deficit has no settled failed target slot."
        raise RuntimeError(msg)
    failed_primary = available[0]
    target_slot = int(failed_primary["reservation"]["target_slot"])
    ordinal = sum(row["reservation"]["origin"] == "replacement" for row in batch_records) + 1
    amendments = _replacement_amendments(
        common.paths.resolve_generation_replacements_directory(parent.batch_storage_name, storage_root=storage_root), batch_id
    )
    high_water = int(amendments[-1]["high_water"]) if amendments else 0
    if ordinal > high_water:
        msg = "Replacement candidate exceeds the explicitly amended pool high-water."
        raise RuntimeError(msg)
    case_index = len(material_records) + 1
    round_index = ordinal
    plan_sha = common.serialization.file_sha256(common.paths.resolve_generation_campaign_plan_path(root))
    sampling_seed = seeding.derive_seed(campaign.root_seed, "campaign_replacement_pool_v1", campaign.profile.id, plan["seed_key"], batch_id)
    sampling_plans = parent.scientific_values["sampling"]["blocks"]
    point = sampling.supplemental_unit_point(
        registry=parent.scientific_values["material"]["parameter_registry"],
        blocks=tuple(sampling_plans),
        block_parameters={block: tuple(spec["parameters"]) for block, spec in sampling_plans.items()},
        seed=sampling_seed,
        ordinal=ordinal,
    )
    block_seeds = {block: seeding.derive_seed(sampling_seed, "supplemental_replacement", block) for block in point}
    design = {
        "schema_kind": "generation_source_supplemental_design",
        "schema_version": 1,
        "campaign_plan_sha256": plan_sha,
        "parent_campaign_digest": campaign.campaign_digest,
        "parent_batch_id": batch_id,
        "parent_scientific_config_digest": parent.scientific_config_digest,
        "parent_failed_case_id": failed_primary["reservation"]["case_id"],
        "candidate_id": "",
        "candidate_ordinal": ordinal,
        "batch_ordinal": ordinal,
        "round": round_index,
        "population": ordinal,
        "target_slot": target_slot,
        "material_case_index": case_index,
        "root_seed": campaign.root_seed,
        "seed_derivation_version": 1,
        "batch_locator_regime": "",
        "sampling_algorithm": "sobol",
        "sampling_seed": sampling_seed,
        "block_seeds": block_seeds,
        "unit_point": point,
        "unit_point_sha256": common.serialization.canonical_json_sha256({"unit_point": point}),
        "physical_values": None,
        "physical_values_sha256": None,
    }
    design["batch_locator_regime"] = _replacement_locators(parent, design)[1]
    design["candidate_id"] = f"replacement__{common.serialization.canonical_json_sha256(design)[:24]}"
    sample = sampling.sample_case(_resolved_replacement_batch(campaign, parent, design), case_index)
    design["physical_values"] = sample.values
    design["physical_values_sha256"] = common.serialization.canonical_json_sha256({"physical_values": sample.values})
    final = _resolved_replacement_batch(campaign, parent, design)
    if sampling.sample_case(final, case_index).values != sample.values:
        msg = "Replacement sample changed after scientific identity was finalized."
        raise RuntimeError(msg)
    return final


def campaign_from_frozen_run(
    manifest: Mapping[str, Any],
    *,
    require_executable: bool = True,
    storage_root: Path | str | None = None,
) -> config_service.CampaignConfig:
    """Reconstruct one v2 campaign and reserved replacements from frozen science."""
    from src.generation.cases import generation_cases_config as config_service  # noqa: PLC0415 -- reconstruct typed campaign from immutable plan
    from src.generation.contracts import generation_contracts_profiles as profiles  # noqa: PLC0415 -- frozen profile registry
    from src.generation.contracts import generation_contracts_templates as templates  # noqa: PLC0415 -- exact template validation

    run_id = str(manifest.get("campaign_run_id"))
    parts = common.paths.campaign_run_parts(run_id)
    if parts is None:
        msg = "Frozen campaign reconstruction requires a canonical v2 run handle."
        raise ValueError(msg)
    task, name = parts
    storage_value = manifest.get("remote_storage_root")
    if (
        not isinstance(storage_value, str)
        or not Path(storage_value).is_absolute()
        or ".." in Path(storage_value).parts
        or Path(storage_value).parent == Path(storage_value)
    ):
        msg = "Frozen campaign run lacks an absolute persisted storage root."
        raise ValueError(msg)
    storage = Path(storage_value) if storage_root is None else common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    plan = load_campaign_plan(task, name, storage_root=storage)
    if (
        manifest.get("campaign_name") != plan["campaign_name"]
        or manifest.get("source_name") != name
        or manifest.get("simulation_profile") != task
        or manifest.get("campaign_digest") != plan["campaign_digest"]
        or manifest.get("campaign_id") != plan["campaign_id"]
        or manifest.get("execution_config_digest") != common.serialization.canonical_json_sha256(plan["execution_values"])
    ):
        msg = "Campaign-run identity conflicts with its immutable resolved plan."
        raise ValueError(msg)
    planned_names = [batch["batch_name"] for batch in plan["batches"]]
    if manifest.get("selected_batch_names") != planned_names:
        msg = "Canonical campaign run must own the complete frozen campaign plan."
        raise ValueError(msg)
    project = common.paths.get_project_root().resolve()
    source_locator = manifest.get("campaign_config")
    if not isinstance(source_locator, str):
        msg = "Frozen campaign has no authored configuration locator."
        raise ValueError(msg)  # noqa: TRY004 -- invalid persisted JSON evidence is reported by the established ValueError contract
    source_relative = Path(source_locator)
    if (
        source_relative.is_absolute()
        or ".." in source_relative.parts
        or source_relative.parts[:1] != ("configs",)
        or source_relative.suffix != ".yaml"
    ):
        msg = "Frozen campaign authored configuration locator is unsafe."
        raise ValueError(msg)
    source_path = project / source_relative
    if source_path.resolve(strict=False) != source_path:
        msg = "Frozen campaign authored configuration locator escapes its repository."
        raise ValueError(msg)
    if source_locator not in {record["relative_path"] for record in plan["source_files"]}:
        msg = "Campaign-run authored locator is absent from its frozen source records."
        raise ValueError(msg)
    source_paths = tuple(project / str(record["relative_path"]) for record in plan["source_files"])
    template_relative = str(plan["template"]["relative_path"])
    if require_executable:
        template = templates.resolve_template_identity(template_relative, repository_root=project)
        if template.sha256 != plan["template"]["sha256"]:
            msg = "Current executable template differs from the frozen campaign plan."
            raise ValueError(msg)
    else:
        absolute = project / template_relative
        template = templates.ResolvedTemplateIdentity(template_relative, absolute, absolute.with_suffix(".sha256"), plan["template"]["sha256"])
    profile = profiles.resolve_profile(task)
    batches = tuple(
        config_service.GenerationConfig(
            source_path=source_path,
            profile=profile,
            material_family=str(item["material_family"]),
            material_role="source",
            evaluation_regime="parameter_extension" if item["sampling_regime"] == "parameter_ood" else "reference_range",
            sampling_regime=str(item["sampling_regime"]),
            batch_name=str(item["batch_name"]),
            scientific_values=copy.deepcopy(item["scientific_values"]),
            execution_values=copy.deepcopy(item["execution_values"]),
            template=template,
            case_indices=tuple(int(index) for index in item["case_indices"]),
            seed_base=item["seed_base"],
            assignments={int(index): copy.deepcopy(value) for index, value in item["assignments"].items()},
            scientific_config_digest=str(item["scientific_config_digest"]),
            case_input_config_digest=str(item["case_input_config_digest"]),
            batch_identity=str(item["batch_identity"]),
            batch_id=str(item["batch_id"]),
            batch_storage_name=str(item["batch_storage_name"]),
        )
        for item in plan["batches"]
    )
    campaign = config_service.CampaignConfig(
        source_path=source_path,
        source_label=plan.get("source_label"),
        authored_value=copy.deepcopy(plan.get("authored_value")),
        campaign_name=plan["campaign_name"],
        campaign_digest=plan["campaign_digest"],
        package_request_digest=common.serialization.canonical_json_sha256([]),
        campaign_id=plan["campaign_id"],
        campaign_purpose=plan["campaign_purpose"],
        domain_profile=str(plan["domain_profile"]),
        evaluation_regimes=(),
        profile=profile,
        template=template,
        material_roles={"seen": tuple(plan["materials"]), "near_family_ood": (), "far_family_ood": (), "extreme_family_ood": ()},
        material_memberships={},
        membership={},
        source_registry=copy.deepcopy(plan["source_registry"]),
        source_files=source_paths,
        total_case_count=int(plan["total_case_count"]),
        batches=batches,
        dataset_packages=(),
        duplicate_case_input_policy=plan["duplicate_case_input_policy"],
        execution_values=copy.deepcopy(plan["execution_values"]),
        schema_version=_CAMPAIGN_PLAN_SCHEMA_VERSION,
        root_seed=int(plan["root_seed"]),
        sampling_mode=str(plan["sampling_mode"]),
        selected_targets={material: int(count) for material, count in plan["selected_targets"].items()},
        composition_sources=tuple(copy.deepcopy(plan.get("composition_sources", ()))),
        scientific_overrides=copy.deepcopy(plan["scientific_overrides"]),
        producer_identity=copy.deepcopy(plan["producer_identity"]),
        execution_overrides=copy.deepcopy(plan.get("execution_overrides", {})),
    )
    replacements = tuple(
        replacement_batch_from_reservation(campaign, row["reservation"], storage_root=storage)
        for row in candidate_records(task, name, storage_root=storage)
        if row["reservation"]["origin"] == "replacement"
    )
    if len({batch.batch_name for batch in (*batches, *replacements)}) != len(batches) + len(replacements):
        msg = "Frozen replacement batch names are duplicated."
        raise ValueError(msg)
    return replace(campaign, batches=(*batches, *replacements), total_case_count=campaign.total_case_count + len(replacements))
