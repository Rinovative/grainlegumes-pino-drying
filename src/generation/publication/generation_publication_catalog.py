"""Expose bounded read-only candidate and raw-case evidence for one campaign.

Responsibilities:
  - Project primary, replacement, reused, successful, and failed case records
  - Summarize one Source's replacement policy and candidate outcomes as immutable Dataset evidence
  - Resolve exact immutable source references before cross-campaign reuse
  - Preserve individual failure inputs, diagnostics, and attempt lineage
  - Render a campaign's derived cases.csv view with outcomes, typed reasons, sampled parameters and solver logs

Design principles:
  - One task and campaign name bound every query
  - Manifest evidence, not directory order or modification time, establishes outcome
  - Expensive artifact hashing occurs only on explicit source admission
  - The cases view reads only immutable records, so re-rendering a settled campaign is byte-identical
  - A successful row reads the evidence admission requires strictly for either task; only evidence the producer may omit stays optional

This module does NOT:
  - Allocate cases, select split membership, or package Dataset payloads
  - Search all scientific storage on every status or viewer request
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.generation.runtime.generation_runtime_batch import TerminalCaseEvidence
    from src.generation.runtime.generation_runtime_timing import CaseTimingEvidence

from src import common
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cases import generation_cases_input as input_service
from src.generation.contracts import generation_contracts_campaign_identity as identity

from . import generation_publication_campaign_evidence as campaign_evidence

# Fixed leading columns of a campaign's derived cases.csv; sampled parameters and the failed solver log follow.
_CASES_VIEW_COLUMNS: Final = (
    "material",
    "case_id",
    "origin",
    "outcome",
    "reason_code",
    "detail",
    "duration_s",
    "simulated_end_h",
    "final_bulk_moisture_wb",
)


@dataclass(frozen=True, slots=True)
class AdmittedRawCase:
    """Bind one admitted completed case to its immutable source design."""

    case: TerminalCaseEvidence
    hdf5_path: Path
    _source_identity_json: str
    _design_provenance_json: str
    _scientific_config_json: str
    _git_commit: str

    @property
    def source_identity(self) -> dict[str, Any]:
        """Return independent JSON-ready exact source identity evidence."""
        return json.loads(self._source_identity_json)

    @property
    def design_provenance(self) -> dict[str, Any]:
        """Return independent JSON-ready original and supplemental design evidence."""
        return json.loads(self._design_provenance_json)

    def load_timing(self) -> CaseTimingEvidence:
        """Admit status and timing against this exact source case and frozen science."""
        from src.generation.runtime import generation_runtime_timing as timing_service  # noqa: PLC0415 -- selected source read

        context = _SourceCaseTimingContext(
            admitted_case=self.case,
            batch_id=self.source_identity["source_batch_id"],
            simulation_profile=self.source_identity["task"],
            git_commit=self._git_commit,
            _scientific_config_json=self._scientific_config_json,
        )
        return timing_service.load_case_timing(self.case, batch=context)


@dataclass(frozen=True, slots=True)
class _SourceCaseTimingContext:
    """Supply the timing reader with admitted single-case source evidence."""

    admitted_case: TerminalCaseEvidence
    batch_id: str
    simulation_profile: str
    git_commit: str
    _scientific_config_json: str

    def case(self, case_id: str) -> TerminalCaseEvidence:
        """Resolve only the exact already admitted source member."""
        if case_id != self.admitted_case.case_id:
            msg = "Timing case is not the admitted source member."
            raise ValueError(msg)
        return self.admitted_case

    def scientific_config_payload(self) -> dict[str, Any]:
        """Return the frozen resolved science bound by the source plan."""
        return json.loads(self._scientific_config_json)


def list_campaigns(
    task: str,
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Discover source campaigns, excluding test collections unless requested."""
    from src.generation.contracts import generation_contracts_profiles as profiles  # noqa: PLC0415 -- bounded task validation

    profiles.resolve_profile(task)
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    task_root = common.paths.get_generation_root(storage_root=storage) / task
    if task_root.is_symlink() or not task_root.resolve(strict=False).is_relative_to(storage):
        msg = "Canonical task namespace is unsafe."
        raise ValueError(msg)
    if not task_root.is_dir():
        return ()
    campaigns: list[dict[str, Any]] = []
    for directory in sorted(task_root.iterdir(), key=lambda item: item.name):
        if directory.is_symlink():
            msg = "Canonical task namespace contains a symlink."
            raise ValueError(msg)
        common.paths.admit_generation_campaign_record_directory(directory)
        if not directory.is_dir() or not common.paths.resolve_generation_campaign_plan_path(directory).exists():
            continue
        plan = lifecycle.load_campaign_plan(task, directory.name, storage_root=storage)
        campaigns.append(
            {
                "task": task,
                "campaign_name": plan["campaign_name"],
                "source_name": plan["source_name"],
                "campaign_digest": plan["campaign_digest"],
                "campaign_purpose": plan["campaign_purpose"],
                "domain_profile": plan["domain_profile"],
                "root_seed": plan["root_seed"],
                "sampling_mode": plan["sampling_mode"],
                "materials": plan["materials"],
                "original_design_sizes": plan["original_design_sizes"],
                "selected_targets": plan["selected_targets"],
            }
        )
    return tuple(campaigns)


def _selected_keys(task: str, campaign_name: str, *, storage_root: Path | str | None) -> set[tuple[str, str]]:
    """Return final exact selection, or an empty set while incomplete."""
    try:
        selected = lifecycle.selected_successes(task, campaign_name, storage_root=storage_root)
    except RuntimeError:
        return set()
    return {(row["reservation"]["material_family"], row["reservation"]["case_id"]) for row in selected}


def list_campaign_cases(
    task: str,
    campaign_name: str,
    *,
    storage_root: Path | str | None = None,
    selected_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    """List case outcomes and exact lineage from one frozen campaign namespace."""
    plan = lifecycle.load_campaign_plan(task, campaign_name, storage_root=storage_root)
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    selected = _selected_keys(task, campaign_name, storage_root=storage_root)
    rows: list[dict[str, Any]] = []
    for record in lifecycle.candidate_records(task, campaign_name, storage_root=storage_root):
        reservation = record["reservation"]
        decision = record["decision"]
        material = reservation["material_family"]
        designs = [batch for batch in plan["batches"] if batch["batch_id"] == reservation["batch_id"]]
        if len(designs) != 1:
            msg = "Candidate has no unique original source design."
            raise ValueError(msg)
        original_design_size = len(designs[0]["case_indices"])
        case_id = reservation["case_id"]
        is_selected = (material, case_id) in selected
        if selected_only and not is_selected:
            continue
        evidence = {} if decision is None else decision["evidence"]
        input_manifest_relative = evidence.get("input_manifest_relative")
        if input_manifest_relative is not None and not isinstance(input_manifest_relative, str):
            msg = "Candidate input manifest locator is malformed."
            raise ValueError(msg)
        case_directory = record["case_directory"]
        inputs_directory = common.paths.resolve_generation_case_inputs_directory(case_directory)
        input_case = inputs_directory / common.paths.GENERATION_CASE_PAYLOAD_FILENAME
        input_reference = inputs_directory / common.paths.GENERATION_INPUT_REFERENCE_FILENAME
        input_case_relative = None
        input_reference_relative = None
        if input_case.exists():
            if input_case.is_symlink() or not input_case.is_file() or input_case.resolve() != input_case or not input_case.is_relative_to(storage):
                msg = "Canonical candidate input case is unsafe."
                raise ValueError(msg)
            input_case_relative = input_case.relative_to(storage).as_posix()
        if input_reference.exists():
            if (
                input_reference.is_symlink()
                or not input_reference.is_file()
                or input_reference.resolve() != input_reference
                or not input_reference.is_relative_to(storage)
            ):
                msg = "Canonical candidate input reference is unsafe."
                raise ValueError(msg)
            input_reference_relative = input_reference.relative_to(storage).as_posix()
            reference = json.loads(input_reference.read_text(encoding="utf-8"))
            if (
                not isinstance(reference, dict)
                or reference.get("schema_kind") != "generation_campaign_input_reference"
                or not common.serialization.is_schema_version(reference.get("schema_version"), input_service.INPUT_REFERENCE_SCHEMA_VERSION)
            ):
                msg = "Canonical candidate input reference schema is malformed."
                raise ValueError(msg)
            input_generation_id = reference.get("input_generation_id")
            batch_locator = reference.get("batch_storage_name")
            parts = common.paths.campaign_batch_parts(batch_locator) if isinstance(batch_locator, str) else None
            if (
                not isinstance(input_generation_id, str)
                or parts is None
                or parts[:3] != (task, campaign_name, material)
                or reference.get("case_id") != case_id
                or (input_case_relative is not None and common.serialization.file_sha256(input_case) != reference.get("case_json_sha256"))
            ):
                msg = "Canonical candidate input reference disagrees with its case."
                raise ValueError(msg)
            if not isinstance(batch_locator, str):
                msg = "Canonical candidate input batch locator is malformed."
                raise ValueError(msg)
            manifest_path = (
                common.paths.resolve_generation_input_generation_metadata_directory(batch_locator, input_generation_id, storage_root=storage)
                / "input_generation_manifest.json"
            )
            if (
                manifest_path.is_symlink()
                or not manifest_path.is_file()
                or manifest_path.resolve() != manifest_path
                or not manifest_path.is_relative_to(storage)
            ):
                msg = "Canonical candidate input manifest is missing or unsafe."
                raise ValueError(msg)
            derived_relative = manifest_path.relative_to(storage).as_posix()
            if input_manifest_relative is not None and input_manifest_relative != derived_relative:
                msg = "Candidate decision input manifest differs from its immutable input reference."
                raise ValueError(msg)
            input_manifest_relative = derived_relative
        rows.append(
            {
                "task": task,
                "campaign_name": plan["campaign_name"],
                "source_name": campaign_name,
                "campaign_digest": plan["campaign_digest"],
                "material_family": material,
                "domain_profile": "parameter_extension" if reservation["sampling_regime"] == "parameter_ood" else "reference_range",
                "original_design_size": original_design_size,
                "batch_id": reservation["batch_id"],
                "sampling_regime": reservation["sampling_regime"],
                "case_id": case_id,
                "origin": reservation["origin"],
                "design_key": reservation["design_key"],
                "design_evidence": copy.deepcopy(reservation.get("design_evidence")),
                "candidate_position": reservation["candidate_position"],
                "target_slot": reservation["target_slot"],
                "root_seed": reservation["root_seed"],
                "derived_seeds": copy.deepcopy(reservation["derived_seeds"]),
                "sample": copy.deepcopy(reservation["sample"]),
                "outcome": "pending" if decision is None else decision["outcome"],
                "selected": is_selected,
                "case_input_id": None if decision is None else decision["case_input_id"],
                "simulation_case_id": None if decision is None else decision["simulation_case_id"],
                "case_hdf5_relative": None if decision is None else decision["case_hdf5_relative"],
                "case_hdf5_sha256": None if decision is None else decision["case_hdf5_sha256"],
                "input_manifest_relative": input_manifest_relative,
                "input_case_relative": input_case_relative,
                "input_reference_relative": input_reference_relative,
                "attempts": copy.deepcopy(record["attempts"]),
                "failure": None
                if decision is None or decision["outcome"] == "successful"
                else {
                    "stage": decision["stage"],
                    "class": decision["class"],
                    "reason_code": decision["reason_code"],
                    "detail": decision["detail"],
                },
                "parameters": copy.deepcopy(evidence.get("parameters")),
                "fields": copy.deepcopy(evidence.get("fields")),
                "schedules": copy.deepcopy(evidence.get("schedules")),
                "diagnostics": copy.deepcopy(evidence.get("diagnostics")),
                "source_provenance": copy.deepcopy(evidence.get("source_provenance")),
                "resources": copy.deepcopy(evidence.get("resources")),
                "duration_seconds": evidence.get("duration_seconds"),
            }
        )
    return tuple(rows)


def source_outcome_evidence(
    task: str,
    source_name: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Report one Source's replacement policy, every candidate outcome, and per-material failure counts.

    The result is derived only from frozen reservations, decisions and capacity
    amendments, so a Dataset can bind it as immutable evidence of the population
    its selected successes were drawn from.
    """
    plan = lifecycle.load_campaign_plan(task, source_name, storage_root=storage_root)
    candidates = [
        {
            "case_id": row["case_id"],
            "material_family": row["material_family"],
            "sampling_regime": row["sampling_regime"],
            "origin": row["origin"],
            "candidate_position": row["candidate_position"],
            "target_slot": row["target_slot"],
            "parent_failed_case_id": None if row["origin"] == "primary" else row["design_evidence"]["parent_failed_case_id"],
            "outcome": row["outcome"],
            "selected": row["selected"],
            "failure_stage": None if row["failure"] is None else row["failure"]["stage"],
            "failure_class": None if row["failure"] is None else row["failure"]["class"],
            "reason_code": None if row["failure"] is None else row["failure"]["reason_code"],
        }
        for row in list_campaign_cases(task, source_name, storage_root=storage_root)
    ]
    materials: dict[str, dict[str, int]] = {}
    for material in plan["selected_targets"]:
        rows = [candidate for candidate in candidates if candidate["material_family"] == material]
        originals = [candidate for candidate in rows if candidate["origin"] == "primary"]
        replacements = [candidate for candidate in rows if candidate["origin"] == "replacement"]
        materials[material] = {
            "original_candidates": len(originals),
            "original_failed": sum(candidate["outcome"] == "failed" for candidate in originals),
            "replacement_candidates": len(replacements),
            "replacement_failed": sum(candidate["outcome"] == "failed" for candidate in replacements),
            "successful": sum(candidate["outcome"] == "successful" for candidate in rows),
            "pending": sum(candidate["outcome"] == "pending" for candidate in rows),
        }
    return {
        "source_name": source_name,
        "campaign_digest": plan["campaign_digest"],
        "replacement_policy": lifecycle.replacement_policy(task, source_name, storage_root=storage_root),
        "materials": materials,
        "candidates": candidates,
    }


def _optional_json_object(path: Path) -> dict[str, Any] | None:
    """Return one optional JSON object for the derived cases view, or ``None`` when it is absent or unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _optional_number(value: object) -> float | None:
    """Return one finite number of a derived view, or ``None``."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _view_cell(value: object) -> str:
    """Render one cases-view value deterministically; a float keeps its shortest exact representation."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return repr(value) if isinstance(value, float) else str(value)


def _case_measurements(task: str, row: Mapping[str, Any], *, campaign_directory: Path) -> dict[str, object]:
    """Return a listed case's runtime, simulated end, final bulk moisture and last failed solver log where recorded.

    A successful row reads what admission requires of a success, for either task: its status and timing (both belong to the processed
    publication) and, for a transient case, its canonical final bulk moisture; an unreadable one raises with the case. Only values the producer
    may omit stay blank: a stationary solve has no time axis, so its status carries no ``t_stop_exact``. A failed row keeps its attempt files
    optional, because an attempt can end before it writes them.
    """
    from src.generation.publication import generation_publication_attempt as attempt_service  # noqa: PLC0415 -- cases-view read only
    from src.generation.publication import generation_publication_storage as storage_service  # noqa: PLC0415 -- HDF5 reader for the cases view

    case_id = str(row["case_id"])
    case_directory = campaign_directory / str(row["material_family"]) / case_id
    measured: dict[str, object] = {}
    if row["outcome"] == "successful":
        result = common.paths.resolve_generation_case_result_directory(case_directory)
        evidence = common.paths.resolve_generation_result_evidence_directory(result)
        status = campaign_evidence.load_json_object(evidence / "status.json", label=f"status of successful case {case_id}")
        timing = campaign_evidence.load_json_object(evidence / "timing.json", label=f"timing of successful case {case_id}")
        measured["duration_s"] = _optional_number(timing.get("runtime_s"))
        measured["simulated_end_h"] = _optional_number(status.get("t_stop_exact"))
        if task == "transient_drying":
            canonical = result / common.paths.GENERATION_RESULT_HDF5_FILENAME
            try:
                measured["final_bulk_moisture_wb"] = storage_service.read_transient_final_bulk_moisture(canonical)
            except (OSError, TypeError, ValueError) as error:
                message = f"Could not read the final bulk moisture of successful case {case_id}: {canonical}"
                raise ValueError(message) from error
        return measured
    runs = attempt_service.attempt_run_directories(common.paths.resolve_generation_case_attempts_directory(case_directory))
    if row["outcome"] == "failed" and runs:
        attempt = _optional_json_object(runs[-1] / "attempt.json") or {}
        simulated_seconds = _optional_number(attempt.get("final_simulation_time"))
        solver_log = runs[-1] / "payload" / "runtime" / "stdout.log"
        measured["duration_s"] = _optional_number(attempt.get("elapsed_seconds"))
        measured["simulated_end_h"] = None if simulated_seconds is None else simulated_seconds / 3600.0
        measured["solver_log"] = (
            solver_log.relative_to(campaign_directory).as_posix() if solver_log.is_file() and not solver_log.is_symlink() else None
        )
    return measured


def campaign_cases_csv(
    task: str,
    campaign_name: str,
    *,
    storage_root: Path | str | None = None,
) -> str:
    """Render the derived ``cases.csv`` view of one campaign: one row per candidate in material and case order.

    Columns are the outcome and typed reason, runtime, simulated end and final bulk
    moisture where recorded, every sampled parameter its decision recorded and, for
    a failed case, its last failure bundle's ``stdout.log`` relative to the campaign.
    The text depends only on immutable records, so re-rendering a settled campaign is
    byte-identical; it never decides, selects, or validates anything.
    """
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    campaign_directory = common.paths.resolve_generation_campaign_directory(task, campaign_name, storage_root=storage)
    rows = list_campaign_cases(task, campaign_name, storage_root=storage)
    sampled_values = [dict(row["parameters"]) if isinstance(row["parameters"], dict) else {} for row in rows]
    parameters = sorted({name for sampled in sampled_values for name in sampled})
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow((*_CASES_VIEW_COLUMNS, *parameters, "solver_log"))
    for row, sampled in zip(rows, sampled_values, strict=True):
        failure = row["failure"] or {}
        values = _case_measurements(task, row, campaign_directory=campaign_directory)
        cells = (
            row["material_family"],
            row["case_id"],
            "original" if row["origin"] == "primary" else row["origin"],
            row["outcome"],
            failure.get("reason_code"),
            failure.get("detail"),
            values.get("duration_s"),
            values.get("simulated_end_h"),
            values.get("final_bulk_moisture_wb"),
            *(sampled.get(name) for name in parameters),
            values.get("solver_log"),
        )
        writer.writerow(tuple(_view_cell(cell) for cell in cells))
    return buffer.getvalue()


def _contained_file(relative: str, *, storage_root: Path) -> Path:
    """Admit one storage-relative file without symlink or escape."""
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        msg = "Source case reference must be a contained storage-relative path."
        raise ValueError(msg)
    candidate = storage_root / path
    if not candidate.resolve(strict=False).is_relative_to(storage_root) or any(
        ancestor.is_symlink() for ancestor in (candidate, *candidate.parents) if ancestor.is_relative_to(storage_root)
    ):
        msg = "Source case reference is a symlink or escapes storage."
        raise ValueError(msg)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def resolve_raw_case_reference(
    task: str,
    source: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> AdmittedRawCase:
    """Admit one successful canonical raw case by exact identities and hashes."""
    required = {
        "campaign_name",
        "campaign_digest",
        "material_family",
        "case_id",
        "case_input_id",
        "simulation_case_id",
        "case_hdf5_sha256",
    }
    if set(source) != required:
        msg = "Raw case selector must contain exactly the seven source identity fields."
        raise ValueError(msg)
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    design_name = common.paths.validate_logical_name(source.get("campaign_name"), label="source_campaign_name")
    identity.source_name(design_name, source.get("campaign_digest"))
    material = common.paths.validate_logical_name(source.get("material_family"), label="source_material")
    case_id = common.paths.validate_logical_name(source.get("case_id"), label="source_case_id")
    # A Source folder may carry an authored label, so locate it through its frozen design name and digest.
    located = [
        (campaign["source_name"], row)
        for campaign in list_campaigns(task, storage_root=storage)
        if campaign["campaign_name"] == design_name and campaign["campaign_digest"] == source.get("campaign_digest")
        for row in list_campaign_cases(task, campaign["source_name"], storage_root=storage)
        if row["material_family"] == material and row["case_id"] == case_id
    ]
    if len(located) != 1 or located[0][1]["outcome"] != "successful":
        msg = "Raw case reference does not select one successful canonical source."
        raise ValueError(msg)
    campaign_name, row = located[0]
    for key in ("campaign_digest", "case_input_id", "simulation_case_id", "case_hdf5_sha256"):
        if source.get(key) != row[key]:
            msg = f"Raw case source identity disagrees at {key!r}."
            raise ValueError(msg)
    relative = row["case_hdf5_relative"]
    if not isinstance(relative, str):
        msg = "Successful raw case lacks a canonical HDF5 locator."
        raise ValueError(msg)  # noqa: TRY004 -- invalid persisted JSON evidence is reported by the established ValueError contract
    path = _contained_file(relative, storage_root=storage)
    if common.serialization.file_sha256(path) != row["case_hdf5_sha256"]:
        msg = "Canonical raw case HDF5 hash disagrees with frozen successful evidence."
        raise ValueError(msg)
    from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence  # noqa: PLC0415 -- frozen run admission
    from src.generation.runtime import generation_runtime_batch as runtime_batch  # noqa: PLC0415 -- completed case admission

    root = common.paths.resolve_generation_campaign_directory(task, campaign_name, storage_root=storage)
    plan = lifecycle.load_campaign_plan(task, campaign_name, storage_root=storage)
    matches = [batch for batch in plan["batches"] if batch["batch_id"] == row["batch_id"]]
    if len(matches) != 1:
        msg = "Raw case has no unique frozen source design."
        raise ValueError(msg)
    design = matches[0]
    run_id = f"source~{task}~{campaign_name}"
    run = campaign_evidence.load_campaign_run(run_id, storage_root=storage)
    campaign = lifecycle.campaign_from_frozen_run(run, require_executable=False, storage_root=storage)
    records = [
        record
        for record in lifecycle.candidate_records(task, campaign_name, storage_root=storage)
        if record["reservation"]["material_family"] == material and record["reservation"]["case_id"] == case_id
    ]
    if len(records) != 1 or records[0]["decision"] is None:
        msg = "Successful raw case lacks one immutable reservation and decision."
        raise ValueError(msg)
    record = records[0]
    reservation = record["reservation"]
    if reservation["origin"] == "replacement":
        batch = lifecycle.replacement_batch_from_reservation(campaign, reservation, storage_root=storage)
    else:
        matching_batch = next((item for item in campaign.batches if item.batch_id == row["batch_id"]), None)
        if matching_batch is None:
            msg = "Raw case has no unique admitted source batch."
            raise ValueError(msg)
        batch = matching_batch
    if batch is None:
        msg = "Raw case has no unique admitted source batch."
        raise ValueError(msg)
    if not case_id.startswith("case_") or not case_id[5:].isdigit():
        msg = "Raw case selector has a malformed candidate ID."
        raise ValueError(msg)
    case_index = int(case_id[5:])
    case = runtime_batch.admit_completed_case(batch, case_index, storage_root=storage, validation_depth="routine")
    if (
        case.hdf5_path != path
        or case.case_input_id != row["case_input_id"]
        or case.simulation_case_id != row["simulation_case_id"]
        or case.case_hdf5_sha256 != row["case_hdf5_sha256"]
    ):
        msg = "Admitted raw case differs from its successful source decision."
        raise ValueError(msg)
    plan_sha = common.serialization.file_sha256(common.paths.resolve_generation_campaign_plan_path(root))
    pool_path = common.paths.resolve_generation_selected_pool_path(root)
    pool_sha = None
    if pool_path.exists():
        lifecycle.load_selected_pool(task, campaign_name, storage_root=storage)
        pool_sha = common.serialization.file_sha256(pool_path)
    source_identity = {
        "schema_kind": "generation_source_case_reference",
        "task": task,
        "campaign_name": plan["campaign_name"],
        "source_name": campaign_name,
        "campaign_digest": row["campaign_digest"],
        "source_plan_sha256": plan_sha,
        "material_family": material,
        "domain_profile": row["domain_profile"],
        "source_batch_id": batch.batch_id,
        "source_design_parent_batch_id": reservation["batch_id"],
        "case_id": case_id,
        "case_input_id": row["case_input_id"],
        "simulation_case_id": row["simulation_case_id"],
        "case_hdf5_relative": relative,
        "case_hdf5_sha256": row["case_hdf5_sha256"],
        "reservation_sha256": common.serialization.file_sha256(common.paths.resolve_generation_case_reservation_path(record["case_directory"])),
        "decision_sha256": common.serialization.file_sha256(common.paths.resolve_generation_case_decision_path(record["case_directory"])),
        "selected_pool_sha256": pool_sha,
    }
    design_provenance = {
        "sampling_method": plan["sampling_mode"],
        "original_design_size": len(design["case_indices"]),
        "candidate_position": reservation["candidate_position"],
        "origin": reservation["origin"],
        "root_seed": reservation["root_seed"],
        "candidate_seed": reservation["candidate_seed"],
        "derived_seeds": reservation["derived_seeds"],
        "design_evidence": reservation["design_evidence"],
    }
    return AdmittedRawCase(
        case=case,
        hdf5_path=path,
        _source_identity_json=json.dumps(source_identity, sort_keys=True),
        _design_provenance_json=json.dumps(design_provenance, sort_keys=True),
        _scientific_config_json=json.dumps(batch.scientific_values, sort_keys=True),
        _git_commit=str(case.metadata_payload()["git_commit"]),
    )
