"""
===============================================================================
generation_smoke.py
===============================================================================
Validate, evidence, and compare native technical runtime smoke campaigns.
Responsibilities:
  - Publish profile-scoped evidence only after a complete technical workflow
  - Match production readiness to semantic mapping, template, and runtime identity
  - Require contrasting terminal steady and transient Slurm cases
  - Bind retained inputs, exports, HDF5, packages, loaders, and source identities
  - Report paired airflow differences and transient mass-balance observations
  - Write and revalidate one immutable runtime-validation receipt
Design principles:
  - Existing campaign and all-workflow receipts remain the lifecycle authorities
  - Equivalence and mass balance are reported without invented pass tolerances
  - Profile evidence excludes Git commit from its validity identity
This module does NOT:
  - Run COMSOL, submit Slurm jobs, infer export mappings, or launch production
  - Treat a technical smoke as experimental validation of scientific priors
===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np

from src import common, domain

from . import generation_campaign as campaign_runtime
from . import generation_workflow as workflow_service
from .cases import generation_cases_config as config_service
from .contracts import generation_contracts_comsol_spreadsheet as spreadsheet_contract
from .contracts import generation_contracts_mapping as mapping_contract
from .contracts import generation_contracts_profiles as profiles
from .publication import generation_publication_campaign_evidence as campaign_evidence
from .publication import generation_publication_storage as storage_service
from .runtime import generation_runtime_batch as runtime_service
from .runtime import generation_runtime_comsol as comsol_service
from .runtime import generation_runtime_preflight as preflight_service
from .runtime import generation_runtime_workspace as workspace_service

if TYPE_CHECKING:
    from collections.abc import Sequence

REAL_SMOKE_SCHEMA_KIND: Final = "vp2_real_runtime_smoke"
REAL_SMOKE_SCHEMA_VERSION: Final = 1
TECHNICAL_SMOKE_EVIDENCE_SCHEMA_KIND: Final = "generation_technical_smoke_evidence"
TECHNICAL_SMOKE_EVIDENCE_SCHEMA_VERSION: Final = 1
TECHNICAL_SMOKE_PURPOSE: Final = "technical_runtime_smoke"
_SHARED_FIELD_NAMES: Final = (
    *domain.fields.PERMEABILITY_FIELDS,
    *domain.fields.POROSITY_FIELDS,
    *domain.fields.BOUNDARY_FIELDS,
)
_AIRFLOW_FIELD_NAMES: Final = domain.fields.STATE_FIELDS
_MINIMUM_CONTRASTING_CASES: Final = 2
_EXPECTED_PROFILE_COUNT: Final = 2
_GIT_SHA_LENGTH: Final = 40
_SHA256_LENGTH: Final = 64
_COMSOL_EXACT_VERSION_PATTERN: Final = re.compile(r"(?<![0-9.])([0-9]+(?:[.][0-9]+){2,3})(?![0-9.])")
_TECHNICAL_SMOKE_EVIDENCE_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "status",
        "recorded_at",
        "simulation_profile",
        "mapping_contract_sha256",
        "template",
        "comsol",
        "technical_smoke_contract_sha256",
        "technical_smoke_campaign_id",
        "campaign_run_id",
        "git_commit",
        "required_case_count",
        "cases",
        "workflow_gate_sha256",
        "evidence_digest",
    }
)
_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "status",
        "recorded_at",
        "git_commit",
        "material_family_inventory",
        "source_binding",
        "templates",
        "profile_mappings",
        "campaigns",
        "cases",
        "comsol",
        "slurm",
        "dataset_packages",
        "cpu_source_retention",
        "template_equivalence",
        "mass_balance",
        "acceptance_tolerances",
        "receipt_digest",
    }
)


@dataclass(frozen=True, slots=True)
class _CaseEvidence:
    """Hold one validated receipt record plus arrays used for comparisons."""

    record: dict[str, Any]
    static: dict[str, np.ndarray]
    stationary_fixed: dict[str, float]
    scalars: dict[str, float]
    schedule: np.ndarray | None
    global_values: np.ndarray | None
    initial_state: dict[str, np.ndarray]


def _utc_now() -> str:
    """Return one timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required non-symlink JSON object."""
    if not path.is_file() or path.is_symlink():
        message = f"{label} is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"{label} is unreadable: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain one JSON object: {path}"
        raise TypeError(message)
    return value


def _text(value: Any, *, label: str) -> str:
    """Decode one HDF5 text scalar or attribute."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        message = f"{label} must be text."
        raise TypeError(message)
    return value


def _field_map(dataset: h5py.Dataset, *, label: str) -> dict[str, np.ndarray]:
    """Load one named HDF5 field tensor as isolated float64 arrays."""
    names = json.loads(_text(dataset.attrs.get("field_names"), label=f"{label}.field_names"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        message = f"{label} field names are malformed."
        raise TypeError(message)
    values = np.asarray(dataset, dtype=np.float64)
    if values.shape[0] != len(names):
        message = f"{label} leading dimension disagrees with field names."
        raise ValueError(message)
    return {name: np.asarray(values[index], dtype=np.float64) for index, name in enumerate(names)}


def _scalar_map(dataset: h5py.Dataset, *, label: str) -> dict[str, float]:
    """Load one named HDF5 scalar vector."""
    names = json.loads(_text(dataset.attrs.get("field_names"), label=f"{label}.field_names"))
    values = np.asarray(dataset, dtype=np.float64)
    if not isinstance(names, list) or values.shape != (len(names),):
        message = f"{label} scalar names or shape are malformed."
        raise ValueError(message)
    return {str(name): float(value) for name, value in zip(names, values, strict=True)}


def _relative(path: Path, *, storage: Path) -> str:
    """Return one storage-relative path after containment validation."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(storage).as_posix()
    except ValueError as error:
        message = f"Smoke artifact escapes the configured storage root: {resolved}"
        raise ValueError(message) from error


def _input_adapter_header(path: Path, *, delimiter: str) -> list[str]:
    """Read the first non-comment CSV header without loading its payload."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith(("%", "#")):
                    continue
                return [value.strip() for value in next(csv.reader([line], delimiter=delimiter))]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        message = f"Could not read retained CSV header: {path}"
        raise ValueError(message) from error
    message = f"Retained CSV contains no header: {path}"
    raise ValueError(message)


def _input_inventory(
    directory: Path,
    case_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate retained input bytes and exact profile-owned headers."""
    identities = case_payload.get("input_files")
    if not isinstance(identities, Mapping) or not identities:
        message = "Case provenance has no input-file identities."
        raise TypeError(message)
    expected = set(identities)
    actual = {path.name for path in directory.iterdir() if path.is_file() and not path.is_symlink()} if directory.is_dir() else set()
    if actual != expected:
        message = f"Retained input membership differs from case.json: expected {sorted(expected)}, got {sorted(actual)}."
        raise RuntimeError(message)
    contracts = case_payload["input_contract"]
    by_filename = {contract["filename"]: contract for contract in contracts.values()}
    records: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = directory / name
        identity = identities[name]
        contract = by_filename.get(name)
        if not isinstance(identity, Mapping) or not isinstance(contract, Mapping):
            message = f"Retained input contract is malformed for {name!r}."
            raise TypeError(message)
        sha256 = common.serialization.file_sha256(path)
        size_bytes = path.stat().st_size
        if sha256 != identity.get("sha256") or size_bytes != identity.get("size_bytes"):
            message = f"Retained input identity differs from case.json: {path}"
            raise RuntimeError(message)
        header = _input_adapter_header(path, delimiter=str(contract["delimiter"]))
        if header != list(contract["columns"]):
            message = f"Retained input header differs from its explicit contract: {path}"
            raise ValueError(message)
        records.append(
            {
                "filename": name,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "header": header,
            }
        )
    return records


def _required_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required HDF5 dataset after explicit runtime narrowing."""
    member = handle.get(name)
    if not isinstance(member, h5py.Dataset):
        message = f"Canonical HDF5 dataset is missing or malformed: {name}"
        raise TypeError(message)
    return member


def _source_exports(handle: h5py.File) -> dict[str, dict[str, Any]]:
    """Load canonical raw-export identities from HDF5 provenance."""
    dataset = handle.get("provenance/source_exports_json")
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
        message = "Canonical HDF5 source-export provenance is missing."
        raise TypeError(message)
    value = json.loads(_text(dataset[()], label="provenance/source_exports_json"))
    if not isinstance(value, dict) or not value:
        message = "Canonical HDF5 source-export provenance must be a non-empty object."
        raise TypeError(message)
    return value


def _export_inventory(
    directory: Path,
    source_exports: Mapping[str, Mapping[str, Any]],
    output_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate retained export bytes and report exact mapping observations."""
    contracts = {str(contract["role"]): contract for contract in output_contract["exports"]}
    actual = (
        {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and not path.is_symlink()}
        if directory.is_dir()
        else set()
    )
    if actual != set(source_exports):
        message = "Retained raw-export membership differs from canonical HDF5 provenance."
        raise RuntimeError(message)
    records: list[dict[str, Any]] = []
    for relative_path in sorted(source_exports):
        identity = source_exports[relative_path]
        role = str(identity["role"])
        contract = contracts.get(role)
        if contract is None:
            message = f"Raw export uses undeclared role {role!r}."
            raise ValueError(message)
        path = directory / relative_path
        sha256 = common.serialization.file_sha256(path)
        size_bytes = path.stat().st_size
        if sha256 != identity.get("sha256") or size_bytes != identity.get("size_bytes"):
            message = f"Retained raw-export identity changed: {path}"
            raise RuntimeError(message)
        observation = spreadsheet_contract.validate_export_mapping_observation(
            path,
            delimiter=str(contract["delimiter"]),
            columns=contract["columns"],
            units=contract["units"],
            wide_temporal=role == profiles.TRANSIENT_RAW_EXPORT_ROLE,
        )
        records.append(
            {
                "relative_path": relative_path,
                "role": role,
                "sha256": sha256,
                "size_bytes": size_bytes,
                **observation,
            }
        )
    return records


def _load_hdf5(
    path: Path, *, profile_id: str
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, float],
    np.ndarray | None,
    np.ndarray | None,
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Load bounded comparison data after complete canonical HDF5 validation."""
    identity = storage_service.validate_case_hdf5(path, expected_profile=profile_id)
    with h5py.File(path, "r") as handle:
        static_dataset = _required_dataset(handle, "static/fields")
        static = _field_map(static_dataset, label="static/fields")
        fixed_name = "stationary_fixed/values" if profile_id == profiles.STEADY_FLOW_PROFILE else "scalar/values"
        fixed_dataset = _required_dataset(handle, fixed_name)
        all_scalars = _scalar_map(fixed_dataset, label="case scalars")
        fixed = {name: all_scalars[name] for name in profiles.STATIONARY_FIXED_FIELDS}
        scalars = {} if profile_id == profiles.STEADY_FLOW_PROFILE else all_scalars
        schedule: np.ndarray | None = None
        global_values: np.ndarray | None = None
        transient_dataset: h5py.Dataset | None = None
        initial_state: dict[str, np.ndarray] = {}
        if profile_id == profiles.TRANSIENT_DRYING_PROFILE:
            schedule = np.asarray(
                _required_dataset(handle, "schedule/values"),
                dtype=np.float64,
            )
            global_values = np.asarray(
                _required_dataset(handle, "global/values"),
                dtype=np.float64,
            )
            transient_dataset = _required_dataset(handle, "transient/fields")
            names = json.loads(
                _text(
                    transient_dataset.attrs.get("field_names"),
                    label="transient/fields.field_names",
                )
            )
            initial = np.asarray(transient_dataset[0], dtype=np.float64)
            initial_state = {str(name): np.asarray(initial[index], dtype=np.float64) for index, name in enumerate(names)}
        exports = _source_exports(handle)
        shapes = {
            "static": list(static_dataset.shape),
            "transient": (None if transient_dataset is None else list(transient_dataset.shape)),
            "schedule": None if schedule is None else list(schedule.shape),
            "global": None if global_values is None else list(global_values.shape),
        }
    return static, fixed, scalars, schedule, global_values, initial_state, exports, {"identity": identity, "shapes": shapes}


def _case_evidence(
    config: config_service.GenerationConfig,
    case_index: int,
    *,
    storage: Path,
) -> _CaseEvidence:
    """Validate and summarize one published smoke case."""
    publication = runtime_service.validate_completed_case(config, case_index, storage_root=storage)
    raw = runtime_service.raw_case_directory(config, case_index, storage_root=storage)
    processed = runtime_service.processed_case_directory(config, case_index, storage_root=storage)
    case_payload = _json_object(raw / "case.json", label="smoke canonical raw case")
    timing = _json_object(processed / "timing.json", label="smoke case timing")
    execution = _json_object(processed / "execution_provenance.json", label="smoke execution provenance")
    status = _json_object(processed / "status.json", label="smoke solver status")
    if (
        timing.get("scheduler_kind") != "slurm"
        or not timing.get("scheduler_job_id")
        or execution.get("result", {}).get("state") != "succeeded"
        or status.get("solver_success") is not True
    ):
        message = f"Technical smoke case lacks successful native Slurm evidence: {config.case_id(case_index)}."
        raise RuntimeError(message)
    if config.execution_values["retention_policy"] != "full":
        message = "Technical smoke execution must use full attempt and solved-model retention."
        raise RuntimeError(message)
    solved = processed / comsol_service.RETAINED_MODEL_FILENAME
    if not solved.is_file() or solved.is_symlink() or solved.stat().st_size <= 0:
        message = f"Technical smoke solved model was not retained: {solved}"
        raise FileNotFoundError(message)
    hdf5_path = processed / "case.h5"
    static, fixed, scalars, schedule, global_values, initial_state, exports, hdf5 = _load_hdf5(
        hdf5_path,
        profile_id=config.profile.id,
    )
    input_records = _input_inventory(raw / "inputs", case_payload)
    expected_inputs = {"fields.csv"} if config.profile.id == profiles.STEADY_FLOW_PROFILE else {"fields.csv", "scalars.csv", "schedule.csv"}
    if {record["filename"] for record in input_records} != expected_inputs:
        message = f"Smoke input adapter membership is invalid for profile {config.profile.id!r}."
        raise ValueError(message)
    export_records = _export_inventory(
        processed / "comsol_exports",
        exports,
        config.scientific_values["output_contract"],
    )
    record = {
        "profile": config.profile.id,
        "batch_id": config.batch_id,
        "case_index": case_index,
        "case_id": case_payload["case_id"],
        "case_input_id": case_payload["case_input_id"],
        "simulation_case_id": case_payload["simulation_case_id"],
        "material_family": case_payload["material_family"],
        "sampling_regime": case_payload["sampling_regime"],
        "git_commit": case_payload["git_commit"],
        "scientific_config_digest": case_payload["scientific_config_digest"],
        "case_input_config_digest": case_payload["case_input_config_digest"],
        "template": dict(case_payload["template"]),
        "export_contract_sha256": case_payload["export_contract_sha256"],
        "input_files": input_records,
        "raw_exports": export_records,
        "raw_relative_path": _relative(raw, storage=storage),
        "processed_relative_path": _relative(processed, storage=storage),
        "hdf5": {
            **hdf5,
            "sha256": common.serialization.file_sha256(hdf5_path),
            "size_bytes": hdf5_path.stat().st_size,
        },
        "solved_model": {
            "sha256": common.serialization.file_sha256(solved),
            "size_bytes": solved.stat().st_size,
        },
        "timing": {
            "sha256": common.serialization.file_sha256(processed / "timing.json"),
            "runtime_s": timing["runtime_s"],
            "hostname": timing["hostname"],
            "scheduler_kind": timing["scheduler_kind"],
            "scheduler_job_id": timing["scheduler_job_id"],
            "scheduler_array_job_id": timing["scheduler_array_job_id"],
            "scheduler_array_task_id": timing["scheduler_array_task_id"],
            "requested_cores": timing["requested_cores"],
            "executable": timing["executable"],
        },
        "execution_provenance_sha256": common.serialization.file_sha256(processed / "execution_provenance.json"),
        "status_sha256": common.serialization.file_sha256(processed / "status.json"),
        "status": status,
        "publication": {
            "provenance_sha256": common.serialization.file_sha256(processed / "provenance.json"),
            "case_input_id": publication["case_input_id"],
            "simulation_case_id": publication["simulation_case_id"],
        },
        "seed_evidence": case_payload["seed_evidence"],
        "sampled_values": case_payload["sampled_values"],
        "scalars_consumed": scalars,
    }
    return _CaseEvidence(
        record=record,
        static=static,
        stationary_fixed=fixed,
        scalars=scalars,
        schedule=schedule,
        global_values=global_values,
        initial_state=initial_state,
    )


def _validate_campaign(
    run_id: str,
    *,
    expected_profile: str,
    storage: Path,
) -> tuple[
    config_service.CampaignConfig,
    dict[str, Any],
    dict[str, Any],
    tuple[_CaseEvidence, ...],
]:
    """Validate one retained technical-smoke workflow."""
    workflow = workflow_service.validate_completed_workflow(
        run_id,
        storage_root=storage,
    )
    campaign = campaign_evidence.campaign_for_run(
        run_id,
        storage_root=storage,
    )
    terminal = campaign_runtime.validate_terminal_campaign(
        run_id,
        storage_root=storage,
    )
    valid_batch_count = len(campaign.batches) == 1
    batch = campaign.batches[0] if valid_batch_count else None
    if (
        campaign.campaign_purpose != TECHNICAL_SMOKE_PURPOSE
        or campaign.profile.id != expected_profile
        or not valid_batch_count
        or batch is None
        or len(batch.case_indices) < _MINIMUM_CONTRASTING_CASES
        or batch.sampling_regime != "natural"
        or workflow["cleanup_requested"] is not False
        or workflow["cpu_cleanup_complete"] != {"status": "skipped_by_request", "evidence": None}
    ):
        message = f"Campaign run {run_id!r} is not a retained {expected_profile!r} technical smoke with at least two natural cases."
        raise ValueError(message)
    cases = tuple(_case_evidence(batch, case_index, storage=storage) for case_index in batch.case_indices)
    case_count = len(cases)
    if len({case.record["case_input_id"] for case in cases}) != case_count:
        message = f"Technical smoke {expected_profile!r} reused a case-input identity."
        raise RuntimeError(message)
    if len({case.record["simulation_case_id"] for case in cases}) != case_count:
        message = f"Technical smoke {expected_profile!r} reused a simulation identity."
        raise RuntimeError(message)
    return campaign, terminal, workflow, cases


def parse_comsol_exact_version(version_output: str) -> str:
    """Return the exact COMSOL runtime version from bounded command output."""
    if not isinstance(version_output, str) or not version_output.strip():
        message = "COMSOL version output must be non-empty text."
        raise ValueError(message)
    versions = tuple(dict.fromkeys(_COMSOL_EXACT_VERSION_PATTERN.findall(version_output)))
    if len(versions) != 1:
        message = f"COMSOL version output must contain exactly one full runtime version; found {list(versions)}."
        raise ValueError(message)
    return versions[0]


def _output_identity(campaign: config_service.CampaignConfig) -> tuple[str, str, str]:
    """Return one profile, mapping, and template identity for a campaign."""
    identities = {
        (
            batch.profile.id,
            mapping_contract.mapping_contract_sha256(
                batch.profile.id,
                batch.scientific_values["output_contract"],
            ),
            batch.template_sha256,
        )
        for batch in campaign.batches
    }
    if len(identities) != 1:
        message = "One campaign must resolve exactly one output mapping and template identity."
        raise RuntimeError(message)
    return next(iter(identities))


def _technical_smoke_campaign_for(
    campaign: config_service.CampaignConfig,
) -> config_service.CampaignConfig:
    """Resolve the unique maintained technical-smoke contract for one profile."""
    if campaign.campaign_purpose == TECHNICAL_SMOKE_PURPOSE:
        smoke = campaign
    else:
        candidates = [
            candidate
            for candidate in config_service.discover_campaign_configs(
                common.paths.get_project_root(),
                require_executable=True,
            )
            if candidate.campaign_purpose == TECHNICAL_SMOKE_PURPOSE and candidate.profile.id == campaign.profile.id
        ]
        if len(candidates) != 1:
            message = (
                "Production readiness requires exactly one maintained technical-smoke "
                f"campaign for profile {campaign.profile.id!r}; discovered {len(candidates)}."
            )
            raise ValueError(message)
        smoke = candidates[0]
    valid_batch = len(smoke.batches) == 1 and smoke.batches[0].sampling_regime == "natural"
    if not valid_batch or smoke.total_case_count < _MINIMUM_CONTRASTING_CASES:
        message = f"Technical-smoke contract for profile {campaign.profile.id!r} must contain one natural batch with at least two required cases."
        raise ValueError(message)
    return smoke


def build_technical_smoke_evidence_context(
    campaign_path: Path | str,
    *,
    comsol_version_output: str,
) -> dict[str, Any]:
    """Resolve the profile-scoped identity required from technical-smoke evidence."""
    campaign = config_service.load_campaign_config(campaign_path, require_executable=True)
    smoke = _technical_smoke_campaign_for(campaign)
    requested_identity = _output_identity(campaign)
    smoke_identity = _output_identity(smoke)
    if requested_identity != smoke_identity:
        message = (
            f"Current {campaign.profile.id!r} operation and its maintained technical smoke do not resolve the same mapping and template contract."
        )
        raise RuntimeError(message)
    simulation_profile, contract_sha256, template_sha256 = requested_identity
    return {
        "simulation_profile": simulation_profile,
        "mapping_contract_sha256": contract_sha256,
        "template_sha256": template_sha256,
        "comsol_version": parse_comsol_exact_version(comsol_version_output),
        "verifier_schema_kind": TECHNICAL_SMOKE_EVIDENCE_SCHEMA_KIND,
        "verifier_schema_version": TECHNICAL_SMOKE_EVIDENCE_SCHEMA_VERSION,
        "technical_smoke_contract_sha256": smoke.campaign_digest,
        "technical_smoke_campaign_id": smoke.campaign_id,
        "required_case_count": smoke.total_case_count,
    }


def evaluate_technical_smoke_evidence(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one technical-smoke evidence record against current semantics."""
    if report.get("schema_kind") != TECHNICAL_SMOKE_EVIDENCE_SCHEMA_KIND:
        return {"valid": False, "classification": "malformed", "reasons": ["schema kind is invalid"]}
    stale_reasons: list[str] = []
    if report.get("schema_version") != expected.get("verifier_schema_version"):
        stale_reasons.append("technical-smoke verifier version differs")
    if report.get("simulation_profile") != expected.get("simulation_profile"):
        stale_reasons.append("simulation profile differs")
    if report.get("mapping_contract_sha256") != expected.get("mapping_contract_sha256"):
        stale_reasons.append("output mapping contract changed")
    template = report.get("template")
    if not isinstance(template, Mapping) or template.get("sha256") != expected.get("template_sha256"):
        stale_reasons.append("template SHA-256 changed")
    comsol = report.get("comsol")
    if not isinstance(comsol, Mapping) or comsol.get("exact_version") != expected.get("comsol_version"):
        stale_reasons.append("COMSOL version changed")
    if report.get("technical_smoke_contract_sha256") != expected.get("technical_smoke_contract_sha256"):
        stale_reasons.append("technical-smoke campaign contract changed")
    if report.get("technical_smoke_campaign_id") != expected.get("technical_smoke_campaign_id"):
        stale_reasons.append("technical-smoke campaign identity changed")
    if report.get("required_case_count") != expected.get("required_case_count"):
        stale_reasons.append("technical-smoke required case count changed")
    if stale_reasons:
        return {"valid": False, "classification": "stale", "reasons": stale_reasons}
    cases = report.get("cases")
    failure_reasons: list[str] = []
    if report.get("status") != "technical_smoke_complete":
        failure_reasons.append(f"technical-smoke status is {report.get('status')!r}")
    if not isinstance(cases, list) or len(cases) != expected.get("required_case_count"):
        failure_reasons.append("technical-smoke evidence does not contain every required case")
    elif any(
        not isinstance(case, Mapping)
        or set(case)
        != {
            "case_id",
            "case_input_id",
            "simulation_case_id",
            "hdf5_sha256",
            "publication_provenance_sha256",
        }
        or not isinstance(case.get("case_id"), str)
        or not case.get("case_id")
        or not isinstance(case.get("case_input_id"), str)
        or not case.get("case_input_id")
        or not isinstance(case.get("simulation_case_id"), str)
        or not case.get("simulation_case_id")
        or re.fullmatch(r"[0-9a-f]{64}", str(case.get("hdf5_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(case.get("publication_provenance_sha256"))) is None
        for case in cases
    ):
        failure_reasons.append("technical-smoke case success evidence is malformed")
    elif (
        len({str(case["case_id"]) for case in cases}) != len(cases)
        or len({str(case["case_input_id"]) for case in cases}) != len(cases)
        or len({str(case["simulation_case_id"]) for case in cases}) != len(cases)
    ):
        failure_reasons.append("technical-smoke case success identities are not unique")
    if failure_reasons:
        return {"valid": False, "classification": "failed", "reasons": failure_reasons}
    return {"valid": True, "classification": "valid", "reasons": []}


def technical_smoke_evidence_path(
    campaign_run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the canonical profile evidence path alongside campaign metadata."""
    return (
        campaign_evidence.campaign_run_directory(
            campaign_run_id,
            storage_root=storage_root,
        )
        / campaign_evidence.TECHNICAL_SMOKE_EVIDENCE_FILENAME
    )


def load_technical_smoke_evidence(
    path: Path | str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Load one self-digested immutable technical-smoke evidence record."""
    evidence_path = Path(path).expanduser().resolve()
    report = _json_object(evidence_path, label="technical-smoke evidence")
    if set(report) != _TECHNICAL_SMOKE_EVIDENCE_KEYS:
        message = f"Technical-smoke evidence schema is invalid: {evidence_path}"
        raise ValueError(message)
    run_id = report.get("campaign_run_id")
    if not isinstance(run_id, str):
        message = f"Technical-smoke evidence campaign-run identity is malformed: {evidence_path}"
        raise TypeError(message)
    if storage_root is not None and evidence_path != technical_smoke_evidence_path(run_id, storage_root=storage_root):
        message = f"Technical-smoke evidence is outside its canonical campaign path: {evidence_path}"
        raise ValueError(message)
    template = report.get("template")
    comsol = report.get("comsol")
    cases = report.get("cases")
    if (
        report.get("schema_kind") != TECHNICAL_SMOKE_EVIDENCE_SCHEMA_KIND
        or not isinstance(report.get("schema_version"), int)
        or isinstance(report.get("schema_version"), bool)
        or not isinstance(report.get("status"), str)
        or not isinstance(report.get("recorded_at"), str)
        or not report.get("recorded_at")
        or not isinstance(report.get("simulation_profile"), str)
        or not report.get("simulation_profile")
        or re.fullmatch(r"[0-9a-f]{64}", str(report.get("mapping_contract_sha256"))) is None
        or not isinstance(template, dict)
        or not isinstance(template.get("relative_path"), str)
        or not template.get("relative_path")
        or re.fullmatch(r"[0-9a-f]{64}", str(template.get("sha256"))) is None
        or not isinstance(comsol, dict)
        or set(comsol) != {"exact_version", "version_output"}
        or not isinstance(comsol.get("exact_version"), str)
        or not isinstance(comsol.get("version_output"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(report.get("technical_smoke_contract_sha256"))) is None
        or not isinstance(report.get("technical_smoke_campaign_id"), str)
        or not report.get("technical_smoke_campaign_id")
        or re.fullmatch(r"[0-9a-f]{40}", str(report.get("git_commit"))) is None
        or not isinstance(report.get("required_case_count"), int)
        or isinstance(report.get("required_case_count"), bool)
        or report.get("required_case_count", 0) < _MINIMUM_CONTRASTING_CASES
        or not isinstance(cases, list)
        or re.fullmatch(r"[0-9a-f]{64}", str(report.get("workflow_gate_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(report.get("evidence_digest"))) is None
    ):
        message = f"Technical-smoke evidence fields are malformed: {evidence_path}"
        raise ValueError(message)
    try:
        reported_version = parse_comsol_exact_version(comsol["version_output"])
    except ValueError as error:
        message = f"Technical-smoke COMSOL version evidence is malformed: {evidence_path}"
        raise ValueError(message) from error
    if reported_version != comsol["exact_version"]:
        message = f"Technical-smoke COMSOL version evidence disagrees internally: {evidence_path}"
        raise ValueError(message)
    payload = dict(report)
    observed_digest = payload.pop("evidence_digest")
    if common.serialization.canonical_json_sha256(payload) != observed_digest:
        message = f"Technical-smoke evidence self-digest is invalid: {evidence_path}"
        raise RuntimeError(message)
    return report


def discover_technical_smoke_evidence(
    *,
    storage_root: Path | str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Find current evidence for one profile using a bounded campaign scan."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    root = common.paths.get_generation_meta_root(storage_root=storage) / "campaigns"
    if not root.is_dir() or root.is_symlink():
        return {
            "status": "technical_smoke_evidence_missing",
            "reason": "no safe Generation campaign-evidence directory exists",
            "valid_report": None,
            "inspected_reports": [],
        }
    candidates = tuple(
        sorted(
            (
                directory / campaign_evidence.TECHNICAL_SMOKE_EVIDENCE_FILENAME
                for directory in root.iterdir()
                if directory.is_dir() and not directory.is_symlink() and (directory / campaign_evidence.TECHNICAL_SMOKE_EVIDENCE_FILENAME).is_file()
            ),
            reverse=True,
        )
    )
    inspected: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            report = load_technical_smoke_evidence(candidate, storage_root=storage)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            inspected.append({"path": str(candidate), "classification": "malformed", "reasons": [str(error)]})
            continue
        if report.get("simulation_profile") != expected.get("simulation_profile"):
            continue
        evaluation = evaluate_technical_smoke_evidence(report, expected)
        inspected.append({"path": str(candidate), **evaluation})
        if evaluation["valid"]:
            return {
                "status": "technical_smoke_evidence_valid",
                "reason": None,
                "valid_report": str(candidate),
                "inspected_reports": inspected,
            }
    relevant = [record for record in inspected if record.get("classification") != "malformed"]
    if not relevant:
        status = "technical_smoke_evidence_missing"
        reason = "no completed technical-smoke evidence exists for the current simulation profile"
    else:
        status = (
            "technical_smoke_evidence_invalid"
            if any(record["classification"] == "failed" for record in relevant)
            else "technical_smoke_evidence_stale"
        )
        reason = "; ".join(dict.fromkeys(reason for record in relevant for reason in record["reasons"]))
    return {
        "status": status,
        "reason": reason,
        "valid_report": None,
        "inspected_reports": inspected,
    }


def technical_smoke_evidence_status(
    campaign_path: Path | str,
    *,
    storage_root: Path | str,
    comsol_version_output: str,
) -> dict[str, Any]:
    """Return current technical-smoke evidence status for one selected profile."""
    expected = build_technical_smoke_evidence_context(
        campaign_path,
        comsol_version_output=comsol_version_output,
    )
    return {
        "expected_identity": expected,
        **discover_technical_smoke_evidence(storage_root=storage_root, expected=expected),
    }


def finalize_technical_smoke_evidence(
    campaign_run_id: str,
    *,
    comsol_version_output: str,
    storage_root: Path | str | None = None,
) -> Path:
    """Publish evidence only after every required technical-smoke case succeeds."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    declared = campaign_evidence.campaign_for_run(campaign_run_id, storage_root=storage)
    campaign, terminal, workflow, cases = _validate_campaign(
        campaign_run_id,
        expected_profile=declared.profile.id,
        storage=storage,
    )
    expected = build_technical_smoke_evidence_context(
        campaign.source_path,
        comsol_version_output=comsol_version_output,
    )
    if len(cases) != expected["required_case_count"]:
        message = "Completed technical-smoke evidence does not cover every required campaign case."
        raise RuntimeError(message)
    path = technical_smoke_evidence_path(campaign_run_id, storage_root=storage)
    if path.exists():
        report = load_technical_smoke_evidence(path, storage_root=storage)
        evaluation = evaluate_technical_smoke_evidence(report, expected)
        if not evaluation["valid"]:
            message = f"Existing immutable technical-smoke evidence is not current: {evaluation['reasons']}"
            raise ValueError(message)
        return path
    payload: dict[str, Any] = {
        "schema_kind": TECHNICAL_SMOKE_EVIDENCE_SCHEMA_KIND,
        "schema_version": TECHNICAL_SMOKE_EVIDENCE_SCHEMA_VERSION,
        "status": "technical_smoke_complete",
        "recorded_at": _utc_now(),
        "simulation_profile": expected["simulation_profile"],
        "mapping_contract_sha256": expected["mapping_contract_sha256"],
        "template": {
            "relative_path": campaign.template_relative_path,
            "sha256": expected["template_sha256"],
        },
        "comsol": {
            "exact_version": expected["comsol_version"],
            "version_output": comsol_version_output[:4096],
        },
        "technical_smoke_contract_sha256": expected["technical_smoke_contract_sha256"],
        "technical_smoke_campaign_id": expected["technical_smoke_campaign_id"],
        "campaign_run_id": campaign_run_id,
        "git_commit": terminal["git_commit"],
        "required_case_count": expected["required_case_count"],
        "cases": [
            {
                "case_id": case.record["case_id"],
                "case_input_id": case.record["case_input_id"],
                "simulation_case_id": case.record["simulation_case_id"],
                "hdf5_sha256": case.record["hdf5"]["sha256"],
                "publication_provenance_sha256": case.record["publication"]["provenance_sha256"],
            }
            for case in cases
        ],
        "workflow_gate_sha256": workflow["workflow_gate_sha256"],
    }
    payload["evidence_digest"] = common.serialization.canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    common.serialization.atomic_write_json(path, payload)
    load_technical_smoke_evidence(path, storage_root=storage)
    return path


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    """Return one exact numeric-array identity independent of HDF5 layout."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": digest.hexdigest()}


def _difference_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> dict[str, Any]:
    """Return observed same-grid differences without applying a pass tolerance."""
    if left.shape != right.shape or left.shape != (y_axis.size, x_axis.size):
        message = "Template-equivalence fields do not share the configured grid."
        raise ValueError(message)
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    absolute = np.abs(difference)
    flat_index = int(np.argmax(absolute))
    y_index, x_index = np.unravel_index(flat_index, difference.shape)
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), np.finfo(np.float64).tiny)
    relative = absolute / denominator
    return {
        "maximum_absolute_difference": float(absolute[y_index, x_index]),
        "maximum_relative_difference": float(np.max(relative)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "l2_difference": float(np.linalg.norm(difference.ravel(), ord=2)),
        "maximum_difference_location": {
            "x_index": int(x_index),
            "y_index": int(y_index),
            "x_m": float(x_axis[x_index]),
            "y_m": float(y_axis[y_index]),
        },
        "acceptance_tolerance": None,
    }


def _variation_report(
    cases: Sequence[_CaseEvidence],
    *,
    profile_id: str,
) -> dict[str, Any]:
    """Require configured technical cases to include contrasting inputs."""
    if len(cases) < _MINIMUM_CONTRASTING_CASES:
        message = f"Technical smoke for {profile_id!r} has fewer than {_MINIMUM_CONTRASTING_CASES} contrasting cases."
        raise ValueError(message)
    reference = cases[0]
    contrasts = cases[1:]
    spatial_differences = {
        name: max(float(np.max(np.abs(reference.static[name] - case.static[name]))) for case in contrasts) for name in _SHARED_FIELD_NAMES
    }
    required_spatial = ("p_in_bc", "Kxx", "eps_bed")
    if any(spatial_differences[name] == 0.0 for name in required_spatial):
        message = f"Configured {profile_id} smoke did not vary required airflow inputs {required_spatial}."
        raise RuntimeError(message)
    input_identities = {common.serialization.canonical_json_sha256(case.record["input_files"]) for case in cases}
    report: dict[str, Any] = {
        "case_count": len(cases),
        "spatial_maximum_absolute_differences": spatial_differences,
        "input_hash_sets_distinct": len(input_identities) > 1,
    }
    if profile_id == profiles.TRANSIENT_DRYING_PROFILE:
        if reference.schedule is None or any(case.schedule is None for case in contrasts):
            message = "Transient smoke cases have no canonical schedules."
            raise RuntimeError(message)
        schedule_difference = max(float(np.max(np.abs(reference.schedule - case.schedule))) for case in contrasts if case.schedule is not None)
        moisture_difference = max(float(np.max(np.abs(reference.static["X_0_db_field"] - case.static["X_0_db_field"]))) for case in contrasts)
        common_scalar_names = set(reference.scalars)
        for case in contrasts:
            common_scalar_names.intersection_update(case.scalars)
        changed_scalars = sorted(name for name in common_scalar_names if any(reference.scalars[name] != case.scalars[name] for case in contrasts))
        if schedule_difference == 0.0 or moisture_difference == 0.0 or not changed_scalars:
            message = "Configured transient smoke did not vary schedule, initial moisture, and a case-dependent scalar."
            raise RuntimeError(message)
        report.update(
            {
                "schedule_maximum_absolute_difference": schedule_difference,
                "initial_moisture_maximum_absolute_difference": (moisture_difference),
                "changed_case_dependent_scalars": changed_scalars,
                "scalar_handoff_consumed": True,
            }
        )
    return report


def _paired_smoke_batches(
    steady_campaign: config_service.CampaignConfig,
    transient_campaign: config_service.CampaignConfig,
) -> tuple[
    config_service.GenerationConfig,
    config_service.GenerationConfig,
]:
    """Return two structurally paired technical-smoke batches."""
    steady_batch = steady_campaign.batches[0]
    transient_batch = transient_campaign.batches[0]
    if (
        steady_campaign.paired_equivalence_seed is None
        or steady_campaign.paired_equivalence_seed != transient_campaign.paired_equivalence_seed
        or steady_batch.material_family != transient_batch.material_family
        or steady_batch.material_role != transient_batch.material_role
        or steady_batch.case_indices != transient_batch.case_indices
    ):
        message = "Technical-smoke campaigns must pair one material, role, case inventory, and configured equivalence seed across both profiles."
        raise ValueError(message)
    return steady_batch, transient_batch


def _equivalence_report(
    steady_cases: Sequence[_CaseEvidence],
    transient_cases: Sequence[_CaseEvidence],
    *,
    steady_config: config_service.GenerationConfig,
    transient_config: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Compare paired template airflow outputs on their configured grid."""
    steady_grid = steady_config.scientific_values["grid"]
    transient_grid = transient_config.scientific_values["grid"]
    if steady_grid != transient_grid:
        message = "Paired technical-smoke campaigns resolve different grids."
        raise ValueError(message)
    nx = int(steady_grid["nx"])
    ny = int(steady_grid["ny"])
    x_axis = np.linspace(
        0.0,
        float(steady_grid["Lx"]),
        nx,
        dtype=np.float64,
    )
    y_axis = np.linspace(
        0.0,
        float(steady_grid["Ly"]),
        ny,
        dtype=np.float64,
    )
    pairs: list[dict[str, Any]] = []
    for steady, transient in zip(
        steady_cases,
        transient_cases,
        strict=True,
    ):
        shared = {}
        for name in _SHARED_FIELD_NAMES:
            if not np.array_equal(steady.static[name], transient.static[name]):
                message = f"Paired templates did not consume identical shared input {name!r}."
                raise RuntimeError(message)
            shared[name] = _array_identity(steady.static[name])
        if steady.stationary_fixed != transient.stationary_fixed:
            message = "Paired templates did not bind identical stationary fixed values."
            raise RuntimeError(message)
        pair_identity = common.serialization.canonical_json_sha256(
            {
                "shared_fields": shared,
                "stationary_fixed": steady.stationary_fixed,
            }
        )
        pairs.append(
            {
                "case_index": steady.record["case_index"],
                "steady_simulation_case_id": (steady.record["simulation_case_id"]),
                "transient_simulation_case_id": (transient.record["simulation_case_id"]),
                "paired_input_identity": pair_identity,
                "shared_input_identities": shared,
                "stationary_fixed": steady.stationary_fixed,
                "observed_differences": {
                    name: _difference_metrics(
                        steady.static[name],
                        transient.static[name],
                        x_axis=x_axis,
                        y_axis=y_axis,
                    )
                    for name in _AIRFLOW_FIELD_NAMES
                },
            }
        )
    dx = float(steady_grid["dx"])
    dy = float(steady_grid["dy"])
    return {
        "status": "observed_no_acceptance_threshold",
        "grid_shape": [ny, nx],
        "grid_spacing_m": dx if dx == dy else None,
        "grid_spacing_by_axis_m": {
            "x": dx,
            "y": dy,
        },
        "grid_extent_m": {
            "x": float(steady_grid["Lx"]),
            "y": float(steady_grid["Ly"]),
            "z": float(steady_grid["Lz"]),
        },
        "pair_count": len(pairs),
        "pairs": pairs,
        "acceptance_tolerance": None,
    }


def _residual_metrics(values: np.ndarray, times: np.ndarray, *, unit: str) -> dict[str, Any]:
    """Return observed finite residual statistics without an acceptance threshold."""
    absolute = np.abs(values)
    index = int(np.argmax(absolute))
    return {
        "maximum_absolute": float(absolute[index]),
        "time_of_maximum_absolute_h": float(times[index]),
        "final": float(values[-1]),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "l2": float(np.linalg.norm(values, ord=2)),
        "unit": unit,
        "acceptance_tolerance": None,
    }


def _mass_balance_case(case: _CaseEvidence) -> dict[str, Any]:
    """Report observed differential/integral balance and reconstructed moisture."""
    if case.global_values is None or not case.initial_state:
        message = "Mass-balance diagnostics require one transient case."
        raise TypeError(message)
    values = case.global_values
    columns = {name: values[:, index] for index, name in enumerate(profiles.GLOBAL_FIELD_NAMES)}
    times_h = columns["t"]
    times_s = times_h * 3600.0
    total_water = columns["m_w_gr"] + columns["m_v_gas"]
    net_boundary = columns["m_dot_v_in"] - columns["m_dot_v_out"]
    derivative = np.gradient(total_water, times_s, edge_order=1)
    differential = derivative - net_boundary
    steps = np.diff(times_s)
    trapezoids = 0.5 * (net_boundary[1:] + net_boundary[:-1]) * steps
    accumulated = np.concatenate((np.asarray([0.0]), np.cumsum(trapezoids)))
    integral = total_water - total_water[0] - accumulated
    stored = columns["mt_mass_balance"]
    f_surf = float(case.scalars["f_surf"])
    rho = case.static["rho_bu_dry"]
    x_initial = case.static["X_0_db_field"]
    water = domain.moisture.granular_water_content(
        case.initial_state["w_surf"],
        case.initial_state["w_int"],
        f_surf,
    )
    reconstructed_x_db = water / rho
    reconstructed_x_wb = reconstructed_x_db / (1.0 + reconstructed_x_db)
    expected_x_wb = x_initial / (1.0 + x_initial)
    return {
        "case_id": case.record["case_id"],
        "simulation_case_id": case.record["simulation_case_id"],
        "stored_mt_mass_balance": _residual_metrics(stored, times_h, unit="kg/s"),
        "differential_total_water_residual": _residual_metrics(differential, times_h, unit="kg/s"),
        "integral_total_water_closure": _residual_metrics(integral, times_h, unit="kg"),
        "initial_reconstruction": {
            "w_gr_formula": "f_surf*w_surf + (1-f_surf)*w_int",
            "maximum_absolute_X_db_error": float(np.max(np.abs(reconstructed_x_db - x_initial))),
            "maximum_absolute_X_wb_error": float(np.max(np.abs(reconstructed_x_wb - expected_x_wb))),
        },
        "sign_validation": {
            "m_dot_evap_minimum_kg_per_s": float(np.min(columns["m_dot_evap"])),
            "m_dot_v_in_minimum_kg_per_s": float(np.min(columns["m_dot_v_in"])),
            "m_dot_v_out_minimum_kg_per_s": float(np.min(columns["m_dot_v_out"])),
            "Q_evap_sign_relation": "Q_evap=-h_fg*m_evap",
            "canonical_nonnegative_mass_flows_validated_by_hdf5": True,
        },
        "acceptance_tolerance": None,
    }


def _source_binding(
    campaigns: Sequence[config_service.CampaignConfig],
) -> dict[str, Any]:
    """Return exact source-file and resolved-campaign identities."""
    repository = common.paths.get_project_root().resolve()
    source_paths = {source_path.resolve() for campaign in campaigns for source_path in campaign.source_files}
    records = []
    for path in sorted(source_paths):
        if not path.is_file() or path.is_symlink():
            message = f"Required smoke source is missing or unsafe: {path}"
            raise FileNotFoundError(message)
        try:
            relative_path = path.relative_to(repository).as_posix()
        except ValueError as error:
            message = f"Smoke source escapes the configured project root: {path}"
            raise ValueError(message) from error
        records.append(
            {
                "relative_path": relative_path,
                "sha256": common.serialization.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    campaign_records = [
        {
            "campaign_id": campaign.campaign_id,
            "campaign_digest": campaign.campaign_digest,
            "simulation_profile": campaign.profile.id,
        }
        for campaign in campaigns
    ]
    return {
        "files": records,
        "resolved_campaigns": campaign_records,
        "bundle_digest": common.serialization.canonical_json_sha256(
            {
                "files": records,
                "resolved_campaigns": campaign_records,
            }
        ),
    }


def _template_binding(
    campaigns: Sequence[config_service.CampaignConfig],
) -> dict[str, Any]:
    """Return exact configured template identities."""
    return {
        campaign.profile.id: {
            "relative_path": campaign.template_relative_path,
            "sha256": campaign.template_sha256,
            "size_bytes": campaign.template_path.stat().st_size,
        }
        for campaign in campaigns
    }


def _repository_commit() -> str:
    """Return the current exact repository commit."""
    result = subprocess.run(  # noqa: S603 -- fixed Git argument vector
        ["git", "-C", str(common.paths.get_project_root()), "rev-parse", "HEAD"],  # noqa: S607 -- site PATH owns Git
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != _GIT_SHA_LENGTH or any(character not in "0123456789abcdef" for character in commit):
        message = f"Repository commit is malformed: {commit!r}."
        raise ValueError(message)
    return commit


def _campaign_record(
    run_id: str,
    campaign: config_service.CampaignConfig,
    terminal: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> dict[str, Any]:
    """Return campaign, terminal, mapping, and lifecycle identity evidence."""
    return {
        "campaign_run_id": run_id,
        "campaign_id": campaign.campaign_id,
        "campaign_name": campaign.campaign_name,
        "campaign_purpose": campaign.campaign_purpose,
        "campaign_digest": campaign.campaign_digest,
        "campaign_config": terminal["campaign_config"],
        "campaign_config_sha256": common.serialization.file_sha256(campaign.source_path),
        "profile": campaign.profile.id,
        "selected_batch_ids": [batch["batch_id"] for batch in terminal["batches"]],
        "git_commit": terminal["git_commit"],
        "slurm_job_ids": list(terminal["slurm_job_ids"]),
        "dataset_ids": list(workflow["dataset_ids"]),
        "workflow_gate_sha256": workflow["workflow_gate_sha256"],
        "cpu_source_retention": workflow["cpu_cleanup_complete"],
    }


def _profile_mapping_binding(campaigns: Sequence[config_service.CampaignConfig]) -> dict[str, Any]:
    """Return one explicit export and conditioning binding per profile."""
    result: dict[str, Any] = {}
    for campaign in campaigns:
        batch = campaign.batches[0]
        output = batch.scientific_values["output_contract"]
        conditioning = batch.scientific_values["steady_flow_conditioning"]
        if conditioning is None:
            message = f"Profile {campaign.profile.id!r} has no confirmed steady-flow conditioning."
            raise RuntimeError(message)
        result[campaign.profile.id] = {
            "export_contract_sha256": common.serialization.canonical_json_sha256(output),
            "exports": output["exports"],
            "steady_flow_conditioning": conditioning,
            "steady_flow_conditioning_sha256": common.serialization.canonical_json_sha256(conditioning),
        }
    return result


def _dataset_records(workflows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return exact package hashes, inspections, and two-worker smoke evidence."""
    records: list[dict[str, Any]] = []
    for workflow in workflows:
        for dataset_id, package_hash, inspection, loader_smoke in zip(
            workflow["dataset_ids"],
            workflow["dataset_package_hashes"],
            workflow["package_inspection_results"],
            workflow["loader_smoke_results"],
            strict=True,
        ):
            if set(loader_smoke) != {"workers_0", "workers_2"}:
                message = f"Dataset {dataset_id!r} lacks both required loader-smoke worker counts."
                raise RuntimeError(message)
            records.append(
                {
                    "dataset_id": dataset_id,
                    **package_hash,
                    "inspection": inspection,
                    "loader_smoke": loader_smoke,
                }
            )
    if len({record["dataset_id"] for record in records}) != len(records):
        message = "Technical smoke workflows produced duplicate dataset package IDs."
        raise RuntimeError(message)
    return records


def _paired_comsol_contract(
    campaigns: Sequence[config_service.CampaignConfig],
) -> dict[str, str]:
    """Return one agreed configured COMSOL module, executable, and version."""
    modules = {str(campaign.execution_values["site"]["comsol_module"]) for campaign in campaigns}
    executables = {str(campaign.execution_values["site"]["comsol_executable"]) for campaign in campaigns}
    if len(modules) != 1 or len(executables) != 1:
        message = "Paired technical smoke campaigns must agree on one COMSOL module and executable."
        raise RuntimeError(message)
    module = next(iter(modules))
    executable = next(iter(executables))
    version = preflight_service.configured_module_version(module)
    if version is None:
        message = f"Configured COMSOL module must expose a version suffix: {module!r}."
        raise ValueError(message)
    return {
        "module": module,
        "executable": executable,
        "required_version": version,
    }


def _build_payload(
    steady_run_id: str,
    transient_run_id: str,
    *,
    storage: Path,
    recorded_at: str,
    comsol_version_output: str,
) -> dict[str, Any]:
    """Recompute the complete receipt payload except its self-digest."""
    steady_campaign, steady_terminal, steady_workflow, steady_cases = _validate_campaign(
        steady_run_id,
        expected_profile=profiles.STEADY_FLOW_PROFILE,
        storage=storage,
    )
    transient_campaign, transient_terminal, transient_workflow, transient_cases = _validate_campaign(
        transient_run_id,
        expected_profile=profiles.TRANSIENT_DRYING_PROFILE,
        storage=storage,
    )
    steady_batch, transient_batch = _paired_smoke_batches(
        steady_campaign,
        transient_campaign,
    )
    commits = {steady_terminal["git_commit"], transient_terminal["git_commit"]}
    current_commit = _repository_commit()
    if commits != {current_commit}:
        message = "Technical smoke runs and current repository do not share one exact Git commit."
        raise RuntimeError(message)
    comsol_contract = _paired_comsol_contract((steady_campaign, transient_campaign))
    if not isinstance(comsol_version_output, str) or not preflight_service.reported_version_matches(
        comsol_version_output,
        comsol_contract["required_version"],
    ):
        message = f"Real-smoke receipt version output does not report configured COMSOL version {comsol_contract['required_version']!r}."
        raise ValueError(message)
    source_hosts = {steady_workflow["cpu_source_host"], transient_workflow["cpu_source_host"]}
    if len(source_hosts) != 1:
        message = "Paired technical smoke runs came from different CPU hosts."
        raise RuntimeError(message)
    steady_variation = _variation_report(steady_cases, profile_id=profiles.STEADY_FLOW_PROFILE)
    transient_variation = _variation_report(transient_cases, profile_id=profiles.TRANSIENT_DRYING_PROFILE)
    all_cases = (*steady_cases, *transient_cases)
    profile_mappings = _profile_mapping_binding((steady_campaign, transient_campaign))
    payload: dict[str, Any] = {
        "schema_kind": REAL_SMOKE_SCHEMA_KIND,
        "schema_version": REAL_SMOKE_SCHEMA_VERSION,
        "status": "observations_complete_no_scientific_acceptance_threshold",
        "recorded_at": recorded_at,
        "git_commit": current_commit,
        "material_family_inventory": [steady_batch.material_family],
        "source_binding": _source_binding((steady_campaign, transient_campaign)),
        "templates": _template_binding((steady_campaign, transient_campaign)),
        "profile_mappings": profile_mappings,
        "campaigns": [
            _campaign_record(steady_run_id, steady_campaign, steady_terminal, steady_workflow),
            _campaign_record(transient_run_id, transient_campaign, transient_terminal, transient_workflow),
        ],
        "cases": [case.record for case in all_cases],
        "comsol": {
            "required_version": comsol_contract["required_version"],
            "version_command": [comsol_contract["executable"], "-version"],
            "version_output": comsol_version_output[:4096],
            "source_host": next(iter(source_hosts)),
        },
        "slurm": {
            "steady_job_ids": list(steady_terminal["slurm_job_ids"]),
            "transient_job_ids": list(transient_terminal["slurm_job_ids"]),
            "case_job_ids": [case.record["timing"]["scheduler_job_id"] for case in all_cases],
        },
        "dataset_packages": _dataset_records((steady_workflow, transient_workflow)),
        "cpu_source_retention": {
            "steady": steady_workflow["cpu_cleanup_complete"],
            "transient": transient_workflow["cpu_cleanup_complete"],
            "review_required_before_cleanup": True,
        },
        "template_equivalence": _equivalence_report(
            steady_cases,
            transient_cases,
            steady_config=steady_batch,
            transient_config=transient_batch,
        ),
        "mass_balance": {
            "status": "observed_no_acceptance_threshold",
            "cases": [_mass_balance_case(case) for case in transient_cases],
            "acceptance_tolerance": None,
        },
        "acceptance_tolerances": {
            "template_equivalence": None,
            "mass_balance": None,
        },
    }
    payload["campaigns"][0]["input_variation"] = steady_variation
    payload["campaigns"][1]["input_variation"] = transient_variation
    return payload


def real_smoke_receipt_path(
    steady_run_id: str,
    transient_run_id: str,
    *,
    storage_root: Path | str | None = None,
) -> Path:
    """Return the canonical immutable receipt path for one paired smoke."""
    steady = common.paths.validate_logical_name(steady_run_id, label="steady_campaign_run_id")
    transient = common.paths.validate_logical_name(transient_run_id, label="transient_campaign_run_id")
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    return common.paths.get_generation_meta_root(storage_root=storage) / "real_smoke" / f"{steady}--{transient}.json"


def validate_real_smoke_receipt(
    path: Path | str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Recompute every durable binding and validate one immutable receipt."""
    receipt_path = Path(path).expanduser().resolve()
    receipt = _json_object(receipt_path, label="real runtime-smoke receipt")
    if (
        set(receipt) != _RECEIPT_KEYS
        or receipt.get("schema_kind") != REAL_SMOKE_SCHEMA_KIND
        or receipt.get("schema_version") != REAL_SMOKE_SCHEMA_VERSION
    ):
        message = f"Real runtime-smoke receipt schema is invalid: {receipt_path}"
        raise ValueError(message)
    campaigns = receipt.get("campaigns")
    if not isinstance(campaigns, list) or len(campaigns) != _EXPECTED_PROFILE_COUNT:
        message = "Real runtime-smoke receipt must bind exactly two campaigns."
        raise ValueError(message)
    steady_run_id = campaigns[0].get("campaign_run_id")
    transient_run_id = campaigns[1].get("campaign_run_id")
    if not isinstance(steady_run_id, str) or not isinstance(transient_run_id, str):
        message = "Real runtime-smoke campaign run IDs are missing."
        raise TypeError(message)
    expected_path = real_smoke_receipt_path(steady_run_id, transient_run_id, storage_root=storage_root)
    if receipt_path != expected_path:
        message = f"Real runtime-smoke receipt is outside its canonical immutable path: {receipt_path}"
        raise ValueError(message)
    comsol = receipt.get("comsol")
    recorded_at = receipt.get("recorded_at")
    if not isinstance(comsol, dict) or not isinstance(comsol.get("version_output"), str) or not isinstance(recorded_at, str) or not recorded_at:
        message = "Real runtime-smoke timestamp or COMSOL evidence is malformed."
        raise TypeError(message)
    expected = _build_payload(
        steady_run_id,
        transient_run_id,
        storage=workspace_service.resolve_storage_root(storage_root, create=False),
        recorded_at=recorded_at,
        comsol_version_output=comsol["version_output"],
    )
    digest = common.serialization.canonical_json_sha256(expected)
    expected["receipt_digest"] = digest
    if receipt != expected:
        message = f"Real runtime-smoke receipt no longer matches current artifacts or source: {receipt_path}"
        raise ValueError(message)
    return receipt


def validate_current_real_smoke_receipts(
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Require at least one immutable real-smoke receipt valid for current source."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    root = common.paths.get_generation_meta_root(storage_root=storage) / "real_smoke"
    if not root.is_dir() or root.is_symlink():
        message = f"No safe real runtime-smoke receipt directory exists: {root}"
        raise FileNotFoundError(message)
    candidates = tuple(sorted(path for path in root.glob("*.json") if path.is_file() and not path.is_symlink()))
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            receipt = validate_real_smoke_receipt(candidate, storage_root=storage)
        except (OSError, TypeError, ValueError) as error:  # noqa: PERF203 -- isolate stale receipts
            invalid.append({"path": str(candidate), "error": str(error)})
        else:
            valid.append(
                {
                    "path": str(candidate),
                    "receipt_digest": str(receipt["receipt_digest"]),
                    "campaign_run_ids": [str(campaign["campaign_run_id"]) for campaign in receipt["campaigns"]],
                }
            )
    if not valid:
        message = f"No real runtime-smoke receipt is valid for current source; candidates={len(candidates)}, invalid={invalid}."
        raise ValueError(message)
    return {
        "schema_kind": "vp2_current_real_runtime_smoke_receipts",
        "schema_version": 1,
        "status": "current_runtime_evidence_complete",
        "valid_receipts": valid,
        "invalid_or_stale_receipts": invalid,
    }


def finalize_real_smoke(
    steady_run_id: str,
    transient_run_id: str,
    *,
    comsol_version_output: str,
    storage_root: Path | str | None = None,
) -> Path:
    """Write or validate the immutable paired real-runtime smoke receipt."""
    storage = workspace_service.resolve_storage_root(storage_root, create=False)
    path = real_smoke_receipt_path(steady_run_id, transient_run_id, storage_root=storage)
    if path.exists():
        validate_real_smoke_receipt(path, storage_root=storage)
        return path
    payload = _build_payload(
        steady_run_id,
        transient_run_id,
        storage=storage,
        recorded_at=_utc_now(),
        comsol_version_output=comsol_version_output,
    )
    payload["receipt_digest"] = common.serialization.canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    common.serialization.atomic_write_json(path, payload)
    validate_real_smoke_receipt(path, storage_root=storage)
    return path
