"""
===============================================================================
generation_smoke.py
===============================================================================
Validate and bind one two-profile native technical runtime smoke.
Responsibilities:
  - Require two terminal steady and two terminal transient Slurm cases
  - Bind retained inputs, exports, HDF5, packages, loaders, and source identities
  - Report paired airflow differences and transient mass-balance observations
  - Write and revalidate one immutable runtime-validation receipt
Design principles:
  - Existing campaign and all-workflow receipts remain the lifecycle authorities
  - Equivalence and mass balance are reported without invented pass tolerances
  - Current source, template, mapping, material, and decision identities are bound
This module does NOT:
  - Run COMSOL, submit Slurm jobs, infer export mappings, or launch production
  - Treat a technical smoke as experimental validation of scientific priors
===============================================================================
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import h5py
import numpy as np

from src import common, domain

from . import generation_campaign_runtime as campaign_runtime
from . import generation_config as config_service
from . import generation_materials as materials
from . import generation_profiles as profiles
from . import generation_runtime as runtime_service
from . import generation_storage as storage_service
from . import generation_workflow as workflow_service
from . import generation_workspace as workspace_service

if TYPE_CHECKING:
    from collections.abc import Sequence

REAL_SMOKE_SCHEMA_KIND: Final = "vp2_real_runtime_smoke"
REAL_SMOKE_SCHEMA_VERSION: Final = 1
STEADY_SMOKE_CAMPAIGN_ID: Final = "steady_flow_technical_runtime_smoke_v1"
TRANSIENT_SMOKE_CAMPAIGN_ID: Final = "transient_drying_technical_runtime_smoke_v1"
TECHNICAL_SMOKE_PURPOSE: Final = "technical_runtime_smoke"
_SHARED_FIELD_NAMES: Final = ("Kxx", "Kxy", "Kyy", "eps_bed", "p_in_bc")
_AIRFLOW_FIELD_NAMES: Final = ("p", "u", "v")
_EXPECTED_CASE_COUNT: Final = 2
_EXPECTED_PROFILE_COUNT: Final = 2
_GIT_SHA_LENGTH: Final = 40
_SHA256_LENGTH: Final = 64
_SOURCE_RELATIVE_PATHS: Final = (
    "configs/generation/sources.yaml",
    "configs/generation/registry.yaml",
    "configs/generation/common.yaml",
    "configs/generation/operations/fixed_bed.yaml",
    *(f"configs/generation/materials/{family}.yaml" for family in materials.MATERIAL_FAMILIES),
    "configs/generation/profiles/steady_flow.yaml",
    "configs/generation/profiles/transient_drying.yaml",
    "configs/generation/campaigns/steady_flow/family_generalization.yaml",
    "configs/generation/campaigns/transient_drying/family_generalization.yaml",
    "configs/generation/campaigns/steady_flow/technical_smoke.yaml",
    "configs/generation/campaigns/transient_drying/technical_smoke.yaml",
)
_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "status",
        "recorded_at",
        "git_commit",
        "decision_source",
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


def _csv_header(path: Path, *, delimiter: str) -> list[str]:
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
        header = _csv_header(path, delimiter=str(contract["delimiter"]))
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
    """Validate retained export bytes and report actual versus mapped headers."""
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
        header = _csv_header(path, delimiter=str(contract["delimiter"]))
        if contract["mapping_state"] == "mapping_probe_required":
            message = f"Real smoke cannot bind unresolved required mapping role {role!r}."
            raise RuntimeError(message)
        expected_sources = list(contract["columns"].values())
        missing = sorted(set(expected_sources).difference(header))
        if missing:
            message = f"Export mapping for {role!r} is missing actual headers {missing}: {path}"
            raise ValueError(message)
        records.append(
            {
                "relative_path": relative_path,
                "role": role,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "actual_header": header,
                "configured_columns": dict(contract["columns"]),
                "mapping_state_before_runtime_receipt": contract["mapping_state"],
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
    case_payload = _json_object(processed / "case.json", label="smoke case provenance")
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
    if not config.execution_values["retention"]["retain_raw_csv"] or not config.execution_values["retention"]["retain_solved_model"]:
        message = "Technical smoke execution must retain raw CSV and solved model evidence."
        raise RuntimeError(message)
    solved = processed / "solved.mph"
    if not solved.is_file() or solved.is_symlink() or solved.stat().st_size <= 0:
        message = f"Technical smoke solved model was not retained: {solved}"
        raise FileNotFoundError(message)
    hdf5_path = processed / "case.h5"
    static, fixed, scalars, schedule, global_values, initial_state, exports, hdf5 = _load_hdf5(
        hdf5_path,
        profile_id=config.profile.id,
    )
    input_records = _input_inventory(raw / "raw_csv" / "inputs", case_payload)
    expected_inputs = {"fields.csv"} if config.profile.id == profiles.STEADY_FLOW_PROFILE else {"fields.csv", "scalars.csv", "schedule.csv"}
    if {record["filename"] for record in input_records} != expected_inputs:
        message = f"Smoke input adapter membership is invalid for profile {config.profile.id!r}."
        raise ValueError(message)
    export_records = _export_inventory(
        raw / "raw_csv" / "exports",
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
    expected_campaign_id: str,
    expected_profile: str,
    storage: Path,
) -> tuple[config_service.CampaignConfig, dict[str, Any], dict[str, Any], tuple[_CaseEvidence, ...]]:
    """Validate one retained two-case technical smoke workflow."""
    workflow = workflow_service.validate_completed_workflow(run_id, storage_root=storage)
    campaign = campaign_runtime.campaign_for_run(run_id, storage_root=storage)
    terminal = campaign_runtime.validate_terminal_campaign(run_id, storage_root=storage)
    if (
        campaign.campaign_id != expected_campaign_id
        or campaign.campaign_purpose != TECHNICAL_SMOKE_PURPOSE
        or campaign.profile.id != expected_profile
        or len(campaign.batches) != 1
        or len(campaign.batches[0].case_indices) != _EXPECTED_CASE_COUNT
        or campaign.batches[0].sampling_regime != "natural"
        or campaign.batches[0].material_family != "lentil"
        or workflow["cleanup_requested"] is not False
        or workflow["cpu_cleanup_complete"] != {"status": "skipped_by_request", "evidence": None}
    ):
        message = f"Campaign run {run_id!r} is not the canonical retained two-case {expected_profile} smoke."
        raise ValueError(message)
    cases = tuple(_case_evidence(campaign.batches[0], case_index, storage=storage) for case_index in campaign.batches[0].case_indices)
    if len({case.record["case_input_id"] for case in cases}) != _EXPECTED_CASE_COUNT:
        message = f"Technical smoke {expected_profile!r} reused a case-input identity."
        raise RuntimeError(message)
    if len({case.record["simulation_case_id"] for case in cases}) != _EXPECTED_CASE_COUNT:
        message = f"Technical smoke {expected_profile!r} reused a simulation identity."
        raise RuntimeError(message)
    return campaign, terminal, workflow, cases


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
        message = "Template-equivalence fields do not share the canonical grid."
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


def _variation_report(cases: Sequence[_CaseEvidence], *, profile_id: str) -> dict[str, Any]:
    """Require the two technical cases to exercise distinct authoritative inputs."""
    first, second = cases
    spatial_differences = {name: float(np.max(np.abs(first.static[name] - second.static[name]))) for name in _SHARED_FIELD_NAMES}
    required_spatial = ("p_in_bc", "Kxx", "eps_bed")
    if any(spatial_differences[name] == 0.0 for name in required_spatial):
        message = f"Two-case {profile_id} smoke did not vary required airflow inputs {required_spatial}."
        raise RuntimeError(message)
    report: dict[str, Any] = {
        "spatial_maximum_absolute_differences": spatial_differences,
        "input_hash_sets_distinct": first.record["input_files"] != second.record["input_files"],
    }
    if profile_id == profiles.TRANSIENT_DRYING_PROFILE:
        if first.schedule is None or second.schedule is None:
            message = "Transient smoke cases have no canonical schedules."
            raise RuntimeError(message)
        schedule_difference = float(np.max(np.abs(first.schedule - second.schedule)))
        moisture_difference = float(np.max(np.abs(first.static["X_0_db_field"] - second.static["X_0_db_field"])))
        fixed = set(profiles.STATIONARY_FIXED_FIELDS) | {"f_wet_dm_max"}
        changed_scalars = sorted(
            name for name in first.scalars.keys() & second.scalars.keys() if name not in fixed and first.scalars[name] != second.scalars[name]
        )
        if schedule_difference == 0.0 or moisture_difference == 0.0 or not changed_scalars:
            message = "Two-case transient smoke did not vary schedule, initial moisture, and a case-dependent scalar."
            raise RuntimeError(message)
        report.update(
            {
                "schedule_maximum_absolute_difference": schedule_difference,
                "initial_moisture_maximum_absolute_difference": moisture_difference,
                "changed_case_dependent_scalars": changed_scalars,
                "scalar_handoff_consumed": True,
            }
        )
    return report


def _equivalence_report(
    steady_cases: Sequence[_CaseEvidence],
    transient_cases: Sequence[_CaseEvidence],
) -> dict[str, Any]:
    """Compare paired template airflow outputs while requiring exact shared inputs."""
    x_axis = np.linspace(0.0, 1.2, 401, dtype=np.float64)
    y_axis = np.linspace(0.0, 0.75, 251, dtype=np.float64)
    pairs: list[dict[str, Any]] = []
    for steady, transient in zip(steady_cases, transient_cases, strict=True):
        shared = {}
        for name in _SHARED_FIELD_NAMES:
            if not np.array_equal(steady.static[name], transient.static[name]):
                message = f"Paired templates did not consume identical shared input {name!r}."
                raise RuntimeError(message)
            shared[name] = _array_identity(steady.static[name])
        if steady.stationary_fixed != transient.stationary_fixed:
            message = "Paired templates did not bind identical stationary fixed values."
            raise RuntimeError(message)
        pair_identity = common.serialization.canonical_json_sha256({"shared_fields": shared, "stationary_fixed": steady.stationary_fixed})
        pairs.append(
            {
                "case_index": steady.record["case_index"],
                "steady_simulation_case_id": steady.record["simulation_case_id"],
                "transient_simulation_case_id": transient.record["simulation_case_id"],
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
    return {
        "status": "observed_no_acceptance_threshold",
        "grid_shape": [251, 401],
        "grid_spacing_m": 0.003,
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


def _source_binding() -> dict[str, Any]:
    """Return exact repository scientific-source identities for readiness binding."""
    repository = common.paths.get_project_root().resolve()
    records = []
    for relative in _SOURCE_RELATIVE_PATHS:
        path = (repository / relative).resolve()
        if not path.is_file() or path.is_symlink():
            message = f"Required smoke source is missing or unsafe: {path}"
            raise FileNotFoundError(message)
        records.append(
            {
                "relative_path": relative,
                "sha256": common.serialization.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "files": records,
        "bundle_digest": common.serialization.canonical_json_sha256(records),
    }


def _template_binding() -> dict[str, Any]:
    """Return exact current template identities."""
    return {
        profile_id: {
            "relative_path": profile.template_relative_path,
            "sha256": profile.template_sha256,
            "size_bytes": profile.template_path.stat().st_size,
        }
        for profile_id in profiles.available_profiles()
        for profile in (profiles.get_profile(profile_id),)
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
        expected_campaign_id=STEADY_SMOKE_CAMPAIGN_ID,
        expected_profile=profiles.STEADY_FLOW_PROFILE,
        storage=storage,
    )
    transient_campaign, transient_terminal, transient_workflow, transient_cases = _validate_campaign(
        transient_run_id,
        expected_campaign_id=TRANSIENT_SMOKE_CAMPAIGN_ID,
        expected_profile=profiles.TRANSIENT_DRYING_PROFILE,
        storage=storage,
    )
    commits = {steady_terminal["git_commit"], transient_terminal["git_commit"]}
    current_commit = _repository_commit()
    if commits != {current_commit}:
        message = "Technical smoke runs and current repository do not share one exact Git commit."
        raise RuntimeError(message)
    if not isinstance(comsol_version_output, str) or "6.4" not in comsol_version_output:
        message = "Real-smoke receipt requires observed COMSOL 6.4 version output."
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
        "decision_source": {
            "artifact": materials.VP2_DECISION_ARTIFACT,
            "schema_version": materials.VP2_DECISION_SCHEMA_VERSION,
            "sha256": materials.VP2_DECISION_SHA256,
        },
        "material_family_inventory": list(materials.MATERIAL_FAMILIES),
        "source_binding": _source_binding(),
        "templates": _template_binding(),
        "profile_mappings": profile_mappings,
        "campaigns": [
            _campaign_record(steady_run_id, steady_campaign, steady_terminal, steady_workflow),
            _campaign_record(transient_run_id, transient_campaign, transient_terminal, transient_workflow),
        ],
        "cases": [case.record for case in all_cases],
        "comsol": {
            "required_version": "6.4",
            "version_command": ["comsol", "-version"],
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
        "template_equivalence": _equivalence_report(steady_cases, transient_cases),
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
