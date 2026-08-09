"""
===============================================================================
generation_case.py
===============================================================================
Create deterministic scientific inputs and isolated COMSOL work directories.
Responsibilities:
  - Resolve one typed sample into spatial, scalar, and schedule adapter tables
  - Derive exact adapter-input and profile-bound simulation identities
  - Copy one immutable template into a fresh disposable work directory
Design principles:
  - Adapter bytes are deterministic, case-local, and identity-bound
  - Every stochastic stream and block row is explicit in case provenance
  - Source templates are read and copied but never opened as output targets
This module does NOT:
  - Run COMSOL, infer template mappings, or publish canonical HDF5 results
  - Treat CSV as a canonical learning-view artifact
===============================================================================
"""

from __future__ import annotations

import csv
import io
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import common

from . import generation_config as config_contract
from . import generation_fields as fields_service
from . import generation_profiles as profiles
from . import generation_sampling as sampling_service
from . import generation_schedule as schedule_service
from . import generation_source as source_service
from . import generation_workspace as workspace_service

CASE_SCHEMA_KIND = "simulation_case"
CASE_SCHEMA_VERSION = 1


def compute_case_input_id(payload: dict[str, Any]) -> str:
    """Compute the exact generated-input identity from persisted evidence."""
    try:
        identity_payload = {
            "schema_version": payload["schema_version"],
            "case_input_config_digest": payload["case_input_config_digest"],
            "material_family": payload["material_family"],
            "sampling_regime": payload["sampling_regime"],
            "ood": payload["ood"],
            "case_index": payload["case_index"],
            "seed_evidence": payload["seed_evidence"],
            "block_provenance": payload["block_provenance"],
            "sampled_values": payload["sampled_values"],
            "coupled_selections": payload["coupled_selections"],
            "input_files": payload["input_files"],
        }
    except (KeyError, TypeError) as error:
        msg = "Case-input identity payload is incomplete or malformed."
        raise ValueError(msg) from error
    return common.serialization.canonical_json_sha256(identity_payload)


def compute_simulation_case_id(payload: dict[str, Any]) -> str:
    """Compute the profile-, template-, and export-bound simulation identity."""
    try:
        identity_payload = {
            "schema_version": payload["schema_version"],
            "case_input_id": payload["case_input_id"],
            "simulation_profile": payload["simulation_profile"],
            "template_sha256": payload["template"]["sha256"],
            "export_contract_sha256": payload["export_contract_sha256"],
            "steady_flow_conditioning_digest": payload["steady_flow_conditioning_digest"],
        }
    except (KeyError, TypeError) as error:
        msg = "Simulation-case identity payload is incomplete or malformed."
        raise ValueError(msg) from error
    return common.serialization.canonical_json_sha256(identity_payload)


@dataclass(frozen=True, slots=True)
class CaseBundle:
    """One generated case input bundle and its two canonical identities."""

    directory: Path
    case_id: str
    case_input_id: str
    simulation_case_id: str
    case_payload: dict[str, Any]
    input_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """One isolated work directory containing adapters and a model copy."""

    bundle: CaseBundle
    work_directory: Path
    work_root: Path
    workspace_run_id: str
    workspace_marker: Path
    model_path: Path
    exports_directory: Path
    runtime_directory: Path


class CasePreparationError(RuntimeError):
    """Report preparation failure while preserving marked-workspace identity."""

    def __init__(
        self,
        message: str,
        *,
        work_directory: Path,
        work_root: Path,
        workspace_run_id: str,
    ) -> None:
        """Initialize one preparation error with its cleanup boundaries."""
        super().__init__(message)
        self.work_directory = work_directory
        self.work_root = work_root
        self.workspace_run_id = workspace_run_id


def _format_number(value: float) -> str:
    """Format one finite number with locale-independent round-trip precision."""
    if not math.isfinite(value):
        msg = f"Cannot serialize non-finite numeric value {value!r}."
        raise ValueError(msg)
    return format(float(value), ".17g")


def _table_text(rows: list[list[str]], *, delimiter: str) -> str:
    """Serialize one stable table with LF line endings."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    return stream.getvalue()


def _write_spatial_file(destination: Path, fields: fields_service.SpatialFields, spec: dict[str, Any]) -> Path:
    """Write the exact spatial adapter in deterministic column-major order."""
    columns = list(spec["columns"])
    missing = [name for name in columns if name not in fields.columns]
    if missing:
        msg = f"Spatial adapter requests unknown generated fields: {missing}."
        raise ValueError(msg)
    flattened = [fields.columns[name].ravel(order="F") for name in columns]
    rows = [columns]
    rows.extend([_format_number(float(value)) for value in values] for values in zip(*flattened, strict=True))
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
    return path


def _write_scalar_file(
    destination: Path,
    spec: dict[str, Any],
    profile_id: str,
    values: dict[str, Any],
    units: dict[str, str],
) -> tuple[Path, list[dict[str, Any]]]:
    """Write and validate the sole profile-owned long-form scalar handoff."""
    field_names = profiles.scalar_input_fields(profile_id)
    expected_units = profiles.scalar_input_units(profile_id)
    package_fixed = {"T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}
    entries: list[dict[str, Any]] = []
    for name, expected_unit in zip(field_names, expected_units, strict=True):
        if name not in values:
            msg = f"Required scalar adapter value {name!r} is unresolved."
            raise ValueError(msg)
        number = float(values[name])
        if not math.isfinite(number):
            msg = f"Required scalar adapter value {name!r} is non-finite."
            raise ValueError(msg)
        unit = units.get(name)
        if unit != expected_unit:
            msg = f"Scalar adapter unit for {name!r} must be {expected_unit!r}, got {unit!r}."
            raise ValueError(msg)
        entries.append(
            {
                "name": name,
                "value": number,
                "unit": unit,
                "owner": "package_fixed" if name in package_fixed else "case_dependent",
            }
        )
    rows = [["name", "value", "unit"]]
    rows.extend([[entry["name"], _format_number(entry["value"]), entry["unit"]] for entry in entries])
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
    return path, entries


def _write_schedule_file(destination: Path, spec: dict[str, Any], schedule: schedule_service.Schedule) -> Path:
    """Write the exact four-column regular schedule adapter."""
    columns = list(spec["columns"])
    if tuple(columns) != profiles.SCHEDULE_FIELDS or schedule.values.shape != (169, len(columns)):
        msg = "Generated schedule does not satisfy the exact adapter shape and field contract."
        raise ValueError(msg)
    rows = [columns]
    rows.extend([_format_number(float(value)) for value in row] for row in schedule.values)
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
    return path


def _require_empty_destination(
    destination: Path,
    *,
    allow_workspace_marker: bool,
) -> None:
    """Create one new bundle destination or require only its ownership marker."""
    if destination.exists() and not destination.is_dir():
        msg = f"Case bundle destination is not a directory: {destination}"
        raise FileExistsError(msg)
    destination.mkdir(parents=True, exist_ok=True)
    allowed = {workspace_service.CASE_WORKSPACE_MARKER} if allow_workspace_marker else set()
    actual = {entry.name for entry in destination.iterdir()}
    if actual != allowed:
        msg = f"Case bundle destination must contain exactly {sorted(allowed)} before generation: {destination}"
        raise FileExistsError(msg)


def _subseeds(config: config_contract.GenerationConfig, case_index: int) -> dict[str, int]:
    """Return only the profile-owned stable case sub-seeds."""
    seed_base = config.seed_base
    if seed_base is None:
        message = f"Batch {config.batch_name!r} has no resolved seed."
        raise ValueError(message)
    labels = ["bed", "pressure_bc"]
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        labels.extend(("initial_moisture", "schedule_shared", "schedule_independent"))
    return {label: config_contract.derive_seed(seed_base, "case", str(case_index), label) for label in labels}


def generate_case_input_bundle(
    config: config_contract.GenerationConfig,
    case_index: int,
    destination: Path | str,
    *,
    _allow_workspace_marker: bool = False,
) -> CaseBundle:
    """Generate one deterministic profile-specific scientific input bundle."""
    case_id = config.case_id(case_index)
    case_seed = config.case_seed(case_index)
    assignment = config.case_assignment(case_index)
    bundle_dir = Path(destination).expanduser().resolve()
    _require_empty_destination(
        bundle_dir,
        allow_workspace_marker=_allow_workspace_marker,
    )
    sample = sampling_service.sample_case(config, case_index)
    values = dict(sample.values)
    units = dict(sample.units)
    fixed = config.scientific_values["scientific_fixed_values"]
    stationary_fixed_entries: list[dict[str, Any]] = []
    for name, unit in zip(
        profiles.STATIONARY_FIXED_FIELDS,
        profiles.STATIONARY_FIXED_UNITS,
        strict=True,
    ):
        value = fixed[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            msg = f"Package-fixed stationary scalar {name!r} is unresolved."
            raise ValueError(msg)
        number = float(value)
        expected = profiles.STATIONARY_FIXED_VALUES[name]
        if number != expected:
            msg = f"Package-fixed stationary value {name!r} does not match the canonical template contract."
            raise ValueError(msg)
        values[name] = number
        units[name] = unit
        stationary_fixed_entries.append(
            {
                "name": name,
                "value": number,
                "unit": unit,
                "owner": "package_fixed",
                "runtime_source": "canonical_template",
            }
        )
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        values["f_wet_dm_max"] = fixed["f_wet_dm_max"]
        units["f_wet_dm_max"] = "1"
    subseeds = _subseeds(config, case_index)

    schedule: schedule_service.Schedule | None = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        schedule = schedule_service.generate_schedule(
            values,
            config.scientific_values["time"],
            fixed,
            seeds={name: subseeds[name] for name in ("schedule_shared", "schedule_independent")},
        )
        values.update(schedule.derived_scalars)
        units["T_in_ref"] = "K"

    family_contract = config.scientific_values["material"]
    spatial_seed_names: tuple[str, ...] = ("bed", "pressure_bc")
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        spatial_seed_names = (*spatial_seed_names, "initial_moisture")
    fields = fields_service.generate_spatial_fields(
        config.profile.id,
        config.scientific_values["grid"],
        values,
        seeds={name: subseeds[name] for name in spatial_seed_names},
        family_bounds=(family_contract["initial_moisture_bounds"] if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE else None),
    )
    input_contract = config.scientific_values["input_contract"]
    spatial_path = _write_spatial_file(bundle_dir, fields, input_contract["spatial"])
    input_paths_list = [spatial_path]
    scalar_path: Path | None = None
    scalar_entries: list[dict[str, Any]] | None = None
    if config.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        scalar_path, scalar_entries = _write_scalar_file(
            bundle_dir,
            input_contract["scalar"],
            config.profile.id,
            values,
            units,
        )
        input_paths_list.append(scalar_path)
    if schedule is not None:
        input_paths_list.append(_write_schedule_file(bundle_dir, input_contract["schedule"], schedule))
    input_paths = tuple(sorted(input_paths_list, key=lambda item: item.name))
    input_files = {
        path.name: {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in input_paths
    }
    export_contract_sha256 = common.serialization.canonical_json_sha256(config.scientific_values["output_contract"])
    steady_flow_conditioning = config.scientific_values["steady_flow_conditioning"]
    steady_flow_conditioning_digest = common.serialization.canonical_json_sha256(steady_flow_conditioning)
    seed_evidence = {
        "batch_seed": config.seed_base,
        "case_seed": case_seed,
        "derivation": "sha256(generator_version|batch_seed|semantic_labels)",
        "subseeds": subseeds,
    }
    case_payload: dict[str, Any] = {
        "schema_kind": CASE_SCHEMA_KIND,
        "schema_version": CASE_SCHEMA_VERSION,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "scientific_config_digest": config.scientific_config_digest,
        "case_input_config_digest": config.case_input_config_digest,
        "case_id": case_id,
        "case_index": case_index,
        "generator_version": config_contract.GENERATOR_VERSION,
        "git_commit": source_service.required_git_commit(),
        "material_family": assignment["material_family"],
        "sampling_regime": assignment["sampling_regime"],
        "ood": sample.ood_provenance,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "steady_flow_conditioning": steady_flow_conditioning,
        "steady_flow_conditioning_digest": steady_flow_conditioning_digest,
        "stationary_fixed_ownership": config.scientific_values["stationary_fixed_ownership"],
        "stationary_fixed_values": stationary_fixed_entries,
        "seed_evidence": seed_evidence,
        "block_provenance": sample.block_provenance,
        "sampled_values": values,
        "sampled_units": units,
        "coupled_selections": sample.coupled_selections,
        "spatial_diagnostics": fields.metadata,
        "input_contract": input_contract,
        "template": {
            "relative_path": config.profile.template_relative_path,
            "filename": config.template_path.name,
            "sha256": config.template_sha256,
        },
        "export_contract_sha256": export_contract_sha256,
        "input_files": input_files,
    }
    if scalar_path is not None and scalar_entries is not None:
        case_payload["scalar_handoff"] = {
            "mechanism": "case_local_long_form_csv",
            "filename": scalar_path.name,
            "fresh_per_case": True,
            "runtime_validation": "required",
            "entries": scalar_entries,
        }
        case_payload["scalars"] = scalar_entries
    if schedule is not None:
        case_payload["schedule_diagnostics"] = schedule.metadata
    case_payload["case_input_id"] = compute_case_input_id(case_payload)
    case_payload["simulation_case_id"] = compute_simulation_case_id(case_payload)
    common.serialization.atomic_write_json(bundle_dir / "case.json", case_payload)
    return CaseBundle(
        directory=bundle_dir,
        case_id=case_id,
        case_input_id=case_payload["case_input_id"],
        simulation_case_id=case_payload["simulation_case_id"],
        case_payload=case_payload,
        input_paths=input_paths,
    )


def _require_template_digest(path: Path, expected_sha256: str, *, message: str) -> None:
    """Require one template copy to retain its preflight digest."""
    if common.serialization.file_sha256(path) != expected_sha256:
        raise RuntimeError(message)


def prepare_case_work_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> PreparedCase:
    """Prepare a fresh case directory beside a digest-verified disposable model."""
    case_id = config.case_id(case_index)
    _require_template_digest(
        config.template_path,
        config.template_sha256,
        message=f"COMSOL template changed after configuration preflight: {config.template_path}",
    )
    storage = workspace_service.resolve_storage_root(storage_root, create=True)
    work_directory, root, marker_path = workspace_service.create_case_workspace(
        config,
        case_id=case_id,
        storage_root=storage,
        work_root=work_root,
    )
    run_id = workspace_service.workspace_run_id(config)
    model_path = work_directory / "model.mph"
    try:
        bundle = generate_case_input_bundle(
            config,
            case_index,
            work_directory,
            _allow_workspace_marker=True,
        )
        shutil.copyfile(config.template_path, model_path)
        _require_template_digest(
            model_path,
            config.template_sha256,
            message=f"Copied COMSOL template digest mismatch in {work_directory}.",
        )
        exports_directory = work_directory / config.scientific_values["output_contract"]["exports_root"]
        runtime_directory = work_directory / "runtime"
        exports_directory.mkdir()
        runtime_directory.mkdir()
        _require_template_digest(
            config.template_path,
            config.template_sha256,
            message=f"Source COMSOL template changed during case preparation: {config.template_path}",
        )
    except BaseException as error:
        message = f"Could not prepare isolated case workspace {work_directory}: {error}"
        raise CasePreparationError(
            message,
            work_directory=work_directory,
            work_root=root,
            workspace_run_id=run_id,
        ) from error
    return PreparedCase(
        bundle=bundle,
        work_directory=work_directory,
        work_root=root,
        workspace_run_id=run_id,
        workspace_marker=marker_path,
        model_path=model_path,
        exports_directory=exports_directory,
        runtime_directory=runtime_directory,
    )
