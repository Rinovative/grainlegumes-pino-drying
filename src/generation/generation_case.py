"""
===============================================================================
generation_case.py
===============================================================================
Create deterministic scientific inputs and isolated COMSOL work directories.
Responsibilities:
  - Resolve one typed sample into spatial, scalar, and schedule adapter tables
  - Derive distinct profile-neutral case-input and profile-bound simulation IDs
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
import os
import shutil
import tempfile
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

CASE_SCHEMA_KIND = "simulation_case"
CASE_SCHEMA_VERSION = 4


def compute_case_input_id(payload: dict[str, Any]) -> str:
    """Compute the profile-neutral scientific input identity from persisted evidence."""
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
    model_path: Path
    exports_directory: Path
    runtime_directory: Path


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
    values: dict[str, Any],
    units: dict[str, str],
) -> tuple[Path, list[dict[str, Any]]]:
    """Write the exact generic long-form scalar adapter."""
    entries: list[dict[str, Any]] = []
    for name in profiles.SCALAR_INPUT_FIELDS:
        if name not in values:
            msg = f"Required scalar adapter value {name!r} is unresolved."
            raise ValueError(msg)
        number = float(values[name])
        if not math.isfinite(number):
            msg = f"Required scalar adapter value {name!r} is non-finite."
            raise ValueError(msg)
        unit = units.get(name)
        if not isinstance(unit, str) or not unit:
            msg = f"Required scalar adapter unit {name!r} is unresolved."
            raise ValueError(msg)
        entries.append({"name": name, "value": number, "unit": unit})
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


def _require_empty_destination(destination: Path) -> None:
    """Create one new bundle destination or require an existing empty directory."""
    if destination.exists() and not destination.is_dir():
        msg = f"Case bundle destination is not a directory: {destination}"
        raise FileExistsError(msg)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        msg = f"Case bundle destination must be empty: {destination}"
        raise FileExistsError(msg)


def _subseeds(config: config_contract.GenerationConfig, case_index: int) -> dict[str, int]:
    """Return stable case-owned field and schedule sub-seeds."""
    seed_base = config.seed_base
    if seed_base is None:
        message = f"Batch {config.batch_name!r} has no resolved seed."
        raise ValueError(message)
    return {
        label: config_contract.derive_seed(seed_base, "case", str(case_index), label)
        for label in ("bed", "pressure_bc", "initial_moisture", "schedule_shared", "schedule_independent")
    }


def generate_case_input_bundle(
    config: config_contract.GenerationConfig,
    case_index: int,
    destination: Path | str,
) -> CaseBundle:
    """Generate one deterministic, profile-pairable scientific input bundle."""
    case_id = config.case_id(case_index)
    case_seed = config.case_seed(case_index)
    assignment = config.case_assignment(case_index)
    bundle_dir = Path(destination).expanduser().resolve()
    _require_empty_destination(bundle_dir)
    sample = sampling_service.sample_case(config, case_index)
    values = dict(sample.values)
    subseeds = _subseeds(config, case_index)
    schedule = schedule_service.generate_schedule(
        values,
        config.scientific_values["time"],
        config.scientific_values["scientific_fixed_values"],
        seeds={name: subseeds[name] for name in ("schedule_shared", "schedule_independent")},
    )
    values.update(schedule.derived_scalars)
    values["T_flow_ref"] = 0.5 * (float(values["T_in_ref"]) + float(values["T_init"]))
    values["f_wet_dm_max"] = config.scientific_values["scientific_fixed_values"]["f_wet_dm_max"]
    units = dict(sample.units)
    units.update({"f_wet_dm_max": "1", "T_in_ref": "K", "T_flow_ref": "K"})
    family_contract = config.scientific_values["material"]
    fields = fields_service.generate_spatial_fields(
        config.scientific_values["grid"],
        values,
        seeds={name: subseeds[name] for name in ("bed", "pressure_bc", "initial_moisture")},
        family_bounds=family_contract["initial_moisture_bounds"],
    )
    input_contract = config.scientific_values["input_contract"]
    spatial_path = _write_spatial_file(bundle_dir, fields, input_contract["spatial"])
    scalar_path, scalar_entries = _write_scalar_file(bundle_dir, input_contract["scalar"], values, units)
    schedule_path = _write_schedule_file(bundle_dir, input_contract["schedule"], schedule)
    input_paths = tuple(sorted((spatial_path, scalar_path, schedule_path), key=lambda item: item.name))
    input_files = {path.name: {"sha256": common.serialization.file_sha256(path), "size_bytes": path.stat().st_size} for path in input_paths}
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
        "seed_evidence": seed_evidence,
        "block_provenance": sample.block_provenance,
        "sampled_values": values,
        "sampled_units": units,
        "coupled_selections": sample.coupled_selections,
        "spatial_diagnostics": fields.metadata,
        "schedule_diagnostics": schedule.metadata,
        "scalars": scalar_entries,
        "input_contract": input_contract,
        "template": {
            "relative_path": config.profile.template_relative_path,
            "filename": config.template_path.name,
            "sha256": config.template_sha256,
        },
        "export_contract_sha256": export_contract_sha256,
        "input_files": input_files,
    }
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


def _resolve_work_root(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None,
    work_root: Path | str | None,
) -> Path:
    """Resolve a node-local work root, falling back to private generation state."""
    if work_root is not None:
        root = Path(work_root).expanduser()
    elif os.environ.get("TMPDIR"):
        root = Path(os.environ["TMPDIR"]).expanduser()
    else:
        root = common.paths.get_generation_state_root(storage_root=storage_root) / config.profile.id / config.batch_id / "work"
    root.mkdir(parents=True, exist_ok=True)
    canonical = root.resolve()
    project_root = common.paths.get_project_root().resolve()
    if canonical == project_root or canonical.is_relative_to(project_root):
        msg = f"Case work directories cannot be created inside the repository: {canonical}"
        raise ValueError(msg)
    return canonical


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
    root = _resolve_work_root(config, storage_root=storage_root, work_root=work_root)
    work_directory = Path(tempfile.mkdtemp(prefix=f"{case_id}_work_", dir=root)).resolve()
    model_path = work_directory / "model.mph"
    try:
        bundle = generate_case_input_bundle(config, case_index, work_directory)
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
    except BaseException:
        shutil.rmtree(work_directory, ignore_errors=True)
        raise
    return PreparedCase(
        bundle=bundle,
        work_directory=work_directory,
        model_path=model_path,
        exports_directory=exports_directory,
        runtime_directory=runtime_directory,
    )
