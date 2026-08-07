"""
===============================================================================
generation_case.py
===============================================================================
Create deterministic profile-qualified inputs and isolated COMSOL work directories.
Responsibilities:
  - Write configured spatial, scalar, schedule, and canonical case files
  - Derive input-bound deterministic case identity and file hashes
  - Copy the immutable template into one fresh node-local work directory
Design principles:
  - Every COMSOL adapter is generated from canonical case provenance
  - Stable UTF-8, newlines, headers, ordering, and numeric formatting are explicit
  - A source template is read and hashed but never used as an output target
This module does NOT:
  - Run COMSOL, collect exports, or publish completed cases
  - Define drying-specific scalars, schedules, or output fields
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
from itertools import pairwise
from pathlib import Path
from typing import Any

from src import common

from . import generation_config as config_contract
from . import generation_fields as fields_service
from . import generation_sampling as sampling_service

CASE_SCHEMA_KIND = "simulation_case"
CASE_SCHEMA_VERSION = 1
_MINIMUM_SCHEDULE_ROWS = 2


def compute_case_identity(payload: dict[str, Any]) -> str:
    """Compute one identity from a generation-time or persisted case payload."""
    try:
        seed = payload["seed"] if "seed" in payload else payload["seed_evidence"]["case_seed"]
        template_sha256 = payload["template_sha256"] if "template_sha256" in payload else payload["template"]["sha256"]
        identity_payload = {
            "schema_version": payload["schema_version"],
            "simulation_profile": payload["simulation_profile"],
            "batch_identity": payload["batch_identity"],
            "case_id": payload["case_id"],
            "case_index": payload["case_index"],
            "generator_version": payload["generator_version"],
            "seed": seed,
            "template_sha256": template_sha256,
            "export_contract_sha256": payload["export_contract_sha256"],
            "input_files": payload["input_files"],
        }
    except (KeyError, TypeError) as error:
        message = "Case identity payload is missing or has malformed identity-bound fields."
        raise ValueError(message) from error
    return common.serialization.canonical_json_sha256(identity_payload)


@dataclass(frozen=True, slots=True)
class CaseBundle:
    """One generated case input bundle and its canonical identity."""

    directory: Path
    case_id: str
    case_identity: str
    case_payload: dict[str, Any]
    input_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """One isolated case work directory containing a disposable model copy."""

    bundle: CaseBundle
    work_directory: Path
    model_path: Path
    exports_directory: Path
    runtime_directory: Path


def _format_number(value: float) -> str:
    """Format one finite number with locale-independent round-trip precision."""
    if not math.isfinite(value):
        message = f"Cannot serialize non-finite numeric value {value!r}."
        raise ValueError(message)
    return format(float(value), ".17g")


def _table_text(rows: list[list[str]], *, delimiter: str) -> str:
    """Serialize one stable UTF-8-compatible table with LF line endings."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    return stream.getvalue()


def _write_spatial_files(
    destination: Path,
    spatial_fields: fields_service.SpatialFields,
    specs: list[dict[str, Any]],
) -> list[Path]:
    """Write every configured spatial adapter in deterministic column-major order."""
    paths: list[Path] = []
    for spec in specs:
        missing = [name for name in spec["columns"] if name not in spatial_fields.columns]
        if missing:
            message = f"Spatial adapter {spec['filename']!r} requests unknown generated fields: {missing}."
            raise ValueError(message)
        rows = [list(spec["columns"])]
        flattened = [spatial_fields.columns[name].ravel(order="F") for name in spec["columns"]]
        rows.extend([_format_number(float(value)) for value in values] for values in zip(*flattened, strict=True))
        path = destination / spec["filename"]
        common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
        paths.append(path)
    return paths


def _write_scalar_file(
    destination: Path,
    spec: dict[str, Any] | None,
    entries: list[dict[str, Any]],
) -> Path | None:
    """Write one generic scalar adapter, or no file when defaults remain authoritative."""
    if spec is None:
        if entries:
            message = "Scalar entries cannot exist without inputs.scalar_file configuration."
            raise ValueError(message)
        return None
    if not entries and not spec.get("required_when_empty", False):
        return None
    if spec["format"] == "long":
        rows: list[list[str]] = []
        if spec["include_header"]:
            rows.append(["name", "value", "unit"])
        rows.extend([entry["name"], _format_number(float(entry["value"])), str(entry.get("unit", ""))] for entry in entries)
    else:
        rows = []
        if spec["include_header"]:
            rows.append([entry["name"] for entry in entries])
        if entries:
            rows.append([_format_number(float(entry["value"])) for entry in entries])
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(rows, delimiter=spec["delimiter"]))
    return path


def _validated_schedule_rows(
    spec: dict[str, Any],
    rows: list[list[float]],
) -> list[list[float]]:
    """Validate optional case-overridden schedule rows at their owning boundary."""
    if len(rows) < 2:  # noqa: PLR2004 -- a schedule requires explicit endpoints
        message = "A configured schedule must contain explicit first and final rows."
        raise ValueError(message)
    width = len(spec["columns"])
    validated: list[list[float]] = []
    for index, row in enumerate(rows):
        if len(row) != width:
            message = f"Schedule row {index} has width {len(row)}; expected {width}."
            raise ValueError(message)
        numeric = [float(value) for value in row]
        if not all(math.isfinite(value) for value in numeric):
            message = f"Schedule row {index} contains non-finite values."
            raise ValueError(message)
        validated.append(numeric)
    if any(right[0] <= left[0] for left, right in pairwise(validated)):
        message = "Schedule time must be strictly increasing."
        raise ValueError(message)
    return validated


def _write_schedule_file(
    destination: Path,
    spec: dict[str, Any] | None,
    rows: list[list[float]] | None,
) -> Path | None:
    """Write one optional schedule adapter without fabricating absent schedules."""
    if spec is None:
        if rows is not None:
            message = "Schedule rows cannot exist without inputs.schedule_file configuration."
            raise ValueError(message)
        return None
    if rows is None:
        message = "Configured schedule adapter is missing its explicit rows."
        raise ValueError(message)
    validated = _validated_schedule_rows(spec, rows)
    table = [list(spec["columns"]), *[[_format_number(value) for value in row] for row in validated]]
    path = destination / spec["filename"]
    common.serialization.atomic_write_text(path, _table_text(table, delimiter=spec["delimiter"]))
    return path


def _require_empty_destination(destination: Path) -> None:
    """Create one new bundle destination or require an existing empty directory."""
    if destination.exists() and not destination.is_dir():
        message = f"Case bundle destination is not a directory: {destination}"
        raise FileExistsError(message)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        message = f"Case bundle destination must be empty: {destination}"
        raise FileExistsError(message)


def generate_case_input_bundle(
    config: config_contract.GenerationConfig,
    case_index: int,
    destination: Path | str,
) -> CaseBundle:
    """
    Generate one deterministic case-owned input bundle.

    ``case.json`` is the canonical scalar and provenance representation. Text
    tables are case-local COMSOL adapters and their exact bytes are hash-bound.
    """
    case_id = config.case_id(case_index)
    case_seed = config.case_seed(case_index)
    bundle_dir = Path(destination).expanduser().resolve()
    _require_empty_destination(bundle_dir)
    parameters, scalar_entries, schedule_rows = sampling_service.resolve_case_values(config, case_index)
    spatial_fields = fields_service.generate_spatial_fields(
        config.values["generator"]["domain"],
        parameters,
        seed=case_seed,
    )
    input_paths = _write_spatial_files(bundle_dir, spatial_fields, config.values["inputs"]["spatial_files"])
    scalar_path = _write_scalar_file(bundle_dir, config.values["inputs"].get("scalar_file"), scalar_entries)
    if scalar_path is not None:
        input_paths.append(scalar_path)
    schedule_path = _write_schedule_file(bundle_dir, config.values["inputs"].get("schedule_file"), schedule_rows)
    if schedule_path is not None:
        input_paths.append(schedule_path)
    input_files = {
        path.name: {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(input_paths, key=lambda item: item.name)
    }
    export_contract_sha256 = common.serialization.canonical_json_sha256(config.values["exports"])
    case_identity_payload = {
        "schema_version": CASE_SCHEMA_VERSION,
        "simulation_profile": config.profile.id,
        "batch_identity": config.batch_identity,
        "case_id": case_id,
        "case_index": case_index,
        "generator_version": config_contract.GENERATOR_VERSION,
        "seed": case_seed,
        "template_sha256": config.template_sha256,
        "export_contract_sha256": export_contract_sha256,
        "input_files": input_files,
    }
    case_identity = compute_case_identity(case_identity_payload)
    scalar_provenance = [
        {"name": entry["name"], "value": float(entry["value"]), **({"unit": entry["unit"]} if "unit" in entry else {})} for entry in scalar_entries
    ]
    case_payload = {
        "schema_kind": CASE_SCHEMA_KIND,
        "schema_version": CASE_SCHEMA_VERSION,
        "simulation_profile": config.profile.id,
        "batch_id": config.batch_id,
        "batch_identity": config.batch_identity,
        "case_id": case_id,
        "case_identity": case_identity,
        "case_index": case_index,
        "generator_version": config_contract.GENERATOR_VERSION,
        "available_learning_views": list(config.profile.available_learning_views),
        "airflow_source": config.profile.airflow_source,
        "seed_evidence": {"batch_seed": config.seed_base, "case_seed": case_seed, "derivation": "batch_seed + case_index"},
        "generator_parameters": parameters,
        "generator_metadata": spatial_fields.metadata,
        "scalars": scalar_provenance,
        "spatial_input_filenames": [spec["filename"] for spec in config.values["inputs"]["spatial_files"]],
        "spatial_input_contracts": [
            {
                "filename": spec["filename"],
                "delimiter": spec["delimiter"],
                "columns": list(spec["columns"]),
            }
            for spec in config.values["inputs"]["spatial_files"]
        ],
        "scalar_filename": None if scalar_path is None else scalar_path.name,
        "schedule_filename": None if schedule_path is None else schedule_path.name,
        "template": {
            "relative_path": config.profile.template_relative_path,
            "filename": config.template_path.name,
            "sha256": config.template_sha256,
        },
        "export_contract": config.values["exports"],
        "export_contract_sha256": export_contract_sha256,
        "input_files": input_files,
    }
    common.serialization.atomic_write_json(bundle_dir / "case.json", case_payload)
    return CaseBundle(
        directory=bundle_dir,
        case_id=case_id,
        case_identity=case_identity,
        case_payload=case_payload,
        input_paths=tuple(sorted(input_paths, key=lambda item: item.name)),
    )


def _resolve_work_root(
    config: config_contract.GenerationConfig,
    *,
    storage_root: Path | str | None,
    work_root: Path | str | None,
) -> Path:
    """Resolve a node-local work root, falling back to generation state storage."""
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
        message = f"Case work directories cannot be created inside the repository: {canonical}"
        raise ValueError(message)
    return canonical


def prepare_case_work_directory(
    config: config_contract.GenerationConfig,
    case_index: int,
    *,
    storage_root: Path | str | None = None,
    work_root: Path | str | None = None,
) -> PreparedCase:
    """
    Prepare one fresh isolated work directory beside a disposable model copy.

    The source template digest is checked before and after copying. The source
    path is never opened for writing and is never a subprocess output target.
    """
    case_id = config.case_id(case_index)
    if common.serialization.file_sha256(config.template_path) != config.template_sha256:
        message = f"COMSOL template changed after configuration preflight: {config.template_path}"
        raise RuntimeError(message)
    root = _resolve_work_root(config, storage_root=storage_root, work_root=work_root)
    work_directory = Path(tempfile.mkdtemp(prefix=f"{case_id}_work_", dir=root)).resolve()
    model_path = work_directory / "model.mph"
    try:
        bundle = generate_case_input_bundle(config, case_index, work_directory)
        shutil.copyfile(config.template_path, model_path)
        if common.serialization.file_sha256(model_path) != config.template_sha256:
            message = f"Copied COMSOL template digest mismatch in {work_directory}."
            raise RuntimeError(message)  # noqa: TRY301 -- enclosing cleanup owns partial work
        exports_directory = work_directory / config.values["exports"]["root"]
        runtime_directory = work_directory / "runtime"
        exports_directory.mkdir()
        runtime_directory.mkdir()
        if common.serialization.file_sha256(config.template_path) != config.template_sha256:
            message = f"Source COMSOL template changed during case preparation: {config.template_path}"
            raise RuntimeError(message)  # noqa: TRY301 -- enclosing cleanup owns partial work
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
