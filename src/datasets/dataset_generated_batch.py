"""
===============================================================================
dataset_generated_batch.py
===============================================================================
Admit current profile-qualified simulation batches for task-owned interpretation.
Responsibilities:
  - Validate terminal batch, case-publication, membership, and hash evidence
  - Materialize the steady-flow learning view from either simulation profile
  - Apply TaskSpec field order and permeability storage representations
Design principles:
  - Only the current Python producer schema is admitted
  - Available learning views and source profiles are explicit persisted metadata
  - Scientific inputs and outputs fail closed before tensor construction
This module does NOT:
  - Admit earlier generated-batch schemas or producer-progress files
  - Build or publish final training datasets or fit normalizers
  - Define transient-drying training fields or temporal transformations
===============================================================================
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from src import common, domain
from src.generation import generation_case as generation_case_service
from src.generation import generation_profiles

from . import dataset_identity as identity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.domain.tasks.domain_task_spec import TaskSpec

BATCH_MANIFEST_SCHEMA_KIND = "simulation_batch_manifest"
BATCH_MANIFEST_SCHEMA_VERSION = 1
_CASE_ID_PATTERN = re.compile(r"case_[0-9]{4,}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GRID_RTOL = 1e-8
_COORDINATE_ATOL = 1e-12
_MINIMUM_AXIS_POINTS = 2
_MINIMUM_TABLE_ROWS = 2
_CASE_PAYLOAD_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "simulation_profile",
        "batch_id",
        "batch_identity",
        "case_id",
        "case_identity",
        "case_index",
        "generator_version",
        "available_learning_views",
        "airflow_source",
        "seed_evidence",
        "generator_parameters",
        "generator_metadata",
        "scalars",
        "spatial_input_filenames",
        "spatial_input_contracts",
        "scalar_filename",
        "schedule_filename",
        "template",
        "export_contract",
        "export_contract_sha256",
        "input_files",
    }
)
_PUBLICATION_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "stage",
        "simulation_profile",
        "batch_id",
        "batch_identity",
        "case_id",
        "case_identity",
        "template_sha256",
        "export_contract_sha256",
        "available_learning_views",
        "airflow_source",
        "artifacts",
    }
)
_SUCCESS_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "stage",
        "batch_id",
        "case_id",
        "case_identity",
        "provenance_sha256",
    }
)


class GeneratedBatchError(RuntimeError):
    """Report invalid or incomplete current generation evidence."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    if not path.is_file() or path.is_symlink():
        message = f"Required generated-batch file is missing or unsafe: {path}"
        raise FileNotFoundError(message)
    return common.serialization.file_sha256(path)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required JSON object with contextual errors."""
    if not path.is_file() or path.is_symlink():
        message = f"Missing required {label}: {path}"
        raise FileNotFoundError(message)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        message = f"Could not read {label}: {path}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"{label} must contain a JSON object: {path}"
        raise TypeError(message)
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    """Return one lowercase SHA-256 digest."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        message = f"{label} must be one lowercase SHA-256 digest."
        raise ValueError(message)
    return value


def _batch_paths(batch_id: str, *, storage_root: Path | str | None) -> tuple[Path, Path, Path]:
    """Return current metadata, raw, and processed batch roots."""
    safe_id = common.paths.validate_logical_name(batch_id, label="batch_id")
    meta = common.paths.get_generation_meta_root(storage_root=storage_root) / safe_id
    raw = common.paths.resolve_generated_batch_dir(safe_id, stage="raw", storage_root=storage_root)
    processed = common.paths.resolve_generated_batch_dir(safe_id, stage="processed", storage_root=storage_root)
    return meta, raw, processed


def load_batch_manifest(
    batch_id: str,
    *,
    storage_root: Path | str | None = None,
) -> tuple[dict[str, Any], Path, str]:
    """Load and validate one current terminal batch manifest and success marker."""
    meta, _raw, _processed = _batch_paths(batch_id, storage_root=storage_root)
    manifest_path = meta / "batch_manifest.json"
    success_path = meta / "_SUCCESS"
    manifest = _json_object(manifest_path, label="terminal batch manifest")
    success = _json_object(success_path, label="terminal batch success marker")
    manifest_sha256 = sha256_file(manifest_path)
    expected_keys = {
        "schema_kind",
        "schema_version",
        "status",
        "simulation_profile",
        "available_learning_views",
        "airflow_source",
        "batch_id",
        "batch_identity",
        "template",
        "export_contract_sha256",
        "intended_case_indices",
        "cases",
    }
    if set(manifest) != expected_keys:
        message = f"Terminal batch manifest keys do not match the current schema: {manifest_path}"
        raise GeneratedBatchError(message)
    if manifest["schema_kind"] != BATCH_MANIFEST_SCHEMA_KIND or manifest["schema_version"] != BATCH_MANIFEST_SCHEMA_VERSION:
        message = f"Unsupported generated-batch schema: {manifest_path}"
        raise GeneratedBatchError(message)
    if manifest["status"] != "complete" or manifest["batch_id"] != batch_id:
        message = f"Terminal batch is not complete or has the wrong identity: {manifest_path}"
        raise GeneratedBatchError(message)
    if success != {
        "schema_kind": "simulation_batch_success",
        "schema_version": 1,
        "simulation_profile": manifest["simulation_profile"],
        "batch_id": batch_id,
        "batch_identity": manifest["batch_identity"],
        "manifest_sha256": manifest_sha256,
    }:
        message = f"Terminal batch success marker does not bind the manifest: {success_path}"
        raise GeneratedBatchError(message)
    profile = manifest["simulation_profile"]
    views = manifest["available_learning_views"]
    airflow_source = manifest["airflow_source"]
    expected_profile_metadata = {
        "steady_flow": (["steady_flow"], "comsol_steady_reference"),
        "transient_drying": (["steady_flow", "transient_drying"], "comsol_coupled_reference"),
    }
    if profile not in expected_profile_metadata or (views, airflow_source) != expected_profile_metadata[profile]:
        message = f"Terminal batch has invalid profile learning-view or airflow provenance: {manifest_path}"
        raise GeneratedBatchError(message)
    template = manifest["template"]
    if not isinstance(template, dict) or set(template) != {"relative_path", "sha256"}:
        message = f"Terminal batch template identity is malformed: {manifest_path}"
        raise GeneratedBatchError(message)
    profile_contract = generation_profiles.get_profile(profile)
    if template != {
        "relative_path": profile_contract.template_relative_path,
        "sha256": profile_contract.template_sha256,
    }:
        message = f"Terminal batch template identity disagrees with profile {profile!r}: {manifest_path}"
        raise GeneratedBatchError(message)
    _require_sha256(manifest["batch_identity"], label="terminal batch_identity")
    _require_sha256(manifest["export_contract_sha256"], label="terminal export_contract_sha256")
    indices = manifest["intended_case_indices"]
    records = manifest["cases"]
    if (
        not isinstance(indices, list)
        or not indices
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in indices)
        or indices != sorted(set(indices))
        or not isinstance(records, list)
        or len(records) != len(indices)
    ):
        message = f"Terminal batch membership is malformed: {manifest_path}"
        raise GeneratedBatchError(message)
    expected_ids = [f"case_{value:04d}" for value in indices]
    for expected_index, expected_id, record in zip(indices, expected_ids, records, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "case_index",
            "case_id",
            "case_identity",
            "success_sha256",
            "provenance_sha256",
        }:
            message = f"Terminal batch case record is malformed for {expected_id}: {manifest_path}"
            raise GeneratedBatchError(message)
        if record["case_index"] != expected_index or record["case_id"] != expected_id:
            message = f"Terminal batch case order is invalid for {expected_id}: {manifest_path}"
            raise GeneratedBatchError(message)
        for key in ("case_identity", "success_sha256", "provenance_sha256"):
            _require_sha256(record[key], label=f"{expected_id}.{key}")
    return manifest, manifest_path, manifest_sha256


def _validate_artifacts(directory: Path, provenance: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate exact publication membership and every declared artifact digest."""
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        message = f"Case publication has no artifact map: {directory}"
        raise GeneratedBatchError(message)
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path.name not in {"provenance.json", "_SUCCESS"}
    }
    if actual != set(artifacts):
        message = f"Case publication membership mismatch: {directory}"
        raise GeneratedBatchError(message)
    normalized: dict[str, dict[str, Any]] = {}
    for relative, raw_identity in artifacts.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(raw_identity, dict)
            or set(raw_identity) != {"sha256", "size_bytes"}
        ):
            message = f"Case artifact identity is malformed for {relative!r}: {directory}"
            raise GeneratedBatchError(message)
        digest = _require_sha256(raw_identity["sha256"], label=f"artifact {relative!r} sha256")
        size = raw_identity["size_bytes"]
        path = directory / relative
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or path.stat().st_size != size or sha256_file(path) != digest:
            message = f"Case artifact integrity failure: {path}"
            raise GeneratedBatchError(message)
        normalized[relative] = {"sha256": digest, "size_bytes": size}
    return normalized


def _validate_case_publication(
    directory: Path,
    *,
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate one current raw or processed case publication."""
    case_id = str(record["case_id"])
    success_path = directory / "_SUCCESS"
    provenance_path = directory / "provenance.json"
    success = _json_object(success_path, label=f"{stage} case success marker")
    provenance = _json_object(provenance_path, label=f"{stage} case provenance")
    case = _json_object(directory / "case.json", label=f"{stage} canonical case")
    if (
        set(success) != _SUCCESS_KEYS
        or success["schema_kind"] != "simulation_case_success"
        or success["schema_version"] != 1
        or set(provenance) != _PUBLICATION_KEYS
        or provenance["schema_kind"] != "simulation_case_publication"
        or provenance["schema_version"] != 1
        or set(case) != _CASE_PAYLOAD_KEYS
        or case["schema_kind"] != "simulation_case"
        or case["schema_version"] != 1
    ):
        message = f"Case publication schema is not current for {case_id}: {directory}"
        raise GeneratedBatchError(message)
    if sha256_file(success_path) != record["success_sha256"] and stage == "processed":
        message = f"Terminal manifest success digest mismatch for {case_id}."
        raise GeneratedBatchError(message)
    if sha256_file(provenance_path) != record["provenance_sha256"] and stage == "processed":
        message = f"Terminal manifest provenance digest mismatch for {case_id}."
        raise GeneratedBatchError(message)
    if success.get("provenance_sha256") != sha256_file(provenance_path) or (
        success.get("batch_id"),
        success.get("case_id"),
        success.get("case_identity"),
        success.get("stage"),
    ) != (manifest["batch_id"], case_id, record["case_identity"], stage):
        message = f"Case success marker does not bind publication identity for {case_id}."
        raise GeneratedBatchError(message)
    expected = (
        manifest["simulation_profile"],
        manifest["batch_id"],
        case_id,
        record["case_identity"],
        stage,
    )
    if (
        provenance.get("simulation_profile"),
        provenance.get("batch_id"),
        provenance.get("case_id"),
        provenance.get("case_identity"),
        provenance.get("stage"),
    ) != expected:
        message = f"Case publication identity mismatch for {case_id}."
        raise GeneratedBatchError(message)
    if (
        provenance.get("batch_identity") != manifest["batch_identity"]
        or provenance.get("template_sha256") != manifest["template"]["sha256"]
        or provenance.get("export_contract_sha256") != manifest["export_contract_sha256"]
        or provenance.get("available_learning_views") != manifest["available_learning_views"]
        or provenance.get("airflow_source") != manifest["airflow_source"]
        or case.get("simulation_profile") != manifest["simulation_profile"]
        or case.get("batch_id") != manifest["batch_id"]
        or case.get("batch_identity") != manifest["batch_identity"]
        or case.get("case_id") != case_id
        or case.get("case_identity") != record["case_identity"]
        or case.get("available_learning_views") != manifest["available_learning_views"]
        or case.get("airflow_source") != manifest["airflow_source"]
        or case.get("template")
        != {
            "relative_path": manifest["template"]["relative_path"],
            "filename": Path(manifest["template"]["relative_path"]).name,
            "sha256": manifest["template"]["sha256"],
        }
        or case.get("export_contract_sha256") != manifest["export_contract_sha256"]
        or generation_case_service.compute_case_identity(case) != record["case_identity"]
    ):
        message = f"Canonical case metadata disagrees with terminal profile identity for {case_id}."
        raise GeneratedBatchError(message)
    artifacts = _validate_artifacts(directory, provenance)
    return case, artifacts


def validate_exact_source_membership(
    batch_id: str,
    manifest: Mapping[str, Any],
    *,
    storage_root: Path | str | None = None,
) -> None:
    """Require exact raw and processed directory membership for a terminal batch."""
    _meta, raw, processed = _batch_paths(batch_id, storage_root=storage_root)
    expected = {str(record["case_id"]) for record in manifest["cases"]}
    for root in (raw, processed):
        entries = tuple(root.iterdir()) if root.is_dir() else ()
        actual = {entry.name for entry in entries}
        unsafe = [entry.name for entry in entries if not entry.is_dir() or entry.is_symlink()]
        if actual != expected or unsafe:
            message = (
                f"Terminal generated-batch membership mismatch under {root}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}, unsafe={sorted(unsafe)}."
            )
            raise GeneratedBatchError(message)


def validate_terminal_batch(
    batch_id: str,
    *,
    storage_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate terminal manifest, exact membership, publications, and hashes."""
    manifest, _manifest_path, _manifest_sha256 = load_batch_manifest(batch_id, storage_root=storage_root)
    validate_exact_source_membership(batch_id, manifest, storage_root=storage_root)
    _meta, raw, processed = _batch_paths(batch_id, storage_root=storage_root)
    for record in manifest["cases"]:
        case_id = record["case_id"]
        raw_case, _raw_artifacts = _validate_case_publication(raw / case_id, manifest=manifest, record=record, stage="raw")
        processed_case, _processed_artifacts = _validate_case_publication(
            processed / case_id,
            manifest=manifest,
            record=record,
            stage="processed",
        )
        if raw_case != processed_case:
            message = f"Raw and processed canonical case payloads disagree for {case_id}."
            raise GeneratedBatchError(message)
    return manifest


def _read_numeric_table(path: Path, *, delimiter: str = ";") -> tuple[list[str], np.ndarray]:
    """Read one finite rectangular header-bearing numeric table."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.lstrip().startswith(("%", "#"))]
    except (OSError, UnicodeDecodeError) as error:
        message = f"Could not read numeric case table: {path}"
        raise ValueError(message) from error
    rows = list(csv.reader(lines, delimiter=delimiter))
    if len(rows) < _MINIMUM_TABLE_ROWS:
        message = f"Numeric case table must contain a header and data: {path}"
        raise ValueError(message)
    header = [value.strip() for value in rows[0]]
    if not header or len(header) != len(set(header)) or any(len(row) != len(header) for row in rows[1:]):
        message = f"Numeric case table has an invalid header or row width: {path}"
        raise ValueError(message)
    try:
        values = np.asarray([[float(value.strip()) for value in row] for row in rows[1:]], dtype=np.float64)
    except ValueError as error:
        message = f"Numeric case table contains malformed data: {path}"
        raise ValueError(message) from error
    if not np.isfinite(values).all():
        message = f"Numeric case table contains non-finite data: {path}"
        raise ValueError(message)
    return header, values


def _cartesian_fields(header: Sequence[str], values: np.ndarray, *, path: Path) -> tuple[tuple[int, int], dict[str, np.ndarray]]:
    """Canonicalize table columns onto one complete uniform y-by-x grid."""
    try:
        x_position = header.index("x")
        y_position = header.index("y")
    except ValueError as error:
        message = f"Cartesian case table must contain x and y columns: {path}"
        raise ValueError(message) from error
    x_axis = np.unique(values[:, x_position])
    y_axis = np.unique(values[:, y_position])
    if x_axis.size < _MINIMUM_AXIS_POINTS or y_axis.size < _MINIMUM_AXIS_POINTS or values.shape[0] != x_axis.size * y_axis.size:
        message = f"Case table does not contain one complete two-dimensional Cartesian grid: {path}"
        raise ValueError(message)
    for axis, label in ((x_axis, "x"), (y_axis, "y")):
        differences = np.diff(axis)
        if np.any(differences <= 0) or not np.allclose(differences, differences[0], rtol=_GRID_RTOL, atol=_COORDINATE_ATOL):
            message = f"Case table {label}-axis is not strictly increasing and uniform: {path}"
            raise ValueError(message)
    x_lookup = {value: index for index, value in enumerate(x_axis)}
    y_lookup = {value: index for index, value in enumerate(y_axis)}
    fields = {name: np.full((y_axis.size, x_axis.size), np.nan, dtype=np.float64) for name in header}
    occupied: set[tuple[int, int]] = set()
    for row in values:
        coordinate = y_lookup[row[y_position]], x_lookup[row[x_position]]
        if coordinate in occupied:
            message = f"Case table repeats one Cartesian coordinate: {path}"
            raise ValueError(message)
        occupied.add(coordinate)
        for index, name in enumerate(header):
            fields[name][coordinate] = row[index]
    if not all(np.isfinite(field).all() for field in fields.values()):
        message = f"Case table Cartesian grid contains missing values: {path}"
        raise ValueError(message)
    return (y_axis.size, x_axis.size), fields


def _steady_flow_fields(
    input_fields: Mapping[str, np.ndarray],
    output_fields: Mapping[str, np.ndarray],
    *,
    task: TaskSpec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply the task-owned steady-flow channel and permeability contract."""
    if task.id != "steady_flow":
        message = f"Generated airflow views support only the registered steady_flow task, got {task.id!r}."
        raise ValueError(message)
    if not np.allclose(input_fields["x"], output_fields["x"], rtol=0.0, atol=_COORDINATE_ATOL) or not np.allclose(
        input_fields["y"], output_fields["y"], rtol=0.0, atol=_COORDINATE_ATOL
    ):
        message = "Generated input and steady-flow learning-view coordinates disagree."
        raise ValueError(message)
    raw_kxx = input_fields["Kxx"]
    raw_kxy = input_fields["Kxy"]
    raw_kyy = input_fields["Kyy"]
    determinant = raw_kxx * raw_kyy - raw_kxy**2
    if np.any(raw_kxx <= 0) or np.any(raw_kyy <= 0) or np.any(determinant <= 0):
        message = "Generated permeability tensor must be positive definite at every point."
        raise ValueError(message)
    values: dict[str, np.ndarray] = {
        "x": input_fields["x"],
        "y": input_fields["y"],
        "kxx": np.log10(raw_kxx),
        "kxy": raw_kxy / np.sqrt(raw_kxx * raw_kyy),
        "kyy": np.log10(raw_kyy),
        "eps": input_fields["eps"],
        "p_bc": input_fields["p_bc"],
        "p": output_fields["p"],
        "u": output_fields["u"],
        "v": output_fields["v"],
    }
    if np.any((values["eps"] <= 0) | (values["eps"] > 1)):
        message = "Generated porosity must satisfy 0 < eps <= 1."
        raise ValueError(message)
    expected = set(task.input_names) | set(task.output_names)
    if set(values) != expected:
        message = "Current steady-flow TaskSpec no longer matches the generated learning-view field contract."
        raise RuntimeError(message)
    converted: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            message = f"Generated field {name!r} is non-finite after float32 conversion."
            raise ValueError(message)
        converted[name] = array.copy()
    return (
        {name: converted[name] for name in task.input_names},
        {name: converted[name] for name in task.output_names},
    )


def interpret_generated_case(
    batch_id: str,
    case_id: str,
    *,
    task: TaskSpec,
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    storage_root: Path | str | None = None,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any], str]:
    """Interpret and fingerprint one validated current-profile steady-flow view."""
    if _CASE_ID_PATTERN.fullmatch(case_id) is None or case_id != record.get("case_id"):
        message = f"Invalid manifest-bound case identifier: {case_id!r}."
        raise ValueError(message)
    _meta, raw_root, processed_root = _batch_paths(batch_id, storage_root=storage_root)
    raw_case, raw_artifacts = _validate_case_publication(raw_root / case_id, manifest=manifest, record=record, stage="raw")
    processed_case, processed_artifacts = _validate_case_publication(
        processed_root / case_id,
        manifest=manifest,
        record=record,
        stage="processed",
    )
    if raw_case != processed_case:
        message = f"Raw and processed canonical case payloads disagree for {case_id}."
        raise GeneratedBatchError(message)
    if "steady_flow" not in manifest["available_learning_views"]:
        message = f"Batch {batch_id!r} does not advertise a validated steady_flow learning view."
        raise ValueError(message)
    input_names = raw_case.get("spatial_input_filenames")
    input_contracts = raw_case.get("spatial_input_contracts")
    if (
        not isinstance(input_names, list)
        or len(input_names) != 1
        or not isinstance(input_names[0], str)
        or not isinstance(input_contracts, list)
        or len(input_contracts) != 1
        or not isinstance(input_contracts[0], dict)
        or set(input_contracts[0]) != {"filename", "delimiter", "columns"}
        or input_contracts[0]["filename"] != input_names[0]
        or input_contracts[0]["delimiter"] not in {",", ";", "\t"}
    ):
        message = f"Case {case_id!r} must bind one generated spatial input contract."
        raise GeneratedBatchError(message)
    input_path = raw_root / case_id / input_names[0]
    view_relative = "learning_views/steady_flow/fields.csv"
    view_path = processed_root / case_id / view_relative
    if input_names[0] not in raw_artifacts or view_relative not in processed_artifacts:
        message = f"Case {case_id!r} does not bind required steady-flow source artifacts."
        raise GeneratedBatchError(message)
    input_header, input_values = _read_numeric_table(input_path, delimiter=input_contracts[0]["delimiter"])
    view_header, view_values = _read_numeric_table(view_path)
    expected_input_header = [source for _canonical, source in generation_profiles.get_profile(manifest["simulation_profile"]).spatial_field_mapping]
    if input_contracts[0]["columns"] != expected_input_header or input_header != expected_input_header:
        message = f"Generated spatial input fields do not match the profile contract for {case_id}."
        raise ValueError(message)
    if view_header != ["x", "y", *task.output_names]:
        message = f"Steady-flow learning-view fields do not match TaskSpec order for {case_id}."
        raise ValueError(message)
    spatial_shape, input_grid = _cartesian_fields(input_header, input_values, path=input_path)
    output_shape, output_grid = _cartesian_fields(view_header, view_values, path=view_path)
    if output_shape != spatial_shape:
        message = f"Generated input and steady-flow view shapes disagree for {case_id}: {spatial_shape} != {output_shape}."
        raise ValueError(message)
    case_inputs, case_outputs = _steady_flow_fields(input_grid, output_grid, task=task)
    inputs = torch.stack([torch.from_numpy(case_inputs[name]) for name in task.input_names])
    outputs = torch.stack([torch.from_numpy(case_outputs[name]) for name in task.output_names])
    scalar_parameters = {
        item["name"]: item["value"]
        for item in raw_case.get("scalars", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("value"), (int, float))
    }
    metadata = {
        "case_id": case_id,
        "case_index": raw_case["case_index"],
        "simulation_profile": manifest["simulation_profile"],
        "available_learning_views": list(manifest["available_learning_views"]),
        "airflow_source": manifest["airflow_source"],
        "template_sha256": manifest["template"]["sha256"],
        "generator_version": raw_case["generator_version"],
        "seed": raw_case["seed_evidence"]["case_seed"],
        "parameters": {**raw_case["generator_parameters"], **scalar_parameters},
        "geometry": raw_case["generator_metadata"]["geometry"],
    }
    stable_source = {
        "case_id": case_id,
        "case_identity": record["case_identity"],
        "simulation_profile": manifest["simulation_profile"],
        "template_sha256": manifest["template"]["sha256"],
        "airflow_source": manifest["airflow_source"],
        "input": raw_artifacts[input_names[0]],
        "steady_flow_view": processed_artifacts[view_relative],
    }
    fingerprint = identity.compute_case_fingerprint(
        task=task,
        case_id=case_id,
        source_identity=stable_source,
        source_metadata=metadata,
        inputs=inputs,
        outputs=outputs,
    )
    return spatial_shape, inputs, outputs, metadata, stable_source, fingerprint


def source_provenance(manifest: Mapping[str, Any], *, manifest_sha256: str) -> dict[str, Any]:
    """Return explicit operational provenance for one current simulation batch."""
    return {
        "batch_manifest_sha256": manifest_sha256,
        "simulation_profile": manifest["simulation_profile"],
        "template_sha256": manifest["template"]["sha256"],
        "airflow_source": manifest["airflow_source"],
        "available_learning_views": list(manifest["available_learning_views"]),
        "cases": [
            {
                "case_id": record["case_id"],
                "case_identity": record["case_identity"],
                "success_sha256": record["success_sha256"],
                "provenance_sha256": record["provenance_sha256"],
            }
            for record in manifest["cases"]
        ],
    }


def load_generated_batch(
    batch_name: str,
    *,
    task_id: str = "steady_flow",
    storage_root: Path | str | None = None,
    show_progress: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Load a validated current generated-batch prefix without publishing tensors."""
    task = domain.tasks.registry.get_task(task_id)
    if task.id != "steady_flow":
        message = f"Generated airflow views support only the steady_flow task, got {task.id!r}."
        raise ValueError(message)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        message = f"max_cases must be a positive integer or None, got {max_cases!r}."
        raise ValueError(message)
    manifest, manifest_path, manifest_sha256 = load_batch_manifest(batch_name, storage_root=storage_root)
    validate_exact_source_membership(batch_name, manifest, storage_root=storage_root)
    records = manifest["cases"] if max_cases is None else manifest["cases"][:max_cases]
    generated_identity = identity.build_generated_batch_identity(manifest)
    rows: list[dict[str, Any]] = []
    iterator = tqdm(records, desc=f"Loading {batch_name}", unit="case", disable=not show_progress)
    for record in iterator:
        _shape, inputs, outputs, metadata, _source, _fingerprint = interpret_generated_case(
            batch_name,
            record["case_id"],
            task=task,
            manifest=manifest,
            record=record,
            storage_root=storage_root,
        )
        rows.append(
            {
                **{name: inputs[index].numpy() for index, name in enumerate(task.input_names)},
                **{name: outputs[index].numpy() for index, name in enumerate(task.output_names)},
                "meta": metadata,
            }
        )
    return {
        "batch_name": batch_name,
        "generation_root": common.paths.get_generation_root(storage_root=storage_root),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "generated_batch_identity": generated_identity,
        "sample_ids": [record["case_id"] for record in records],
        "available_case_count": len(manifest["cases"]),
        "rows": rows,
        "task": task,
    }
