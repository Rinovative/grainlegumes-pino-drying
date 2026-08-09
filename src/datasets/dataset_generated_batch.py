"""
===============================================================================
dataset_generated_batch.py
===============================================================================
Admit canonical HDF5 simulation batches for task-owned interpretation.
Responsibilities:
  - Validate terminal batch, case publication, identity, membership, and hashes
  - Read the steady-flow view from case.h5 for either simulation profile
  - Apply TaskSpec field order and permeability representations
Design principles:
  - Only the current dual-identity HDF5 producer schema is admitted
  - Source profile, template, material family, and learning views remain metadata
  - Scientific fields fail closed before tensor construction
This module does NOT:
  - Admit CSV learning views or historical generation schemas
  - Publish final training datasets or define transient tensor semantics
===============================================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src import common, domain
from src.generation import generation_case as generation_case_service
from src.generation import generation_materials, generation_profiles, generation_source, generation_storage

from . import dataset_identity as identity

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.domain.tasks.domain_task_spec import TaskSpec

BATCH_MANIFEST_SCHEMA_KIND = "simulation_batch_manifest"
BATCH_MANIFEST_SCHEMA_VERSION = 1
_CASE_ID_PATTERN = re.compile(r"case_[0-9]{4,}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class GeneratedBatchError(RuntimeError):
    """Report invalid or incomplete current generation evidence."""


def _hdf5_dataset(handle: h5py.File, name: str) -> h5py.Dataset:
    """Return one required canonical HDF5 dataset."""
    value = handle.get(name)
    if not isinstance(value, h5py.Dataset):
        msg = f"Canonical HDF5 member {name!r} must be a dataset."
        raise GeneratedBatchError(msg)
    return value


def _string_list_attribute(dataset: h5py.Dataset, name: str) -> list[str]:
    """Decode one required JSON string-list dataset attribute."""
    value = dataset.attrs.get(name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        msg = f"Canonical HDF5 attribute {name!r} must be text."
        raise GeneratedBatchError(msg)
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        msg = f"Canonical HDF5 attribute {name!r} must contain a JSON string list."
        raise GeneratedBatchError(msg)
    return decoded


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one safe regular file."""
    if not path.is_file() or path.is_symlink():
        msg = f"Required generated-batch file is missing or unsafe: {path}"
        raise FileNotFoundError(msg)
    return common.serialization.file_sha256(path)


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one required JSON object with contextual errors."""
    if not path.is_file() or path.is_symlink():
        msg = f"Missing required {label}: {path}"
        raise FileNotFoundError(msg)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        msg = f"Could not read {label}: {path}"
        raise ValueError(msg) from error
    if not isinstance(value, dict):
        msg = f"{label} must contain a JSON object: {path}"
        raise TypeError(msg)
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    """Return one lowercase SHA-256 digest."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{label} must be one lowercase SHA-256 digest."
        raise ValueError(msg)
    return value


def _batch_paths(batch_id: str, *, storage_root: Path | str | None) -> tuple[Path, Path, Path]:
    """Return current metadata, input-provenance, and processed batch roots."""
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
        "batch_name",
        "batch_id",
        "batch_identity",
        "material_family",
        "sampling_regime",
        "git_commit",
        "scientific_config_digest",
        "template",
        "export_contract_sha256",
        "intended_case_indices",
        "cases",
    }
    if set(manifest) != expected_keys:
        msg = f"Terminal batch manifest keys do not match the current schema: {manifest_path}"
        raise GeneratedBatchError(msg)
    if manifest["schema_kind"] != BATCH_MANIFEST_SCHEMA_KIND or manifest["schema_version"] != BATCH_MANIFEST_SCHEMA_VERSION:
        msg = f"Unsupported generated-batch schema: {manifest_path}"
        raise GeneratedBatchError(msg)
    if manifest["status"] != "complete" or manifest["batch_id"] != batch_id:
        msg = f"Terminal batch is not complete or has the wrong identity: {manifest_path}"
        raise GeneratedBatchError(msg)
    generation_source.validate_git_commit(manifest["git_commit"])
    expected_batch_name = f"{manifest['simulation_profile']}__{manifest['material_family']}__{manifest['sampling_regime']}"
    if manifest["batch_name"] != expected_batch_name or not batch_id.startswith(f"{expected_batch_name}__"):
        msg = f"Terminal batch name or immutable identifier is malformed: {manifest_path}"
        raise GeneratedBatchError(msg)
    if success != {
        "schema_kind": "simulation_batch_success",
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
        "simulation_profile": manifest["simulation_profile"],
        "batch_id": batch_id,
        "batch_identity": manifest["batch_identity"],
        "manifest_sha256": manifest_sha256,
    }:
        msg = f"Terminal success marker does not bind the manifest: {success_path}"
        raise GeneratedBatchError(msg)
    profile = manifest["simulation_profile"]
    expected_metadata = {
        "steady_flow": (["steady_flow"], "comsol_steady_reference"),
        "transient_drying": (["steady_flow", "transient_drying"], "comsol_coupled_reference"),
    }
    if (
        profile not in expected_metadata
        or (
            manifest["available_learning_views"],
            manifest["airflow_source"],
        )
        != expected_metadata[profile]
    ):
        msg = f"Terminal batch profile learning-view provenance is invalid: {manifest_path}"
        raise GeneratedBatchError(msg)
    template = manifest["template"]
    profile_contract = generation_profiles.get_profile(profile)
    if template != {
        "relative_path": profile_contract.template_relative_path,
        "sha256": profile_contract.template_sha256,
    }:
        msg = f"Terminal template identity disagrees with profile {profile!r}: {manifest_path}"
        raise GeneratedBatchError(msg)
    for key in ("batch_identity", "scientific_config_digest", "export_contract_sha256"):
        _require_sha256(manifest[key], label=f"terminal {key}")
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
        msg = f"Terminal batch membership is malformed: {manifest_path}"
        raise GeneratedBatchError(msg)
    expected_record_keys = {
        "case_index",
        "case_id",
        "material_family",
        "case_input_id",
        "simulation_case_id",
        "success_sha256",
        "provenance_sha256",
        "case_hdf5_sha256",
    }
    for expected_index, record in zip(indices, records, strict=True):
        expected_id = f"case_{expected_index:04d}"
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            msg = f"Terminal case record is malformed for {expected_id}: {manifest_path}"
            raise GeneratedBatchError(msg)
        if record["case_index"] != expected_index or record["case_id"] != expected_id:
            msg = f"Terminal case order is invalid for {expected_id}: {manifest_path}"
            raise GeneratedBatchError(msg)
        if record["material_family"] not in generation_materials.MATERIAL_FAMILIES or record["material_family"] != manifest["material_family"]:
            msg = f"Terminal case has the wrong material_family for {expected_id}: {manifest_path}"
            raise GeneratedBatchError(msg)
        for key in ("case_input_id", "simulation_case_id", "success_sha256", "provenance_sha256", "case_hdf5_sha256"):
            _require_sha256(record[key], label=f"{expected_id}.{key}")
    return manifest, manifest_path, manifest_sha256


def _validate_artifacts(directory: Path, provenance: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate exact publication membership and every declared artifact digest."""
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        msg = f"Case publication has no artifact map: {directory}"
        raise GeneratedBatchError(msg)
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path.name not in {"provenance.json", "_SUCCESS"}
    }
    if actual != set(artifacts):
        msg = f"Case publication membership mismatch: {directory}"
        raise GeneratedBatchError(msg)
    normalized: dict[str, dict[str, Any]] = {}
    for relative, raw_identity in artifacts.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(raw_identity, dict)
            or set(raw_identity) != {"sha256", "size_bytes"}
        ):
            msg = f"Case artifact identity is malformed for {relative!r}: {directory}"
            raise GeneratedBatchError(msg)
        digest = _require_sha256(raw_identity["sha256"], label=f"artifact {relative!r} sha256")
        size = raw_identity["size_bytes"]
        path = directory / relative
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or path.stat().st_size != size or sha256_file(path) != digest:
            msg = f"Case artifact integrity failure: {path}"
            raise GeneratedBatchError(msg)
        normalized[relative] = {"sha256": digest, "size_bytes": size}
    return normalized


def _validate_case_publication(
    directory: Path,
    *,
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    stage: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate one current compact-input or processed case publication."""
    case_id = str(record["case_id"])
    success_path = directory / "_SUCCESS"
    provenance_path = directory / "provenance.json"
    success = _json_object(success_path, label=f"{stage} case success marker")
    provenance = _json_object(provenance_path, label=f"{stage} case provenance")
    case = _json_object(directory / "case.json", label=f"{stage} canonical case provenance")
    if (
        success.get("schema_kind") != "simulation_case_success"
        or success.get("schema_version") != BATCH_MANIFEST_SCHEMA_VERSION
        or provenance.get("schema_kind") != "simulation_case_publication"
        or provenance.get("schema_version") != BATCH_MANIFEST_SCHEMA_VERSION
        or case.get("schema_kind") != "simulation_case"
        or case.get("schema_version") != generation_case_service.CASE_SCHEMA_VERSION
    ):
        msg = f"Case publication schema is not current for {case_id}: {directory}"
        raise GeneratedBatchError(msg)
    if stage == "processed" and (
        sha256_file(success_path) != record["success_sha256"] or sha256_file(provenance_path) != record["provenance_sha256"]
    ):
        msg = f"Terminal manifest publication digest mismatch for {case_id}."
        raise GeneratedBatchError(msg)
    if success.get("provenance_sha256") != sha256_file(provenance_path):
        msg = f"Case success marker does not bind provenance for {case_id}."
        raise GeneratedBatchError(msg)
    expected = manifest["batch_id"], case_id, stage, record["case_input_id"], record["simulation_case_id"]
    if (
        success.get("batch_id"),
        success.get("case_id"),
        success.get("stage"),
        success.get("case_input_id"),
        success.get("simulation_case_id"),
    ) != expected or (
        provenance.get("batch_id"),
        provenance.get("case_id"),
        provenance.get("stage"),
        provenance.get("case_input_id"),
        provenance.get("simulation_case_id"),
    ) != expected:
        msg = f"Case publication identity mismatch for {case_id}."
        raise GeneratedBatchError(msg)
    if (
        case.get("batch_id") != manifest["batch_id"]
        or case.get("batch_identity") != manifest["batch_identity"]
        or case.get("scientific_config_digest") != manifest["scientific_config_digest"]
        or case.get("simulation_profile") != manifest["simulation_profile"]
        or case.get("case_id") != case_id
        or case.get("case_input_id") != record["case_input_id"]
        or case.get("simulation_case_id") != record["simulation_case_id"]
        or case.get("material_family") != record["material_family"]
        or case.get("sampling_regime") != manifest["sampling_regime"]
        or case.get("git_commit") != manifest["git_commit"]
        or provenance.get("git_commit") != manifest["git_commit"]
        or case.get("available_learning_views") != manifest["available_learning_views"]
        or case.get("airflow_source") != manifest["airflow_source"]
        or case.get("template")
        != {
            "relative_path": manifest["template"]["relative_path"],
            "filename": Path(manifest["template"]["relative_path"]).name,
            "sha256": manifest["template"]["sha256"],
        }
        or case.get("export_contract_sha256") != manifest["export_contract_sha256"]
        or generation_case_service.compute_case_input_id(case) != record["case_input_id"]
        or generation_case_service.compute_simulation_case_id(case) != record["simulation_case_id"]
    ):
        msg = f"Canonical case metadata disagrees with terminal identity for {case_id}."
        raise GeneratedBatchError(msg)
    artifacts = _validate_artifacts(directory, provenance)
    if stage == "processed":
        if "case.h5" not in artifacts or artifacts["case.h5"]["sha256"] != record["case_hdf5_sha256"]:
            msg = f"Canonical case.h5 is not manifest-bound for {case_id}."
            raise GeneratedBatchError(msg)
        hdf5_identity = generation_storage.validate_case_hdf5(directory / "case.h5", expected_profile=manifest["simulation_profile"])
        if (
            hdf5_identity["case_input_id"] != record["case_input_id"]
            or hdf5_identity["simulation_case_id"] != record["simulation_case_id"]
            or hdf5_identity["git_commit"] != manifest["git_commit"]
        ):
            msg = f"Canonical HDF5 identity mismatch for {case_id}."
            raise GeneratedBatchError(msg)
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
            msg = (
                f"Terminal generated-batch membership mismatch under {root}: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}, unsafe={sorted(unsafe)}."
            )
            raise GeneratedBatchError(msg)


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
        raw_case, _raw_artifacts = _validate_case_publication(raw / record["case_id"], manifest=manifest, record=record, stage="raw")
        processed_case, _processed_artifacts = _validate_case_publication(
            processed / record["case_id"],
            manifest=manifest,
            record=record,
            stage="processed",
        )
        if raw_case != processed_case:
            msg = f"Raw and processed case metadata disagree for {record['case_id']}."
            raise GeneratedBatchError(msg)
    return manifest


def _steady_flow_fields(
    static: Mapping[str, np.ndarray],
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    task: TaskSpec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply the canonical steady-flow channel and permeability contract."""
    if task.id != "steady_flow":
        msg = f"Generated airflow views support only steady_flow, got {task.id!r}."
        raise ValueError(msg)
    x_grid: np.ndarray
    y_grid: np.ndarray
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    raw_kxx = static["Kxx"]
    raw_kxy = static["Kxy"]
    raw_kyy = static["Kyy"]
    determinant = raw_kxx * raw_kyy - raw_kxy**2
    if np.any(raw_kxx <= 0) or np.any(raw_kyy <= 0) or np.any(determinant <= 0):
        msg = "Canonical permeability tensor must be positive definite at every point."
        raise ValueError(msg)
    values: dict[str, np.ndarray] = {
        "x": x_grid,
        "y": y_grid,
        "Kxx": np.log10(raw_kxx),
        "Kxy": raw_kxy / np.sqrt(raw_kxx * raw_kyy),
        "Kyy": np.log10(raw_kyy),
        "eps_bed": static["eps_bed"],
        "p_in_bc": static["p_in_bc"],
        "p": static["p"],
        "u": static["u"],
        "v": static["v"],
    }
    if np.any((values["eps_bed"] <= 0) | (values["eps_bed"] > 1)):
        msg = "Canonical porosity must satisfy 0 < eps <= 1."
        raise ValueError(msg)
    expected = set(task.input_names) | set(task.output_names)
    if set(values) != expected:
        msg = "Current steady_flow TaskSpec no longer matches the canonical HDF5 view."
        raise RuntimeError(msg)
    converted = {name: np.asarray(value, dtype=np.float32).copy() for name, value in values.items()}
    if not all(np.isfinite(value).all() for value in converted.values()):
        msg = "Steady-flow learning view is non-finite after float32 conversion."
        raise ValueError(msg)
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
    """Interpret and fingerprint one validated canonical steady-flow HDF5 view."""
    if _CASE_ID_PATTERN.fullmatch(case_id) is None or case_id != record.get("case_id"):
        msg = f"Invalid manifest-bound case identifier: {case_id!r}."
        raise ValueError(msg)
    _meta, raw_root, processed_root = _batch_paths(batch_id, storage_root=storage_root)
    raw_case, _raw_artifacts = _validate_case_publication(raw_root / case_id, manifest=manifest, record=record, stage="raw")
    processed_case, processed_artifacts = _validate_case_publication(
        processed_root / case_id,
        manifest=manifest,
        record=record,
        stage="processed",
    )
    if raw_case != processed_case:
        msg = f"Raw and processed case metadata disagree for {case_id}."
        raise GeneratedBatchError(msg)
    if "steady_flow" not in manifest["available_learning_views"]:
        msg = f"Batch {batch_id!r} does not advertise a steady_flow view."
        raise ValueError(msg)
    hdf5_path = processed_root / case_id / "case.h5"
    with h5py.File(hdf5_path, "r") as handle:
        x_axis = np.asarray(_hdf5_dataset(handle, "coords/x"), dtype=np.float64)
        y_axis = np.asarray(_hdf5_dataset(handle, "coords/y"), dtype=np.float64)
        static_dataset = _hdf5_dataset(handle, "static/fields")
        static_values = np.asarray(static_dataset, dtype=np.float32)
        names = _string_list_attribute(static_dataset, "field_names")
    if names != list(generation_profiles.static_field_names(str(manifest["simulation_profile"]))):
        msg = f"Canonical static HDF5 field order is invalid for {case_id}."
        raise ValueError(msg)
    static = {name: static_values[index] for index, name in enumerate(names)}
    case_inputs, case_outputs = _steady_flow_fields(static, x_axis=x_axis, y_axis=y_axis, task=task)
    inputs = torch.stack([torch.from_numpy(case_inputs[name]) for name in task.input_names])
    outputs = torch.stack([torch.from_numpy(case_outputs[name]) for name in task.output_names])
    metadata = {
        "case_id": case_id,
        "case_index": raw_case["case_index"],
        "case_input_id": raw_case["case_input_id"],
        "simulation_case_id": raw_case["simulation_case_id"],
        "material_family": raw_case["material_family"],
        "sampling_regime": raw_case["sampling_regime"],
        "ood": raw_case["ood"],
        "simulation_profile": manifest["simulation_profile"],
        "available_learning_views": list(manifest["available_learning_views"]),
        "airflow_source": manifest["airflow_source"],
        "template_sha256": manifest["template"]["sha256"],
        "generator_version": raw_case["generator_version"],
        "seed": raw_case["seed_evidence"]["case_seed"],
        "parameters": raw_case["sampled_values"],
        "geometry": raw_case["spatial_diagnostics"]["geometry"],
        "schedule_class": (raw_case["schedule_diagnostics"]["schedule_class"] if "schedule_diagnostics" in raw_case else None),
    }
    source = {
        "case_id": case_id,
        "case_input_id": record["case_input_id"],
        "simulation_case_id": record["simulation_case_id"],
        "simulation_profile": manifest["simulation_profile"],
        "template_sha256": manifest["template"]["sha256"],
        "airflow_source": manifest["airflow_source"],
        "case_hdf5": processed_artifacts["case.h5"],
    }
    fingerprint = identity.compute_case_fingerprint(
        task=task,
        case_id=case_id,
        source_identity=source,
        source_metadata=metadata,
        inputs=inputs,
        outputs=outputs,
    )
    return (y_axis.size, x_axis.size), inputs, outputs, metadata, source, fingerprint


def source_provenance(manifest: Mapping[str, Any], *, manifest_sha256: str) -> dict[str, Any]:
    """Return explicit operational provenance for one canonical simulation batch."""
    return {
        "batch_manifest_sha256": manifest_sha256,
        "simulation_profile": manifest["simulation_profile"],
        "template_sha256": manifest["template"]["sha256"],
        "scientific_config_digest": manifest["scientific_config_digest"],
        "git_commit": manifest["git_commit"],
        "airflow_source": manifest["airflow_source"],
        "available_learning_views": list(manifest["available_learning_views"]),
        "cases": [
            {
                "case_id": record["case_id"],
                "material_family": record["material_family"],
                "case_input_id": record["case_input_id"],
                "simulation_case_id": record["simulation_case_id"],
                "case_hdf5_sha256": record["case_hdf5_sha256"],
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
    """Load a validated generated-batch prefix without publishing tensors."""
    task = domain.tasks.registry.get_task(task_id)
    if task.id != "steady_flow":
        msg = f"Generated airflow views support only steady_flow, got {task.id!r}."
        raise ValueError(msg)
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        msg = f"max_cases must be a positive integer or None, got {max_cases!r}."
        raise ValueError(msg)
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
