"""
===============================================================================
dataset_identity.py
===============================================================================
Define and verify persisted final-dataset and split identity contracts.

Responsibilities:
  - Validate the single current task-aware training-dataset schema
  - Compute stable case, dataset, and split-membership fingerprints
  - Bind persisted tensors and ordered sample identities to task contracts

Design principles:
  - Identity is content-addressed, deterministic, and fail-closed
  - Fingerprints are computed before publication and verified on consumption
  - Ordered membership remains distinct from dataset-level content identity

This module does NOT:
  - Load final datasets into model-ready samples. ``dataset_factory`` owns that
  - Choose training/evaluation ratios or random seeds. Experiment services own them
  - Publish dataset files. The generation-domain builder owns publication
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import domain

if TYPE_CHECKING:
    from torch import Tensor

    from src.domain.tasks.domain_task_spec import TaskSpec

TRAINING_DATASET_SCHEMA_VERSION = 1
TRAINING_DATASET_SCHEMA_KIND = "training_dataset"
GENERATED_BATCH_IDENTITY_SCHEMA_VERSION = 2
CASE_FINGERPRINT_VERSION = 1
SPLIT_SCHEMA_VERSION = 1
_SHA256_HEX_LENGTH = 64
_TENSOR_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_FINITE_CHECK_CHUNK_ELEMENTS = 1024 * 1024
_GENERATED_BATCH_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "batch_id",
        "simulation_profile",
        "batch_identity",
        "scientific_config_digest",
        "template",
        "export_contract_sha256",
        "available_learning_views",
        "airflow_source",
        "intended_case_ids",
        "cases",
        "batch_manifest_identity_sha256",
    }
)
_GENERATED_TEMPLATE_KEYS = frozenset({"relative_path", "sha256"})
_GENERATED_CASE_KEYS = frozenset(
    {
        "case_id",
        "material_family",
        "case_input_id",
        "simulation_case_id",
        "case_hdf5_sha256",
        "success_sha256",
        "provenance_sha256",
    }
)
_PROFILE_METADATA = {
    "steady_flow": (("steady_flow",), "comsol_steady_reference"),
    "transient_drying": (("steady_flow", "transient_drying"), "comsol_coupled_reference"),
}

_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "schema_kind",
        "dataset_id",
        "task",
        "data_contract_digest",
        "fields",
        "tensor_layout",
        "sample_count",
        "spatial_shape",
        "sample_ids",
        "generated_batch_identity",
        "source_identities",
        "source_metadata",
        "source_provenance",
        "case_fingerprints",
        "tensor_metadata",
        "inputs",
        "outputs",
        "dataset_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """
    Portable ordered identity returned after strict dataset validation.

    ``data_contract_digest`` identifies learned-data compatibility, distinct
    from the complete run TaskSpec contract digest.
    """

    dataset_id: str
    task: str
    data_contract_digest: str
    fingerprint: str
    sample_ids: tuple[str, ...]
    sample_count: int
    spatial_shape: tuple[int, ...]
    generated_batch_identity_sha256: str | None = field(default=None, compare=False, repr=False)
    generated_batch_identity: dict[str, Any] | None = field(default=None, compare=False, repr=False)
    source_metadata: tuple[dict[str, Any], ...] | None = field(default=None, compare=False, repr=False)
    source_provenance: dict[str, Any] | None = field(default=None, compare=False, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation used by split identity."""
        return {
            "dataset_id": self.dataset_id,
            "task": self.task,
            "data_contract_digest": self.data_contract_digest,
            "fingerprint": self.fingerprint,
            "sample_ids": list(self.sample_ids),
            "sample_count": self.sample_count,
            "spatial_shape": list(self.spatial_shape),
        }


def _canonical_json(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        msg = f"{label} must be JSON-serializable without non-finite values."
        raise TypeError(msg) from error


def _update_hash(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def _tensor_dtype(tensor: Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _update_tensor_hash(hasher: Any, tensor: Tensor) -> None:
    import torch  # noqa: PLC0415

    contiguous = tensor.detach().cpu()
    if not contiguous.is_contiguous():
        contiguous = contiguous.contiguous()
    byte_count = contiguous.numel() * contiguous.element_size()
    hasher.update(byte_count.to_bytes(8, byteorder="big", signed=False))
    byte_view = contiguous.view(torch.uint8).numpy().data.cast("B")
    for offset in range(0, len(byte_view), _TENSOR_HASH_CHUNK_BYTES):
        hasher.update(byte_view[offset : offset + _TENSOR_HASH_CHUNK_BYTES])


def _content_fingerprint(metadata: Mapping[str, Any], tensors: Sequence[tuple[str, Tensor]]) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, _canonical_json(dict(metadata), label="Fingerprint metadata"))
    for label, tensor in tensors:
        _update_hash(hasher, label.encode("utf-8"))
        _update_tensor_hash(hasher, tensor)
    return hasher.hexdigest()


def _require_non_empty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string."
        raise TypeError(msg)
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    digest = _require_non_empty_string(value, label=label)
    if len(digest) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in digest):
        msg = f"{label} must be a 64-character lowercase hexadecimal SHA-256 digest."
        raise ValueError(msg)
    return digest


def _require_string_sequence(value: Any, *, label: str, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        msg = f"{label} must be a list or tuple of strings."
        raise TypeError(msg)
    values = tuple(_require_non_empty_string(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if unique and len(values) != len(set(values)):
        duplicates = sorted({item for item in values if values.count(item) > 1})
        msg = f"{label} contains duplicate identifiers: {duplicates}."
        raise ValueError(msg)
    return values


def _require_sha256_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    values = _require_string_sequence(value, label=label, unique=False)
    return tuple(_require_sha256(item, label=f"{label}[{index}]") for index, item in enumerate(values))


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer."
        raise TypeError(msg)
    if value <= 0:
        msg = f"{label} must be positive, got {value}."
        raise ValueError(msg)
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    return value


def _json_mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    mapping = dict(_require_mapping(value, label=label))
    return json.loads(_canonical_json(mapping, label=label).decode("utf-8"))


def _require_exact_mapping(
    value: Any,
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    """Return an isolated JSON mapping with exactly the declared keys."""
    normalized = _json_mapping_copy(value, label=label)
    missing = sorted(set(expected).difference(normalized))
    unexpected = sorted(set(normalized).difference(expected))
    if missing or unexpected:
        msg = f"{label} keys do not match. Missing: {missing}. Unexpected: {unexpected}."
        raise ValueError(msg)
    return normalized


def _validate_generated_batch_identity(
    value: Any,
    *,
    sample_ids: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Validate the current profile-qualified generated-batch identity."""
    batch_identity = _require_exact_mapping(value, _GENERATED_BATCH_IDENTITY_KEYS, label=label)
    schema_version = batch_identity["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != GENERATED_BATCH_IDENTITY_SCHEMA_VERSION:
        msg = f"{label}.schema_version must be integer {GENERATED_BATCH_IDENTITY_SCHEMA_VERSION}."
        raise ValueError(msg)
    _require_non_empty_string(batch_identity["batch_id"], label=f"{label}.batch_id")
    profile = _require_non_empty_string(
        batch_identity["simulation_profile"],
        label=f"{label}.simulation_profile",
    )
    if profile not in _PROFILE_METADATA:
        msg = f"{label}.simulation_profile is unsupported: {profile!r}."
        raise ValueError(msg)
    _require_sha256(batch_identity["batch_identity"], label=f"{label}.batch_identity")
    _require_sha256(batch_identity["scientific_config_digest"], label=f"{label}.scientific_config_digest")
    _require_sha256(
        batch_identity["export_contract_sha256"],
        label=f"{label}.export_contract_sha256",
    )
    template = _require_exact_mapping(
        batch_identity["template"],
        _GENERATED_TEMPLATE_KEYS,
        label=f"{label}.template",
    )
    relative_path = _require_non_empty_string(
        template["relative_path"],
        label=f"{label}.template.relative_path",
    )
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts or not relative_path.endswith(".mph"):
        msg = f"{label}.template.relative_path must be one safe relative .mph path."
        raise ValueError(msg)
    _require_sha256(template["sha256"], label=f"{label}.template.sha256")
    expected_views, expected_source = _PROFILE_METADATA[profile]
    views = _require_string_sequence(
        batch_identity["available_learning_views"],
        label=f"{label}.available_learning_views",
        unique=True,
    )
    if views != expected_views or batch_identity["airflow_source"] != expected_source:
        msg = f"{label} has learning-view or airflow provenance inconsistent with {profile!r}."
        raise ValueError(msg)
    intended = _require_string_sequence(
        batch_identity["intended_case_ids"],
        label=f"{label}.intended_case_ids",
        unique=True,
    )
    if intended != tuple(sample_ids):
        msg = f"{label}.intended_case_ids must equal the dataset sample_ids in order."
        raise ValueError(msg)
    raw_cases = batch_identity["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != len(intended):
        msg = f"{label}.cases must align one-to-one with intended_case_ids."
        raise ValueError(msg)
    normalized_cases: list[dict[str, Any]] = []
    for index, (case_id, raw_case) in enumerate(zip(intended, raw_cases, strict=True)):
        case_label = f"{label}.cases[{index}]"
        case = _require_exact_mapping(raw_case, _GENERATED_CASE_KEYS, label=case_label)
        if case["case_id"] != case_id:
            msg = f"{label}.cases must follow intended_case_ids exactly."
            raise ValueError(msg)
        _require_non_empty_string(case["material_family"], label=f"{case_label}.material_family")
        for key in ("case_input_id", "simulation_case_id", "case_hdf5_sha256", "success_sha256", "provenance_sha256"):
            _require_sha256(case[key], label=f"{case_label}.{key}")
        normalized_cases.append(case)
    batch_identity["template"] = template
    batch_identity["cases"] = normalized_cases
    content = {key: batch_identity[key] for key in batch_identity if key != "batch_manifest_identity_sha256"}
    expected_digest = hashlib.sha256(_canonical_json(content, label=label)).hexdigest()
    actual_digest = _require_sha256(
        batch_identity["batch_manifest_identity_sha256"],
        label=f"{label}.batch_manifest_identity_sha256",
    )
    if actual_digest != expected_digest:
        msg = f"{label}.batch_manifest_identity_sha256 does not match its content."
        raise ValueError(msg)
    return batch_identity


def build_generated_batch_identity(source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build the current scientific identity from one admitted terminal manifest."""
    manifest = _require_mapping(source_manifest, label="Source manifest")
    indices = manifest.get("intended_case_indices")
    if not isinstance(indices, list):
        msg = "Source manifest.intended_case_indices must be a list."
        raise TypeError(msg)
    intended = [f"case_{index:04d}" for index in indices]
    records = manifest.get("cases")
    if not isinstance(records, list):
        msg = "Source manifest.cases must be a list."
        raise TypeError(msg)
    cases: list[dict[str, Any]] = []
    for index, record_value in enumerate(records):
        record = _require_mapping(record_value, label=f"Source manifest.cases[{index}]")
        cases.append(
            {
                "case_id": record.get("case_id"),
                "material_family": record.get("material_family"),
                "case_input_id": record.get("case_input_id"),
                "simulation_case_id": record.get("simulation_case_id"),
                "case_hdf5_sha256": record.get("case_hdf5_sha256"),
                "success_sha256": record.get("success_sha256"),
                "provenance_sha256": record.get("provenance_sha256"),
            }
        )
    content: dict[str, Any] = {
        "schema_version": GENERATED_BATCH_IDENTITY_SCHEMA_VERSION,
        "batch_id": manifest.get("batch_id"),
        "simulation_profile": manifest.get("simulation_profile"),
        "batch_identity": manifest.get("batch_identity"),
        "scientific_config_digest": manifest.get("scientific_config_digest"),
        "template": manifest.get("template"),
        "export_contract_sha256": manifest.get("export_contract_sha256"),
        "available_learning_views": manifest.get("available_learning_views"),
        "airflow_source": manifest.get("airflow_source"),
        "intended_case_ids": intended,
        "cases": cases,
    }
    result = dict(content)
    result["batch_manifest_identity_sha256"] = hashlib.sha256(_canonical_json(content, label="Generated batch identity")).hexdigest()
    return _validate_generated_batch_identity(
        result,
        sample_ids=intended,
        label="Generated batch identity",
    )


def build_generated_package_identity(
    *,
    dataset_name: str,
    simulation_profile: str,
    campaign_digest: str,
    template: Mapping[str, Any],
    export_contract_sha256: str,
    available_learning_views: Sequence[str],
    airflow_source: str,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one aggregate generated-source identity for a multi-batch package."""
    normalized_cases = [_json_mapping_copy(case, label=f"Package cases[{index}]") for index, case in enumerate(cases)]
    sample_ids = [str(case.get("case_id")) for case in normalized_cases]
    source_digest_payload = {
        "dataset_name": dataset_name,
        "simulation_profile": simulation_profile,
        "campaign_digest": campaign_digest,
        "template": dict(template),
        "export_contract_sha256": export_contract_sha256,
        "available_learning_views": list(available_learning_views),
        "airflow_source": airflow_source,
        "cases": normalized_cases,
    }
    source_digest = hashlib.sha256(_canonical_json(source_digest_payload, label="Generated package source identity")).hexdigest()
    content: dict[str, Any] = {
        "schema_version": GENERATED_BATCH_IDENTITY_SCHEMA_VERSION,
        "batch_id": dataset_name,
        "simulation_profile": simulation_profile,
        "batch_identity": source_digest,
        "scientific_config_digest": campaign_digest,
        "template": dict(template),
        "export_contract_sha256": export_contract_sha256,
        "available_learning_views": list(available_learning_views),
        "airflow_source": airflow_source,
        "intended_case_ids": sample_ids,
        "cases": normalized_cases,
    }
    result = dict(content)
    result["batch_manifest_identity_sha256"] = hashlib.sha256(_canonical_json(content, label="Generated package identity")).hexdigest()
    return _validate_generated_batch_identity(
        result,
        sample_ids=sample_ids,
        label="Generated package identity",
    )


def _require_tensor(value: Any, *, label: str, rank: int) -> Tensor:
    import torch  # noqa: PLC0415

    if not isinstance(value, torch.Tensor):
        msg = f"{label} must be a torch.Tensor."
        raise TypeError(msg)
    tensor = value
    if tensor.layout != torch.strided:
        msg = f"{label} must be a dense strided tensor."
        raise TypeError(msg)
    if tensor.ndim != rank:
        msg = f"{label} must have rank {rank}, got shape {tuple(tensor.shape)}."
        raise ValueError(msg)
    if tensor.dtype != torch.float32:
        msg = f"{label} must use torch.float32, got {tensor.dtype}."
        raise TypeError(msg)
    cpu_tensor = tensor.detach().cpu()
    flat = cpu_tensor.reshape(-1)
    for offset in range(0, flat.numel(), _FINITE_CHECK_CHUNK_ELEMENTS):
        if not bool(torch.isfinite(flat[offset : offset + _FINITE_CHECK_CHUNK_ELEMENTS]).all().item()):
            msg = f"{label} must contain only finite values."
            raise ValueError(msg)
    return cpu_tensor


def _tensor_metadata(tensor: Tensor) -> dict[str, Any]:
    return {"dtype": _tensor_dtype(tensor), "shape": list(tensor.shape)}


def _require_exact_keys(payload: Mapping[str, Any], *, label: str) -> None:
    missing = sorted(_REQUIRED_KEYS.difference(payload))
    unexpected = sorted(set(payload).difference(_REQUIRED_KEYS))
    if missing or unexpected:
        msg = f"{label} schema keys do not match. Missing: {missing}. Unexpected: {unexpected}."
        raise ValueError(msg)


def validate_dataset_data_contract_digest(
    value: Any,
    *,
    task: TaskSpec,
    label: str = "Dataset data_contract_digest",
) -> str:
    """
    Admit one persisted digest against the task's learned-data contract.

    Parameters
    ----------
    value : Any
        Persisted digest from a dataset payload, identity, or metadata package.
    task : TaskSpec
        Authoritative current task whose learned-data contract must match.
    label : str, optional
        Evidence label used in validation errors.

    Returns
    -------
    str
        The exact admitted persisted digest.

    Raises
    ------
    TypeError
        If ``value`` is not a non-empty string.
    ValueError
        If ``value`` is not a SHA-256 digest or is not bound to the current
        learned-data contract.

    Notes
    -----
    Datasets persist only ``TaskSpec.data_contract_digest``. Complete-task
    digests and prior data-contract digests are rejected without conversion.

    """
    digest = _require_sha256(value, label=label)
    if digest != task.data_contract_digest:
        msg = f"{label} is not compatible with the learned-data contract for task {task.id!r}."
        raise ValueError(msg)
    return digest


def _validate_task_header(payload: Mapping[str, Any], task: TaskSpec, *, label: str) -> None:
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != TRAINING_DATASET_SCHEMA_VERSION:
        msg = f"{label} schema_version must be integer {TRAINING_DATASET_SCHEMA_VERSION}."
        raise ValueError(msg)
    if payload.get("schema_kind") != TRAINING_DATASET_SCHEMA_KIND:
        msg = f"{label} schema_kind must be {TRAINING_DATASET_SCHEMA_KIND!r}."
        raise ValueError(msg)
    if payload.get("task") != task.id:
        msg = f"{label} task must be {task.id!r}, got {payload.get('task')!r}."
        raise ValueError(msg)
    validate_dataset_data_contract_digest(
        payload.get("data_contract_digest"),
        task=task,
        label=f"{label}.data_contract_digest",
    )
    fields = _require_mapping(payload.get("fields"), label=f"{label}.fields")
    if set(fields) != {"inputs", "outputs"}:
        msg = f"{label}.fields must contain exactly 'inputs' and 'outputs'."
        raise ValueError(msg)
    inputs = _require_string_sequence(fields["inputs"], label=f"{label}.fields.inputs", unique=True)
    outputs = _require_string_sequence(fields["outputs"], label=f"{label}.fields.outputs", unique=True)
    domain.field_sets.validate_ordered_fields(inputs, task.input_names, label=f"{label}.fields.inputs")
    domain.field_sets.validate_ordered_fields(outputs, task.output_names, label=f"{label}.fields.outputs")
    layout = _require_string_sequence(payload.get("tensor_layout"), label=f"{label}.tensor_layout", unique=True)
    if layout != task.tensor_layout:
        msg = f"{label}.tensor_layout must equal {list(task.tensor_layout)}, got {list(layout)}."
        raise ValueError(msg)


def source_file_identity(path: Path | str) -> dict[str, Any]:
    """Return basename, exact size, and streaming SHA-256 for one source file."""
    source_path = Path(path)
    hasher = hashlib.sha256()
    size = 0
    with source_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
    return {"name": source_path.name, "size_bytes": size, "sha256": hasher.hexdigest()}


def canonical_metadata_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical byte count and SHA-256 of portable JSON metadata."""
    encoded = _canonical_json(dict(value), label="Source metadata identity")
    return {"canonical_size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def compute_case_fingerprint(
    *,
    task: TaskSpec,
    case_id: str,
    source_identity: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    inputs: Tensor,
    outputs: Tensor,
) -> str:
    """Compute one in-memory scientific case fingerprint without persistence."""
    normalized_case_id = _require_non_empty_string(case_id, label="case_id")
    rank = len(task.tensor_layout) - 1
    normalized_inputs = _require_tensor(inputs, label="case inputs", rank=rank)
    normalized_outputs = _require_tensor(outputs, label="case outputs", rank=rank)
    if normalized_inputs.shape[0] != task.in_channels or normalized_outputs.shape[0] != task.out_channels:
        msg = "Case tensor channel counts do not match the task contract."
        raise ValueError(msg)
    if normalized_inputs.shape[1:] != normalized_outputs.shape[1:]:
        msg = "Case input/output spatial shapes differ."
        raise ValueError(msg)
    if normalized_inputs.dtype != normalized_outputs.dtype:
        msg = "Case input/output tensor dtypes differ."
        raise ValueError(msg)
    metadata = {
        "case_fingerprint_version": CASE_FINGERPRINT_VERSION,
        "task": task.id,
        "data_contract_digest": task.data_contract_digest,
        "fields": {"inputs": list(task.input_names), "outputs": list(task.output_names)},
        "tensor_layout": list(task.tensor_layout[1:]),
        "case_id": normalized_case_id,
        "source_identity": _json_mapping_copy(source_identity, label="source_identity"),
        "source_metadata": _json_mapping_copy(source_metadata, label="source_metadata"),
        "tensor_metadata": {
            "inputs": _tensor_metadata(normalized_inputs),
            "outputs": _tensor_metadata(normalized_outputs),
        },
    }
    return _content_fingerprint(metadata, (("inputs", normalized_inputs), ("outputs", normalized_outputs)))


def build_training_dataset_payload(
    *,
    task: TaskSpec,
    dataset_id: str,
    sample_ids: Sequence[str],
    generated_batch_identity: Mapping[str, Any],
    source_identities: Sequence[Mapping[str, Any]],
    source_metadata: Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Any],
    case_fingerprints: Sequence[str],
    inputs: Tensor,
    outputs: Tensor,
) -> dict[str, Any]:
    """Build the only maintained final training-dataset payload."""
    normalized_dataset_id = _require_non_empty_string(dataset_id, label="dataset_id")
    normalized_sample_ids = _require_string_sequence(sample_ids, label="sample_ids", unique=True)
    normalized_case_fingerprints = _require_sha256_sequence(case_fingerprints, label="case_fingerprints")
    normalized_batch_identity = _validate_generated_batch_identity(
        generated_batch_identity,
        sample_ids=normalized_sample_ids,
        label="generated_batch_identity",
    )
    normalized_sources = [_json_mapping_copy(value, label=f"source_identities[{index}]") for index, value in enumerate(source_identities)]
    normalized_metadata = [_json_mapping_copy(value, label=f"source_metadata[{index}]") for index, value in enumerate(source_metadata)]
    normalized_provenance = _json_mapping_copy(source_provenance, label="source_provenance")
    sample_count = len(normalized_sample_ids)
    if sample_count <= 0:
        msg = "Training datasets must contain at least one sample."
        raise ValueError(msg)
    if not (len(normalized_sources) == len(normalized_metadata) == len(normalized_case_fingerprints) == sample_count):
        msg = "sample_ids, source identities, source metadata, and case fingerprints must align one-to-one."
        raise ValueError(msg)
    rank = len(task.tensor_layout)
    normalized_inputs = _require_tensor(inputs, label="inputs", rank=rank)
    normalized_outputs = _require_tensor(outputs, label="outputs", rank=rank)
    if normalized_inputs.shape[0] != sample_count or normalized_outputs.shape[0] != sample_count:
        msg = f"Dataset tensor sample counts must equal {sample_count}."
        raise ValueError(msg)
    if normalized_inputs.shape[1] != task.in_channels or normalized_outputs.shape[1] != task.out_channels:
        msg = "Dataset tensor channel counts do not match the task contract."
        raise ValueError(msg)
    if normalized_inputs.shape[2:] != normalized_outputs.shape[2:]:
        msg = "Dataset input/output spatial shapes differ."
        raise ValueError(msg)
    if normalized_inputs.dtype != normalized_outputs.dtype:
        msg = "Dataset input/output tensor dtypes differ."
        raise ValueError(msg)
    payload: dict[str, Any] = {
        "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
        "schema_kind": TRAINING_DATASET_SCHEMA_KIND,
        "dataset_id": normalized_dataset_id,
        "task": task.id,
        "data_contract_digest": task.data_contract_digest,
        "fields": {"inputs": list(task.input_names), "outputs": list(task.output_names)},
        "tensor_layout": list(task.tensor_layout),
        "sample_count": sample_count,
        "spatial_shape": list(normalized_inputs.shape[2:]),
        "sample_ids": list(normalized_sample_ids),
        "generated_batch_identity": normalized_batch_identity,
        "source_identities": normalized_sources,
        "source_metadata": normalized_metadata,
        "source_provenance": normalized_provenance,
        "case_fingerprints": list(normalized_case_fingerprints),
        "tensor_metadata": {
            "inputs": _tensor_metadata(normalized_inputs),
            "outputs": _tensor_metadata(normalized_outputs),
        },
        "inputs": normalized_inputs,
        "outputs": normalized_outputs,
        "dataset_fingerprint": "",
    }
    fingerprint_metadata = {
        key: value for key, value in payload.items() if key not in {"inputs", "outputs", "dataset_fingerprint", "dataset_id", "source_provenance"}
    }
    payload["dataset_fingerprint"] = _content_fingerprint(
        fingerprint_metadata,
        (("inputs", normalized_inputs), ("outputs", normalized_outputs)),
    )
    validate_training_dataset_payload(payload, task=task)
    return payload


def validate_training_dataset_payload(
    payload: Any,
    *,
    task: TaskSpec,
    verify_content: bool = False,
) -> DatasetIdentity:
    """Validate one version-1 training dataset and optionally rehash all content."""
    mapping = _require_mapping(payload, label="Training dataset")
    _require_exact_keys(mapping, label="Training dataset")
    _validate_task_header(mapping, task, label="Training dataset")
    dataset_id = _require_non_empty_string(mapping.get("dataset_id"), label="Training dataset.dataset_id")
    sample_count = _require_positive_int(mapping.get("sample_count"), label="Training dataset.sample_count")
    sample_ids = _require_string_sequence(mapping.get("sample_ids"), label="Training dataset.sample_ids", unique=True)
    if len(sample_ids) != sample_count:
        msg = f"Training dataset sample_count={sample_count} does not match {len(sample_ids)} sample_ids."
        raise ValueError(msg)
    generated_batch_identity = _validate_generated_batch_identity(
        mapping.get("generated_batch_identity"),
        sample_ids=sample_ids,
        label="Training dataset.generated_batch_identity",
    )
    source_identities = mapping.get("source_identities")
    source_metadata = mapping.get("source_metadata")
    if not isinstance(source_identities, (list, tuple)) or len(source_identities) != sample_count:
        msg = "Training dataset.source_identities must align one-to-one with sample_ids."
        raise ValueError(msg)
    if not isinstance(source_metadata, (list, tuple)) or len(source_metadata) != sample_count:
        msg = "Training dataset.source_metadata must align one-to-one with sample_ids."
        raise ValueError(msg)
    normalized_sources = [
        _json_mapping_copy(value, label=f"Training dataset.source_identities[{index}]") for index, value in enumerate(source_identities)
    ]
    normalized_metadata = [
        _json_mapping_copy(value, label=f"Training dataset.source_metadata[{index}]") for index, value in enumerate(source_metadata)
    ]
    source_provenance = _json_mapping_copy(
        mapping.get("source_provenance"),
        label="Training dataset.source_provenance",
    )
    case_fingerprints = _require_sha256_sequence(
        mapping.get("case_fingerprints"),
        label="Training dataset.case_fingerprints",
    )
    if len(case_fingerprints) != sample_count:
        msg = "Training dataset.case_fingerprints must align one-to-one with sample_ids."
        raise ValueError(msg)
    rank = len(task.tensor_layout)
    inputs = _require_tensor(mapping.get("inputs"), label="Training dataset.inputs", rank=rank)
    outputs = _require_tensor(mapping.get("outputs"), label="Training dataset.outputs", rank=rank)
    if inputs.shape[0] != sample_count or outputs.shape[0] != sample_count:
        msg = f"Training dataset tensor sample counts must equal {sample_count}."
        raise ValueError(msg)
    if inputs.shape[1] != task.in_channels or outputs.shape[1] != task.out_channels:
        msg = "Training dataset tensor channel counts do not match task fields."
        raise ValueError(msg)
    if inputs.shape[2:] != outputs.shape[2:]:
        msg = "Training dataset input/output spatial shapes differ."
        raise ValueError(msg)
    spatial_shape_value = mapping.get("spatial_shape")
    if not isinstance(spatial_shape_value, (list, tuple)) or len(spatial_shape_value) != rank - 2:
        msg = f"Training dataset.spatial_shape must contain exactly {rank - 2} dimensions."
        raise ValueError(msg)
    spatial_shape = tuple(_require_positive_int(value, label="spatial_shape") for value in spatial_shape_value)
    if tuple(inputs.shape[2:]) != spatial_shape:
        msg = f"Training dataset spatial_shape {spatial_shape} does not match tensors {tuple(inputs.shape[2:])}."
        raise ValueError(msg)
    tensor_metadata = _require_mapping(mapping.get("tensor_metadata"), label="Training dataset.tensor_metadata")
    expected_tensor_metadata = {"inputs": _tensor_metadata(inputs), "outputs": _tensor_metadata(outputs)}
    if dict(tensor_metadata) != expected_tensor_metadata:
        msg = "Training dataset tensor_metadata does not match its tensors."
        raise ValueError(msg)
    fingerprint = _require_sha256(mapping.get("dataset_fingerprint"), label="Training dataset.dataset_fingerprint")
    if verify_content:
        normalized_mapping = dict(mapping)
        normalized_mapping["generated_batch_identity"] = generated_batch_identity
        normalized_mapping["source_identities"] = normalized_sources
        normalized_mapping["source_metadata"] = normalized_metadata
        normalized_mapping["source_provenance"] = source_provenance
        fingerprint_metadata = {
            key: value
            for key, value in normalized_mapping.items()
            if key not in {"inputs", "outputs", "dataset_fingerprint", "dataset_id", "source_provenance"}
        }
        expected = _content_fingerprint(fingerprint_metadata, (("inputs", inputs), ("outputs", outputs)))
        if fingerprint != expected:
            msg = f"Training dataset fingerprint mismatch for {dataset_id!r}."
            raise ValueError(msg)
    return DatasetIdentity(
        dataset_id=dataset_id,
        task=task.id,
        data_contract_digest=str(mapping["data_contract_digest"]),
        fingerprint=fingerprint,
        sample_ids=sample_ids,
        sample_count=sample_count,
        spatial_shape=spatial_shape,
        generated_batch_identity_sha256=str(generated_batch_identity["batch_manifest_identity_sha256"]),
        generated_batch_identity=generated_batch_identity,
        source_metadata=tuple(normalized_metadata),
        source_provenance=source_provenance,
    )


def membership_digest(
    *,
    role: str,
    dataset_fingerprint: str,
    sample_ids: Sequence[str],
    indices: Sequence[int],
) -> str:
    """Hash exact ordered split membership against one dataset fingerprint."""
    normalized_role = _require_non_empty_string(role, label="role")
    normalized_fingerprint = _require_sha256(dataset_fingerprint, label="dataset_fingerprint")
    normalized_sample_ids = _require_string_sequence(sample_ids, label="sample_ids", unique=True)
    normalized_indices: list[int] = []
    for position, value in enumerate(indices):
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"indices[{position}] must be an integer."
            raise TypeError(msg)
        if value < 0 or value >= len(normalized_sample_ids):
            msg = f"indices[{position}]={value} is out of bounds for {len(normalized_sample_ids)} sample_ids."
            raise IndexError(msg)
        normalized_indices.append(value)
    if len(normalized_indices) != len(set(normalized_indices)):
        msg = "indices must not contain duplicates."
        raise ValueError(msg)
    content = {
        "role": normalized_role,
        "dataset_fingerprint": normalized_fingerprint,
        "indices": normalized_indices,
        "sample_ids": [normalized_sample_ids[index] for index in normalized_indices],
    }
    return hashlib.sha256(_canonical_json(content, label="Membership payload")).hexdigest()
