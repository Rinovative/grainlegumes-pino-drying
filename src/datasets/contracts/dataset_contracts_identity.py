"""Define and verify persisted final-dataset and split identity contracts.

Responsibilities:
  - Validate the single current task-aware training-dataset schema
  - Compute stable package, case, dataset, and split-membership fingerprints
  - Bind persisted tensors and ordered sample identities to task contracts

Design principles:
  - Identity is content-addressed, deterministic, and fail-closed
  - Fingerprints are computed before publication and verified on consumption
  - Ordered membership remains distinct from dataset-level content identity

This module does NOT:
  - Load final datasets into model-ready samples. ``src.datasets.runtime.factory`` owns that
  - Choose training/evaluation ratios or random seeds. The authored Dataset plan owns them
  - Publish dataset files. Dataset package services own publication
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from src import domain

if TYPE_CHECKING:
    from torch import Tensor

    from src.domain.tasks.domain_task_spec import TaskSpec

PACKAGE_SELECTION_SCHEMA_VERSION = 3
TRAINING_DATASET_SCHEMA_VERSION = 1
TRAINING_DATASET_SCHEMA_KIND = "dataset_steady_tensor_payload"
CASE_FINGERPRINT_VERSION = 1
TRAINING_TENSOR_DTYPE: Final = "float32"
_SHA256_HEX_LENGTH = 64
_TENSOR_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_FINITE_CHECK_CHUNK_ELEMENTS = 1024 * 1024
_DATASET_CONVERSION_CONTRACTS: Final = {
    "steady_flow": {
        "schema_kind": "dataset_conversion_contract",
        "schema_version": 1,
        "source_schema_version": 1,
        "tensor_dtype": TRAINING_TENSOR_DTYPE,
        "case_fingerprint_version": CASE_FINGERPRINT_VERSION,
        "permeability_representation": {
            "Kxx": "log10",
            "Kxy": "Kxy/sqrt(Kxx*Kyy)",
            "Kyy": "log10",
        },
        "payload_semantics": "ordered_eager_task_tensors",
    },
    "transient_drying": {
        "schema_kind": "dataset_conversion_contract",
        "schema_version": 1,
        "source_schema_version": 1,
        "index_schema_version": 1,
        "transition_selection": "consecutive_regular_states_excluding_irregular_stop",
        "target_semantics": "next_state_minus_current_state",
        "payload_semantics": "portable_source_hdf5_transition_index",
    },
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
        "source_selection_identity",
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
    source_selection_identity: dict[str, Any] | None = field(default=None, compare=False, repr=False)
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


def combined_dataset_id(dataset_ids: Sequence[str]) -> str:
    """Return the stable logical identity for one ordered OOD package selection."""
    identifiers = tuple(dataset_ids)
    if not identifiers or any(not isinstance(item, str) or not item for item in identifiers):
        msg = "Combined OOD package identifiers must be non-empty strings."
        raise ValueError(msg)
    if len(identifiers) != len(set(identifiers)):
        msg = "Combined OOD package identifiers must be unique."
        raise ValueError(msg)
    if len(identifiers) == 1:
        return identifiers[0]
    payload = json.dumps(
        list(identifiers),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"combined_ood__{hashlib.sha256(payload).hexdigest()[:16]}"


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


def dataset_conversion_contract_identity(dataset_view: str) -> dict[str, Any]:
    """Return the maintained content-conversion contract for one Dataset view."""
    contract = _DATASET_CONVERSION_CONTRACTS.get(dataset_view)
    if contract is None:
        message = f"Unsupported Dataset conversion view {dataset_view!r}."
        raise ValueError(message)
    digest = hashlib.sha256(_canonical_json(contract, label="Dataset conversion contract")).hexdigest()
    return {
        "schema_kind": "dataset_conversion_identity",
        "schema_version": 1,
        "dataset_view": dataset_view,
        "contract_sha256": digest,
    }


def package_semantic_identity_payload(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Select the positive content and interpretation dependencies of a package."""
    value = dict(provenance)
    if value.get("schema_kind") != "dataset_selection_manifest" or value.get("schema_version") != PACKAGE_SELECTION_SCHEMA_VERSION:
        msg = "Dataset package uses an unsupported selection contract."
        raise ValueError(msg)
    dataset_view = _require_non_empty_string(value.get("dataset_view"), label="Dataset package dataset_view")
    expected_conversion = dataset_conversion_contract_identity(dataset_view)
    builder = value.get("builder_identity")
    conversion = {key: builder.get(key) for key in expected_conversion} if isinstance(builder, Mapping) else None
    if conversion != expected_conversion:
        message = "Dataset package builder_identity does not match the maintained conversion contract."
        raise ValueError(message)
    if not isinstance(builder, Mapping) or set(builder) != set(expected_conversion) | {"implementation_sha256"}:
        message = "Dataset builder implementation evidence is invalid."
        raise ValueError(message)
    _require_sha256(builder["implementation_sha256"], label="Dataset builder implementation")
    source_cases = value.get("source_case_identities")
    if not isinstance(source_cases, list) or not source_cases:
        message = "Dataset package source_case_identities must be a non-empty ordered list."
        raise ValueError(message)
    projected_cases: list[dict[str, Any]] = []
    package_ids: set[str] = set()
    physical_inputs: set[str] = set()
    simulations: set[str] = set()
    for index, source_case in enumerate(source_cases):
        if not isinstance(source_case, Mapping) or set(source_case) != {"package_case_id", "source", "design"}:
            msg = f"Dataset source_case_identities[{index}] has an unsupported contract."
            raise ValueError(msg)
        if not isinstance(source_case["source"], Mapping) or not isinstance(source_case["design"], Mapping):
            msg = f"Dataset source_case_identities[{index}] lacks source or design evidence."
            raise TypeError(msg)
        package_id = _require_non_empty_string(source_case["package_case_id"], label="package_case_id")
        source = source_case["source"]
        design = source_case["design"]
        required_source = {
            "task",
            "campaign_name",
            "campaign_digest",
            "source_plan_sha256",
            "material_family",
            "domain_profile",
            "source_batch_id",
            "case_id",
            "case_input_id",
            "simulation_case_id",
            "case_hdf5_relative",
            "case_hdf5_sha256",
            "reservation_sha256",
            "decision_sha256",
        }
        required_design = {
            "sampling_method",
            "original_design_size",
            "candidate_position",
            "origin",
            "root_seed",
            "candidate_seed",
            "derived_seeds",
            "design_evidence",
        }
        if not required_source.issubset(source) or not required_design.issubset(design):
            msg = f"Dataset source_case_identities[{index}] lacks exact source or design provenance."
            raise ValueError(msg)
        for key in ("campaign_digest", "source_plan_sha256", "case_hdf5_sha256", "reservation_sha256", "decision_sha256"):
            _require_sha256(source[key], label=f"source.{key}")
        if source.get("selected_pool_sha256") is not None:
            _require_sha256(source["selected_pool_sha256"], label="source.selected_pool_sha256")
        original_size = design["original_design_size"]
        candidate_position = design["candidate_position"]
        if (
            isinstance(original_size, bool)
            or not isinstance(original_size, int)
            or original_size < 1
            or isinstance(candidate_position, bool)
            or not isinstance(candidate_position, int)
            or candidate_position < 0
        ):
            msg = "Dataset source original design size or candidate position is invalid."
            raise ValueError(msg)
        if package_id in package_ids or source["case_input_id"] in physical_inputs or source["simulation_case_id"] in simulations:
            msg = "Dataset selection repeats a package or physical source case."
            raise ValueError(msg)
        package_ids.add(package_id)
        physical_inputs.add(source["case_input_id"])
        simulations.add(source["simulation_case_id"])
        projected_cases.append(copy.deepcopy(dict(source_case)))
    package_keys = (
        "schema_kind",
        "schema_version",
        "dataset_view",
        "registered_task_id",
        "source_task",
        "materials",
        "channel_contract",
        "channel_contract_digest",
        "grid",
        "builder_identity",
    )
    missing = [key for key in package_keys if key not in value]
    if missing:
        message = f"Dataset package semantic identity is missing fields {missing}."
        raise ValueError(message)
    payload = {key: copy.deepcopy(value[key]) for key in package_keys}
    from . import dataset_contracts_partition as partition_contract  # noqa: PLC0415

    fixed = value.get("case_partition")
    if fixed is not None:
        fixed = partition_contract.validate_fixed_partition(
            fixed,
            case_ids=[entry["package_case_id"] for entry in projected_cases],
            material_by_case={entry["package_case_id"]: entry["source"]["material_family"] for entry in projected_cases},
        )
    payload["case_partition"] = copy.deepcopy(fixed)
    payload["source_case_identities"] = projected_cases
    return payload


def package_content_digest(value: Mapping[str, Any]) -> str:
    """Return the digest of a Dataset's scientific content, without split-generation provenance.

    Two Datasets with the same interpretation, source cases, builder and ordered
    Train/validation/test/OOD membership are the same content even when other
    split seeds or recipes produced that membership. The seeds and recipe stay in
    the manifest as provenance.
    """
    payload = package_semantic_identity_payload(value)
    partition = payload.get("case_partition")
    if isinstance(partition, dict):
        payload["case_partition"] = {key: item for key, item in partition.items() if key not in {"seed_input", "seed", "recipe"}}
    return hashlib.sha256(_canonical_json(payload, label="Dataset content identity")).hexdigest()


def package_identity_from_provenance(provenance: Mapping[str, Any]) -> tuple[str, str]:
    """Return the current package identity from semantic provenance."""
    payload = package_semantic_identity_payload(provenance)
    digest = hashlib.sha256(_canonical_json(payload, label="Dataset package identity")).hexdigest()
    name = _require_non_empty_string(provenance.get("dataset_name"), label="Dataset package dataset_name")
    return name, digest


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
    if _tensor_dtype(tensor) != TRAINING_TENSOR_DTYPE:
        msg = f"{label} must use torch.{TRAINING_TENSOR_DTYPE}, got {tensor.dtype}."
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


def _training_fingerprint_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Select payload metadata while excluding source provenance and locators."""
    return {
        key: value for key, value in payload.items() if key not in {"inputs", "outputs", "dataset_fingerprint", "dataset_id", "source_provenance"}
    }


def build_training_dataset_payload(
    *,
    task: TaskSpec,
    dataset_id: str,
    sample_ids: Sequence[str],
    source_selection_identity: Mapping[str, Any],
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
    normalized_selection_identity = _json_mapping_copy(source_selection_identity, label="source_selection_identity")
    if (
        set(normalized_selection_identity) != {"schema_kind", "schema_version", "source_cases"}
        or normalized_selection_identity["schema_kind"] != "dataset_source_selection"
        or normalized_selection_identity["schema_version"] != 1
        or not isinstance(normalized_selection_identity["source_cases"], list)
        or len(normalized_selection_identity["source_cases"]) != len(normalized_sample_ids)
    ):
        msg = "Steady Dataset source selection identity is invalid."
        raise ValueError(msg)
    normalized_sources = [_json_mapping_copy(value, label=f"source_identities[{index}]") for index, value in enumerate(source_identities)]
    normalized_metadata = [_json_mapping_copy(value, label=f"source_metadata[{index}]") for index, value in enumerate(source_metadata)]
    normalized_provenance = _json_mapping_copy(source_provenance, label="source_provenance")
    if normalized_selection_identity["source_cases"] != normalized_provenance.get("source_case_identities"):
        msg = "Steady payload selection identity disagrees with Dataset provenance."
        raise ValueError(msg)
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
        "source_selection_identity": normalized_selection_identity,
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
    fingerprint_metadata = _training_fingerprint_metadata(payload)
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
    source_selection_identity = _json_mapping_copy(mapping.get("source_selection_identity"), label="source_selection_identity")
    if (
        set(source_selection_identity) != {"schema_kind", "schema_version", "source_cases"}
        or source_selection_identity["schema_kind"] != "dataset_source_selection"
        or source_selection_identity["schema_version"] != 1
        or not isinstance(source_selection_identity["source_cases"], list)
        or len(source_selection_identity["source_cases"]) != sample_count
    ):
        msg_0 = "Steady Dataset source selection identity is invalid."
        message = msg_0
        raise ValueError(message)
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
    if source_selection_identity["source_cases"] != source_provenance.get("source_case_identities"):
        msg_0 = "Steady payload selection identity disagrees with Dataset provenance."
        message = msg_0
        raise ValueError(message)
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
        normalized_mapping["source_selection_identity"] = source_selection_identity
        normalized_mapping["source_identities"] = normalized_sources
        normalized_mapping["source_metadata"] = normalized_metadata
        normalized_mapping["source_provenance"] = source_provenance
        fingerprint_metadata = _training_fingerprint_metadata(normalized_mapping)
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
        source_selection_identity=source_selection_identity,
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
