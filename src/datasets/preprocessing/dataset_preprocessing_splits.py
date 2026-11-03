"""Own identity-bound dataset combinations and admitted split membership evidence.

Responsibilities:
  - Construct identity-preserving ordered dataset views and combinations
  - Validate the persisted split schema against exact dataset identities
  - Expose immutable role evidence and reproduce caller-owned split payloads

Design principles:
  - Train, validation and ID-test follow the Dataset-owned fixed case partition
  - Additional evaluation membership remains identity-bound and order-sensitive
  - Persisted tensor membership is isolated at every admission boundary

This module does NOT:
  - Fit preprocessing or construct DataLoaders
  - Persist split payloads or resolve experiment configuration
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src import domain
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_partition as partition_contract

if TYPE_CHECKING:
    from collections.abc import Sized


SPLIT_SCHEMA_VERSION = 2
_SPLIT_INDEX_KEYS = ("train_indices", "validation_indices", "id_test_indices", "additional_evaluation_indices")
_SPLIT_REQUIRED_KEYS = frozenset({"schema_version", "task", "task_contract_digest", *_SPLIT_INDEX_KEYS, "metadata"})
_UNCHECKED: Any = object()


SplitRole = Literal["train", "validation", "id_test", "additional_evaluation"]
SPLIT_ROLES: tuple[SplitRole, ...] = ("train", "validation", "id_test", "additional_evaluation")
ID_SPLIT_ROLES: tuple[SplitRole, ...] = ("train", "validation", "id_test")


@dataclass(frozen=True, slots=True)
class SplitRoleEvidence:
    """Immutable identity and ordered membership evidence for one split role."""

    name: SplitRole
    source: identity.DatasetIdentity
    index_values: tuple[int, ...]
    count: int
    full_count: int
    membership_digest: str

    @property
    def indices(self) -> Tensor:
        """Return an isolated CPU ``long`` tensor in persisted order."""
        return torch.tensor(self.index_values, dtype=torch.long)


@dataclass(frozen=True, slots=True)
class SplitContract:
    """Immutable admitted split schema with role-oriented evidence access."""

    schema_version: int
    task: str
    task_contract_digest: str
    split_seed: int
    case_partition: Mapping[str, Any]
    case_partition_digest: str
    train: SplitRoleEvidence
    validation: SplitRoleEvidence
    id_test: SplitRoleEvidence
    additional_evaluation: SplitRoleEvidence | None

    @property
    def roles(self) -> tuple[SplitRole, ...]:
        """Return the roles with admitted membership in canonical order."""
        return ID_SPLIT_ROLES if self.additional_evaluation is None else SPLIT_ROLES

    def role(self, name: SplitRole) -> SplitRoleEvidence:
        """Return immutable evidence for one admitted role."""
        if name in ID_SPLIT_ROLES:
            return cast("SplitRoleEvidence", getattr(self, name))
        if name == "additional_evaluation":
            if self.additional_evaluation is None:
                msg = "This split has no additional evaluation membership."
                raise ValueError(msg)
            return self.additional_evaluation
        msg = f"Unknown split role {name!r}."
        raise ValueError(msg)

    def sources(self) -> dict[str, identity.DatasetIdentity]:
        """Return the dataset identity behind each identity-bearing role."""
        result = {"train": self.train.source}
        if self.additional_evaluation is not None:
            result["additional_evaluation"] = self.additional_evaluation.source
        return result

    def scientific_payload(self) -> dict[str, Any]:
        """Return the replay-scientific split identity bound into checkpoints."""
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "task_contract_digest": self.task_contract_digest,
            "split_seed": self.split_seed,
            "case_partition_digest": self.case_partition_digest,
            "datasets": {role: source.as_dict() for role, source in self.sources().items()},
            "membership_digests": {role: self.role(role).membership_digest for role in self.roles},
        }

    def as_payload(self) -> dict[str, Any]:
        """Reproduce the caller-owned ``split_indices.pt`` payload."""
        additional = self.additional_evaluation
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "task_contract_digest": self.task_contract_digest,
            "train_indices": self.train.indices,
            "validation_indices": self.validation.indices,
            "id_test_indices": self.id_test.indices,
            "additional_evaluation_indices": None if additional is None else additional.indices,
            "metadata": {
                "datasets": {
                    "train": self.train.source.as_dict(),
                    "additional_evaluation": None if additional is None else additional.source.as_dict(),
                },
                "split_seed": self.split_seed,
                "case_partition": dict(self.case_partition),
                "case_partition_digest": self.case_partition_digest,
                "counts": {
                    "n_train_full": self.train.full_count,
                    "n_train": self.train.count,
                    "n_validation": self.validation.count,
                    "n_id_test": self.id_test.count,
                    "n_additional_evaluation_full": None if additional is None else additional.full_count,
                    "n_additional_evaluation": None if additional is None else additional.count,
                },
                "membership_digests": {role: self.role(role).membership_digest for role in self.roles},
            },
        }


def role_positions(source: identity.DatasetIdentity, case_ids: Sequence[str], *, label: str) -> tuple[int, ...]:
    """Map ordered Dataset case IDs to positions in one verified identity."""
    position_by_id = {sample_id: index for index, sample_id in enumerate(source.sample_ids)}
    try:
        return tuple(position_by_id[case_id] for case_id in case_ids)
    except KeyError as error:
        msg = f"Split role {label!r} references case {error.args[0]!r}, which {source.dataset_id!r} does not contain."
        raise ValueError(msg) from error


def role_evidence(name: SplitRole, source: identity.DatasetIdentity, positions: Sequence[int]) -> SplitRoleEvidence:
    """Bind one ordered role membership to its source identity."""
    values = tuple(int(value) for value in positions)
    return SplitRoleEvidence(
        name=name,
        source=source,
        index_values=values,
        count=len(values),
        full_count=source.sample_count,
        membership_digest=identity.membership_digest(
            role=name,
            dataset_fingerprint=source.fingerprint,
            sample_ids=source.sample_ids,
            indices=list(values),
        ),
    )


class _CombinedIdentityDataset(Dataset[Mapping[str, Any]]):
    """Concatenate independent evaluation packages under one derived loader identity."""

    def __init__(self, sources: list[Dataset[Mapping[str, Any]]]) -> None:
        """Validate compatible component identities and bind ordered membership."""
        if not sources:
            msg = "Combined OOD dataset requires at least one component."
            raise ValueError(msg)
        identities = [getattr(source, "identity", None) for source in sources]
        if not all(isinstance(item, identity.DatasetIdentity) for item in identities):
            msg = "Every combined OOD component must expose a verified DatasetIdentity."
            raise TypeError(msg)
        typed = cast("list[identity.DatasetIdentity]", identities)
        first = typed[0]
        if any(
            item.task != first.task or item.data_contract_digest != first.data_contract_digest or item.spatial_shape != first.spatial_shape
            for item in typed[1:]
        ):
            msg = "Combined OOD packages must share task, learned-data contract, and spatial shape."
            raise ValueError(msg)
        input_fields = [getattr(source, "input_fields", None) for source in sources]
        output_fields = [getattr(source, "output_fields", None) for source in sources]
        if any(value != input_fields[0] for value in input_fields[1:]) or any(value != output_fields[0] for value in output_fields[1:]):
            msg = "Combined OOD packages must share input and output field order."
            raise ValueError(msg)
        self.input_fields = input_fields[0]
        self.output_fields = output_fields[0]
        self.sources = tuple(sources)
        self.offsets: list[tuple[int, int]] = []
        sample_ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        total = 0
        for source, item in zip(sources, typed, strict=True):
            start = total
            total += len(cast("Sized", source))
            self.offsets.append((start, total))
            sample_ids.extend(f"{item.dataset_id}::{sample_id}" for sample_id in item.sample_ids)
            if item.source_metadata is not None:
                metadata.extend(item.source_metadata)
        payload = json.dumps(
            {
                "components": [
                    {
                        "dataset_id": item.dataset_id,
                        "fingerprint": item.fingerprint,
                        "sample_ids": item.sample_ids,
                    }
                    for item in typed
                ]
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.identity = identity.DatasetIdentity(
            dataset_id=identity.combined_dataset_id(tuple(item.dataset_id for item in typed)),
            task=first.task,
            data_contract_digest=first.data_contract_digest,
            fingerprint=hashlib.sha256(payload).hexdigest(),
            sample_ids=tuple(sample_ids),
            sample_count=len(sample_ids),
            spatial_shape=first.spatial_shape,
            source_metadata=tuple(metadata) if len(metadata) == len(sample_ids) else None,
            source_provenance={"component_dataset_ids": [item.dataset_id for item in typed]},
        )

    def __len__(self) -> int:
        """Return combined OOD sample count."""
        return self.identity.sample_count

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Return one component sample in configured package order."""
        if index < 0:
            index += len(self)
        for source, (start, stop) in zip(self.sources, self.offsets, strict=True):
            if start <= index < stop:
                return source[index - start]
        raise IndexError(index)


def combine_identity_datasets(sources: Sequence[Dataset[Mapping[str, Any]]]) -> Dataset[Mapping[str, Any]]:
    """Return one verified dataset or an identity-bound ordered combination."""
    normalized = list(sources)
    if not normalized:
        msg = "At least one dataset package is required."
        raise ValueError(msg)
    if not all(isinstance(getattr(source, "identity", None), identity.DatasetIdentity) for source in normalized):
        msg = "Every dataset package must expose a verified DatasetIdentity."
        raise TypeError(msg)
    return normalized[0] if len(normalized) == 1 else _CombinedIdentityDataset(normalized)


def _required_metadata_count(metadata: Mapping[str, Any], key: str) -> int:
    """
    Parse one required positive count from persisted split metadata.

    Booleans are rejected even though they are integer subclasses, preserving an
    unambiguous schema for sample counts and spatial dimensions.
    """
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"split_indices.pt metadata {key!r} must be an integer."
        raise TypeError(msg)
    if value <= 0:
        msg = f"split_indices.pt metadata {key!r} must be positive, got {value}."
        raise ValueError(msg)
    return value


def _normalized_seed(value: Any, *, label: str) -> int:
    """Return an integer split seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{label} must be an integer."
        raise TypeError(msg)
    return value


def _validated_index_tensor(split_info: Mapping[str, Any], key: str) -> Tensor:
    """
    Isolate one persisted membership tensor in canonical CPU ``long`` form.

    Admission requires a non-empty, one-dimensional, unique integer tensor.
    Booleans, floating/complex values, and duplicate membership are rejected.
    The returned clone cannot mutate the loaded persistence payload.
    """
    value = split_info.get(key)
    if not isinstance(value, Tensor):
        msg = f"split_indices.pt key {key!r} must be a torch.Tensor."
        raise TypeError(msg)
    if value.ndim != 1:
        msg = f"split_indices.pt key {key!r} must be one-dimensional, got shape {tuple(value.shape)}."
        raise ValueError(msg)
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        msg = f"split_indices.pt key {key!r} must contain integer indices, got dtype {value.dtype}."
        raise TypeError(msg)
    if value.numel() == 0:
        msg = f"split_indices.pt key {key!r} must not be empty."
        raise ValueError(msg)
    if torch.unique(value).numel() != value.numel():
        msg = f"split_indices.pt key {key!r} must not contain duplicate indices."
        raise ValueError(msg)
    return value.to(dtype=torch.long, device="cpu").clone()


def _validate_index_bounds(indices: Tensor, *, key: str, full_count: int) -> None:
    """Reject negative or out-of-range saved indices."""
    min_index = int(indices.min().item())
    max_index = int(indices.max().item())
    if min_index < 0 or max_index >= full_count:
        msg = f"split_indices.pt key {key!r} is out of bounds for full count {full_count}. The observed index range is {min_index}..{max_index}."
        raise ValueError(msg)


def _identity_from_mapping(value: Any, *, label: str) -> identity.DatasetIdentity:
    """
    Reconstruct one exact persisted dataset identity without additional keys.

    The mapping must contain only logical identity, task and learned-data digest,
    fingerprint, ordered unique sample IDs, count, and positive spatial shape. String and
    count validation fails before the immutable ``DatasetIdentity`` is returned.
    """
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    required = {"dataset_id", "task", "data_contract_digest", "fingerprint", "sample_ids", "sample_count", "spatial_shape"}
    missing = sorted(required.difference(value))
    unexpected = sorted(set(value).difference(required))
    if missing or unexpected:
        msg = f"{label} keys do not match. Missing: {missing}. Unexpected: {unexpected}."
        raise ValueError(msg)
    sample_ids_raw = value["sample_ids"]
    if not isinstance(sample_ids_raw, list) or not all(isinstance(item, str) and item for item in sample_ids_raw):
        msg = f"{label}.sample_ids must be a list of non-empty strings."
        raise TypeError(msg)
    sample_ids = tuple(sample_ids_raw)
    if len(sample_ids) != len(set(sample_ids)):
        msg = f"{label}.sample_ids must be unique."
        raise ValueError(msg)
    sample_count = _required_metadata_count(value, "sample_count")
    if len(sample_ids) != sample_count:
        msg = f"{label}.sample_count={sample_count} does not match {len(sample_ids)} sample_ids."
        raise ValueError(msg)
    spatial_shape_raw = value["spatial_shape"]
    if not isinstance(spatial_shape_raw, list) or not spatial_shape_raw:
        msg = f"{label}.spatial_shape must be a non-empty list."
        raise TypeError(msg)
    spatial_shape = tuple(_required_metadata_count({str(index): item}, str(index)) for index, item in enumerate(spatial_shape_raw))
    strings: dict[str, str] = {}
    for key in ("dataset_id", "task", "data_contract_digest", "fingerprint"):
        item = value[key]
        if not isinstance(item, str) or not item:
            msg = f"{label}.{key} must be a non-empty string."
            raise TypeError(msg)
        strings[key] = item
    return identity.DatasetIdentity(
        dataset_id=strings["dataset_id"],
        task=strings["task"],
        data_contract_digest=strings["data_contract_digest"],
        fingerprint=strings["fingerprint"],
        sample_ids=sample_ids,
        sample_count=sample_count,
        spatial_shape=spatial_shape,
    )


def _validate_expected_identity(
    saved: identity.DatasetIdentity,
    expected: identity.DatasetIdentity | None,
    *,
    label: str,
) -> None:
    """
    Bind a saved split to the complete currently loaded dataset identity.

    When an expected identity is supplied, dataclass equality compares the
    logical ID, persisted data-provenance digest, content fingerprint, ordered
    sample IDs, count, and spatial shape. No partial or path-based match is
    accepted.
    """
    if expected is not None and saved != expected:
        msg = f"split_indices.pt {label} dataset identity does not match the loaded dataset. Saved: {saved.as_dict()}. Loaded: {expected.as_dict()}."
        raise ValueError(msg)


def _validate_split_task_identity(
    *,
    task_id: str,
    task_contract_digest: str,
    identities: Sequence[tuple[str, identity.DatasetIdentity]],
) -> None:
    """Bind split-level task identity and nested learned-data identities."""
    try:
        registered_task = domain.tasks.registry.get_task(task_id)
    except (KeyError, ValueError):
        registered_task = None

    if registered_task is not None and task_contract_digest != registered_task.contract_digest:
        msg = f"split_indices.pt task_contract_digest does not match the current registered task {task_id!r}."
        raise ValueError(msg)
    for label, saved_identity in identities:
        if saved_identity.task != task_id:
            msg = f"split_indices.pt {label} identity does not match its task header."
            raise ValueError(msg)
        if registered_task is not None:
            identity.validate_dataset_data_contract_digest(
                saved_identity.data_contract_digest,
                task=registered_task,
                label=f"split_indices.pt {label} dataset data_contract_digest",
            )
    if registered_task is None and len({item.data_contract_digest for _, item in identities}) > 1:
        msg = "split_indices.pt identities do not share one learned-data contract."
        raise ValueError(msg)


def _admit_case_partition(metadata: Mapping[str, Any], *, split_seed: int) -> Mapping[str, Any]:
    """Admit the recorded Dataset fixed partition and its digest."""
    case_partition = metadata["case_partition"]
    if not isinstance(case_partition, Mapping) or not {"seed", *ID_SPLIT_ROLES}.issubset(case_partition):
        msg = "split_indices.pt metadata.case_partition is not a Dataset fixed partition."
        raise ValueError(msg)
    if metadata["case_partition_digest"] != partition_contract.partition_digest(case_partition):
        msg = "split_indices.pt case partition digest mismatch."
        raise ValueError(msg)
    if case_partition["seed"] != split_seed:
        msg = "split_indices.pt split_seed disagrees with its Dataset case partition."
        raise ValueError(msg)
    return case_partition


def _admit_additional_evaluation(
    split_info: Mapping[str, Any],
    source: identity.DatasetIdentity | None,
) -> SplitRoleEvidence | None:
    """Admit optional additional evaluation membership against its saved identity."""
    if source is None:
        if split_info["additional_evaluation_indices"] is not None:
            msg = "split_indices.pt has additional evaluation indices without an identity."
            raise ValueError(msg)
        return None
    indices = _validated_index_tensor(split_info, "additional_evaluation_indices")
    _validate_index_bounds(indices, key="additional_evaluation_indices", full_count=source.sample_count)
    return role_evidence("additional_evaluation", source, [int(value) for value in indices.tolist()])


def admit_split_contract(
    split_info: Mapping[str, Any],
    *,
    train_identity: identity.DatasetIdentity | None = None,
    additional_evaluation_identity: identity.DatasetIdentity | None = _UNCHECKED,
    expected_split_seed: int | None = None,
) -> SplitContract:
    """
    Admit immutable split evidence against exact ordered dataset identity.

    Parameters
    ----------
    split_info : Mapping[str, Any]
        Current ``split_indices.pt`` payload.
    train_identity : DatasetIdentity | None, optional
        Loaded training dataset identity to bind.
    additional_evaluation_identity : DatasetIdentity | None, optional
        Loaded combined additional-evaluation identity to bind; ``None`` binds
        the absence of additional evaluation. Omission skips this check.
    expected_split_seed : int | None, optional
        Effective Dataset partition seed.

    Returns
    -------
    SplitContract
        Immutable task, source identity, partition, membership, and isolated
        role evidence.

    Raises
    ------
    TypeError
        If the payload, nested metadata, identities, or membership tensors have
        incompatible runtime types.
    ValueError
        If schema keys/version, task binding, partition, counts, seeds, dataset
        identity, or ordered membership digests disagree.

    Notes
    -----
    Train, validation and ID-test membership must equal the Dataset-owned fixed
    case partition in its recorded order. Additional evaluation is the declared
    OOD membership of the selected evaluation Datasets, or absent.

    """
    if not isinstance(split_info, Mapping):
        msg = "split_indices.pt must contain a mapping."
        raise TypeError(msg)
    missing = sorted(_SPLIT_REQUIRED_KEYS.difference(split_info))
    unexpected = sorted(set(split_info).difference(_SPLIT_REQUIRED_KEYS))
    if missing or unexpected:
        msg = f"split_indices.pt schema keys do not match the current Dataset-partition contract. Missing: {missing}. Unexpected: {unexpected}."
        raise ValueError(msg)
    schema_version = split_info.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != SPLIT_SCHEMA_VERSION:
        msg = f"split_indices.pt schema_version must be the current value {SPLIT_SCHEMA_VERSION}."
        raise ValueError(msg)

    task = split_info.get("task")
    digest = split_info.get("task_contract_digest")
    if not isinstance(task, str) or not task or not isinstance(digest, str) or not digest:
        msg = "split_indices.pt must contain non-empty task and task_contract_digest strings."
        raise TypeError(msg)
    metadata = split_info.get("metadata")
    if not isinstance(metadata, Mapping):
        msg = "split_indices.pt must contain a metadata mapping."
        raise TypeError(msg)
    required_metadata_keys = {"datasets", "split_seed", "case_partition", "case_partition_digest", "counts", "membership_digests"}
    if set(metadata) != required_metadata_keys:
        msg = f"split_indices.pt metadata keys must be exactly {sorted(required_metadata_keys)}."
        raise ValueError(msg)
    datasets_meta = metadata["datasets"]
    if not isinstance(datasets_meta, Mapping) or set(datasets_meta) != {"train", "additional_evaluation"}:
        msg = "split_indices.pt metadata.datasets must contain exactly train and additional_evaluation."
        raise ValueError(msg)
    saved_train_identity = _identity_from_mapping(datasets_meta["train"], label="metadata.datasets.train")
    raw_additional = datasets_meta["additional_evaluation"]
    saved_additional_identity = (
        None if raw_additional is None else _identity_from_mapping(raw_additional, label="metadata.datasets.additional_evaluation")
    )
    _validate_split_task_identity(
        task_id=task,
        task_contract_digest=digest,
        identities=[("train", saved_train_identity)]
        + ([] if saved_additional_identity is None else [("additional_evaluation", saved_additional_identity)]),
    )
    _validate_expected_identity(saved_train_identity, train_identity, label="train")
    if additional_evaluation_identity is not _UNCHECKED and saved_additional_identity != additional_evaluation_identity:
        msg = "split_indices.pt additional evaluation identity does not match the loaded evaluation Datasets."
        raise ValueError(msg)

    saved_split_seed = _normalized_seed(metadata["split_seed"], label="metadata.split_seed")
    if expected_split_seed is not None and saved_split_seed != _normalized_seed(expected_split_seed, label="Expected split_seed"):
        msg = "split_indices.pt split_seed does not match the Dataset partition."
        raise ValueError(msg)
    case_partition = _admit_case_partition(metadata, split_seed=saved_split_seed)

    evidence: dict[str, SplitRoleEvidence] = {}
    for role in ID_SPLIT_ROLES:
        indices = _validated_index_tensor(split_info, f"{role}_indices")
        _validate_index_bounds(indices, key=f"{role}_indices", full_count=saved_train_identity.sample_count)
        expected_positions = role_positions(saved_train_identity, case_partition[role], label=role)
        if tuple(int(value) for value in indices.tolist()) != expected_positions:
            msg = f"split_indices.pt {role} membership disagrees with the Dataset case partition."
            raise ValueError(msg)
        evidence[role] = role_evidence(role, saved_train_identity, expected_positions)
    id_sets = [set(evidence[role].index_values) for role in ID_SPLIT_ROLES]
    if any(left & right for index, left in enumerate(id_sets) for right in id_sets[index + 1 :]):
        msg = "split_indices.pt Train/validation/ID-test memberships overlap."
        raise ValueError(msg)

    additional_evidence = _admit_additional_evaluation(split_info, saved_additional_identity)

    counts = metadata["counts"]
    expected_counts = {
        "n_train_full": saved_train_identity.sample_count,
        "n_train": evidence["train"].count,
        "n_validation": evidence["validation"].count,
        "n_id_test": evidence["id_test"].count,
        "n_additional_evaluation_full": None if additional_evidence is None else additional_evidence.full_count,
        "n_additional_evaluation": None if additional_evidence is None else additional_evidence.count,
    }
    if counts != expected_counts:
        msg = f"split_indices.pt counts are inconsistent with its membership: {counts} != {expected_counts}."
        raise ValueError(msg)
    expected_membership = {role: item.membership_digest for role, item in evidence.items()}
    if additional_evidence is not None:
        expected_membership["additional_evaluation"] = additional_evidence.membership_digest
    if metadata["membership_digests"] != expected_membership:
        msg = "split_indices.pt ordered membership digests mismatch."
        raise ValueError(msg)
    return SplitContract(
        schema_version=schema_version,
        task=task,
        task_contract_digest=digest,
        split_seed=saved_split_seed,
        case_partition=dict(case_partition),
        case_partition_digest=metadata["case_partition_digest"],
        train=evidence["train"],
        validation=evidence["validation"],
        id_test=evidence["id_test"],
        additional_evaluation=additional_evidence,
    )
