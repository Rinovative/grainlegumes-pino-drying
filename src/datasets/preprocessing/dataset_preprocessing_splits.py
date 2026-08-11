"""
===============================================================================
dataset_preprocessing_splits.py
===============================================================================
Own identity-bound dataset combinations and admitted split membership evidence.
Responsibilities:
  - Construct identity-preserving ordered dataset views and combinations
  - Validate the persisted split schema against exact dataset identities
  - Expose immutable role evidence and reproduce caller-owned split payloads
Design principles:
  - Train and evaluation partition one ordered source identity exactly once
  - OOD selection remains identity-bound and order-sensitive
  - Persisted tensor membership is isolated at every admission boundary
This module does NOT:
  - Fit preprocessing or construct DataLoaders
  - Persist split payloads or resolve experiment configuration
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src import domain
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_views as views

if TYPE_CHECKING:
    from collections.abc import Sized


SPLIT_SCHEMA_VERSION = 1
_SPLIT_INDEX_KEYS = ("train_indices", "eval_indices", "ood_indices")
_SPLIT_REQUIRED_KEYS = frozenset({"schema_version", "task", "task_contract_digest", *_SPLIT_INDEX_KEYS, "metadata"})
_SPLIT_COUNT_METADATA_KEYS = {
    "train_indices": "n_train",
    "eval_indices": "n_eval",
    "ood_indices": "n_ood",
}


SplitRole = Literal["train", "eval", "ood"]
SPLIT_ROLES: tuple[SplitRole, ...] = ("train", "eval", "ood")


@dataclass(frozen=True, slots=True)
class SplitSettings:
    """Normalized deterministic split settings."""

    train_ratio: float
    ood_fraction: float
    split_seed: int


@dataclass(frozen=True, slots=True)
class SplitRoleEvidence:
    """Immutable identity and ordered membership evidence for one split role."""

    name: SplitRole
    source: identity.DatasetIdentity
    index_values: tuple[int, ...]
    count: int
    full_count: int
    membership_digest: str
    ratio: float
    seed: int

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
    train_ratio: float
    ood_fraction: float
    split_seed: int
    train: SplitRoleEvidence
    eval: SplitRoleEvidence
    ood: SplitRoleEvidence

    def role(self, name: SplitRole) -> SplitRoleEvidence:
        """Return immutable evidence for ``train``, ``eval``, or ``ood``."""
        if name == "train":
            return self.train
        if name == "eval":
            return self.eval
        if name == "ood":
            return self.ood
        msg = f"Unknown split role {name!r}."
        raise ValueError(msg)

    def as_payload(self) -> dict[str, Any]:
        """Reproduce the current caller-owned ``split_indices.pt`` payload."""
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "task_contract_digest": self.task_contract_digest,
            "train_indices": self.train.indices,
            "eval_indices": self.eval.indices,
            "ood_indices": self.ood.indices,
            "metadata": {
                "datasets": {
                    "train": self.train.source.as_dict(),
                    "ood": self.ood.source.as_dict(),
                },
                "n_train_full": self.train.full_count,
                "n_train": self.train.count,
                "n_eval": self.eval.count,
                "n_ood_full": self.ood.full_count,
                "n_ood": self.ood.count,
                "train_ratio": self.train_ratio,
                "ood_fraction": self.ood_fraction,
                "split_seed": self.split_seed,
                "membership_digests": {
                    "train": self.train.membership_digest,
                    "eval": self.eval.membership_digest,
                    "ood": self.ood.membership_digest,
                },
            },
        }


def admit_split_settings(
    *,
    train_ratio: Any,
    ood_fraction: Any,
    split_seed: Any,
) -> SplitSettings:
    """Normalize split ratios and membership seed through one contract owner."""
    return SplitSettings(
        train_ratio=_normalized_fraction(train_ratio, label="train_ratio", allow_one=False),
        ood_fraction=_normalized_fraction(ood_fraction, label="ood_fraction", allow_one=True),
        split_seed=_normalized_seed(split_seed, label="split_seed"),
    )


def select_identity_dataset(
    source: Dataset[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    label: str,
) -> Dataset[Mapping[str, Any]]:
    """Return an identity-bound ordered dataset view without tensor copies."""
    return _IdentityDatasetView(source, list(indices), label=label)


class _IdentityDatasetView(Dataset[Mapping[str, Any]]):
    """Expose one identity-bound ordered view without copying tensor payloads."""

    def __init__(self, source: Dataset[Mapping[str, Any]], indices: list[int], *, label: str) -> None:
        """Bind an ordered subset to a deterministic derived identity."""
        source_identity = getattr(source, "identity", None)
        if not isinstance(source_identity, identity.DatasetIdentity):
            msg = f"{label} source must expose a verified DatasetIdentity."
            raise TypeError(msg)
        if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(cast("Sized", source)):
            msg = f"{label} indices must be a non-empty unique in-range selection."
            raise ValueError(msg)
        self.source = source
        self.indices = tuple(indices)
        sample_ids = tuple(source_identity.sample_ids[index] for index in indices)
        fingerprint_payload = json.dumps(
            {
                "source_fingerprint": source_identity.fingerprint,
                "label": label,
                "sample_ids": sample_ids,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        source_metadata = source_identity.source_metadata
        self.identity = identity.DatasetIdentity(
            dataset_id=source_identity.dataset_id,
            task=source_identity.task,
            data_contract_digest=source_identity.data_contract_digest,
            fingerprint=hashlib.sha256(fingerprint_payload).hexdigest(),
            sample_ids=sample_ids,
            sample_count=len(sample_ids),
            spatial_shape=source_identity.spatial_shape,
            source_metadata=(None if source_metadata is None else tuple(source_metadata[index] for index in indices)),
            source_provenance=source_identity.source_provenance,
        )

    def __len__(self) -> int:
        """Return selected sample count."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        """Return one source sample in view order."""
        return self.source[self.indices[index]]


class _CombinedIdentityDataset(Dataset[Mapping[str, Any]]):
    """Concatenate independent OOD packages under one derived loader identity."""

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


def package_id_membership(dataset: Dataset[Mapping[str, Any]]) -> dict[views.IdMembership, list[int]] | None:
    """Return exact package-owned ID membership when present."""
    dataset_identity = getattr(dataset, "identity", None)
    if not isinstance(dataset_identity, identity.DatasetIdentity) or dataset_identity.source_metadata is None:
        return None
    values = [metadata.get("dataset_membership") for metadata in dataset_identity.source_metadata]
    if all(value is None for value in values):
        return None
    if any(value not in views.ID_MEMBERSHIPS for value in values):
        msg = "Package-owned ID membership must label every sample train, validation, or id_test."
        raise ValueError(msg)
    membership: dict[views.IdMembership, list[int]] = {
        cast("views.IdMembership", role): [index for index, value in enumerate(values) if value == role] for role in views.ID_MEMBERSHIPS
    }
    if any(not indices for indices in membership.values()):
        msg = "Package-owned train, validation, and id_test memberships must all be non-empty."
        raise ValueError(msg)
    return membership


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


def _normalized_fraction(value: Any, *, label: str, allow_one: bool) -> float:
    """
    Normalize a finite non-boolean split fraction to the supported interval.

    The lower bound is always open. ``allow_one`` selects ``(0, 1]`` for OOD
    selection or ``(0, 1)`` when a non-empty complementary split is required.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{label} must be numeric."
        raise TypeError(msg)
    fraction = float(value)
    upper_bound_valid = fraction <= 1.0 if allow_one else fraction < 1.0
    if not math.isfinite(fraction) or fraction <= 0.0 or not upper_bound_valid:
        interval = "(0, 1]" if allow_one else "(0, 1)"
        msg = f"{label} must be in {interval}, got {value!r}."
        raise ValueError(msg)
    return fraction


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


def _validate_train_eval_partition(train_indices: Tensor, eval_indices: Tensor, *, n_train_full: int) -> None:
    """
    Require an exact disjoint partition of every ordered training source index.

    Validation is independent of index order: concatenated membership must sort
    to ``range(n_train_full)`` exactly once, with no train/eval overlap.
    """
    overlap = train_indices[torch.isin(train_indices, eval_indices)]
    if overlap.numel():
        msg = f"Saved train/eval indices must be disjoint. Overlapping indices include {overlap[:10].tolist()}."
        raise ValueError(msg)
    combined = torch.cat((train_indices, eval_indices))
    expected = torch.arange(n_train_full, dtype=torch.long)
    if combined.numel() != n_train_full or not torch.equal(torch.sort(combined).values, expected):
        missing = expected[~torch.isin(expected, combined)]
        msg = f"Saved train/eval indices must cover every source index exactly once. Missing indices include {missing[:10].tolist()}."
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
    train_identity: identity.DatasetIdentity,
    ood_identity: identity.DatasetIdentity,
) -> None:
    """Bind split-level task identity and nested learned-data identities."""
    try:
        registered_task = domain.tasks.registry.get_task(task_id)
    except (KeyError, ValueError):
        registered_task = None

    if registered_task is not None and task_contract_digest != registered_task.contract_digest:
        msg = f"split_indices.pt task_contract_digest does not match the current registered task {task_id!r}."
        raise ValueError(msg)

    saved_identities = (("train", train_identity), ("ood", ood_identity))
    for label, saved_identity in saved_identities:
        if saved_identity.task != task_id:
            msg = f"split_indices.pt {label} identity does not match its task header."
            raise ValueError(msg)
        if registered_task is not None:
            identity.validate_dataset_data_contract_digest(
                saved_identity.data_contract_digest,
                task=registered_task,
                label=f"split_indices.pt {label} dataset data_contract_digest",
            )

    if registered_task is None and train_identity.data_contract_digest != ood_identity.data_contract_digest:
        msg = "split_indices.pt train and OOD identities do not share one learned-data contract."
        raise ValueError(msg)


def admit_split_contract(
    split_info: Mapping[str, Any],
    *,
    train_identity: identity.DatasetIdentity | None = None,
    ood_identity: identity.DatasetIdentity | None = None,
    expected_train_ratio: float | None = None,
    expected_ood_fraction: float | None = None,
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
    ood_identity : DatasetIdentity | None, optional
        Loaded OOD dataset identity to bind.
    expected_train_ratio : float | None, optional
        Effective training fraction.
    expected_ood_fraction : float | None, optional
        Effective OOD fraction.
    expected_split_seed : int | None, optional
        Effective split seed.

    Returns
    -------
    SplitContract
        Immutable task, source identity, settings, membership, and isolated
        role evidence.

    Raises
    ------
    TypeError
        If the payload, nested metadata, identities, settings, or membership
        tensors have incompatible runtime types.
    ValueError
        If schema keys/version, task binding, counts, fractions, seeds, index
        partition, dataset identity, or ordered membership digests disagree.

    Notes
    -----
    Train and eval must partition the full training dataset exactly once. OOD is
    a non-empty subset of its source dataset. Optional expected settings and
    identities bind an existing split to the effective run configuration without
    mutating the supplied mapping. A current split header identifies the complete
    TaskSpec, while nested dataset identities retain their independently validated
    learned-data provenance.

    """
    if not isinstance(split_info, Mapping):
        msg = "split_indices.pt must contain a mapping."
        raise TypeError(msg)
    missing = sorted(_SPLIT_REQUIRED_KEYS.difference(split_info))
    unexpected = sorted(set(split_info).difference(_SPLIT_REQUIRED_KEYS))
    if missing or unexpected:
        msg = f"split_indices.pt schema keys do not match. Missing: {missing}. Unexpected: {unexpected}."
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
    required_metadata_keys = {
        "datasets",
        "n_train_full",
        "n_train",
        "n_eval",
        "n_ood_full",
        "n_ood",
        "train_ratio",
        "ood_fraction",
        "split_seed",
        "membership_digests",
    }
    missing_metadata = sorted(required_metadata_keys.difference(metadata))
    unexpected_metadata = sorted(set(metadata).difference(required_metadata_keys))
    if missing_metadata or unexpected_metadata:
        msg = f"split_indices.pt metadata keys do not match. Missing: {missing_metadata}. Unexpected: {unexpected_metadata}."
        raise ValueError(msg)
    datasets_meta = metadata.get("datasets")
    if not isinstance(datasets_meta, Mapping) or set(datasets_meta) != {"train", "ood"}:
        msg = "split_indices.pt metadata.datasets must contain exactly train and ood identities."
        raise ValueError(msg)
    saved_train_identity = _identity_from_mapping(datasets_meta["train"], label="metadata.datasets.train")
    saved_ood_identity = _identity_from_mapping(datasets_meta["ood"], label="metadata.datasets.ood")
    _validate_split_task_identity(
        task_id=task,
        task_contract_digest=digest,
        train_identity=saved_train_identity,
        ood_identity=saved_ood_identity,
    )
    _validate_expected_identity(saved_train_identity, train_identity, label="train")
    _validate_expected_identity(saved_ood_identity, ood_identity, label="OOD")

    saved_train_ratio = _normalized_fraction(metadata.get("train_ratio"), label="metadata.train_ratio", allow_one=False)
    saved_ood_fraction = _normalized_fraction(metadata.get("ood_fraction"), label="metadata.ood_fraction", allow_one=True)
    saved_split_seed = _normalized_seed(metadata.get("split_seed"), label="metadata.split_seed")
    if expected_train_ratio is not None and saved_train_ratio != _normalized_fraction(
        expected_train_ratio,
        label="Expected train_ratio",
        allow_one=False,
    ):
        msg = "split_indices.pt train_ratio does not match the effective config."
        raise ValueError(msg)
    if expected_ood_fraction is not None and saved_ood_fraction != _normalized_fraction(
        expected_ood_fraction,
        label="Expected ood_fraction",
        allow_one=True,
    ):
        msg = "split_indices.pt ood_fraction does not match the effective config."
        raise ValueError(msg)
    if expected_split_seed is not None and saved_split_seed != _normalized_seed(expected_split_seed, label="Expected split_seed"):
        msg = "split_indices.pt split_seed does not match the effective config."
        raise ValueError(msg)

    n_train_full = _required_metadata_count(metadata, "n_train_full")
    n_ood_full = _required_metadata_count(metadata, "n_ood_full")
    if n_train_full != saved_train_identity.sample_count or n_ood_full != saved_ood_identity.sample_count:
        msg = "split_indices.pt full counts do not match saved ordered dataset identities."
        raise ValueError(msg)
    validated = {key: _validated_index_tensor(split_info, key) for key in _SPLIT_INDEX_KEYS}
    saved_counts = {
        "n_train": _required_metadata_count(metadata, "n_train"),
        "n_eval": _required_metadata_count(metadata, "n_eval"),
        "n_ood": _required_metadata_count(metadata, "n_ood"),
    }
    for index_key, count_key in _SPLIT_COUNT_METADATA_KEYS.items():
        if saved_counts[count_key] != int(validated[index_key].numel()):
            msg = f"split_indices.pt metadata {count_key!r} does not match {index_key!r}."
            raise ValueError(msg)
    expected_counts = {
        "n_train": int(saved_train_ratio * n_train_full),
        "n_eval": n_train_full - int(saved_train_ratio * n_train_full),
        "n_ood": int(saved_ood_fraction * n_ood_full),
    }
    if saved_counts != expected_counts:
        msg = f"split_indices.pt counts are inconsistent with saved settings: {saved_counts} != {expected_counts}."
        raise ValueError(msg)

    _validate_index_bounds(validated["train_indices"], key="train_indices", full_count=n_train_full)
    _validate_index_bounds(validated["eval_indices"], key="eval_indices", full_count=n_train_full)
    _validate_index_bounds(validated["ood_indices"], key="ood_indices", full_count=n_ood_full)
    _validate_train_eval_partition(validated["train_indices"], validated["eval_indices"], n_train_full=n_train_full)

    membership = metadata.get("membership_digests")
    if not isinstance(membership, Mapping) or set(membership) != {"train", "eval", "ood"}:
        msg = "split_indices.pt metadata.membership_digests must contain exactly train, eval, and ood."
        raise ValueError(msg)
    sources = {"train": saved_train_identity, "eval": saved_train_identity, "ood": saved_ood_identity}
    index_keys = {"train": "train_indices", "eval": "eval_indices", "ood": "ood_indices"}
    for role, source in sources.items():
        expected_membership = identity.membership_digest(
            role=role,
            dataset_fingerprint=source.fingerprint,
            sample_ids=source.sample_ids,
            indices=[int(value) for value in validated[index_keys[role]].tolist()],
        )
        if membership.get(role) != expected_membership:
            msg = f"split_indices.pt {role} ordered membership digest mismatch."
            raise ValueError(msg)
    return SplitContract(
        schema_version=schema_version,
        task=task,
        task_contract_digest=digest,
        train_ratio=saved_train_ratio,
        ood_fraction=saved_ood_fraction,
        split_seed=saved_split_seed,
        train=SplitRoleEvidence(
            name="train",
            source=saved_train_identity,
            index_values=tuple(int(value) for value in validated["train_indices"].tolist()),
            count=saved_counts["n_train"],
            full_count=n_train_full,
            membership_digest=cast("str", membership["train"]),
            ratio=saved_train_ratio,
            seed=saved_split_seed,
        ),
        eval=SplitRoleEvidence(
            name="eval",
            source=saved_train_identity,
            index_values=tuple(int(value) for value in validated["eval_indices"].tolist()),
            count=saved_counts["n_eval"],
            full_count=n_train_full,
            membership_digest=cast("str", membership["eval"]),
            ratio=1.0 - saved_train_ratio,
            seed=saved_split_seed,
        ),
        ood=SplitRoleEvidence(
            name="ood",
            source=saved_ood_identity,
            index_values=tuple(int(value) for value in validated["ood_indices"].tolist()),
            count=saved_counts["n_ood"],
            full_count=n_ood_full,
            membership_digest=cast("str", membership["ood"]),
            ratio=saved_ood_fraction,
            seed=saved_split_seed,
        ),
    )
