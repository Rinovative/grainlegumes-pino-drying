"""
===============================================================================
dataset_base.py
===============================================================================
Provide dataset splitting, validation, and dataloader construction helpers.

Responsibilities:
  - Build deterministic train/eval/OOD splits bound to dataset identity
  - Construct data processors and dataloaders
  - Return split indices for persistence by callers

Design principles:
  - Split creation uses explicit membership, loader, and worker seeds
  - Reused splits validate task, ordered membership, and dataset fingerprints
  - Per-channel normalizers are fit from training membership only
  - Returned split metadata is complete but remains caller-owned for persistence

This module does NOT:
  - Save split indices, normalizer state, checkpoints, or run metadata
  - Construct simulation samples or define dataset fingerprint algorithms
  - Resolve dataset paths, devices, tasks, or experiment configuration defaults
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src import domain

from . import dataset_factory as runtime_factory
from . import dataset_identity as identity

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Sized

    from src.domain.tasks.domain_task_spec import TaskSpec


_SPLIT_INDEX_KEYS = ("train_indices", "eval_indices", "ood_indices")
_SPLIT_REQUIRED_KEYS = frozenset({"schema_version", "task", "task_contract_digest", *_SPLIT_INDEX_KEYS, "metadata"})
_SPLIT_COUNT_METADATA_KEYS = {
    "train_indices": "n_train",
    "eval_indices": "n_eval",
    "ood_indices": "n_ood",
}
_REQUIRED_NORMALIZER_STATE_KEYS = (
    "in_normalizer.mean",
    "in_normalizer.std",
    "out_normalizer.mean",
    "out_normalizer.std",
)
_NORMALIZER_STATE_RANK = 4
_NORMALIZER_DENOMINATOR_FLOOR = 1e-7
_SHA256_HEX_LENGTH = 64
_NORMALIZER_ARTIFACT_SCHEMA_KIND = "dataset_bound_normalizer"
_NORMALIZER_ARTIFACT_SCHEMA_VERSION = 1
_NORMALIZER_ARTIFACT_KEYS = frozenset(
    {
        "schema_kind",
        "schema_version",
        "task",
        "data_contract_digest",
        "dataset_id",
        "dataset_fingerprint",
        "train_membership_digest",
        "train_sample_count",
        "state",
    }
)


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
            dataset_id=combined_dataset_id(tuple(item.dataset_id for item in typed)),
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


def _package_id_membership(dataset: Dataset[Mapping[str, Any]]) -> dict[str, list[int]] | None:
    """Return exact package-owned ID membership when present."""
    dataset_identity = getattr(dataset, "identity", None)
    if not isinstance(dataset_identity, identity.DatasetIdentity) or dataset_identity.source_metadata is None:
        return None
    values = [metadata.get("dataset_membership") for metadata in dataset_identity.source_metadata]
    if all(value is None for value in values):
        return None
    if any(value not in {"train", "validation", "id_test"} for value in values):
        msg = "Package-owned ID membership must label every sample train, validation, or id_test."
        raise ValueError(msg)
    membership = {role: [index for index, value in enumerate(values) if value == role] for role in ("train", "validation", "id_test")}
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


def validate_split_info(
    split_info: Mapping[str, Any],
    *,
    train_identity: identity.DatasetIdentity | None = None,
    ood_identity: identity.DatasetIdentity | None = None,
    expected_train_ratio: float | None = None,
    expected_ood_fraction: float | None = None,
    expected_split_seed: int | None = None,
) -> dict[str, Tensor]:
    """
    Validate split membership against exact ordered dataset identity.

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
    dict[str, Tensor]
        Isolated CPU ``long`` train, eval, and OOD membership tensors in their
        persisted order.

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
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != identity.SPLIT_SCHEMA_VERSION:
        msg = f"split_indices.pt schema_version must be the current value {identity.SPLIT_SCHEMA_VERSION}."
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
    return validated


def data_processor_from_state(
    state: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> DefaultDataProcessor:
    """
    Reconstruct a data processor from persisted normalizer tensors.

    Parameters
    ----------
    state : Mapping[str, Any]
        Exact four-key normalizer state. Means and standard deviations must be
        finite real tensors shaped ``[1, channels, 1, 1]``. Standard deviations
        may be zero because the processor applies its denominator floor.
    device : torch.device | str, optional
        Target processor device for isolated state tensors.

    Returns
    -------
    DefaultDataProcessor
        Processor containing detached, cloned input/output normalization state
        on ``device`` with normalization axes ``[0, 2, 3]``.

    Raises
    ------
    TypeError
        If the state/key tensors have incompatible container, dtype, or layout
        types.
    ValueError
        If keys, shapes, finiteness, mean/std pairing, or non-negative standard
        deviations violate the persisted normalizer contract.

    """
    if not isinstance(state, Mapping):
        msg = "Saved normalizer state must be a mapping."
        raise TypeError(msg)
    missing = sorted(set(_REQUIRED_NORMALIZER_STATE_KEYS).difference(state))
    unexpected = sorted(set(state).difference(_REQUIRED_NORMALIZER_STATE_KEYS))
    if missing or unexpected:
        msg = f"Saved normalizer state keys do not match. Missing: {missing}. Unexpected: {unexpected}."
        raise ValueError(msg)

    tensors: dict[str, Tensor] = {}
    for key in _REQUIRED_NORMALIZER_STATE_KEYS:
        value = state.get(key)
        if not isinstance(value, Tensor):
            msg = f"Saved normalizer state {key!r} must be a torch.Tensor."
            raise TypeError(msg)
        if not value.is_floating_point() or value.is_complex():
            msg = f"Saved normalizer state {key!r} must be a real floating-point tensor."
            raise TypeError(msg)
        if value.ndim != _NORMALIZER_STATE_RANK or value.shape[0] != 1 or value.shape[2:] != (1, 1):
            msg = f"Saved normalizer state {key!r} must have shape (1, channels, 1, 1), got {tuple(value.shape)}."
            raise ValueError(msg)
        tensors[key] = value

    if tensors["in_normalizer.mean"].shape != tensors["in_normalizer.std"].shape:
        msg = "Saved input normalizer mean/std shapes do not match."
        raise ValueError(msg)
    if tensors["out_normalizer.mean"].shape != tensors["out_normalizer.std"].shape:
        msg = "Saved output normalizer mean/std shapes do not match."
        raise ValueError(msg)
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        msg = "Saved normalizer state contains non-finite values."
        raise ValueError(msg)
    if torch.any(tensors["in_normalizer.std"] < 0) or torch.any(tensors["out_normalizer.std"] < 0):
        msg = "Saved normalizer standard deviations must be non-negative."
        raise ValueError(msg)
    target_device = torch.device(device)
    in_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3], eps=_NORMALIZER_DENOMINATOR_FLOOR)
    out_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3], eps=_NORMALIZER_DENOMINATOR_FLOOR)
    processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
    )
    in_normalizer.mean = tensors["in_normalizer.mean"].detach().clone().to(target_device)
    in_normalizer.std = tensors["in_normalizer.std"].detach().clone().to(target_device)
    out_normalizer.mean = tensors["out_normalizer.mean"].detach().clone().to(target_device)
    out_normalizer.std = tensors["out_normalizer.std"].detach().clone().to(target_device)
    processor.device = target_device
    return processor


def _normalizer_split_identity(
    split_info: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, int]:
    """Return exact train-dataset, membership-digest, and sample-count evidence."""
    metadata = split_info.get("metadata")
    if not isinstance(metadata, Mapping):
        message = "Split contract metadata is required for normalizer identity."
        raise TypeError(message)
    datasets_metadata = metadata.get("datasets")
    membership_digests = metadata.get("membership_digests")
    if not isinstance(datasets_metadata, Mapping) or not isinstance(membership_digests, Mapping):
        message = "Split contract lacks dataset or membership identity for the normalizer."
        raise TypeError(message)
    train_dataset = datasets_metadata.get("train")
    train_membership_digest = membership_digests.get("train")
    train_indices = split_info.get("train_indices")
    if (
        not isinstance(train_dataset, Mapping)
        or not isinstance(train_membership_digest, str)
        or len(train_membership_digest) != _SHA256_HEX_LENGTH
        or not isinstance(train_indices, Tensor)
        or train_indices.ndim != 1
        or train_indices.numel() < 1
    ):
        message = "Split contract train identity is malformed for normalizer binding."
        raise ValueError(message)
    return train_dataset, train_membership_digest, int(train_indices.numel())


def build_normalizer_artifact(
    data_processor: DefaultDataProcessor,
    *,
    task: TaskSpec,
    split_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one persisted normalizer bound to exact train data membership."""
    train_dataset, membership_digest, train_sample_count = _normalizer_split_identity(split_info)
    state = {key: value.detach().cpu().clone() for key, value in data_processor.state_dict().items()}
    data_processor_from_state(state, device="cpu")
    dataset_id = train_dataset.get("dataset_id")
    dataset_fingerprint = train_dataset.get("fingerprint")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(dataset_fingerprint, str)
        or len(dataset_fingerprint) != _SHA256_HEX_LENGTH
        or train_dataset.get("data_contract_digest") != task.data_contract_digest
    ):
        message = "Split train dataset identity is incompatible with the task normalizer contract."
        raise ValueError(message)
    return {
        "schema_kind": _NORMALIZER_ARTIFACT_SCHEMA_KIND,
        "schema_version": _NORMALIZER_ARTIFACT_SCHEMA_VERSION,
        "task": task.id,
        "data_contract_digest": task.data_contract_digest,
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "train_membership_digest": membership_digest,
        "train_sample_count": train_sample_count,
        "state": state,
    }


def validate_normalizer_artifact(
    artifact: Mapping[str, Any],
    *,
    task: TaskSpec,
    split_info: Mapping[str, Any],
) -> dict[str, Tensor]:
    """Validate and return isolated tensors from one dataset-bound normalizer."""
    if not isinstance(artifact, Mapping) or set(artifact) != _NORMALIZER_ARTIFACT_KEYS:
        message = "Saved normalizer artifact keys do not match the dataset-bound schema."
        raise ValueError(message)
    train_dataset, membership_digest, train_sample_count = _normalizer_split_identity(split_info)
    expected = {
        "schema_kind": _NORMALIZER_ARTIFACT_SCHEMA_KIND,
        "schema_version": _NORMALIZER_ARTIFACT_SCHEMA_VERSION,
        "task": task.id,
        "data_contract_digest": task.data_contract_digest,
        "dataset_id": train_dataset.get("dataset_id"),
        "dataset_fingerprint": train_dataset.get("fingerprint"),
        "train_membership_digest": membership_digest,
        "train_sample_count": train_sample_count,
    }
    observed = {key: artifact.get(key) for key in expected}
    if observed != expected:
        message = "Saved normalizer identity does not match the exact task, dataset, fingerprint, and train membership."
        raise ValueError(message)
    state = artifact.get("state")
    if not isinstance(state, Mapping):
        message = "Saved normalizer artifact state must be a mapping."
        raise TypeError(message)
    data_processor_from_state(state, device="cpu")
    return {key: cast("Tensor", value).detach().cpu().clone() for key, value in state.items()}


def _make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """
    Create a worker_init_fn for deterministic DataLoader worker seeding.

    When num_workers > 0, PyTorch spawns worker processes. Each worker
    must have its RNG seeded independently but deterministically.

    Parameters
    ----------
    base_seed : int
        Base seed for the worker pool.

    Returns
    -------
    callable
        Function to pass as worker_init_fn to DataLoader.

    """

    def worker_init_fn(worker_id: int) -> None:
        """Seed the worker's random state."""
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))  # noqa: NPY002 -- worker process RNG
        torch.manual_seed(worker_seed)

    return worker_init_fn


def create_dataloaders(
    path_train: str,
    path_test_ood: str | Sequence[str],
    *,
    task: TaskSpec,
    train_dataset_id: str,
    ood_dataset_id: str | Sequence[str],
    batch_size: int = 16,
    train_ratio: float = 0.8,
    ood_fraction: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    split_seed: int = 9,
    loader_seed: int | None = None,
    worker_seed: int | None = None,
    split_indices: Mapping[str, Any] | None = None,
    data_processor: DefaultDataProcessor | None = None,
) -> tuple[DataLoader, dict[str, DataLoader], DefaultDataProcessor, dict[str, Any]]:
    """
    Create task-aware dataloaders bound to exact dataset identity.

    Parameters
    ----------
    path_train : str
        Current resolved training dataset path.
    path_test_ood : str or Sequence[str]
        One or more independently resolved OOD package paths.
    task : TaskSpec
        Authoritative task contract.
    train_dataset_id : str
        Expected logical training dataset identifier.
    ood_dataset_id : str or Sequence[str]
        Expected logical OOD package identifiers in path order.
    batch_size : int, optional
        Batch size for all loaders.
    train_ratio : float, optional
        Fraction in ``(0, 1)`` assigned to training. The remainder is evaluation.
        Counts use ``int(train_ratio * full_count)``.
    ood_fraction : float, optional
        Fraction in ``(0, 1]`` selected from the OOD dataset, also rounded down
        with ``int``.
    num_workers : int, optional
        Training DataLoader worker count.
    pin_memory : bool, optional
        Whether the training loader pins host memory.
    persistent_workers : bool, optional
        Whether nonzero training workers persist across epochs.
    split_seed : int, optional
        Deterministic train/eval and OOD membership seed.
    loader_seed : int | None, optional
        Shuffled training-loader generator seed. Defaults to ``split_seed`` but
        does not change membership.
    worker_seed : int | None, optional
        Base Python/NumPy/PyTorch worker seed. Worker ``i`` receives
        ``worker_seed + i``. The default is ``loader_seed``.
    split_indices : Mapping[str, Any] | None, optional
        Saved exact membership to validate against datasets, settings, and
        membership digests. Omission creates deterministic new membership.
    data_processor : DefaultDataProcessor | None, optional
        Restored processor. When omitted, input/output normalizers are fit only
        on the selected training subset over TaskSpec normalization axes.

    Returns
    -------
    tuple[DataLoader, dict[str, DataLoader], DefaultDataProcessor, dict[str, Any]]
        A shuffled train loader, non-shuffled ``eval`` and ``ood`` loaders, a fitted
        or supplied processor, and the complete current split contract. This
        function does not persist the contract or processor.

    Raises
    ------
    TypeError
        If seeds/settings, restored split state, or factory datasets violate
        required types or fail to expose verified ``DatasetIdentity`` objects.
    ValueError
        If ratios select an empty split, logical IDs disagree with payloads, or
        saved membership/settings/identity fail strict validation.

    Notes
    -----
    ``num_workers=0`` forces ``persistent_workers=False``. Evaluation and OOD
    loaders always use the main process without pinned memory. Fitting a new
    processor materializes the complete selected training tensors in memory.
    Caller-supplied processors are reused without refitting.

    """
    train_ratio = _normalized_fraction(train_ratio, label="train_ratio", allow_one=False)
    ood_fraction = _normalized_fraction(ood_fraction, label="ood_fraction", allow_one=True)
    split_seed = _normalized_seed(split_seed, label="split_seed")
    loader_seed = _normalized_seed(split_seed if loader_seed is None else loader_seed, label="loader_seed")
    worker_seed = _normalized_seed(loader_seed if worker_seed is None else worker_seed, label="worker_seed")
    if num_workers == 0:
        persistent_workers = False

    raw_train = runtime_factory.create_steady_dataset(path_train, task=task)
    raw_train_identity = getattr(raw_train, "identity", None)
    if not isinstance(raw_train_identity, identity.DatasetIdentity):
        msg = "Task dataset factory must expose a verified DatasetIdentity."
        raise TypeError(msg)
    if raw_train_identity.dataset_id != train_dataset_id:
        msg = f"Resolved training dataset identifier does not match its payload: {raw_train_identity.dataset_id!r}/{train_dataset_id!r}."
        raise ValueError(msg)

    ood_paths = (path_test_ood,) if isinstance(path_test_ood, str) else tuple(path_test_ood)
    ood_ids = (ood_dataset_id,) if isinstance(ood_dataset_id, str) else tuple(ood_dataset_id)
    if not ood_paths or len(ood_paths) != len(ood_ids):
        msg = "OOD package paths and identifiers must be non-empty and aligned."
        raise ValueError(msg)
    ood_sources = [runtime_factory.create_steady_dataset(path, task=task) for path in ood_paths]
    for expected_id, source_dataset in zip(ood_ids, ood_sources, strict=True):
        source_identity = getattr(source_dataset, "identity", None)
        if not isinstance(source_identity, identity.DatasetIdentity) or source_identity.dataset_id != expected_id:
            msg = f"Resolved OOD package identifier does not match its payload: {expected_id!r}."
            raise ValueError(msg)
    ood_full = combine_identity_datasets(ood_sources)
    ood_identity = getattr(ood_full, "identity", None)
    if not isinstance(ood_identity, identity.DatasetIdentity):
        msg = "Combined OOD packages did not expose a verified DatasetIdentity."
        raise TypeError(msg)

    package_membership = _package_id_membership(raw_train)
    id_test_set: Dataset[Mapping[str, Any]] | None = None
    if package_membership is None:
        full_train: Dataset[Mapping[str, Any]] = raw_train
        explicit_train_count = None
    else:
        train_positions = package_membership["train"]
        validation_positions = package_membership["validation"]
        full_train = _IdentityDatasetView(
            raw_train,
            [*train_positions, *validation_positions],
            label="package train+validation",
        )
        id_test_set = _IdentityDatasetView(
            raw_train,
            package_membership["id_test"],
            label="package id_test",
        )
        explicit_train_count = len(train_positions)
        train_ratio = explicit_train_count / len(cast("Sized", full_train))

    train_identity = getattr(full_train, "identity", None)
    if not isinstance(train_identity, identity.DatasetIdentity):
        msg = "Resolved training view did not expose a verified DatasetIdentity."
        raise TypeError(msg)
    n_train_full = len(cast("Sized", full_train))
    n_ood_full = len(cast("Sized", ood_full))
    if split_indices is None:
        n_train = int(train_ratio * n_train_full)
        n_eval = n_train_full - n_train
        n_ood = int(ood_fraction * n_ood_full)
        if min(n_train, n_eval, n_ood) <= 0:
            msg = f"Split settings must select non-empty train/eval/OOD sets. Received train={n_train}, eval={n_eval}, ood={n_ood}."
            raise ValueError(msg)
        if explicit_train_count is None:
            train_random, eval_random = random_split(
                full_train,
                [n_train, n_eval],
                generator=torch.Generator().manual_seed(split_seed),
            )
            train_indices = torch.tensor(train_random.indices, dtype=torch.long)
            eval_indices = torch.tensor(eval_random.indices, dtype=torch.long)
        else:
            n_train = explicit_train_count
            n_eval = n_train_full - n_train
            train_indices = torch.arange(n_train, dtype=torch.long)
            eval_indices = torch.arange(n_train, n_train_full, dtype=torch.long)
        ood_random, _ = random_split(
            ood_full,
            [n_ood, n_ood_full - n_ood],
            generator=torch.Generator().manual_seed(split_seed),
        )
        ood_indices = torch.tensor(ood_random.indices, dtype=torch.long)
        membership_digests = {
            "train": identity.membership_digest(
                role="train",
                dataset_fingerprint=train_identity.fingerprint,
                sample_ids=train_identity.sample_ids,
                indices=[int(value) for value in train_indices.tolist()],
            ),
            "eval": identity.membership_digest(
                role="eval",
                dataset_fingerprint=train_identity.fingerprint,
                sample_ids=train_identity.sample_ids,
                indices=[int(value) for value in eval_indices.tolist()],
            ),
            "ood": identity.membership_digest(
                role="ood",
                dataset_fingerprint=ood_identity.fingerprint,
                sample_ids=ood_identity.sample_ids,
                indices=[int(value) for value in ood_indices.tolist()],
            ),
        }
        split_info: dict[str, Any] = {
            "schema_version": identity.SPLIT_SCHEMA_VERSION,
            "task": task.id,
            "task_contract_digest": task.contract_digest,
            "train_indices": train_indices,
            "eval_indices": eval_indices,
            "ood_indices": ood_indices,
            "metadata": {
                "datasets": {
                    "train": train_identity.as_dict(),
                    "ood": ood_identity.as_dict(),
                },
                "n_train_full": n_train_full,
                "n_train": n_train,
                "n_eval": n_eval,
                "n_ood_full": n_ood_full,
                "n_ood": n_ood,
                "train_ratio": train_ratio,
                "ood_fraction": ood_fraction,
                "split_seed": split_seed,
                "membership_digests": membership_digests,
            },
        }
    else:
        split_info = dict(split_indices)

    validated_indices = validate_split_info(
        split_info,
        train_identity=train_identity,
        ood_identity=ood_identity,
        expected_train_ratio=train_ratio,
        expected_ood_fraction=ood_fraction,
        expected_split_seed=split_seed,
    )
    split_info.update(validated_indices)
    train_set = Subset(full_train, validated_indices["train_indices"].tolist())
    eval_set = Subset(full_train, validated_indices["eval_indices"].tolist())
    ood_subset = Subset(ood_full, validated_indices["ood_indices"].tolist())

    if data_processor is None:
        xs_train: list[Tensor] = []
        ys_train: list[Tensor] = []
        fitting_loader = runtime_factory.make_data_loader(
            train_set,
            runtime_factory.LoaderSettings(batch_size=batch_size),
        )
        for batch in fitting_loader:
            xs_train.append(batch["x"])
            ys_train.append(batch["y"])
        x_train = torch.cat(xs_train, dim=0)
        y_train = torch.cat(ys_train, dim=0)
        normalization_axes = list(task.normalization_axes)
        in_norm = UnitGaussianNormalizer(dim=normalization_axes, eps=_NORMALIZER_DENOMINATOR_FLOOR)
        in_norm.fit(x_train)
        out_norm = UnitGaussianNormalizer(dim=normalization_axes, eps=_NORMALIZER_DENOMINATOR_FLOOR)
        out_norm.fit(y_train)
        data_processor = DefaultDataProcessor(in_normalizer=in_norm, out_normalizer=out_norm)

    generator = torch.Generator().manual_seed(loader_seed)
    worker_init = _make_worker_init_fn(worker_seed) if num_workers > 0 else None
    train_loader = runtime_factory.make_data_loader(
        train_set,
        runtime_factory.LoaderSettings(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            shuffle=True,
        ),
        generator=generator,
        worker_init_fn=worker_init,
    )
    evaluation_settings = runtime_factory.LoaderSettings(batch_size=batch_size)
    eval_loader = runtime_factory.make_data_loader(eval_set, evaluation_settings)
    ood_loader = runtime_factory.make_data_loader(ood_subset, evaluation_settings)
    test_loaders = {"eval": eval_loader, "ood": ood_loader}
    if id_test_set is not None:
        test_loaders["id_test"] = runtime_factory.make_data_loader(
            id_test_set,
            evaluation_settings,
        )
    return train_loader, test_loaders, cast("DefaultDataProcessor", data_processor), split_info
