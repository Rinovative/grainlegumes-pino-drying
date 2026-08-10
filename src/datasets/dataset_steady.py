"""
===============================================================================
dataset_steady.py
===============================================================================
Load and select verified steady-flow tensor Dataset packages.
Responsibilities:
  - Load steady tensor payloads in authoritative TaskSpec channel order
  - Expose immutable identity-bound runtime selections without tensor copies
  - Apply manifest-owned ID-membership and parameter-OOD selectors
Design principles:
  - TaskSpec remains the sole learned-channel contract
  - Payload fingerprints are verified before runtime access
  - Selection identity binds source fingerprint, selector, and ordered sample IDs
This module does NOT:
  - Resolve package lifecycle paths or orchestrate DataLoaders
  - Fit normalizers, choose experiment splits, or build tensor payloads
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from torch.utils.data import Dataset

from src import domain

from . import dataset_identity as identity

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


class SteadyFlowMetadata(TypedDict, total=False):
    """Describe the source identity required on new steady package items."""

    dataset_id: str
    simulation_case_id: str
    case_input_id: str
    source_batch_id: str
    source_simulation_profile: str
    material_family: str
    evaluation_regime: str
    dataset_membership: str
    source_hdf5_sha256: str
    generator: dict[str, Any]


class SteadyFlowItem(TypedDict):
    """Expose one steady input/target pair and isolated source metadata."""

    x: torch.Tensor
    y: torch.Tensor
    meta: SteadyFlowMetadata


class SteadyFlowDataset(Dataset[SteadyFlowItem]):
    """Load and verify one steady tensor payload in authoritative task order."""

    def __init__(self, data_path: Path | str, *, task: TaskSpec) -> None:
        """Load one exact payload and verify its complete content fingerprint."""
        path = Path(data_path)
        if not path.is_file() or path.is_symlink():
            message = f"Steady training dataset is missing or unsafe: {path}."
            raise FileNotFoundError(message)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            message = f"Steady training dataset must contain one dictionary payload: {path}."
            raise TypeError(message)
        self.path = path
        self.task = task
        self.input_fields = list(task.input_names)
        self.output_fields = list(task.output_names)
        self.data = payload
        self.identity = identity.validate_training_dataset_payload(
            payload,
            task=task,
            verify_content=True,
        )
        self.inputs = payload["inputs"]
        self.outputs = payload["outputs"]

    def __len__(self) -> int:
        """Return the verified ordered steady sample count."""
        return self.identity.sample_count

    def __getitem__(self, index: int) -> SteadyFlowItem:
        """Return one task-ordered pair and a defensive metadata copy."""
        raw_meta = self.data["source_metadata"][index]
        if not isinstance(raw_meta, Mapping):
            message = f"Steady source_metadata[{index}] must be a mapping: {self.path}."
            raise TypeError(message)
        return {
            "x": self.inputs[index],
            "y": self.outputs[index],
            "meta": cast("SteadyFlowMetadata", deepcopy(dict(raw_meta))),
        }


class SteadyDatasetSelection(Dataset[SteadyFlowItem]):
    """Expose one identity-bound steady case selection without tensor copies."""

    def __init__(
        self,
        source: SteadyFlowDataset,
        indices: Sequence[int],
        *,
        selector: str,
    ) -> None:
        """Validate an ordered unique selection and derive its exact identity."""
        selected = tuple(indices)
        if not selected or len(selected) != len(set(selected)) or min(selected) < 0 or max(selected) >= len(source):
            message = f"Steady selector {selector!r} must resolve non-empty unique in-range positions."
            raise ValueError(message)
        self.source = source
        self.indices = selected
        self.input_fields = source.input_fields
        self.output_fields = source.output_fields
        sample_ids = tuple(source.identity.sample_ids[index] for index in selected)
        fingerprint_payload = json.dumps(
            {
                "source_fingerprint": source.identity.fingerprint,
                "selector": selector,
                "sample_ids": sample_ids,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        source_metadata = source.identity.source_metadata
        self.identity = identity.DatasetIdentity(
            dataset_id=source.identity.dataset_id,
            task=source.identity.task,
            data_contract_digest=source.identity.data_contract_digest,
            fingerprint=hashlib.sha256(fingerprint_payload).hexdigest(),
            sample_ids=sample_ids,
            sample_count=len(sample_ids),
            spatial_shape=source.identity.spatial_shape,
            source_metadata=(None if source_metadata is None else tuple(source_metadata[index] for index in selected)),
            source_provenance=source.identity.source_provenance,
        )

    def __len__(self) -> int:
        """Return selected case count."""
        return len(self.indices)

    def __getitem__(self, index: int) -> SteadyFlowItem:
        """Return one selected source sample."""
        return self.source[self.indices[index]]


def create_dataset(
    data_path: Path | str,
    *,
    task: TaskSpec | None = None,
) -> SteadyFlowDataset:
    """
    Construct the verified steady runtime Dataset implementation.

    Parameters
    ----------
    data_path : Path | str
        Persisted steady tensor payload.
    task : TaskSpec | None, optional
        Exact learned-data contract. The registered steady-flow task is used
        when omitted.

    Returns
    -------
    SteadyFlowDataset
        Verified task-ordered runtime Dataset.

    """
    resolved_task = domain.tasks.registry.get_task("steady_flow") if task is None else task
    return SteadyFlowDataset(data_path, task=resolved_task)


def _case_positions(
    dataset: SteadyFlowDataset,
    package_case_ids: Sequence[str],
    *,
    selector: str,
) -> tuple[int, ...]:
    """Resolve package case IDs to ordered steady payload positions."""
    position_by_id = {sample_id: index for index, sample_id in enumerate(dataset.identity.sample_ids)}
    if len(position_by_id) != len(dataset.identity.sample_ids):
        message = "Steady dataset sample identities are duplicated."
        raise ValueError(message)
    try:
        positions = tuple(position_by_id[case_id] for case_id in package_case_ids)
    except KeyError as error:
        message = f"Manifest selector {selector!r} references a missing steady sample {error.args[0]!r}."
        raise ValueError(message) from error
    if not positions:
        message = f"Manifest selector {selector!r} is unavailable in this steady package."
        raise ValueError(message)
    return positions


def select_dataset(
    dataset: SteadyFlowDataset,
    manifest: Mapping[str, Any],
    *,
    evaluation_regime: str,
    membership: str | None,
    ood_group: str | None,
) -> Dataset[SteadyFlowItem]:
    """
    Apply an explicit manifest-owned steady membership or OOD-group selector.

    Parameters
    ----------
    dataset : SteadyFlowDataset
        Verified complete steady package runtime.
    manifest : Mapping[str, Any]
        Validated package manifest containing case selectors.
    evaluation_regime : str
        Package evaluation regime.
    membership : str | None
        Optional ID membership selector.
    ood_group : str | None
        Optional parameter-OOD group selector.

    Returns
    -------
    Dataset[SteadyFlowItem]
        Complete source or immutable identity-bound selection.

    """
    package_case_ids: Sequence[str] | None = None
    selector = "all"
    if evaluation_regime == "id" and membership is not None:
        split_membership = manifest["split_membership"]
        package_case_ids = split_membership.get(membership)
        selector = f"id/{membership}"
    elif evaluation_regime == "parameter_ood" and ood_group is not None:
        available = manifest["available_ood_groups"]
        if ood_group not in available:
            message = f"Steady-flow parameter-OOD group {ood_group!r} is unavailable; available selectors are {available}."
            raise ValueError(message)
        package_case_ids = manifest["ood_group_indexes"].get(ood_group)
        selector = f"parameter_ood/{ood_group}"
    if package_case_ids is None:
        return dataset
    if not isinstance(package_case_ids, list) or not all(isinstance(value, str) for value in package_case_ids):
        message = f"Manifest selector {selector!r} is malformed."
        raise TypeError(message)
    return SteadyDatasetSelection(
        dataset,
        _case_positions(dataset, package_case_ids, selector=selector),
        selector=selector,
    )
