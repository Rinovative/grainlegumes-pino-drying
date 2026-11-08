"""Own the one persisted Experiment split schema of both tasks and its admission.

Responsibilities:
  - Build the split of a run from manifest-bound Dataset identities and the Dataset-owned fixed partition
  - Admit a persisted split with one function for both tasks, read a run's saved split and expose immutable role evidence
  - Name the additional Evaluation role from its ordered parts for every artifact producer and reader
  - Replay a saved split only when it equals the split rebuilt from the current Datasets
  - Refuse a reopened source Dataset whose identity or fixed partition contradicts its saved split part
  - Construct identity-preserving ordered combinations of evaluation Datasets

Design principles:
  - Train, validation and ID-test follow the Dataset-owned fixed case partition in its recorded order
  - Every additional Evaluation source keeps its own Dataset identity, case membership and assignment
  - The shared core binds manifest identity and case membership; one task extension holds only the transient item expansion
  - Admission is fail-closed: exact keys, task contract, digests and membership must agree

This module does NOT:
  - Fit preprocessing or construct DataLoaders
  - Write split payloads or resolve experiment configuration
  - Choose Dataset roles; Dataset manifests own the fixed partition and the declared OOD cases

Notes:
  Persisted payload (``split_indices.pt``):
    ``schema_kind``, ``schema_version``, ``task``, ``task_contract_digest``, ``split_seed``, ``case_partition``,
    ``case_partition_digest``, ``dataset_identity`` (``train`` and the ordered ``additional_evaluation`` list),
    ``roles`` (``train``, ``validation``, ``id_test`` and ``additional_evaluation.parts``, each with ordered
    ``case_ids`` and a ``membership_digest``), ``evaluation_assignments`` (one per additional part) and
    ``extension`` (``None`` for steady flow; the transient tensorizer selection, Train and evaluation sampling and
    spatial view).
  A Dataset identity is the manifest binding (``dataset_id``, ``dataset_digest``, ``channel_contract_digest``,
  ordered ``case_ids``) plus the identity its payload format states (steady ``fingerprint`` and ``spatial_shape``;
  transient ``index_digest`` and ``configured_regular_horizon``).

"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src import common, domain
from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views

if TYPE_CHECKING:
    from collections.abc import Sized
    from pathlib import Path


SPLIT_SCHEMA_KIND: Final = "experiment_case_split"
SPLIT_SCHEMA_VERSION: Final = 1
SplitRole = Literal["train", "validation", "id_test", "additional_evaluation"]
ID_SPLIT_ROLES: tuple[SplitRole, ...] = ("train", "validation", "id_test")
_SPLIT_KEYS: Final = frozenset(
    {
        "schema_kind",
        "schema_version",
        "task",
        "task_contract_digest",
        "split_seed",
        "case_partition",
        "case_partition_digest",
        "dataset_identity",
        "roles",
        "evaluation_assignments",
        "extension",
    }
)
_CORE_IDENTITY_KEYS: Final = frozenset({"dataset_id", "dataset_digest", "channel_contract_digest", "case_ids"})
_PAYLOAD_IDENTITY_KEYS: Final = {
    "steady_flow": frozenset({"fingerprint", "spatial_shape"}),
    "transient_drying": frozenset({"index_digest", "configured_regular_horizon"}),
}
_TRANSIENT_EXTENSION_KEYS: Final = frozenset({"tensorizer", "sampling", "evaluation_sampling", "spatial_view"})
_SPATIAL_VIEW_KEYS: Final = ("spatial_stride", "canonical_ny", "canonical_nx", "effective_ny", "effective_nx")
# The one producer assigns every additional part "ood"; the reopen check in require_source_identity maps that role to the "ood" partition.
_ASSIGNMENT_ROLES: Final = frozenset({"ood"})
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_MINIMUM_SPATIAL_AXIS_LENGTH: Final = 2
_SPATIAL_RANK: Final = 2


@dataclass(frozen=True, slots=True)
class SplitRoleEvidence:
    """Immutable Dataset identity and ordered case membership of one split role or Evaluation source."""

    name: SplitRole
    dataset: Mapping[str, Any]
    case_ids: tuple[str, ...]
    membership_digest: str

    @property
    def dataset_id(self) -> str:
        """Return the logical Dataset ID of this role's source."""
        return str(self.dataset["dataset_id"])

    @property
    def count(self) -> int:
        """Return the number of member cases."""
        return len(self.case_ids)

    @property
    def full_count(self) -> int:
        """Return the number of cases in the complete source Dataset."""
        return len(self.dataset["case_ids"])

    @property
    def index_values(self) -> tuple[int, ...]:
        """Return member positions in the source Dataset's ordered cases."""
        position_by_id = {case_id: index for index, case_id in enumerate(self.dataset["case_ids"])}
        return tuple(position_by_id[case_id] for case_id in self.case_ids)

    @property
    def indices(self) -> Tensor:
        """Return an isolated CPU ``long`` tensor of member positions in persisted order."""
        return torch.tensor(self.index_values, dtype=torch.long)


@dataclass(frozen=True, slots=True)
class SplitContract:
    """Immutable admitted split with role-oriented evidence access for both tasks."""

    task: str
    task_contract_digest: str
    split_seed: int
    case_partition: Mapping[str, Any]
    case_partition_digest: str
    train: SplitRoleEvidence
    validation: SplitRoleEvidence
    id_test: SplitRoleEvidence
    additional_evaluation: tuple[SplitRoleEvidence, ...]
    evaluation_assignments: tuple[Mapping[str, Any], ...]
    extension: Mapping[str, Any] | None

    @property
    def schema_version(self) -> int:
        """Return the persisted schema version."""
        return SPLIT_SCHEMA_VERSION

    @property
    def roles(self) -> tuple[SplitRole, ...]:
        """Return the fixed-partition roles in canonical order."""
        return ID_SPLIT_ROLES

    def role(self, name: SplitRole) -> SplitRoleEvidence:
        """Return immutable evidence for one fixed-partition role."""
        if name not in ID_SPLIT_ROLES:
            msg = f"Split role {name!r} is not a fixed-partition role; additional Evaluation sources are listed separately."
            raise ValueError(msg)
        return cast("SplitRoleEvidence", getattr(self, name))

    @property
    def additional_evaluation_name(self) -> str | None:
        """Return the artifact Dataset name of the additional Evaluation role from its parts in split order, if any."""
        if not self.additional_evaluation:
            return None
        return identity.combined_dataset_id(tuple(part.dataset_id for part in self.additional_evaluation))

    def membership_digests(self) -> dict[str, Any]:
        """Return every role's ordered-membership digest, additional sources in order."""
        return {
            **{role: self.role(role).membership_digest for role in ID_SPLIT_ROLES},
            "additional_evaluation": [part.membership_digest for part in self.additional_evaluation],
        }

    def as_payload(self) -> dict[str, Any]:
        """Reproduce the caller-owned ``split_indices.pt`` payload."""

        def membership(evidence: SplitRoleEvidence) -> dict[str, Any]:
            return {"case_ids": list(evidence.case_ids), "membership_digest": evidence.membership_digest}

        return copy.deepcopy(
            {
                "schema_kind": SPLIT_SCHEMA_KIND,
                "schema_version": SPLIT_SCHEMA_VERSION,
                "task": self.task,
                "task_contract_digest": self.task_contract_digest,
                "split_seed": self.split_seed,
                "case_partition": dict(self.case_partition),
                "case_partition_digest": self.case_partition_digest,
                "dataset_identity": {
                    "train": dict(self.train.dataset),
                    "additional_evaluation": [dict(part.dataset) for part in self.additional_evaluation],
                },
                "roles": {
                    **{role: membership(self.role(role)) for role in ID_SPLIT_ROLES},
                    "additional_evaluation": {"parts": [membership(part) for part in self.additional_evaluation]},
                },
                "evaluation_assignments": [dict(item) for item in self.evaluation_assignments],
                "extension": None if self.extension is None else dict(self.extension),
            }
        )


def role_positions(source: identity.DatasetIdentity, case_ids: Sequence[str], *, label: str) -> tuple[int, ...]:
    """Map ordered Dataset case IDs to positions in one verified identity."""
    position_by_id = {sample_id: index for index, sample_id in enumerate(source.sample_ids)}
    try:
        return tuple(position_by_id[case_id] for case_id in case_ids)
    except KeyError as error:
        msg = f"Split role {label!r} references case {error.args[0]!r}, which {source.dataset_id!r} does not contain."
        raise ValueError(msg) from error


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


def split_dataset_identity(binding: identity.ManifestBinding, payload_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the persisted split identity of one runtime: its manifest binding plus its payload-format identity."""
    return copy.deepcopy({**binding.split_identity(), **dict(payload_identity)})


def require_source_identity(runtime: Any, expected: Mapping[str, Any], *, role: SplitRole, case_ids: Sequence[str]) -> None:
    """
    Refuse a reopened source Dataset whose identity or fixed partition differs from its saved split part.

    Parameters
    ----------
    runtime
        The reopened full-package runtime, carrying its ``manifest_binding`` and ``payload_identity``.
    expected
        The saved split part's Dataset identity.
    role
        The saved split role of the part; its partition entry is the role itself, except ``ood`` for an additional part.
    case_ids
        The saved ordered case membership of the part.

    Raises
    ------
    RuntimeError
        If the runtime's Dataset ID, manifest digest, channel contract, ordered case IDs or payload identity
        (steady fingerprint and spatial shape; transient index digest and horizon) differ from ``expected``, or the
        current manifest's fixed partition does not hold exactly ``case_ids`` for the part's role.

    """
    binding = getattr(runtime, "manifest_binding", None)
    if binding is None or split_dataset_identity(binding, runtime.payload_identity) != dict(expected):
        msg = f"Current package {expected.get('dataset_id')!r} contradicts its saved split Dataset identity."
        raise RuntimeError(msg)
    partition = binding.case_partition
    partition_role = "ood" if role == "additional_evaluation" else role
    if not isinstance(partition, Mapping) or partition.get(partition_role) != list(case_ids):
        msg = f"Saved {role} membership of {binding.dataset_id!r} contradicts the Dataset's fixed partition."
        raise RuntimeError(msg)


def reject_role_leakage(
    train: identity.ManifestBinding,
    case_partition: Mapping[str, Any],
    additional_evaluation: Sequence[tuple[identity.ManifestBinding, Sequence[str]]],
) -> None:
    """
    Refuse roles that share a physical source case, or a material holdout that reuses a Train material, for both tasks.

    Raises
    ------
    ValueError
        If two roles or parts contain the same case input or simulation, or an additional Evaluation part (every
        part is a Dataset-declared material holdout) contains a material family of the Train cases.

    """
    roles: list[tuple[str, identity.ManifestBinding, Sequence[str]]] = [(role, train, case_partition[role]) for role in ID_SPLIT_ROLES]
    roles += [(f"additional_evaluation[{index}]", binding, case_ids) for index, (binding, case_ids) in enumerate(additional_evaluation)]
    seen: set[str] = set()
    for label, binding, case_ids in roles:
        tokens = {token for case_id in case_ids for token in binding.physical_sources[case_id]}
        if seen & tokens:
            msg = f"Experiment {label} reuses a physical source case from another role."
            raise ValueError(msg)
        seen |= tokens
    train_materials = {train.materials[case_id] for case_id in case_partition["train"]}
    for binding, case_ids in additional_evaluation:
        if train_materials & {binding.materials[case_id] for case_id in case_ids}:
            msg = "Declared family-OOD evaluation reuses a training material family."
            raise ValueError(msg)


def resolved_event(split: SplitContract, *, items: int, grid: Sequence[int]) -> dict[str, Any]:
    """Return the ``dataset_resolved`` startup fields both tasks report for a resolved split."""
    return {
        "dataset_id": split.train.dataset_id,
        "cases": split.train.count,
        "items": items,
        "grid": "x".join(map(str, grid)),
        "additional_sources": len(split.additional_evaluation),
    }


def _membership_digest(role: str, dataset: Mapping[str, Any], case_ids: Sequence[str]) -> str:
    """Bind one ordered case membership to its role and source manifest digest."""
    return common.serialization.canonical_json_sha256({"role": role, "dataset_digest": dataset["dataset_digest"], "case_ids": list(case_ids)})


def build_split(
    *,
    task: str,
    train_dataset: Mapping[str, Any],
    case_partition: Mapping[str, Any],
    additional_evaluation: Sequence[tuple[Mapping[str, Any], Sequence[str], Mapping[str, Any]]],
    extension: Mapping[str, Any] | None,
) -> SplitContract:
    """
    Build and admit the split of one run from manifest-bound Dataset identities.

    Parameters
    ----------
    task : str
        Registered task ID.
    train_dataset : Mapping[str, Any]
        Split identity of the training Dataset (:func:`split_dataset_identity`).
    case_partition : Mapping[str, Any]
        The training Dataset's validated fixed case partition.
    additional_evaluation : Sequence[tuple[Mapping[str, Any], Sequence[str], Mapping[str, Any]]]
        Ordered ``(split identity, declared case IDs, evaluation assignment)`` per additional Evaluation source.
    extension : Mapping[str, Any] | None
        Task extension: ``None`` for steady flow, the transient item expansion otherwise.

    Returns
    -------
    SplitContract
        The admitted split.

    """
    registered = domain.tasks.registry.get_task(task)

    def membership(role: str, dataset: Mapping[str, Any], case_ids: Sequence[str]) -> dict[str, Any]:
        return {"case_ids": list(case_ids), "membership_digest": _membership_digest(role, dataset, case_ids)}

    payload = {
        "schema_kind": SPLIT_SCHEMA_KIND,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "task": task,
        "task_contract_digest": registered.contract_digest,
        "split_seed": case_partition["seed"],
        "case_partition": copy.deepcopy(dict(case_partition)),
        "case_partition_digest": partition_contract.partition_digest(case_partition),
        "dataset_identity": {"train": dict(train_dataset), "additional_evaluation": [dict(dataset) for dataset, _, _ in additional_evaluation]},
        "roles": {
            **{role: membership(role, train_dataset, case_partition[role]) for role in ID_SPLIT_ROLES},
            "additional_evaluation": {
                "parts": [membership("additional_evaluation", dataset, case_ids) for dataset, case_ids, _ in additional_evaluation]
            },
        },
        "evaluation_assignments": [dict(assignment) for _, _, assignment in additional_evaluation],
        "extension": None if extension is None else dict(extension),
    }
    return admit_split(payload)


def read_saved_split(run_dir: Path | str) -> SplitContract:
    """
    Load and admit the split persisted in one run directory.

    Raises
    ------
    FileNotFoundError
        If the run has no saved split.
    ValueError, TypeError
        If the file is unreadable or its payload is not an admissible split.

    """
    path = common.paths.resolve_split_indices_path(run_dir)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, EOFError, pickle.UnpicklingError) as error:
        msg = f"Saved split is unreadable: {path}."
        raise ValueError(msg) from error
    return admit_split(payload)


def resolve_split(rebuilt: SplitContract, saved: Mapping[str, Any] | None) -> SplitContract:
    """Return the rebuilt split, or the admitted saved split when it equals the rebuilt one exactly."""
    if saved is None:
        return rebuilt
    admitted = admit_split(saved)
    if admitted != rebuilt:
        msg = "Saved split does not match the current Dataset manifests, partition, evaluation membership or item expansion."
        raise ValueError(msg)
    return admitted


def _sha256_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        msg = f"Split {label} must be a lowercase SHA-256 digest."
        raise ValueError(msg)
    return value


def _case_list(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value) or len(set(value)) != len(value):
        msg = f"Split {label} case IDs must be a non-empty list of unique non-empty strings."
        raise ValueError(msg)
    return tuple(value)


def _admit_dataset_identity(value: Any, *, task: str, label: str) -> dict[str, Any]:
    """Admit one Dataset identity: manifest binding plus the task's payload-format identity."""
    expected_keys = _CORE_IDENTITY_KEYS | _PAYLOAD_IDENTITY_KEYS[task]
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        msg = f"Split {label} Dataset identity keys must be exactly {sorted(expected_keys)}."
        raise ValueError(msg)
    common.paths.validate_logical_name(value["dataset_id"], label=f"split {label} dataset_id")
    _sha256_text(value["dataset_digest"], label=f"{label} dataset_digest")
    if value["channel_contract_digest"] != views.get_view(task).contract_digest:
        msg = f"Split {label} Dataset channel contract disagrees with the task's Dataset view."
        raise ValueError(msg)
    _case_list(value["case_ids"], label=f"{label} Dataset")
    if task == "steady_flow":
        _sha256_text(value["fingerprint"], label=f"{label} fingerprint")
        shape = value["spatial_shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != _SPATIAL_RANK
            or any(isinstance(axis, bool) or not isinstance(axis, int) or axis < _MINIMUM_SPATIAL_AXIS_LENGTH for axis in shape)
        ):
            msg = f"Split {label} spatial_shape must be two integer axis lengths of at least {_MINIMUM_SPATIAL_AXIS_LENGTH}."
            raise ValueError(msg)
    else:
        _sha256_text(value["index_digest"], label=f"{label} index_digest")
        temporal = transient_contract.TRANSIENT_STEP_CONTRACT
        horizon = value["configured_regular_horizon"]
        if (
            not isinstance(horizon, Mapping)
            or set(horizon) != {"value", "unit", "source"}
            or isinstance(horizon["value"], bool)
            or not isinstance(horizon["value"], (int, float))
            or not math.isfinite(horizon["value"])
            or horizon["value"] <= 0
            or [horizon["unit"], horizon["source"]] != [temporal.time_unit, temporal.temporal.configured_horizon_source]
        ):
            msg = f"Split {label} configured_regular_horizon must be the positive horizon of the transient contract."
            raise ValueError(msg)
    return copy.deepcopy(dict(value))


def _admit_membership(value: Any, *, role: SplitRole, dataset: Mapping[str, Any], label: str) -> SplitRoleEvidence:
    """Admit one ordered case membership against its source identity and digest."""
    if not isinstance(value, Mapping) or set(value) != {"case_ids", "membership_digest"}:
        msg = f"Split {label} membership must contain exactly case_ids and membership_digest."
        raise ValueError(msg)
    case_ids = _case_list(value["case_ids"], label=label)
    if not set(case_ids) <= set(dataset["case_ids"]):
        msg = f"Split {label} references cases that {dataset['dataset_id']!r} does not contain."
        raise ValueError(msg)
    if value["membership_digest"] != _membership_digest(role, dataset, case_ids):
        msg = f"Split {label} membership digest mismatch."
        raise ValueError(msg)
    return SplitRoleEvidence(name=role, dataset=dataset, case_ids=case_ids, membership_digest=value["membership_digest"])


def _admit_assignment(value: Any, *, dataset_id: str, index: int) -> dict[str, Any]:
    """Admit one explicit evaluation assignment without inferring roles from Dataset names."""
    if not isinstance(value, Mapping) or set(value) != {"dataset_id", "role", "relation", "assumptions"}:
        msg = f"Split evaluation assignment {index} has invalid keys."
        raise ValueError(msg)
    if value["dataset_id"] != dataset_id or value["role"] not in _ASSIGNMENT_ROLES:
        msg = f"Split evaluation assignment {index} disagrees with its additional Evaluation source."
        raise ValueError(msg)
    if not isinstance(value["relation"], str) or not value["relation"].strip():
        msg = "Split evaluation relation must explain the declared comparison."
        raise ValueError(msg)
    assumptions = value["assumptions"]
    if not isinstance(assumptions, list) or any(not isinstance(item, str) or not item.strip() for item in assumptions):
        msg = "Split evaluation assumptions must be a list of non-empty statements."
        raise ValueError(msg)
    return copy.deepcopy(dict(value))


def _admit_spatial_view(value: Any) -> dict[str, int]:
    """Admit one boundary-preserving canonical and effective spatial view."""
    if not isinstance(value, Mapping) or set(value) != set(_SPATIAL_VIEW_KEYS):
        msg = "Transient split spatial_view has unexpected fields."
        raise ValueError(msg)
    stride = transient_contract.validate_spatial_stride(value["spatial_stride"])
    axes = tuple(value[key] for key in _SPATIAL_VIEW_KEYS[1:])
    if any(isinstance(axis, bool) or not isinstance(axis, int) or axis < _MINIMUM_SPATIAL_AXIS_LENGTH for axis in axes):
        msg = "Transient split spatial-view axes must be integers >= 2."
        raise ValueError(msg)
    if transient_contract.resolve_spatial_view((axes[0], axes[1]), stride) != (axes[2], axes[3]):
        msg = "Transient split effective grid does not match its canonical grid and stride."
        raise ValueError(msg)
    return dict(value)


def _admit_extension(value: Any, *, task: str) -> dict[str, Any] | None:
    """Admit the task extension: none for steady flow, the exact transient item expansion otherwise."""
    if task == "steady_flow":
        if value is not None:
            msg = "A steady split carries no task extension."
            raise ValueError(msg)
        return None
    from src.learning.transient.learning_transient_contracts import TransientTensorizerSpec  # noqa: PLC0415 -- learning imports datasets

    if not isinstance(value, Mapping) or set(value) != _TRANSIENT_EXTENSION_KEYS:
        msg = f"Transient split extension keys must be exactly {sorted(_TRANSIENT_EXTENSION_KEYS)}."
        raise ValueError(msg)
    canonical = {
        "tensorizer": TransientTensorizerSpec.from_mapping(value["tensorizer"]).selection_dict(),
        "sampling": transient_contract.TransientSamplingSpec.from_mapping(value["sampling"]).as_dict(),
        "evaluation_sampling": transient_contract.TransientSamplingSpec.from_mapping(value["evaluation_sampling"]).as_dict(),
        "spatial_view": _admit_spatial_view(value["spatial_view"]),
    }
    if canonical != dict(value):
        msg = "Transient split extension is not in canonical form."
        raise ValueError(msg)
    if canonical["evaluation_sampling"] != transient_contract.EVALUATION_SAMPLING.as_dict():
        msg = "Transient split evaluation_sampling must be whole-trajectory evaluation."
        raise ValueError(msg)
    return copy.deepcopy(canonical)


def admit_split(value: Mapping[str, Any]) -> SplitContract:
    """
    Admit one persisted split of either task.

    Parameters
    ----------
    value : Mapping[str, Any]
        Current ``split_indices.pt`` payload.

    Returns
    -------
    SplitContract
        Immutable task, Dataset identity, partition, membership and extension evidence.

    Raises
    ------
    TypeError
        If the payload is not a mapping or its seed is not an integer.
    ValueError
        If schema, task contract, Dataset identities, partition, memberships, assignments or the task extension
        disagree with the current contract.

    """
    if not isinstance(value, Mapping):
        msg = "split_indices.pt must contain a mapping."
        raise TypeError(msg)
    if (
        set(value) != _SPLIT_KEYS
        or value["schema_kind"] != SPLIT_SCHEMA_KIND
        or not common.serialization.is_schema_version(value["schema_version"], SPLIT_SCHEMA_VERSION)
    ):
        msg = f"split_indices.pt does not match the current {SPLIT_SCHEMA_KIND!r} schema version {SPLIT_SCHEMA_VERSION}."
        raise ValueError(msg)
    task = value["task"]
    if task not in _PAYLOAD_IDENTITY_KEYS:
        msg = f"Split task must be one of {sorted(_PAYLOAD_IDENTITY_KEYS)}."
        raise ValueError(msg)
    if value["task_contract_digest"] != domain.tasks.registry.get_task(task).contract_digest:
        msg = f"split_indices.pt task_contract_digest does not match the current registered task {task!r}."
        raise ValueError(msg)
    seed = value["split_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = "Split seed must be an integer."
        raise TypeError(msg)
    partition = value["case_partition"]
    if not isinstance(partition, Mapping) or partition.get("seed") != seed or not {*ID_SPLIT_ROLES, "ood"} <= set(partition):
        msg = "split_indices.pt case_partition is not the Dataset fixed partition of its split seed."
        raise ValueError(msg)
    if value["case_partition_digest"] != partition_contract.partition_digest(partition):
        msg = "split_indices.pt case partition digest mismatch."
        raise ValueError(msg)
    datasets_value = value["dataset_identity"]
    roles = value["roles"]
    if not isinstance(datasets_value, Mapping) or set(datasets_value) != {"train", "additional_evaluation"}:
        msg = "Split dataset_identity must contain exactly train and additional_evaluation."
        raise ValueError(msg)
    if not isinstance(roles, Mapping) or set(roles) != {*ID_SPLIT_ROLES, "additional_evaluation"}:
        msg = "Split roles must contain exactly train, validation, id_test and additional_evaluation."
        raise ValueError(msg)
    train_dataset = _admit_dataset_identity(datasets_value["train"], task=task, label="train")
    evidence = {role: _admit_membership(roles[role], role=role, dataset=train_dataset, label=role) for role in ID_SPLIT_ROLES}
    for role in ID_SPLIT_ROLES:
        if list(evidence[role].case_ids) != list(partition[role]):
            msg = f"split_indices.pt {role} membership disagrees with the Dataset case partition."
            raise ValueError(msg)
    partition_contract.require_disjoint_roles(partition)
    raw_identities = datasets_value["additional_evaluation"]
    raw_parts = roles["additional_evaluation"].get("parts") if isinstance(roles["additional_evaluation"], Mapping) else None
    raw_assignments = value["evaluation_assignments"]
    if (
        not isinstance(raw_identities, list)
        or not isinstance(raw_parts, list)
        or set(roles["additional_evaluation"]) != {"parts"}
        or not isinstance(raw_assignments, list)
        or not len(raw_identities) == len(raw_parts) == len(raw_assignments)
    ):
        msg = "Split additional Evaluation identities, memberships and assignments must be aligned lists."
        raise ValueError(msg)
    additional = []
    assignments = []
    for index, (raw_identity, raw_part, raw_assignment) in enumerate(zip(raw_identities, raw_parts, raw_assignments, strict=True)):
        label = f"additional_evaluation[{index}]"
        dataset = _admit_dataset_identity(raw_identity, task=task, label=label)
        additional.append(_admit_membership(raw_part, role="additional_evaluation", dataset=dataset, label=label))
        assignments.append(_admit_assignment(raw_assignment, dataset_id=dataset["dataset_id"], index=index))
    if len({part.dataset_id for part in additional}) != len(additional):
        msg = "split_indices.pt additional Evaluation parts must come from distinct Datasets."
        raise ValueError(msg)
    for part in additional:
        if part.dataset_id == train_dataset["dataset_id"] and (dict(part.dataset) != train_dataset or list(part.case_ids) != list(partition["ood"])):
            msg = "split_indices.pt additional Evaluation part of the training Dataset must hold exactly its fixed OOD cases."
            raise ValueError(msg)
    return SplitContract(
        task=task,
        task_contract_digest=value["task_contract_digest"],
        split_seed=seed,
        case_partition=copy.deepcopy(dict(partition)),
        case_partition_digest=value["case_partition_digest"],
        train=evidence["train"],
        validation=evidence["validation"],
        id_test=evidence["id_test"],
        additional_evaluation=tuple(additional),
        evaluation_assignments=tuple(assignments),
        extension=_admit_extension(value["extension"], task=task),
    )
