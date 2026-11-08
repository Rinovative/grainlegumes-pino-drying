"""Protect the split admission and role-leakage contract that both tasks share.

Responsibilities:
  - Check that roles never share a physical source case and material holdouts never reuse a Train material
  - Check that persisted splits refuse malformed payload identities and repeated or altered training-Dataset parts
  - Check that a reopened part must match the current manifest's fixed partition
  - Check that admission refuses a partition with a case in two roles and a transient evaluation sampling other than whole trajectories

Design principles:
  - Manifest bindings and splits are synthetic and test-owned; no package or tensor is read

This module does NOT:
  - Build loaders or fit scaling; the task loader tests and the lifecycle tests own those paths
"""

from __future__ import annotations

import copy
import types
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from support.synthetic_task import synthetic_split_payload

from src.datasets.contracts import dataset_contracts_identity as identity
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.preprocessing import dataset_preprocessing_splits as splits

if TYPE_CHECKING:
    from collections.abc import Callable

_TRAIN_PARTITION = {"train": ["t1"], "validation": ["t2"], "id_test": ["t3"]}


def _binding(dataset_id: str, cases: dict[str, tuple[str, str]]) -> identity.ManifestBinding:
    """Bind synthetic cases given as ``case_id -> (material family, physical source)``."""
    return identity.ManifestBinding(
        dataset_id=dataset_id,
        dataset_digest="d" * 64,
        channel_contract_digest="c" * 64,
        case_ids=tuple(cases),
        physical_sources={case_id: (source, f"simulation_{source}") for case_id, (_, source) in cases.items()},
        materials={case_id: material for case_id, (material, _) in cases.items()},
        case_partition=None,
    )


@pytest.mark.parametrize(
    ("train_sources", "holdout", "match"),
    [
        (("a", "b", "c"), ("chickpea", "d"), None),
        (("a", "a", "c"), ("chickpea", "d"), "validation reuses a physical source"),
        (("a", "b", "c"), ("chickpea", "a"), r"additional_evaluation\[0\] reuses a physical source"),
        (("a", "b", "c"), ("lentil", "d"), "reuses a training material"),
    ],
    ids=("disjoint", "id-roles-share-a-source", "holdout-shares-a-source", "holdout-reuses-train-material"),
)
def test_role_leakage_refuses_shared_sources_and_reused_training_materials(
    train_sources: tuple[str, str, str], holdout: tuple[str, str], match: str | None
) -> None:
    """Refuse leakage by physical source across every role and by material family for declared holdouts."""
    train = _binding("train_data", {case_id: ("lentil", source) for case_id, source in zip(("t1", "t2", "t3"), train_sources, strict=True)})
    additional = [(_binding("ood_data", {"o1": holdout}), ("o1",))]
    if match is None:
        splits.reject_role_leakage(train, _TRAIN_PARTITION, additional)
        return
    with pytest.raises(ValueError, match=match):
        splits.reject_role_leakage(train, _TRAIN_PARTITION, additional)


def _break_steady_shape(dataset: dict[str, Any]) -> None:
    dataset["spatial_shape"] = [8]


def _break_transient_horizon(dataset: dict[str, Any]) -> None:
    dataset["configured_regular_horizon"] = {**dataset["configured_regular_horizon"], "unit": "undeclared"}


@pytest.mark.parametrize(
    ("task", "mutate", "match"),
    [
        ("steady_flow", _break_steady_shape, "spatial_shape must be two integer axis lengths"),
        ("transient_drying", _break_transient_horizon, "configured_regular_horizon must be the positive horizon"),
    ],
    ids=("steady", "transient"),
)
def test_split_admission_refuses_a_malformed_payload_identity(task: str, mutate: Callable[[dict[str, Any]], None], match: str) -> None:
    """Admit only the payload identity each task's runtime states about itself."""
    payload = synthetic_split_payload(task, additional=())
    mutate(payload["dataset_identity"]["train"])
    with pytest.raises(ValueError, match=match):
        splits.admit_split(payload)


@pytest.mark.parametrize(
    ("role", "case_ids", "refused"),
    [
        ("validation", ["t2"], False),
        ("validation", ["t1"], True),
        ("additional_evaluation", ["t4", "t5"], True),
    ],
    ids=("id-role-matches", "id-role-contradicts-partition", "additional-part-contradicts-ood-partition"),
)
def test_require_source_identity_binds_a_reopened_part_to_the_fixed_partition(role: splits.SplitRole, case_ids: list[str], refused: bool) -> None:
    """Refuse a reopened part whose saved membership differs from the current manifest's fixed partition."""
    partition = {**_TRAIN_PARTITION, "ood": ["t4"]}
    binding = replace(
        _binding("train_data", {case_id: ("lentil", f"s{case_id}") for case_id in ("t1", "t2", "t3", "t4")}),
        case_partition=partition,
    )
    payload_identity = {"fingerprint": "f" * 64, "spatial_shape": [8, 8]}
    runtime = types.SimpleNamespace(manifest_binding=binding, payload_identity=payload_identity)
    expected = splits.split_dataset_identity(binding, payload_identity)
    if not refused:
        splits.require_source_identity(runtime, expected, role=role, case_ids=case_ids)
        return
    with pytest.raises(RuntimeError, match="contradicts the Dataset's fixed partition"):
        splits.require_source_identity(runtime, expected, role=role, case_ids=case_ids)


def test_split_admission_refuses_repeated_parts_and_an_altered_training_part() -> None:
    """Keep each additional source distinct and the training Dataset's part exactly its fixed OOD cases."""
    payload = synthetic_split_payload(
        "steady_flow", additional=(("ood_a", ("chickpea__case_0001__00000000000a",)), ("ood_b", ("pea__case_0002__00000000000b",)))
    )
    aligned = (payload["dataset_identity"]["additional_evaluation"], payload["roles"]["additional_evaluation"]["parts"])
    for parts in (*aligned, payload["evaluation_assignments"]):
        parts[1] = copy.deepcopy(parts[0])
    with pytest.raises(ValueError, match="parts must come from distinct Datasets"):
        splits.admit_split(payload)
    with pytest.raises(ValueError, match="part of the training Dataset must hold exactly its fixed OOD cases"):
        synthetic_split_payload("steady_flow", additional=(("train_data", ("lentil__case_0001__000000000001",)),))


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_split_admission_refuses_a_case_in_train_and_ood(task: str) -> None:
    """Refuse a self-consistent partition that lists a Train case also as OOD."""
    payload = synthetic_split_payload(task)
    partition = payload["case_partition"]
    partition["ood"] = [*partition["ood"], partition["train"][0]]
    payload["case_partition_digest"] = partition_contract.partition_digest(partition)
    with pytest.raises(ValueError, match="roles must be disjoint"):
        splits.admit_split(payload)


def test_split_admission_refuses_a_transient_evaluation_sampling_other_than_whole_trajectories() -> None:
    """Bind the persisted transient evaluation sampling to the one specification all loaders evaluate with."""
    payload = synthetic_split_payload("transient_drying")
    payload["extension"]["evaluation_sampling"] = {"mode": "one_step_transition"}
    with pytest.raises(ValueError, match="must be whole-trajectory evaluation"):
        splits.admit_split(payload)
