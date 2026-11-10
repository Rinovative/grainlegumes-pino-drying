"""Protect the split admission and role-leakage contract that both tasks share.

Responsibilities:
  - Check that roles never share a physical source case, parameter-OOD cases stay out of ID roles and material-OOD
    parts never reuse a Train material, while parameter-OOD parts hold ID materials by design
  - Check that material and parameter OOD stay separate parts of one split, an empty role adding none
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
_DESIGN_KINDS = {"t1": "train", "t2": "validation", "t3": "test", "t4": "test", "p1": "parameter_ood", "o1": "test"}


def _binding(dataset_id: str, cases: dict[str, tuple[str, str]], *, kind: str | None = None) -> identity.ManifestBinding:
    """Bind synthetic cases given as ``case_id -> (material family, physical source)``; design kinds are ``kind`` or ``_DESIGN_KINDS``."""
    return identity.ManifestBinding(
        dataset_id=dataset_id,
        dataset_digest="d" * 64,
        channel_contract_digest="c" * 64,
        case_ids=tuple(cases),
        physical_sources={case_id: (source, f"simulation_{source}") for case_id, (_, source) in cases.items()},
        materials={case_id: material for case_id, (material, _) in cases.items()},
        design_kinds={case_id: kind or _DESIGN_KINDS[case_id] for case_id in cases},
        case_partition=None,
    )


@pytest.mark.parametrize(
    ("train_sources", "holdout", "match"),
    [
        (("a", "b", "c"), ("chickpea", "d"), None),
        (("a", "a", "c"), ("chickpea", "d"), "validation reuses a physical source"),
        (("a", "b", "c"), ("chickpea", "a"), r"material_ood\[0\] reuses a physical source"),
        (("a", "b", "c"), ("lentil", "d"), "reuses a training material"),
    ],
    ids=("disjoint", "id-roles-share-a-source", "holdout-shares-a-source", "holdout-reuses-train-material"),
)
def test_role_leakage_refuses_shared_sources_and_reused_training_materials(
    train_sources: tuple[str, str, str], holdout: tuple[str, str], match: str | None
) -> None:
    """Refuse leakage by physical source across every role and by material family for declared holdouts."""
    train = _binding("train_data", {case_id: ("lentil", source) for case_id, source in zip(("t1", "t2", "t3"), train_sources, strict=True)})
    material_ood: dict[splits.OodRole, list[tuple[identity.ManifestBinding, tuple[str, ...]]]] = {
        "material_ood": [(_binding("ood_data", {"o1": holdout}), ("o1",))]
    }
    if match is None:
        splits.reject_role_leakage(train, _TRAIN_PARTITION, material_ood)
        return
    with pytest.raises(ValueError, match=match):
        splits.reject_role_leakage(train, _TRAIN_PARTITION, material_ood)


def test_parameter_ood_cases_stay_out_of_id_roles_and_the_family_rule_binds_only_material_ood() -> None:
    """A parameter-OOD case of a Train material is admitted as parameter OOD, refused in Train and refused as material OOD."""
    cases = {"t1": ("lentil", "a"), "t2": ("lentil", "b"), "t3": ("lentil", "c"), "p1": ("lentil", "p")}
    train = _binding("train_data", cases)
    splits.reject_role_leakage(train, _TRAIN_PARTITION, {"parameter_ood": [(train, ("p1",))]})
    with pytest.raises(ValueError, match="material-OOD evaluation reuses a training material family"):
        splits.reject_role_leakage(train, _TRAIN_PARTITION, {"material_ood": [(train, ("p1",))]})
    with pytest.raises(ValueError, match="Train, validation or ID-test holds parameter-OOD cases"):
        splits.reject_role_leakage(train, {**_TRAIN_PARTITION, "train": ["t1", "p1"]}, {})


@pytest.mark.parametrize("task", ["steady_flow", "transient_drying"])
def test_material_and_parameter_ood_are_separate_parts_of_one_split(task: str) -> None:
    """Keep each OOD role its own parts and assignments; an empty role has none and is not an evaluated role."""
    parameter = ("lentil__case_0009__000000000009",)
    material = (("ood_a", ("chickpea__case_0001__00000000000a",)), ("ood_b", ("pea__case_0002__00000000000b",)))
    split = splits.admit_split(synthetic_split_payload(task, material_ood=material, parameter_ood=parameter))

    assert split.ood_roles == ("material_ood", "parameter_ood")
    assert [part.dataset_id for part in split.ood_parts("material_ood")] == ["ood_a", "ood_b"]
    assert [part.dataset_id for part in split.ood_parts("parameter_ood")] == ["train_data"]
    assert split.ood_parts("parameter_ood")[0].case_ids == parameter
    assert {role: [item["role"] for item in items] for role, items in split.evaluation_assignments.items()} == {
        "material_ood": ["material_ood", "material_ood"],
        "parameter_ood": ["parameter_ood"],
    }
    assert set(split.membership_digests()) == {"train", "validation", "id_test", "material_ood", "parameter_ood"}
    material_only = splits.admit_split(synthetic_split_payload(task, material_ood=material[:1]))
    assert material_only.ood_roles == ("material_ood",)
    assert material_only.ood_parts("parameter_ood") == ()


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
    payload = synthetic_split_payload(task, material_ood=())
    mutate(payload["dataset_identity"]["train"])
    with pytest.raises(ValueError, match=match):
        splits.admit_split(payload)


@pytest.mark.parametrize(
    ("role", "case_ids", "refused"),
    [
        ("validation", ["t2"], False),
        ("validation", ["t1"], True),
        ("material_ood", ["t4", "t5"], True),
    ],
    ids=("id-role-matches", "id-role-contradicts-partition", "material-ood-part-contradicts-ood-partition"),
)
def test_require_source_identity_binds_a_reopened_part_to_the_fixed_partition(role: splits.SplitRole, case_ids: list[str], refused: bool) -> None:
    """Refuse a reopened part whose saved membership differs from the current manifest's fixed partition."""
    partition = {**_TRAIN_PARTITION, "parameter_ood": [], "ood": ["t4"]}
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


_SUBSET_TRAIN = ("l1", "l2", "l3", "l4", "c1", "c2", "c3", "c4")


@pytest.mark.parametrize(
    ("train_subset", "case_ids", "refused"),
    [
        (None, list(_SUBSET_TRAIN), False),
        (2, ["l1", "l2", "c1", "c2"], False),
        (4, ["l1", "l2", "c1", "c2"], True),
        (None, ["l1", "l2", "c1", "c2"], True),
        (2, ["c1", "c2", "l1", "l2"], True),
    ],
    ids=("full", "subset-matches", "claimed-subset-larger-than-saved", "null-with-a-saved-subset", "subset-out-of-order"),
)
def test_a_reopened_train_part_must_be_exactly_its_saved_train_subset(train_subset: int | None, case_ids: list[str], refused: bool) -> None:
    """The saved Train membership equals the first ``train_subset`` Train cases per ID material, in order; null is the whole Train role."""
    cases = {case_id: ("lentil" if case_id.startswith("l") else "chickpea", f"s{case_id}") for case_id in _SUBSET_TRAIN}
    partition = {"train": list(_SUBSET_TRAIN), "validation": [], "id_test": [], "parameter_ood": [], "ood": []}
    binding = replace(_binding("train_data", cases, kind="train"), case_partition=partition)
    payload_identity = {"fingerprint": "f" * 64, "spatial_shape": [8, 8]}
    runtime = types.SimpleNamespace(manifest_binding=binding, payload_identity=payload_identity)
    expected = splits.split_dataset_identity(binding, payload_identity)
    if not refused:
        splits.require_source_identity(runtime, expected, role="train", case_ids=case_ids, train_subset=train_subset)
        return
    with pytest.raises(RuntimeError, match="contradicts the Dataset's fixed partition"):
        splits.require_source_identity(runtime, expected, role="train", case_ids=case_ids, train_subset=train_subset)


def test_split_admission_refuses_repeated_parts_and_an_altered_training_part() -> None:
    """Keep each material-OOD source distinct and a training-Dataset part exactly its declared cases of that role."""
    payload = synthetic_split_payload(
        "steady_flow", material_ood=(("ood_a", ("chickpea__case_0001__00000000000a",)), ("ood_b", ("pea__case_0002__00000000000b",)))
    )
    aligned = (payload["dataset_identity"]["material_ood"], payload["roles"]["material_ood"]["parts"])
    for parts in (*aligned, payload["evaluation_assignments"]["material_ood"]):
        parts[1] = copy.deepcopy(parts[0])
    with pytest.raises(ValueError, match="material_ood parts must come from distinct Datasets"):
        splits.admit_split(payload)
    with pytest.raises(ValueError, match="material_ood part of the training Dataset must hold exactly its declared material_ood cases"):
        synthetic_split_payload("steady_flow", material_ood=(("train_data", ("lentil__case_0001__000000000001",)),))
    # Parameter OOD comes only from the training Dataset; a split that drops its declared parameter-OOD cases is refused.
    with_parameter = synthetic_split_payload("steady_flow", parameter_ood=("lentil__case_0009__000000000009",))
    for section in (with_parameter["dataset_identity"], with_parameter["evaluation_assignments"]):
        section["parameter_ood"] = []
    with_parameter["roles"]["parameter_ood"]["parts"] = []
    with pytest.raises(ValueError, match="parameter_ood must be exactly the training Dataset's declared parameter-OOD cases"):
        splits.admit_split(with_parameter)


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
