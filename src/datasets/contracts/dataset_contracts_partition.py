"""Resolve and validate immutable Dataset case partition evidence.

Responsibilities:
  - Derive each case's role from its material role and its Generation design kind
  - Bind ordered Train, validation, ID-test, parameter-OOD and material-OOD membership to explicit material roles
  - Select the Train subset of a run: the first N Train cases per ID material in stored order
  - Refuse training on a Dataset without a fixed partition or without Train cases, for every Dataset view

Design principles:
  - Physical Sources remain role-free; Dataset membership is immutable
  - Roles follow the fixed designs, never a seed: Train keeps the Sobol order of its design
  - An OOD material contributes only its test design; a parameter-OOD design belongs to an ID material

This module does NOT:
  - Generate physical cases or derive sampling seeds
  - Expand transitions, fit scalers or choose model settings
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

CASE_ROLES: Final = ("train", "validation", "id_test", "parameter_ood", "ood")
# The partition role each OOD Evaluation role reads: held-out materials, and ID materials outside the parameter support.
OOD_PARTITION_ROLES: Final = {"material_ood": "ood", "parameter_ood": "parameter_ood"}
_PARTITION_KEYS: Final = frozenset({"material_roles", *CASE_ROLES})
_ID_ROLE_BY_DESIGN_KIND: Final = {"train": "train", "validation": "validation", "test": "id_test", "parameter_ood": "parameter_ood"}


def _sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case_evidence(source: Sequence[str], evidence: object, *, label: str) -> Mapping[str, str]:
    """Require one non-empty text value per source case."""
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != set(source)
        or any(not isinstance(value, str) or not value for value in evidence.values())
    ):
        msg = f"Partition lacks exact source {label} evidence."
        raise ValueError(msg)
    return evidence


def build_fixed_partition(
    case_ids: Sequence[str],
    *,
    material_by_case: Mapping[str, str],
    design_kind_by_case: Mapping[str, str],
    material_roles: Mapping[str, str],
) -> dict[str, Any]:
    """Build ordered role membership from each case's material role and design kind, in source order.

    An ID material's train, validation, test and parameter-OOD designs become ``train``, ``validation``,
    ``id_test`` and ``parameter_ood``; an OOD material's test design becomes ``ood``.

    Raises
    ------
    ValueError
        If the cases are not unique, material roles do not map every material to id or ood, or a design kind does
        not fit its material role.

    """
    source = tuple(case_ids)
    if len(source) != len(set(source)):
        msg = "Fixed partition source cases must be unique."
        raise ValueError(msg)
    materials = _case_evidence(source, material_by_case, label="material")
    kinds = _case_evidence(source, design_kind_by_case, label="design kind")
    if (
        not isinstance(material_roles, Mapping)
        or set(material_roles) != set(materials.values())
        or any(role not in {"id", "ood"} for role in material_roles.values())
    ):
        msg = "Partition material roles must map each material to id or ood."
        raise ValueError(msg)
    result: dict[str, Any] = {"material_roles": dict(material_roles), **{role: [] for role in CASE_ROLES}}
    for case_id in source:
        material_role = material_roles[materials[case_id]]
        role = ("ood" if kinds[case_id] == "test" else None) if material_role == "ood" else _ID_ROLE_BY_DESIGN_KIND.get(kinds[case_id])
        if role is None:
            msg = f"Case {case_id!r} has design kind {kinds[case_id]!r}, which a {material_role} material cannot hold."
            raise ValueError(msg)
        result[role].append(case_id)
    return result


def evaluation_cases(partition: Mapping[str, Any], role: str) -> tuple[str, ...]:
    """Return the Dataset-declared cases of one OOD Evaluation role (``material_ood`` or ``parameter_ood``) in stored order."""
    return tuple(partition[OOD_PARTITION_ROLES[role]])


def train_subset_cases(partition: Mapping[str, Any], *, material_by_case: Mapping[str, str], train_subset: int | None) -> tuple[str, ...]:
    """Return the run's Train cases: all of them, or the first ``train_subset`` per ID material in stored Train order.

    Raises
    ------
    ValueError
        If an ID material holds fewer Train cases than ``train_subset``.

    """
    train = tuple(partition["train"])
    if train_subset is None:
        return train
    by_material: dict[str, list[str]] = {}
    for case_id in train:
        by_material.setdefault(material_by_case[case_id], []).append(case_id)
    short = sorted(material for material, cases in by_material.items() if len(cases) < train_subset)
    if short:
        msg = f"data.train_subset {train_subset} exceeds the Train cases of {short}."
        raise ValueError(msg)
    selected = {case_id for cases in by_material.values() for case_id in cases[:train_subset]}
    return tuple(case_id for case_id in train if case_id in selected)


def require_disjoint_roles(value: Mapping[str, Any]) -> None:
    """Require every case in at most one partition role, and at most once within it.

    Raises
    ------
    ValueError
        If a role is not a list of case IDs or a case is repeated within or across roles.

    """
    for key in CASE_ROLES:
        items = value[key]
        if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
            msg = f"Fixed partition {key} cases must be a list of case IDs."
            raise ValueError(msg)
    combined = [item for key in CASE_ROLES for item in value[key]]
    if len(combined) != len(set(combined)):
        msg = "Fixed partition roles must be disjoint and hold each case once."
        raise ValueError(msg)


def validate_fixed_partition(
    value: Mapping[str, Any],
    *,
    case_ids: Sequence[str],
    material_by_case: Mapping[str, str],
    design_kind_by_case: Mapping[str, str],
) -> dict[str, Any]:
    """Validate that a stored partition is exactly the one its material roles and the cases' design kinds determine."""
    if not isinstance(value, Mapping) or set(value) != _PARTITION_KEYS:
        msg = "Fixed partition has unexpected fields."
        raise ValueError(msg)
    require_disjoint_roles(value)
    expected = build_fixed_partition(
        case_ids, material_by_case=material_by_case, design_kind_by_case=design_kind_by_case, material_roles=value["material_roles"]
    )
    if dict(value) != expected:
        msg = "Fixed partition membership does not match its material roles and the design kinds of its cases."
        raise ValueError(msg)
    return expected


def validate_training_partition(
    value: Mapping[str, Any] | None,
    *,
    dataset_id: str,
    case_ids: Sequence[str],
    material_by_case: Mapping[str, str],
    design_kind_by_case: Mapping[str, str],
) -> dict[str, Any]:
    """Require the fixed partition of a training Dataset to exist and hold Train cases, then validate it.

    Evaluation-only Datasets may hold every case as OOD; training on one is refused for every Dataset view.
    """
    if value is not None:
        fixed = validate_fixed_partition(value, case_ids=case_ids, material_by_case=material_by_case, design_kind_by_case=design_kind_by_case)
        if fixed["train"]:
            return fixed
    msg = f"Training Dataset {dataset_id!r} has no fixed case partition with Train cases; build it from a Dataset plan."
    raise ValueError(msg)


def partition_digest(value: Mapping[str, Any]) -> str:
    """Return a canonical digest for JSON-compatible fixed partition evidence."""
    return _sha256(value)
