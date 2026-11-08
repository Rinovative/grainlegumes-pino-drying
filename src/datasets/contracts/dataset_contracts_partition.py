"""Resolve and validate immutable Dataset case partition evidence.

Responsibilities:
  - Preserve the existing seeded material partition algorithm and rounding
  - Bind ordered Dataset ID/OOD case membership to explicit roles and seeds
  - Refuse training on a Dataset without a fixed partition or without Train cases, for every Dataset view

Design principles:
  - Physical Sources remain role-free; Dataset membership is immutable
  - No network seed influences fixed Dataset partitions

This module does NOT:
  - Generate physical cases or derive new sampling seeds
  - Expand transitions, fit scalers or choose model settings
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_PARTITION_KEYS = {"seed_input", "seed", "recipe", "material_roles", "train", "validation", "id_test", "ood"}
_ROLES = ("train", "validation", "id_test")
_MINIMUM_ID_CASES = 3


def _sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_case_partition(
    case_ids: Sequence[str],
    recipe: Mapping[str, Any],
    *,
    split_seed: int,
    material_by_case: Mapping[str, str],
) -> dict[str, tuple[str, ...]]:
    """Resolve one experiment-owned case partition before expanding transitions."""
    if not isinstance(recipe, Mapping) or recipe.get("kind") not in {"exact", "seeded_material_fraction"}:
        msg = "Experiment case partition must be exact or seeded_material_fraction."
        raise ValueError(msg)
    source = tuple(case_ids)
    if len(source) < _MINIMUM_ID_CASES or len(source) != len(set(source)):
        msg = "Experiment training Dataset needs at least three distinct cases."
        raise TypeError(msg)
    if (
        not isinstance(material_by_case, Mapping)
        or set(material_by_case) != set(source)
        or any(not isinstance(material, str) or not material for material in material_by_case.values())
    ):
        msg = "Experiment partition lacks exact source material evidence."
        raise ValueError(msg)
    if recipe["kind"] == "exact":
        if set(recipe) != {"kind", "train", "validation", "id_test"}:
            msg = "Exact experiment case partition has invalid keys."
            raise ValueError(msg)
        result: dict[str, tuple[str, ...]] = {}
        for role in _ROLES:
            value = recipe[role]
            if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
                msg = f"Exact experiment {role} cases must be a non-empty list."
                raise ValueError(msg)
            result[role] = tuple(value)
        combined = tuple(item for role in result.values() for item in role)
        if len(combined) != len(set(combined)) or set(combined) != set(source):
            msg = "Exact experiment partition must cover source cases once without overlap."
            raise ValueError(msg)
        return result
    if set(recipe) != {"kind", "train_fraction", "validation_fraction", "id_test_fraction"}:
        msg = "Seeded experiment partition has invalid keys."
        raise ValueError(msg)
    fractions = tuple(recipe[key] for key in ("train_fraction", "validation_fraction", "id_test_fraction"))
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0.0 < float(value) < 1.0
        for value in fractions
    ):
        msg = "Seeded experiment partition fractions must be finite values between zero and one."
        raise ValueError(msg)
    if not np.isclose(sum(float(value) for value in fractions), 1.0, atol=1.0e-12, rtol=0.0):
        msg = "Seeded experiment partition fractions must sum to one."
        raise ValueError(msg)
    material_groups: dict[str, list[str]] = {}
    for case_id in source:
        material_groups.setdefault(material_by_case[case_id], []).append(case_id)
    resolved: dict[str, list[str]] = {role: [] for role in _ROLES}
    for material, cases in material_groups.items():
        material_seed = int(_sha256({"split_seed": split_seed, "material_family": material})[:8], 16)
        shuffled = list(cases)
        random.Random(material_seed).shuffle(shuffled)  # noqa: S311 -- unchanged reproducible case partition
        train_count = int(len(cases) * float(fractions[0]))
        validation_count = int(len(cases) * float(fractions[1]))
        test_count = len(cases) - train_count - validation_count
        if min(train_count, validation_count, test_count) < 1:
            msg = f"Seeded partition leaves material {material!r} without a complete case in every role."
            raise ValueError(msg)
        resolved["train"].extend(shuffled[:train_count])
        resolved["validation"].extend(shuffled[train_count : train_count + validation_count])
        resolved["id_test"].extend(shuffled[train_count + validation_count :])
    return {role: tuple(cases) for role, cases in resolved.items()}


def build_fixed_partition(
    case_ids: Sequence[str],
    *,
    material_by_case: Mapping[str, str],
    material_roles: Mapping[str, str],
    recipe: Mapping[str, Any],
    seed_input: int,
    seed: int,
) -> dict[str, Any]:
    """Build ordered ID/OOD split evidence using the declared effective seed."""
    source = tuple(case_ids)
    if not isinstance(material_roles, Mapping):
        msg = "Partition material roles must map each material to id or ood."
        raise TypeError(msg)
    if (
        not isinstance(material_by_case, Mapping)
        or set(material_by_case) != set(source)
        or any(not isinstance(material, str) or not material for material in material_by_case.values())
    ):
        msg = "Experiment partition lacks exact source material evidence."
        raise ValueError(msg)
    materials = tuple(dict.fromkeys(material_by_case[case_id] for case_id in source))
    if set(material_roles) != set(materials) or any(not isinstance(value, str) or value not in {"id", "ood"} for value in material_roles.values()):
        msg = "Partition material roles must map each material to id or ood."
        raise ValueError(msg)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (seed_input, seed)):
        msg = "Partition seeds must be integers."
        raise ValueError(msg)
    id_cases = tuple(case_id for case_id in source if material_roles[material_by_case[case_id]] == "id")
    ood_cases = tuple(case_id for case_id in source if material_roles[material_by_case[case_id]] == "ood")
    resolved = (
        resolve_case_partition(id_cases, recipe, split_seed=seed, material_by_case={case_id: material_by_case[case_id] for case_id in id_cases})
        if id_cases
        else dict.fromkeys(_ROLES, ())
    )
    result: dict[str, Any] = {
        "seed_input": seed_input,
        "seed": seed,
        "recipe": dict(recipe),
        "material_roles": dict(material_roles),
        **{role: list(resolved[role]) for role in _ROLES},
        "ood": list(ood_cases),
    }
    validate_fixed_partition(result, case_ids=source, material_by_case=material_by_case)
    return result


def require_disjoint_roles(value: Mapping[str, Any]) -> None:
    """Require every case in at most one of Train, validation, ID-test and OOD, and at most once within it.

    Raises
    ------
    ValueError
        If a role is not a list of case IDs or a case is repeated within or across roles.

    """
    for key in (*_ROLES, "ood"):
        items = value[key]
        if not isinstance(items, list) or any(not isinstance(item, str) or not item for item in items):
            msg = f"Fixed partition {key} cases must be a list of case IDs."
            raise ValueError(msg)
    combined = [item for key in (*_ROLES, "ood") for item in value[key]]
    if len(combined) != len(set(combined)):
        msg = "Fixed partition roles must be disjoint and hold each case once."
        raise ValueError(msg)


def validate_fixed_partition(
    value: Mapping[str, Any],
    *,
    case_ids: Sequence[str],
    material_by_case: Mapping[str, str],
) -> dict[str, Any]:
    """Validate exact coverage, material ownership, and reproducible membership."""
    if not isinstance(value, Mapping) or set(value) != _PARTITION_KEYS:
        msg = "Fixed partition has unexpected fields."
        raise ValueError(msg)
    source = tuple(case_ids)
    if len(source) != len(set(source)):
        msg = "Fixed partition source cases must be unique."
        raise ValueError(msg)
    if (
        isinstance(value["seed_input"], bool)
        or not isinstance(value["seed_input"], int)
        or isinstance(value["seed"], bool)
        or not isinstance(value["seed"], int)
    ):
        msg = "Partition seeds must be integers."
        raise TypeError(msg)
    roles = value["material_roles"]
    if (
        not isinstance(roles, Mapping)
        or not isinstance(material_by_case, Mapping)
        or set(material_by_case) != set(source)
        or any(not isinstance(material, str) or not material for material in material_by_case.values())
    ):
        msg = "Partition lacks exact source material and role evidence."
        raise ValueError(msg)
    materials = tuple(dict.fromkeys(material_by_case[case_id] for case_id in source))
    if set(roles) != set(materials) or any(not isinstance(role, str) or role not in {"id", "ood"} for role in roles.values()):
        msg = "Partition material roles must map each material to id or ood."
        raise ValueError(msg)
    require_disjoint_roles(value)
    if {item for key in (*_ROLES, "ood") for item in value[key]} != set(source):
        msg = "Fixed partition must cover source cases once without overlap."
        raise ValueError(msg)
    expected_ood = [case_id for case_id in source if roles[material_by_case[case_id]] == "ood"]
    if value["ood"] != expected_ood:
        msg = "Fixed partition OOD cases must preserve source order and material roles."
        raise ValueError(msg)
    id_cases = tuple(case_id for case_id in source if roles[material_by_case[case_id]] == "id")
    expected_id = (
        resolve_case_partition(
            id_cases,
            value["recipe"],
            split_seed=value["seed"],
            material_by_case={case_id: material_by_case[case_id] for case_id in id_cases},
        )
        if id_cases
        else dict.fromkeys(_ROLES, ())
    )
    if any(tuple(value[role]) != expected_id[role] for role in _ROLES):
        msg = "Fixed partition ID case membership does not match its recipe and seed."
        raise ValueError(msg)
    return dict(value)


def validate_training_partition(
    value: Mapping[str, Any] | None,
    *,
    dataset_id: str,
    case_ids: Sequence[str],
    material_by_case: Mapping[str, str],
) -> dict[str, Any]:
    """Require the fixed partition of a training Dataset to exist and hold Train cases, then validate it.

    Evaluation-only Datasets may hold every case as OOD; training on one is refused for every Dataset view.
    """
    if value is not None:
        fixed = validate_fixed_partition(value, case_ids=case_ids, material_by_case=material_by_case)
        if fixed["train"]:
            return fixed
    msg = f"Training Dataset {dataset_id!r} has no fixed case partition with Train cases; build it from a Dataset plan."
    raise ValueError(msg)


def partition_digest(value: Mapping[str, Any]) -> str:
    """Return a canonical digest for JSON-compatible fixed partition evidence."""
    return _sha256(value)
