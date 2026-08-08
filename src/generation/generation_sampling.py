"""
===============================================================================
generation_sampling.py
===============================================================================
Generate deterministic blockwise design-of-experiments values.
Responsibilities:
  - Build independent Latin-hypercube or Sobol designs for four physical blocks
  - Apply typed registry transforms, coupled selections, and parameter OOD tails
  - Persist block seeds, row permutations, row indices, and selection evidence
Design principles:
  - Each block owns an independent label-derived stream and row permutation
  - Coupled sets, family identity, and seeds are metadata rather than dimensions
  - Case values are independent of process ordering, concurrency, and resume
This module does NOT:
  - Define material ranges, invent unresolved values, or generate spatial fields
  - Evaluate arbitrary expressions or select a simulation profile
===============================================================================
"""

from __future__ import annotations

import copy
import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue] -- SciPy typing omits the public export
from scipy.stats import qmc

from src import common

from . import generation_materials as materials
from . import generation_registry as registry_service

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .generation_config import GenerationConfig

LHS_MAXIMIN_CANDIDATES = 50
SOBOL_SKIP = 1000
SOBOL_LEAP = 200
_NEAREST_NEIGHBOR_COUNT = 2


@dataclass(frozen=True, slots=True)
class CaseSample:
    """One resolved case-level scientific sample and complete DOE evidence."""

    values: dict[str, Any]
    units: dict[str, str]
    coupled_selections: dict[str, str]
    block_provenance: dict[str, dict[str, Any]]
    ood_provenance: dict[str, Any]


def _seed(seed_base: int, *labels: str) -> int:
    """Delegate semantic seed derivation to the scientific configuration owner."""
    from .generation_config import derive_seed  # noqa: PLC0415 -- breaks the config/sampling import cycle

    return derive_seed(seed_base, *labels)


def _lhs_design(*, count: int, dimensions: int, seed: int) -> np.ndarray:
    """Select the most separated of the maintained deterministic LHS candidates."""
    best: np.ndarray | None = None
    best_separation = -math.inf
    for iteration in range(LHS_MAXIMIN_CANDIDATES):
        candidate_rng = np.random.default_rng(np.random.SeedSequence([seed, iteration]))
        candidate = qmc.LatinHypercube(d=dimensions, scramble=True, rng=candidate_rng).random(count)
        if count < _NEAREST_NEIGHBOR_COUNT:
            separation = math.inf
        else:
            distances, _indices = cKDTree(candidate).query(candidate, k=_NEAREST_NEIGHBOR_COUNT)
            separation = float(np.min(distances[:, 1]))
        if separation > best_separation:
            best = candidate
            best_separation = separation
    if best is None:
        msg = "Latin-hypercube generation produced no candidate design."
        raise RuntimeError(msg)
    return best


def _sobol_design(*, count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return the maintained scrambled, skipped, and leaped Sobol design."""
    sampler = qmc.Sobol(d=dimensions, scramble=True, rng=seed)
    sampler.fast_forward(SOBOL_SKIP)
    design: np.ndarray = np.empty((count, dimensions), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for row in range(count):
            design[row] = sampler.random(1)[0]
            if row + 1 < count:
                sampler.fast_forward(SOBOL_LEAP)
    return design


def unit_design(method: str, *, count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return one deterministic unit design for an authoritative physical block."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        msg = "Design count must be a non-negative integer."
        raise ValueError(msg)
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 0:
        msg = "Design dimensions must be a non-negative integer."
        raise ValueError(msg)
    if count == 0 or dimensions == 0:
        return np.empty((count, dimensions), dtype=np.float64)
    if method == "lhs":
        return _lhs_design(count=count, dimensions=dimensions, seed=seed)
    if method == "sobol":
        return _sobol_design(count=count, dimensions=dimensions, seed=seed)
    msg = f"Unsupported blockwise sampling method: {method!r}."
    raise ValueError(msg)


def _array_sha256(value: np.ndarray) -> str:
    """Return a stable shape-, dtype-, and byte-bound array digest."""
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(length) for length in contiguous.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def build_sampling_plan(*, registry: Mapping[str, Mapping[str, Any]], case_count: int, seed_base: int, method: str) -> dict[str, dict[str, Any]]:
    """Build serializable independent design and permutation provenance."""
    plan: dict[str, dict[str, Any]] = {}
    dimensions = materials.sampling_block_dimensions(registry)
    for block, parameters in materials.SAMPLING_BLOCKS.items():
        dimension = dimensions[block]
        design_seed = _seed(seed_base, "sampling_block", block, "design")
        permutation_seed = _seed(seed_base, "sampling_block", block, "permutation")
        permutation = np.random.default_rng(permutation_seed).permutation(case_count).astype(np.int64)
        design = unit_design(method, count=case_count, dimensions=dimension, seed=design_seed)
        plan[block] = {
            "label": block,
            "parameters": list(parameters),
            "effective_dimension": dimension,
            "design_seed": design_seed,
            "design_sha256": _array_sha256(design),
            "permutation_seed": permutation_seed,
            "permutation": permutation.tolist(),
            "permutation_sha256": _array_sha256(permutation),
        }
    return plan


def _interval_value(entry: Mapping[str, Any], coordinate: float, *, use_ood: bool) -> float:
    """Map one unit coordinate through a registry-owned physical transform."""
    bounds = entry["ood"] if use_ood else entry
    lower = float(bounds["lower"])
    upper = float(bounds["upper"])
    transform = str(entry["transform"])
    if transform == "linear":
        return lower + coordinate * (upper - lower)
    if transform == "log":
        return math.exp(math.log(lower) + coordinate * (math.log(upper) - math.log(lower)))
    if transform == "logit":
        lower_t = math.log(lower / (1.0 - lower))
        upper_t = math.log(upper / (1.0 - upper))
        transformed = lower_t + coordinate * (upper_t - lower_t)
        return 1.0 / (1.0 + math.exp(-transformed))
    if transform == "phase":
        return (lower + coordinate * (upper - lower)) % (2.0 * math.pi)
    msg = f"Unsupported interval transform {transform!r}."
    raise ValueError(msg)


def _integer_value(entry: Mapping[str, Any], coordinate: float, *, use_ood: bool) -> int:
    """Map one unit coordinate onto an inclusive integer interval."""
    bounds = entry["ood"] if use_ood else entry
    lower = int(bounds["lower"])
    upper = int(bounds["upper"])
    width = upper - lower + 1
    return lower + min(math.floor(coordinate * width), width - 1)


def _simplex_value(entry: Mapping[str, Any], coordinates: np.ndarray, *, ood_index: int | None) -> dict[str, float]:
    """Map n-1 unit coordinates to one complete n-component simplex vector."""
    components = tuple(entry["components"])
    boundaries = np.concatenate(([0.0], np.sort(coordinates), [1.0]))
    id_weights = np.diff(boundaries)
    if ood_index is None:
        return {name: float(id_weights[index]) for index, name in enumerate(components)}
    values = entry.get("ood_values", [])
    if not values:
        msg = "Selected simplex OOD unit has no configured OOD vector."
        raise ValueError(msg)
    selected = copy.deepcopy(values[ood_index % len(values)])
    ood_weights = np.asarray([selected[name] for name in components], dtype=np.float64)
    gap = float(np.linalg.norm(ood_weights - id_weights))
    if not math.isfinite(gap) or gap <= 0.0:
        msg = "Selected simplex OOD vector has no simplex-coordinate separation from its ID block row."
        raise ValueError(msg)
    return selected


def _select_coupled(
    name: str,
    entry: Mapping[str, Any],
    *,
    seed_base: int,
    case_index: int,
    use_ood: bool,
) -> dict[str, Any]:
    """Select one complete coupled record from a label-derived stream."""
    kind = str(entry["kind"])
    primary = "pairs" if kind == "paired_parameter_set" else "sets"
    ood_key = "ood_pairs" if kind == "paired_parameter_set" else "ood_sets"
    candidates = entry.get(ood_key, []) if use_ood else entry[primary]
    if not candidates:
        msg = f"Selected coupled unit {name!r} has no {'OOD' if use_ood else 'ID'} records."
        raise ValueError(msg)
    rng = np.random.default_rng(_seed(seed_base, "coupled_selection", name, str(case_index), "ood" if use_ood else "id"))
    return copy.deepcopy(candidates[int(rng.integers(0, len(candidates)))])


def _ood_key(entry: Mapping[str, Any]) -> str | None:
    """Return the configured OOD inventory key for one typed entry."""
    return {
        "interval": "ood",
        "integer": "ood",
        "categorical": "ood_choices",
        "simplex": "ood_values",
        "parameter_set": "ood_sets",
        "paired_parameter_set": "ood_pairs",
    }.get(str(entry["kind"]))


def _ood_selections(
    registry: Mapping[str, Mapping[str, Any]],
    assignment: Mapping[str, Any],
    *,
    balance_parameters: bool,
    seed_base: int,
    case_index: int,
) -> tuple[str, ...]:
    """Select configured OOD units from exactly the assigned physical group."""
    group = assignment["ood_group"]
    if group is None:
        return ()
    candidates = [
        name for name, entry in registry.items() if entry.get("ood_group") == group and (key := _ood_key(entry)) is not None and bool(entry.get(key))
    ]
    if not candidates:
        msg = f"Parameter-OOD group {group!r} has no configured disjoint OOD units."
        raise ValueError(msg)
    count = int(assignment["ood_units_per_case"])
    if count > len(candidates):
        msg = f"Parameter-OOD case requests {count} units from only {len(candidates)} configured candidates in {group!r}."
        raise ValueError(msg)
    if balance_parameters:
        offset = (int(assignment["regime_index"]) // len(materials.OOD_GROUPS)) * count
        return tuple(candidates[(offset + index) % len(candidates)] for index in range(count))
    rng = np.random.default_rng(_seed(seed_base, "ood_selection", group, str(case_index)))
    indices = rng.choice(len(candidates), size=count, replace=False)
    return tuple(candidates[int(index)] for index in indices)


def _non_numerical_values(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    selected_ood: frozenset[str],
    seed_base: int,
    case_index: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve fixed, categorical, and complete coupled registry values."""
    values: dict[str, Any] = {}
    selections: dict[str, str] = {}
    for name, entry in registry.items():
        kind = entry["kind"]
        if kind == "fixed":
            values[name] = float(entry["value"])
        elif kind == "categorical":
            choices = entry.get("ood_choices", []) if name in selected_ood else entry["choices"]
            if not choices:
                msg = f"Categorical parameter {name!r} has no selectable values."
                raise ValueError(msg)
            rng = np.random.default_rng(_seed(seed_base, "categorical_selection", name, str(case_index)))
            values[name] = copy.deepcopy(choices[int(rng.integers(0, len(choices)))])
        elif kind in {"parameter_set", "paired_parameter_set"}:
            record = _select_coupled(name, entry, seed_base=seed_base, case_index=case_index, use_ood=name in selected_ood)
            selections[name] = str(record["id"])
            overlap = set(values).intersection(record["values"])
            if overlap:
                msg = f"Coupled parameter set {name!r} conflicts with independently resolved values {sorted(overlap)}."
                raise ValueError(msg)
            values.update(record["values"])
    return values, selections


def _block_values(
    registry: Mapping[str, Mapping[str, Any]],
    block: str,
    row: np.ndarray,
    *,
    selected_ood: frozenset[str],
    seed_base: int,
    case_index: int,
) -> dict[str, Any]:
    """Map one physical block row through its typed family registry."""
    values: dict[str, Any] = {}
    column = 0
    for name in materials.SAMPLING_BLOCKS[block]:
        entry = registry[name]
        dimension = registry_service.effective_dimension(entry)
        coordinates = row[column : column + dimension]
        column += dimension
        if entry["kind"] == "interval":
            values[name] = _interval_value(entry, float(coordinates[0]), use_ood=name in selected_ood)
        elif entry["kind"] == "integer":
            values[name] = _integer_value(entry, float(coordinates[0]), use_ood=name in selected_ood)
        elif entry["kind"] == "simplex":
            ood_index = None
            if name in selected_ood:
                count = len(entry.get("ood_values", []))
                ood_index = 0 if count == 0 else _seed(seed_base, "simplex_ood", name, str(case_index)) % count
            values[name] = _simplex_value(entry, coordinates, ood_index=ood_index)
        elif dimension != 0:
            msg = f"Registry parameter {name!r} owns an unsupported numerical dimension."
            raise RuntimeError(msg)
    expected = materials.sampling_block_dimensions(registry)[block]
    if column != expected:
        msg = f"Sampling block {block!r} consumed {column} coordinates; expected {expected}."
        raise RuntimeError(msg)
    return values


def _units(registry: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Return stable scalar and coupled-component units."""
    units: dict[str, str] = {}
    for name, entry in registry.items():
        if entry["kind"] in {"parameter_set", "paired_parameter_set"}:
            units.update(entry["units"])
        else:
            units[name] = str(entry["unit"])
    return units


def sample_case(config: GenerationConfig, case_index: int) -> CaseSample:
    """Resolve one deterministic case from independent physical block designs."""
    config.case_id(case_index)
    assignment = config.case_assignment(case_index)
    if config.seed_base is None:
        msg = f"Batch {config.batch_name!r} has no executable sampling seed."
        raise ValueError(msg)
    family = assignment["material_family"]
    family_contract = config.scientific_values["material"]
    registry = family_contract["parameter_registry"]
    policy = config.scientific_values["parameter_ood"]
    selected = frozenset(
        _ood_selections(
            registry,
            assignment,
            balance_parameters=bool(policy["balance_parameters"]),
            seed_base=config.seed_base,
            case_index=case_index,
        )
    )
    values, coupled_selections = _non_numerical_values(
        registry,
        selected_ood=selected,
        seed_base=config.seed_base,
        case_index=case_index,
    )
    position = config.case_indices.index(case_index)
    block_provenance: dict[str, dict[str, Any]] = {}
    for block, plan in config.scientific_values["sampling"]["blocks"].items():
        row_index = int(plan["permutation"][position])
        design = unit_design(
            config.scientific_values["sampling"]["method"],
            count=len(config.case_indices),
            dimensions=int(plan["effective_dimension"]),
            seed=int(plan["design_seed"]),
        )
        if _array_sha256(design) != plan["design_sha256"]:
            msg = f"Sampling design digest changed for block {block!r}."
            raise RuntimeError(msg)
        numerical_block_parameters = {name for name in materials.SAMPLING_BLOCKS[block] if registry_service.effective_dimension(registry[name]) > 0}
        overlap = set(values).intersection(numerical_block_parameters)
        if overlap:
            msg = f"Block {block!r} conflicts with coupled or fixed values {sorted(overlap)}."
            raise ValueError(msg)
        values.update(
            _block_values(
                registry,
                block,
                design[row_index],
                selected_ood=selected,
                seed_base=config.seed_base,
                case_index=case_index,
            )
        )
        block_provenance[block] = {
            "design_seed": plan["design_seed"],
            "design_sha256": plan["design_sha256"],
            "permutation_seed": plan["permutation_seed"],
            "permutation_sha256": plan["permutation_sha256"],
            "case_position": position,
            "block_row_index": row_index,
        }
    simplex = values.pop("schedule.component_weights")
    if not isinstance(simplex, dict):
        msg = "schedule.component_weights did not resolve to a complete simplex mapping."
        raise TypeError(msg)
    values["schedule.component_weights"] = simplex
    values = registry_service.resolve_derived_values(registry, values, defer_missing=True)
    lower = family_contract["initial_moisture_bounds"]["lower"]
    upper = family_contract["initial_moisture_bounds"]["upper"]
    mean = float(values["initial_moisture.mean_db"])
    amplitude = float(values["initial_moisture.amplitude_db"])
    if lower is None or upper is None:
        msg = f"Executable material family {family!r} has unresolved initial-moisture bounds."
        raise ValueError(msg)
    maximum_amplitude = min(mean - float(lower), float(upper) - mean)
    if amplitude < 0 or amplitude > maximum_amplitude:
        message = f"Sampled initial-moisture amplitude {amplitude} exceeds the no-clipping bound {maximum_amplitude} for family {family!r}."
        raise ValueError(message)
    return CaseSample(
        values=values,
        units=_units(registry),
        coupled_selections=coupled_selections,
        block_provenance=block_provenance,
        ood_provenance={
            "group": assignment["ood_group"],
            "selected_units": sorted(selected),
            "units_per_case": assignment["ood_units_per_case"],
        },
    )


def sample_all_cases(config: GenerationConfig) -> dict[int, CaseSample]:
    """Return all configured samples in canonical case-index order."""
    return {case_index: sample_case(config, case_index) for case_index in config.case_indices}


def sampling_plan_sha256(config: GenerationConfig) -> str:
    """Return the deterministic identity of all persisted block plans."""
    return common.serialization.canonical_json_sha256(config.scientific_values["sampling"]["blocks"])
