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
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue] -- SciPy typing omits the public export
from scipy.stats import beta as beta_distribution
from scipy.stats import qmc

from src import common

from . import generation_materials as materials
from . import generation_registry as registry_service
from . import generation_seeding as seeding

if TYPE_CHECKING:
    from collections.abc import Mapping

LHS_MAXIMIN_CANDIDATES = 50
SOBOL_SKIP = 1000
SOBOL_LEAP = 200
_NEAREST_NEIGHBOR_COUNT = 2


class _SamplingConfig(Protocol):
    """Describe the resolved batch surface consumed by case sampling."""

    @property
    def scientific_values(self) -> dict[str, Any]:
        """Return the resolved scientific configuration."""
        ...

    @property
    def case_indices(self) -> tuple[int, ...]:
        """Return batch case indices in canonical order."""
        ...

    @property
    def seed_base(self) -> int | None:
        """Return the resolved batch seed when sampling is executable."""
        ...

    @property
    def batch_name(self) -> str:
        """Return the human-readable batch name."""
        ...

    def case_id(self, case_index: int) -> str:
        """Return the canonical case identifier for one batch member."""
        ...

    def case_assignment(self, case_index: int) -> dict[str, Any]:
        """Return one isolated resolved case assignment."""
        ...


@dataclass(frozen=True, slots=True)
class CaseSample:
    """One resolved case-level scientific sample and complete DOE evidence."""

    values: dict[str, Any]
    units: dict[str, str]
    coupled_selections: dict[str, str]
    block_provenance: dict[str, dict[str, Any]]
    conditional_supports: dict[str, dict[str, Any]]
    ood_provenance: dict[str, Any]


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


def build_sampling_plan(
    *,
    registry: Mapping[str, Mapping[str, Any]],
    case_count: int,
    seed_base: int,
    method: str,
    blocks: tuple[str, ...],
    block_parameters: Mapping[str, tuple[str, ...]],
    block_seed_bases: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build serializable designs for only the profile-active blocks."""
    dimensions = materials.sampling_block_dimensions(
        registry,
        blocks=blocks,
        block_parameters=block_parameters,
    )
    overrides = {} if block_seed_bases is None else dict(block_seed_bases)
    unknown_overrides = sorted(set(overrides).difference(blocks))
    if unknown_overrides:
        message = f"Sampling seed overrides target inactive blocks {unknown_overrides}."
        raise ValueError(message)
    plan: dict[str, dict[str, Any]] = {}
    for block in blocks:
        parameters = block_parameters[block]
        dimension = dimensions[block]
        paired = block in overrides
        block_seed_base = overrides.get(block, seed_base)
        namespace = "paired_equivalence_block" if paired else "sampling_block"
        design_seed = seeding.derive_seed(block_seed_base, namespace, block, "design")
        permutation_seed = seeding.derive_seed(
            block_seed_base,
            namespace,
            block,
            "permutation",
        )
        permutation = np.random.default_rng(permutation_seed).permutation(case_count).astype(np.int64)
        design = unit_design(
            method,
            count=case_count,
            dimensions=dimension,
            seed=design_seed,
        )
        plan[block] = {
            "label": block,
            "parameters": list(parameters),
            "effective_dimension": dimension,
            "seed_origin": ("paired_equivalence" if paired else "profile_batch"),
            "design_seed": design_seed,
            "design_sha256": _array_sha256(design),
            "permutation_seed": permutation_seed,
            "permutation": permutation.tolist(),
            "permutation_sha256": _array_sha256(permutation),
        }
    return plan


def _interval_value(entry: Mapping[str, Any], coordinate: float, *, bounds: Mapping[str, Any]) -> float:
    """Map one unit coordinate through a selected registry-owned interval."""
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


def _integer_value(coordinate: float, *, bounds: Mapping[str, Any]) -> int:
    """Map one unit coordinate onto a selected inclusive integer interval."""
    lower = int(bounds["lower"])
    upper = int(bounds["upper"])
    width = upper - lower + 1
    return lower + min(math.floor(coordinate * width), width - 1)


def _selected_interval(
    name: str,
    entry: Mapping[str, Any],
    *,
    use_ood: bool,
    seed_base: int,
    case_index: int,
) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
    """Select one deterministic scalar OOD tail and report its separation."""
    if not use_ood:
        return entry, None
    intervals = entry.get("ood", [])
    if not intervals:
        msg = f"Selected scalar OOD unit {name!r} has no configured tails."
        raise ValueError(msg)
    index = seeding.derive_seed(seed_base, "scalar_ood_tail", name, str(case_index)) % len(intervals)
    bounds = intervals[index]
    transform = str(entry.get("transform", "linear"))
    id_lower = registry_service.transformed_coordinate(float(entry["lower"]), entry)
    id_upper = registry_service.transformed_coordinate(float(entry["upper"]), entry)
    tail_lower = registry_service.transformed_coordinate(float(bounds["lower"]), entry)
    tail_upper = registry_service.transformed_coordinate(float(bounds["upper"]), entry)
    gap = id_lower - tail_upper if float(bounds["upper"]) < float(entry["lower"]) else tail_lower - id_upper
    id_width = id_upper - id_lower
    return bounds, {
        "selection_kind": "scalar_interval",
        "tail_id": f"{name}__tail_{index + 1}",
        "tail_index": index,
        "transform": transform,
        "id_interval": [float(entry["lower"]), float(entry["upper"])],
        "ood_interval": [float(bounds["lower"]), float(bounds["upper"])],
        "transformed_gap": gap,
        "transform_space_distance": gap,
        "transformed_gap_fraction": gap / id_width,
        "transformed_width_fraction": (tail_upper - tail_lower) / id_width,
        "hard_boundary": bool(bounds.get("hard_boundary", False)),
    }


def _remap_interval(
    entry: Mapping[str, Any],
    coordinate: float,
    *,
    lower: float,
    upper: float,
    label: str,
) -> float:
    """Map one existing DOE coordinate through a non-empty conditional interval."""
    if lower > upper:
        msg = f"Conditional support for {label!r} is empty: lower={lower}, upper={upper}."
        raise ValueError(msg)
    return _interval_value(entry, coordinate, bounds={"lower": lower, "upper": upper})


def _condition_operation_values(
    values: dict[str, Any],
    *,
    registry: Mapping[str, Mapping[str, Any]],
    coordinates: Mapping[str, float],
    bounds: Mapping[str, Mapping[str, Any]],
    fixed: Mapping[str, Any],
) -> None:
    """Enforce joint schedule support without clipping generated schedules."""
    for base_name, amplitude_name, minimum_name, maximum_name in (
        ("T_in_base", "T_in_amp", "T_in_min", "T_in_max"),
        ("omega_in_base", "omega_in_amp", "omega_min", "omega_max"),
    ):
        base_bounds = bounds[base_name]
        amplitude_bounds = bounds[amplitude_name]
        amplitude_lower = float(amplitude_bounds["lower"])
        operational_lower = float(fixed[minimum_name])
        operational_upper = float(fixed[maximum_name])
        base_lower = max(float(base_bounds["lower"]), operational_lower + amplitude_lower)
        base_upper = min(float(base_bounds["upper"]), operational_upper - amplitude_lower)
        base = _remap_interval(
            registry[base_name],
            coordinates[base_name],
            lower=base_lower,
            upper=base_upper,
            label=base_name,
        )
        amplitude_upper = min(
            float(amplitude_bounds["upper"]),
            base - operational_lower,
            operational_upper - base,
        )
        amplitude = _remap_interval(
            registry[amplitude_name],
            coordinates[amplitude_name],
            lower=amplitude_lower,
            upper=amplitude_upper,
            label=amplitude_name,
        )
        values[base_name] = base
        values[amplitude_name] = amplitude

    duration_name = "schedule.event_duration_rel"
    width_name = "schedule.event_width_rel"
    duration_bounds = bounds[duration_name]
    width_bounds = bounds[width_name]
    width_lower = float(width_bounds["lower"])
    duration_lower = max(float(duration_bounds["lower"]), 2.0 * width_lower)
    duration = _remap_interval(
        registry[duration_name],
        coordinates[duration_name],
        lower=duration_lower,
        upper=float(duration_bounds["upper"]),
        label=duration_name,
    )
    width = _remap_interval(
        registry[width_name],
        coordinates[width_name],
        lower=width_lower,
        upper=min(float(width_bounds["upper"]), 0.5 * duration),
        label=width_name,
    )
    values[duration_name] = duration
    values[width_name] = width


def _condition_initial_moisture_values(
    values: dict[str, Any],
    *,
    registry: Mapping[str, Mapping[str, Any]],
    coordinates: Mapping[str, float],
    bounds: Mapping[str, Mapping[str, Any]],
    family_bounds: Mapping[str, float],
    field_constraint: Mapping[str, Any],
    natural_support_departure_allowed: bool,
) -> None:
    """Enforce the supplied natural or high-tail field guard without clipping."""
    mean_name = "initial_moisture.mean_db"
    amplitude_name = "initial_moisture.amplitude_db"
    mean_bounds = bounds[mean_name]
    amplitude_bounds = bounds[amplitude_name]
    amplitude_lower = float(amplitude_bounds["lower"])
    field_upper: float | None
    if natural_support_departure_allowed:
        field_lower = float(field_constraint["minimum_db"])
        field_upper = None
        mean_upper = float(mean_bounds["upper"])
    else:
        field_lower = float(family_bounds["lower"])
        field_upper = float(family_bounds["upper"])
        mean_upper = min(float(mean_bounds["upper"]), field_upper - amplitude_lower)
    mean = _remap_interval(
        registry[mean_name],
        coordinates[mean_name],
        lower=max(float(mean_bounds["lower"]), field_lower + amplitude_lower),
        upper=mean_upper,
        label=mean_name,
    )
    if natural_support_departure_allowed:
        amplitude_upper = min(float(amplitude_bounds["upper"]), mean - field_lower)
    else:
        if field_upper is None:
            message = "Natural initial-moisture support requires a finite upper bound."
            raise RuntimeError(message)
        amplitude_upper = min(float(amplitude_bounds["upper"]), mean - field_lower, field_upper - mean)
    amplitude = _remap_interval(
        registry[amplitude_name],
        coordinates[amplitude_name],
        lower=amplitude_lower,
        upper=amplitude_upper,
        label=amplitude_name,
    )
    values[mean_name] = mean
    values[amplitude_name] = amplitude


def _simplex_value(
    entry: Mapping[str, Any],
    coordinates: np.ndarray,
    *,
    ood_index: int | None,
    seed: int,
) -> tuple[dict[str, float], float | None]:
    """Map two coordinates through the binding truncated Dirichlet simplex."""
    components = tuple(entry["components"])
    alpha = np.asarray(entry["alpha"], dtype=np.float64)
    clipped = np.clip(np.asarray(coordinates, dtype=np.float64), np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    first = float(beta_distribution.ppf(clipped[0], alpha[0], np.sum(alpha[1:])))
    second_fraction = float(beta_distribution.ppf(clipped[1], alpha[1], alpha[2]))
    id_weights = np.asarray(
        [first, (1.0 - first) * second_fraction, (1.0 - first) * (1.0 - second_fraction)],
        dtype=np.float64,
    )
    minimum = float(entry["minimum_each"])
    maximum = float(entry["maximum_each"])
    if np.any((id_weights < minimum) | (id_weights > maximum)):
        random = np.random.default_rng(seed)
        for _ in range(10000):
            candidate = random.dirichlet(alpha)
            if np.all((candidate >= minimum) & (candidate <= maximum)):
                id_weights = candidate
                break
        else:
            msg = "Could not draw a feasible truncated Dirichlet simplex vector."
            raise RuntimeError(msg)
    if ood_index is None:
        return ({name: float(id_weights[index]) for index, name in enumerate(components)}, None)
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
    return selected, gap


def _select_coupled(
    name: str,
    entry: Mapping[str, Any],
    *,
    seed_base: int,
    case_index: int,
    use_ood: bool,
) -> dict[str, Any]:
    """Select one complete coupled record from a label-derived stream."""
    if entry["kind"] != "parameter_set":
        message = f"Coupled selection {name!r} must be one parameter_set."
        raise ValueError(message)
    candidates = entry.get("ood_sets", []) if use_ood else entry["sets"]
    if not candidates:
        msg = f"Selected coupled unit {name!r} has no {'OOD' if use_ood else 'ID'} records."
        raise ValueError(msg)
    rng = np.random.default_rng(seeding.derive_seed(seed_base, "coupled_selection", name, str(case_index), "ood" if use_ood else "id"))
    return copy.deepcopy(candidates[int(rng.integers(0, len(candidates)))])


def _ood_key(entry: Mapping[str, Any]) -> str | None:
    """Return the configured OOD inventory key for one typed entry."""
    return {
        "interval": "ood",
        "conditional_interval": "parameter_ood",
        "integer": "ood",
        "categorical": "ood_choices",
        "simplex": "ood_values",
        "parameter_set": "ood_sets",
    }.get(str(entry["kind"]))


def eligible_ood_units(
    family_contract: Mapping[str, Any],
    *,
    groups: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    """Return profile-projected OOD units in canonical registry order."""
    registry = family_contract["parameter_registry"]
    units: list[dict[str, str]] = []
    for name, entry in registry.items():
        group = entry.get("ood_group")
        key = _ood_key(entry)
        if group in groups and key is not None and bool(entry.get(key)):
            units.append(
                {
                    "unit_id": name,
                    "ood_group": str(group),
                    "unit_kind": str(entry["kind"]),
                }
            )
    for name, contract in family_contract["coupled_ood_records"].items():
        group = contract["ood_group"]
        if group in groups and contract["records"]:
            units.append(
                {
                    "unit_id": name,
                    "ood_group": str(group),
                    "unit_kind": "complete_atomic_record",
                }
            )
    identifiers = [unit["unit_id"] for unit in units]
    if len(identifiers) != len(set(identifiers)):
        message = "Profile-projected OOD inventory contains duplicate unit identifiers."
        raise ValueError(message)
    return tuple(units)


def allocate_ood_units(
    eligible_units: tuple[Mapping[str, str], ...],
    *,
    case_count: int,
) -> tuple[dict[str, str], ...]:
    """Allocate cases approximately evenly over every eligible atomic unit."""
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 1:
        message = "Parameter-OOD allocation requires a positive integer case count."
        raise ValueError(message)
    if not eligible_units:
        message = "Parameter-OOD allocation has no profile-eligible configured units."
        raise ValueError(message)
    return tuple(copy.deepcopy(dict(eligible_units[index % len(eligible_units)])) for index in range(case_count))


def _ood_selections(
    family_contract: Mapping[str, Any],
    assignment: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    groups: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate and return the one unit preallocated to this OOD case."""
    group = assignment["ood_group"]
    if group is None:
        return ()
    if group not in groups:
        message = f"Assigned OOD group {group!r} is not active for this profile."
        raise ValueError(message)
    eligible = eligible_ood_units(family_contract, groups=groups)
    if list(eligible) != policy.get("eligible_units"):
        message = "Resolved parameter-OOD eligibility changed after campaign planning."
        raise ValueError(message)
    by_identifier = {unit["unit_id"]: unit for unit in eligible}
    selected = assignment.get("ood_unit_id")
    if selected not in by_identifier:
        message = f"Assigned parameter-OOD unit {selected!r} is not profile eligible."
        raise ValueError(message)
    if by_identifier[selected]["ood_group"] != group:
        message = f"Assigned parameter-OOD unit {selected!r} disagrees with group {group!r}."
        raise ValueError(message)
    if assignment.get("ood_units_per_case") != 1:
        message = "Each parameter-OOD case must activate exactly one eligible unit."
        raise ValueError(message)
    return (str(selected),)


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
            rng = np.random.default_rng(seeding.derive_seed(seed_base, "categorical_selection", name, str(case_index)))
            values[name] = copy.deepcopy(choices[int(rng.integers(0, len(choices)))])
        elif kind == "parameter_set":
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
    parameters: tuple[str, ...],
    selected_ood: frozenset[str],
    seed_base: int,
    case_index: int,
    fixed: Mapping[str, Any],
    family_bounds: Mapping[str, float],
    initial_moisture_field_constraint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float]]:
    """Map one physical block row through typed and jointly constrained supports."""
    values: dict[str, Any] = {}
    coordinates_by_name: dict[str, float] = {}
    bounds_by_name: dict[str, Mapping[str, Any]] = {}
    ood_details: dict[str, dict[str, Any]] = {}
    conditional_coordinates: dict[str, float] = {}
    column = 0
    for name in parameters:
        entry = registry[name]
        dimension = registry_service.effective_dimension(entry)
        coordinates = row[column : column + dimension]
        column += dimension
        if entry["kind"] == "conditional_interval":
            conditional_coordinates[name] = float(coordinates[0])
        elif entry["kind"] in {"interval", "integer"}:
            coordinate = float(coordinates[0])
            bounds, detail = _selected_interval(
                name,
                entry,
                use_ood=name in selected_ood,
                seed_base=seed_base,
                case_index=case_index,
            )
            coordinates_by_name[name] = coordinate
            bounds_by_name[name] = bounds
            if entry["kind"] == "interval":
                values[name] = _interval_value(entry, coordinate, bounds=bounds)
            else:
                values[name] = _integer_value(coordinate, bounds=bounds)
            if detail is not None:
                ood_details[name] = detail
        elif entry["kind"] == "simplex":
            ood_index = None
            if name in selected_ood:
                count = len(entry.get("ood_values", []))
                ood_index = 0 if count == 0 else seeding.derive_seed(seed_base, "simplex_ood", name, str(case_index)) % count
            simplex_values, transform_space_distance = _simplex_value(
                entry,
                coordinates,
                ood_index=ood_index,
                seed=seeding.derive_seed(seed_base, "truncated_dirichlet", name, str(case_index)),
            )
            values[name] = simplex_values
            if ood_index is not None:
                ood_details[name] = {
                    "selection_kind": "complete_simplex",
                    "record_index": ood_index,
                    "values": copy.deepcopy(values[name]),
                    "transform_space_distance": transform_space_distance,
                }
        elif dimension != 0:
            msg = f"Registry parameter {name!r} owns an unsupported numerical dimension."
            raise RuntimeError(msg)
    expected = materials.sampling_block_dimensions(
        registry,
        blocks=(block,),
        block_parameters={block: parameters},
    )[block]
    if column != expected:
        msg = f"Sampling block {block!r} consumed {column} coordinates; expected {expected}."
        raise RuntimeError(msg)
    if block == "operation":
        _condition_operation_values(
            values,
            registry=registry,
            coordinates=coordinates_by_name,
            bounds=bounds_by_name,
            fixed=fixed,
        )
    elif block == "initial_moisture":
        _condition_initial_moisture_values(
            values,
            registry=registry,
            coordinates=coordinates_by_name,
            bounds=bounds_by_name,
            family_bounds=family_bounds,
            field_constraint=initial_moisture_field_constraint,
            natural_support_departure_allowed=bool(selected_ood.intersection(materials.INITIAL_MOISTURE_LEVEL_PARAMETERS)),
        )
    for name, detail in ood_details.items():
        detail["realized_value"] = copy.deepcopy(values[name])
    return values, ood_details, conditional_coordinates


def _conditional_evidence(
    name: str,
    entry: Mapping[str, Any],
    support: Mapping[str, Any],
    *,
    value: float,
    coordinate: float | None,
    selected_tail: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return complete realized support evidence for one conditional coordinate."""
    transformed_value = registry_service.transformed_coordinate(value, entry)
    natural = support["id_interval"]
    lower_t = float(natural["transformed_lower"])
    upper_t = float(natural["transformed_upper"])
    width = float(natural["transformed_width"])
    if selected_tail is None:
        support_kind = "natural"
        transformed_distance = 0.0
        transformed_distance_fraction = 0.0
        ood_interval = None
        tail_id = None
        physical_interpretation = "material_natural_global_packing"
    else:
        direction = str(selected_tail["direction"])
        support_kind = str(selected_tail["support_kind"])
        transformed_distance = lower_t - transformed_value if direction == "lower" else transformed_value - upper_t
        transformed_distance_fraction = transformed_distance / width
        ood_interval = [float(selected_tail["lower"]), float(selected_tail["upper"])]
        tail_id = f"{name}__{support_kind}"
        physical_interpretation = str(selected_tail["physical_interpretation"])
    return {
        "selection_kind": "conditional_scalar_interval",
        "support_kind": support_kind,
        "support_resolver": support["support_resolver"],
        "conditioning_coordinate": support["conditioning_coordinate"],
        "material_kappa_nominal": support["material_kappa_nominal"],
        "sampled_kappa_mean": support["sampled_kappa_mean"],
        "eps_bed_cal_ref": support["eps_bed_cal_ref"],
        "packing_porosity_mean_support": copy.deepcopy(support["packing_porosity_mean_support"]),
        "A_KC_reference": support["A_KC_reference"],
        "id_interval": [float(natural["lower"]), float(natural["upper"])],
        "id_transformed_interval": [lower_t, upper_t],
        "id_transformed_width": width,
        "ood_interval": ood_interval,
        "tail_id": tail_id,
        "transform": str(entry["transform"]),
        "conditional_unit_coordinate": coordinate,
        "transformed_support_coordinate": transformed_value,
        "transformed_ood_distance": transformed_distance,
        "transformed_ood_distance_fraction": transformed_distance_fraction,
        "transform_space_distance": transformed_distance,
        "transformed_gap": None if selected_tail is None else float(selected_tail["transformed_gap"]),
        "transformed_width": None if selected_tail is None else float(selected_tail["transformed_width"]),
        "transformed_gap_fraction": None if selected_tail is None else float(selected_tail["transformed_gap_fraction"]),
        "transformed_width_fraction": None if selected_tail is None else float(selected_tail["transformed_width_fraction"]),
        "physical_interpretation": physical_interpretation,
        "ood_basis": support["ood_basis"],
        "ood_status": support["ood_status"],
        "available_ood_directions": [str(tail["direction"]) for tail in support["available_ood_tails"]],
        "unavailable_ood_directions": copy.deepcopy(support["unavailable_ood_directions"]),
        "realized_value": value,
    }


def _apply_conditional_values(
    family_contract: Mapping[str, Any],
    *,
    coordinates: Mapping[str, float],
    selected_ood: frozenset[str],
    values: dict[str, Any],
    seed_base: int,
    case_index: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Resolve every conditional coordinate after all conditioning values and records."""
    registry = family_contract["parameter_registry"]
    expected = {name for name, entry in registry.items() if entry["kind"] == "conditional_interval"}
    if set(coordinates) != expected:
        message = f"Conditional coordinate coverage changed: expected={sorted(expected)}, actual={sorted(coordinates)}."
        raise ValueError(message)
    evidence: dict[str, dict[str, Any]] = {}
    ood_details: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        entry = registry[name]
        support = registry_service.resolve_conditional_support(
            entry,
            values=values,
            material_contract=family_contract,
        )
        selected_tail = None
        bounds: Mapping[str, Any] = support["id_interval"]
        if name in selected_ood:
            tails = support["available_ood_tails"]
            if not tails:
                message = f"Selected conditional OOD unit {name!r} has no physically feasible tail."
                raise ValueError(message)
            index = seeding.derive_seed(seed_base, "conditional_ood_tail", name, str(case_index)) % len(tails)
            selected_tail = tails[index]
            bounds = selected_tail
        coordinate = float(coordinates[name])
        realized = _interval_value(entry, coordinate, bounds=bounds)
        values[name] = realized
        detail = _conditional_evidence(
            name,
            entry,
            support,
            value=realized,
            coordinate=coordinate,
            selected_tail=selected_tail,
        )
        evidence[name] = detail
        if selected_tail is not None:
            ood_details[name] = copy.deepcopy(detail)
    return evidence, ood_details


def _nominal_conditional_supports(
    family_contract: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve conditional evidence for explicit configured nominal cases."""
    result: dict[str, dict[str, Any]] = {}
    for name, entry in family_contract["parameter_registry"].items():
        if entry["kind"] != "conditional_interval":
            continue
        support = registry_service.resolve_conditional_support(
            entry,
            values=values,
            material_contract=family_contract,
        )
        value = float(values[name])
        lower = float(support["id_interval"]["lower"])
        upper = float(support["id_interval"]["upper"])
        if not lower <= value <= upper:
            message = f"Explicit nominal conditional value {name!r} lies outside its resolved natural support."
            raise ValueError(message)
        transformed = registry_service.transformed_coordinate(value, entry)
        lower_t = float(support["id_interval"]["transformed_lower"])
        width = float(support["id_interval"]["transformed_width"])
        coordinate = (transformed - lower_t) / width
        result[name] = _conditional_evidence(
            name,
            entry,
            support,
            value=value,
            coordinate=coordinate,
            selected_tail=None,
        )
    return result


def _coupled_transform_space_distance(
    registry: Mapping[str, Mapping[str, Any]],
    values: Mapping[str, float],
) -> tuple[float, dict[str, float | None]]:
    """Return normalized distance from one complete record to its ID support."""
    component_distances: dict[str, float | None] = {}
    squared = 0.0
    for component, value in values.items():
        entry = registry[component]
        if entry["kind"] != "interval":
            component_distances[component] = None
            continue
        lower_t = registry_service.transformed_coordinate(float(entry["lower"]), entry)
        upper_t = registry_service.transformed_coordinate(float(entry["upper"]), entry)
        value_t = registry_service.transformed_coordinate(float(value), entry)
        width = upper_t - lower_t
        distance = max(lower_t - value_t, value_t - upper_t, 0.0) / width
        component_distances[component] = distance
        squared += distance * distance
    result = math.sqrt(squared)
    if not math.isfinite(result) or result <= 0.0:
        msg = "Complete coupled OOD record has no positive normalized transform-space distance."
        raise ValueError(msg)
    return result, component_distances


def _apply_coupled_ood_records(
    family_contract: Mapping[str, Any],
    *,
    selected_ood: frozenset[str],
    values: dict[str, Any],
    selections: dict[str, str],
    seed_base: int,
    case_index: int,
) -> dict[str, dict[str, Any]]:
    """Apply selected complete density or kinetics records atomically."""
    details: dict[str, dict[str, Any]] = {}
    for name, contract in family_contract["coupled_ood_records"].items():
        if name not in selected_ood:
            continue
        records = contract["records"]
        if not records:
            msg = f"Selected coupled OOD unit {name!r} has no records."
            raise ValueError(msg)
        index = seeding.derive_seed(seed_base, "coupled_ood_record", name, str(case_index)) % len(records)
        record = records[index]
        components = tuple(contract["components"])
        before = {component: copy.deepcopy(values[component]) for component in components}
        values.update(copy.deepcopy(record["values"]))
        selections[name] = str(record["id"])
        transform_space_distance, component_distances = _coupled_transform_space_distance(
            family_contract["parameter_registry"],
            record["values"],
        )
        details[name] = {
            "selection_kind": "complete_coupled_record",
            "record_id": record["id"],
            "record_index": index,
            "components": list(components),
            "id_row_values": before,
            "ood_record_values": copy.deepcopy(record["values"]),
            "transform_space_distance": transform_space_distance,
            "component_transform_space_distances": component_distances,
            "distance_normalization": "component_distance_to_id_support_divided_by_component_id_width",
            "metadata": copy.deepcopy(record["metadata"]),
        }
    return details


def _units(registry: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Return stable scalar and coupled-component units."""
    units: dict[str, str] = {}
    for name, entry in registry.items():
        if entry["kind"] == "parameter_set":
            units.update(entry["units"])
        else:
            units[name] = str(entry["unit"])
    return units


def _explicit_nominal_values(
    registry: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve only authored nominal values and complete atomic records."""
    values: dict[str, Any] = {}
    coupled_selections: dict[str, str] = {}
    for name, entry in registry.items():
        kind = str(entry["kind"])
        if kind == "derived":
            continue
        if kind == "fixed":
            values[name] = copy.deepcopy(entry["value"])
            continue
        if kind in {"interval", "conditional_interval", "integer"}:
            if "nominal" not in entry:
                message = f"Explicit pilot nominal is missing for resolved parameter owner/key {name!r}."
                raise ValueError(message)
            nominal = entry["nominal"]
            if isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(float(nominal)):
                message = f"Explicit pilot nominal for {name!r} must be finite numeric data."
                raise ValueError(message)
            if kind == "integer":
                integer = int(nominal)
                if float(integer) != float(nominal):
                    message = f"Explicit pilot nominal for integer {name!r} is not integral."
                    raise ValueError(message)
                values[name] = integer
            else:
                values[name] = float(nominal)
            continue
        if kind == "simplex":
            nominal = entry.get("nominal")
            components = tuple(entry["components"])
            if not isinstance(nominal, dict) or set(nominal) != set(components):
                message = f"Explicit pilot nominal simplex is missing or incomplete for owner/key {name!r}."
                raise ValueError(message)
            weights = {component: float(nominal[component]) for component in components}
            if not all(math.isfinite(value) for value in weights.values()) or not math.isclose(
                sum(weights.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                message = f"Explicit pilot nominal simplex {name!r} is non-finite or does not sum to one."
                raise ValueError(message)
            minimum = float(entry["minimum_each"])
            maximum = float(entry["maximum_each"])
            if any(not minimum <= value <= maximum for value in weights.values()):
                message = f"Explicit pilot nominal simplex {name!r} violates its configured component bounds."
                raise ValueError(message)
            values[name] = weights
            continue
        if kind == "categorical":
            if "nominal" not in entry:
                message = f"Explicit pilot nominal is missing for categorical owner/key {name!r}."
                raise ValueError(message)
            nominal = copy.deepcopy(entry["nominal"])
            if nominal not in entry["choices"]:
                message = f"Explicit pilot nominal for {name!r} is not a configured natural choice."
                raise ValueError(message)
            values[name] = nominal
            continue
        if kind == "parameter_set":
            records = entry.get("sets")
            if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
                message = (
                    f"Explicit pilot nominal atomic record for owner/key {name!r} must resolve to exactly one complete configured natural record."
                )
                raise ValueError(message)
            record = records[0]
            components = tuple(entry["components"])
            record_values = record.get("values")
            if not isinstance(record_values, dict) or set(record_values) != set(components):
                message = f"Explicit pilot nominal atomic record {name!r} is incomplete."
                raise ValueError(message)
            overlap = set(values).intersection(record_values)
            if overlap:
                message = f"Explicit pilot nominal atomic record {name!r} conflicts with values {sorted(overlap)}."
                raise ValueError(message)
            values.update(copy.deepcopy(record_values))
            coupled_selections[name] = str(record["id"])
            continue
        message = f"Explicit pilot nominal resolution does not support registry kind {kind!r} for {name!r}."
        raise ValueError(message)
    return values, coupled_selections


def _sample_nominal_case(config: _SamplingConfig, case_index: int) -> CaseSample:
    """Return one fail-closed explicit configured nominal with no seed search."""
    registry = config.scientific_values["material"]["parameter_registry"]
    values, coupled_selections = _explicit_nominal_values(registry)
    values = registry_service.resolve_derived_values(
        registry,
        values,
        defer_missing=True,
    )
    return CaseSample(
        values=values,
        units=_units(registry),
        coupled_selections=coupled_selections,
        conditional_supports=_nominal_conditional_supports(
            config.scientific_values["material"],
            values,
        ),
        block_provenance={
            block: {
                "sampling_kind": "explicit_configured_nominal",
                "case_position": config.case_indices.index(case_index),
                "seed_search": False,
                "stochastic_field_seed_origin": "normal_label_derived_case_substreams",
            }
            for block in config.scientific_values["sampling"]["blocks"]
        },
        ood_provenance={
            "group": None,
            "active_ood_group": None,
            "selected_units": [],
            "active_unit_id": None,
            "active_record_id": None,
            "units_per_case": 0,
            "transform_space_distance": 0.0,
            "natural_support_state": "nominal_reference",
            "selections": {},
            "nominal_source": "explicit_resolved_registry_nominals_and_complete_records",
            "seed_search": False,
        },
    )


def sample_case(config: _SamplingConfig, case_index: int) -> CaseSample:
    """Resolve one deterministic case from profile-active block designs."""
    config.case_id(case_index)
    assignment = config.case_assignment(case_index)
    pilot_kind = assignment.get("pilot_case_kind")
    if pilot_kind == "nominal_reference":
        return _sample_nominal_case(config, case_index)
    if pilot_kind not in {None, "natural_pilot"}:
        message = f"Unsupported pilot case kind {pilot_kind!r}."
        raise ValueError(message)
    if config.seed_base is None:
        message = f"Batch {config.batch_name!r} has no executable sampling seed."
        raise ValueError(message)
    family_contract = config.scientific_values["material"]
    registry = family_contract["parameter_registry"]
    policy = config.scientific_values["parameter_ood"]
    policy_groups = tuple(policy["groups"])
    selected = frozenset(
        _ood_selections(
            family_contract,
            assignment,
            policy=policy,
            groups=policy_groups,
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
    ood_details: dict[str, dict[str, Any]] = {}
    conditional_coordinates: dict[str, float] = {}
    family_bounds = family_contract.get("initial_moisture_bounds", {})
    initial_moisture_field_constraint = family_contract.get("initial_moisture_field_constraint", {})
    for block, plan in config.scientific_values["sampling"]["blocks"].items():
        row_index = int(plan["permutation"][position])
        design = unit_design(
            config.scientific_values["sampling"]["method"],
            count=len(config.case_indices),
            dimensions=int(plan["effective_dimension"]),
            seed=int(plan["design_seed"]),
        )
        if _array_sha256(design) != plan["design_sha256"]:
            message = f"Sampling design digest changed for block {block!r}."
            raise RuntimeError(message)
        parameters = tuple(str(name) for name in plan["parameters"])
        numerical_block_parameters = {name for name in parameters if registry_service.effective_dimension(registry[name]) > 0}
        overlap = set(values).intersection(numerical_block_parameters)
        if overlap:
            message = f"Block {block!r} conflicts with coupled or fixed values {sorted(overlap)}."
            raise ValueError(message)
        block_values, block_ood_details, block_conditional_coordinates = _block_values(
            registry,
            block,
            design[row_index],
            parameters=parameters,
            selected_ood=selected,
            seed_base=config.seed_base,
            case_index=case_index,
            fixed=config.scientific_values["scientific_fixed_values"],
            family_bounds=family_bounds,
            initial_moisture_field_constraint=initial_moisture_field_constraint,
        )
        values.update(block_values)
        ood_details.update(block_ood_details)
        overlap = set(conditional_coordinates).intersection(block_conditional_coordinates)
        if overlap:
            message = f"Conditional coordinates were resolved by multiple blocks: {sorted(overlap)}."
            raise ValueError(message)
        conditional_coordinates.update(block_conditional_coordinates)
        block_provenance[block] = {
            "seed_origin": plan["seed_origin"],
            "design_seed": plan["design_seed"],
            "design_sha256": plan["design_sha256"],
            "permutation_seed": plan["permutation_seed"],
            "permutation_sha256": plan["permutation_sha256"],
            "case_position": position,
            "block_row_index": row_index,
        }
    coupled_ood_details = _apply_coupled_ood_records(
        family_contract,
        selected_ood=selected,
        values=values,
        selections=coupled_selections,
        seed_base=config.seed_base,
        case_index=case_index,
    )
    ood_details.update(coupled_ood_details)
    conditional_supports, conditional_ood_details = _apply_conditional_values(
        family_contract,
        coordinates=conditional_coordinates,
        selected_ood=selected,
        values=values,
        seed_base=config.seed_base,
        case_index=case_index,
    )
    ood_details.update(conditional_ood_details)
    if "schedule.component_weights" in values:
        simplex = values["schedule.component_weights"]
        if not isinstance(simplex, dict):
            message = "schedule.component_weights did not resolve to a complete simplex mapping."
            raise TypeError(message)
    values = registry_service.resolve_derived_values(
        registry,
        values,
        defer_missing=True,
    )
    if "initial_moisture_bounds" in family_contract:
        materials.initial_moisture_generation_bounds(
            family_contract,
            values,
            active_ood_unit=next(iter(selected), None),
        )
    selected_units = sorted(selected)
    if selected_units:
        if len(selected_units) != 1:
            message = "Parameter-OOD provenance requires exactly one active coordinate or complete record."
            raise ValueError(message)
        active_unit_id = selected_units[0]
        active_detail = ood_details[active_unit_id]
        transform_space_distance = active_detail.get("transform_space_distance")
        if isinstance(transform_space_distance, bool) or not isinstance(transform_space_distance, (int, float)):
            message = f"Parameter-OOD unit {active_unit_id!r} has no numeric transform-space distance."
            raise ValueError(message)
        active_record_id = active_detail.get(
            "record_id",
            active_detail.get("tail_id"),
        )
        if active_record_id is None and "record_index" in active_detail:
            record_number = int(active_detail["record_index"]) + 1
            active_record_id = f"{active_unit_id}__record_{record_number}"
    else:
        active_unit_id = None
        active_record_id = None
        transform_space_distance = 0.0
    return CaseSample(
        values=values,
        units=_units(registry),
        coupled_selections=coupled_selections,
        block_provenance=block_provenance,
        conditional_supports=conditional_supports,
        ood_provenance={
            "group": assignment["ood_group"],
            "active_ood_group": assignment["ood_group"],
            "selected_units": selected_units,
            "active_unit_id": active_unit_id,
            "active_record_id": active_record_id,
            "units_per_case": assignment["ood_units_per_case"],
            "transform_space_distance": float(transform_space_distance),
            "natural_support_state": ("natural" if not selected_units else "parameter_ood"),
            "selections": ood_details,
        },
    )


def sample_all_cases(config: _SamplingConfig) -> dict[int, CaseSample]:
    """Return all configured samples in canonical case-index order."""
    return {case_index: sample_case(config, case_index) for case_index in config.case_indices}


def sampling_plan_sha256(config: _SamplingConfig) -> str:
    """Return the deterministic identity of all persisted block plans."""
    return common.serialization.canonical_json_sha256(config.scientific_values["sampling"]["blocks"])
