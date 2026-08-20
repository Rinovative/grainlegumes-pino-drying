"""
generation_contracts_materials.py

Resolve compact role-neutral material records into typed scientific registries.
Responsibilities:
  - Discover material identifiers and resolve profile-specific coordinate contracts
  - Merge disjoint common, operation, and family-specific scientific owners
  - Resolve atomic density, Oswin, and kinetics records with effective provenance
Design principles:
  - Material files contain family science and no campaign roles or shared controls
  - Density calibration is always sampled-rho/fixed-reference-epsilon in natural data
  - Every effective parameter exposes its scientific source and interpretation
This module does NOT:
  - Assign campaign roles, counts, seeds, dataset membership, or runtime mappings
  - Invent scientific values, derivations, sources, or compatibility modes
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Any, Final

import yaml

from src import common

from . import generation_contracts_porosity as porosity_service
from . import generation_contracts_provenance as provenance_service
from . import generation_contracts_registry as registry_service

_MATERIAL_FAMILY_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]*")
_PARAMETER_REGISTRY_SCHEMA_VERSION: Final = 1
POROSITY_GENERATOR_PARAMETERS: Final = (
    "porosity.smooth_len_rel",
    "porosity.texture_amp",
)
INITIAL_MOISTURE_LEVEL_PARAMETERS: Final = (
    "initial_moisture.mean_db",
    "initial_moisture.amplitude_db",
)


_REGISTRY_METADATA_KEYS: Final = frozenset({"report_symbol", "description"})
_SCOPE_KEYS: Final = {
    "common_name",
    "species",
    "market_class",
    "product_form",
    "coat_or_hull_state",
    "description",
}
_ALLOWED_SCOPE_VALUES: Final = MappingProxyType(
    {
        "product_form": frozenset({"whole_seed", "whole_achene"}),
        "coat_or_hull_state": frozenset({"intact", "hull_intact"}),
    }
)


def validate_material_family(value: object) -> str:
    """Return one safe role-neutral material-family identifier."""
    if not isinstance(value, str) or _MATERIAL_FAMILY_PATTERN.fullmatch(value) is None:
        message = f"Material family must match {_MATERIAL_FAMILY_PATTERN.pattern!r}; received {value!r}."
        raise ValueError(message)
    return value


def available_material_families() -> tuple[str, ...]:
    """Discover validated role-neutral material identifiers from configuration files."""
    root = common.paths.get_project_root() / "configs" / "generation" / "materials"
    paths = tuple(sorted(root.glob("*.yaml")))
    if not paths:
        message = f"Generation material configuration directory contains no YAML files: {root}."
        raise FileNotFoundError(message)
    families: list[str] = []
    for path in paths:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            message = f"Generation material configuration is not valid YAML: {path}."
            raise ValueError(message) from error
        if not isinstance(payload, Mapping):
            message = f"Generation material configuration must be a mapping: {path}."
            raise TypeError(message)
        family = validate_material_family(payload.get("material_family"))
        if payload.get("schema_kind") != "generation_material" or payload.get("schema_version") != 1:
            message = f"Unsupported generation material schema: {path}."
            raise ValueError(message)
        if family != path.stem:
            message = f"Generation material filename and material_family disagree: {path}."
            raise ValueError(message)
        families.append(family)
    if len(families) != len(set(families)):
        message = f"Generation material configurations contain duplicate identifiers: {families}."
        raise ValueError(message)
    return tuple(families)


def _profile_applicability(entry: Mapping[str, Any], *, name: str) -> tuple[str, ...]:
    """Return one registry entry's ordered profile applicability."""
    value = entry.get("profile_applicability")
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        message = f"Registry parameter {name!r} must declare non-empty profile_applicability."
        raise ValueError(message)
    return tuple(value)


def profile_parameter_names(
    registry: Mapping[str, Mapping[str, Any]],
    profile_id: str,
) -> tuple[str, ...]:
    """Return registry parameters applicable to one profile in authored order."""
    names = tuple(name for name, entry in registry.items() if profile_id in _profile_applicability(entry, name=name))
    if not names:
        message = f"Registry declares no parameters for simulation profile {profile_id!r}."
        raise ValueError(message)
    return names


def sampling_blocks(registry: Mapping[str, Mapping[str, Any]]) -> Mapping[str, tuple[str, ...]]:
    """Derive ordered sampling-block membership from registry declarations."""
    grouped: dict[str, list[str]] = {}
    for name, entry in registry.items():
        block = entry.get("block")
        if block is None:
            continue
        if not isinstance(block, str) or not block:
            message = f"Registry parameter {name!r} block must be non-empty text."
            raise ValueError(message)
        grouped.setdefault(block, []).append(name)
    if not grouped:
        message = "Registry must declare at least one sampling block."
        raise ValueError(message)
    return MappingProxyType({block: tuple(names) for block, names in grouped.items()})


def active_sampling_blocks(
    registry: Mapping[str, Mapping[str, Any]],
    profile_id: str,
) -> tuple[str, ...]:
    """Derive numerical blocks consumed by one profile from the registry."""
    applicable = set(profile_parameter_names(registry, profile_id))
    return tuple(block for block, names in sampling_blocks(registry).items() if applicable.intersection(names))


def active_ood_groups(
    registry: Mapping[str, Mapping[str, Any]],
    profile_id: str,
) -> tuple[str, ...]:
    """Derive profile-active parameter-OOD groups in authored registry order."""
    applicable = set(profile_parameter_names(registry, profile_id))
    groups: list[str] = []
    for name, entry in registry.items():
        if name not in applicable:
            continue
        group = entry.get("ood_group")
        if group is None:
            continue
        if not isinstance(group, str) or not group:
            message = f"Registry parameter {name!r} ood_group must be non-empty text."
            raise ValueError(message)
        if group not in groups:
            groups.append(group)
    return tuple(groups)


def derived_parameter_names(registry: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return parameters whose values are derived by the typed registry."""
    return tuple(name for name, entry in registry.items() if entry.get("kind") == "derived")


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return one isolated string-keyed mapping."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return copy.deepcopy(dict(value))


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str, optional: set[str] | None = None) -> None:
    """Require one exact mapping schema."""
    allowed_optional = set() if optional is None else optional
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected | allowed_optional))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise ValueError(message)


def _finite(value: Any, *, label: str) -> float:
    """Return one finite non-boolean scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        message = f"{label} must be finite."
        raise ValueError(message)
    return float(value)


def _semantic_entry(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, str], int | None]:
    """Separate executable semantics, coordinate order, and catalogue labels."""
    entry = _mapping(value, label=label)
    sampling_order_value = entry.pop("sampling_order", None)
    if sampling_order_value is None:
        sampling_order = None
    elif isinstance(sampling_order_value, bool) or not isinstance(sampling_order_value, int) or sampling_order_value <= 0:
        message = f"{label}.sampling_order must be a positive integer."
        raise ValueError(message)
    else:
        sampling_order = sampling_order_value
    metadata: dict[str, str] = {}
    for key in _REGISTRY_METADATA_KEYS:
        item = entry.pop(key, None)
        if not isinstance(item, str) or not item:
            message = f"{label}.{key} must be non-empty text."
            raise ValueError(message)
        metadata[key] = item
    return entry, metadata, sampling_order


def _validate_sampling_block_contract(
    definitions: Mapping[str, Mapping[str, Any]],
    sampling_orders: Mapping[str, int],
) -> dict[str, tuple[str, ...]]:
    """Derive registry block membership from entry-owned block and coordinate order."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    expected: set[str] = set()
    for name, entry in definitions.items():
        block = entry.get("block")
        if block is None:
            if name in sampling_orders:
                message = f"Non-sampled registry parameter {name!r} cannot declare sampling_order."
                raise ValueError(message)
            continue
        if not isinstance(block, str) or not block:
            message = f"Registry parameter {name!r} block must be non-empty text."
            raise ValueError(message)
        expected.add(name)
        order = sampling_orders.get(name)
        if order is None:
            message = f"Sampled registry parameter {name!r} must declare sampling_order."
            raise ValueError(message)
        grouped.setdefault(block, []).append((order, name))
    unknown = sorted(set(sampling_orders).difference(expected))
    if unknown:
        message = f"Only sampled registry parameters may declare sampling_order: {unknown}."
        raise ValueError(message)
    if not grouped:
        message = "Generation parameter registry must declare at least one sampling block."
        raise ValueError(message)
    normalized: dict[str, tuple[str, ...]] = {}
    for block, ordered_names in grouped.items():
        ranks = [order for order, _name in ordered_names]
        expected_ranks = list(range(1, len(ordered_names) + 1))
        if sorted(ranks) != expected_ranks:
            message = f"Sampling block {block!r} orders must be exactly {expected_ranks}; received {sorted(ranks)}."
            raise ValueError(message)
        normalized[block] = tuple(name for _order, name in sorted(ordered_names))
    return normalized


def validate_semantic_registry(
    value: Any,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
    dict[str, tuple[str, ...]],
]:
    """Validate registry-owned parameter semantics and sampling order."""
    config = _mapping(value, label="generation parameter registry")
    _exact_keys(
        config,
        {"schema_kind", "schema_version", "parameters"},
        label="generation parameter registry",
    )
    if config["schema_kind"] != "generation_parameter_registry" or config["schema_version"] != _PARAMETER_REGISTRY_SCHEMA_VERSION:
        message = "Unsupported generation parameter-registry schema."
        raise ValueError(message)
    raw = _mapping(config["parameters"], label="generation parameter registry.parameters")
    if not raw:
        message = "generation parameter registry.parameters must be non-empty."
        raise ValueError(message)
    definitions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, str]] = {}
    sampling_orders: dict[str, int] = {}
    symbols: dict[str, str] = {}
    for name, item in raw.items():
        definition, entry_metadata, sampling_order = _semantic_entry(item, label=f"parameters.{name}")
        definitions[name] = definition
        metadata[name] = entry_metadata
        if sampling_order is not None:
            sampling_orders[name] = sampling_order
        symbol = entry_metadata["report_symbol"]
        if symbol in symbols:
            message = f"Report symbol {symbol!r} is assigned to both {symbols[symbol]!r} and {name!r}."
            raise ValueError(message)
        symbols[symbol] = name
    block_contract = _validate_sampling_block_contract(definitions, sampling_orders)
    return definitions, metadata, block_contract


def resolve_value_record(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Resolve one numeric owner record and expand its supplied provenance."""
    record = _mapping(value, label=label)
    provenance = record.pop("provenance", None)
    if provenance is None:
        message = f"{label} must contain provenance."
        raise ValueError(message)
    record["provenance"] = provenance_service.resolve_provenance(provenance, sources=sources, label=f"{label}.provenance")
    return record


def practical_target_moisture_wb(exact_reference_wb: float) -> float:
    """Round one safe-storage reference to a whole wet-basis percentage point."""
    reference = _finite(
        exact_reference_wb,
        label="exact safe-storage moisture reference",
    )
    if not 0 < reference < 1:
        message = "Exact safe-storage moisture reference must lie inside (0, 1)."
        raise ValueError(message)
    percentage = Decimal(str(reference)) * Decimal(100)
    return float(percentage.quantize(Decimal(1), rounding=ROUND_HALF_UP) / Decimal(100))


def _resolve_synthetic_ood_provenance(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Resolve one OOD provenance record and require synthetic-design evidence."""
    resolved = provenance_service.resolve_provenance(value, sources=sources, label=label)
    if resolved.get("evidence") != "synthetic_design":
        message = f"{label}.evidence must be synthetic_design."
        raise ValueError(message)
    return resolved


def _support_record(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
    allow_ood: bool = True,
) -> dict[str, Any]:
    """Flatten one compact nominal/support record for registry merging."""
    record = resolve_value_record(value, sources=sources, label=label)
    if "ood_provenance" not in record:
        message = f"{label}.ood_provenance is required."
        raise ValueError(message)
    record["ood_provenance"] = _resolve_synthetic_ood_provenance(record["ood_provenance"], sources=sources, label=f"{label}.ood_provenance")
    required = {"nominal", "support", "transform", "distribution", "provenance", "ood_provenance"}
    optional = {"ood_supports"} if allow_ood else set()
    _exact_keys(record, required, optional=optional, label=label)
    support = _mapping(record.pop("support"), label=f"{label}.support")
    _exact_keys(support, {"lower", "upper"}, label=f"{label}.support")
    result = {
        "lower": support["lower"],
        "upper": support["upper"],
        "nominal": record["nominal"],
        "distribution": record["distribution"],
        "provenance": record["provenance"],
        "ood_provenance": record["ood_provenance"],
    }
    if "ood_supports" in record:
        result["ood"] = copy.deepcopy(record["ood_supports"])
    return result


def _validate_material_scope(value: Any, *, label: str) -> dict[str, str]:
    """Validate one compact controlled material identity block."""
    scope = _mapping(value, label=label)
    _exact_keys(scope, _SCOPE_KEYS, label=label)
    for key, item in scope.items():
        if not isinstance(item, str) or not item or item.strip() != item:
            message = f"{label}.{key} must be non-empty trimmed text."
            raise ValueError(message)
    for key, allowed in _ALLOWED_SCOPE_VALUES.items():
        if scope[key] not in allowed:
            message = f"{label}.{key} must be one of {sorted(allowed)}."
            raise ValueError(message)
    return {key: str(item) for key, item in scope.items()}


def _validate_packing_support(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    """Validate the material natural support for realized mean porosity."""
    support = resolve_value_record(value, sources=sources, label=label)
    _exact_keys(support, {"lower", "upper", "unit", "provenance"}, label=label)
    lower = _finite(support["lower"], label=f"{label}.lower")
    upper = _finite(support["upper"], label=f"{label}.upper")
    if support["unit"] != "1" or not 0 < lower < upper < 1:
        message = f"{label} must be one ordered dimensionless porosity support inside (0, 1)."
        raise ValueError(message)
    support["lower"] = lower
    support["upper"] = upper
    return support


def _atomic_ood_location(
    definitions: Mapping[str, Mapping[str, Any]],
    components: tuple[str, ...],
    *,
    label: str,
) -> tuple[str, str]:
    """Return the one registry-owned OOD group and block for atomic components."""
    locations: set[tuple[str, str]] = set()
    for name in components:
        entry = definitions[name]
        group = entry.get("ood_group")
        block = entry.get("block")
        if group is None and block is None:
            continue
        if not isinstance(group, str) or not group or not isinstance(block, str) or not block:
            message = f"{label} component {name!r} must declare both ood_group and block."
            raise ValueError(message)
        locations.add((group, block))
    if len(locations) != 1:
        message = f"{label} sampled components must share one OOD group and block; received {sorted(locations)}."
        raise ValueError(message)
    return next(iter(locations))


def _validate_density_record(
    value: Any,
    *,
    packing_support: Mapping[str, Any],
    definitions: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate natural sampled density, fixed reference epsilon, and atomic OOD records."""
    density = resolve_value_record(value, sources=sources, label=label)
    if "ood_provenance" not in density:
        message = f"{label}.ood_provenance is required."
        raise ValueError(message)
    density["ood_provenance"] = _resolve_synthetic_ood_provenance(density["ood_provenance"], sources=sources, label=f"{label}.ood_provenance")
    _exact_keys(
        density,
        {"record_id", "reference", "rho_bu_dry_ref_support", "ood_records", "selection_rule", "provenance", "ood_provenance"},
        label=label,
    )
    if not isinstance(density["record_id"], str) or not density["record_id"]:
        message = f"{label}.record_id must be non-empty text."
        raise ValueError(message)
    if density["selection_rule"] != "complete_record_atomic":
        message = f"{label}.selection_rule must be complete_record_atomic."
        raise ValueError(message)
    reference = _mapping(density["reference"], label=f"{label}.reference")
    _exact_keys(reference, {"rho_bu_dry_ref", "eps_bed_cal_ref"}, label=f"{label}.reference")
    rho = _finite(reference["rho_bu_dry_ref"], label=f"{label}.reference.rho_bu_dry_ref")
    eps = _finite(reference["eps_bed_cal_ref"], label=f"{label}.reference.eps_bed_cal_ref")
    if rho <= 0 or not 0 < eps < 1:
        message = f"{label}.reference is internally inconsistent."
        raise ValueError(message)
    particle = round(rho / (1.0 - eps), 6)
    if not float(packing_support["lower"]) <= eps <= float(packing_support["upper"]):
        message = f"{label}.reference.eps_bed_cal_ref must lie inside the material packing support."
        raise ValueError(message)
    support = _mapping(density["rho_bu_dry_ref_support"], label=f"{label}.rho_bu_dry_ref_support")
    _exact_keys(support, {"lower", "upper", "transform"}, label=f"{label}.rho_bu_dry_ref_support")
    lower = _finite(support["lower"], label=f"{label}.rho_bu_dry_ref_support.lower")
    upper = _finite(support["upper"], label=f"{label}.rho_bu_dry_ref_support.upper")
    if support["transform"] != "log" or not 0 < lower <= rho <= upper:
        message = f"{label}.rho_bu_dry_ref_support must be positive, ordered, and contain the reference."
        raise ValueError(message)
    raw_ood = density["ood_records"]
    if not isinstance(raw_ood, list):
        message = f"{label}.ood_records must be a list."
        raise TypeError(message)
    ood_records: list[dict[str, Any]] = []
    expected_ids = ("loose_low_density", "dense_high_density")
    if len(raw_ood) != len(expected_ids):
        message = f"{label}.ood_records must contain exactly {list(expected_ids)}."
        raise ValueError(message)
    identities: set[str] = set()
    for index, raw in enumerate(raw_ood):
        item_label = f"{label}.ood_records[{index}]"
        item = _mapping(raw, label=item_label)
        _exact_keys(item, {"record_id", "rho_bu_dry_ref", "eps_bed_cal_ref"}, optional={"basis"}, label=item_label)
        if item["record_id"] != expected_ids[index]:
            message = f"{item_label}.record_id must be {expected_ids[index]!r}."
            raise ValueError(message)
        identity = item["record_id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            message = f"{item_label}.record_id must be unique non-empty text."
            raise ValueError(message)
        identities.add(identity)
        values = {
            "rho_bu_dry_ref": _finite(item["rho_bu_dry_ref"], label=f"{item_label}.rho_bu_dry_ref"),
            "eps_bed_cal_ref": _finite(item["eps_bed_cal_ref"], label=f"{item_label}.eps_bed_cal_ref"),
        }
        if values["rho_bu_dry_ref"] <= 0 or not 0 < values["eps_bed_cal_ref"] < 1:
            message = f"{item_label} has invalid density-calibration values."
            raise ValueError(message)
        if lower <= values["rho_bu_dry_ref"] <= upper:
            message = f"{item_label}.rho_bu_dry_ref must be disjoint from natural support."
            raise ValueError(message)
        expected_loose = index == 0
        directional = (
            values["rho_bu_dry_ref"] < lower and values["eps_bed_cal_ref"] > float(packing_support["upper"])
            if expected_loose
            else values["rho_bu_dry_ref"] > upper and values["eps_bed_cal_ref"] < float(packing_support["lower"])
        )
        if not directional:
            message = f"{item_label} does not retain its required density/porosity OOD direction."
            raise ValueError(message)
        inferred_particle = values["rho_bu_dry_ref"] / (1.0 - values["eps_bed_cal_ref"])
        if not math.isclose(inferred_particle, particle, rel_tol=0.0, abs_tol=2e-5):
            message = f"{item_label} must preserve the reference inferred dry particle density."
            raise ValueError(message)
        metadata = {"basis": copy.deepcopy(item["basis"])} if "basis" in item else {}
        ood_records.append({"id": identity, "values": values, "metadata": metadata})
    density["reference"] = {"rho_bu_dry_ref": rho, "eps_bed_cal_ref": eps, "inferred_rho_particle_dry": particle}
    density["rho_bu_dry_ref_support"] = {"lower": lower, "upper": upper, "transform": "log"}
    density["ood_records"] = ood_records
    rho_owner = {
        "lower": lower,
        "upper": upper,
        "nominal": rho,
        "distribution": "uniform_in_log_transform_space",
        "provenance": copy.deepcopy(density["provenance"]),
    }
    eps_owner = {"value": eps, "nominal": eps, "provenance": copy.deepcopy(density["provenance"])}
    components = ("rho_bu_dry_ref", "eps_bed_cal_ref")
    ood_group, block = _atomic_ood_location(
        definitions,
        components,
        label=f"{label}.ood_records",
    )
    contract = {
        "ood_group": ood_group,
        "block": block,
        "components": list(components),
        "units": {"rho_bu_dry_ref": "kg/m^3", "eps_bed_cal_ref": "1"},
        "records": ood_records,
        "selection_rule": "complete_record_atomic",
        "provenance": copy.deepcopy(density["provenance"]),
        "ood_provenance": copy.deepcopy(density["ood_provenance"]),
        "natural_packing_support": {
            "lower": float(packing_support["lower"]),
            "upper": float(packing_support["upper"]),
        },
    }
    return density, rho_owner, {"eps_owner": eps_owner, "ood_contract": contract}


def _validate_oswin_record(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one immutable complete Modified-Oswin coefficient record."""
    oswin = resolve_value_record(value, sources=sources, label=label)
    _exact_keys(oswin, {"record_id", "components", "units", "equation", "temperature_variable", "provenance"}, label=label)
    components = _mapping(oswin["components"], label=f"{label}.components")
    units = _mapping(oswin["units"], label=f"{label}.units")
    names = ("A_osw", "B_osw", "C_osw")
    if tuple(components) != names or tuple(units) != names or units != {"A_osw": "1", "B_osw": "1/K", "C_osw": "1"}:
        message = f"{label} must contain exact A_osw/B_osw/C_osw components and units."
        raise ValueError(message)
    values = {name: _finite(components[name], label=f"{label}.components.{name}") for name in names}
    if values["A_osw"] <= 0 or values["C_osw"] <= 0:
        message = f"{label} has nonpositive A_osw or C_osw."
        raise ValueError(message)
    if not isinstance(oswin["record_id"], str) or not oswin["record_id"]:
        message = f"{label}.record_id must be non-empty text."
        raise ValueError(message)
    owner = {
        "sets": [{"id": oswin["record_id"], "values": values}],
        "provenance": copy.deepcopy(oswin["provenance"]),
    }
    oswin["components"] = values
    return oswin, owner


def _validate_kinetics_record(
    value: Any,
    *,
    definitions: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate natural component supports and complete kinetics OOD records."""
    kinetics = resolve_value_record(value, sources=sources, label=label)
    if "ood_provenance" not in kinetics:
        message = f"{label}.ood_provenance is required."
        raise ValueError(message)
    kinetics["ood_provenance"] = _resolve_synthetic_ood_provenance(kinetics["ood_provenance"], sources=sources, label=f"{label}.ood_provenance")
    _exact_keys(kinetics, {"record_id", "components", "ood_records", "selection_rule", "provenance", "ood_provenance"}, label=label)
    if kinetics["selection_rule"] != "complete_record_atomic":
        message = f"{label}.selection_rule must be complete_record_atomic."
        raise ValueError(message)
    components = _mapping(kinetics["components"], label=f"{label}.components")
    expected = ("r_surf_0", "r_int_surf", "f_surf")
    if tuple(components) != expected:
        message = f"{label}.components must be exactly {list(expected)}."
        raise ValueError(message)
    owners: dict[str, dict[str, Any]] = {}
    normalized_components: dict[str, Any] = {}
    for name in expected:
        component = _mapping(components[name], label=f"{label}.components.{name}")
        _exact_keys(component, {"nominal", "support", "transform"}, label=f"{label}.components.{name}")
        support = _mapping(component["support"], label=f"{label}.components.{name}.support")
        _exact_keys(support, {"lower", "upper"}, label=f"{label}.components.{name}.support")
        lower = _finite(support["lower"], label=f"{label}.components.{name}.support.lower")
        upper = _finite(support["upper"], label=f"{label}.components.{name}.support.upper")
        nominal = _finite(component["nominal"], label=f"{label}.components.{name}.nominal")
        if not lower <= nominal <= upper:
            message = f"{label}.components.{name} support does not contain its nominal."
            raise ValueError(message)
        normalized_components[name] = {"nominal": nominal, "support": {"lower": lower, "upper": upper}, "transform": component["transform"]}
        owners[name] = {
            "lower": lower,
            "upper": upper,
            "nominal": nominal,
            "distribution": f"uniform_in_{component['transform']}_transform_space",
            "provenance": copy.deepcopy(kinetics["provenance"]),
        }
    raw_records = kinetics["ood_records"]
    if not isinstance(raw_records, list):
        message = f"{label}.ood_records must be a list."
        raise TypeError(message)
    records: list[dict[str, Any]] = []
    expected_ids = ("slow_internal_limited", "fast_surface_exposed")
    if len(raw_records) != len(expected_ids):
        message = f"{label}.ood_records must contain exactly {list(expected_ids)}."
        raise ValueError(message)
    identities: set[str] = set()
    for index, raw in enumerate(raw_records):
        item_label = f"{label}.ood_records[{index}]"
        item = _mapping(raw, label=item_label)
        _exact_keys(item, {"record_id", *expected}, label=item_label)
        identity = item.pop("record_id")
        if identity != expected_ids[index]:
            message = f"{item_label}.record_id must be {expected_ids[index]!r}."
            raise ValueError(message)
        if not isinstance(identity, str) or not identity or identity in identities:
            message = f"{item_label}.record_id must be unique non-empty text."
            raise ValueError(message)
        identities.add(identity)
        values = {name: _finite(item[name], label=f"{item_label}.{name}") for name in expected}
        if normalized_components["r_surf_0"]["support"]["lower"] <= values["r_surf_0"] <= normalized_components["r_surf_0"]["support"]["upper"]:
            message = f"{item_label}.r_surf_0 must be disjoint from its natural support."
            raise ValueError(message)
        multipliers = (0.25, 0.5, -0.15) if index == 0 else (3.5, 1.5, 0.15)
        expected_values = {
            "r_surf_0": multipliers[0] * normalized_components["r_surf_0"]["nominal"],
            "r_int_surf": multipliers[1] * normalized_components["r_int_surf"]["nominal"],
            "f_surf": normalized_components["f_surf"]["nominal"] + multipliers[2],
        }
        if any(not math.isclose(values[name], expected_values[name], rel_tol=0.0, abs_tol=1e-14) for name in expected):
            message = f"{item_label} violates the required common kinetics OOD design equation."
            raise ValueError(message)
        records.append({"id": identity, "values": values, "metadata": {}})
    kinetics["components"] = normalized_components
    kinetics["ood_records"] = records
    ood_group, block = _atomic_ood_location(
        definitions,
        expected,
        label=f"{label}.ood_records",
    )
    contract = {
        "ood_group": ood_group,
        "block": block,
        "components": list(expected),
        "units": {"r_surf_0": "1/s", "r_int_surf": "1", "f_surf": "1"},
        "records": records,
        "selection_rule": "complete_record_atomic",
        "provenance": copy.deepcopy(kinetics["provenance"]),
        "ood_provenance": copy.deepcopy(kinetics["ood_provenance"]),
    }
    return kinetics, owners, contract


def _validate_material_ood_inventory(registry: Mapping[str, Mapping[str, Any]]) -> None:
    """Require complete scalar OOD roles with their exact directional inventory."""
    expected = {
        "kappa_mean": (True, True),
        "k_gr": (True, True),
        "cp_gr_dry": (True, True),
        "initial_moisture.mean_db": (False, True),
        "initial_moisture.amplitude_db": (False, True),
    }
    for name, (requires_lower, requires_upper) in expected.items():
        entry = registry.get(name)
        if not isinstance(entry, Mapping):
            message = f"Material scalar OOD role {name!r} is missing from the resolved registry."
            raise TypeError(message)
        tails = entry.get("ood")
        if not isinstance(tails, list):
            message = f"Material scalar OOD role {name!r} must declare interval tails."
            raise TypeError(message)
        lower, upper = float(entry["lower"]), float(entry["upper"])
        directions = [
            "lower" if float(tail["upper"]) < lower else "upper" if float(tail["lower"]) > upper else "overlap"
            for tail in tails
            if isinstance(tail, Mapping) and {"lower", "upper"}.issubset(tail)
        ]
        expected_directions = (["lower"] if requires_lower else []) + (["upper"] if requires_upper else [])
        if directions != expected_directions:
            message = f"Material scalar OOD role {name!r} must declare exact directions {expected_directions}."
            raise ValueError(message)
        if not isinstance(entry.get("ood_provenance"), Mapping):
            message = f"Material scalar OOD role {name!r} requires OOD provenance."
            raise TypeError(message)


def _merge_registry(
    definitions: Mapping[str, Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge exact disjoint semantic and value owners."""
    unknown = sorted(set(owners).difference(definitions))
    if unknown:
        message = f"Scientific value owners contain undeclared parameters: {unknown}."
        raise ValueError(message)
    missing: list[str] = []
    merged: dict[str, dict[str, Any]] = {}
    for name, definition in definitions.items():
        entry = copy.deepcopy(dict(definition))
        additions = copy.deepcopy(dict(owners.get(name, {})))
        overlap = set(entry).intersection(additions)
        if overlap:
            message = f"Parameter {name!r} has duplicate semantic/value keys {sorted(overlap)}."
            raise ValueError(message)
        entry.update(additions)
        if entry.get("kind") != "derived" and name not in owners:
            missing.append(name)
        merged[name] = entry
    if missing:
        message = f"Scientific parameters have no value owner: {sorted(missing)}."
        raise ValueError(message)
    registry = registry_service.validate_parameter_registry(merged)
    validate_vp2_registry(registry)
    return registry


def _derived_provenance(
    name: str,
    registry: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, str]],
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic inherited provenance for one configured derivation rule."""
    source_refs: list[str] = []
    for source_name in registry[name]["sources"]:
        source_entry = registry.get(source_name)
        provenance = None if source_entry is None else source_entry.get("provenance")
        if isinstance(provenance, Mapping):
            for source_ref in provenance.get("source_refs", []):
                if isinstance(source_ref, str) and source_ref not in source_refs:
                    source_refs.append(source_ref)
    value = {
        "evidence": "derived",
        "source_refs": source_refs,
        "method": metadata[name]["description"],
        "verification": provenance_service.REPRODUCED_VERIFICATION,
    }
    return provenance_service.resolve_provenance(value, sources=sources, label=f"derived parameter {name} provenance")


def validate_profile_registry(
    registry: Mapping[str, Mapping[str, Any]],
    profile_id: str,
) -> dict[str, int]:
    """Validate one registry projection against authored profile applicability."""
    applicable = set(profile_parameter_names(registry, profile_id))
    inapplicable = sorted(set(registry).difference(applicable))
    if inapplicable:
        message = f"Profile {profile_id!r} registry contains inapplicable parameters {inapplicable}."
        raise ValueError(message)
    blocks = active_sampling_blocks(registry, profile_id)
    if not blocks:
        message = f"Profile {profile_id!r} registry has no active sampling block."
        raise ValueError(message)
    return sampling_block_dimensions(registry, blocks=blocks)


def validate_vp2_registry(registry: Mapping[str, Mapping[str, Any]]) -> str:
    """Validate the algorithm-bound coupled and derived parameter contracts."""
    expected_derived = {
        "bed.structure.fine_weight": ("complement_of_one", ("bed.structure.coarse_weight",)),
        "initial_moisture.structure.fine_weight": ("complement_of_one", ("initial_moisture.structure.coarse_weight",)),
        "T_init": ("copy", ("T_amb",)),
        "r_surf": ("copy", ("r_surf_0",)),
        "r_int": ("product", ("r_int_surf", "r_surf")),
    }
    required = {
        "rho_bu_dry_ref",
        "eps_bed_cal_ref",
        "schedule.component_weights",
        "oswin",
        *expected_derived,
    }
    missing = sorted(required.difference(registry))
    if missing:
        message = f"Registry is missing algorithm-bound parameters {missing}."
        raise ValueError(message)
    if registry["rho_bu_dry_ref"]["kind"] != "interval" or registry["eps_bed_cal_ref"]["kind"] != "fixed":
        message = "Natural density must sample rho_bu_dry_ref while eps_bed_cal_ref remains fixed."
        raise ValueError(message)
    if (
        registry["rho_bu_dry_ref"].get("atomic_record") != "density_calibration"
        or registry["eps_bed_cal_ref"].get("atomic_record") != "density_calibration"
    ):
        message = "Density components must declare one density_calibration atomic record."
        raise ValueError(message)
    simplex = registry["schedule.component_weights"]
    if (
        simplex["kind"] != "simplex"
        or tuple(simplex["components"]) != ("smooth", "event", "trend")
        or simplex.get("selection") != "truncated_dirichlet"
    ):
        message = "schedule.component_weights must use the supported ordered smooth/event/trend truncated-Dirichlet protocol."
        raise ValueError(message)
    oswin = registry["oswin"]
    if oswin["kind"] != "parameter_set" or tuple(oswin["components"]) != ("A_osw", "B_osw", "C_osw"):
        message = "oswin must be one complete A_osw/B_osw/C_osw parameter set."
        raise ValueError(message)
    for name, (derivation, source_names) in expected_derived.items():
        entry = registry[name]
        if entry["kind"] != "derived" or entry["derivation"] != derivation or tuple(entry["sources"]) != source_names:
            message = f"Derived parameter {name!r} violates its supported rule."
            raise ValueError(message)
    return "sampled_rho_fixed_reference_epsilon"


def sampling_block_dimensions(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    blocks: tuple[str, ...] | None = None,
    block_parameters: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, int]:
    """Return numerical dimensions for selected registry-owned blocks."""
    membership = sampling_blocks(registry) if block_parameters is None else block_parameters
    selected = tuple(membership) if blocks is None else blocks
    if not selected or any(block not in membership for block in selected):
        message = f"Unknown or empty active sampling blocks {selected}."
        raise ValueError(message)
    return {block: sum(registry_service.effective_dimension(registry[name]) for name in membership[block]) for block in selected}


def sampling_coordinate_labels(
    registry: Mapping[str, Mapping[str, Any]],
    profile_id: str,
    *,
    block_parameters: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Return profile-qualified numerical coordinate labels in configured order."""
    membership = sampling_blocks(registry) if block_parameters is None else block_parameters
    labels: list[str] = []
    for block in active_sampling_blocks(registry, profile_id):
        for name in membership[block]:
            dimension = registry_service.effective_dimension(registry[name])
            if dimension == 1:
                labels.append(name)
            else:
                labels.extend(f"{name}[{index}]" for index in range(1, dimension + 1))
    return tuple(labels)


def project_material_for_profile(
    material: Mapping[str, Any],
    profile_id: str,
    *,
    sampling_block_contract: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Return only registry-declared material science applicable to one profile."""
    full_registry = material["parameter_registry"]
    active_names = profile_parameter_names(full_registry, profile_id)
    active_name_set = set(active_names)
    projected_registry = {name: copy.deepcopy(entry) for name, entry in full_registry.items() if name in active_name_set}
    blocks = tuple(block for block, names in sampling_block_contract.items() if active_name_set.intersection(names))
    active_block_parameters = {block: tuple(name for name in sampling_block_contract[block] if name in active_name_set) for block in blocks}
    extra_provenance = {"A_osw", "B_osw", "C_osw"} if "oswin" in active_name_set else set()
    projected = {
        "material_family": material["material_family"],
        "material_scope": copy.deepcopy(material["material_scope"]),
        "packing_porosity_mean_support": copy.deepcopy(material["packing_porosity_mean_support"]),
        "porosity_coupling": copy.deepcopy(material["porosity_coupling"]),
        "parameter_registry": projected_registry,
        "effective_parameter_provenance": {
            name: copy.deepcopy(value)
            for name, value in material["effective_parameter_provenance"].items()
            if name in active_name_set or name in extra_provenance
        },
        "active_sampling_blocks": list(blocks),
        "active_coordinate_names": [name for block in blocks for name in active_block_parameters[block]],
    }
    if profile_id == "transient_drying":
        projected.update(
            {
                "initial_moisture_bounds": copy.deepcopy(material["initial_moisture_bounds"]),
                "initial_moisture_field_constraint": copy.deepcopy(material["initial_moisture_field_constraint"]),
                "coupled_ood_records": copy.deepcopy(material["coupled_ood_records"]),
                "atomic_records": copy.deepcopy(material["atomic_records"]),
            }
        )
    else:
        projected["coupled_ood_records"] = {}
        projected["atomic_records"] = {}
    return projected


def initial_moisture_generation_bounds(
    material_contract: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    active_ood_unit: str | None,
) -> dict[str, Any]:
    """Resolve the natural envelope or supplied high-tail target guard."""
    natural = _mapping(material_contract["initial_moisture_bounds"], label="initial-moisture natural bounds")
    constraint = _mapping(
        material_contract["initial_moisture_field_constraint"],
        label="initial-moisture field constraint",
    )
    mean = _finite(values["initial_moisture.mean_db"], label="initial_moisture.mean_db")
    amplitude = _finite(values["initial_moisture.amplitude_db"], label="initial_moisture.amplitude_db")
    natural_lower = _finite(natural["lower"], label="initial-moisture natural lower bound")
    natural_upper = _finite(natural["upper"], label="initial-moisture natural upper bound")
    allow_departure = active_ood_unit in INITIAL_MOISTURE_LEVEL_PARAMETERS
    lower = _finite(constraint["minimum_db"], label="initial-moisture target-separation minimum") if allow_departure else natural_lower
    upper = mean + amplitude if allow_departure else natural_upper
    maximum_amplitude = mean - lower
    if not allow_departure:
        maximum_amplitude = min(maximum_amplitude, upper - mean)
    if amplitude < 0 or amplitude > maximum_amplitude or lower >= upper:
        message = (
            f"Initial-moisture amplitude {amplitude} exceeds the no-clipping bound "
            f"{maximum_amplitude} for material {material_contract['material_family']!r}."
        )
        raise ValueError(message)
    return {
        "lower": lower,
        "upper": upper,
        "natural_lower": natural_lower,
        "natural_upper": natural_upper,
        "natural_support_departure_allowed": allow_departure,
        "active_ood_unit": active_ood_unit,
        "target_separation_constraint": copy.deepcopy(constraint),
    }


def resolve_material_definition(
    definitions: Mapping[str, Mapping[str, Any]],
    registry_metadata: Mapping[str, Mapping[str, str]],
    common_values: Any,
    operation_values: Any,
    material_config: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve one compact role-neutral material file with complete provenance."""
    material = _mapping(material_config, label="material configuration")
    expected = {
        "schema_kind",
        "schema_version",
        "material_family",
        "material_scope",
        "permeability",
        "packing_porosity_mean_support",
        "density_calibration",
        "thermal_properties",
        "initial_moisture",
        "target_moisture",
        "oswin",
        "two_compartment_kinetics",
    }
    _exact_keys(material, expected, label="material configuration")
    family = validate_material_family(material["material_family"])
    if material["schema_kind"] != "generation_material" or material["schema_version"] != 1:
        message = "Unsupported role-neutral material configuration schema."
        raise ValueError(message)
    scope = _validate_material_scope(material["material_scope"], label=f"material {family}.material_scope")
    packing = _validate_packing_support(
        material["packing_porosity_mean_support"],
        sources=sources,
        label=f"material {family}.packing_porosity_mean_support",
    )
    owners: dict[str, Mapping[str, Any]] = {}
    schedule_record_id: str | None = None
    for source_name, raw_owner in (("common", common_values), ("operation", operation_values)):
        raw_values = _mapping(raw_owner, label=f"{source_name} parameter values")
        owner: dict[str, dict[str, Any]] = {}
        for name, raw_entry in raw_values.items():
            entry = resolve_value_record(
                raw_entry,
                sources=sources,
                label=f"{source_name} parameter values.{name}",
            )
            if name == "schedule.component_weights":
                record_id = entry.pop("record_id", None)
                if source_name != "operation" or not isinstance(record_id, str) or not record_id:
                    message = "The operation-owned schedule simplex must declare one non-empty record_id."
                    raise ValueError(message)
                schedule_record_id = record_id
            owner[name] = entry
        overlap = set(owners).intersection(owner)
        if overlap:
            message = f"Scientific parameter values have duplicate owners: {sorted(overlap)}."
            raise ValueError(message)
        owners.update(owner)
    if schedule_record_id is None:
        message = "Operation parameter values must own the schedule simplex atomic record."
        raise ValueError(message)
    owners["kappa_mean"] = _support_record(material["permeability"], sources=sources, label=f"material {family}.permeability")
    thermal = _mapping(material["thermal_properties"], label=f"material {family}.thermal_properties")
    _exact_keys(thermal, {"k_gr", "cp_gr_dry"}, label=f"material {family}.thermal_properties")
    owners["k_gr"] = _support_record(thermal["k_gr"], sources=sources, label=f"material {family}.thermal_properties.k_gr")
    owners["cp_gr_dry"] = _support_record(thermal["cp_gr_dry"], sources=sources, label=f"material {family}.thermal_properties.cp_gr_dry")
    initial = _mapping(material["initial_moisture"], label=f"material {family}.initial_moisture")
    _exact_keys(
        initial,
        {"mean_db", "amplitude_db", "field_support", "margin_above_target_db"},
        label=f"material {family}.initial_moisture",
    )
    owners["initial_moisture.mean_db"] = _support_record(initial["mean_db"], sources=sources, label=f"material {family}.initial_moisture.mean_db")
    owners["initial_moisture.amplitude_db"] = _support_record(
        initial["amplitude_db"],
        sources=sources,
        label=f"material {family}.initial_moisture.amplitude_db",
    )
    field_support = resolve_value_record(initial["field_support"], sources=sources, label=f"material {family}.initial_moisture.field_support")
    _exact_keys(field_support, {"lower", "upper", "unit", "provenance"}, label=f"material {family}.initial_moisture.field_support")
    initial_bounds = {
        "lower": _finite(field_support["lower"], label=f"material {family}.initial_moisture.field_support.lower"),
        "upper": _finite(field_support["upper"], label=f"material {family}.initial_moisture.field_support.upper"),
    }
    if field_support["unit"] != "kg/kg" or initial_bounds["lower"] >= initial_bounds["upper"]:
        message = f"Material {family!r} initial-moisture field support is invalid."
        raise ValueError(message)
    density, rho_owner, density_parts = _validate_density_record(
        material["density_calibration"],
        packing_support=packing,
        definitions=definitions,
        sources=sources,
        label=f"material {family}.density_calibration",
    )
    owners["rho_bu_dry_ref"] = rho_owner
    owners["eps_bed_cal_ref"] = density_parts["eps_owner"]
    oswin, oswin_owner = _validate_oswin_record(material["oswin"], sources=sources, label=f"material {family}.oswin")
    owners["oswin"] = oswin_owner
    kinetics, kinetics_owners, kinetics_contract = _validate_kinetics_record(
        material["two_compartment_kinetics"],
        definitions=definitions,
        sources=sources,
        label=f"material {family}.two_compartment_kinetics",
    )
    owners.update(kinetics_owners)
    target = resolve_value_record(material["target_moisture"], sources=sources, label=f"material {family}.target_moisture")
    _exact_keys(
        target,
        {"target_moisture_wb", "provenance"},
        label=f"material {family}.target_moisture",
    )
    target_wb = _finite(
        target["target_moisture_wb"],
        label=f"material {family}.target_moisture.target_moisture_wb",
    )
    if not 0 < target_wb < 1:
        message = f"Material {family!r} target moisture must lie inside (0, 1)."
        raise ValueError(message)
    if practical_target_moisture_wb(target_wb) != target_wb:
        message = f"Material {family!r} target moisture must use a whole wet-basis percentage point."
        raise ValueError(message)
    target_db = target_wb / (1.0 - target_wb)
    target = {
        "target_moisture_wb": target_wb,
        "provenance": copy.deepcopy(target["provenance"]),
    }
    margin_above_target_db = _finite(
        initial["margin_above_target_db"],
        label=f"material {family}.initial_moisture.margin_above_target_db",
    )
    field_minimum_db = target_db + margin_above_target_db
    if margin_above_target_db <= 0 or field_minimum_db <= 0:
        message = f"Material {family!r} initial-moisture target margin must be positive."
        raise ValueError(message)
    expected_field_constraint = f"min(X_0_db_field) >= {field_minimum_db:.8f} kg/kg"
    field_constraint = {
        "authored_expression": expected_field_constraint,
        "minimum_db": field_minimum_db,
        "margin_above_target_db": margin_above_target_db,
        "unit": "kg/kg",
        "derivation": {
            "kind": "derived_from_configured_target",
            "verification": "mathematically_reproduced",
            "inputs": ["X_target_db", f"{margin_above_target_db:g} kg/kg"],
            "formula_or_method": f"minimum_db = X_target_db + {margin_above_target_db:g} kg/kg",
        },
    }
    owners["X_target_wb"] = {"value": target_wb, "nominal": target_wb, "provenance": copy.deepcopy(target["provenance"])}
    registry = _merge_registry(definitions, owners)
    _validate_material_ood_inventory(registry)
    kappa_entry = registry["kappa_mean"]
    coupling = porosity_service.resolve_porosity_coupling(
        material_family=family,
        material_kappa_nominal=float(kappa_entry["nominal"]),
        eps_bed_cal_ref=float(registry["eps_bed_cal_ref"]["value"]),
        authored_permeability_support=kappa_entry,
        packing_porosity_mean_support=packing,
        eps_min_global=float(registry["eps_min_global"]["value"]),
        eps_max_global=float(registry["eps_max_global"]["value"]),
        authored_kappa_ood=kappa_entry.get("ood"),
    )
    kappa_entry["lower"] = coupling["effective_joint_permeability_support"]["lower"]
    kappa_entry["upper"] = coupling["effective_joint_permeability_support"]["upper"]
    effective: dict[str, Any] = {}
    for name, entry in registry.items():
        entry_provenance = entry.get("provenance")
        if isinstance(entry_provenance, Mapping):
            effective[name] = copy.deepcopy(dict(entry_provenance))
        elif entry.get("kind") == "derived":
            derived = _derived_provenance(name, registry, registry_metadata, sources)
            registry[name]["provenance"] = copy.deepcopy(derived)
            effective[name] = derived
        else:
            message = f"Resolved parameter {name!r} has no effective provenance."
            raise ValueError(message)
    for component in ("A_osw", "B_osw", "C_osw"):
        effective[component] = copy.deepcopy(effective["oswin"])
    return {
        "material_family": family,
        "material_scope": scope,
        "packing_porosity_mean_support": packing,
        "porosity_coupling": coupling,
        "initial_moisture_bounds": initial_bounds,
        "initial_moisture_field_constraint": field_constraint,
        "parameter_registry": registry,
        "effective_parameter_provenance": effective,
        "coupled_ood_records": {
            "density_calibration": density_parts["ood_contract"],
            "two_compartment_kinetics": kinetics_contract,
        },
        "atomic_records": {
            "density_calibration": density,
            "oswin": oswin,
            "two_compartment_kinetics": kinetics,
            "schedule_simplex": {
                "record_id": schedule_record_id,
                "provenance": copy.deepcopy(registry["schedule.component_weights"]["provenance"]),
            },
        },
        "target_moisture": target,
        "initial_moisture_field_provenance": field_support["provenance"],
    }
