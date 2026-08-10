"""
===============================================================================
generation_materials.py
===============================================================================
Resolve compact role-neutral material records into typed scientific registries.
Responsibilities:
  - Define exact material identifiers and profile-specific coordinate contracts
  - Merge disjoint common, operation, and family-specific scientific owners
  - Resolve atomic density, Oswin, and kinetics records with effective provenance
Design principles:
  - Material files contain family science and no campaign roles or shared controls
  - Density calibration is always sampled-rho/fixed-reference-epsilon in natural data
  - Every effective parameter exposes its supplied source and interpretation
This module does NOT:
  - Assign campaign roles, counts, seeds, dataset membership, or runtime mappings
  - Invent scientific values, derivations, sources, or compatibility modes
===============================================================================
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from . import generation_provenance as provenance_service
from . import generation_registry as registry_service

MATERIAL_FAMILIES: Final = (
    "lentil",
    "chickpea",
    "kidney_bean",
    "field_pea",
    "rapeseed",
    "sunflower_seed",
)
VP2_DECISION_ARTIFACT: Final = "VP2_Parameter_Decisions.yaml"
VP2_DECISION_SCHEMA_VERSION: Final = "1.1.0"
VP2_DECISION_SHA256: Final = "774ce0e39bf989ad77b5fe80e37c364f46ff83b3c6be1bd7410ea4c72d7269f5"
_SIMPLEX_MINIMUM_EACH: Final = 0.05
_SIMPLEX_MAXIMUM_EACH: Final = 0.8
STEADY_DIMENSION: Final = 28
TRANSIENT_DIMENSION: Final = 54

AIRFLOW_PARAMETERS: Final = (
    "kappa_mean",
    "kappa_cv",
    "bed.structure.coarse_len_rel",
    "bed.structure.fine_len_rel",
    "bed.structure.coarse_weight",
    "bed.structure.cross_scale_corr",
    "bed.structure.fine_ani_x",
    "bed.structure.fine_ani_y",
    "bed.perturbations.amplitude",
    "bed.perturbations.granularity",
    "bed.perturbations.sign_bias",
    "permeability.anisotropy.max_ratio",
    "permeability.anisotropy.exponent",
    "permeability.anisotropy.strength",
    "permeability.orientation.jitter",
    "permeability.orientation.smooth_len_rel",
    "porosity.kc_anchor_factor",
    "porosity.smooth_len_rel",
    "porosity.texture_amp",
    "pressure_bc.mean",
    "pressure_bc.sin_amp",
    "pressure_bc.sin_freq",
    "pressure_bc.sin_phase",
    "pressure_bc.gauss_count",
    "pressure_bc.gauss_amp",
    "pressure_bc.gauss_width",
    "pressure_bc.gauss_jitter",
    "pressure_bc.linear_amp",
)
INITIAL_MOISTURE_PARAMETERS: Final = (
    "initial_moisture.mean_db",
    "initial_moisture.amplitude_db",
    "initial_moisture.structure.coarse_len_rel",
    "initial_moisture.structure.fine_len_rel",
    "initial_moisture.structure.coarse_weight",
    "initial_moisture.structure.cross_scale_corr",
    "initial_moisture.structure.fine_ani_x",
    "initial_moisture.structure.fine_ani_y",
)
OPERATION_PARAMETERS: Final = (
    "T_in_base",
    "T_in_amp",
    "omega_in_base",
    "omega_in_amp",
    "schedule.corr",
    "schedule.timescale_rel",
    "schedule.component_weights",
    "schedule.event_count",
    "schedule.event_duration_rel",
    "schedule.event_width_rel",
    "T_amb",
)
MATERIAL_PROPERTY_PARAMETERS: Final = (
    "rho_bu_dry_ref",
    "k_gr",
    "cp_gr_dry",
    "r_surf_0",
    "r_int_surf",
    "f_surf",
)
SAMPLING_BLOCKS: Final = MappingProxyType(
    {
        "airflow": AIRFLOW_PARAMETERS,
        "initial_moisture": INITIAL_MOISTURE_PARAMETERS,
        "operation": OPERATION_PARAMETERS,
        "material_properties": MATERIAL_PROPERTY_PARAMETERS,
    }
)
SAMPLING_BLOCK_DIMENSIONS: Final = MappingProxyType({"airflow": 28, "initial_moisture": 8, "operation": 12, "material_properties": 6})
PROFILE_SAMPLING_BLOCKS: Final = MappingProxyType(
    {
        "steady_flow": ("airflow",),
        "transient_drying": tuple(SAMPLING_BLOCKS),
    }
)
PROFILE_OOD_GROUPS: Final = MappingProxyType(
    {
        "steady_flow": ("bed", "operation"),
        "transient_drying": ("bed", "operation", "initial_moisture", "material_properties"),
    }
)
OOD_GROUPS: Final = PROFILE_OOD_GROUPS["transient_drying"]
DERIVED_PARAMETERS: Final = (
    "bed.structure.fine_weight",
    "initial_moisture.structure.fine_weight",
    "T_init",
    "r_surf",
    "r_int",
    "T_in_ref",
)
POROSITY_GENERATOR_PARAMETERS: Final = (
    "porosity.kc_anchor_factor",
    "porosity.smooth_len_rel",
    "porosity.texture_amp",
)
INITIAL_MOISTURE_LEVEL_PARAMETERS: Final = (
    "initial_moisture.mean_db",
    "initial_moisture.amplitude_db",
)
SUPPORT_PARAMETERS: Final = (
    "eps_min_global",
    "eps_max_global",
    "eps_bed_cal_ref",
    "X_target_wb",
    "oswin",
    *DERIVED_PARAMETERS,
)
EXPECTED_PARAMETERS: Final = frozenset(name for names in SAMPLING_BLOCKS.values() for name in names) | frozenset(SUPPORT_PARAMETERS)
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
        "market_class": frozenset(
            {
                "red_or_brown_dry_lentil",
                "kabuli",
                "red_kidney",
                "yellow_or_green_field_pea",
                "canola_quality_rapeseed",
                "high_oleic_oil_type",
            }
        ),
        "product_form": frozenset({"whole_seed", "whole_achene"}),
        "coat_or_hull_state": frozenset({"intact", "hull_intact"}),
    }
)
_COUPLED_COMPONENTS: Final = MappingProxyType(
    {
        "density_calibration": ("rho_bu_dry_ref", "eps_bed_cal_ref"),
        "two_compartment_kinetics": ("r_surf_0", "r_int_surf", "f_surf"),
    }
)
_CORRELATION_REPORT_SYMBOLS: Final = MappingProxyType(
    {
        "bed.structure.cross_scale_corr": r"\rho_b",
        "initial_moisture.structure.cross_scale_corr": r"\rho_X",
        "schedule.corr": r"\rho_{T,\omega}",
    }
)


def available_material_families() -> tuple[str, ...]:
    """Return exact role-neutral material identifiers in canonical order."""
    return MATERIAL_FAMILIES


def active_sampling_blocks(profile_id: str) -> tuple[str, ...]:
    """Return exact numerical blocks consumed by one simulation profile."""
    try:
        return PROFILE_SAMPLING_BLOCKS[profile_id]
    except KeyError as error:
        message = f"Unknown simulation profile {profile_id!r}."
        raise ValueError(message) from error


def active_ood_groups(profile_id: str) -> tuple[str, ...]:
    """Return exact parameter-OOD groups consumed by one simulation profile."""
    try:
        return PROFILE_OOD_GROUPS[profile_id]
    except KeyError as error:
        message = f"Unknown simulation profile {profile_id!r}."
        raise ValueError(message) from error


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


def validate_decision_source(value: Any, *, label: str) -> dict[str, str]:
    """Validate the immutable handoff decision identity copied into production YAML."""
    decision = _mapping(value, label=label)
    _exact_keys(decision, {"artifact", "schema_version", "sha256"}, label=label)
    expected = {
        "artifact": VP2_DECISION_ARTIFACT,
        "schema_version": VP2_DECISION_SCHEMA_VERSION,
        "sha256": VP2_DECISION_SHA256,
    }
    if decision != expected:
        message = f"{label} must bind the validated VP2 decision artifact identity {expected}."
        raise ValueError(message)
    return {key: str(item) for key, item in decision.items()}


def _semantic_entry(value: Any, *, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Separate executable semantics from catalogue-only labels."""
    entry = _mapping(value, label=label)
    metadata: dict[str, str] = {}
    for key in _REGISTRY_METADATA_KEYS:
        item = entry.pop(key, None)
        if not isinstance(item, str) or not item:
            message = f"{label}.{key} must be non-empty text."
            raise ValueError(message)
        metadata[key] = item
    return entry, metadata


def validate_semantic_registry(value: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """Validate the source-owned parameter-semantic inventory."""
    config = _mapping(value, label="generation parameter registry")
    _exact_keys(config, {"schema_kind", "schema_version", "decision_source", "parameters"}, label="generation parameter registry")
    if config["schema_kind"] != "generation_parameter_registry" or config["schema_version"] != 1:
        message = "Unsupported generation parameter-registry schema."
        raise ValueError(message)
    validate_decision_source(config["decision_source"], label="generation parameter registry.decision_source")
    raw = _mapping(config["parameters"], label="generation parameter registry.parameters")
    if set(raw) != EXPECTED_PARAMETERS:
        missing = sorted(EXPECTED_PARAMETERS.difference(raw))
        unknown = sorted(set(raw).difference(EXPECTED_PARAMETERS))
        message = f"Parameter semantics mismatch: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    definitions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, str]] = {}
    symbols: dict[str, str] = {}
    for name, item in raw.items():
        definitions[name], metadata[name] = _semantic_entry(item, label=f"parameters.{name}")
        symbol = metadata[name]["report_symbol"]
        if symbol in symbols:
            message = f"Report symbol {symbol!r} is assigned to both {symbols[symbol]!r} and {name!r}."
            raise ValueError(message)
        symbols[symbol] = name
    for name, expected in _CORRELATION_REPORT_SYMBOLS.items():
        if metadata[name]["report_symbol"] != expected:
            message = f"Correlation parameter {name!r} must use report symbol {expected!r}."
            raise ValueError(message)
    return definitions, metadata


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


def _support_record(
    value: Any,
    *,
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
    allow_ood: bool = True,
) -> dict[str, Any]:
    """Flatten one compact nominal/support record for registry merging."""
    record = resolve_value_record(value, sources=sources, label=label)
    required = {"nominal", "support", "transform", "distribution", "provenance"}
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


def _validate_density_record(
    value: Any,
    *,
    packing_support: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate natural sampled density, fixed reference epsilon, and atomic OOD records."""
    density = resolve_value_record(value, sources=sources, label=label)
    _exact_keys(
        density,
        {"record_id", "reference", "rho_bu_dry_ref_support", "ood_records", "selection_rule", "provenance"},
        label=label,
    )
    if not isinstance(density["record_id"], str) or not density["record_id"]:
        message = f"{label}.record_id must be non-empty text."
        raise ValueError(message)
    if density["selection_rule"] != "complete_record_atomic":
        message = f"{label}.selection_rule must be complete_record_atomic."
        raise ValueError(message)
    reference = _mapping(density["reference"], label=f"{label}.reference")
    _exact_keys(reference, {"rho_bu_dry_ref", "eps_bed_cal_ref", "inferred_rho_particle_dry"}, label=f"{label}.reference")
    rho = _finite(reference["rho_bu_dry_ref"], label=f"{label}.reference.rho_bu_dry_ref")
    eps = _finite(reference["eps_bed_cal_ref"], label=f"{label}.reference.eps_bed_cal_ref")
    particle = _finite(reference["inferred_rho_particle_dry"], label=f"{label}.reference.inferred_rho_particle_dry")
    if rho <= 0 or not 0 < eps < 1 or not math.isclose(rho / (1.0 - eps), particle, rel_tol=2e-9, abs_tol=2e-6):
        message = f"{label}.reference is internally inconsistent."
        raise ValueError(message)
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
    identities: set[str] = set()
    for index, raw in enumerate(raw_ood):
        item_label = f"{label}.ood_records[{index}]"
        item = _mapping(raw, label=item_label)
        _exact_keys(
            item,
            {"record_id", "rho_bu_dry_ref", "eps_bed_cal_ref"},
            optional={"supplied_status", "supplied_derivation"},
            label=item_label,
        )
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
        metadata = {key: copy.deepcopy(item[key]) for key in ("supplied_status", "supplied_derivation") if key in item}
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
    contract = {
        "ood_group": "material_properties",
        "block": "material_properties",
        "components": ["rho_bu_dry_ref", "eps_bed_cal_ref"],
        "units": {"rho_bu_dry_ref": "kg/m^3", "eps_bed_cal_ref": "1"},
        "records": ood_records,
        "selection_rule": "complete_record_atomic",
        "provenance": copy.deepcopy(density["provenance"]),
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
    sources: Mapping[str, Mapping[str, Any]],
    label: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Validate natural component supports and complete kinetics OOD records."""
    kinetics = resolve_value_record(value, sources=sources, label=label)
    _exact_keys(kinetics, {"record_id", "components", "ood_records", "selection_rule", "provenance"}, label=label)
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
    identities: set[str] = set()
    for index, raw in enumerate(raw_records):
        item_label = f"{label}.ood_records[{index}]"
        item = _mapping(raw, label=item_label)
        _exact_keys(item, {"record_id", *expected}, label=item_label)
        identity = item.pop("record_id")
        if not isinstance(identity, str) or not identity or identity in identities:
            message = f"{item_label}.record_id must be unique non-empty text."
            raise ValueError(message)
        identities.add(identity)
        values = {name: _finite(item[name], label=f"{item_label}.{name}") for name in expected}
        if normalized_components["r_surf_0"]["support"]["lower"] <= values["r_surf_0"] <= normalized_components["r_surf_0"]["support"]["upper"]:
            message = f"{item_label}.r_surf_0 must be disjoint from its natural support."
            raise ValueError(message)
        records.append({"id": identity, "values": values, "metadata": {}})
    kinetics["components"] = normalized_components
    kinetics["ood_records"] = records
    contract = {
        "ood_group": "material_properties",
        "block": "material_properties",
        "components": list(expected),
        "units": {"r_surf_0": "1/s", "r_int_surf": "1", "f_surf": "1"},
        "records": records,
        "selection_rule": "complete_record_atomic",
        "provenance": copy.deepcopy(kinetics["provenance"]),
    }
    return kinetics, owners, contract


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
        if name not in DERIVED_PARAMETERS and name not in owners:
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
    """Build deterministic inherited provenance for one supplied derivation rule."""
    source_refs: list[str] = []
    for source_name in registry[name]["sources"]:
        source_entry = registry.get(source_name)
        provenance = None if source_entry is None else source_entry.get("provenance")
        if isinstance(provenance, Mapping):
            for source_ref in provenance.get("source_refs", []):
                if isinstance(source_ref, str) and source_ref not in source_refs:
                    source_refs.append(source_ref)
    if "vp2_decision_contract" not in source_refs:
        source_refs.append("vp2_decision_contract")
    value = {
        "source_refs": source_refs,
        "status": "derived",
        "derivation": {
            "kind": "derived_from_configured_value",
            "origin": "supplied_by_handoff",
            "verification": "declared_only",
            "description": metadata[name]["description"],
        },
        "confidence": "inherits_configured_source_confidence",
        "validity": {"equation_or_method": metadata[name]["description"]},
    }
    return provenance_service.resolve_provenance(value, sources=sources, label=f"derived parameter {name} provenance")


def validate_profile_registry(
    registry: Mapping[str, Mapping[str, Any]],
    profile_id: str,
) -> dict[str, int]:
    """Validate one exact profile projection of the canonical registry."""
    blocks = active_sampling_blocks(profile_id)
    expected = {name for block in blocks for name in SAMPLING_BLOCKS[block]}
    if profile_id == "steady_flow":
        expected.update({"bed.structure.fine_weight", "eps_min_global", "eps_max_global", "eps_bed_cal_ref"})
    else:
        expected.update(SUPPORT_PARAMETERS)
    if set(registry) != expected:
        missing = sorted(expected.difference(registry))
        unknown = sorted(set(registry).difference(expected))
        message = f"Profile {profile_id!r} registry projection mismatch: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    for name, entry in registry.items():
        applicability = entry.get("profile_applicability")
        if not isinstance(applicability, list) or profile_id not in applicability:
            message = f"Registry parameter {name!r} is inapplicable to profile {profile_id!r}."
            raise ValueError(message)
    dimensions = sampling_block_dimensions(registry, blocks=blocks)
    expected_dimensions = {block: SAMPLING_BLOCK_DIMENSIONS[block] for block in blocks}
    if dimensions != expected_dimensions:
        message = f"Profile {profile_id!r} block dimensions must be {expected_dimensions}, got {dimensions}."
        raise ValueError(message)
    expected_total = STEADY_DIMENSION if profile_id == "steady_flow" else TRANSIENT_DIMENSION
    if sum(dimensions.values()) != expected_total:
        message = f"Profile {profile_id!r} effective dimension changed."
        raise ValueError(message)
    return dimensions


def validate_vp2_registry(registry: Mapping[str, Mapping[str, Any]]) -> str:
    """Validate the unique 28D/54D profile contract and atomic memberships."""
    if set(registry) != EXPECTED_PARAMETERS:
        missing = sorted(EXPECTED_PARAMETERS.difference(registry))
        unknown = sorted(set(registry).difference(EXPECTED_PARAMETERS))
        message = f"VP2 parameter registry mismatch: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    for block, parameters in SAMPLING_BLOCKS.items():
        actual = {name for name, entry in registry.items() if entry.get("block") == block}
        if actual != set(parameters):
            message = f"Sampling block {block!r} must contain exactly {sorted(parameters)}, got {sorted(actual)}."
            raise ValueError(message)
        dimension = sum(registry_service.effective_dimension(registry[name]) for name in parameters)
        if dimension != SAMPLING_BLOCK_DIMENSIONS[block]:
            message = f"Sampling block {block!r} has dimension {dimension}; expected {SAMPLING_BLOCK_DIMENSIONS[block]}."
            raise ValueError(message)
    if sum(SAMPLING_BLOCK_DIMENSIONS.values()) != TRANSIENT_DIMENSION or SAMPLING_BLOCK_DIMENSIONS["airflow"] != STEADY_DIMENSION:
        message = "Canonical profile dimensions must remain steady=28 and transient=54."
        raise RuntimeError(message)
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
        or simplex.get("alpha") != [5.5, 3.0, 1.5]
        or simplex.get("minimum_each") != _SIMPLEX_MINIMUM_EACH
        or simplex.get("maximum_each") != _SIMPLEX_MAXIMUM_EACH
    ):
        message = "schedule.component_weights must be the binding complete truncated Dirichlet simplex."
        raise ValueError(message)
    oswin = registry["oswin"]
    if oswin["kind"] != "parameter_set" or tuple(oswin["components"]) != ("A_osw", "B_osw", "C_osw"):
        message = "oswin must be one complete A_osw/B_osw/C_osw parameter set."
        raise ValueError(message)
    expected_derived = {
        "bed.structure.fine_weight": ("complement_of_one", ("bed.structure.coarse_weight",)),
        "initial_moisture.structure.fine_weight": ("complement_of_one", ("initial_moisture.structure.coarse_weight",)),
        "T_init": ("copy", ("T_amb",)),
        "r_surf": ("copy", ("r_surf_0",)),
        "r_int": ("product", ("r_int_surf", "r_surf")),
        "T_in_ref": ("schedule_time_average", ("schedule",)),
    }
    for name, (derivation, source_names) in expected_derived.items():
        entry = registry[name]
        if entry["kind"] != "derived" or entry["derivation"] != derivation or tuple(entry["sources"]) != source_names:
            message = f"Derived parameter {name!r} violates its supplied rule."
            raise ValueError(message)
    return "sampled_rho_fixed_reference_epsilon"


def sampling_block_dimensions(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    blocks: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Return exact numerical dimensions for selected active blocks."""
    selected = tuple(SAMPLING_BLOCKS) if blocks is None else blocks
    if not selected or any(block not in SAMPLING_BLOCKS for block in selected):
        message = f"Unknown or empty active sampling blocks {selected}."
        raise ValueError(message)
    result: dict[str, int] = {}
    for block in selected:
        missing = [name for name in SAMPLING_BLOCKS[block] if name not in registry]
        if missing:
            message = f"Active block {block!r} is missing registry parameters {missing}."
            raise ValueError(message)
        result[block] = sum(registry_service.effective_dimension(registry[name]) for name in SAMPLING_BLOCKS[block])
        if result[block] != SAMPLING_BLOCK_DIMENSIONS[block]:
            message = f"Active block {block!r} has dimension {result[block]}, expected {SAMPLING_BLOCK_DIMENSIONS[block]}."
            raise ValueError(message)
    return result


def sampling_coordinate_labels(
    registry: Mapping[str, Mapping[str, Any]],
    profile_id: str,
) -> tuple[str, ...]:
    """Return exact profile-qualified numerical coordinate labels."""
    labels: list[str] = []
    for block in active_sampling_blocks(profile_id):
        for name in SAMPLING_BLOCKS[block]:
            if name not in registry:
                message = f"Active sampling coordinate {name!r} is missing from the registry."
                raise ValueError(message)
            dimension = registry_service.effective_dimension(registry[name])
            if dimension == 1:
                labels.append(name)
            else:
                labels.extend(f"{name}[{index}]" for index in range(1, dimension + 1))
    return tuple(labels)


def project_material_for_profile(material: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    """Return only material science consumed by one simulation profile."""
    blocks = active_sampling_blocks(profile_id)
    active_names = {name for block in blocks for name in SAMPLING_BLOCKS[block]}
    if profile_id == "steady_flow":
        active_names.update({"bed.structure.fine_weight", "eps_min_global", "eps_max_global", "eps_bed_cal_ref"})
    else:
        active_names.update(SUPPORT_PARAMETERS)
    projected = {
        "material_family": material["material_family"],
        "decision_source": copy.deepcopy(material["decision_source"]),
        "material_scope": copy.deepcopy(material["material_scope"]),
        "packing_porosity_mean_support": copy.deepcopy(material["packing_porosity_mean_support"]),
        "parameter_registry": {name: copy.deepcopy(entry) for name, entry in material["parameter_registry"].items() if name in active_names},
        "effective_parameter_provenance": {
            name: copy.deepcopy(value)
            for name, value in material["effective_parameter_provenance"].items()
            if name in active_names or (profile_id == "transient_drying" and name in {"A_osw", "B_osw", "C_osw"})
        },
        "active_sampling_blocks": list(blocks),
        "active_coordinate_names": [name for block in blocks for name in SAMPLING_BLOCKS[block]],
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
        "decision_source",
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
    family = material["material_family"]
    if material["schema_kind"] != "generation_material" or material["schema_version"] != 1 or family not in MATERIAL_FAMILIES:
        message = "Unsupported or unknown role-neutral material configuration."
        raise ValueError(message)
    decision = validate_decision_source(material["decision_source"], label=f"material {family} decision_source")
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
    _exact_keys(initial, {"mean_db", "amplitude_db", "field_support", "field_constraint"}, label=f"material {family}.initial_moisture")
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
        sources=sources,
        label=f"material {family}.density_calibration",
    )
    owners["rho_bu_dry_ref"] = rho_owner
    owners["eps_bed_cal_ref"] = density_parts["eps_owner"]
    oswin, oswin_owner = _validate_oswin_record(material["oswin"], sources=sources, label=f"material {family}.oswin")
    owners["oswin"] = oswin_owner
    kinetics, kinetics_owners, kinetics_contract = _validate_kinetics_record(
        material["two_compartment_kinetics"],
        sources=sources,
        label=f"material {family}.two_compartment_kinetics",
    )
    owners.update(kinetics_owners)
    target = resolve_value_record(material["target_moisture"], sources=sources, label=f"material {family}.target_moisture")
    _exact_keys(
        target,
        {
            "selected_simulation_target_wb",
            "selected_target_db",
            "market_acceptance_moisture_wb",
            "safe_storage_moisture_wb",
            "provenance",
        },
        label=f"material {family}.target_moisture",
    )
    target_wb = _finite(target["selected_simulation_target_wb"], label=f"material {family}.target_moisture.selected_simulation_target_wb")
    target_db = _finite(target["selected_target_db"], label=f"material {family}.target_moisture.selected_target_db")
    if not 0 < target_wb < 1 or not math.isclose(target_wb / (1.0 - target_wb), target_db, rel_tol=1e-14, abs_tol=1e-14):
        message = f"Material {family!r} target moisture basis conversion is inconsistent."
        raise ValueError(message)
    field_minimum_db = target_db + 0.01
    expected_field_constraint = f"min(X_0_db_field) >= {field_minimum_db:.8f} kg/kg"
    if initial["field_constraint"] != expected_field_constraint or field_minimum_db <= 0:
        message = f"Material {family!r} initial-moisture field constraint must be exactly {expected_field_constraint!r}."
        raise ValueError(message)
    field_constraint = {
        "authored_expression": expected_field_constraint,
        "minimum_db": field_minimum_db,
        "margin_above_target_db": 0.01,
        "unit": "kg/kg",
        "derivation": {
            "kind": "derived_from_configured_target",
            "origin": "supplied_by_handoff",
            "verification": "mathematically_reproduced",
            "inputs": ["X_target_db", "0.01 kg/kg"],
            "formula_or_method": "minimum_db = X_target_db + 0.01 kg/kg",
        },
        "decision_source": copy.deepcopy(decision),
    }
    owners["X_target_wb"] = {"value": target_wb, "nominal": target_wb, "provenance": copy.deepcopy(target["provenance"])}
    registry = _merge_registry(definitions, owners)
    effective: dict[str, Any] = {}
    for name, entry in registry.items():
        entry_provenance = entry.get("provenance")
        if isinstance(entry_provenance, Mapping):
            effective[name] = copy.deepcopy(dict(entry_provenance))
        elif name in DERIVED_PARAMETERS:
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
        "decision_source": decision,
        "material_scope": scope,
        "packing_porosity_mean_support": packing,
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
