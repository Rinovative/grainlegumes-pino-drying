"""
===============================================================================
generation_materials.py
===============================================================================
Resolve role-neutral material definitions into typed scientific registries.
Responsibilities:
  - Define exact material identifiers, sampling blocks, and numerical dimensions
  - Merge disjoint registry, common, operation, and material value ownership
  - Validate material evidence metadata and density-calibration alternatives
Design principles:
  - Material files contain natural properties and never experimental roles
  - Every scientific value has one declarative owner
  - Unresolved literature values are allowed only in non-executable templates
This module does NOT:
  - Assign seen or family-OOD roles, counts, seeds, or dataset membership
  - Sample designs, generate fields, or invent scientific values
===============================================================================
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from . import generation_registry as registry_service

MATERIAL_FAMILIES: Final = (
    "lentil",
    "chickpea",
    "kidney_bean",
    "field_pea",
    "almond",
)

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
    "porosity.anchor_rel",
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
SAMPLING_BLOCK_DIMENSIONS: Final = MappingProxyType(
    {
        "airflow": 28,
        "initial_moisture": 8,
        "operation": 12,
        "material_properties": 6,
    }
)
OOD_GROUPS: Final = ("bed", "operation", "initial_moisture", "material_properties")

DERIVED_PARAMETERS: Final = (
    "bed.structure.fine_weight",
    "initial_moisture.structure.fine_weight",
    "T_init",
    "r_surf",
    "r_int",
    "T_in_ref",
    "T_flow_ref",
)
SUPPORT_PARAMETERS: Final = (
    "eps_min_global",
    "eps_max_global",
    "eps_bed_cal_ref",
    "X_target_wb",
    "oswin",
    *DERIVED_PARAMETERS,
)
OPTIONAL_PARAMETERS: Final = ("density_calibration",)
EXPECTED_PARAMETERS: Final = frozenset(name for names in SAMPLING_BLOCKS.values() for name in names) | frozenset(SUPPORT_PARAMETERS)

_REGISTRY_METADATA_KEYS: Final = frozenset({"report_symbol", "description"})
_EVIDENCE_REQUIRED_KEYS: Final = frozenset(
    {
        "source",
        "evidence_type",
        "confidence",
        "temperature_range",
        "humidity_range",
        "cultivar_or_market_class",
        "product_form",
        "status",
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


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return one isolated string-keyed mapping."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        message = f"{label} must be a mapping with string keys."
        raise TypeError(message)
    return copy.deepcopy(dict(value))


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    """Require one exact mapping schema."""
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        message = f"{label} keys are invalid: missing={missing}, unknown={unknown}."
        raise ValueError(message)


def _semantic_entry(value: Any, *, label: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Separate validator-owned parameter semantics from catalogue metadata."""
    entry = _mapping(value, label=label)
    metadata: dict[str, str] = {}
    for key in _REGISTRY_METADATA_KEYS:
        item = entry.pop(key, None)
        if not isinstance(item, str) or not item:
            message = f"{label}.{key} must be non-empty text."
            raise ValueError(message)
        metadata[key] = item
    return entry, metadata


def _validate_report_symbols(metadata: Mapping[str, Mapping[str, str]]) -> None:
    """Require unique symbols and the exact conventional correlation notation."""
    owners: dict[str, str] = {}
    for name, entry in metadata.items():
        symbol = entry["report_symbol"]
        if symbol in owners:
            message = f"Report symbol {symbol!r} is assigned to both {owners[symbol]!r} and {name!r}."
            raise ValueError(message)
        owners[symbol] = name
    for name, expected in _CORRELATION_REPORT_SYMBOLS.items():
        actual = metadata[name]["report_symbol"]
        if actual != expected:
            message = f"Correlation parameter {name!r} must use report symbol {expected!r}, got {actual!r}."
            raise ValueError(message)


def validate_semantic_registry(value: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """Validate the complete source-owned parameter-semantic inventory."""
    config = _mapping(value, label="generation parameter registry")
    _exact_keys(config, {"schema_kind", "schema_version", "parameters"}, label="generation parameter registry")
    if config["schema_kind"] != "generation_parameter_registry" or config["schema_version"] != 1:
        message = "Unsupported generation parameter-registry schema."
        raise ValueError(message)
    raw_parameters = _mapping(config["parameters"], label="generation parameter registry.parameters")
    missing = sorted(EXPECTED_PARAMETERS.difference(raw_parameters))
    unknown = sorted(set(raw_parameters).difference(EXPECTED_PARAMETERS | set(OPTIONAL_PARAMETERS)))
    if missing or unknown:
        message = f"Parameter semantics mismatch: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    definitions: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for name, raw in raw_parameters.items():
        definitions[name], metadata[name] = _semantic_entry(raw, label=f"parameters.{name}")
    _validate_report_symbols(metadata)
    return definitions, metadata


def _validate_evidence(value: Any, *, label: str) -> dict[str, Any]:
    """Validate one unresolved or sourced material-evidence record."""
    evidence = _mapping(value, label=label)
    _exact_keys(evidence, set(_EVIDENCE_REQUIRED_KEYS), label=label)
    if evidence["evidence_type"] not in {"measured", "fitted", "inferred", "assumed", "unresolved"}:
        message = f"{label}.evidence_type is unsupported."
        raise ValueError(message)
    if evidence["confidence"] not in {"high", "medium", "low", "unresolved"}:
        message = f"{label}.confidence is unsupported."
        raise ValueError(message)
    if evidence["status"] not in {"resolved", "unresolved"}:
        message = f"{label}.status must be resolved or unresolved."
        raise ValueError(message)
    for key in ("source", "temperature_range", "humidity_range", "cultivar_or_market_class", "product_form"):
        if evidence[key] is not None and not isinstance(evidence[key], (str, list)):
            message = f"{label}.{key} must be null, text, or an explicit range."
            raise TypeError(message)
    return evidence


def _initial_moisture_bounds(value: Any, *, label: str, allow_unresolved: bool) -> dict[str, float | None]:
    """Validate natural dry-basis initial-moisture bounds."""
    bounds = _mapping(value, label=label)
    _exact_keys(bounds, {"lower", "upper"}, label=label)
    normalized: dict[str, float | None] = {}
    for key in ("lower", "upper"):
        number = bounds[key]
        if number is None and allow_unresolved:
            normalized[key] = None
        elif isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            message = f"{label}.{key} must be finite."
            raise ValueError(message)
        else:
            normalized[key] = float(number)
    lower = normalized["lower"]
    upper = normalized["upper"]
    if lower is not None and upper is not None and lower > upper:
        message = f"{label}.lower must not exceed {label}.upper."
        raise ValueError(message)
    return normalized


def _merge_registry(
    definitions: Mapping[str, Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
    *,
    allow_unresolved: bool,
) -> dict[str, dict[str, Any]]:
    """Merge exact disjoint value owners into validator-ready entries."""
    paired_density = "density_calibration" in owners
    if paired_density and ({"rho_bu_dry_ref", "eps_bed_cal_ref"} & set(owners)):
        message = "Paired density calibration cannot coexist with independent rho/eps values."
        raise ValueError(message)
    missing_values: list[str] = []
    registry: dict[str, dict[str, Any]] = {}
    for name, definition in definitions.items():
        if name in OPTIONAL_PARAMETERS and name not in owners:
            continue
        entry = copy.deepcopy(dict(definition))
        if paired_density and name in {"rho_bu_dry_ref", "eps_bed_cal_ref"}:
            entry = {
                "kind": "derived",
                "unit": entry["unit"],
                "derivation": "selected_parameter_set_component",
                "sources": ["density_calibration"],
            }
        additions = copy.deepcopy(dict(owners.get(name, {})))
        overlap = set(entry).intersection(additions)
        if overlap:
            message = f"Parameter {name!r} has duplicate semantic/value keys {sorted(overlap)}."
            raise ValueError(message)
        entry.update(additions)
        if name not in DERIVED_PARAMETERS and not (paired_density and name in {"rho_bu_dry_ref", "eps_bed_cal_ref"}) and name not in owners:
            missing_values.append(name)
        registry[name] = entry
    if missing_values:
        message = f"Scientific parameters have no value owner: {sorted(missing_values)}."
        raise ValueError(message)
    validated = registry_service.validate_parameter_registry(registry, allow_unresolved=allow_unresolved)
    validate_vp2_registry(validated)
    return validated


def _require_derivation(
    registry: Mapping[str, Mapping[str, Any]],
    name: str,
    derivation: str,
    sources: tuple[str, ...],
) -> None:
    """Require one exact supported derived declaration."""
    entry = registry[name]
    if entry["kind"] != "derived" or entry["derivation"] != derivation or tuple(entry["sources"]) != sources:
        message = f"Parameter {name!r} must use derivation {derivation!r} from {list(sources)}."
        raise ValueError(message)


def validate_density_calibration_mode(registry: Mapping[str, Mapping[str, Any]]) -> str:
    """Require one mutually exclusive density/porosity calibration mode."""
    density = registry.get("density_calibration")
    rho = registry["rho_bu_dry_ref"]
    eps = registry["eps_bed_cal_ref"]
    if density is None:
        if rho["kind"] != "interval" or rho.get("block") != "material_properties" or eps["kind"] != "fixed":
            message = "Independent density mode requires sampled rho_bu_dry_ref and fixed eps_bed_cal_ref."
            raise ValueError(message)
        return "fixed_eps_sampled_rho"
    if density["kind"] != "paired_parameter_set" or tuple(density["components"]) != ("rho_bu_dry_ref", "eps_bed_cal_ref"):
        message = "density_calibration must pair rho_bu_dry_ref and eps_bed_cal_ref."
        raise ValueError(message)
    for name in ("rho_bu_dry_ref", "eps_bed_cal_ref"):
        entry = registry[name]
        if entry["kind"] != "derived" or tuple(entry["sources"]) != ("density_calibration",):
            message = f"{name} must be derived only from density_calibration in paired mode."
            raise ValueError(message)
    return "paired_density_calibration"


def sampling_block_dimensions(
    registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    """Return effective numerical dimensions for the active density mode."""
    validate_vp2_registry(registry)
    return {block: sum(registry_service.effective_dimension(registry[name]) for name in parameters) for block, parameters in SAMPLING_BLOCKS.items()}


def validate_vp2_registry(registry: Mapping[str, Mapping[str, Any]]) -> str:
    """Validate final block membership, dimensions, coupled sets, and derivations."""
    names = set(registry)
    missing = sorted(EXPECTED_PARAMETERS.difference(names))
    unknown = sorted(names.difference(EXPECTED_PARAMETERS | set(OPTIONAL_PARAMETERS)))
    if missing or unknown:
        message = f"VP2 parameter registry mismatch: missing={missing}, unknown={unknown}."
        raise ValueError(message)
    density_mode = validate_density_calibration_mode(registry)
    for block, parameters in SAMPLING_BLOCKS.items():
        expected = set(parameters)
        expected_dimension = SAMPLING_BLOCK_DIMENSIONS[block]
        if density_mode == "paired_density_calibration" and block == "material_properties":
            expected.remove("rho_bu_dry_ref")
            expected_dimension -= 1
        actual = {name for name, entry in registry.items() if entry.get("block") == block}
        if actual != expected:
            message = f"Sampling block {block!r} must contain exactly {sorted(expected)}, got {sorted(actual)}."
            raise ValueError(message)
        dimension = sum(registry_service.effective_dimension(registry[name]) for name in expected)
        if dimension != expected_dimension:
            message = f"Sampling block {block!r} has dimension {dimension}; expected {expected_dimension}."
            raise ValueError(message)
    simplex = registry["schedule.component_weights"]
    if simplex["kind"] != "simplex" or tuple(simplex["components"]) != ("smooth", "event", "trend"):
        message = "schedule.component_weights must be the smooth/event/trend simplex."
        raise ValueError(message)
    oswin = registry["oswin"]
    if oswin["kind"] != "parameter_set" or tuple(oswin["components"]) != ("A_osw", "B_osw", "C_osw"):
        message = "oswin must be one coupled A_osw/B_osw/C_osw parameter set."
        raise ValueError(message)
    if registry["X_target_wb"]["kind"] != "fixed":
        message = "X_target_wb must be material fixed."
        raise ValueError(message)
    _require_derivation(
        registry,
        "bed.structure.fine_weight",
        "complement_of_one",
        ("bed.structure.coarse_weight",),
    )
    _require_derivation(
        registry,
        "initial_moisture.structure.fine_weight",
        "complement_of_one",
        ("initial_moisture.structure.coarse_weight",),
    )
    _require_derivation(registry, "T_init", "copy", ("T_amb",))
    _require_derivation(registry, "r_surf", "copy", ("r_surf_0",))
    _require_derivation(registry, "r_int", "product", ("r_int_surf", "r_surf"))
    _require_derivation(registry, "T_in_ref", "schedule_time_average", ("schedule",))
    _require_derivation(registry, "T_flow_ref", "mean", ("T_in_ref", "T_init"))
    return density_mode


def _validate_taxonomy(value: Any, *, label: str, allow_unresolved: bool) -> dict[str, Any]:
    """Validate one explicit role-neutral taxonomic description."""
    taxonomy = _mapping(value, label=label)
    _exact_keys(
        taxonomy,
        {"common_name", "species", "market_class", "cultivar", "specificity_status"},
        label=label,
    )
    if not isinstance(taxonomy["common_name"], str) or not taxonomy["common_name"]:
        message = f"{label}.common_name must be non-empty text."
        raise ValueError(message)
    for key in ("species", "market_class", "cultivar"):
        if taxonomy[key] is not None and (not isinstance(taxonomy[key], str) or not taxonomy[key]):
            message = f"{label}.{key} must be null or non-empty text."
            raise ValueError(message)
    if taxonomy["specificity_status"] not in {"resolved", "unresolved"}:
        message = f"{label}.specificity_status must be resolved or unresolved."
        raise ValueError(message)
    if not allow_unresolved and taxonomy["specificity_status"] != "resolved":
        message = f"{label} remains scientifically unresolved."
        raise ValueError(message)
    return taxonomy


def _validate_product_form(value: Any, *, label: str, allow_unresolved: bool) -> dict[str, Any]:
    """Validate one explicit whole/split, shell, and skin product description."""
    product = _mapping(value, label=label)
    _exact_keys(
        product,
        {
            "whole_or_split",
            "shell_state",
            "skin_or_seed_coat_state",
            "description",
            "specificity_status",
        },
        label=label,
    )
    if product["whole_or_split"] not in {"whole", "split"}:
        message = f"{label}.whole_or_split must be whole or split."
        raise ValueError(message)
    for key in ("shell_state", "skin_or_seed_coat_state"):
        if product[key] is not None and (not isinstance(product[key], str) or not product[key]):
            message = f"{label}.{key} must be null or non-empty text."
            raise ValueError(message)
    if not isinstance(product["description"], str) or not product["description"]:
        message = f"{label}.description must be non-empty text."
        raise ValueError(message)
    if product["specificity_status"] not in {"resolved", "unresolved"}:
        message = f"{label}.specificity_status must be resolved or unresolved."
        raise ValueError(message)
    if not allow_unresolved and product["specificity_status"] != "resolved":
        message = f"{label} remains scientifically unresolved."
        raise ValueError(message)
    return product


def resolve_material_definition(
    definitions: Mapping[str, Mapping[str, Any]],
    common_values: Any,
    operation_values: Any,
    material_config: Any,
    *,
    allow_unresolved: bool,
) -> dict[str, Any]:
    """Resolve one role-neutral material file with global and operation owners."""
    material = _mapping(material_config, label="material configuration")
    _exact_keys(
        material,
        {
            "schema_kind",
            "schema_version",
            "material_family",
            "executable",
            "taxonomy",
            "product_form",
            "parameter_values",
            "evidence",
        },
        label="material configuration",
    )
    material_family = material["material_family"]
    if material["schema_kind"] != "generation_material" or material["schema_version"] != 1 or material_family not in MATERIAL_FAMILIES:
        message = "Unsupported or unknown role-neutral material configuration."
        raise ValueError(message)
    if not isinstance(material["executable"], bool) or (not material["executable"] and not allow_unresolved):
        message = f"Material {material_family!r} is non-executable because scientific evidence remains unresolved."
        raise ValueError(message)
    taxonomy = _validate_taxonomy(
        material["taxonomy"],
        label=f"material {material_family} taxonomy",
        allow_unresolved=allow_unresolved,
    )
    product_form = _validate_product_form(
        material["product_form"],
        label=f"material {material_family} product_form",
        allow_unresolved=allow_unresolved,
    )
    material_values = _mapping(material["parameter_values"], label=f"material {material_family} parameter values")
    if "initial_moisture_bounds" not in material_values:
        message = f"Material {material_family!r} must own initial_moisture_bounds in parameter_values."
        raise ValueError(message)
    initial_moisture_bounds = material_values.pop("initial_moisture_bounds")

    owner_maps = (
        _mapping(common_values, label="common parameter values"),
        _mapping(operation_values, label="operation parameter values"),
        material_values,
    )
    owners: dict[str, Mapping[str, Any]] = {}
    for owner in owner_maps:
        overlap = set(owners).intersection(owner)
        if overlap:
            message = f"Scientific parameter values have duplicate owners: {sorted(overlap)}."
            raise ValueError(message)
        owners.update(owner)
    unknown = sorted(set(owners).difference(definitions))
    if unknown:
        message = f"Scientific value owners contain undeclared parameters: {unknown}."
        raise ValueError(message)

    evidence = _mapping(material["evidence"], label=f"material {material_family} evidence")
    expected_evidence = set(material_values) | {"initial_moisture_bounds"}
    if set(evidence) != expected_evidence:
        message = f"Material {material_family!r} evidence must cover every material-owned quantity."
        raise ValueError(message)
    validated_evidence = {name: _validate_evidence(record, label=f"material {material_family} evidence.{name}") for name, record in evidence.items()}
    if not allow_unresolved and any(record["status"] != "resolved" for record in validated_evidence.values()):
        message = f"Material {material_family!r} contains unresolved evidence records."
        raise ValueError(message)
    registry = _merge_registry(definitions, owners, allow_unresolved=allow_unresolved)
    return {
        "material_family": material_family,
        "taxonomy": taxonomy,
        "product_form": product_form,
        "density_calibration_mode": validate_density_calibration_mode(registry),
        "initial_moisture_bounds": _initial_moisture_bounds(
            initial_moisture_bounds,
            label=f"material {material_family} initial_moisture_bounds",
            allow_unresolved=allow_unresolved,
        ),
        "parameter_registry": registry,
        "evidence": validated_evidence,
    }
