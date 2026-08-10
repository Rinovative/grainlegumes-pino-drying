"""
===============================================================================
generation_inventory.py
===============================================================================
Audit profile-qualified parameter ownership, dimensions, and effective consumers.
Responsibilities:
  - Report exact 28-D steady and 54-D transient coordinate inventories
  - Trace each resolved parameter to one authored owner and downstream consumer
  - Keep scientific, generated, COMSOL-adapter, output, and execution state separate
Design principles:
  - Inventory is derived from the active profile projection, never a global superset
  - Missing ownership or consumers remain explicit scientific-audit evidence
  - Execution configuration cannot own scientific values
This module does NOT:
  - Supply scientific values, infer COMSOL binary mappings, or execute simulations
  - Treat a passing ownership audit as proof of scientific plausibility
===============================================================================
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from . import generation_materials as materials
from . import generation_profiles as profiles
from . import generation_registry as registry_service

COMMON_VALUE_PARAMETERS: Final = frozenset({"eps_min_global", "eps_max_global"})
OPERATION_VALUE_PARAMETERS: Final = frozenset(
    set(materials.AIRFLOW_PARAMETERS).difference({"kappa_mean"})
    | set(materials.INITIAL_MOISTURE_PARAMETERS).difference(materials.INITIAL_MOISTURE_LEVEL_PARAMETERS)
    | set(materials.OPERATION_PARAMETERS)
)

PARAMETER_OWNERS: Final = MappingProxyType(
    {
        name: (
            "configs/generation/common.yaml"
            if name in COMMON_VALUE_PARAMETERS
            else "configs/generation/operations/fixed_bed.yaml"
            if name in OPERATION_VALUE_PARAMETERS
            else "configs/generation/registry.yaml"
            if name in materials.DERIVED_PARAMETERS
            else "configs/generation/materials/<material>.yaml"
        )
        for name in materials.EXPECTED_PARAMETERS
    }
)

_BED_STRUCTURE_PARAMETERS: Final = frozenset(
    name for name in materials.EXPECTED_PARAMETERS if name.startswith(("bed.structure.", "bed.perturbations."))
)
_PERMEABILITY_PARAMETERS: Final = frozenset(
    {"kappa_mean", "kappa_cv"} | {name for name in materials.EXPECTED_PARAMETERS if name.startswith("permeability.")}
)
_POROSITY_PARAMETERS: Final = frozenset((*materials.POROSITY_GENERATOR_PARAMETERS, "kappa_mean", "eps_min_global", "eps_max_global"))
_PRESSURE_PARAMETERS: Final = frozenset(name for name in materials.AIRFLOW_PARAMETERS if name.startswith("pressure_bc."))
_INITIAL_MOISTURE_PARAMETERS: Final = frozenset(name for name in materials.EXPECTED_PARAMETERS if name.startswith("initial_moisture."))
_SCHEDULE_PARAMETERS: Final = frozenset(materials.OPERATION_PARAMETERS)
_DENSITY_FIELD_PARAMETERS: Final = frozenset({"rho_bu_dry_ref", "eps_bed_cal_ref"})
_TRANSIENT_SCALAR_PARAMETERS: Final = frozenset(
    {
        "T_init",
        "T_amb",
        "T_in_ref",
        "eps_bed_cal_ref",
        "rho_bu_dry_ref",
        "k_gr",
        "cp_gr_dry",
        "X_target_wb",
        "r_surf_0",
        "r_int_surf",
        "f_surf",
        "oswin",
    }
)
_PHYSICAL_FORMULA_RECORDS: Final = frozenset({"r_surf", "r_int"})
_PARAMETER_CONSUMER_GROUPS: Final = MappingProxyType(
    {
        "generation_fields._bed_structure": _BED_STRUCTURE_PARAMETERS,
        "generation_fields._permeability_fields": _PERMEABILITY_PARAMETERS,
        "generation_fields._porosity_field": _POROSITY_PARAMETERS,
        "generation_fields._pressure_boundary": _PRESSURE_PARAMETERS,
        "generation_fields._initial_moisture": _INITIAL_MOISTURE_PARAMETERS,
        "generation_schedule.generate_schedule": _SCHEDULE_PARAMETERS,
        "generation_fields derived dry-density fields": _DENSITY_FIELD_PARAMETERS,
        "generation_case transient scalar COMSOL adapter": _TRANSIENT_SCALAR_PARAMETERS,
        "common.physical_formulas COMSOL expression contract": _PHYSICAL_FORMULA_RECORDS,
    }
)

COMSOL_INPUT_SOURCES: Final = MappingProxyType(
    {
        "x": "generated Cartesian grid",
        "y": "generated Cartesian grid",
        "Kxx": "generation_fields permeability tensor",
        "Kxy": "generation_fields permeability tensor",
        "Kyy": "generation_fields permeability tensor",
        "eps_bed": "generation_fields porosity map",
        "p_in_bc": "generation_fields inlet-pressure boundary",
        "X_0_db_field": "generation_fields initial-moisture field",
        **{
            name: (
                "common.scientific_fixed_values"
                if name in {"T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}
                else "typed parameter registry or deterministic derivation"
            )
            for name in profiles.TRANSIENT_SCALAR_INPUT_FIELDS
        },
        **dict.fromkeys(profiles.SCHEDULE_FIELDS, "generation_schedule regular hourly nodes"),
    }
)
SEED_GENERATED_VALUES: Final = (
    "bed multiscale realization",
    "bed local perturbation count, locations, widths, orientations, and signs",
    "permeability orientation jitter",
    "inlet-pressure Gaussian details",
    "initial-moisture multiscale realization",
    "schedule harmonic phases and coefficients",
    "schedule event details and activation mask",
)
GLOBAL_FIXED_VALUES: Final = (
    "grid",
    "time",
    "scientific_fixed_values",
    "physical_formulas",
    "input_contract",
    "storage",
)
EXECUTION_ONLY_VALUES: Final = ("site", "runtime", "retention", "cluster")
OUTPUT_ONLY_VALUES: Final = tuple(
    dict.fromkeys(
        (
            *profiles.STEADY_STATIC_FIELD_NAMES,
            *profiles.TRANSIENT_STATIC_FIELD_NAMES,
            *profiles.TRANSIENT_FIELD_NAMES,
            *profiles.GLOBAL_FIELD_NAMES,
            *profiles.FINAL_STATUS_FIELDS,
        )
    )
)


@dataclass(frozen=True, slots=True)
class InventoryReport:
    """One mechanical profile-qualified ownership and consumer audit."""

    profile_id: str
    sampled_dimensions_by_block: dict[str, int]
    sampled_parameters_by_block: dict[str, tuple[str, ...]]
    sampled_coordinate_names: tuple[str, ...]
    effective_consumers: dict[str, tuple[str, ...]]
    derived_quantities: tuple[str, ...]
    coupled_selections: tuple[str, ...]
    material_fixed_values: tuple[str, ...]
    global_fixed_values: tuple[str, ...]
    seed_generated_values: tuple[str, ...]
    execution_only_values: tuple[str, ...]
    output_only_values: tuple[str, ...]
    configured_but_unused: tuple[str, ...]
    consumed_but_undeclared: tuple[str, ...]
    total_effective_dimension: int


def parameter_owner(name: str) -> str:
    """Return the one authoritative configuration owner for a parameter."""
    try:
        return PARAMETER_OWNERS[name]
    except KeyError as error:
        message = f"Parameter {name!r} has no declared scientific owner."
        raise ValueError(message) from error


def parameter_consumers(name: str, profile_id: str) -> tuple[str, ...]:
    """Return effective downstream consumers for one profile-applicable parameter."""
    if profile_id not in materials.PROFILE_SAMPLING_BLOCKS:
        message = f"Unknown generation profile {profile_id!r}."
        raise ValueError(message)
    return tuple(label for label, names in _PARAMETER_CONSUMER_GROUPS.items() if name in names)


def _infer_profile(registry: Mapping[str, Mapping[str, Any]]) -> str:
    """Infer only the two exact maintained registry projections."""
    names = set(registry)
    transient_expected = set(materials.EXPECTED_PARAMETERS)
    steady_expected: set[str] = set(materials.AIRFLOW_PARAMETERS)
    steady_expected.update({"bed.structure.fine_weight", "eps_min_global", "eps_max_global"})
    if names == transient_expected:
        return profiles.TRANSIENT_DRYING_PROFILE
    if names == steady_expected:
        return profiles.STEADY_FLOW_PROFILE
    message = "Registry is neither the exact steady-flow nor transient-drying projection."
    raise ValueError(message)


def audit_parameter_registry(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    profile_id: str | None = None,
) -> InventoryReport:
    """Audit one fully resolved profile registry and return its exact inventory."""
    selected_profile = _infer_profile(registry) if profile_id is None else profile_id
    dimensions = materials.validate_profile_registry(registry, selected_profile)
    declared = set(registry)
    consumers = {name: parameter_consumers(name, selected_profile) for name in registry}
    configured_but_unused = tuple(sorted(name for name, values in consumers.items() if not values))
    consumed_but_undeclared: tuple[str, ...] = ()
    if not declared.issubset(PARAMETER_OWNERS):
        message = "Parameter-owner inventory must cover the exact active registry."
        raise ValueError(message)
    if any(owner.startswith("execution") for owner in PARAMETER_OWNERS.values()):
        message = "Execution settings cannot own scientific parameters."
        raise ValueError(message)
    for name, entry in registry.items():
        if entry["kind"] != "derived":
            continue
        missing_sources = set(entry["sources"]).difference(declared | {"schedule"})
        if missing_sources:
            message = f"Derived parameter {name!r} has undeclared sources {sorted(missing_sources)}."
            raise ValueError(message)
    adapter_names: set[str] = set(profiles.STEADY_SPATIAL_INPUT_FIELDS)
    if selected_profile == profiles.TRANSIENT_DRYING_PROFILE:
        adapter_names.update(profiles.TRANSIENT_SPATIAL_INPUT_FIELDS)
        adapter_names.update(profiles.TRANSIENT_SCALAR_INPUT_FIELDS)
        adapter_names.update(profiles.SCHEDULE_FIELDS)
    if not adapter_names.issubset(COMSOL_INPUT_SOURCES):
        message = "Every profile-active COMSOL input adapter value must have one explicit source."
        raise ValueError(message)
    blocks = materials.active_sampling_blocks(selected_profile)
    sampled_by_block = {
        block: tuple(name for name in materials.SAMPLING_BLOCKS[block] if registry_service.effective_dimension(registry[name]) > 0)
        for block in blocks
    }
    coordinate_names = materials.sampling_coordinate_labels(registry, selected_profile)
    material_fixed = tuple(
        name
        for name, entry in registry.items()
        if parameter_owner(name) == "configs/generation/materials/<material>.yaml" and entry["kind"] == "fixed"
    )
    return InventoryReport(
        profile_id=selected_profile,
        sampled_dimensions_by_block=dimensions,
        sampled_parameters_by_block=sampled_by_block,
        sampled_coordinate_names=coordinate_names,
        effective_consumers=consumers,
        derived_quantities=tuple(name for name, entry in registry.items() if entry["kind"] == "derived"),
        coupled_selections=tuple(name for name, entry in registry.items() if entry["kind"] == "parameter_set"),
        material_fixed_values=material_fixed,
        global_fixed_values=GLOBAL_FIXED_VALUES,
        seed_generated_values=SEED_GENERATED_VALUES,
        execution_only_values=EXECUTION_ONLY_VALUES,
        output_only_values=OUTPUT_ONLY_VALUES,
        configured_but_unused=configured_but_unused,
        consumed_but_undeclared=consumed_but_undeclared,
        total_effective_dimension=sum(dimensions.values()),
    )


def audit_campaign(campaign: Any) -> dict[str, InventoryReport]:
    """Audit every batch registry and scientific/execution separation in a campaign."""
    reports: dict[str, InventoryReport] = {}
    for batch in campaign.batches:
        if "execution" in batch.scientific_values or "cluster" in batch.scientific_values:
            message = f"Batch {batch.batch_name!r} leaks execution settings into scientific identity."
            raise ValueError(message)
        registry = batch.scientific_values["material"]["parameter_registry"]
        reports[batch.batch_name] = audit_parameter_registry(registry, profile_id=batch.profile.id)
    return reports


_ATOMIC_COMPONENT_PARENTS: Final = MappingProxyType(
    {
        "A_osw": "oswin",
        "B_osw": "oswin",
        "C_osw": "oswin",
    }
)
_ATOMIC_RECORD_COMPONENTS: Final = MappingProxyType(
    {
        "density_calibration": ("rho_bu_dry_ref", "eps_bed_cal_ref"),
        "oswin": ("A_osw", "B_osw", "C_osw"),
        "two_compartment_kinetics": ("r_surf_0", "r_int_surf", "f_surf"),
        "schedule_simplex": ("smooth", "event", "trend"),
    }
)
_ATOMIC_RECORD_REGISTRY_NAMES: Final = MappingProxyType(
    {
        "density_calibration": ("rho_bu_dry_ref", "eps_bed_cal_ref"),
        "oswin": ("oswin",),
        "two_compartment_kinetics": ("r_surf_0", "r_int_surf", "f_surf"),
        "schedule_simplex": ("schedule.component_weights",),
    }
)
_PROFILE_ORDER: Final = (profiles.STEADY_FLOW_PROFILE, profiles.TRANSIENT_DRYING_PROFILE)


def _ood_inventory(material: Mapping[str, Any], name: str) -> list[Any]:
    """Return configured family-specific OOD choices for one scalar or record."""
    if name in material["coupled_ood_records"]:
        return copy.deepcopy(material["coupled_ood_records"][name]["records"])
    registry_name = "schedule.component_weights" if name == "schedule_simplex" else name
    entry = material["parameter_registry"].get(registry_name)
    if not isinstance(entry, Mapping):
        return []
    key = {
        "interval": "ood",
        "integer": "ood",
        "categorical": "ood_choices",
        "simplex": "ood_values",
        "parameter_set": "ood_sets",
    }.get(str(entry["kind"]))
    return [] if key is None else copy.deepcopy(list(entry.get(key, [])))


def _configured_without_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return configured values and semantics without duplicating provenance."""
    return {name: copy.deepcopy(item) for name, item in value.items() if name != "provenance"}


def _entry_coordinate_labels(name: str, entry: Mapping[str, Any]) -> tuple[str, ...]:
    """Return exact numerical coordinate labels for one registry entry."""
    dimension = registry_service.effective_dimension(entry)
    if dimension == 1:
        return (name,)
    return tuple(f"{name}[{index}]" for index in range(1, dimension + 1))


def _entry_inspection_contract(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-value registry semantics needed to interpret one route entry."""
    contract = {
        "kind": entry["kind"],
        "classification": entry.get("classification"),
        "unit": entry.get("unit", entry.get("units")),
        "transform": entry.get("transform"),
        "block": entry.get("block"),
        "ood_group": entry.get("ood_group"),
        "profile_applicability": copy.deepcopy(entry.get("profile_applicability")),
        "atomic_record": entry.get("atomic_record"),
    }
    return {name: value for name, value in contract.items() if value is not None}


_SCHEDULE_FIXED_CONSUMERS: Final = frozenset(
    {
        "p_ref",
        "T_in_min",
        "T_in_max",
        "omega_min",
        "omega_max",
        "phi_operational_min",
        "phi_operational_max",
        "phi_clip_min",
        "phi_clip_max",
    }
)
_TEMPLATE_FIXED_NAMES: Final = frozenset(
    {
        "T_flow_ref",
        "p_ref",
        "p_out",
        "phi_clip_min",
        "phi_clip_max",
        "cp_w",
        "h_fg",
        "D_v_air",
        "M_v",
        "d_wall",
        "k_wall",
        "h_ext",
        "U_wall",
        "f_wet_dm_max",
        "schedule_interpolation",
    }
)


def _inspect_fixed_value(
    scientific: Mapping[str, Any],
    profile_id: str,
    canonical_name: str,
) -> dict[str, Any] | None:
    """Return one resolved common fixed-value view when it is profile-active."""
    name = canonical_name.removeprefix("scientific_fixed_values.")
    records = scientific["scientific_fixed_records"]
    if name not in records:
        return None
    record = records[name]
    consumers: list[str] = []
    schedule_consumer = profile_id == profiles.TRANSIENT_DRYING_PROFILE and name in _SCHEDULE_FIXED_CONSUMERS
    if schedule_consumer:
        consumers.append("generation_schedule feasibility or psychrometric conversion")
    scalar_adapter = profile_id == profiles.TRANSIENT_DRYING_PROFILE and name in profiles.TRANSIENT_SCALAR_INPUT_FIELDS
    if scalar_adapter:
        consumers.append("generation_case transient scalar COMSOL adapter")
    steady_template_consumer = profile_id == profiles.STEADY_FLOW_PROFILE and name in profiles.STATIONARY_FIXED_FIELDS
    if steady_template_consumer:
        consumers.append("canonical steady COMSOL template fixed conditioning")
    template_consumer = (
        name in _TEMPLATE_FIXED_NAMES
        and not scalar_adapter
        and not (profile_id == profiles.STEADY_FLOW_PROFILE and name in profiles.STATIONARY_FIXED_FIELDS)
    )
    if template_consumer:
        consumers.append("canonical COMSOL template fixed physics; Python has no runtime setter")
    if name == "f_wet_dm_max":
        consumers.append("generation_storage and pilot exact-stop validation")
    provenance = copy.deepcopy(record["provenance"])
    runtime_state = (
        "case_adapter_requires_native_reload_evidence"
        if scalar_adapter
        else "generator_consumed_and_template_fixed_requires_native_verification"
        if schedule_consumer and template_consumer
        else "generator_consumed"
        if schedule_consumer
        else "template_fixed_no_python_runtime_setter"
        if steady_template_consumer or template_consumer
        else "resolved_without_effective_consumer"
    )
    applicability = (
        [profiles.STEADY_FLOW_PROFILE, profiles.TRANSIENT_DRYING_PROFILE]
        if name in profiles.STATIONARY_FIXED_FIELDS
        else [profiles.TRANSIENT_DRYING_PROFILE]
    )
    return {
        "schema_kind": "vp2_resolved_parameter_inspection",
        "schema_version": 1,
        "canonical_name": name,
        "current_owner": "configs/generation/common.yaml",
        "profile_applicability": applicability,
        "effective_dimension": 0,
        "coordinate_labels": [],
        "kind": "derived" if provenance["status"] == "derived" else "fixed",
        "classification": "derived" if provenance["status"] == "derived" else "fixed",
        "unit": record["unit"],
        "configured": _configured_without_provenance(record),
        "provenance": provenance,
        "producer_to_consumer_path": {
            "authored_owner": "configs/generation/common.yaml",
            "resolver": "generation_config._validate_common_config -> profile-qualified fixed-value projection",
            "sampler": "not_sampled",
            "effective_downstream_consumers": consumers,
            "runtime_mapping_state": runtime_state,
            "case_provenance": ("case.json.sampled_values and scalar_handoff" if scalar_adapter else "resolved scientific config only"),
            "hdf5_config_provenance": "provenance/scientific_config_json.scientific_fixed_records",
            "hdf5_realized_provenance": (
                "scalar values and provenance/case_scientific_provenance_json.sampled_values"
                if scalar_adapter
                else "no separate realized value; fixed contract remains in scientific_config_json"
            ),
        },
    }


def _inspect_common_record(
    scientific: Mapping[str, Any],
    profile_id: str,
    canonical_name: str,
) -> dict[str, Any] | None:
    """Return one resolved grid, time, or physical-formula record view."""
    prefix, separator, component = canonical_name.partition(".")
    if prefix not in {"grid", "time", "physical_formulas"}:
        return None
    values = scientific.get(prefix)
    provenance_values = scientific.get(f"{prefix}_provenance")
    if values is None or provenance_values is None:
        message = f"Common record {prefix!r} is not applicable to profile {profile_id!r}."
        raise ValueError(message)
    if not isinstance(values, Mapping) or not isinstance(provenance_values, Mapping):
        message = f"Resolved common record {prefix!r} must be a mapping."
        raise TypeError(message)
    if prefix == "physical_formulas":
        if separator and component not in values:
            message = f"Physical formula {component!r} is unknown."
            raise ValueError(message)
        selected_values = {component: copy.deepcopy(values[component])} if separator else copy.deepcopy(dict(values))
        components = [component] if separator else list(values)
        provenance = copy.deepcopy(dict(provenance_values))
        consumers = [
            "canonical transient COMSOL template expression contract; Python does not set formulas",
            "generation storage validates configured output fields produced by those expressions",
        ]
        runtime_state = "template_formula_requires_native_model_verification"
        record_kind = "derived_physical_field" if separator else "derived_physical_formula_catalogue"
    else:
        if not separator:
            message = f"Inspect {prefix!r} through one component name such as {prefix}.nx."
            raise ValueError(message)
        if component not in values or component not in provenance_values:
            message = f"Common record component {canonical_name!r} is unknown."
            raise ValueError(message)
        selected_values = {component: copy.deepcopy(values[component])}
        components = [component]
        provenance = copy.deepcopy(provenance_values[component])
        if prefix == "grid":
            consumers = ["generation_fields.generate_case_fields and generation_storage coordinate validation"]
            runtime_state = "generator_consumed"
        else:
            consumers = [
                "generation_schedule regular-node generation",
                "generation_case transient run handoff",
                "generation_storage exact-stop and maximum-duration validation",
            ]
            runtime_state = "generator_and_storage_consumed_requires_native_time_reload_evidence"
        record_kind = "derived" if provenance["status"] == "derived" else "fixed"
    return {
        "schema_kind": "vp2_resolved_parameter_inspection",
        "schema_version": 1,
        "canonical_name": canonical_name,
        "current_owner": "configs/generation/common.yaml",
        "profile_applicability": (
            [profiles.STEADY_FLOW_PROFILE, profiles.TRANSIENT_DRYING_PROFILE] if prefix == "grid" else [profiles.TRANSIENT_DRYING_PROFILE]
        ),
        "effective_dimension": 0,
        "coordinate_labels": [],
        "kind": record_kind,
        "classification": "fixed_or_derived_common_record",
        "configured": selected_values,
        "components": components,
        "provenance": provenance,
        "producer_to_consumer_path": {
            "authored_owner": "configs/generation/common.yaml",
            "resolver": "generation_config._validate_common_config -> profile-qualified projection",
            "sampler": "not_sampled",
            "effective_downstream_consumers": consumers,
            "runtime_mapping_state": runtime_state,
            "case_provenance": "resolved case scientific configuration",
            "hdf5_config_provenance": f"provenance/scientific_config_json.{prefix}",
            "hdf5_realized_provenance": (
                "realized coordinates, fields, and time datasets plus the fixed scientific contract"
                if prefix != "physical_formulas"
                else "realized COMSOL output fields plus the fixed formula catalogue"
            ),
        },
    }


def _inspect_porosity_support(
    campaign: Any,
    natural_batches: Mapping[str, Any],
    canonical_name: str,
) -> dict[str, Any] | None:
    """Return the material-specific porosity-support record and inherited provenance."""
    if canonical_name != "packing_porosity_mean_support":
        return None
    role_by_material = {family: role for role, families in campaign.material_roles.items() for family in families}
    material_views: dict[str, Any] = {}
    for family in materials.MATERIAL_FAMILIES:
        if family not in natural_batches:
            continue
        material = natural_batches[family].scientific_values["material"]
        record = material["packing_porosity_mean_support"]
        material_views[family] = {
            "material_role": role_by_material[family],
            "material_scope": copy.deepcopy(material["material_scope"]),
            "configured": _configured_without_provenance(record),
            "provenance": copy.deepcopy(record["provenance"]),
            "applicability": {
                "profiles": [campaign.profile.id],
                "parameter_ood": False,
                "parameter_ood_reason": ("fixed family-specific natural-support guard, not an independent sampled coordinate"),
            },
        }
    return {
        "schema_kind": "vp2_resolved_parameter_inspection",
        "schema_version": 1,
        "canonical_name": canonical_name,
        "current_owner": "configs/generation/materials/<material>.yaml",
        "profile_applicability": [profiles.STEADY_FLOW_PROFILE, profiles.TRANSIENT_DRYING_PROFILE],
        "effective_dimension": 0,
        "coordinate_labels": [],
        "kind": "coupled_record_guard",
        "classification": "fixed_natural_support",
        "producer_to_consumer_path": {
            "authored_owner": "configs/generation/materials/<material>.yaml",
            "resolver": ("generation_materials.resolve_material_definition -> generation_materials.project_material_for_profile"),
            "sampler": "not_sampled; validates the realized porosity-field mean",
            "effective_downstream_consumers": [
                "generation_fields._porosity_field natural-support guard",
                "generation_sentinels static material-support audit",
            ],
            "case_provenance": "case.json scientific_config.material.packing_porosity_mean_support",
            "hdf5_config_provenance": ("provenance/scientific_config_json.material.packing_porosity_mean_support"),
            "hdf5_realized_provenance": ("porosity field and generator diagnostics; support remains in scientific_config_json"),
        },
        "materials": material_views,
    }


def inspect_campaign_parameter(campaign: Any, canonical_name: str) -> dict[str, Any]:
    """Return complete resolved producer, consumer, provenance, and persistence evidence."""
    if not isinstance(canonical_name, str) or not canonical_name:
        message = "Parameter inspection requires one non-empty canonical name."
        raise ValueError(message)
    natural_batch_list = [batch for batch in campaign.batches if batch.sampling_regime == "natural"]
    natural_batches = {batch.material_family: batch for batch in natural_batch_list}
    expected_materials = {family for families in campaign.material_roles.values() for family in families}
    if len(natural_batch_list) != len(natural_batches) or set(natural_batches) != expected_materials:
        message = "Parameter inspection requires exactly one natural batch per campaign material."
        raise ValueError(message)
    first_scientific = next(iter(natural_batches.values())).scientific_values
    non_registry_view = (
        _inspect_fixed_value(first_scientific, campaign.profile.id, canonical_name)
        or _inspect_common_record(first_scientific, campaign.profile.id, canonical_name)
        or _inspect_porosity_support(campaign, natural_batches, canonical_name)
    )
    if non_registry_view is not None:
        return non_registry_view
    first_material = first_scientific["material"]
    parent = _ATOMIC_COMPONENT_PARENTS.get(canonical_name, canonical_name)
    is_atomic_record = canonical_name in _ATOMIC_RECORD_COMPONENTS
    registry_name = "schedule.component_weights" if canonical_name == "schedule_simplex" else parent
    first_entry = first_material["parameter_registry"].get(registry_name)
    if is_atomic_record and canonical_name not in first_material["atomic_records"]:
        message = f"Atomic record {canonical_name!r} is not applicable to profile {campaign.profile.id!r}."
        raise ValueError(message)
    if first_entry is None and not is_atomic_record:
        message = f"Parameter {canonical_name!r} is unknown or not applicable to profile {campaign.profile.id!r}."
        raise ValueError(message)
    route_names = _ATOMIC_RECORD_REGISTRY_NAMES[canonical_name] if is_atomic_record else (registry_name,)
    missing_route_names = [name for name in route_names if name not in first_material["parameter_registry"]]
    if missing_route_names:
        message = f"Parameter inspection route is missing registry entries {missing_route_names}."
        raise ValueError(message)
    route_entries = tuple((name, first_material["parameter_registry"][name]) for name in route_names)
    primary_entry = route_entries[0][1]
    atomic_name = canonical_name if is_atomic_record else primary_entry.get("atomic_record")
    if canonical_name in {"density_calibration", "two_compartment_kinetics", "oswin"}:
        owner = "configs/generation/materials/<material>.yaml"
    elif canonical_name == "schedule_simplex":
        owner = "configs/generation/operations/fixed_bed.yaml"
    else:
        owner = parameter_owner(registry_name)
    role_by_material = {family: role for role, families in campaign.material_roles.items() for family in families}
    materials_view: dict[str, Any] = {}
    for family in materials.MATERIAL_FAMILIES:
        if family not in natural_batches:
            continue
        material = natural_batches[family].scientific_values["material"]
        registry = material["parameter_registry"]
        if is_atomic_record:
            record = material["atomic_records"][canonical_name]
            provenance = record["provenance"]
            configured = _configured_without_provenance(record)
            components = list(_ATOMIC_RECORD_COMPONENTS[canonical_name])
        elif canonical_name in _ATOMIC_COMPONENT_PARENTS:
            entry = registry[parent]
            provenance = material["effective_parameter_provenance"][canonical_name]
            configured = {
                "parent_record": parent,
                "component_inventory": [
                    {
                        "record_id": record["id"],
                        "value": record["values"][canonical_name],
                    }
                    for record in entry["sets"]
                ],
                "unit": entry["units"][canonical_name],
            }
            components = [canonical_name]
        else:
            entry = registry[registry_name]
            provenance = material["effective_parameter_provenance"][registry_name]
            configured = _configured_without_provenance(entry)
            components = list(entry.get("components", ())) or [canonical_name]
        ood_inventory_name = atomic_name or registry_name
        ood_inventory = _ood_inventory(material, ood_inventory_name)
        seen = role_by_material[family] == "seen"
        materials_view[family] = {
            "material_role": role_by_material[family],
            "material_scope": copy.deepcopy(material["material_scope"]),
            "configured": configured,
            "components": components,
            "provenance": copy.deepcopy(provenance),
            "applicability": {
                "profiles": [campaign.profile.id],
                "parameter_ood": seen and bool(ood_inventory),
                "parameter_ood_reason": (
                    "seen material with a configured disjoint OOD tail or complete record"
                    if seen and ood_inventory
                    else "family-OOD materials use natural support only"
                    if not seen
                    else "no supplied compatible OOD tail or complete record"
                ),
                "configured_ood_inventory": ood_inventory,
            },
        }
    applicability = {profile_id for _name, entry in route_entries for profile_id in entry.get("profile_applicability", ())}
    profiles_applicable = [profile_id for profile_id in _PROFILE_ORDER if profile_id in applicability]
    coordinate_labels = tuple(label for name, entry in route_entries for label in _entry_coordinate_labels(name, entry))
    consumers = tuple(dict.fromkeys(consumer for name, _entry in route_entries for consumer in parameter_consumers(name, campaign.profile.id)))
    component_contracts = {name: _entry_inspection_contract(entry) for name, entry in route_entries}
    result: dict[str, Any] = {
        "schema_kind": "vp2_resolved_parameter_inspection",
        "schema_version": 1,
        "canonical_name": canonical_name,
        "current_owner": owner,
        "profile_applicability": profiles_applicable,
        "effective_dimension": len(coordinate_labels),
        "coordinate_labels": list(coordinate_labels),
        "kind": "atomic_record" if is_atomic_record else primary_entry["kind"],
        "classification": "coupled_record" if is_atomic_record else primary_entry.get("classification"),
        "component_contracts": component_contracts,
        "producer_to_consumer_path": {
            "authored_owner": owner,
            "resolver": (
                "generation_config.load_campaign_config -> "
                "generation_materials.resolve_material_definition -> "
                "generation_materials.project_material_for_profile"
            ),
            "sampler": "generation_sampling.sample_case",
            "realized_components": list(_ATOMIC_RECORD_COMPONENTS.get(canonical_name, ())) or [canonical_name],
            "effective_downstream_consumers": list(consumers),
            "case_provenance": ("case.json.sampled_values / block_provenance / ood / coupled_selections"),
            "hdf5_config_provenance": ("provenance/scientific_config_json.material.effective_parameter_provenance"),
            "hdf5_realized_provenance": ("provenance/case_scientific_provenance_json.sampled_values / block_provenance / ood / coupled_selections"),
        },
        "materials": materials_view,
    }
    if atomic_name is not None:
        result["atomic_record"] = atomic_name
    if not is_atomic_record:
        for key in ("block", "unit", "units", "transform", "ood_group"):
            if key in primary_entry:
                output_key = "unit" if key == "units" else key
                result[output_key] = copy.deepcopy(primary_entry[key])
    return result
