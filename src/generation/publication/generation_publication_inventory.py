"""
===============================================================================
generation_publication_inventory.py
===============================================================================
Audit profile-qualified parameter ownership, dimensions, and effective consumers.
Responsibilities:
  - Report exact profile-specific sampled-coordinate inventories
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

from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.contracts import generation_contracts_registry as registry_service

_COMMON_VALUE_PARAMETERS: Final = frozenset({"eps_min_global", "eps_max_global"})
_MATERIAL_VALUE_PARAMETERS: Final = frozenset(
    {
        "kappa_mean",
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "rho_bu_dry_ref",
        "eps_bed_cal_ref",
        "k_gr",
        "cp_gr_dry",
        "X_target_wb",
        "oswin",
        "r_surf_0",
        "r_int_surf",
        "f_surf",
    }
)
_TRANSIENT_SCALAR_PARAMETERS: Final = frozenset(
    {
        "T_amb",
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
_TRANSIENT_ONLY_CONSUMERS: Final = frozenset(
    {
        "generation.cases.generation_cases_fields._initial_moisture",
        "generation.cases.generation_cases_schedule.generate_schedule",
        "generation.cases.generation_cases_fields derived dry-density fields",
        "generation.runtime.generation_runtime_batch admitted COMSOL CLI scalar override",
        "common.physical_formulas COMSOL expression contract",
    }
)

COMSOL_INPUT_SOURCES: Final = MappingProxyType(
    {
        "x": "generated Cartesian grid",
        "y": "generated Cartesian grid",
        "Kxx": "generation.cases.generation_cases_fields permeability tensor",
        "Kxy": "generation.cases.generation_cases_fields permeability tensor",
        "Kyy": "generation.cases.generation_cases_fields permeability tensor",
        "eps_bed": "generation.cases.generation_cases_fields porosity map",
        "p_in_bc": "generation.cases.generation_cases_fields inlet-pressure boundary",
        "X_0_db_field": "generation.cases.generation_cases_fields initial-moisture field",
        **dict.fromkeys(
            profiles.TRANSIENT_SCALAR_INPUT_FIELDS,
            "typed parameter registry or deterministic derivation",
        ),
        **dict.fromkeys(profiles.SCHEDULE_FIELDS, "generation.cases.generation_cases_schedule regular hourly nodes"),
    }
)
SEED_GENERATED_VALUES: Final = (
    "bed multiscale realization",
    "bed local perturbation count, locations, widths, orientations, and signs",
    "permeability orientation jitter",
    "inlet-pressure Gaussian details",
    "initial-moisture multiscale realization",
    "schedule low-pass excitation and filter details",
    "schedule event placements, types, signs, durations, and widths",
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


def parameter_owner(name: str, entry: Mapping[str, Any]) -> str:
    """Return the authoritative configuration owner from one resolved entry."""
    if name in _COMMON_VALUE_PARAMETERS:
        return "configs/generation/common.yaml"
    if entry.get("kind") == "derived":
        return "configs/generation/registry.yaml"
    if name in _MATERIAL_VALUE_PARAMETERS:
        return "configs/generation/materials/<material>.yaml"
    return "configs/generation/operations/fixed_bed.yaml"


def parameter_consumers(
    name: str,
    profile_id: str,
    entry: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return effective implementation consumers for one resolved parameter."""
    profiles.resolve_profile(profile_id)
    consumers: list[str] = []
    if name.startswith(("bed.structure.", "bed.perturbations.")):
        consumers.append("generation.cases.generation_cases_fields._bed_structure")
    if name in {"kappa_mean", "kappa_cv"} or name.startswith("permeability."):
        consumers.append("generation.cases.generation_cases_fields._permeability_fields")
    if name in {*materials.POROSITY_GENERATOR_PARAMETERS, "kappa_mean", "eps_bed_cal_ref", "eps_min_global", "eps_max_global"}:
        consumers.append("generation.cases.generation_cases_fields._porosity_field")
    if name.startswith("pressure_bc."):
        consumers.append("generation.cases.generation_cases_fields._pressure_boundary")
    if name.startswith("initial_moisture."):
        consumers.append("generation.cases.generation_cases_fields._initial_moisture")
    if entry.get("block") == "operation" and name != "T_init":
        consumers.append("generation.cases.generation_cases_schedule.generate_schedule")
    if name == "T_init":
        consumers.append("canonical COMSOL template derived alias and generation.validation.generation_validation_sentinels")
    if name in {"rho_bu_dry_ref", "eps_bed_cal_ref"}:
        consumers.append("generation.cases.generation_cases_fields derived dry-density fields")
    if name in _TRANSIENT_SCALAR_PARAMETERS:
        consumers.append("generation.runtime.generation_runtime_batch admitted COMSOL CLI scalar override")
    if name in {"r_surf", "r_int"}:
        consumers.append("common.physical_formulas COMSOL expression contract")
    return tuple(consumer for consumer in consumers if not (profile_id == profiles.STEADY_FLOW_PROFILE and consumer in _TRANSIENT_ONLY_CONSUMERS))


def _infer_profile(registry: Mapping[str, Mapping[str, Any]]) -> str:
    """Infer the profile represented by one already projected registry."""
    applicability = {name: tuple(entry.get("profile_applicability", ())) for name, entry in registry.items()}
    if not applicability or any(not values for values in applicability.values()):
        message = "Projected registry entries must declare profile_applicability."
        raise ValueError(message)
    if any(profiles.STEADY_FLOW_PROFILE not in values for values in applicability.values()):
        if all(profiles.TRANSIENT_DRYING_PROFILE in values for values in applicability.values()):
            return profiles.TRANSIENT_DRYING_PROFILE
        message = "Projected registry mixes incompatible profile applicability."
        raise ValueError(message)
    return profiles.STEADY_FLOW_PROFILE


def audit_parameter_registry(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    profile_id: str | None = None,
    block_parameters: Mapping[str, tuple[str, ...]] | None = None,
) -> InventoryReport:
    """Audit one fully resolved profile registry and return its exact inventory."""
    selected_profile = _infer_profile(registry) if profile_id is None else profile_id
    materials.validate_profile_registry(registry, selected_profile)
    declared = set(registry)
    consumers = {name: parameter_consumers(name, selected_profile, entry) for name, entry in registry.items()}
    configured_but_unused = tuple(sorted(name for name, values in consumers.items() if not values))
    consumed_but_undeclared: tuple[str, ...] = ()
    owners = {name: parameter_owner(name, entry) for name, entry in registry.items()}
    if any(owner.startswith("execution") for owner in owners.values()):
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
    blocks = materials.active_sampling_blocks(registry, selected_profile)
    block_membership = materials.sampling_blocks(registry) if block_parameters is None else block_parameters
    dimensions = materials.sampling_block_dimensions(
        registry,
        blocks=blocks,
        block_parameters=block_membership,
    )
    sampled_by_block = {
        block: tuple(name for name in block_membership[block] if registry_service.effective_dimension(registry[name]) > 0) for block in blocks
    }
    coordinate_names = materials.sampling_coordinate_labels(
        registry,
        selected_profile,
        block_parameters=block_membership,
    )
    material_fixed = tuple(
        name
        for name, entry in registry.items()
        if parameter_owner(name, entry) == "configs/generation/materials/<material>.yaml" and entry["kind"] == "fixed"
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
        block_parameters = {
            block: tuple(str(name) for name in plan["parameters"]) for block, plan in batch.scientific_values["sampling"]["blocks"].items()
        }
        reports[batch.batch_name] = audit_parameter_registry(
            registry,
            profile_id=batch.profile.id,
            block_parameters=block_parameters,
        )
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
    if entry["kind"] == "conditional_interval":
        registry = material["parameter_registry"]
        values = {
            "kappa_mean": registry["kappa_mean"]["nominal"],
            "eps_bed_cal_ref": registry["eps_bed_cal_ref"]["value"],
            "eps_min_global": registry["eps_min_global"]["value"],
            "eps_max_global": registry["eps_max_global"]["value"],
        }
        support = registry_service.resolve_conditional_support(
            entry,
            values=values,
            material_contract=material,
        )
        return copy.deepcopy(list(support["available_ood_tails"]))
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
        "transform": (
            "conditional_log" if entry.get("kind") == "conditional_interval" and entry.get("transform") == "log" else entry.get("transform")
        ),
        "support_kind": entry.get("support_kind"),
        "support_resolver": entry.get("support_resolver"),
        "conditioning_coordinates": copy.deepcopy(entry.get("conditioning_coordinates")),
        "material_inputs": copy.deepcopy(entry.get("material_inputs")),
        "parameter_ood": entry.get("parameter_ood"),
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
        *profiles.STATIONARY_FIXED_FIELDS,
        "f_wet_dm_max",
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
        consumers.append("generation.cases.generation_cases_schedule feasibility or psychrometric conversion")
    scalar_provenance = profile_id == profiles.TRANSIENT_DRYING_PROFILE and name in profiles.TRANSIENT_SCALAR_INPUT_FIELDS
    if scalar_provenance:
        consumers.append(
            "generation.contracts.generation_contracts_scalar_handoff case provenance and generation.publication.generation_publication_storage HDF5"
        )
    steady_template_consumer = profile_id == profiles.STEADY_FLOW_PROFILE and name in profiles.STATIONARY_FIXED_FIELDS
    if steady_template_consumer:
        consumers.append("canonical steady COMSOL template fixed conditioning")
    template_consumer = name in _TEMPLATE_FIXED_NAMES and not (
        profile_id == profiles.STEADY_FLOW_PROFILE and name in profiles.STATIONARY_FIXED_FIELDS
    )
    if template_consumer:
        consumers.append("canonical COMSOL template fixed physics; Python has no runtime setter")
    if name == "f_wet_dm_max":
        consumers.append("generation.publication.generation_publication_storage and pilot exact-stop validation")
    provenance = copy.deepcopy(record["provenance"])
    runtime_state = (
        "generator_consumed_and_template_fixed_requires_native_verification"
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
        "kind": "derived" if provenance["evidence"] == "derived" else "fixed",
        "classification": "derived" if provenance["evidence"] == "derived" else "fixed",
        "unit": record["unit"],
        "configured": _configured_without_provenance(record),
        "provenance": provenance,
        "producer_to_consumer_path": {
            "authored_owner": "configs/generation/common.yaml",
            "resolver": "generation.cases.generation_cases_config._validate_common_config -> profile-qualified fixed-value projection",
            "sampler": "not_sampled",
            "effective_downstream_consumers": consumers,
            "runtime_mapping_state": runtime_state,
            "case_provenance": ("case.json.sampled_values and scalar_handoff" if scalar_provenance else "resolved scientific config only"),
            "hdf5_config_provenance": "provenance/scientific_config_json.scientific_fixed_records",
            "hdf5_realized_provenance": (
                "scalar values and provenance/case_scientific_provenance_json.sampled_values"
                if scalar_provenance
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
            consumers = [
                "generation.cases.generation_cases_fields.generate_case_fields and "
                "generation.publication.generation_publication_storage coordinate validation"
            ]
            runtime_state = "generator_consumed"
        else:
            consumers = [
                "generation.cases.generation_cases_schedule regular-node generation",
                "generation.cases.generation_cases_case transient run handoff",
                "generation.publication.generation_publication_storage exact-stop and maximum-duration validation",
            ]
            runtime_state = "generator_and_storage_consumed_requires_native_time_reload_evidence"
        record_kind = "derived" if provenance["evidence"] == "derived" else "fixed"
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
            "resolver": "generation.cases.generation_cases_config._validate_common_config -> profile-qualified projection",
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
    for family in natural_batches:
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
            "resolver": (
                "generation.contracts.generation_contracts_materials.resolve_material_definition -> "
                "generation.contracts.generation_contracts_materials.project_material_for_profile"
            ),
            "sampler": "not_sampled; validates the realized porosity-field mean",
            "effective_downstream_consumers": [
                "generation.cases.generation_cases_fields._porosity_field natural-support guard",
                "generation.validation.generation_validation_sentinels static material-support audit",
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
    elif isinstance(first_entry, Mapping):
        owner = parameter_owner(registry_name, first_entry)
    else:
        message = f"Parameter {registry_name!r} does not resolve to one registry mapping."
        raise TypeError(message)
    role_by_material = {family: role for role, families in campaign.material_roles.items() for family in families}
    materials_view: dict[str, Any] = {}
    for family in natural_batches:
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
    consumers = tuple(dict.fromkeys(consumer for name, entry in route_entries for consumer in parameter_consumers(name, campaign.profile.id, entry)))
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
                "generation.cases.generation_cases_config.load_campaign_config -> "
                "generation.contracts.generation_contracts_materials.resolve_material_definition -> "
                "generation.contracts.generation_contracts_materials.project_material_for_profile"
            ),
            "sampler": "generation.cases.generation_cases_sampling.sample_case",
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
        if primary_entry["kind"] == "conditional_interval":
            result.update(
                {
                    "transform": "conditional_log",
                    "support_kind": primary_entry["support_kind"],
                    "support_resolver": primary_entry["support_resolver"],
                    "conditioning_coordinates": copy.deepcopy(primary_entry["conditioning_coordinates"]),
                    "material_inputs": copy.deepcopy(primary_entry["material_inputs"]),
                    "parameter_ood": primary_entry["parameter_ood"],
                }
            )
    return result
