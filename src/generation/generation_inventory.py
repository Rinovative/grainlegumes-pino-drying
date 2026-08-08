"""
===============================================================================
generation_inventory.py
===============================================================================
Audit authoritative generation parameter ownership, dimensions, and consumers.
Responsibilities:
  - Report numerical blocks and non-numerical scientific categories
  - Verify every configured parameter has one owner and one maintained consumer
  - Verify input adapters have explicit sources and execution stays non-scientific
Design principles:
  - Inventory is source owned and derived from active registry/profile contracts
  - Documentation consumes the inventory; tests do not snapshot documentation text
  - Missing, duplicate, or undeclared ownership fails closed
This module does NOT:
  - Supply parameter values, parse COMSOL binaries, or infer export mappings
  - Duplicate typed registry validation or execute simulations
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from . import generation_materials as materials
from . import generation_profiles as profiles
from . import generation_registry as registry_service

if TYPE_CHECKING:
    from collections.abc import Mapping

COMMON_VALUE_PARAMETERS: Final = frozenset({"eps_min_global", "eps_max_global"})
OPERATION_VALUE_PARAMETERS: Final = frozenset(
    {
        "pressure_bc.mean",
        "pressure_bc.sin_amp",
        "pressure_bc.sin_freq",
        "pressure_bc.sin_phase",
        "pressure_bc.gauss_count",
        "pressure_bc.gauss_amp",
        "pressure_bc.gauss_width",
        "pressure_bc.gauss_jitter",
        "pressure_bc.linear_amp",
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
    }
)
FIELD_CONSUMED_PARAMETERS: Final = frozenset(
    {
        "kappa_mean",
        "kappa_cv",
        "bed.structure.coarse_len_rel",
        "bed.structure.fine_len_rel",
        "bed.structure.coarse_weight",
        "bed.structure.fine_weight",
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
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "initial_moisture.structure.coarse_len_rel",
        "initial_moisture.structure.fine_len_rel",
        "initial_moisture.structure.coarse_weight",
        "initial_moisture.structure.fine_weight",
        "initial_moisture.structure.cross_scale_corr",
        "initial_moisture.structure.fine_ani_x",
        "initial_moisture.structure.fine_ani_y",
        "rho_bu_dry_ref",
        "eps_bed_cal_ref",
        "eps_min_global",
        "eps_max_global",
    }
)
OPERATION_CONSUMED_PARAMETERS: Final = frozenset(
    {
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
    }
)
SCALAR_ADAPTER_PARAMETERS: Final = frozenset(profiles.SCALAR_INPUT_FIELDS).difference({"f_wet_dm_max", "A_osw", "B_osw", "C_osw"})
DERIVATION_CONSUMED_PARAMETERS: Final = frozenset(materials.DERIVED_PARAMETERS)
COUPLED_SELECTION_PARAMETERS: Final = frozenset({"oswin"})
ACTIVE_CONSUMED_PARAMETERS: Final = (
    FIELD_CONSUMED_PARAMETERS
    | OPERATION_CONSUMED_PARAMETERS
    | SCALAR_ADAPTER_PARAMETERS
    | DERIVATION_CONSUMED_PARAMETERS
    | COUPLED_SELECTION_PARAMETERS
)

PARAMETER_OWNERS: Final = MappingProxyType(
    {
        name: (
            "common.yaml"
            if name in COMMON_VALUE_PARAMETERS
            else "operations/fixed_bed.yaml"
            if name in OPERATION_VALUE_PARAMETERS
            else "registry.yaml"
            if name in materials.DERIVED_PARAMETERS
            else "materials/<material>.yaml"
        )
        for name in materials.EXPECTED_PARAMETERS | set(materials.OPTIONAL_PARAMETERS)
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
        "p_bc": "generation_fields inlet-pressure boundary",
        "X_0_db_field": "generation_fields initial-moisture field",
        **{
            name: ("common.scientific_fixed_values" if name == "f_wet_dm_max" else "typed parameter registry or deterministic derivation")
            for name in profiles.SCALAR_INPUT_FIELDS
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
EXECUTION_ONLY_VALUES: Final = (
    "site",
    "runtime",
    "retention",
    "cluster",
)
OUTPUT_ONLY_VALUES: Final = tuple(
    dict.fromkeys(
        (
            *profiles.STATIC_FIELD_NAMES,
            *profiles.TRANSIENT_FIELD_NAMES,
            *profiles.GLOBAL_FIELD_NAMES,
            *profiles.FINAL_STATUS_FIELDS,
        )
    )
)


@dataclass(frozen=True, slots=True)
class InventoryReport:
    """One mechanical ownership and consumer audit result."""

    sampled_dimensions_by_block: dict[str, int]
    sampled_parameters_by_block: dict[str, tuple[str, ...]]
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


def audit_parameter_registry(registry: Mapping[str, Mapping[str, Any]]) -> InventoryReport:
    """Audit one fully merged material registry and return its exact inventory."""
    materials.validate_vp2_registry(registry)
    declared = set(registry)
    active_consumers = set(ACTIVE_CONSUMED_PARAMETERS)
    if "density_calibration" in declared:
        active_consumers.add("density_calibration")
    configured_but_unused = tuple(sorted(declared.difference(active_consumers)))
    consumed_but_undeclared = tuple(sorted(active_consumers.difference(declared)))
    if configured_but_unused or consumed_but_undeclared:
        message = (
            "Generation parameter-consumer mismatch: "
            f"configured_but_unused={list(configured_but_unused)}, "
            f"consumed_but_undeclared={list(consumed_but_undeclared)}."
        )
        raise ValueError(message)
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
    adapter_names = set(profiles.SPATIAL_INPUT_FIELDS) | set(profiles.SCALAR_INPUT_FIELDS) | set(profiles.SCHEDULE_FIELDS)
    if adapter_names != set(COMSOL_INPUT_SOURCES):
        message = "Every COMSOL input adapter value must have one explicit source."
        raise ValueError(message)

    sampled_by_block = {
        block: tuple(name for name in names if registry_service.effective_dimension(registry[name]) > 0)
        for block, names in materials.SAMPLING_BLOCKS.items()
    }
    dimensions = {
        block: sum(registry_service.effective_dimension(registry[name]) for name in names) for block, names in materials.SAMPLING_BLOCKS.items()
    }
    material_fixed = tuple(
        name for name, entry in registry.items() if parameter_owner(name) == "materials/<material>.yaml" and entry["kind"] == "fixed"
    )
    return InventoryReport(
        sampled_dimensions_by_block=dimensions,
        sampled_parameters_by_block=sampled_by_block,
        derived_quantities=tuple(name for name, entry in registry.items() if entry["kind"] == "derived"),
        coupled_selections=tuple(name for name, entry in registry.items() if entry["kind"] in {"parameter_set", "paired_parameter_set"}),
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
        reports[batch.batch_name] = audit_parameter_registry(registry)
    return reports
