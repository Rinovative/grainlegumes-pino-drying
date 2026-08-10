"""
===============================================================================
generation_sentinels.py
===============================================================================
Run deterministic no-COMSOL sentinels over both maintained VP2 profiles.
Responsibilities:
  - Exercise all six families under the exact 28-D steady and 54-D transient plans
  - Cover every configured profile-active OOD unit and complete atomic record
  - Validate fields, schedules, moisture, density, and psychrometric invariants
Design principles:
  - Sentinels use isolated derived views and never mutate canonical campaigns
  - Scientific support guards remain active and their failures stay visible
  - Deterministic replay and bounded regeneration retain every scientific guard
This module does NOT:
  - Start COMSOL, mutate campaign YAML, authorize production, or fit parameters
  - Create alternate sampling modes or family-specific evaluation behavior
===============================================================================
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from src import common

from . import generation_config as config_service
from . import generation_fields as field_service
from . import generation_inventory as inventory_service
from . import generation_materials as materials
from . import generation_profiles as profiles
from . import generation_registry as registry_service
from . import generation_sampling as sampling_service
from . import generation_schedule as schedule_service

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_STATIC_SENTINEL_SCHEMA_VERSION: Final = 3
_PRIMARY_SENTINEL_SEED: Final = 202608090
_ID_SENTINEL_CASE_COUNT: Final = 8
_CANONICAL_CASE_COUNTS: Final = {
    profiles.STEADY_FLOW_PROFILE: 1200,
    profiles.TRANSIENT_DRYING_PROFILE: 660,
}
_SCHEDULE_DIAGNOSTIC_NAMES: Final = (
    "min_T_in_bc",
    "max_T_in_bc",
    "min_omega_in_bc",
    "max_omega_in_bc",
    "min_phi_in_bc",
    "max_phi_in_bc",
    "min_phi_source_air",
    "max_phi_source_air",
    "min_heater_temperature_rise",
    "schedule_rejection_count",
    "schedule_acceptance_attempt",
)
_EXPECTED_ROLES: Final = {
    "seen": ("lentil", "chickpea", "kidney_bean"),
    "near_family_ood": ("field_pea",),
    "far_family_ood": ("rapeseed",),
    "extreme_family_ood": ("sunflower_seed",),
}
_EXPECTED_EVALUATION_REGIMES: Final = (
    "id",
    "parameter_ood",
    "near_family_ood",
    "far_family_ood",
    "extreme_family_ood",
)


def _array_sha256(value: np.ndarray) -> str:
    """Return a deterministic digest for one realized sentinel field."""
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"|float64|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sentinel_ood_allocation(
    batch: config_service.GenerationConfig,
    *,
    case_count: int,
) -> tuple[dict[str, str], ...]:
    """Return the same generic eligible-unit allocation used by campaigns."""
    if batch.sampling_regime != "parameter_ood":
        return ()
    groups = materials.active_ood_groups(batch.profile.id)
    eligible = sampling_service.eligible_ood_units(
        batch.scientific_values["material"],
        groups=groups,
    )
    return sampling_service.allocate_ood_units(eligible, case_count=case_count)


def _assignments(
    batch: config_service.GenerationConfig,
    *,
    case_count: int,
) -> dict[int, dict[str, Any]]:
    """Return deterministic sentinel-only assignments for one source batch."""
    allocation = _sentinel_ood_allocation(batch, case_count=case_count)
    assignments: dict[int, dict[str, Any]] = {}
    for case_index in range(1, case_count + 1):
        allocated = allocation[case_index - 1] if allocation else None
        group = None if allocated is None else allocated["ood_group"]
        assignments[case_index] = {
            "case_index": case_index,
            "regime_index": case_index - 1,
            "assignment_role": "static_sentinel",
            "material_family": batch.material_family,
            "sampling_regime": batch.sampling_regime,
            "ood_group": group,
            "ood_unit_id": None if allocated is None else allocated["unit_id"],
            "ood_units_per_case": 1 if allocated is not None else 0,
        }
    return assignments


def _sentinel_view(
    batch: config_service.GenerationConfig,
    *,
    case_count: int,
    seed_base: int,
) -> config_service.GenerationConfig:
    """Return an isolated executable sampling view without changing campaign state."""
    assignments = _assignments(batch, case_count=case_count)
    scientific = copy.deepcopy(batch.scientific_values)
    registry = scientific["material"]["parameter_registry"]
    blocks = materials.active_sampling_blocks(batch.profile.id)
    scientific["case_count"] = case_count
    scientific["assignments"] = [assignments[index] for index in assignments]
    if batch.sampling_regime == "parameter_ood":
        groups = materials.active_ood_groups(batch.profile.id)
        eligible = sampling_service.eligible_ood_units(
            scientific["material"],
            groups=groups,
        )
        allocation = _sentinel_ood_allocation(batch, case_count=case_count)
        scientific["parameter_ood"]["eligible_units"] = [dict(unit) for unit in eligible]
        scientific["parameter_ood"]["case_allocation"] = [{"case_index": index, **dict(unit)} for index, unit in enumerate(allocation, start=1)]
        counts = dict.fromkeys((unit["unit_id"] for unit in eligible), 0)
        for unit in allocation:
            counts[unit["unit_id"]] += 1
        scientific["parameter_ood"]["allocation_counts"] = counts
    scientific["sampling"]["seed_base"] = seed_base
    scientific["sampling"]["blocks"] = sampling_service.build_sampling_plan(
        registry=registry,
        case_count=case_count,
        seed_base=seed_base,
        method=str(scientific["sampling"]["method"]),
        blocks=blocks,
    )
    digest = common.serialization.canonical_json_sha256(scientific)
    return replace(
        batch,
        scientific_values=scientific,
        case_indices=tuple(assignments),
        seed_base=seed_base,
        assignments=assignments,
        scientific_config_digest=digest,
        case_input_config_digest=digest,
        batch_identity=digest,
        batch_id=f"static_sentinel__{batch.batch_name}__{digest[:16]}",
    )


def _subseeds(batch: config_service.GenerationConfig, case_index: int) -> dict[str, int]:
    """Return deterministic field and schedule substreams for one sentinel."""
    if batch.seed_base is None:
        message = "Static sentinel view unexpectedly lacks a seed."
        raise RuntimeError(message)
    return {
        name: config_service.derive_seed(batch.seed_base, "static_sentinel_case", str(case_index), name)
        for name in ("bed", "pressure_bc", "initial_moisture", "schedule_shared", "schedule_independent")
    }


def _design_change_evidence(batch: config_service.GenerationConfig) -> dict[str, Any]:
    """Require every unit-design coordinate and sampled parameter to vary."""
    registry = batch.scientific_values["material"]["parameter_registry"]
    coordinate_changes: dict[str, bool] = {}
    labels = materials.sampling_coordinate_labels(registry, batch.profile.id)
    label_offset = 0
    for plan in batch.scientific_values["sampling"]["blocks"].values():
        design = sampling_service.unit_design(
            batch.scientific_values["sampling"]["method"],
            count=len(batch.case_indices),
            dimensions=int(plan["effective_dimension"]),
            seed=int(plan["design_seed"]),
        )
        block_labels = labels[label_offset : label_offset + design.shape[1]]
        label_offset += design.shape[1]
        for column, label in enumerate(block_labels):
            changed = len({float(value) for value in design[:, column]}) > 1
            if not changed:
                message = f"Static unit-design coordinate {label!r} did not vary."
                raise RuntimeError(message)
            coordinate_changes[label] = True
    if label_offset != len(labels):
        message = "Static coordinate labels disagree with the profile design dimension."
        raise RuntimeError(message)
    samples = [sampling_service.sample_case(batch, case_index) for case_index in batch.case_indices]
    parameter_changes: dict[str, bool] = {}
    for block in materials.active_sampling_blocks(batch.profile.id):
        for name in materials.SAMPLING_BLOCKS[block]:
            serialized = {common.serialization.canonical_json_sha256(sample.values[name]) for sample in samples}
            changed = len(serialized) > 1
            if not changed:
                message = f"Static sampled parameter {name!r} did not vary across the bounded design."
                raise RuntimeError(message)
            parameter_changes[name] = True
    return {
        "coordinate_count": len(labels),
        "coordinate_labels": list(labels),
        "unit_design_coordinate_changed": coordinate_changes,
        "sampled_parameter_changed": parameter_changes,
    }


def _diagnostic_fields(
    batch: config_service.GenerationConfig,
    case_index: int,
    sample: sampling_service.CaseSample,
) -> field_service.SpatialFields:
    """Generate one field realization under its exact sampled OOD attribution."""
    seeds = _subseeds(batch, case_index)
    family = batch.scientific_values["material"]
    moisture_bounds = None
    field_seed_names: tuple[str, ...] = ("bed", "pressure_bc")
    if batch.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        field_seed_names = (*field_seed_names, "initial_moisture")
        moisture_bounds = materials.initial_moisture_generation_bounds(
            family,
            sample.values,
            active_ood_unit=sample.ood_provenance["active_unit_id"],
        )
    return field_service.generate_spatial_fields(
        batch.profile.id,
        batch.scientific_values["grid"],
        sample.values,
        seeds={name: seeds[name] for name in field_seed_names},
        family_bounds=moisture_bounds,
        packing_porosity_mean_support=family["packing_porosity_mean_support"],
        material_kappa_nominal=float(family["parameter_registry"]["kappa_mean"]["nominal"]),
        active_ood_unit=sample.ood_provenance["active_unit_id"],
    )


def _realization_evidence(
    batch: config_service.GenerationConfig,
    case_index: int,
) -> dict[str, Any]:
    """Generate one full-grid profile realization while preserving guard evidence."""
    sample = sampling_service.sample_case(batch, case_index)
    fields = _diagnostic_fields(
        batch,
        case_index,
        sample,
    )
    determinant_minimum = float(fields.metadata["permeability"]["determinant_min"])
    pressure_minimum = float(fields.metadata["pressure_boundary"]["minimum"])
    if determinant_minimum <= 0 or pressure_minimum <= 0:
        message = "Static field sentinel violated SPD permeability or positive inlet pressure."
        raise ValueError(message)
    porosity = fields.columns["eps_bed"]
    porosity_metadata = fields.metadata["porosity"]
    result: dict[str, Any] = {
        "case_index": case_index,
        "material_family": batch.material_family,
        "simulation_profile": batch.profile.id,
        "sampling_regime": batch.sampling_regime,
        "ood": copy.deepcopy(sample.ood_provenance),
        "shape": list(fields.shape),
        "field_sha256": {name: _array_sha256(values) for name, values in fields.columns.items()},
        "permeability_determinant_min": determinant_minimum,
        "pressure_inlet_min": pressure_minimum,
        "porosity": copy.deepcopy(porosity_metadata),
        "porosity_guard_failure": None,
    }
    if batch.profile.id == profiles.STEADY_FLOW_PROFILE:
        return result
    fixed = batch.scientific_values["scientific_fixed_values"]
    seeds = _subseeds(batch, case_index)
    schedule = schedule_service.generate_schedule(
        sample.values,
        batch.scientific_values["time"],
        fixed,
        seeds={name: seeds[name] for name in ("schedule_shared", "schedule_independent")},
    )
    temperature = schedule.values[:, 1]
    humidity_ratio = schedule.values[:, 2]
    relative_humidity = schedule.values[:, 3]
    if (
        np.any((temperature < fixed["T_in_min"]) | (temperature > fixed["T_in_max"]))
        or np.any((humidity_ratio < fixed["omega_min"]) | (humidity_ratio > fixed["omega_max"]))
        or np.any((relative_humidity < fixed["phi_operational_min"]) | (relative_humidity > fixed["phi_operational_max"]))
    ):
        message = "Static schedule sentinel escaped its temperature, humidity-ratio, or RH envelope."
        raise ValueError(message)
    schedule_diagnostics = {name: schedule.metadata[name] for name in _SCHEDULE_DIAGNOSTIC_NAMES}
    if (
        schedule.metadata["column_order"] != ["t", "T_in_bc", "omega_in_bc", "phi_in_bc"]
        or schedule_diagnostics["min_phi_source_air"] <= 0.0
        or schedule_diagnostics["max_phi_source_air"] > 1.0
        or schedule_diagnostics["min_heater_temperature_rise"] < 0.0
        or schedule_diagnostics["schedule_rejection_count"] != schedule_diagnostics["schedule_acceptance_attempt"] - 1
    ):
        message = "Static schedule sentinel violated heater-only diagnostics or the four-column contract."
        raise ValueError(message)
    if float(sample.values["schedule.event_duration_rel"]) < 2.0 * float(sample.values["schedule.event_width_rel"]):
        message = "Static schedule sentinel violated duration >= 2*width."
        raise ValueError(message)
    moisture_metadata = fields.metadata["initial_moisture"]
    target_constraint = batch.scientific_values["material"]["initial_moisture_field_constraint"]
    if float(moisture_metadata["minimum"]) < float(target_constraint["minimum_db"]) - 1e-12:
        message = "Static initial-moisture field violates the supplied target-separation guard."
        raise ValueError(message)
    calibration_porosity = float(sample.values["eps_bed_cal_ref"])
    dry_density = float(sample.values["rho_bu_dry_ref"]) * (1.0 - porosity) / (1.0 - calibration_porosity)
    if not np.isfinite(dry_density).all() or np.any(dry_density <= 0):
        message = "Static density formula produced nonpositive or non-finite values."
        raise ValueError(message)
    oswin_ratio = float(sample.values["A_osw"]) + float(sample.values["B_osw"]) * (float(sample.values["T_init"]) - 273.15)
    oswin_value = 0.01 * oswin_ratio * (0.5 / (1.0 - 0.5)) ** float(sample.values["C_osw"])
    if not math.isfinite(oswin_value) or oswin_value <= 0:
        message = "Static Oswin equilibrium sentinel is nonpositive or non-finite."
        raise ValueError(message)
    result.update(
        {
            "initial_moisture": copy.deepcopy(moisture_metadata),
            "dry_bulk_density_min": float(np.min(dry_density)),
            "oswin_midpoint_X_eq_db": oswin_value,
            "schedule_shape": list(schedule.values.shape),
            "schedule_temperature_range": [float(np.min(temperature)), float(np.max(temperature))],
            "schedule_humidity_ratio_range": [float(np.min(humidity_ratio)), float(np.max(humidity_ratio))],
            "schedule_relative_humidity_range": [float(np.min(relative_humidity)), float(np.max(relative_humidity))],
            "schedule_diagnostics": schedule_diagnostics,
            "event_duration_at_least_twice_width": True,
        }
    )
    return result


def _alternate_interval_value(entry: Mapping[str, Any], current: float) -> float:
    """Return the configured interval endpoint farthest from one current value."""
    candidates = (float(entry["lower"]), float(entry["upper"]))
    return max(candidates, key=lambda value: abs(value - current))


def _maximum_field_difference(left: field_service.SpatialFields, right: field_service.SpatialFields, names: tuple[str, ...]) -> float:
    """Return the largest absolute field change over selected columns."""
    return max(float(np.max(np.abs(left.columns[name] - right.columns[name]))) for name in names)


def _field_coupling_evidence(
    batch: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Prove scalar coupling effects and the shared-background local contract."""
    from . import generation_porosity as porosity_service  # noqa: PLC0415 -- keeps sentinel dependencies explicit

    sentinel = _sentinel_view(
        batch,
        case_count=2,
        seed_base=config_service.derive_seed(
            _PRIMARY_SENTINEL_SEED,
            batch.profile.id,
            "field_coupling",
        ),
    )
    sample = sampling_service.sample_case(sentinel, 1)
    baseline = _diagnostic_fields(sentinel, 1, sample)
    material = sentinel.scientific_values["material"]
    registry = material["parameter_registry"]

    def generated(changes: Mapping[str, float]) -> field_service.SpatialFields:
        values = copy.deepcopy(sample.values)
        values.update(changes)
        return _diagnostic_fields(sentinel, 1, replace(sample, values=values))

    support = registry_service.resolve_conditional_support(
        registry[porosity_service.ANCHOR_PARAMETER_NAME],
        values=sample.values,
        material_contract=material,
    )
    natural = support["id_interval"]
    lower_log = math.log(float(natural["lower"]))
    upper_log = math.log(float(natural["upper"]))
    current_factor = float(sample.values[porosity_service.ANCHOR_PARAMETER_NAME])
    candidate_factors = (
        math.exp(lower_log + 0.25 * (upper_log - lower_log)),
        math.exp(lower_log + 0.75 * (upper_log - lower_log)),
    )
    alternate_factor = max(candidate_factors, key=lambda value: abs(value - current_factor))
    factor_fields = generated({porosity_service.ANCHOR_PARAMETER_NAME: alternate_factor})

    current_kappa = float(sample.values["kappa_mean"])
    kappa_candidates = (0.995 * current_kappa, 1.005 * current_kappa)
    kappa_alternate = max(kappa_candidates, key=lambda value: abs(value - current_kappa))
    kappa_fields = generated({"kappa_mean": kappa_alternate})

    independent_names = (
        "kappa_cv",
        "bed.perturbations.amplitude",
        "permeability.anisotropy.strength",
    )
    independent: dict[str, Any] = {}
    for name in independent_names:
        alternate = _alternate_interval_value(registry[name], float(sample.values[name]))
        fields = generated({name: alternate})
        porosity_identical = np.array_equal(fields.columns["eps_bed"], baseline.columns["eps_bed"])
        permeability_difference = _maximum_field_difference(baseline, fields, ("Kxx", "Kxy", "Kyy"))
        if not porosity_identical or permeability_difference <= 0.0:
            message = f"Local porosity incorrectly depends on permeability-only control {name!r}."
            raise ValueError(message)
        independent[name] = {
            "eps_bed_identical": True,
            "permeability_max_abs_difference": permeability_difference,
        }

    shared_name = "bed.structure.coarse_len_rel"
    shared_fields = generated(
        {
            shared_name: _alternate_interval_value(
                registry[shared_name],
                float(sample.values[shared_name]),
            )
        }
    )
    shared_porosity_difference = _maximum_field_difference(baseline, shared_fields, ("eps_bed",))
    shared_permeability_difference = _maximum_field_difference(baseline, shared_fields, ("Kxx", "Kxy", "Kyy"))
    smooth_name = "porosity.smooth_len_rel"
    smooth_fields = generated(
        {
            smooth_name: _alternate_interval_value(
                registry[smooth_name],
                float(sample.values[smooth_name]),
            )
        }
    )
    amplitude_name = "porosity.texture_amp"
    amplitude_fields = generated(
        {
            amplitude_name: _alternate_interval_value(
                registry[amplitude_name],
                float(sample.values[amplitude_name]),
            )
        }
    )
    factor_difference = _maximum_field_difference(baseline, factor_fields, ("eps_bed",))
    kappa_difference = _maximum_field_difference(baseline, kappa_fields, ("eps_bed",))
    smooth_difference = _maximum_field_difference(baseline, smooth_fields, ("eps_bed",))
    amplitude_difference = _maximum_field_difference(baseline, amplitude_fields, ("eps_bed",))
    if (
        min(
            shared_porosity_difference,
            shared_permeability_difference,
            factor_difference,
            kappa_difference,
            smooth_difference,
            amplitude_difference,
        )
        <= 0.0
    ):
        message = "One corrected porosity or shared-background control has no observable static field effect."
        raise ValueError(message)
    return {
        "texture_source": baseline.metadata["porosity"]["texture_source"],
        "background_field_sha256": baseline.metadata["porosity"]["background_field_sha256"],
        "kappa_mean_eps_bed_max_abs_difference": kappa_difference,
        "kc_anchor_factor_eps_bed_max_abs_difference": factor_difference,
        "shared_background_control": {
            "name": shared_name,
            "eps_bed_max_abs_difference": shared_porosity_difference,
            "permeability_max_abs_difference": shared_permeability_difference,
        },
        "porosity_smooth_len_max_abs_difference": smooth_difference,
        "porosity_texture_amp_max_abs_difference": amplitude_difference,
        "permeability_only_controls": independent,
        "shared_statistical_backbone_confirmed": True,
        "fixed_pointwise_correlation_required": False,
    }


def _nominal_anchor_evidence(
    batch: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Prove the material nominal factor identity and conditional support contract."""
    material = batch.scientific_values["material"]
    registry = material["parameter_registry"]
    entry = registry["porosity.kc_anchor_factor"]
    values = {
        "kappa_mean": float(registry["kappa_mean"]["nominal"]),
        "eps_bed_cal_ref": float(registry["eps_bed_cal_ref"]["value"]),
        "eps_min_global": float(registry["eps_min_global"]["value"]),
        "eps_max_global": float(registry["eps_max_global"]["value"]),
    }
    support = registry_service.resolve_conditional_support(
        entry,
        values=values,
        material_contract=material,
    )
    coefficient = float(support["A_KC_reference"])
    from . import generation_porosity as porosity_service  # noqa: PLC0415 -- keeps sentinel dependencies explicit

    reference = porosity_service.solve_reference_porosity(
        values["kappa_mean"],
        coefficient,
        1.0,
        eps_min_global=values["eps_min_global"],
        eps_max_global=values["eps_max_global"],
    )
    calibration = values["eps_bed_cal_ref"]
    if not math.isclose(reference, calibration, rel_tol=0.0, abs_tol=2e-15):
        message = f"Nominal Kozeny-Carman identity failed for {batch.material_family!r}."
        raise ValueError(message)
    natural = support["id_interval"]
    if not 0.0 < float(natural["lower"]) < 1.0 < float(natural["upper"]):
        message = f"Nominal anchor factor is outside conditional natural support for {batch.material_family!r}."
        raise ValueError(message)
    return {
        "eps_bed_cal_ref": calibration,
        "packing_porosity_mean_support": copy.deepcopy(material["packing_porosity_mean_support"]),
        "material_kappa_nominal": values["kappa_mean"],
        "A_KC_reference": coefficient,
        "nominal_anchor_factor": 1.0,
        "recovered_eps_reference": reference,
        "nominal_recovery_absolute_error": abs(reference - calibration),
        "conditional_id_interval": [float(natural["lower"]), float(natural["upper"])],
        "available_ood_directions": [str(tail["direction"]) for tail in support["available_ood_tails"]],
        "unavailable_ood_directions": copy.deepcopy(support["unavailable_ood_directions"]),
        "status": "pass",
    }


def _anchor_ood_evidence(
    batch: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Realize every feasible conditional anchor tail with one active OOD unit."""
    from . import generation_porosity as porosity_service  # noqa: PLC0415 -- keeps sentinel dependencies explicit

    sentinel = _sentinel_view(
        batch,
        case_count=2,
        seed_base=config_service.derive_seed(
            _PRIMARY_SENTINEL_SEED,
            batch.profile.id,
            batch.material_family,
            "anchor_ood",
        ),
    )
    sample = sampling_service.sample_case(sentinel, 1)
    material = sentinel.scientific_values["material"]
    entry = material["parameter_registry"][porosity_service.ANCHOR_PARAMETER_NAME]
    support = registry_service.resolve_conditional_support(
        entry,
        values=sample.values,
        material_contract=material,
    )
    packing = material["packing_porosity_mean_support"]
    lower = float(packing["lower"])
    upper = float(packing["upper"])
    directions: dict[str, Any] = {}
    for tail in support["available_ood_tails"]:
        direction = str(tail["direction"])
        factor = math.exp(0.5 * (float(tail["transformed_lower"]) + float(tail["transformed_upper"])))
        values = copy.deepcopy(sample.values)
        values[porosity_service.ANCHOR_PARAMETER_NAME] = factor
        ood = copy.deepcopy(sample.ood_provenance)
        ood.update(
            {
                "group": "bed",
                "active_ood_group": "bed",
                "selected_units": [porosity_service.ANCHOR_PARAMETER_NAME],
                "active_unit_id": porosity_service.ANCHOR_PARAMETER_NAME,
                "active_record_id": f"{porosity_service.ANCHOR_PARAMETER_NAME}__ood_{direction}",
                "units_per_case": 1,
                "natural_support_state": "parameter_ood",
            }
        )
        ood_sample = replace(sample, values=values, ood_provenance=ood)
        fields = _diagnostic_fields(sentinel, 1, ood_sample)
        diagnostics = fields.metadata["porosity"]
        reference = float(diagnostics["eps_reference"])
        realized = float(diagnostics["eps_bed_mean"])
        correct_direction = reference > upper and realized > upper if direction == "lower" else reference < lower and realized < lower
        if (
            not correct_direction
            or diagnostics["active_anchor_support_kind"] != f"ood_{direction}"
            or diagnostics["material_support_departure_cause"] != porosity_service.ANCHOR_PARAMETER_NAME
            or float(tail["transformed_gap_fraction"]) < 0.15 - 1e-12
            or float(tail["transformed_width_fraction"]) < 0.25 - 1e-12
            or float(diagnostics["eps_bed_min"]) < float(sample.values["eps_min_global"])
            or float(diagnostics["eps_bed_max"]) > float(sample.values["eps_max_global"])
        ):
            message = f"Conditional anchor OOD sentinel failed for {batch.material_family!r} direction {direction!r}."
            raise ValueError(message)
        directions[direction] = {
            "factor_interval": [float(tail["lower"]), float(tail["upper"])],
            "sampled_factor": factor,
            "reference_porosity_range": copy.deepcopy(tail["reference_porosity_range"]),
            "observed_eps_reference": reference,
            "observed_eps_bed_mean": realized,
            "transformed_gap": float(tail["transformed_gap"]),
            "transformed_width": float(tail["transformed_width"]),
            "transformed_gap_fraction": float(tail["transformed_gap_fraction"]),
            "transformed_width_fraction": float(tail["transformed_width_fraction"]),
            "physical_interpretation": tail["physical_interpretation"],
            "one_active_ood_unit": ood["selected_units"] == [porosity_service.ANCHOR_PARAMETER_NAME],
            "realized_mean_retained_ood_direction": True,
        }
    if not directions:
        message = f"Seen material {batch.material_family!r} has no feasible conditional anchor OOD direction."
        raise ValueError(message)
    return {
        "available_directions": list(directions),
        "unavailable_directions": copy.deepcopy(support["unavailable_ood_directions"]),
        "directions": directions,
        "status": "pass",
    }


def _downstream_ood_attribution_evidence(
    batch: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Prove non-anchor OOD responses do not create a second OOD unit."""
    requested = [
        "kappa_mean",
        "porosity.smooth_len_rel",
        "porosity.texture_amp",
    ]
    if batch.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        requested.append("density_calibration")
    results: dict[str, Any] = {}
    for unit in requested:
        matches = [case_index for case_index in batch.case_indices if batch.case_assignment(case_index)["ood_unit_id"] == unit]
        if not matches:
            message = f"Static attribution sentinel could not find allocated OOD unit {unit!r}."
            raise ValueError(message)
        case_index = matches[0]
        sample = sampling_service.sample_case(batch, case_index)
        if sample.ood_provenance["selected_units"] != [unit]:
            message = f"Static attribution sentinel {unit!r} is not the sole active OOD unit."
            raise ValueError(message)
        fields = _diagnostic_fields(batch, case_index, sample)
        porosity = fields.metadata["porosity"]
        conditional = sample.conditional_supports["porosity.kc_anchor_factor"]
        if (
            conditional["support_kind"] != "natural"
            or porosity["active_anchor_support_kind"] != "natural"
            or porosity["material_support_departure_cause"] is not None
            or not porosity["eps_bed_within_material_natural_support"]
        ):
            message = f"Downstream porosity response to OOD unit {unit!r} was double-counted or escaped natural support."
            raise ValueError(message)
        results[unit] = {
            "case_index": case_index,
            "selected_units": copy.deepcopy(sample.ood_provenance["selected_units"]),
            "anchor_support_kind": conditional["support_kind"],
            "eps_reference": porosity["eps_reference"],
            "eps_bed_mean": porosity["eps_bed_mean"],
            "eps_bed_within_material_natural_support": True,
            "material_support_departure_cause": None,
            "one_active_ood_unit": True,
        }
    return results


def _ood_candidates(batch: config_service.GenerationConfig) -> dict[str, tuple[str, ...]]:
    """Return every configured profile-active OOD unit by physical group."""
    groups = materials.active_ood_groups(batch.profile.id)
    candidates: dict[str, list[str]] = {group: [] for group in groups}
    for unit in sampling_service.eligible_ood_units(
        batch.scientific_values["material"],
        groups=groups,
    ):
        candidates[unit["ood_group"]].append(unit["unit_id"])
    return {group: tuple(names) for group, names in candidates.items()}


def _ood_evidence(batch: config_service.GenerationConfig) -> dict[str, Any]:
    """Cover every OOD unit with one-group/one-unit selection evidence."""
    candidates = _ood_candidates(batch)
    case_count = sum(len(names) for names in candidates.values())
    sentinel = _sentinel_view(
        batch,
        case_count=case_count,
        seed_base=config_service.derive_seed(_PRIMARY_SENTINEL_SEED, batch.profile.id, "parameter_ood"),
    )
    group_cases: dict[str, list[int]] = {group: [] for group in candidates}
    selected_units: set[str] = set()
    complete_records: set[str] = set()
    for case_index in sentinel.case_indices:
        sample = sampling_service.sample_case(sentinel, case_index)
        group = sample.ood_provenance["group"]
        selected = sample.ood_provenance["selected_units"]
        details = sample.ood_provenance["selections"]
        if group not in group_cases or len(selected) != 1 or set(details) != set(selected):
            message = f"Parameter-OOD sentinel case {case_index} violates one-group/one-unit provenance."
            raise ValueError(message)
        group_cases[group].append(case_index)
        selected_units.update(selected)
        detail = details[selected[0]]
        distance = detail.get("transform_space_distance")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or float(distance) <= 0:
            message = f"OOD sentinel {selected[0]!r} has no positive normalized transform-space distance."
            raise ValueError(message)
        if detail["selection_kind"].startswith("complete_"):
            complete_records.add(selected[0])
    expected_units = {name for names in candidates.values() for name in names}
    if selected_units != expected_units:
        missing = sorted(expected_units.difference(selected_units))
        message = f"OOD sentinel missed configured units {missing}."
        raise ValueError(message)
    representative_realizations = {group: _realization_evidence(sentinel, case_indices[0]) for group, case_indices in group_cases.items()}
    return {
        "case_count": case_count,
        "candidates_by_group": {group: list(names) for group, names in candidates.items()},
        "group_cases": group_cases,
        "selected_units": sorted(selected_units),
        "complete_atomic_records": sorted(complete_records),
        "representative_realizations": representative_realizations,
    }


def _validate_campaign(campaign: config_service.CampaignConfig, profile_id: str) -> None:
    """Require one canonical six-family, five-regime maintained campaign."""
    inventory = tuple(family for role in _EXPECTED_ROLES for family in campaign.material_roles[role])
    if (
        campaign.campaign_purpose != "family_generalization"
        or campaign.profile.id != profile_id
        or campaign.material_roles != _EXPECTED_ROLES
        or campaign.evaluation_regimes != _EXPECTED_EVALUATION_REGIMES
        or inventory != materials.MATERIAL_FAMILIES
        or campaign.total_case_count != _CANONICAL_CASE_COUNTS[profile_id]
    ):
        message = f"Static sentinel input is not the canonical {profile_id!r} six-family campaign."
        raise ValueError(message)


def run_static_sentinels(
    steady_campaign_path: Path | str,
    transient_campaign_path: Path | str,
) -> dict[str, Any]:
    """Run complete deterministic implementation sentinels without COMSOL."""
    campaigns = {
        profiles.STEADY_FLOW_PROFILE: config_service.load_campaign_config(
            steady_campaign_path,
            require_executable=False,
        ),
        profiles.TRANSIENT_DRYING_PROFILE: config_service.load_campaign_config(
            transient_campaign_path,
            require_executable=False,
        ),
    }
    profile_evidence: dict[str, Any] = {}
    support_failures: list[dict[str, Any]] = []
    for profile_id, campaign in campaigns.items():
        _validate_campaign(campaign, profile_id)
        family_evidence: dict[str, Any] = {}
        for family in materials.MATERIAL_FAMILIES:
            source = campaign.batch(f"{profile_id}__{family}__natural")
            sentinel = _sentinel_view(
                source,
                case_count=_ID_SENTINEL_CASE_COUNT,
                seed_base=config_service.derive_seed(_PRIMARY_SENTINEL_SEED, profile_id, "material", family),
            )
            report = inventory_service.audit_parameter_registry(
                sentinel.scientific_values["material"]["parameter_registry"],
                profile_id=profile_id,
            )
            change_evidence = _design_change_evidence(sentinel)
            realization = _realization_evidence(sentinel, 1)
            if realization["porosity_guard_failure"] is not None:
                support_failures.append(
                    {
                        "simulation_profile": profile_id,
                        "material_family": family,
                        "message": realization["porosity_guard_failure"],
                        "porosity": realization["porosity"],
                    }
                )
            family_evidence[family] = {
                "inventory": asdict(report),
                "design_change_evidence": change_evidence,
                "nominal_anchor": _nominal_anchor_evidence(source),
                "realization": realization,
            }
        parameter_source = campaign.batch(f"{profile_id}__lentil__parameter_ood")
        anchor_ood = {family: _anchor_ood_evidence(campaign.batch(f"{profile_id}__{family}__natural")) for family in materials.MATERIAL_FAMILIES[:3]}
        parameter_ood = _ood_evidence(parameter_source)
        parameter_ood["eligible_unit_count"] = len(parameter_ood["selected_units"])
        parameter_ood["downstream_porosity_attribution"] = _downstream_ood_attribution_evidence(parameter_source)
        profile_evidence[profile_id] = {
            "family_evidence": family_evidence,
            "field_coupling": _field_coupling_evidence(campaign.batch(f"{profile_id}__lentil__natural")),
            "anchor_ood_by_seen_material": anchor_ood,
            "parameter_ood": parameter_ood,
        }
    status = "pass" if not support_failures else "blocked_by_scientific_sanity_guard"
    return {
        "schema_kind": "vp2_static_generator_sentinels",
        "schema_version": _STATIC_SENTINEL_SCHEMA_VERSION,
        "status": status,
        "implementation_checks_complete": True,
        "decision_sha256": materials.VP2_DECISION_SHA256,
        "material_families": list(materials.MATERIAL_FAMILIES),
        "material_roles": {name: list(values) for name, values in _EXPECTED_ROLES.items()},
        "evaluation_regimes": list(_EXPECTED_EVALUATION_REGIMES),
        "profile_dimensions": {
            profiles.STEADY_FLOW_PROFILE: materials.STEADY_DIMENSION,
            profiles.TRANSIENT_DRYING_PROFILE: materials.TRANSIENT_DIMENSION,
        },
        "profile_evidence": profile_evidence,
        "scientific_support_guard_failures": support_failures,
        "comsol_started": False,
        "production_state_mutated": False,
    }
