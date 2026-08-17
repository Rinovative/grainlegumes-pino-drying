"""
===============================================================================
generation_validation_sentinels.py
===============================================================================
Run deterministic no-COMSOL sentinels over both maintained VP2 profiles.
Responsibilities:
  - Exercise every configured material under each resolved profile sampling plan
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
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from src import common, domain
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_fields as field_service
from src.generation.cases import generation_cases_sampling as sampling_service
from src.generation.cases import generation_cases_schedule as schedule_service
from src.generation.cases import generation_cases_seeding as seeding
from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_porosity as porosity_service
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.publication import generation_publication_inventory as inventory_service

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_STATIC_SENTINEL_SCHEMA_VERSION: Final = 1
_PRIMARY_SENTINEL_SEED: Final = 202608090
_ID_SENTINEL_CASE_COUNT: Final = 8


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
    groups = materials.active_ood_groups(batch.scientific_values["material"]["parameter_registry"], batch.profile.id)
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
    blocks = materials.active_sampling_blocks(batch.scientific_values["material"]["parameter_registry"], batch.profile.id)
    scientific["case_count"] = case_count
    scientific["assignments"] = [assignments[index] for index in assignments]
    if batch.sampling_regime == "parameter_ood":
        groups = materials.active_ood_groups(batch.scientific_values["material"]["parameter_registry"], batch.profile.id)
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
    block_parameters = {block: tuple(str(name) for name in scientific["sampling"]["blocks"][block]["parameters"]) for block in blocks}
    scientific["sampling"]["seed_base"] = seed_base
    scientific["sampling"]["blocks"] = sampling_service.build_sampling_plan(
        registry=registry,
        case_count=case_count,
        seed_base=seed_base,
        method=str(scientific["sampling"]["method"]),
        blocks=blocks,
        block_parameters=block_parameters,
    )
    scientific_digest = config_service.compute_scientific_config_digest(scientific)
    case_input_digest = config_service.compute_case_input_config_digest(scientific)
    return replace(
        batch,
        scientific_values=scientific,
        case_indices=tuple(assignments),
        seed_base=seed_base,
        assignments=assignments,
        scientific_config_digest=scientific_digest,
        case_input_config_digest=case_input_digest,
        batch_identity=scientific_digest,
        batch_id=config_service.build_batch_id(f"static_sentinel__{batch.batch_name}", scientific_digest),
    )


def _subseeds(batch: config_service.GenerationConfig, case_index: int) -> dict[str, int]:
    """Return deterministic field and schedule substreams for one sentinel."""
    if batch.seed_base is None:
        message = "Static sentinel view unexpectedly lacks a seed."
        raise RuntimeError(message)
    return {
        name: seeding.derive_seed(batch.seed_base, "static_sentinel_case", str(case_index), name)
        for name in ("bed", "pressure_bc", "initial_moisture", "packing_scatter", "schedule_shared", "schedule_independent")
    }


def _design_change_evidence(batch: config_service.GenerationConfig) -> dict[str, Any]:
    """Require every unit-design coordinate and sampled parameter to vary."""
    registry = batch.scientific_values["material"]["parameter_registry"]
    coordinate_changes: dict[str, bool] = {}
    plans = batch.scientific_values["sampling"]["blocks"]
    block_parameters = {block: tuple(str(name) for name in plan["parameters"]) for block, plan in plans.items()}
    labels = materials.sampling_coordinate_labels(
        registry,
        batch.profile.id,
        block_parameters=block_parameters,
    )
    label_offset = 0
    for plan in plans.values():
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
    for plan in plans.values():
        for name in plan["parameters"]:
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
    field_seed_names: tuple[str, ...] = ("bed", "pressure_bc", "packing_scatter")
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
        porosity_coupling=family["porosity_coupling"],
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
    schedule_seeds = {name: seeds[name] for name in ("schedule_shared", "schedule_independent")}
    schedule = schedule_service.generate_schedule(
        sample.values,
        batch.scientific_values["time"],
        fixed,
        seeds=schedule_seeds,
    )
    replay = schedule_service.generate_schedule(
        sample.values,
        batch.scientific_values["time"],
        fixed,
        seeds=schedule_seeds,
    )
    if not np.array_equal(schedule.values, replay.values) or schedule.metadata != replay.metadata:
        message = "Static schedule sentinel failed deterministic replay."
        raise ValueError(message)
    temperature = schedule.values[:, 1]
    humidity_ratio = schedule.values[:, 2]
    relative_humidity_extrema = schedule_service.derived_relative_humidity_extrema(
        temperature,
        humidity_ratio,
        pressure=float(fixed["p_ref"]),
    )
    if (
        np.any((temperature < fixed["T_in_min"]) | (temperature > fixed["T_in_max"]))
        or np.any((humidity_ratio < fixed["omega_min"]) | (humidity_ratio > fixed["omega_max"]))
        or relative_humidity_extrema[0] < fixed["phi_operational_min"]
        or relative_humidity_extrema[1] > fixed["phi_operational_max"]
    ):
        message = "Static schedule sentinel escaped its primitive or continuous derived-RH envelope."
        raise ValueError(message)
    schedule_diagnostics = schedule.diagnostics
    minimum_source_phi = float(cast("float", schedule_diagnostics["min_phi_source_air"]))
    maximum_source_phi = float(cast("float", schedule_diagnostics["max_phi_source_air"]))
    minimum_heater_rise = float(cast("float", schedule_diagnostics["min_heater_temperature_rise"]))
    rejection_count = int(cast("int", schedule_diagnostics["schedule_rejection_count"]))
    acceptance_attempt = int(cast("int", schedule_diagnostics["schedule_acceptance_attempt"]))
    if (
        schedule.metadata["column_order"] != list(profiles.SCHEDULE_FIELDS)
        or minimum_source_phi <= 0.0
        or maximum_source_phi > 1.0
        or minimum_heater_rise < 0.0
        or rejection_count != acceptance_attempt - 1
    ):
        message = "Static schedule sentinel violated heater-only diagnostics or the primitive-column contract."
        raise ValueError(message)
    if float(sample.values["schedule.event_duration_rel"]) < 2.0 * float(sample.values["schedule.event_width_rel"]):
        message = "Static schedule sentinel violated duration >= 2*width."
        raise ValueError(message)
    if (
        float(cast("float", schedule_diagnostics["smooth_scale_intervals"])) < schedule_service.MINIMUM_SMOOTH_SCALE_INTERVALS
        or float(cast("float", schedule_diagnostics["minimum_event_width_intervals"])) < schedule_service.MINIMUM_EVENT_WIDTH_INTERVALS
        or float(cast("float", schedule_diagnostics["event_duration_intervals"])) < schedule_service.MINIMUM_EVENT_DURATION_INTERVALS
    ):
        message = "Static schedule sentinel violated the regular-grid temporal-resolution contract."
        raise ValueError(message)
    for amplitude_name, ratio_name, constant_name in (
        ("T_in_amp", "T_in_amp_realization_ratio", "constant_T_in_bc"),
        ("omega_in_amp", "omega_in_amp_realization_ratio", "constant_omega_in_bc"),
    ):
        amplitude = float(sample.values[amplitude_name])
        ratio = schedule_diagnostics[ratio_name]
        constant = bool(schedule_diagnostics[constant_name])
        if amplitude == 0.0:
            valid = ratio is None and constant
        else:
            valid = ratio is not None and math.isclose(float(ratio), 1.0, rel_tol=0.0, abs_tol=2.0e-12) and not constant
        if not valid:
            message = f"Static schedule sentinel violated exact {amplitude_name} semantics."
            raise ValueError(message)
    correlation_error = schedule_diagnostics["absolute_T_omega_correlation_error"]
    both_vary = float(sample.values["T_in_amp"]) > 0.0 and float(sample.values["omega_in_amp"]) > 0.0
    if (both_vary and (correlation_error is None or float(correlation_error) > schedule_service.CORRELATION_TOLERANCE)) or (
        not both_vary and correlation_error is not None
    ):
        message = "Static schedule sentinel violated exact discrete-node correlation semantics."
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
    oswin_value = float(
        domain.moisture.oswin_equilibrium_dry_basis_moisture(
            0.5,
            float(sample.values["T_init"]),
            a_osw=float(sample.values["A_osw"]),
            b_osw=float(sample.values["B_osw"]),
            c_osw=float(sample.values["C_osw"]),
        )
    )
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
            "schedule_relative_humidity_range": list(relative_humidity_extrema),
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
    """Prove global KC coupling effects and the shared-background local contract."""
    sentinel = _sentinel_view(
        batch,
        case_count=2,
        seed_base=seeding.derive_seed(
            _PRIMARY_SENTINEL_SEED,
            batch.profile.id,
            "field_coupling",
        ),
    )
    sample = sampling_service.sample_case(sentinel, 1)
    baseline = _diagnostic_fields(sentinel, 1, sample)
    material = sentinel.scientific_values["material"]
    registry = material["parameter_registry"]
    coupling = material["porosity_coupling"]

    def generated(changes: Mapping[str, float]) -> field_service.SpatialFields:
        values = copy.deepcopy(sample.values)
        values.update(changes)
        return _diagnostic_fields(sentinel, 1, replace(sample, values=values))

    effective = coupling["effective_joint_permeability_support"]
    effective_lower = float(effective["lower"])
    effective_upper = float(effective["upper"])
    effective_width = effective_upper - effective_lower
    current_kappa = float(sample.values["kappa_mean"])
    kappa_candidates = (
        effective_lower + 0.25 * effective_width,
        effective_lower + 0.75 * effective_width,
    )
    kappa_alternate = max(kappa_candidates, key=lambda value: abs(value - current_kappa))
    kappa_fields = generated({"kappa_mean": kappa_alternate})
    baseline_porosity = baseline.metadata["porosity"]
    kappa_porosity = kappa_fields.metadata["porosity"]
    if baseline_porosity["packing_scatter_z"] != kappa_porosity["packing_scatter_z"]:
        message = "Packing scatter changed while probing deterministic kappa_mean coupling."
        raise ValueError(message)

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
        permeability_difference = _maximum_field_difference(baseline, fields, domain.fields.PERMEABILITY_FIELDS)
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
    shared_porosity_difference = _maximum_field_difference(baseline, shared_fields, domain.fields.POROSITY_FIELDS)
    shared_permeability_difference = _maximum_field_difference(baseline, shared_fields, domain.fields.PERMEABILITY_FIELDS)
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
    kappa_difference = _maximum_field_difference(baseline, kappa_fields, domain.fields.POROSITY_FIELDS)
    smooth_difference = _maximum_field_difference(baseline, smooth_fields, domain.fields.POROSITY_FIELDS)
    amplitude_difference = _maximum_field_difference(baseline, amplitude_fields, domain.fields.POROSITY_FIELDS)
    if (
        min(
            shared_porosity_difference,
            shared_permeability_difference,
            kappa_difference,
            smooth_difference,
            amplitude_difference,
        )
        <= 0.0
    ):
        message = "One corrected porosity or shared-background control has no observable static field effect."
        raise ValueError(message)
    return {
        "texture_source": baseline_porosity["texture_source"],
        "background_field_sha256": baseline_porosity["background_field_sha256"],
        "A_KC_reference": baseline_porosity["A_KC_reference"],
        "packing_scatter_seed": baseline_porosity["packing_scatter_seed"],
        "packing_scatter_z": baseline_porosity["packing_scatter_z"],
        "packing_scatter_fixed_during_kappa_probe": True,
        "kappa_mean_eps_bed_max_abs_difference": kappa_difference,
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


def _kc_coupling_evidence(
    batch: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Prove fixed material calibration and resolved joint-support identities."""
    material = batch.scientific_values["material"]
    registry = material["parameter_registry"]
    coupling = material["porosity_coupling"]
    nominal = float(coupling["material_kappa_nominal"])
    calibration = float(coupling["material_eps_bed_cal_ref"])
    coefficient = float(coupling["A_KC_reference"])
    reference = porosity_service.solve_reference_porosity(
        nominal,
        coefficient,
        eps_min_global=float(registry["eps_min_global"]["value"]),
        eps_max_global=float(registry["eps_max_global"]["value"]),
    )
    if not math.isclose(reference, calibration, rel_tol=0.0, abs_tol=2e-15):
        message = f"Nominal Kozeny-Carman identity failed for {batch.material_family!r}."
        raise ValueError(message)
    derived = porosity_service.derive_reference_coefficient(nominal, calibration)
    if not math.isclose(coefficient, derived, rel_tol=0.0, abs_tol=0.0):
        message = f"Stored Kozeny-Carman coefficient is not canonical for {batch.material_family!r}."
        raise ValueError(message)
    effective = coupling["effective_joint_permeability_support"]
    kappa_entry = registry["kappa_mean"]
    if float(kappa_entry["lower"]) != float(effective["lower"]) or float(kappa_entry["upper"]) != float(effective["upper"]):
        message = f"Registry permeability support is not the resolved joint support for {batch.material_family!r}."
        raise ValueError(message)
    return {
        "eps_bed_cal_ref": calibration,
        "material_kappa_nominal": nominal,
        "A_KC_reference": coefficient,
        "recovered_eps_reference": reference,
        "nominal_recovery_absolute_error": abs(reference - calibration),
        "authored_permeability_support": copy.deepcopy(coupling["authored_permeability_support"]),
        "kc_compatible_permeability_support": copy.deepcopy(coupling["kc_compatible_permeability_support"]),
        "effective_joint_permeability_support": copy.deepcopy(effective),
        "natural_porosity_support": copy.deepcopy(coupling["natural_porosity_support"]),
        "authored_support_narrowed": bool(coupling["authored_support_narrowed"]),
        "kappa_ood_directions": sorted(coupling["kappa_ood_porosity_supports"]),
        "status": "pass",
    }


def _kappa_ood_evidence(
    batch: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Realize every authored permeability tail with one active OOD unit."""
    sentinel = _sentinel_view(
        batch,
        case_count=2,
        seed_base=seeding.derive_seed(
            _PRIMARY_SENTINEL_SEED,
            batch.profile.id,
            batch.material_family,
            "kappa_ood",
        ),
    )
    sample = sampling_service.sample_case(sentinel, 1)
    coupling = sentinel.scientific_values["material"]["porosity_coupling"]
    natural = coupling["natural_porosity_support"]
    natural_lower = float(natural["lower"])
    natural_upper = float(natural["upper"])
    directions: dict[str, Any] = {}
    for direction, tail in sorted(coupling["kappa_ood_porosity_supports"].items()):
        permeability = math.sqrt(float(tail["kappa_lower"]) * float(tail["kappa_upper"]))
        values = copy.deepcopy(sample.values)
        values["kappa_mean"] = permeability
        ood = copy.deepcopy(sample.ood_provenance)
        ood.update(
            {
                "group": "transport_structure",
                "active_ood_group": "transport_structure",
                "selected_units": ["kappa_mean"],
                "active_unit_id": "kappa_mean",
                "active_record_id": f"kappa_mean__ood_{direction}",
                "units_per_case": 1,
                "natural_support_state": "parameter_ood",
            }
        )
        fields = _diagnostic_fields(
            sentinel,
            1,
            replace(sample, values=values, ood_provenance=ood),
        )
        diagnostics = fields.metadata["porosity"]
        reference = float(diagnostics["eps_reference"])
        realized = float(diagnostics["eps_bed_mean"])
        if direction == "lower":
            correct_direction = reference < natural_lower and realized < natural_lower
        else:
            correct_direction = reference > natural_upper and realized > natural_upper
        mapped_lower = porosity_service.solve_reference_porosity(
            float(tail["kappa_lower"]),
            float(coupling["A_KC_reference"]),
            eps_min_global=float(sample.values["eps_min_global"]),
            eps_max_global=float(sample.values["eps_max_global"]),
        )
        mapped_upper = porosity_service.solve_reference_porosity(
            float(tail["kappa_upper"]),
            float(coupling["A_KC_reference"]),
            eps_min_global=float(sample.values["eps_min_global"]),
            eps_max_global=float(sample.values["eps_max_global"]),
        )
        if (
            not correct_direction
            or diagnostics["packing_scatter_support_kind"] != f"kappa_mean_ood_{direction}"
            or not math.isclose(mapped_lower, float(tail["porosity_lower"]), rel_tol=0.0, abs_tol=2e-15)
            or not math.isclose(mapped_upper, float(tail["porosity_upper"]), rel_tol=0.0, abs_tol=2e-15)
            or float(diagnostics["eps_bed_min"]) < float(sample.values["eps_min_global"])
            or float(diagnostics["eps_bed_max"]) > float(sample.values["eps_max_global"])
        ):
            message = f"Permeability OOD sentinel failed for {batch.material_family!r} direction {direction!r}."
            raise ValueError(message)
        directions[direction] = {
            "kappa_interval": [float(tail["kappa_lower"]), float(tail["kappa_upper"])],
            "sampled_kappa_mean": permeability,
            "mapped_porosity_interval": [mapped_lower, mapped_upper],
            "observed_eps_kc_trend": float(diagnostics["eps_kc_trend"]),
            "observed_eps_reference": reference,
            "observed_eps_bed_mean": realized,
            "packing_scatter_z": float(diagnostics["packing_scatter_z"]),
            "one_active_ood_unit": ood["selected_units"] == ["kappa_mean"],
            "realized_mean_retained_ood_direction": True,
        }
    if not directions:
        message = f"Seen material {batch.material_family!r} has no authored kappa_mean OOD direction."
        raise ValueError(message)
    return {
        "available_directions": list(directions),
        "directions": directions,
        "status": "pass",
    }


def _downstream_ood_attribution_evidence(
    batch: config_service.GenerationConfig,
) -> dict[str, Any]:
    """Prove only kappa OOD changes the active porosity support."""
    candidates = _ood_candidates(batch)
    case_count = sum(len(names) for names in candidates.values())
    if case_count <= 0:
        message = f"Seen material {batch.material_family!r} has no eligible parameter-OOD units."
        raise ValueError(message)
    sentinel = _sentinel_view(
        batch,
        case_count=case_count,
        seed_base=seeding.derive_seed(
            _PRIMARY_SENTINEL_SEED,
            batch.profile.id,
            batch.material_family,
            "downstream_ood_attribution",
        ),
    )
    requested = [
        "kappa_mean",
        "porosity.smooth_len_rel",
        "porosity.texture_amp",
    ]
    if batch.profile.id == profiles.TRANSIENT_DRYING_PROFILE:
        requested.append("density_calibration")
    coupling = sentinel.scientific_values["material"]["porosity_coupling"]
    natural = coupling["natural_porosity_support"]
    natural_lower = float(natural["lower"])
    natural_upper = float(natural["upper"])
    results: dict[str, Any] = {}
    for unit in requested:
        matches = [case_index for case_index in sentinel.case_indices if sentinel.case_assignment(case_index)["ood_unit_id"] == unit]
        if not matches:
            message = f"Static attribution sentinel could not find eligible OOD unit {unit!r}."
            raise ValueError(message)
        case_index = matches[0]
        sample = sampling_service.sample_case(sentinel, case_index)
        if sample.ood_provenance["selected_units"] != [unit]:
            message = f"Static attribution sentinel {unit!r} is not the sole active OOD unit."
            raise ValueError(message)
        fields = _diagnostic_fields(sentinel, case_index, sample)
        diagnostics = fields.metadata["porosity"]
        support_kind = str(diagnostics["packing_scatter_support_kind"])
        realized = float(diagnostics["eps_bed_mean"])
        if unit == "kappa_mean":
            retained = (support_kind == "kappa_mean_ood_lower" and realized < natural_lower) or (
                support_kind == "kappa_mean_ood_upper" and realized > natural_upper
            )
        else:
            retained = support_kind == "natural" and natural_lower <= realized <= natural_upper
        if not retained or float(diagnostics["A_KC_reference"]) != float(coupling["A_KC_reference"]):
            message = f"Downstream porosity response to OOD unit {unit!r} has incorrect support attribution."
            raise ValueError(message)
        results[unit] = {
            "case_index": case_index,
            "selected_units": copy.deepcopy(sample.ood_provenance["selected_units"]),
            "packing_scatter_support_kind": support_kind,
            "eps_reference": diagnostics["eps_reference"],
            "eps_bed_mean": realized,
            "A_KC_reference": diagnostics["A_KC_reference"],
            "one_active_ood_unit": True,
        }
    return results


def _ood_candidates(batch: config_service.GenerationConfig) -> dict[str, tuple[str, ...]]:
    """Return every configured profile-active OOD unit by physical group."""
    groups = materials.active_ood_groups(batch.scientific_values["material"]["parameter_registry"], batch.profile.id)
    candidates: dict[str, list[str]] = {group: [] for group in groups}
    for unit in sampling_service.eligible_ood_units(
        batch.scientific_values["material"],
        groups=groups,
    ):
        candidates[unit["ood_group"]].append(unit["unit_id"])
    return {group: tuple(names) for group, names in candidates.items() if names}


def _ood_evidence(batch: config_service.GenerationConfig) -> dict[str, Any]:
    """Cover every OOD unit with one-group/one-unit selection evidence."""
    candidates = _ood_candidates(batch)
    case_count = sum(len(names) for names in candidates.values())
    if case_count <= 0:
        message = f"Seen material {batch.material_family!r} has no eligible parameter-OOD units."
        raise ValueError(message)
    sentinel = _sentinel_view(
        batch,
        case_count=case_count,
        seed_base=seeding.derive_seed(_PRIMARY_SENTINEL_SEED, batch.profile.id, "parameter_ood"),
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


def _campaign_inventory(campaign: config_service.CampaignConfig) -> tuple[str, ...]:
    """Return the configured material inventory in role-declaration order."""
    inventory = campaign.material_inventory
    if not inventory or len(inventory) != len(set(inventory)):
        message = f"Campaign {campaign.campaign_id!r} has an empty or duplicate material inventory."
        raise ValueError(message)
    return inventory


def _parameter_ood_source(
    natural_batch: config_service.GenerationConfig,
) -> config_service.GenerationConfig:
    """Return a sentinel-only parameter-OOD source independent of production counts."""
    if natural_batch.sampling_regime != "natural":
        message = "Parameter-OOD sentinels require one natural source batch."
        raise ValueError(message)
    scientific = copy.deepcopy(natural_batch.scientific_values)
    scientific["sampling_regime"] = "parameter_ood"
    scientific["evaluation_regime"] = "parameter_ood"
    scientific["natural_support_state"] = "parameter_ood"
    return replace(
        natural_batch,
        evaluation_regime="parameter_ood",
        sampling_regime="parameter_ood",
        scientific_values=scientific,
    )


def _validate_campaign(
    campaign: config_service.CampaignConfig,
    profile_id: str,
) -> tuple[str, ...]:
    """Require a family campaign with natural support for its full inventory."""
    if campaign.campaign_purpose != "family_generalization" or campaign.profile.id != profile_id:
        message = f"Static sentinel input is not a family-generalization campaign for {profile_id!r}."
        raise ValueError(message)
    inventory = _campaign_inventory(campaign)
    missing_natural = [
        material_family
        for material_family in inventory
        if campaign.find_batch(
            material_family=material_family,
            sampling_regime="natural",
        )
        is None
    ]
    if missing_natural:
        message = f"Static sentinel campaign lacks natural batches for configured materials {missing_natural}."
        raise ValueError(message)
    return inventory


def _sampling_dimension(batch: config_service.GenerationConfig) -> int:
    """Return one resolved batch sampling dimension."""
    dimensions = batch.scientific_values["sampling"]["block_dimensions"]
    return sum(int(value) for value in dimensions.values())


def inspect_sentinel_workload(
    campaign: config_service.CampaignConfig,
) -> dict[str, Any]:
    """Return the bounded config-derived static-sentinel workload plan."""
    inventory = _validate_campaign(campaign, campaign.profile.id)
    parameter_ood: dict[str, Any] = {}
    for material_family in inventory:
        natural = campaign.require_batch(
            material_family=material_family,
            sampling_regime="natural",
        )
        source = campaign.find_batch(
            material_family=material_family,
            sampling_regime="parameter_ood",
        )
        candidates = _ood_candidates(_parameter_ood_source(natural) if source is None else source)
        parameter_ood[material_family] = {
            "case_count": sum(len(names) for names in candidates.values()),
            "eligible_units_by_group": {group: list(names) for group, names in candidates.items()},
        }
    return {
        "natural_materials": list(inventory),
        "natural_cases_per_material": _ID_SENTINEL_CASE_COUNT,
        "natural_case_count": len(inventory) * _ID_SENTINEL_CASE_COUNT,
        "parameter_ood": parameter_ood,
        "parameter_ood_case_count": sum(int(evidence["case_count"]) for evidence in parameter_ood.values()),
        "production_case_count_independent": True,
    }


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
    campaign_contracts: dict[str, Any] = {}
    profile_evidence: dict[str, Any] = {}
    support_failures: list[dict[str, Any]] = []
    for profile_id, campaign in campaigns.items():
        inventory = _validate_campaign(campaign, profile_id)
        natural_sources = {
            material_family: campaign.require_batch(
                material_family=material_family,
                sampling_regime="natural",
            )
            for material_family in inventory
        }
        family_evidence: dict[str, Any] = {}
        for material_family, source in natural_sources.items():
            sentinel = _sentinel_view(
                source,
                case_count=_ID_SENTINEL_CASE_COUNT,
                seed_base=seeding.derive_seed(
                    _PRIMARY_SENTINEL_SEED,
                    profile_id,
                    "material",
                    material_family,
                ),
            )
            report = inventory_service.audit_parameter_registry(
                sentinel.scientific_values["material"]["parameter_registry"],
                profile_id=profile_id,
                block_parameters={
                    block: tuple(str(name) for name in plan["parameters"]) for block, plan in sentinel.scientific_values["sampling"]["blocks"].items()
                },
            )
            change_evidence = _design_change_evidence(sentinel)
            realization = _realization_evidence(sentinel, 1)
            if realization["porosity_guard_failure"] is not None:
                support_failures.append(
                    {
                        "simulation_profile": profile_id,
                        "material_family": material_family,
                        "message": realization["porosity_guard_failure"],
                        "porosity": realization["porosity"],
                    }
                )
            family_evidence[material_family] = {
                "inventory": asdict(report),
                "design_change_evidence": change_evidence,
                "kc_coupling": _kc_coupling_evidence(source),
                "realization": realization,
            }

        inventory_materials = inventory
        kappa_ood = {material_family: _kappa_ood_evidence(natural_sources[material_family]) for material_family in inventory_materials}
        parameter_ood: dict[str, Any] = {}
        for material_family in inventory_materials:
            configured_source = campaign.find_batch(
                material_family=material_family,
                sampling_regime="parameter_ood",
            )
            source = _parameter_ood_source(natural_sources[material_family]) if configured_source is None else configured_source
            evidence = _ood_evidence(source)
            evidence["eligible_unit_count"] = len(evidence["selected_units"])
            evidence["downstream_porosity_attribution"] = _downstream_ood_attribution_evidence(source)
            evidence["configured_production_batch"] = configured_source is not None
            parameter_ood[material_family] = evidence

        dimensions = {_sampling_dimension(source) for source in natural_sources.values()}
        if len(dimensions) != 1:
            message = f"Configured {profile_id!r} materials resolve inconsistent sampling dimensions {sorted(dimensions)}."
            raise ValueError(message)
        sampling_dimension = next(iter(dimensions))
        campaign_contracts[profile_id] = {
            "campaign_id": campaign.campaign_id,
            "campaign_digest": campaign.campaign_digest,
            "material_inventory": list(inventory),
            "material_roles": {role: list(material_families) for role, material_families in campaign.material_roles.items()},
            "evaluation_regimes": list(campaign.evaluation_regimes),
            "production_case_count": campaign.total_case_count,
            "sampling_dimension": sampling_dimension,
        }
        profile_evidence[profile_id] = {
            "family_evidence": family_evidence,
            "field_coupling_by_material": {material_family: _field_coupling_evidence(source) for material_family, source in natural_sources.items()},
            "kappa_mean_ood_by_material": kappa_ood,
            "parameter_ood_by_material": parameter_ood,
        }

    status = "pass" if not support_failures else "blocked_by_scientific_sanity_guard"
    return {
        "schema_kind": "vp2_static_generator_sentinels",
        "schema_version": _STATIC_SENTINEL_SCHEMA_VERSION,
        "status": status,
        "implementation_checks_complete": True,
        "campaign_contracts": campaign_contracts,
        "profile_dimensions": {profile_id: contract["sampling_dimension"] for profile_id, contract in campaign_contracts.items()},
        "profile_evidence": profile_evidence,
        "scientific_support_guard_failures": support_failures,
        "comsol_started": False,
        "production_state_mutated": False,
    }
