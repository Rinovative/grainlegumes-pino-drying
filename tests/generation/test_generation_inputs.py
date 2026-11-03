"""Final configuration, naming, ownership, sampling, and domain contracts.

Responsibilities:
  - Check config-owned sampling, source sentinels, scalar handoff, startup, and retry policy

Design principles:
  - Scientific identity excludes operational provenance and binds active physical controls

This module does NOT:
  - Execute campaigns or alter maintained scientific configuration
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from src import common, generation
from src.generation.cases import generation_cases_fields as fields
from src.generation.cases import generation_cases_sampling as sampling
from src.generation.cases import generation_cases_seeding as seeding
from src.generation.validation import generation_validation_sentinels as sentinels

if TYPE_CHECKING:
    from collections.abc import Callable

if TYPE_CHECKING:
    from pathlib import Path


def _edit_layer(config_path: Path, layer: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Apply one in-place mutation to a test-owned layered campaign YAML file."""
    path = config_path.parent / layer
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_license_wait(updates: dict[str, object]) -> Callable[[dict[str, Any]], None]:
    """Return a mutation applying dotted keys below ``runtime.license_wait``."""

    def mutate(execution: dict[str, Any]) -> None:
        for dotted, value in updates.items():
            target = execution["runtime"]["license_wait"]
            *parents, key = dotted.split(".")
            for parent in parents:
                target = target[parent]
            target[key] = value

    return mutate


@pytest.mark.parametrize("profile", ["steady_flow", "transient_drying"])
def test_small_configuration_controls_membership_and_sampling(generation_config_factory: Any, profile: str) -> None:
    """Resolve a nondefault small source design with a stable sampling order."""
    path, _template = generation_config_factory(
        simulation_profile=profile,
        campaign_purpose="source_generation",
        natural_count=5,
    )
    campaign = generation.cases.config.load_campaign_config(path)
    assert campaign.total_case_count == 5
    assert campaign.profile.id == profile
    assert campaign.selected_targets == {"lentil": 5}
    assert campaign.dataset_packages == ()
    for batch in campaign.batches:
        for design in batch.scientific_values["sampling"]["blocks"].values():
            assert sorted(design["permutation"]) == list(range(5))


def test_kappa_mean_ood_tails_map_to_their_porosity_tails_through_field_generation(generation_config_factory: Any) -> None:
    """Keep each authored permeability tail on its mapped porosity tail in the realized fields."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name("steady_flow", "lentil", "natural"),
    )
    coupling = config.scientific_values["material"]["porosity_coupling"]
    natural = coupling["natural_porosity_support"]
    base_values = dict(sampling.sample_case(config, 1).values)
    lower_guard = float(base_values["eps_min_global"])
    upper_guard = float(base_values["eps_max_global"])
    assert set(coupling["kappa_ood_porosity_supports"]) == {"lower", "upper"}
    for direction, tail in coupling["kappa_ood_porosity_supports"].items():
        for kappa_key, porosity_key in (("kappa_lower", "porosity_lower"), ("kappa_upper", "porosity_upper")):
            mapped = generation.contracts.porosity.solve_reference_porosity(
                float(tail[kappa_key]),
                float(coupling["A_KC_reference"]),
                eps_min_global=lower_guard,
                eps_max_global=upper_guard,
            )
            assert mapped == pytest.approx(float(tail[porosity_key]), rel=0.0, abs=2e-15)
        values = {**base_values, "kappa_mean": math.sqrt(float(tail["kappa_lower"]) * float(tail["kappa_upper"]))}
        diagnostics = fields.generate_spatial_fields(
            config.profile.id,
            config.scientific_values["grid"],
            values,
            seeds={name: seeding.derive_seed(7, "kappa_ood", direction, name) for name in ("bed", "pressure_bc", "packing_scatter")},
            family_bounds=None,
            porosity_coupling=coupling,
            active_ood_unit="kappa_mean",
        ).metadata["porosity"]
        assert diagnostics["packing_scatter_support_kind"] == f"kappa_mean_ood_{direction}"
        if direction == "lower":
            assert max(diagnostics["eps_reference"], diagnostics["eps_bed_mean"]) < float(natural["lower"])
        else:
            assert min(diagnostics["eps_reference"], diagnostics["eps_bed_mean"]) > float(natural["upper"])
        assert diagnostics["eps_bed_min"] >= lower_guard
        assert diagnostics["eps_bed_max"] <= upper_guard


def test_source_sentinels_reject_wrong_purpose_profile_and_missing_natural_batch(
    generation_config_factory: Any,
) -> None:
    """Keep the current Source inventory and purpose admission fail-closed."""
    steady_path, _steady_template = generation_config_factory(simulation_profile="steady_flow", campaign_purpose="source_generation")
    steady = generation.cases.config.load_campaign_config(steady_path)

    with pytest.raises(ValueError, match="lacks natural batches"):
        sentinels.inspect_sentinel_workload(replace(steady, batches=()))
    with pytest.raises(ValueError, match="not a Source campaign"):
        sentinels.inspect_sentinel_workload(replace(steady, campaign_purpose="other"))


def test_semantic_seed_derivation_is_version_bound() -> None:
    """Protect persisted random-substream identity across internal ownership changes."""
    assert (
        seeding.derive_seed(
            12345,
            "generation_batch",
            "profile_a",
            "material_a",
            "regime_a",
            "evaluation_a",
        )
        == 2107745129
    )


def test_incomplete_mapping_declaration_is_discovery_only(
    generation_config_factory: Any,
) -> None:
    """Permit structural discovery only when executable mappings are not required."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    _edit_layer(config_path, "profile.yaml", lambda profile: profile["exports"][0]["columns"].update(x=None))

    campaign = generation.cases.config.load_campaign_config(config_path, require_executable=False)
    assert "x" not in campaign.batches[0].scientific_values["output_contract"]["exports"][0]["columns"]
    with pytest.raises(
        generation.cases.config.GenerationConfigError,
        match="incomplete; an explicit executable mapping declaration is required",
    ):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize("option", ["--exclusive", "--reservation=reserved", "--nodelist=node-a", "--signal=B:TERM@10"])
def test_execution_rejects_nonordinary_scheduler_options(
    generation_config_factory: Any,
    option: str,
) -> None:
    """Prevent execution YAML from reintroducing exclusivity or reservations."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    _edit_layer(config_path, "execution.yaml", lambda execution: execution["cluster"].update(scheduler_options=[option]))

    with pytest.raises(generation.cases.config.GenerationConfigError, match="allocation directives"):
        generation.cases.config.load_campaign_config(config_path)


def test_parameter_ood_allocation_covers_eligible_units_evenly() -> None:
    """Project configured OOD units and allocate compact cases evenly."""
    family_contract = {
        "parameter_registry": {
            "thermal_interval": {
                "kind": "interval",
                "ood_group": "thermal",
                "ood": [{"lower": 1.0, "upper": 2.0}],
            },
            "flow_mode": {
                "kind": "categorical",
                "ood_group": "flow",
                "ood_choices": ["high"],
            },
            "inactive_interval": {
                "kind": "interval",
                "ood_group": "inactive",
                "ood": [{"lower": 3.0, "upper": 4.0}],
            },
        },
        "coupled_ood_records": {
            "sorption_record": {
                "ood_group": "thermal",
                "records": [{"A": 1.0}],
            }
        },
    }

    eligible = sampling.eligible_ood_units(
        family_contract,
        groups=("thermal", "flow"),
    )
    allocation = sampling.allocate_ood_units(eligible, case_count=7)
    counts = {unit["unit_id"]: sum(assigned["unit_id"] == unit["unit_id"] for assigned in allocation) for unit in eligible}

    assert [unit["unit_id"] for unit in eligible] == [
        "thermal_interval",
        "flow_mode",
        "sorption_record",
    ]
    assert min(counts.values()) >= 1
    assert max(counts.values()) - min(counts.values()) <= 1
    assert [assigned["unit_id"] for assigned in allocation[:3]] == [unit["unit_id"] for unit in eligible]


def test_generation_and_hdf5_versions_are_explicit_integers(
    generation_config_factory: Any,
) -> None:
    """Reject named versions and preserve the active persisted schema identities."""
    config_path, _template = generation_config_factory()
    batch = generation.cases.config.load_campaign_config(config_path).batches[0]
    versions = (
        batch.scientific_values["generator_version"],
        batch.scientific_values["storage"]["schema_version"],
        batch.scientific_values["storage"]["converter_version"],
        generation.cases.schedule.SCHEDULE_GENERATOR_VERSION,
        generation.cases.schedule.COMSOL_BOUNDARY_HANDOFF_VERSION,
    )
    assert versions == (1, 1, 1, 1, 1)
    assert all(type(version) is int for version in versions)

    common_path = config_path.parent / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    common["generator_version"] = "named-version"
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    with pytest.raises(generation.cases.config.GenerationConfigError, match="generator_version must be an integer"):
        generation.cases.config.load_campaign_config(config_path)

    common["generator_version"] = 1
    common["storage"]["converter_version"] = "named-version"
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    with pytest.raises(generation.cases.config.GenerationConfigError, match="converter_version must be an integer"):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("rtol", 0.0, "rtol must be positive"),
        ("rtol", -1.0e-4, "rtol must be positive"),
        ("atol", -1.0e-10, "atol must be non-negative"),
        ("rtol", float("nan"), "must be a finite real value"),
        ("atol", float("inf"), "must be a finite real value"),
        ("rtol", "1e-4", "must be a finite real value"),
        ("atol", True, "must be a finite real value"),
        ("rtol", 1.0e-3, "no greater than"),
        ("atol", 1.0e-6, "no greater than"),
    ],
)
def test_transient_initial_state_tolerance_validation(
    generation_config_factory: Any,
    key: str,
    value: object,
    message: str,
) -> None:
    """Reject malformed or unsafe semantic initial-state tolerances."""
    config_path, _template = generation_config_factory()
    common_path = config_path.parent / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    common["validation"]["transient_initial_state"][key] = value
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")

    with pytest.raises(generation.cases.config.GenerationConfigError, match=message):
        generation.cases.config.load_campaign_config(config_path)


def test_startup_ramp_identity_ignores_inactive_policy_and_binds_active_behavior(
    generation_config_factory: Any,
) -> None:
    """Bind enabled ramp behavior while excluding disabled policy values from identity."""
    transient_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        startup_ramp_enabled=False,
        startup_ramp_duration_h=0.25,
    )
    disabled = generation.cases.config.load_campaign_config(transient_path).batches[0]
    operations_path = transient_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))

    operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.375
    operations["boundary_schedule"]["startup_ramp"]["initial_equilibrium_rh_dry_margin"] = 0.075
    operations["boundary_schedule"]["startup_ramp"]["max_relative_humidity"] = 0.95
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")
    inactive_policy_changed = generation.cases.config.load_campaign_config(transient_path).batches[0]
    assert inactive_policy_changed.scientific_values["boundary_schedule"]["startup_ramp"] == {
        "enabled": False,
        "duration_h": 0.375,
        "initial_equilibrium_rh_dry_margin": 0.075,
        "max_relative_humidity": 0.95,
    }
    assert inactive_policy_changed.scientific_config_digest == disabled.scientific_config_digest
    assert inactive_policy_changed.case_input_config_digest == disabled.case_input_config_digest
    assert inactive_policy_changed.batch_id == disabled.batch_id

    operations["boundary_schedule"]["startup_ramp"]["enabled"] = True
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")
    enabled = generation.cases.config.load_campaign_config(transient_path).batches[0]
    assert enabled.scientific_config_digest != disabled.scientific_config_digest
    assert enabled.case_input_config_digest != disabled.case_input_config_digest
    assert enabled.batch_id != disabled.batch_id

    operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.125
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")
    active_duration_changed = generation.cases.config.load_campaign_config(transient_path).batches[0]
    assert active_duration_changed.scientific_config_digest != enabled.scientific_config_digest
    assert active_duration_changed.case_input_config_digest != enabled.case_input_config_digest
    assert active_duration_changed.batch_id != enabled.batch_id

    operations["boundary_schedule"]["startup_ramp"]["initial_equilibrium_rh_dry_margin"] = 0.065
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")
    active_margin_changed = generation.cases.config.load_campaign_config(transient_path).batches[0]
    assert active_margin_changed.scientific_config_digest != active_duration_changed.scientific_config_digest
    assert active_margin_changed.case_input_config_digest != active_duration_changed.case_input_config_digest
    assert active_margin_changed.batch_id != active_duration_changed.batch_id

    operations["boundary_schedule"]["startup_ramp"]["max_relative_humidity"] = 0.92
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")
    active_maximum_changed = generation.cases.config.load_campaign_config(transient_path).batches[0]
    assert active_maximum_changed.scientific_config_digest != active_margin_changed.scientific_config_digest
    assert active_maximum_changed.case_input_config_digest != active_margin_changed.case_input_config_digest
    assert active_maximum_changed.batch_id != active_margin_changed.batch_id

    steady_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        startup_ramp_enabled=True,
        startup_ramp_duration_h=0.25,
    )
    original_steady = generation.cases.config.load_campaign_config(steady_path).batches[0]
    steady_operations_path = steady_path.parent / "operations.yaml"
    steady_operations = yaml.safe_load(steady_operations_path.read_text(encoding="utf-8"))
    steady_operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.375
    steady_operations_path.write_text(yaml.safe_dump(steady_operations, sort_keys=False), encoding="utf-8")
    changed_steady = generation.cases.config.load_campaign_config(steady_path).batches[0]

    assert "boundary_schedule" not in changed_steady.scientific_values
    assert changed_steady.scientific_config_digest == original_steady.scientific_config_digest
    assert changed_steady.case_input_config_digest == original_steady.case_input_config_digest
    assert changed_steady.batch_id == original_steady.batch_id


def test_startup_margin_does_not_change_sampling_or_case_membership(
    generation_config_factory: Any,
) -> None:
    """Keep deterministic preprocessing outside the DoE and seed plan."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
    )
    original = generation.cases.config.load_campaign_config(config_path).batches[0]
    original_sample = generation.cases.sampling.sample_case(original, 1)
    operations_path = config_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    operations["boundary_schedule"]["startup_ramp"]["initial_equilibrium_rh_dry_margin"] = 0.06
    operations["boundary_schedule"]["startup_ramp"]["max_relative_humidity"] = 0.93
    operations_path.write_text(
        yaml.safe_dump(operations, sort_keys=False),
        encoding="utf-8",
    )

    changed = generation.cases.config.load_campaign_config(config_path).batches[0]
    changed_sample = generation.cases.sampling.sample_case(changed, 1)

    assert changed.case_indices == original.case_indices
    assert changed.seed_base == original.seed_base
    assert changed.case_seed(1) == original.case_seed(1)
    assert changed_sample.values == original_sample.values
    assert changed_sample.units == original_sample.units
    assert changed_sample.block_provenance == original_sample.block_provenance
    assert changed_sample.coupled_selections == original_sample.coupled_selections
    assert "initial_equilibrium_rh_dry_margin" not in changed_sample.values
    assert "max_relative_humidity" not in changed_sample.values
    assert changed.scientific_config_digest != original.scientific_config_digest


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("enabled", "true", "enabled must be boolean"),
        ("duration_h", 0.0, "strictly positive and shorter"),
        ("duration_h", 0.75, "strictly positive and shorter"),
        ("initial_equilibrium_rh_dry_margin", 0.0, "strictly inside"),
        ("initial_equilibrium_rh_dry_margin", 1.0, "strictly inside"),
        ("max_relative_humidity", 0.0, "inside"),
        ("max_relative_humidity", 1.01, "inside"),
    ],
)
def test_startup_ramp_configuration_is_strict(
    generation_config_factory: Any,
    key: str,
    value: object,
    message: str,
) -> None:
    """Reject malformed ramp values against an explicit regular interval."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        startup_ramp_enabled=True,
        startup_ramp_duration_h=0.25,
    )
    common_path = config_path.parent / "common.yaml"
    common_config = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    common_config["time"]["interval"] = 0.75
    common_path.write_text(yaml.safe_dump(common_config, sort_keys=False), encoding="utf-8")
    operations_path = config_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    operations["boundary_schedule"]["startup_ramp"][key] = value
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")

    with pytest.raises(generation.cases.config.GenerationConfigError, match=message):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("runtime", "maximum_failed_cases", -1, "must be an integer >= 0"),
        ("runtime", "maximum_failed_cases", 1.5, "must be an integer >= 0"),
        ("runtime", "maximum_failed_cases", True, "must be an integer >= 0"),
        ("cluster", "node_jobs", 0, "must be an integer >= 1"),
        ("cluster", "node_jobs", 1.5, "must be an integer >= 1"),
        ("cluster", "node_jobs", 5, "must be between 1 and 4"),
    ],
)
def test_execution_count_limits_reject_invalid_domains(
    generation_config_factory: Any,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    """Reject invalid failure-threshold and node-job counts."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "execution.yaml", lambda execution: execution[section].update({key: value}))

    with pytest.raises(generation.cases.config.GenerationConfigError, match=message):
        generation.cases.config.load_campaign_config(config_path)


def test_execution_accepts_the_full_per_user_node_job_cap(generation_config_factory: Any) -> None:
    """The cap of four node jobs is a valid execution choice, not only the default two."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "execution.yaml", lambda execution: execution["cluster"].update({"node_jobs": 4}))

    config = generation.cases.config.load_campaign_config(config_path)

    assert config.execution_values["cluster"]["node_jobs"] == 4


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("cluster", "node_job_wall_time", "00:10:00", "must exceed execution.runtime.timeout_seconds plus a 600 s publication margin"),
        ("cluster", "node_job_wall_time", "24:00:01", "must not exceed 24:00:00"),
        ("cluster", "node_job_wall_time", "6h", "must use HH:MM:SS"),
        ("cluster", "memory_per_case", "20", "memory size <int>M, <int>G or <int>T"),
        ("cluster", "memory_per_case", "1.5G", "memory size <int>M, <int>G or <int>T"),
        ("cluster", "node_memory_overhead", "0G", "memory size <int>M, <int>G or <int>T"),
        ("cluster", "node_memory_limit", "100 GB", "memory size <int>M, <int>G or <int>T"),
        ("cluster", "memory_per_case", "30G", "node_memory_limit is 100G"),
        ("cluster", "cores_per_case", 25, "must not exceed site.cores_per_node"),
    ],
)
def test_node_job_contract_rejects_what_does_not_fit(
    generation_config_factory: Any,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    """Fail closed on a node job that cannot hold one case, an ill-formed size, and a node memory above its limit."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "execution.yaml", lambda execution: execution[section].update({key: value}))

    with pytest.raises(generation.cases.config.GenerationConfigError, match=message):
        generation.cases.config.load_campaign_config(config_path)


def test_execution_schema_version_one_is_not_read(generation_config_factory: Any) -> None:
    """A version-1 file is refused with the supported version named, before any field is interpreted."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "execution.yaml", lambda execution: execution.update({"schema_version": 1}))

    with pytest.raises(generation.cases.config.GenerationConfigError, match=r"schema_version 1 is not supported; schema_version 2 .* is required"):
        generation.cases.config.load_campaign_config(config_path)


def test_node_job_resources_derive_concurrency_and_memory_from_the_case_step(generation_config_factory: Any) -> None:
    """Derive the concurrent case steps and the node-job memory, rounded up to whole GiB, from test-owned inputs."""
    config_path, _template = generation_config_factory(cores_per_case=5, cores_per_node=24)
    _edit_layer(
        config_path,
        "execution.yaml",
        lambda execution: execution["cluster"].update({"memory_per_case": "1500M", "node_memory_overhead": "600M", "node_memory_limit": "1T"}),
    )

    cluster = generation.cases.config.load_campaign_config(config_path).execution_values["cluster"]

    # 24 // 5 = 4 case steps of 1500 MiB plus 600 MiB = 6600 MiB, which Slurm needs as 7 GiB.
    assert cluster["concurrent_cases_per_node"] == 4
    assert cluster["node_job_memory"] == "7G"
    assert cluster["memory_per_case"] == "1500M"


@pytest.mark.parametrize(
    "option",
    [
        "--mem=32G",
        "--mem-per-cpu=4G",
        "--dependency=afterok:1",
        "--comment=run",
        "--exclude=hpc111",
        "--nodelist=hpc111",
        "--exclusive",
        "--gres=gpu:1",
        "--gpus-per-node=1",
        "--array=1-2",
        "--nodes=2",
        "--ntasks=2",
        "--partition=long",
        "--qos=high",
        "--reservation=maintenance",
        "--mem 32G",
    ],
)
def test_scheduler_options_cannot_override_pipeline_owned_allocation(generation_config_factory: Any, option: str) -> None:
    """Resources, dependencies, node selection, and GPUs are pipeline-owned, so configuration cannot carry them."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "execution.yaml", lambda execution: execution["cluster"].update({"scheduler_options": [option]}))

    with pytest.raises(generation.cases.config.GenerationConfigError, match="pipeline-owned allocation directives"):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize(
    ("stall_stop", "message"),
    [
        ({"check_after_seconds": 0, "minimum_simulated_seconds": 3600}, "needs positive"),
        ({"check_after_seconds": 900, "minimum_simulated_seconds": -1}, "needs positive"),
        ({"check_after_seconds": 900}, "missing"),
    ],
)
def test_execution_rejects_an_invalid_stall_rule(generation_config_factory: Any, stall_stop: dict[str, float], message: str) -> None:
    """The optional stall rule needs both positive thresholds and nothing else."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "execution.yaml", lambda execution: execution["runtime"].update({"stall_stop": stall_stop}))

    with pytest.raises(generation.cases.config.GenerationConfigError, match=message):
        generation.cases.config.load_campaign_config(config_path)


def test_execution_without_a_stall_rule_resolves_without_one(generation_config_factory: Any) -> None:
    """An execution file without stall_stop keeps only the timeout."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "execution.yaml", lambda execution: execution["runtime"].pop("stall_stop", None))

    campaign = generation.cases.config.load_campaign_config(config_path)

    assert "stall_stop" not in campaign.batches[0].execution_values["runtime"]


def test_valid_config_edits_are_resolved_without_source_synchronization(
    generation_config_factory: Any,
) -> None:
    """Resolve test-owned science and execution edits from their YAML owners."""
    campaign_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
        campaign_purpose="source_generation",
    )
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    common_path = campaign_path.parent / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    operations_path = campaign_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    execution_path = campaign_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))

    campaign["sampling"]["seed_base"] = 12345
    common["grid"].update({"nx": 17, "ny": 11, "Lx": 1.6, "Ly": 1.0})
    common["time"].update({"start": 0.0, "stop": 84.0, "interval": 0.5})
    operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.25
    common["scientific_fixed_values"]["T_flow_ref"]["value"] = 301.15
    common["scientific_fixed_values"]["p_ref"]["value"] = 100000.0
    common["storage"]["compression_level"] = 6
    execution["runtime"].update(
        {
            "timeout_seconds": 4200,
            "graceful_stop_reserve_seconds": 300,
            "maximum_failed_cases": 3,
        }
    )
    execution["cluster"]["node_job_wall_time"] = "02:00:00"
    execution["submission"].update(
        {
            "max_admission_cases": 2,
            "poll_interval_seconds": 7,
        }
    )
    execution["cluster"]["cores_per_case"] = 8
    execution["cluster"]["node_jobs"] = 1
    execution["site"]["cpu_host"] = "synthetic-host"

    renamed_campaign_path = campaign_path.with_name("lentil-d3__reference-range__lhs-s12345.yaml")
    renamed_campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False),
        encoding="utf-8",
    )
    common_path.write_text(
        yaml.safe_dump(common, sort_keys=False),
        encoding="utf-8",
    )
    operations_path.write_text(
        yaml.safe_dump(operations, sort_keys=False),
        encoding="utf-8",
    )
    execution_path.write_text(
        yaml.safe_dump(execution, sort_keys=False),
        encoding="utf-8",
    )

    resolved = generation.cases.config.load_campaign_config(renamed_campaign_path)
    scientific = resolved.batches[0].scientific_values

    assert resolved.total_case_count == 3
    assert resolved.selected_targets == {"lentil": 3}
    assert scientific["campaign_seed"] == 12345
    assert scientific["grid"]["dx"] == pytest.approx(0.1)
    assert scientific["grid"]["dy"] == pytest.approx(0.1)
    assert scientific["time"]["regular_times"] == [index * 0.5 for index in range(169)]
    assert scientific["boundary_schedule"]["startup_ramp"] == {
        "enabled": True,
        "duration_h": 0.25,
        "initial_equilibrium_rh_dry_margin": 0.05,
        "max_relative_humidity": 0.90,
    }
    assert scientific["scientific_fixed_values"]["T_flow_ref"] == 301.15
    assert scientific["scientific_fixed_values"]["p_ref"] == 100000.0
    assert scientific["storage"]["compression_level"] == 6
    assert resolved.execution_values["runtime"]["timeout_seconds"] == 4200.0
    assert resolved.execution_values["runtime"]["maximum_failed_cases"] == 3
    assert resolved.execution_values["runtime"]["graceful_stop_reserve_seconds"] == 300.0
    cluster = resolved.execution_values["cluster"]
    assert cluster["node_job_wall_time"] == "02:00:00"
    assert (cluster["cores_per_case"], cluster["node_jobs"]) == (8, 1)
    # The fixture site has 24 cores: three 8-core case steps of 1G plus the 1G overhead make a 4G node job.
    assert (cluster["concurrent_cases_per_node"], cluster["node_job_memory"]) == (3, "4G")
    assert resolved.execution_values["submission"] == {
        "max_admission_cases": 2,
        "poll_interval_seconds": 7,
    }
    assert resolved.execution_values["site"]["cpu_host"] == "synthetic-host"


def test_sampling_coordinate_order_is_config_owned_and_identity_bound(
    generation_config_factory: Any,
) -> None:
    """Bind explicit registry coordinate order through scientific identity."""
    campaign_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
        campaign_purpose="source_generation",
    )
    registry_path = campaign_path.parent / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    original = generation.cases.config.load_campaign_config(campaign_path)
    original_batch = original.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    original_parameters = original_batch.scientific_values["sampling"]["blocks"]["airflow"]["parameters"]

    left = "bed.structure.cross_scale_corr"
    right = "bed.structure.fine_ani_x"
    left_order = registry["parameters"][left]["sampling_order"]
    right_order = registry["parameters"][right]["sampling_order"]
    registry["parameters"][left]["sampling_order"] = right_order
    registry["parameters"][right]["sampling_order"] = left_order
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )

    reordered = generation.cases.config.load_campaign_config(campaign_path)
    reordered_batch = reordered.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    reordered_parameters = reordered_batch.scientific_values["sampling"]["blocks"]["airflow"]["parameters"]

    assert original_parameters.index(left) == reordered_parameters.index(right)
    assert original_parameters.index(right) == reordered_parameters.index(left)
    assert reordered_batch.scientific_config_digest != (original_batch.scientific_config_digest)
    assert reordered_batch.batch_id != original_batch.batch_id
    assert reordered.campaign_digest != original.campaign_digest


def test_invalid_config_combinations_report_authoritative_owner(
    generation_config_factory: Any,
) -> None:
    """Report the authored source owner for an invalid design count."""

    def expect_failure(
        mutate: Callable[[dict[str, Any]], None],
        expected_key: str | Callable[[dict[str, Any]], str],
    ) -> None:
        campaign_path, _template = generation_config_factory(
            simulation_profile="transient_drying",
            natural_count=3,
            campaign_purpose="source_generation",
        )
        campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
        mutate(campaign)
        campaign_path.write_text(
            yaml.safe_dump(campaign, sort_keys=False),
            encoding="utf-8",
        )
        with pytest.raises(generation.cases.config.GenerationConfigError) as caught:
            generation.cases.config.load_campaign_config(campaign_path)
        details = generation.cases.config.validation_error_details(
            campaign_path,
            caught.value,
        )
        expected_owner = campaign_path.relative_to(common.paths.get_project_root()).as_posix()
        assert details["file"] == expected_owner
        assert details["owner_to_edit"] == expected_owner
        resolved_key = expected_key(campaign) if callable(expected_key) else expected_key
        assert details["key"] == resolved_key
        assert details["actual_value"] != "<missing>"
        assert details["expected_type_or_rule"]

    expect_failure(
        lambda campaign: campaign["sampling"]["counts"]["natural"].__setitem__(
            "lentil",
            0,
        ),
        "sampling.counts.natural.lentil",
    )


def test_schedule_csv_is_the_identity_bound_primitive_no_ramp_handoff(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Persist exact primitive regular support for a test-owned disabled ramp."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        startup_ramp_enabled=False,
        startup_ramp_duration_h=0.25,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "transient_drying",
            "lentil",
            "natural",
        ),
    )
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "no_ramp_case")
    schedule_path = bundle.directory / "schedule.csv"
    header = schedule_path.read_text(encoding="utf-8").splitlines()[0]
    schedule = np.loadtxt(schedule_path, delimiter=";", skiprows=1)
    regular_times = np.asarray(
        config.scientific_values["time"]["regular_times"],
        dtype=np.float64,
    )

    assert header == "t;T_in_bc;omega_in_bc"
    assert schedule.shape == (regular_times.size, 3)
    np.testing.assert_array_equal(schedule[:, 0], regular_times)
    assert regular_times[0] == 0.0
    handoff = bundle.case_payload["schedule_diagnostics"]["boundary_handoff"]
    assert handoff["startup_ramp"]["enabled"] is False
    assert handoff["rejoin_row"] is None
    np.testing.assert_array_equal(schedule[0], handoff["canonical_start_row"])
    schedule_identity = bundle.case_payload["input_files"]["schedule.csv"]
    assert schedule_identity == {
        "sha256": common.serialization.file_sha256(schedule_path),
        "size_bytes": schedule_path.stat().st_size,
    }
    assert bundle.case_payload["input_contract"]["schedule"]["columns"] == [
        "t",
        "T_in_bc",
        "omega_in_bc",
    ]
    assert bundle.case_payload["case_input_id"] == generation.cases.case.compute_case_input_id(bundle.case_payload)
    admitted = generation.cases.admission.admit_input_case(
        bundle.directory,
        source_id="primitive-writer-reader",
        label="primitive writer-reader agreement",
    )
    assert admitted.schedule is not None
    np.testing.assert_array_equal(admitted.schedule, schedule)


def test_admission_rejects_identity_valid_stale_four_column_schedule(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject a stale RH column even after its file and case identities are refreshed."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        startup_ramp_enabled=False,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "transient_drying",
            "lentil",
            "natural",
        ),
    )
    bundle = generation.cases.case.generate_case_input_bundle(
        config,
        1,
        tmp_path / "stale_four_column_case",
    )
    schedule_path = bundle.directory / "schedule.csv"
    lines = schedule_path.read_text(encoding="utf-8").splitlines()
    stale_lines = (
        f"{lines[0]};phi_in_bc",
        *(f"{line};0.5" for line in lines[1:]),
    )
    schedule_path.write_text("\n".join(stale_lines) + "\n", encoding="utf-8")

    payload = copy.deepcopy(bundle.case_payload)
    payload["input_files"]["schedule.csv"] = {
        "sha256": common.serialization.file_sha256(schedule_path),
        "size_bytes": schedule_path.stat().st_size,
    }
    payload["case_input_id"] = generation.cases.case.compute_case_input_id(payload)
    payload["simulation_case_id"] = generation.cases.case.compute_simulation_case_id(payload)
    common.serialization.atomic_write_json(bundle.directory / "case.json", payload)

    with pytest.raises(ValueError, match="Schedule adapter header is invalid"):
        generation.cases.admission.admit_input_case(
            bundle.directory,
            source_id="stale-four-column",
            label="stale four-column rejection",
        )


def test_practical_safe_storage_rounding_uses_whole_percent_wb() -> None:
    """Round exact safe-storage evidence deterministically for operation."""
    rounding = generation.contracts.materials.practical_target_moisture_wb

    assert rounding(0.11894) == pytest.approx(0.12)
    assert rounding(0.14315) == pytest.approx(0.14)


def test_target_reaches_scalar_handoff_and_scientific_identity(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Bind one test-owned target to case inputs and scientific identity."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
    )
    batch_name = generation.cases.config.build_batch_name(
        "transient_drying",
        "lentil",
        "natural",
    )
    original = generation.cases.config.load_generation_config(
        config_path,
        only_batch=batch_name,
    )
    material_path = config_path.parents[3] / "physical_model/materials/lentil.yaml"
    assert tmp_path in material_path.parents
    authored = yaml.safe_load(material_path.read_text(encoding="utf-8"))
    assert set(authored["target_moisture"]) == {
        "target_moisture_wb",
        "provenance",
    }
    authored["target_moisture"]["target_moisture_wb"] = 0.13
    material_path.write_text(
        yaml.safe_dump(authored, sort_keys=False),
        encoding="utf-8",
    )

    changed = generation.cases.config.load_generation_config(
        config_path,
        only_batch=batch_name,
    )
    bundle = generation.cases.case.generate_case_input_bundle(
        changed,
        1,
        tmp_path / "target-case",
    )
    assert changed.scientific_config_digest != original.scientific_config_digest
    assert changed.case_input_config_digest != original.case_input_config_digest
    assert bundle.scalar_handoff is not None
    scalar_values = dict(
        zip(
            bundle.scalar_handoff.field_names,
            bundle.scalar_handoff.values,
            strict=True,
        )
    )
    assert scalar_values["X_target_wb"] == pytest.approx(0.13)

    authored["target_moisture"]["market_acceptance_moisture_wb"] = 0.17
    material_path.write_text(
        yaml.safe_dump(authored, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"target_moisture.*unknown"):
        generation.cases.config.load_generation_config(
            config_path,
            only_batch=batch_name,
        )


def test_scalar_handoff_rejects_an_unknown_name(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect the exact scalar schema after a modified file receives a fresh hash."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        startup_ramp_enabled=False,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "transient_drying",
            "lentil",
            "natural",
        ),
    )
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "case")
    scalar_path = bundle.directory / "scalars.csv"
    content = scalar_path.read_text(encoding="utf-8")
    assert bundle.scalar_handoff is not None
    assert bundle.scalar_handoff.field_names == generation.contracts.profiles.TRANSIENT_SCALAR_INPUT_FIELDS
    assert len(bundle.scalar_handoff.entries) == 12
    non_handoff_fields = {"T_init", "T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}
    assert non_handoff_fields.isdisjoint(bundle.scalar_handoff.field_names)
    assert all(f"{name};" not in content for name in non_handoff_fields)
    assert bundle.case_payload["sampled_values"]["T_init"] == bundle.case_payload["sampled_values"]["T_amb"]
    assert {"T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}.isdisjoint(bundle.case_payload["sampled_values"])
    registry_entry = config.scientific_values["material"]["parameter_registry"]["T_init"]
    assert registry_entry["kind"] == "derived"
    assert registry_entry["derivation"] == "copy"
    assert registry_entry["sources"] == ["T_amb"]
    field_prefix = "T_amb;"
    assert content.count(field_prefix) == 1
    scalar_path.write_text(content.replace(field_prefix, "unexpected_flow_temperature;"), encoding="utf-8")
    payload = copy.deepcopy(bundle.case_payload)
    payload["input_files"]["scalars.csv"] = {
        "sha256": common.serialization.file_sha256(scalar_path),
        "size_bytes": scalar_path.stat().st_size,
    }
    with pytest.raises(ValueError, match="missing, duplicate, unknown"):
        generation.contracts.scalar_handoff.admit_case_scalar_handoff(
            payload,
            bundle.directory,
        )


def test_each_sampled_morphology_control_changes_its_owned_field(generation_config_factory: Any) -> None:
    """Protect material consumption of all 17 independent morphology controls."""
    config_path, _template = generation_config_factory()
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    sample = sampling.sample_case(batch, 1)
    values = sample.values
    registry = batch.scientific_values["material"]["parameter_registry"]
    grid = batch.scientific_values["grid"]
    seeds = {"bed": 101, "pressure_bc": 202, "packing_scatter": 303, "initial_moisture": 404}
    family_contract = batch.scientific_values["material"]
    family_bounds = family_contract["initial_moisture_bounds"]
    porosity_coupling = family_contract["porosity_coupling"]
    baseline = fields.generate_spatial_fields(
        "transient_drying",
        grid,
        values,
        seeds=seeds,
        family_bounds=family_bounds,
        porosity_coupling=porosity_coupling,
        active_ood_unit=None,
    )
    bed_names = (
        "bed.structure.coarse_len_rel",
        "bed.structure.fine_len_rel",
        "bed.structure.coarse_weight",
        "bed.structure.cross_scale_corr",
        "bed.structure.fine_ani_x",
        "bed.structure.fine_ani_y",
        "bed.perturbations.amplitude",
        "bed.perturbations.granularity",
        "bed.perturbations.sign_bias",
    )
    moisture_names = (
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "initial_moisture.structure.coarse_len_rel",
        "initial_moisture.structure.fine_len_rel",
        "initial_moisture.structure.coarse_weight",
        "initial_moisture.structure.cross_scale_corr",
        "initial_moisture.structure.fine_ani_x",
        "initial_moisture.structure.fine_ani_y",
    )
    observed: dict[str, float] = {}
    for name in (*bed_names, *moisture_names):
        variant = copy.deepcopy(values)
        if name == "initial_moisture.mean_db":
            amplitude = float(values["initial_moisture.amplitude_db"])
            candidates = (float(family_bounds["lower"]) + amplitude, float(family_bounds["upper"]) - amplitude)
        elif name == "initial_moisture.amplitude_db":
            mean = float(values["initial_moisture.mean_db"])
            feasible_upper = min(
                float(registry[name]["upper"]),
                mean - float(family_bounds["lower"]),
                float(family_bounds["upper"]) - mean,
            )
            candidates = (float(registry[name]["lower"]), feasible_upper)
        else:
            candidates = (float(registry[name]["lower"]), float(registry[name]["upper"]))
        variant[name] = max(candidates, key=lambda value: abs(value - float(values[name])))
        if name == "bed.structure.coarse_weight":
            variant["bed.structure.fine_weight"] = 1.0 - variant[name]
        elif name == "initial_moisture.structure.coarse_weight":
            variant["initial_moisture.structure.fine_weight"] = 1.0 - variant[name]
        generated = fields.generate_spatial_fields(
            "transient_drying",
            grid,
            variant,
            seeds=seeds,
            family_bounds=family_bounds,
            porosity_coupling=porosity_coupling,
            active_ood_unit=None,
        )
        affected_fields = ("Kxx", "Kxy", "Kyy", "eps_bed") if name in bed_names else ("X_0_db_field",)
        difference = max(float(np.max(np.abs(generated.columns[field] - baseline.columns[field]))) for field in affected_fields)
        scale = max(float(np.max(np.abs(baseline.columns[field]))) for field in affected_fields)
        observed[name] = difference / max(scale, np.finfo(np.float64).tiny)
    assert set(observed) == set(bed_names) | set(moisture_names)
    assert all(effect > 1e-10 for effect in observed.values()), observed


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("rtol", 0.0, "rtol must be positive"),
        ("rtol", -1.0e-5, "rtol must be positive"),
        ("atol", -1.0e-9, "atol must be non-negative"),
        ("rtol", float("nan"), "finite real value"),
        ("atol", float("inf"), "finite real value"),
        ("rtol", True, "finite real value"),
        ("rtol", 1.0e-4, "no greater than"),
        ("atol", 1.0e-8, "no greater than"),
    ],
)
def test_transient_bulk_moisture_tolerance_validation(
    generation_config_factory: Any,
    key: str,
    value: object,
    message: str,
) -> None:
    """Reject malformed or scientifically over-broad bulk tolerances."""
    config_path, _template = generation_config_factory()
    _edit_layer(config_path, "common.yaml", lambda common: common["validation"]["transient_bulk_moisture"].update({key: value}))

    with pytest.raises(
        generation.cases.config.GenerationConfigError,
        match=message,
    ):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize(
    ("updates", "exception", "message"),
    [
        ({"enabled": 1}, TypeError, "must be boolean"),
        ({"initial_delay_seconds": True}, generation.cases.config.GenerationConfigError, "finite real value"),
        ({"initial_delay_seconds": 0}, generation.cases.config.GenerationConfigError, "must be positive"),
        ({"maximum_delay_seconds": float("inf")}, generation.cases.config.GenerationConfigError, "finite real value"),
        ({"maximum_wait_seconds": -1}, generation.cases.config.GenerationConfigError, "must be positive"),
        ({"maximum_wait_seconds": None}, generation.cases.config.GenerationConfigError, "finite real value"),
        (
            {"initial_delay_seconds": 60.0, "maximum_delay_seconds": 59},
            generation.cases.config.GenerationConfigError,
            "must be at least initial_delay_seconds",
        ),
        (
            {"initial_delay_seconds": 60.0, "maximum_wait_seconds": 59},
            generation.cases.config.GenerationConfigError,
            "must be at least initial_delay_seconds",
        ),
        ({"in_allocation_retry.enabled": 1}, TypeError, "must be boolean"),
        ({"in_allocation_retry.maximum_window_seconds": 0}, generation.cases.config.GenerationConfigError, "must be positive"),
        (
            {"in_allocation_retry.pause_after_capacity_failure_seconds": float("inf")},
            generation.cases.config.GenerationConfigError,
            "finite real value",
        ),
    ],
)
def test_license_retry_configuration_rejects_invalid_values(
    generation_config_factory: Any,
    updates: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    """Reject non-boolean controls, non-finite or non-positive delays, and inverted bounds."""
    config_path, _template = generation_config_factory(license_maximum_delay_seconds=300.0, license_maximum_wait_seconds=3600.0)
    _edit_layer(config_path, "execution.yaml", _set_license_wait(updates))

    with pytest.raises(exception, match=message):
        generation.cases.config.load_campaign_config(config_path)


def test_license_retry_configuration_preserves_non_default_values(generation_config_factory: Any) -> None:
    """Resolve a test-owned wait budget and in-allocation window unchanged."""
    config_path, _template = generation_config_factory()
    _edit_layer(
        config_path,
        "execution.yaml",
        _set_license_wait(
            {
                "maximum_wait_seconds": 1800.0,
                "in_allocation_retry.maximum_window_seconds": 17.0,
                "in_allocation_retry.pause_after_capacity_failure_seconds": 3.0,
            }
        ),
    )

    retry = generation.cases.config.load_campaign_config(config_path).batches[0].execution_values["runtime"]["license_wait"]
    assert retry["maximum_wait_seconds"] == 1800.0
    assert retry["in_allocation_retry"] == {"enabled": True, "maximum_window_seconds": 17.0, "pause_after_capacity_failure_seconds": 3.0}


def test_operation_provenance_does_not_change_scientific_identity(
    generation_config_factory: Any,
) -> None:
    """Keep evidence descriptions outside generated-value identity."""
    config_path, _template = generation_config_factory()
    original = generation.cases.config.load_campaign_config(config_path)
    original_batch = original.batches[0]

    _edit_layer(
        config_path,
        "operations.yaml",
        lambda operations: operations["parameter_values"]["pressure_bc.mean"]["provenance"].update(note="Changed evidence description only."),
    )

    changed = generation.cases.config.load_campaign_config(config_path)
    changed_batch = changed.batches[0]
    assert changed_batch.scientific_values != original_batch.scientific_values
    assert changed_batch.scientific_config_digest == original_batch.scientific_config_digest
    assert changed_batch.case_input_config_digest == original_batch.case_input_config_digest
    assert changed_batch.batch_id == original_batch.batch_id
    assert changed.campaign_digest == original.campaign_digest
