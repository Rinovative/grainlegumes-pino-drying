# ruff: noqa: S101, PLR2004
"""Final configuration, naming, ownership, sampling, and domain contracts."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from src import common, generation
from src.datasets.packages import dataset_packages_planning as package_planning
from src.generation.cases import generation_cases_fields as fields
from src.generation.cases import generation_cases_sampling as sampling
from src.generation.cases import generation_cases_seeding as seeding

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


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
    profile_path = config_path.parent / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["exports"][0]["columns"]["x"] = None
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    campaign = generation.cases.config.load_campaign_config(config_path, require_executable=False)
    assert "x" not in campaign.batches[0].scientific_values["output_contract"]["exports"][0]["columns"]
    with pytest.raises(
        generation.cases.config.GenerationConfigError,
        match="incomplete; an explicit executable mapping declaration is required",
    ):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize("option", ["--exclusive", "--reservation=reserved", "--nodelist=node-a"])
def test_execution_rejects_nonordinary_scheduler_options(
    generation_config_factory: Any,
    option: str,
) -> None:
    """Prevent execution YAML from reintroducing exclusivity or reservations."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["cluster"]["scheduler_options"] = [option]
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")

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
    )
    assert versions == (1, 1, 1)
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


def test_startup_ramp_has_transient_input_identity_without_affecting_steady_batches(
    generation_config_factory: Any,
) -> None:
    """Bind the COMSOL handoff policy only to transient scientific inputs."""
    transient_path, _template = generation_config_factory(simulation_profile="transient_drying")
    original_transient = generation.cases.config.load_campaign_config(transient_path).batches[0]
    operations_path = transient_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.25
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")
    changed_transient = generation.cases.config.load_campaign_config(transient_path).batches[0]

    assert changed_transient.scientific_values["boundary_schedule"] == {"startup_ramp": {"enabled": True, "duration_h": 0.25}}
    assert changed_transient.scientific_values["time"] == original_transient.scientific_values["time"]
    assert changed_transient.scientific_config_digest != original_transient.scientific_config_digest
    assert changed_transient.case_input_config_digest != original_transient.case_input_config_digest
    assert changed_transient.batch_id != original_transient.batch_id

    steady_path, _template = generation_config_factory(simulation_profile="steady_flow")
    original_steady = generation.cases.config.load_campaign_config(steady_path).batches[0]
    steady_operations_path = steady_path.parent / "operations.yaml"
    steady_operations = yaml.safe_load(steady_operations_path.read_text(encoding="utf-8"))
    steady_operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.25
    steady_operations_path.write_text(yaml.safe_dump(steady_operations, sort_keys=False), encoding="utf-8")
    changed_steady = generation.cases.config.load_campaign_config(steady_path).batches[0]

    assert "boundary_schedule" not in changed_steady.scientific_values
    assert changed_steady.scientific_config_digest == original_steady.scientific_config_digest
    assert changed_steady.case_input_config_digest == original_steady.case_input_config_digest
    assert changed_steady.batch_id == original_steady.batch_id


def test_startup_ramp_half_hour_duration_resolves(
    generation_config_factory: Any,
) -> None:
    """Resolve the maintained half-hour startup duration as transient science."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    operations_path = config_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.5
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")

    batch = generation.cases.config.load_campaign_config(config_path).batches[0]

    assert batch.scientific_values["boundary_schedule"] == {
        "startup_ramp": {"enabled": True, "duration_h": 0.5},
    }
    assert batch.scientific_values["time"]["interval"] == 1.0


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("enabled", "true", "enabled must be boolean"),
        ("duration_h", 0.0, "strictly positive and shorter"),
        ("duration_h", 1.0, "strictly positive and shorter"),
    ],
)
def test_startup_ramp_configuration_is_strict(
    generation_config_factory: Any,
    key: str,
    value: object,
    message: str,
) -> None:
    """Reject disabled types and durations outside one regular interval."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    operations_path = config_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    operations["boundary_schedule"]["startup_ramp"][key] = value
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")

    with pytest.raises(generation.cases.config.GenerationConfigError, match=message):
        generation.cases.config.load_campaign_config(config_path)


def test_valid_config_edits_are_resolved_without_source_synchronization(
    generation_config_factory: Any,
) -> None:
    """Resolve test-owned science and execution edits from their YAML owners."""
    campaign_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
        campaign_purpose="family_generalization",
    )
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    common_path = campaign_path.parent / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    operations_path = campaign_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    execution_path = campaign_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))

    campaign["sampling"]["seed_base"] = 12345
    campaign["membership"]["seed"] = 12346
    common["grid"].update({"nx": 17, "ny": 11, "Lx": 1.6, "Ly": 1.0})
    common["time"].update({"start": 0.0, "stop": 84.0, "interval": 0.5})
    operations["boundary_schedule"]["startup_ramp"]["duration_h"] = 0.25
    common["scientific_fixed_values"]["T_flow_ref"]["value"] = 301.15
    common["scientific_fixed_values"]["p_ref"]["value"] = 100000.0
    common["storage"]["compression_level"] = 6
    execution["runtime"].update({"timeout_seconds": 4200, "maximum_failures": 3})
    execution["submission"].update(
        {
            "pending_buffer": 2,
            "poll_interval_seconds": 7,
            "max_running_cases": 3,
        }
    )
    execution["cluster"]["cores_per_case"] = 8
    execution["site"]["cpu_host"] = "synthetic-host"

    campaign_path.write_text(
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

    resolved = generation.cases.config.load_campaign_config(campaign_path)
    scientific = resolved.batches[0].scientific_values

    assert resolved.total_case_count == 3
    assert resolved.membership == campaign["membership"]
    assert scientific["campaign_seed"] == 12345
    assert scientific["grid"]["dx"] == pytest.approx(0.1)
    assert scientific["grid"]["dy"] == pytest.approx(0.1)
    assert scientific["time"]["regular_times"] == [index * 0.5 for index in range(169)]
    assert scientific["boundary_schedule"]["startup_ramp"]["duration_h"] == 0.25
    assert scientific["scientific_fixed_values"]["T_flow_ref"] == 301.15
    assert scientific["scientific_fixed_values"]["p_ref"] == 100000.0
    assert scientific["storage"]["compression_level"] == 6
    assert resolved.execution_values["runtime"]["timeout_seconds"] == 4200.0
    assert resolved.execution_values["runtime"]["maximum_failures"] == 3
    assert resolved.execution_values["cluster"]["cores_per_case"] == 8
    assert resolved.execution_values["submission"] == {
        "pending_buffer": 2,
        "poll_interval_seconds": 7,
        "max_running_cases": 3,
    }
    assert resolved.execution_values["site"]["cpu_host"] == "synthetic-host"


def test_sampling_coordinate_order_is_config_owned_and_identity_bound(
    generation_config_factory: Any,
) -> None:
    """Bind explicit registry coordinate order through scientific identity."""
    campaign_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
        campaign_purpose="family_generalization",
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
    """Reject invalid counts, membership, role overlap, and package sources."""

    def expect_failure(
        mutate: Callable[[dict[str, Any]], None],
        expected_key: str | Callable[[dict[str, Any]], str],
    ) -> None:
        campaign_path, _template = generation_config_factory(
            simulation_profile="transient_drying",
            natural_count=3,
            campaign_purpose="family_generalization",
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
    expect_failure(
        lambda campaign: campaign["membership"]["per_seen_material"].__setitem__("train", 999),
        "membership.per_seen_material",
    )
    expect_failure(
        lambda campaign: campaign["material_roles"]["near_family_ood"].append("lentil"),
        "material_roles",
    )
    expect_failure(
        lambda campaign: campaign["dataset_packages"][0].__setitem__(
            "source_role",
            "far_family_ood",
        ),
        "dataset_packages[0]",
    )


def test_readiness_distinguishes_failed_static_sentinels_from_pending(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an executed failing scientific sentinel as a launch blocker."""
    monkeypatch.setattr(
        generation.readiness.sentinel_service,
        "run_static_sentinels",
        lambda *_args: {"status": "blocked_by_scientific_sanity_guard"},
    )
    steady_path, _steady_template = generation_config_factory(
        simulation_profile="steady_flow",
        natural_count=3,
        campaign_purpose="family_generalization",
    )
    transient_path, _transient_template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
        campaign_purpose="family_generalization",
    )
    for campaign_path in (steady_path, transient_path):
        campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
        campaign["profile_config"] = str((campaign_path.parent / "profile.yaml").resolve())
        campaign_path.write_text(
            yaml.safe_dump(campaign, sort_keys=False),
            encoding="utf-8",
        )
    report = generation.readiness.build_readiness_report(
        steady_path,
        transient_path,
        run_static_sentinels=True,
    )
    assert "STATIC_GENERATOR_SENTINELS_BLOCKED" in report["status_lines"]
    assert report["production_ready_for_user_launch"] is False


def test_schedule_csv_is_the_identity_bound_comsol_handoff(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Hash the transformed table while retaining the separate hourly output grid."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "transient_drying",
            "lentil",
            "natural",
        ),
    )
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "startup_case")
    schedule_path = bundle.directory / "inputs" / "schedule.csv"
    schedule = np.loadtxt(schedule_path, delimiter=";", skiprows=1)
    duration_h = config.scientific_values["boundary_schedule"]["startup_ramp"]["duration_h"]

    schedule_times = schedule[:, 0]
    regular_times = np.asarray(
        config.scientific_values["time"]["regular_times"],
        dtype=np.float64,
    )
    assert duration_h == 0.5
    np.testing.assert_array_equal(schedule_times[:3], (0.0, 0.5, 1.0))
    assert schedule_times[-1] == regular_times[-1]
    assert duration_h in schedule_times
    assert 1.0 / 6.0 not in schedule_times
    assert duration_h not in regular_times
    assert np.all(np.diff(schedule_times) > 0.0)
    assert all(time in schedule_times for time in regular_times)
    np.testing.assert_array_equal(
        schedule_times[np.isin(schedule_times, regular_times)],
        regular_times,
    )
    assert schedule[0, 1] == bundle.case_payload["sampled_values"]["T_init"]
    schedule_identity = bundle.case_payload["input_files"]["schedule.csv"]
    assert schedule_identity == {
        "sha256": common.serialization.file_sha256(schedule_path),
        "size_bytes": schedule_path.stat().st_size,
    }
    assert bundle.case_payload["case_input_id"] == generation.cases.case.compute_case_input_id(bundle.case_payload)
    assert bundle.case_payload["schedule_diagnostics"]["boundary_handoff"]["startup_ramp"]["enabled"] is True


def test_scalar_handoff_rejects_an_unknown_name(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect the exact scalar schema after a modified file receives a fresh hash."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "transient_drying",
            "lentil",
            "natural",
        ),
    )
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "case")
    scalar_path = bundle.directory / "inputs" / "scalars.csv"
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
            bundle.directory / "inputs",
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


def test_steady_conditioning_audit_rejects_hidden_case_varying_solver_input(
    generation_config_factory: Any,
) -> None:
    """Require every varying stationary dependency to be a declared task input."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    contract = copy.deepcopy(batch.scientific_values["steady_flow_conditioning"])
    record = {
        "simulation_profile": "steady_flow",
        "steady_flow_conditioning": contract,
    }

    accepted = package_planning.audit_steady_flow_conditioning([record])
    assert accepted["hidden_conditioning"] is False
    assert accepted["T_flow_ref_owner"] == "package_fixed"
    coupled_record = copy.deepcopy(record)
    coupled_record["simulation_profile"] = "transient_drying"
    mixed = package_planning.audit_steady_flow_conditioning([record, coupled_record])
    assert mixed["source_profiles"] == ["steady_flow", "transient_drying"]
    assert mixed["conditioning_contract_digest"] == accepted["conditioning_contract_digest"]

    hidden = copy.deepcopy(record)
    dependency = next(item for item in hidden["steady_flow_conditioning"]["dependencies"] if item["name"] == "T_flow_ref")
    dependency["affects_stationary_solution"] = True
    dependency["owner"] = "model_input"
    with pytest.raises(ValueError, match=r"Hidden steady-flow conditioning.*T_flow_ref"):
        package_planning.audit_steady_flow_conditioning([hidden])


def test_duplicate_source_policy_is_explicit() -> None:
    """Resolve matched physical inputs only through an explicit source policy."""
    physical_id = "c" * 64
    candidates = [
        {
            "package_case_id": "steady_case",
            "simulation_case_id": "1" * 64,
            "case_input_id": physical_id,
            "simulation_profile": "steady_flow",
            "material_family": "lentil",
        },
        {
            "package_case_id": "transient_case",
            "simulation_case_id": "2" * 64,
            "case_input_id": physical_id,
            "simulation_profile": "transient_drying",
            "material_family": "lentil",
        },
    ]
    with pytest.raises(ValueError, match="requires an explicit source preference"):
        package_planning.resolve_duplicate_case_inputs(
            candidates,
            dataset_view="steady_flow",
            policy="reject_duplicates",
        )

    selected, decisions = package_planning.resolve_duplicate_case_inputs(
        list(reversed(candidates)),
        dataset_view="steady_flow",
        policy="prefer_transient_source",
    )
    assert [candidate["package_case_id"] for candidate in selected] == ["transient_case"]
    assert decisions[0]["selected_simulation_case_id"] == "2" * 64
    assert decisions[0]["excluded_simulation_case_ids"] == ["1" * 64]


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
    common_path = config_path.parent / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    common["validation"]["transient_bulk_moisture"][key] = value
    common_path.write_text(
        yaml.safe_dump(common, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        generation.cases.config.GenerationConfigError,
        match=message,
    ):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize(
    ("key", "value", "exception", "message"),
    [
        ("enabled", 1, TypeError, "must be boolean"),
        (
            "initial_delay_seconds",
            True,
            generation.cases.config.GenerationConfigError,
            "finite real value",
        ),
        (
            "initial_delay_seconds",
            0,
            generation.cases.config.GenerationConfigError,
            "must be positive",
        ),
        (
            "maximum_delay_seconds",
            float("inf"),
            generation.cases.config.GenerationConfigError,
            "finite real value",
        ),
        (
            "maximum_wait_seconds",
            -1,
            generation.cases.config.GenerationConfigError,
            "must be positive",
        ),
    ],
)
def test_temporary_license_retry_configuration_validation(
    generation_config_factory: Any,
    key: str,
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Reject non-boolean controls and non-finite or non-positive delays."""
    config_path, _template = generation_config_factory()
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["runtime"]["temporary_license_retry"][key] = value
    execution_path.write_text(
        yaml.safe_dump(execution, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(exception, match=message):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize(
    ("maximum_key", "maximum_value"),
    [
        ("maximum_delay_seconds", 59),
        ("maximum_wait_seconds", 59),
    ],
)
def test_temporary_license_retry_configuration_bounds(
    generation_config_factory: Any,
    maximum_key: str,
    maximum_value: int,
) -> None:
    """Require both retry maxima to cover at least the initial delay."""
    config_path, _template = generation_config_factory()
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["runtime"]["temporary_license_retry"][maximum_key] = maximum_value
    execution_path.write_text(
        yaml.safe_dump(execution, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        generation.cases.config.GenerationConfigError,
        match="must be at least initial_delay_seconds",
    ):
        generation.cases.config.load_campaign_config(config_path)


def test_operation_provenance_does_not_change_scientific_identity(
    generation_config_factory: Any,
) -> None:
    """Keep evidence descriptions outside generated-value identity."""
    config_path, _template = generation_config_factory()
    original = generation.cases.config.load_campaign_config(config_path)
    original_batch = original.batches[0]

    operations_path = config_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    record = operations["parameter_values"]["pressure_bc.mean"]
    record["provenance"]["note"] = "Changed evidence description only."
    operations_path.write_text(
        yaml.safe_dump(operations, sort_keys=False),
        encoding="utf-8",
    )

    changed = generation.cases.config.load_campaign_config(config_path)
    changed_batch = changed.batches[0]
    assert changed_batch.scientific_values != original_batch.scientific_values
    assert changed_batch.scientific_config_digest == original_batch.scientific_config_digest
    assert changed_batch.case_input_config_digest == original_batch.case_input_config_digest
    assert changed_batch.batch_id == original_batch.batch_id
    assert changed.campaign_digest == original.campaign_digest
