# ruff: noqa: S101, PLR2004, SLF001
"""Final configuration, naming, ownership, sampling, and domain contracts."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src import common, datasets, domain, generation
from src.datasets import dataset_transient_contract as transient_contract


def test_production_templates_are_role_neutral_fail_closed_campaigns() -> None:
    """Protect six role-neutral materials and fail-closed campaign launch gates."""
    primary_paths = (
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
    )
    technical_paths = (
        Path("configs/generation/campaigns/steady_flow/technical_smoke.yaml"),
        Path("configs/generation/campaigns/transient_drying/technical_smoke.yaml"),
    )
    expected_roles = {
        "seen": ("lentil", "chickpea", "kidney_bean"),
        "near_family_ood": ("field_pea",),
        "far_family_ood": ("rapeseed",),
        "extreme_family_ood": ("sunflower_seed",),
    }
    campaign_keys = {
        "schema_kind",
        "schema_version",
        "campaign_purpose",
        "sources_config",
        "registry_config",
        "common_config",
        "operations_config",
        "profile_config",
        "execution_config",
        "material_roles",
        "sampling",
        "membership",
        "dataset_packages",
    }
    expected_case_counts = {"steady_flow": 1200, "transient_drying": 660}
    for path in primary_paths:
        authored = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(authored) == campaign_keys
        assert all(
            set(package)
            <= {
                "evaluation_regime",
                "source_role",
                "membership_seed",
                "membership_counts_per_material",
            }
            for package in authored["dataset_packages"]
        )
        campaign = generation.config.load_campaign_config(path, require_executable=False)
        assert campaign.profile.id == path.parent.name
        assert campaign.material_roles == expected_roles
        assert len(campaign.batches) == 9
        assert sum(len(batch.case_indices) for batch in campaign.batches) == expected_case_counts[path.parent.name]
        extreme_case_count = sum(len(batch.case_indices) for batch in campaign.batches if batch.evaluation_regime == "extreme_family_ood")
        reduced = campaign.without_extreme_family_ood()
        assert reduced.material_roles == campaign.material_roles
        assert reduced.total_case_count == campaign.total_case_count - extreme_case_count
        assert all(batch.evaluation_regime != "extreme_family_ood" for batch in reduced.batches)
        assert all(package["evaluation_regime"] != "extreme_family_ood" for package in reduced.dataset_packages)
        assert [
            package["evaluation_regime"]
            for package in campaign.dataset_packages
            if package["dataset_view"] == campaign.profile.available_learning_views[-1]
        ] == ["id", "parameter_ood", "near_family_ood", "far_family_ood", "extreme_family_ood"]
        assert all(
            export["mapping_state"] in {"declared_unverified", "mapping_probe_required"}
            for export in campaign.batches[0].scientific_values["output_contract"]["exports"]
        )
        with pytest.raises(generation.config.GenerationConfigError, match="unconfirmed required export mappings"):
            generation.config.load_campaign_config(path)
    for path in technical_paths:
        campaign = generation.config.load_campaign_config(path, require_executable=False)
        assert campaign.campaign_purpose == "technical_runtime_smoke"
        assert len(campaign.batches) == 1
        assert len(campaign.batches[0].case_indices) == 2
        assert all(package["evaluation_regime"] == "id" for package in campaign.dataset_packages)
        assert all("membership" not in package for package in campaign.dataset_packages)
        assert all(
            package["split_eligibility"] == {"train": False, "validation": False, "id_test": False, "parameter_ood": False}
            for package in campaign.dataset_packages
        )
        with pytest.raises(generation.config.GenerationConfigError, match="unconfirmed required export mappings"):
            generation.config.load_campaign_config(path)

    expected_keys = {
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
    material_paths = tuple(sorted(Path("configs/generation/materials").glob("*.yaml")))
    assert tuple(path.stem for path in material_paths) == tuple(sorted(generation.materials.MATERIAL_FAMILIES))
    for path in material_paths:
        material = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(material) == expected_keys
        assert material["material_family"] == path.stem
        assert material["decision_source"]["sha256"] == generation.materials.VP2_DECISION_SHA256
        assert not ({"role", "count", "sampling_regime", "dataset_membership"} & set(material))
        assert set(material["material_scope"]) == {
            "common_name",
            "species",
            "market_class",
            "product_form",
            "coat_or_hull_state",
            "description",
        }
        assert all(
            "provenance" in material[record]
            for record in (
                "permeability",
                "packing_porosity_mean_support",
                "density_calibration",
                "target_moisture",
                "oswin",
                "two_compartment_kinetics",
            )
        )
    sunflower = yaml.safe_load(Path("configs/generation/materials/sunflower_seed.yaml").read_text(encoding="utf-8"))
    assert sunflower["material_scope"]["product_form"] == "whole_achene"
    assert sunflower["material_scope"]["coat_or_hull_state"] == "hull_intact"


def test_parameter_ood_allocation_covers_profile_eligible_units_evenly() -> None:
    """Protect exact one-unit OOD assignments derived from profile applicability."""
    paths = (
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
    )
    for path in paths:
        campaign = generation.config.load_campaign_config(path, require_executable=False)
        for batch in (item for item in campaign.batches if item.sampling_regime == "parameter_ood"):
            policy = batch.scientific_values["parameter_ood"]
            eligible = policy["eligible_units"]
            allocation = policy["case_allocation"]
            counts = policy["allocation_counts"]
            assert len(allocation) == len(batch.case_indices)
            assert set(counts) == {unit["unit_id"] for unit in eligible}
            assert min(counts.values()) >= 1
            assert max(counts.values()) - min(counts.values()) <= 1
            assert {unit["ood_group"] for unit in eligible} == set(generation.materials.active_ood_groups(campaign.profile.id))
            for case_index in (batch.case_indices[0], batch.case_indices[-1]):
                assignment = batch.case_assignment(case_index)
                sample = generation.sampling.sample_case(batch, case_index)
                assert sample.ood_provenance["active_unit_id"] == assignment["ood_unit_id"]
                assert sample.ood_provenance["active_ood_group"] == assignment["ood_group"]
                assert sample.ood_provenance["units_per_case"] == 1


def test_generation_and_hdf5_versions_are_integer_one(
    generation_config_factory: Any,
) -> None:
    """Reject pre-dataset named version strings at the owning config loader."""
    config_path, _template = generation_config_factory()
    batch = generation.config.load_campaign_config(config_path).batches[0]
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
    with pytest.raises(generation.config.GenerationConfigError, match="generator_version must be an integer"):
        generation.config.load_campaign_config(config_path)

    common["generator_version"] = 1
    common["storage"]["converter_version"] = "named-version"
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    with pytest.raises(generation.config.GenerationConfigError, match="converter_version must be an integer"):
        generation.config.load_campaign_config(config_path)


def test_final_names_dimensions_inventory_and_profile_pairing(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect canonical names, complete ownership, dimensions, and paired inputs."""
    config_path = Path("configs/generation/campaigns/transient_drying/family_generalization.yaml")
    with monkeypatch.context() as production_environment:
        production_environment.setenv("PROJECT_ROOT", str(Path.cwd()))
        campaign = generation.config.load_campaign_config(config_path, require_executable=False)
    expected_batches = [f"transient_drying__{material_family}__natural" for material_family in generation.materials.MATERIAL_FAMILIES] + [
        f"transient_drying__{material_family}__parameter_ood" for material_family in generation.materials.MATERIAL_FAMILIES[:3]
    ]
    assert [batch.batch_name for batch in campaign.batches] == expected_batches
    regimes = ("id", "parameter_ood", "near_family_ood", "far_family_ood", "extreme_family_ood")
    package_materials = {
        "id": "lentil+chickpea+kidney_bean",
        "parameter_ood": "lentil+chickpea+kidney_bean",
        "near_family_ood": "field_pea",
        "far_family_ood": "rapeseed",
        "extreme_family_ood": "sunflower_seed",
    }
    expected_packages = [
        f"{dataset_view}__{package_materials[regime]}__{regime}" for dataset_view in ("steady_flow", "transient_drying") for regime in regimes
    ]
    assert [package["dataset_name"] for package in campaign.dataset_packages] == expected_packages
    assert regimes == datasets.views.PACKAGE_REGIMES
    assert campaign.material_memberships["extreme_family_ood"] == ("sunflower_seed",)
    assert all(
        "sunflower_seed" not in campaign.material_memberships[membership] for membership in ("train", "validation", "id_test", "parameter_ood")
    )
    steady_membership = campaign.dataset_packages[0]["membership"]
    transient_membership = campaign.dataset_packages[5]["membership"]
    assert steady_membership == transient_membership
    assert {
        "seed": steady_membership["seed"],
        "per_seen_material": steady_membership["per_seen_material"],
    } == campaign.membership
    assert steady_membership["totals"] == {"train": 288, "validation": 36, "id_test": 36}
    reports = generation.inventory.audit_campaign(campaign)
    for report in reports.values():
        assert report.sampled_dimensions_by_block == {
            "airflow": 28,
            "initial_moisture": 8,
            "operation": 12,
            "material_properties": 6,
        }
        assert report.total_effective_dimension == 54
        assert report.configured_but_unused == ()
        assert report.consumed_but_undeclared == ()

    registry = campaign.batches[0].scientific_values["material"]["parameter_registry"]
    expected_airflow_parameters = (
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
    expected_schedule_controls = (
        "schedule.corr",
        "schedule.timescale_rel",
        "schedule.component_weights",
        "schedule.event_count",
        "schedule.event_duration_rel",
        "schedule.event_width_rel",
    )
    assert expected_airflow_parameters == generation.materials.AIRFLOW_PARAMETERS
    assert tuple(name for name in generation.materials.OPERATION_PARAMETERS if name.startswith("schedule.")) == expected_schedule_controls
    assert set(generation.materials.SAMPLING_BLOCKS) == {
        "airflow",
        "initial_moisture",
        "operation",
        "material_properties",
    }

    final_names = {
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
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "initial_moisture.structure.coarse_len_rel",
        "initial_moisture.structure.fine_len_rel",
        "initial_moisture.structure.coarse_weight",
        "initial_moisture.structure.fine_weight",
        "initial_moisture.structure.cross_scale_corr",
        "initial_moisture.structure.fine_ani_x",
        "initial_moisture.structure.fine_ani_y",
    }
    assert final_names.issubset(registry)
    assert registry["kappa_mean"]["unit"] == "m^2"
    assert registry["kappa_cv"]["unit"] == "1"
    assert registry["bed.structure.fine_weight"]["sources"] == ["bed.structure.coarse_weight"]
    assert registry["initial_moisture.structure.fine_weight"]["sources"] == ["initial_moisture.structure.coarse_weight"]
    raw_registry = yaml.safe_load(Path("configs/generation/registry.yaml").read_text(encoding="utf-8"))
    _definitions, metadata = generation.materials.validate_semantic_registry(raw_registry)
    symbols = {name: entry["report_symbol"] for name, entry in metadata.items()}
    assert len(set(symbols.values())) == len(symbols)
    assert {
        name: symbols[name]
        for name in (
            "bed.structure.cross_scale_corr",
            "initial_moisture.structure.cross_scale_corr",
            "schedule.corr",
        )
    } == {
        "bed.structure.cross_scale_corr": r"\rho_b",
        "initial_moisture.structure.cross_scale_corr": r"\rho_X",
        "schedule.corr": r"\rho_{T,\omega}",
    }

    canonical_scientific_names = (
        set(registry)
        | set(generation.profiles.STEADY_SPATIAL_INPUT_FIELDS)
        | set(generation.profiles.TRANSIENT_SPATIAL_INPUT_FIELDS)
        | set(generation.profiles.STATIONARY_FIXED_FIELDS)
        | set(generation.profiles.TRANSIENT_SCALAR_INPUT_FIELDS)
        | set(generation.profiles.SCHEDULE_FIELDS)
        | set(generation.profiles.STEADY_STATIC_FIELD_NAMES)
        | set(generation.profiles.TRANSIENT_STATIC_FIELD_NAMES)
        | set(generation.profiles.TRANSIENT_FIELD_NAMES)
        | set(generation.profiles.GLOBAL_FIELD_NAMES)
        | set(generation.profiles.FINAL_STATUS_FIELDS)
    )
    assert not {name for name in canonical_scientific_names if name.endswith(("_K", "_Pa", "_m", "_h"))}
    common = yaml.safe_load(Path("configs/generation/common.yaml").read_text(encoding="utf-8"))
    assert common["physical_formulas"]["w_gr"] == "f_surf*w_surf + (1-f_surf)*w_int"
    assert common["physical_formulas"]["f_wet_dm"] == ("integral(rho_bu_dry*indicator(X_wb>X_target_wb))/integral(rho_bu_dry)")
    assert common["scientific_fixed_values"]["T_flow_ref"]["value"] == 300.65
    assert common["scientific_fixed_values"]["f_wet_dm_max"]["value"] == 0.05
    assert {"f_wet_dm", "m_w_gr", "m_v_gas", "m_dot_evap", "m_dot_v_in", "m_dot_v_out"}.issubset(generation.profiles.GLOBAL_FIELD_NAMES)
    assert "f_wet_dm_final" in generation.profiles.FINAL_STATUS_FIELDS

    steady_path, _ = generation_config_factory(simulation_profile="steady_flow")
    transient_path, _ = generation_config_factory(simulation_profile="transient_drying")
    steady = generation.config.load_generation_config(steady_path, only_batch="steady_flow__lentil__natural")
    transient = generation.config.load_generation_config(
        transient_path,
        only_batch="transient_drying__lentil__natural",
    )
    assert steady.case_input_config_digest != transient.case_input_config_digest
    assert steady.scientific_config_digest != transient.scientific_config_digest
    steady_bundle = generation.case.generate_case_input_bundle(steady, 1, tmp_path / "steady")
    transient_bundle = generation.case.generate_case_input_bundle(transient, 1, tmp_path / "transient")
    assert steady_bundle.case_input_id != transient_bundle.case_input_id
    assert steady_bundle.simulation_case_id != transient_bundle.simulation_case_id
    assert {path.name for path in steady_bundle.input_paths} == {"fields.csv"}
    assert {path.name for path in transient_bundle.input_paths} == {"fields.csv", "scalars.csv", "schedule.csv"}
    assert transient_bundle.case_payload["schedule_diagnostics"]["generator_kind"] == "compositional_mixed"
    assert transient_bundle.case_payload["schedule_diagnostics"]["generator_version"] == 1
    assert not (steady_bundle.directory / "scalars.csv").exists()
    assert not (steady_bundle.directory / "schedule.csv").exists()
    assert "scalar_handoff" not in steady_bundle.case_payload
    assert "scalars" not in steady_bundle.case_payload
    assert steady_bundle.case_payload["stationary_fixed_values"] == [
        {
            "name": name,
            "value": generation.profiles.STATIONARY_FIXED_VALUES[name],
            "unit": unit,
            "owner": "package_fixed",
            "runtime_source": "canonical_template",
        }
        for name, unit in zip(
            generation.profiles.STATIONARY_FIXED_FIELDS,
            generation.profiles.STATIONARY_FIXED_UNITS,
            strict=True,
        )
    ]
    with (steady_bundle.directory / "fields.csv").open(encoding="utf-8", newline="") as stream:
        steady_fields = list(csv.reader(stream, delimiter=";"))
    with (transient_bundle.directory / "fields.csv").open(encoding="utf-8", newline="") as stream:
        transient_fields = list(csv.reader(stream, delimiter=";"))
    assert tuple(steady_fields[0]) == generation.profiles.STEADY_SPATIAL_INPUT_FIELDS
    assert tuple(transient_fields[0]) == generation.profiles.TRANSIENT_SPATIAL_INPUT_FIELDS
    assert [row[:7] for row in transient_fields[1:]] == steady_fields[1:]
    with (transient_bundle.directory / "scalars.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        transient_scalars = list(csv.DictReader(stream, delimiter=";"))
    assert tuple(row["name"] for row in transient_scalars) == generation.profiles.TRANSIENT_SCALAR_INPUT_FIELDS


def test_readiness_distinguishes_failed_static_sentinels_from_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an executed failing scientific sentinel as a launch blocker."""
    monkeypatch.setattr(
        generation.readiness.sentinel_service,
        "run_static_sentinels",
        lambda *_args: {"status": "blocked_by_scientific_sanity_guard"},
    )
    report = generation.readiness.build_readiness_report(
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
        run_static_sentinels=True,
    )
    assert "STATIC_GENERATOR_SENTINELS_BLOCKED" in report["status_lines"]
    assert report["production_ready_for_user_launch"] is False


def test_resolved_parameter_inspection_exposes_atomic_provenance_and_coordinates() -> None:
    """Expose true design coordinates and inherited complete-record provenance."""
    transient = generation.config.load_campaign_config(
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
        require_executable=False,
    )
    report = next(iter(generation.inventory.audit_campaign(transient).values()))
    assert len(report.sampled_coordinate_names) == report.total_effective_dimension == 54
    assert "schedule.component_weights[1]" in report.sampled_coordinate_names
    assert "schedule.component_weights[2]" in report.sampled_coordinate_names

    oswin = generation.inventory.inspect_campaign_parameter(transient, "oswin")
    oswin_component = generation.inventory.inspect_campaign_parameter(transient, "A_osw")
    density = generation.inventory.inspect_campaign_parameter(transient, "density_calibration")
    density_component = generation.inventory.inspect_campaign_parameter(transient, "eps_bed_cal_ref")
    kinetics = generation.inventory.inspect_campaign_parameter(transient, "two_compartment_kinetics")
    simplex = generation.inventory.inspect_campaign_parameter(transient, "schedule_simplex")
    schedule_bound = generation.inventory.inspect_campaign_parameter(transient, "T_in_min")
    clip_bound = generation.inventory.inspect_campaign_parameter(transient, "phi_clip_min")
    permeability_mean = generation.inventory.inspect_campaign_parameter(transient, "kappa_mean")
    template_fixed = generation.inventory.inspect_campaign_parameter(transient, "cp_w")
    grid_spacing = generation.inventory.inspect_campaign_parameter(transient, "grid.dx")
    wall_coefficient = generation.inventory.inspect_campaign_parameter(transient, "U_wall")
    time_stop = generation.inventory.inspect_campaign_parameter(transient, "time.stop")
    density_formula = generation.inventory.inspect_campaign_parameter(
        transient,
        "physical_formulas.rho_bu_dry",
    )
    porosity_support = generation.inventory.inspect_campaign_parameter(
        transient,
        "packing_porosity_mean_support",
    )
    assert oswin_component["atomic_record"] == "oswin"
    assert density_component["atomic_record"] == "density_calibration"
    assert density["coordinate_labels"] == ["rho_bu_dry_ref"]
    assert kinetics["coordinate_labels"] == ["r_surf_0", "r_int_surf", "f_surf"]
    assert simplex["coordinate_labels"] == [
        "schedule.component_weights[1]",
        "schedule.component_weights[2]",
    ]
    assert "generation_schedule.generate_schedule" in simplex["producer_to_consumer_path"]["effective_downstream_consumers"]
    assert schedule_bound["producer_to_consumer_path"]["runtime_mapping_state"] == "generator_consumed"
    assert clip_bound["producer_to_consumer_path"]["runtime_mapping_state"] == ("generator_consumed_and_template_fixed_requires_native_verification")
    assert clip_bound["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "generation_schedule feasibility or psychrometric conversion",
        "canonical COMSOL template fixed physics; Python has no runtime setter",
    ]
    assert permeability_mean["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "generation_fields._permeability_fields",
        "generation_fields._porosity_field",
    ]
    assert template_fixed["producer_to_consumer_path"]["runtime_mapping_state"] == ("template_fixed_no_python_runtime_setter")
    assert template_fixed["provenance"]["derivation"]["origin"] == "supplied_by_handoff"
    assert grid_spacing["configured"] == {"dx": 0.003}
    assert grid_spacing["provenance"]["status"] == "derived"
    assert grid_spacing["provenance"]["derivation"]["verification"] == "mathematically_reproduced"
    assert wall_coefficient["configured"]["value"] == 3.687943262411348
    assert wall_coefficient["provenance"]["derivation"]["verification"] == "mathematically_reproduced"
    authored_common = yaml.safe_load(Path("configs/generation/common.yaml").read_text(encoding="utf-8"))
    assert "dx" not in authored_common["grid"]
    assert "dy" not in authored_common["grid"]
    assert "value" not in authored_common["scientific_fixed_values"]["U_wall"]
    assert time_stop["configured"] == {"stop": 168.0}
    assert density_formula["configured"] == {"rho_bu_dry": "rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_cal_ref)"}
    assert density_formula["producer_to_consumer_path"]["runtime_mapping_state"] == ("template_formula_requires_native_model_verification")
    assert set(porosity_support["materials"]) == set(generation.materials.MATERIAL_FAMILIES)
    assert all(not material["applicability"]["parameter_ood"] for material in porosity_support["materials"].values())
    for family in generation.materials.MATERIAL_FAMILIES:
        assert oswin_component["materials"][family]["provenance"] == oswin["materials"][family]["provenance"]
        assert density_component["materials"][family]["provenance"] == density["materials"][family]["provenance"]

    steady = generation.config.load_campaign_config(
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        require_executable=False,
    )
    steady_reference_pressure = generation.inventory.inspect_campaign_parameter(steady, "p_ref")
    assert steady_reference_pressure["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "canonical steady COMSOL template fixed conditioning"
    ]
    assert steady_reference_pressure["producer_to_consumer_path"]["runtime_mapping_state"] == ("template_fixed_no_python_runtime_setter")
    with pytest.raises(ValueError, match="not applicable to profile 'steady_flow'"):
        generation.inventory.inspect_campaign_parameter(steady, "density_calibration")
    with pytest.raises(ValueError, match="not applicable to profile 'steady_flow'"):
        generation.inventory.inspect_campaign_parameter(steady, "time.stop")


def test_scalar_handoff_rejects_an_unknown_name(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect the exact scalar schema after a modified file receives a fresh hash."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    config = generation.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    bundle = generation.case.generate_case_input_bundle(config, 1, tmp_path / "case")
    scalar_path = bundle.directory / "scalars.csv"
    content = scalar_path.read_text(encoding="utf-8")
    old = "T_flow_ref;"
    assert content.count(old) == 1
    scalar_path.write_text(content.replace(old, "unexpected_flow_temperature;"), encoding="utf-8")
    payload = copy.deepcopy(bundle.case_payload)
    payload["input_files"]["scalars.csv"] = {
        "sha256": common.serialization.file_sha256(scalar_path),
        "size_bytes": scalar_path.stat().st_size,
    }
    with pytest.raises(ValueError, match="missing, duplicate, unknown"):
        generation.storage._transient_scalar_values(
            payload,
            bundle.directory,
        )


def test_each_sampled_morphology_control_changes_its_owned_field(generation_config_factory: Any) -> None:
    """Protect material consumption of all 17 independent morphology controls."""
    config_path, _template = generation_config_factory()
    campaign = generation.config.load_campaign_config(config_path)
    batch = campaign.batch("transient_drying__lentil__natural")
    sample = generation.sampling.sample_case(batch, 1)
    values = sample.values
    registry = batch.scientific_values["material"]["parameter_registry"]
    grid = {"Lx": 1.2, "Ly": 0.75, "Lz": 0.8, "dx": 0.015, "dy": 0.015, "nx": 81, "ny": 51}
    seeds = {"bed": 101, "pressure_bc": 202, "initial_moisture": 303}
    family_contract = batch.scientific_values["material"]
    family_bounds = family_contract["initial_moisture_bounds"]
    porosity_support = family_contract["packing_porosity_mean_support"]
    baseline = generation.fields.generate_spatial_fields(
        "transient_drying",
        grid,
        values,
        seeds=seeds,
        family_bounds=family_bounds,
        packing_porosity_mean_support=porosity_support,
        material_kappa_nominal=float(registry["kappa_mean"]["nominal"]),
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
        generated = generation.fields.generate_spatial_fields(
            "transient_drying",
            grid,
            variant,
            seeds=seeds,
            family_bounds=family_bounds,
            packing_porosity_mean_support=porosity_support,
            material_kappa_nominal=float(registry["kappa_mean"]["nominal"]),
            active_ood_unit=None,
        )
        fields = ("Kxx", "Kxy", "Kyy", "eps_bed") if name in bed_names else ("X_0_db_field",)
        difference = max(float(np.max(np.abs(generated.columns[field] - baseline.columns[field]))) for field in fields)
        scale = max(float(np.max(np.abs(baseline.columns[field]))) for field in fields)
        observed[name] = difference / max(scale, np.finfo(np.float64).tiny)
    assert set(observed) == set(bed_names) | set(moisture_names)
    assert all(effect > 1e-10 for effect in observed.values()), observed


def test_domain_moisture_dataset_membership_and_task_scope() -> None:
    """Protect one moisture implementation, deterministic ID splits, and task scope."""
    dry_basis = np.asarray([0.0, 0.25, 1.0])
    wet_basis = domain.moisture.dry_basis_to_wet_basis(dry_basis)
    np.testing.assert_allclose(wet_basis, [0.0, 0.2, 0.5])
    np.testing.assert_allclose(domain.moisture.wet_basis_to_dry_basis(wet_basis), dry_basis)
    water = domain.moisture.granular_water_content([1.0, 2.0], [3.0, 4.0], 0.5)
    np.testing.assert_allclose(water, [2.0, 3.0])
    np.testing.assert_allclose(domain.moisture.dry_basis_moisture(water, [20.0, 30.0]), [0.1, 0.1])
    assert domain.moisture.bulk_wet_basis_moisture(water, [20.0, 30.0]) == pytest.approx(5.0 / 55.0)

    candidates = [
        {
            "material_family": "lentil",
            "package_case_id": f"lentil/case_{index}",
            "case_input_id": character * 64,
        }
        for index, character in enumerate("abcdef", start=1)
    ]
    plan = {
        "evaluation_regime": "id",
        "materials": ["lentil"],
        "membership_seed": 19,
        "membership_counts_per_material": {"train": 2, "validation": 1, "id_test": 1},
    }
    membership_a = datasets.packages._shared_id_membership(copy.deepcopy(plan), copy.deepcopy(candidates))
    membership_b = datasets.packages._shared_id_membership(copy.deepcopy(plan), list(reversed(copy.deepcopy(candidates))))
    assert membership_a == membership_b
    assert len(membership_a) == 4
    assert list(membership_a.values()).count("train") == 2
    assert list(membership_a.values()).count("validation") == 1
    assert list(membership_a.values()).count("id_test") == 1

    task = domain.tasks.registry.get_task("steady_flow")
    assert task.input_names == ("x", "y", "Kxx", "Kxy", "Kyy", "eps_bed", "p_in_bc")
    assert task.output_names == ("p", "u", "v")
    assert task.tensor_layout == ("batch", "channel", "y", "x")
    assert task.default_datasets.train == "lhs_var80_seed3001"
    assert task.default_datasets.ood == ("lhs_var120_seed4001",)
    assert "transient_drying" not in domain.tasks.registry.available_tasks()
    assert datasets.views.available_views() == ("steady_flow", "transient_drying")
    assert datasets.views.get_view("steady_flow").trainable is True
    assert datasets.views.get_view("transient_drying").trainable is False
    contract = datasets.transient_contract.TRANSIENT_STEP_CONTRACT
    assert tuple(field.name for field in contract.dynamic_state) == ("T", "phi", "w_surf", "w_int")
    assert tuple(field.name for field in contract.static_spatial_conditioning) == (
        "x",
        "y",
        "u",
        "v",
        "p",
        "eps_bed",
        "rho_bu_dry",
    )
    assert tuple(field.name for field in contract.step_boundary_conditioning) == (
        "T_in_bc_t_n",
        "T_in_bc_t_np1",
        "phi_in_bc_t_n",
        "phi_in_bc_t_np1",
        "T_amb",
    )
    assert tuple(field.name for field in contract.scalar_conditioning) == (
        "r_surf_0",
        "r_int_surf",
        "f_surf",
        "A_osw",
        "B_osw",
        "C_osw",
        "k_gr",
        "cp_gr_dry",
    )
    assert tuple(field.name for field in contract.archived_ablation_fields) == (
        "Kxx",
        "Kxy",
        "Kyy",
        "p_in_bc",
        "X_0_db_field",
    )
    assert contract.time_step == 1.0
    assert contract.time_unit == "h"
    assert contract.canonical_storage_representation == "absolute_physical_states"


def test_steady_conditioning_audit_rejects_hidden_case_varying_solver_input(
    generation_config_factory: Any,
) -> None:
    """Require every varying stationary dependency to be a declared task input."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    campaign = generation.config.load_campaign_config(config_path)
    batch = campaign.batch("steady_flow__lentil__natural")
    contract = copy.deepcopy(batch.scientific_values["steady_flow_conditioning"])
    record = {
        "simulation_profile": "steady_flow",
        "steady_flow_conditioning": contract,
    }

    accepted = datasets.packages.audit_steady_flow_conditioning([record])
    assert accepted["hidden_conditioning"] is False
    assert accepted["T_flow_ref_owner"] == "package_fixed"
    coupled_record = copy.deepcopy(record)
    coupled_record["simulation_profile"] = "transient_drying"
    mixed = datasets.packages.audit_steady_flow_conditioning([record, coupled_record])
    assert mixed["source_profiles"] == ["steady_flow", "transient_drying"]
    assert mixed["conditioning_contract_digest"] == accepted["conditioning_contract_digest"]

    hidden = copy.deepcopy(record)
    dependency = next(item for item in hidden["steady_flow_conditioning"]["dependencies"] if item["name"] == "T_flow_ref")
    dependency["affects_stationary_solution"] = True
    dependency["owner"] = "model_input"
    with pytest.raises(ValueError, match=r"Hidden steady-flow conditioning.*T_flow_ref"):
        datasets.packages.audit_steady_flow_conditioning([hidden])


def test_parameter_ood_eligibility_uses_registry_dependency_blocks() -> None:
    """Include airflow OOD in both views while excluding transient-only changes from steady."""
    config_path = Path("configs/generation/campaigns/transient_drying/family_generalization.yaml")
    campaign = generation.config.load_campaign_config(config_path, require_executable=False)
    batch = campaign.batch("transient_drying__lentil__parameter_ood")
    registry = batch.scientific_values["material"]["parameter_registry"]

    def candidate(parameter: str) -> dict[str, Any]:
        entry = registry[parameter]
        return {
            "package_case_id": f"case__{parameter}",
            "batch": batch,
            "case_payload": {
                "ood": {
                    "group": entry["ood_group"],
                    "selected_units": [parameter],
                    "units_per_case": 1,
                    "selections": {
                        parameter: {
                            "selection_kind": "scalar_interval",
                            "transformed_gap": 0.2,
                        }
                    },
                },
                "sampled_values": {parameter: 1.0},
                "coupled_selections": {},
                "block_provenance": {entry["block"]: {"design": "synthetic"}},
            },
        }

    airflow = candidate("pressure_bc.mean")
    steady_eligible, steady_parameters, steady_evidence, _reason = datasets.packages._ood_eligibility(
        airflow,
        view=datasets.views.get_view("steady_flow"),
    )
    transient_eligible, transient_parameters, _transient_evidence, _reason = datasets.packages._ood_eligibility(
        airflow,
        view=datasets.views.get_view("transient_drying"),
    )
    assert steady_eligible is transient_eligible is True
    assert steady_parameters == transient_parameters == ("pressure_bc.mean",)
    assert steady_evidence["parameters"][0]["block"] == "airflow"

    moisture = candidate("initial_moisture.mean_db")
    eligible, parameters, evidence, reason = datasets.packages._ood_eligibility(
        moisture,
        view=datasets.views.get_view("steady_flow"),
    )
    assert eligible is False
    assert parameters == ()
    assert evidence["group"] == "initial_moisture"
    assert "steady_flow" in str(reason)
    assert (
        datasets.packages._ood_eligibility(
            moisture,
            view=datasets.views.get_view("transient_drying"),
        )[0]
        is True
    )


def test_duplicate_source_policy_and_id_ood_overlap_are_explicit() -> None:
    """Resolve matched physical inputs deterministically and reject train/OOD leakage."""
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
        datasets.packages.resolve_duplicate_case_inputs(
            candidates,
            dataset_view="steady_flow",
            policy="reject_duplicates",
        )

    selected, decisions = datasets.packages.resolve_duplicate_case_inputs(
        list(reversed(candidates)),
        dataset_view="steady_flow",
        policy="prefer_transient_source",
    )
    assert [candidate["package_case_id"] for candidate in selected] == ["transient_case"]
    assert decisions[0]["selected_simulation_case_id"] == "2" * 64
    assert decisions[0]["excluded_simulation_case_ids"] == ["1" * 64]

    id_package = datasets.packages._PreparedPackage(
        plan={"evaluation_regime": "id"},
        batch_records=[],
        candidates=[
            {
                "simulation_case_id": "3" * 64,
                "case_input_id": physical_id,
                "dataset_membership": "train",
            }
        ],
        excluded=[],
        membership={},
        source_decisions=[],
        steady_conditioning=None,
    )
    ood_package = datasets.packages._PreparedPackage(
        plan={"evaluation_regime": "parameter_ood"},
        batch_records=[],
        candidates=[
            {
                "simulation_case_id": "4" * 64,
                "case_input_id": physical_id,
                "dataset_membership": "parameter_ood",
            }
        ],
        excluded=[],
        membership={},
        source_decisions=[],
        steady_conditioning=None,
    )
    with pytest.raises(ValueError, match="ID training and OOD package source overlap"):
        datasets.packages._validate_no_id_ood_overlap((id_package, ood_package))


def test_heater_schedule_retries_complete_realization_deterministically(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an infeasible whole schedule and replay the accepted retry exactly."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    batch = generation.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )
    sample = generation.sampling.sample_case(batch, 1)
    assert batch.seed_base is not None
    seeds = {name: generation.config.derive_seed(batch.seed_base, "case", "1", name) for name in ("schedule_shared", "schedule_independent")}
    original = generation.schedule._candidate_schedule

    def force_first_rejection(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original(*args, **kwargs)
        if kwargs["attempt"] != 1:
            return result
        temperature, *remaining = result
        rejected = np.full_like(temperature, float(sample.values["T_amb"]) - 1.0)
        return (rejected, *remaining)

    monkeypatch.setattr(generation.schedule, "_candidate_schedule", force_first_rejection)
    first = generation.schedule.generate_schedule(
        sample.values,
        batch.scientific_values["time"],
        batch.scientific_values["scientific_fixed_values"],
        seeds=seeds,
    )
    second = generation.schedule.generate_schedule(
        sample.values,
        batch.scientific_values["time"],
        batch.scientific_values["scientific_fixed_values"],
        seeds=seeds,
    )
    np.testing.assert_array_equal(first.values, second.values)
    assert first.metadata == second.metadata
    assert first.metadata["schedule_rejection_count"] == 1
    assert first.metadata["schedule_acceptance_attempt"] == 2
    assert first.metadata["min_heater_temperature_rise"] >= 0.0
    assert 0.0 < first.metadata["min_phi_source_air"] <= first.metadata["max_phi_source_air"] <= 1.0
    assert first.metadata["column_order"] == ["t", "T_in_bc", "omega_in_bc", "phi_in_bc"]
    assert first.metadata["phi_source_air_usage"] == "validation_and_provenance_only"


def _markdown_row(section: str, label: str) -> list[str]:
    """Return cells from one uniquely labelled Markdown table row."""
    rows = [line for line in section.splitlines() if line.startswith(f"| {label} |")]
    assert len(rows) == 1
    return [cell.strip() for cell in rows[0].strip("|").split("|")]


def _code_tokens(cell: str) -> tuple[str, ...]:
    """Return the inline-code tokens in one Markdown cell."""
    parts = cell.split("`")
    return tuple(parts[index] for index in range(1, len(parts), 2))


def test_documented_learning_views_match_programmatic_contracts() -> None:
    """Protect exact steady and transient channels, dtype, shape, and step semantics."""
    text = Path("docs/generation_parameter_reference.md").read_text(encoding="utf-8")
    steady_section = text.split("### Steady learning view", 1)[1].split(
        "### Transient Neural-Operator learning view",
        1,
    )[0]
    transient_section = text.split("### Transient Neural-Operator learning view", 1)[1].split(
        "## Inspect the resolved contract",
        1,
    )[0]
    task = domain.tasks.registry.get_task("steady_flow")
    assert _code_tokens(_markdown_row(steady_section, "Inputs")[1]) == task.input_names
    assert _code_tokens(_markdown_row(steady_section, "Targets")[1]) == task.output_names

    contract = transient_contract.TRANSIENT_STEP_CONTRACT
    rows = {
        "State at `t_n`": (contract.dynamic_state, (len(contract.dynamic_state), *contract.spatial_shape)),
        "Static fields": (
            contract.static_spatial_conditioning,
            (len(contract.static_spatial_conditioning), *contract.spatial_shape),
        ),
        "Material/scientific scalars": (contract.scalar_conditioning, (len(contract.scalar_conditioning),)),
        "Target": (contract.target_increments, (len(contract.target_increments), *contract.spatial_shape)),
    }
    for label, (fields, shape) in rows.items():
        cells = _markdown_row(transient_section, label)
        assert _code_tokens(cells[1]) == (contract.tensor_dtype,)
        assert _code_tokens(cells[2]) == ("[" + ",".join(str(value) for value in shape) + "]",)
        assert _code_tokens(cells[3])[: len(fields)] == tuple(field.name for field in fields)
    boundary = _markdown_row(transient_section, "Boundary conditioning")
    boundary_names = tuple(field.name.replace("_t_np1", "(t_n+1)").replace("_t_n", "(t_n)") for field in contract.step_boundary_conditioning)
    assert _code_tokens(boundary[1]) == (contract.tensor_dtype,)
    assert _code_tokens(boundary[2]) == (f"[{len(boundary_names)}]",)
    assert _code_tokens(boundary[3]) == boundary_names
    target = _markdown_row(transient_section, "Target")
    assert _code_tokens(target[3])[-1] == "q(t_n+1) - q(t_n)"
    assert _code_tokens(_markdown_row(transient_section, "Time step")[3]) == ("dt = 1 h",)
    assert contract.time_step == 1.0
    assert contract.time_unit == "h"
    assert "`omega_in_bc` remains schedule/provenance" in transient_section
    assert "Material family is metadata, not a one-hot channel" in transient_section
    assert all(f"`{field.name}`" in transient_section for field in contract.archived_ablation_fields)
    assert "Exact irregular stop states are" in transient_section


def test_source_catalogue_and_executable_refs_match_canonical_registry() -> None:
    """Protect one exact human-readable row per source and complete ref resolution."""
    registry = yaml.safe_load(Path("configs/generation/sources.yaml").read_text(encoding="utf-8"))
    sources = registry["sources"]
    source_by_key = {record["source_key"]: record for record in sources}
    assert len(source_by_key) == len(sources)
    text = Path("docs/generation_parameter_reference.md").read_text(encoding="utf-8")
    catalogue = text.split("<!-- source-catalogue:start -->", 1)[1].split(
        "<!-- source-catalogue:end -->",
        1,
    )[0]
    rows = [line for line in catalogue.splitlines() if line.startswith("| `")]
    assert len(rows) == len(sources)
    documented = {}
    for line in rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == 5
        key = cells[0].strip("`")
        assert key not in documented
        documented[key] = cells
    assert set(documented) == set(source_by_key)

    used_refs: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source_ref", "source_refs"}:
                    refs = [item] if isinstance(item, str) else item
                    if isinstance(refs, list):
                        used_refs.update(ref for ref in refs if isinstance(ref, str))
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for path in Path("configs/generation").rglob("*.yaml"):
        if path.name != "sources.yaml":
            collect(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert used_refs <= set(source_by_key)
    for key, source in source_by_key.items():
        cells = documented[key]
        assert cells[1] == " ".join(source["citation"].split())
        assert cells[2] == " ".join(source["identifier"].split())
        assert cells[3].strip("`") == " ".join(source["canonical_locator"].split())
        expected_prefix = "Executable source_refs" if key in used_refs else "Supplied supporting source"
        assert cells[4].startswith(expected_prefix)
        assert catalogue.count(f"| `{key}` |") == 1
