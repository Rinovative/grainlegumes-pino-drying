# ruff: noqa: S101, PLR2004, SLF001
"""Final configuration, naming, ownership, sampling, and domain contracts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src import datasets, domain, generation


def test_production_templates_are_role_neutral_fail_closed_campaigns() -> None:
    """Protect nested campaigns, uniform material schemas, and fail-closed science."""
    campaign_paths = sorted(Path("configs/generation/campaigns").rglob("*.yaml"))
    assert campaign_paths
    assert not tuple(Path("configs/generation/campaigns").glob("*.yaml"))
    for path in campaign_paths:
        campaign = generation.config.load_campaign_config(path, require_executable=False)
        assert campaign.profile.id == path.parent.name
        assert campaign.batches
        assert all(not batch.case_indices for batch in campaign.batches)
        assert all("__" in batch.batch_name and len(batch.batch_id.rsplit("__", 1)[-1]) == 16 for batch in campaign.batches)
        assert all("version" not in package["dataset_name"] and "profile" not in package["dataset_name"] for package in campaign.dataset_packages)
        with pytest.raises(generation.config.GenerationConfigError, match="non-executable"):
            generation.config.load_campaign_config(path)

    expected_keys = {
        "schema_kind",
        "schema_version",
        "material_family",
        "executable",
        "taxonomy",
        "product_form",
        "parameter_values",
        "evidence",
    }
    expected_scopes = {
        "lentil": (None, "whole", None),
        "chickpea": ("Kabuli", "whole", None),
        "kidney_bean": ("red kidney bean", "whole", None),
        "field_pea": ("yellow field pea", "whole", None),
        "almond": (None, "whole", "hard_shell_removed"),
    }
    assert tuple(expected_scopes) == generation.materials.MATERIAL_FAMILIES
    for material_family in generation.materials.MATERIAL_FAMILIES:
        material = yaml.safe_load(Path(f"configs/generation/materials/{material_family}.yaml").read_text(encoding="utf-8"))
        assert set(material) == expected_keys
        assert material["material_family"] == material_family
        assert not ({"role", "count", "sampling_regime", "dataset_membership"} & set(material))
        assert material["executable"] is False
        assert set(material["evidence"]) == set(material["parameter_values"])
        taxonomy = material["taxonomy"]
        product_form = material["product_form"]
        market_class, whole_or_split, shell_state = expected_scopes[material_family]
        assert taxonomy["market_class"] == market_class
        assert product_form["whole_or_split"] == whole_or_split
        assert product_form["shell_state"] == shell_state
        assert taxonomy["species"] is taxonomy["cultivar"] is None
        assert product_form["skin_or_seed_coat_state"] is None
        assert taxonomy["specificity_status"] == "unresolved"
        assert product_form["specificity_status"] == "unresolved"


def test_final_names_dimensions_inventory_and_profile_pairing(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect canonical names, complete ownership, dimensions, and paired inputs."""
    config_path, _template = generation_config_factory(material_families=generation.materials.MATERIAL_FAMILIES)
    campaign = generation.config.load_campaign_config(config_path)
    expected_batches = [f"transient_drying__{material_family}__natural" for material_family in generation.materials.MATERIAL_FAMILIES] + [
        f"transient_drying__{material_family}__parameter_ood" for material_family in generation.materials.MATERIAL_FAMILIES[:3]
    ]
    assert [batch.batch_name for batch in campaign.batches] == expected_batches
    assert [package["dataset_name"] for package in campaign.dataset_packages] == [
        "transient_drying__lentil+chickpea+kidney_bean__id",
        "transient_drying__lentil+chickpea+kidney_bean__parameter_ood",
        "transient_drying__field_pea__near_family_ood",
        "transient_drying__almond__far_family_ood",
    ]
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
    old_generator_names = {
        "a_max",
        "a_gamma",
        "tensor_strength",
        "theta_jitter",
        "theta_smooth_rel",
        "A_rel",
        "eps_smooth_rel",
        "texture_amp",
        "p_inlet_mean",
        "a_sin",
        "f_sin",
        "phi_sin",
        "k_gauss",
        "a_gauss",
        "sigma_gauss",
        "gauss_jitter",
        "a_lin",
        "rho_schedule",
        "sched_timescale_rel",
        "schedule_component_weights",
        "sched_event_count",
        "sched_event_duration_rel",
        "sched_event_width_rel",
    }
    assert old_generator_names.isdisjoint(registry)
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
    rejected_registry = copy.deepcopy(raw_registry)
    for old_name, new_name in zip(
        sorted(old_generator_names),
        (
            "porosity.anchor_rel",
            "permeability.anisotropy.exponent",
            "pressure_bc.linear_amp",
            "permeability.anisotropy.max_ratio",
            "pressure_bc.gauss_amp",
            "pressure_bc.sin_amp",
            "porosity.smooth_len_rel",
            "pressure_bc.sin_freq",
            "pressure_bc.gauss_jitter",
            "pressure_bc.gauss_count",
            "pressure_bc.mean",
            "pressure_bc.sin_phase",
            "schedule.corr",
            "schedule.event_count",
            "schedule.event_duration_rel",
            "schedule.event_width_rel",
            "schedule.timescale_rel",
            "schedule.component_weights",
            "pressure_bc.gauss_width",
            "permeability.anisotropy.strength",
            "porosity.texture_amp",
            "permeability.orientation.jitter",
            "permeability.orientation.smooth_len_rel",
        ),
        strict=True,
    ):
        rejected_registry["parameters"][old_name] = rejected_registry["parameters"].pop(new_name)
    with pytest.raises(ValueError, match="Parameter semantics mismatch"):
        generation.materials.validate_semantic_registry(rejected_registry)

    canonical_scientific_names = (
        set(registry)
        | set(generation.profiles.SPATIAL_INPUT_FIELDS)
        | set(generation.profiles.SCALAR_INPUT_FIELDS)
        | set(generation.profiles.SCHEDULE_FIELDS)
        | set(generation.profiles.STATIC_FIELD_NAMES)
        | set(generation.profiles.TRANSIENT_FIELD_NAMES)
        | set(generation.profiles.GLOBAL_FIELD_NAMES)
        | set(generation.profiles.FINAL_STATUS_FIELDS)
    )
    assert not {name for name in canonical_scientific_names if name.endswith(("_K", "_Pa", "_m", "_h"))}
    common = yaml.safe_load(Path("configs/generation/common.yaml").read_text(encoding="utf-8"))
    assert common["physical_formulas"]["f_wet_dm"] == ("integral(rho_bu_dry * indicator(X_wb > X_target_wb)) / integral(rho_bu_dry)")
    assert common["scientific_fixed_values"]["f_wet_dm_max"] == 0.05
    assert {"f_wet_dm", "m_w_gr", "m_v_gas", "m_dot_evap", "m_dot_v_in", "m_dot_v_out"}.issubset(generation.profiles.GLOBAL_FIELD_NAMES)
    assert "f_wet_dm_final" in generation.profiles.FINAL_STATUS_FIELDS

    steady_path, _ = generation_config_factory(simulation_profile="steady_flow")
    transient_path, _ = generation_config_factory(simulation_profile="transient_drying")
    steady = generation.config.load_generation_config(steady_path, only_batch="steady_flow__lentil__natural")
    transient = generation.config.load_generation_config(
        transient_path,
        only_batch="transient_drying__lentil__natural",
    )
    assert steady.case_input_config_digest == transient.case_input_config_digest
    assert steady.scientific_config_digest != transient.scientific_config_digest
    steady_bundle = generation.case.generate_case_input_bundle(steady, 1, tmp_path / "steady")
    transient_bundle = generation.case.generate_case_input_bundle(transient, 1, tmp_path / "transient")
    assert steady_bundle.case_input_id == transient_bundle.case_input_id
    assert steady_bundle.simulation_case_id != transient_bundle.simulation_case_id
    for filename in ("fields.csv", "scalars.csv", "schedule.csv"):
        assert (steady_bundle.directory / filename).read_bytes() == (transient_bundle.directory / filename).read_bytes()


def test_each_sampled_morphology_control_changes_its_owned_field(generation_config_factory: Any) -> None:
    """Protect material consumption of all 17 independent morphology controls."""
    config_path, _template = generation_config_factory(natural_count=1)
    campaign = generation.config.load_campaign_config(config_path)
    batch = campaign.batch("transient_drying__lentil__natural")
    sample = generation.sampling.sample_case(batch, 1)
    values = sample.values
    registry = batch.scientific_values["material"]["parameter_registry"]
    grid = {"Lx": 1.2, "Ly": 0.75, "dx": 0.015, "dy": 0.015, "nx": 81, "ny": 51}
    seeds = {"bed": 101, "pressure_bc": 202, "initial_moisture": 303}
    family_bounds = batch.scientific_values["material"]["initial_moisture_bounds"]
    baseline = generation.fields.generate_spatial_fields(grid, values, seeds=seeds, family_bounds=family_bounds)
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
        candidates = (float(registry[name]["lower"]), float(registry[name]["upper"]))
        variant[name] = max(candidates, key=lambda value: abs(value - float(values[name])))
        if name == "bed.structure.coarse_weight":
            variant["bed.structure.fine_weight"] = 1.0 - variant[name]
        elif name == "initial_moisture.structure.coarse_weight":
            variant["initial_moisture.structure.fine_weight"] = 1.0 - variant[name]
        generated = generation.fields.generate_spatial_fields(grid, variant, seeds=seeds, family_bounds=family_bounds)
        fields = ("Kxx", "Kxy", "Kyy", "eps_bed") if name in bed_names else ("X_0_db_field",)
        difference = max(float(np.max(np.abs(generated.columns[field] - baseline.columns[field]))) for field in fields)
        scale = max(float(np.max(np.abs(baseline.columns[field]))) for field in fields)
        observed[name] = difference / max(scale, np.finfo(np.float64).tiny)
    assert set(observed) == set(bed_names) | set(moisture_names)
    assert all(effect > 1e-10 for effect in observed.values()), observed


def test_paired_density_mode_reduces_only_the_material_block(generation_config_factory: Any) -> None:
    """Protect the evidence-preserving paired density alternative and its sampler."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    campaign = generation.config.load_campaign_config(config_path)
    batch = copy.deepcopy(campaign.batch("steady_flow__lentil__natural"))
    registry = batch.scientific_values["material"]["parameter_registry"]
    semantic = yaml.safe_load(Path("configs/generation/registry.yaml").read_text(encoding="utf-8"))["parameters"]["density_calibration"]
    semantic.pop("report_symbol")
    semantic.pop("description")
    semantic["pairs"] = [{"id": "paired-id", "values": {"rho_bu_dry_ref": 550.0, "eps_bed_cal_ref": 0.45}}]
    semantic["ood_pairs"] = [{"id": "paired-ood", "values": {"rho_bu_dry_ref": 650.0, "eps_bed_cal_ref": 0.55}}]
    registry["density_calibration"] = semantic
    for name, unit in (("rho_bu_dry_ref", "kg/m^3"), ("eps_bed_cal_ref", "1")):
        registry[name] = {
            "kind": "derived",
            "unit": unit,
            "derivation": "selected_parameter_set_component",
            "sources": ["density_calibration"],
        }
    assert generation.materials.validate_vp2_registry(registry) == "paired_density_calibration"
    dimensions = generation.materials.sampling_block_dimensions(registry)
    assert dimensions == {
        "airflow": 28,
        "initial_moisture": 8,
        "operation": 12,
        "material_properties": 5,
    }
    assert generation.inventory.audit_parameter_registry(registry).total_effective_dimension == 53
    assert batch.seed_base is not None
    batch.scientific_values["sampling"]["blocks"] = generation.sampling.build_sampling_plan(
        registry=registry,
        case_count=1,
        seed_base=batch.seed_base,
        method="lhs",
    )
    sample = generation.sampling.sample_case(batch, 1)
    assert sample.coupled_selections["density_calibration"] == "paired-id"
    assert sample.values["rho_bu_dry_ref"] == 550.0
    assert sample.values["eps_bed_cal_ref"] == 0.45


def test_domain_moisture_dataset_membership_and_task_scope() -> None:
    """Protect one moisture implementation, deterministic ID splits, and task scope."""
    dry_basis = np.asarray([0.0, 0.25, 1.0])
    wet_basis = domain.moisture.dry_basis_to_wet_basis(dry_basis)
    np.testing.assert_allclose(wet_basis, [0.0, 0.2, 0.5])
    np.testing.assert_allclose(domain.moisture.wet_basis_to_dry_basis(wet_basis), dry_basis)
    water = domain.moisture.granular_water_content([1.0, 2.0], [3.0, 4.0])
    np.testing.assert_allclose(domain.moisture.dry_basis_moisture(water, [20.0, 30.0]), [0.2, 0.2])
    assert domain.moisture.bulk_wet_basis_moisture(water, [20.0, 30.0]) == pytest.approx(10.0 / 60.0)

    candidates = [
        {
            "material_family": "lentil",
            "package_case_id": f"lentil/case_{index}",
            "record": {"simulation_case_id": character * 64},
        }
        for index, character in enumerate("abcdef", start=1)
    ]
    plan = {
        "evaluation_regime": "id",
        "materials": ["lentil"],
        "membership_seed": 19,
        "membership_counts_per_material": {"train": 2, "validation": 1, "id_test": 1},
    }
    selected_a, membership_a = datasets.packages._assign_membership(copy.deepcopy(plan), copy.deepcopy(candidates))
    selected_b, membership_b = datasets.packages._assign_membership(copy.deepcopy(plan), list(reversed(copy.deepcopy(candidates))))
    assert membership_a == membership_b
    assert [item["package_case_id"] for item in selected_a] == [item["package_case_id"] for item in selected_b]
    assert len({case_id for values in membership_a.values() for case_id in values}) == 4

    task = domain.tasks.registry.get_task("steady_flow")
    assert task.input_names == ("x", "y", "Kxx", "Kxy", "Kyy", "eps_bed", "p_bc")
    assert task.output_names == ("p", "u", "v")
    assert task.tensor_layout == ("batch", "channel", "y", "x")
    assert task.default_datasets.train == "lhs_var80_seed3001"
    assert task.default_datasets.ood == ("lhs_var120_seed4001",)
    assert "transient_drying" not in domain.tasks.registry.available_tasks()
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
        "T_in_t_n",
        "T_in_t_np1",
        "phi_in_t_n",
        "phi_in_t_np1",
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
        "p_bc",
        "X_0_db_field",
    )
    assert contract.time_step == 1.0
    assert contract.time_unit == "h"
    assert contract.canonical_storage_representation == "absolute_physical_states"
