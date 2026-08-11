# ruff: noqa: S101, SLF001
"""Material-calibrated Kozeny-Carman porosity invariants."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src import generation
from src.datasets.packages import dataset_packages_planning as package_planning
from src.generation.cases import generation_cases_fields as fields
from src.generation.cases import generation_cases_sampling as sampling
from src.generation.contracts import generation_contracts_porosity as porosity
from src.generation.contracts import generation_contracts_registry as registry_service
from src.generation.publication import generation_publication_inventory as inventory


def _campaign_path(profile: str) -> Path:
    """Return the maintained family-generalization campaign path."""
    return Path(f"configs/generation/campaigns/{profile}/family_generalization.yaml")


def test_conditional_anchor_support_reconstructs_configured_material_records() -> None:
    """Protect nominal calibration and exact transformed tails for every configured family."""
    campaign = generation.cases.config.load_campaign_config(
        _campaign_path("transient_drying"),
        require_executable=False,
    )
    material_ids = campaign.material_inventory
    assert material_ids
    assert len(material_ids) == len(set(material_ids))
    common = yaml.safe_load(Path("configs/generation/common.yaml").read_text(encoding="utf-8"))
    eps_min = float(common["parameter_values"]["eps_min_global"]["value"])
    eps_max = float(common["parameter_values"]["eps_max_global"]["value"])
    ood_gap_fraction, ood_width_fraction = registry_service.ood_separation_fractions()
    ood_outer_fraction = ood_gap_fraction + ood_width_fraction

    for material_id in material_ids:
        material = yaml.safe_load(Path(f"configs/generation/materials/{material_id}.yaml").read_text(encoding="utf-8"))
        permeability = float(material["permeability"]["nominal"])
        calibration_porosity = float(material["density_calibration"]["reference"]["eps_bed_cal_ref"])
        packing_support = material["packing_porosity_mean_support"]
        support = porosity.resolve_anchor_factor_support(
            sampled_kappa_mean=permeability,
            material_kappa_nominal=permeability,
            eps_bed_cal_ref=calibration_porosity,
            packing_porosity_mean_support=packing_support,
            eps_min_global=eps_min,
            eps_max_global=eps_max,
            ood_gap_fraction=ood_gap_fraction,
            ood_width_fraction=ood_width_fraction,
        )
        natural = support["id_interval"]
        assert float(natural["lower"]) < 1.0 < float(natural["upper"])
        reconstructed = porosity.solve_reference_porosity(
            permeability,
            float(support["A_KC_reference"]),
            1.0,
            eps_min_global=eps_min,
            eps_max_global=eps_max,
        )
        assert reconstructed == pytest.approx(calibration_porosity, abs=2.0e-15)

        lower_log = float(natural["transformed_lower"])
        upper_log = float(natural["transformed_upper"])
        width = upper_log - lower_log
        tails = {tail["direction"]: tail for tail in support["available_ood_tails"]}
        assert set(tails) == {"lower", "upper"}
        assert support["unavailable_ood_directions"] == []
        expected = {
            "lower": (lower_log - ood_outer_fraction * width, lower_log - ood_gap_fraction * width),
            "upper": (upper_log + ood_gap_fraction * width, upper_log + ood_outer_fraction * width),
        }
        for direction, tail in tails.items():
            assert float(tail["transformed_lower"]) == pytest.approx(expected[direction][0])
            assert float(tail["transformed_upper"]) == pytest.approx(expected[direction][1])
            assert float(tail["transformed_gap_fraction"]) == pytest.approx(ood_gap_fraction)
            assert float(tail["transformed_width_fraction"]) == pytest.approx(ood_width_fraction)
            midpoint = math.sqrt(float(tail["lower"]) * float(tail["upper"]))
            reference = porosity.solve_reference_porosity(
                permeability,
                float(support["A_KC_reference"]),
                midpoint,
                eps_min_global=eps_min,
                eps_max_global=eps_max,
            )
            if direction == "lower":
                assert reference > float(packing_support["upper"])
            else:
                assert reference < float(packing_support["lower"])

    steady_campaign = generation.cases.config.load_campaign_config(
        _campaign_path("steady_flow"),
        require_executable=False,
    )
    steady_reference = inventory.inspect_campaign_parameter(
        steady_campaign,
        "eps_bed_cal_ref",
    )
    transient_reference = inventory.inspect_campaign_parameter(
        campaign,
        "eps_bed_cal_ref",
    )
    assert steady_reference["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "generation.cases.generation_cases_fields._porosity_field"
    ]
    assert transient_reference["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "generation.cases.generation_cases_fields._porosity_field",
        "generation.cases.generation_cases_fields derived dry-density fields",
        "generation.cases.generation_cases_case transient scalar COMSOL adapter",
    ]


def test_anchor_ood_is_seen_only_and_uses_one_active_unit() -> None:
    """Protect conditional tail evidence and one-unit attribution in both profiles."""
    anchor = porosity.ANCHOR_PARAMETER_NAME
    ood_gap_fraction, ood_width_fraction = registry_service.ood_separation_fractions()
    for profile in generation.contracts.available_profile_ids():
        campaign = generation.cases.config.load_campaign_config(
            _campaign_path(profile),
            require_executable=False,
        )
        parameter_batches = tuple(batch for batch in campaign.batches if batch.sampling_regime == "parameter_ood")
        assert {batch.material_role for batch in parameter_batches} == {"seen"}
        for batch in parameter_batches:
            case_index = next(index for index in batch.case_indices if batch.case_assignment(index)["ood_unit_id"] == anchor)
            sample = sampling.sample_case(batch, case_index)
            evidence = sample.conditional_supports[anchor]
            assert sample.ood_provenance["active_unit_id"] == anchor
            assert sample.ood_provenance["units_per_case"] == 1
            assert set(sample.ood_provenance["selections"]) == {anchor}
            assert evidence["support_kind"] in {"ood_lower", "ood_upper"}
            assert evidence["support_kind"] == porosity.classify_anchor_factor(
                sample.values[anchor],
                porosity.resolve_anchor_factor_support(
                    sampled_kappa_mean=sample.values["kappa_mean"],
                    material_kappa_nominal=evidence["material_kappa_nominal"],
                    eps_bed_cal_ref=sample.values["eps_bed_cal_ref"],
                    packing_porosity_mean_support=evidence["packing_porosity_mean_support"],
                    eps_min_global=sample.values["eps_min_global"],
                    eps_max_global=sample.values["eps_max_global"],
                    ood_gap_fraction=ood_gap_fraction,
                    ood_width_fraction=ood_width_fraction,
                ),
            )
            package_evidence = package_planning._parameter_evidence(
                {
                    "package_case_id": f"{profile}:{batch.material_family}:{case_index}",
                    "batch": batch,
                    "case_payload": {
                        "ood": sample.ood_provenance,
                        "sampled_values": sample.values,
                        "coupled_selections": sample.coupled_selections,
                    },
                }
            )
            assert package_evidence["selected_units"] == [anchor]
            assert package_evidence["units_per_case"] == 1
            parameter = package_evidence["parameters"][0]
            assert parameter["kind"] == "conditional_interval"
            assert parameter["transform"] == "conditional_log"
            assert parameter["id_support"] == evidence["id_interval"]
            assert parameter["ood_support"] == evidence["ood_interval"]


def test_global_coupling_and_local_permeability_paths_are_distinct() -> None:
    """Protect kappa/factor mean effects and bitwise local-porosity independence."""
    campaign = generation.cases.config.load_campaign_config(
        _campaign_path("transient_drying"),
        require_executable=False,
    )
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    sample = sampling.sample_case(batch, batch.case_indices[0])
    material = batch.scientific_values["material"]
    registry = material["parameter_registry"]
    values = dict(sample.values)
    values["kappa_mean"] = float(registry["kappa_mean"]["nominal"])
    values["eps_bed_cal_ref"] = float(material["atomic_records"]["density_calibration"]["reference"]["eps_bed_cal_ref"])
    values[porosity.ANCHOR_PARAMETER_NAME] = 1.0
    grid: dict[str, Any] = {
        "Lx": 0.6,
        "Ly": 0.375,
        "Lz": 0.8,
        "dx": 0.015,
        "dy": 0.015,
        "nx": 41,
        "ny": 26,
    }
    seeds = {"bed": 101, "pressure_bc": 202, "initial_moisture": 303}

    def generate(candidate: dict[str, Any]) -> fields.SpatialFields:
        return fields.generate_spatial_fields(
            "transient_drying",
            grid,
            candidate,
            seeds=seeds,
            family_bounds=material["initial_moisture_bounds"],
            packing_porosity_mean_support=material["packing_porosity_mean_support"],
            material_kappa_nominal=float(registry["kappa_mean"]["nominal"]),
            active_ood_unit=None,
        )

    baseline = generate(values)
    changed_kappa = dict(values)
    changed_kappa["kappa_mean"] *= 1.02
    kappa_fields = generate(changed_kappa)
    changed_factor = dict(values)
    changed_factor[porosity.ANCHOR_PARAMETER_NAME] *= 1.02
    factor_fields = generate(changed_factor)
    changed_local = dict(values)
    kappa_cv = registry["kappa_cv"]
    changed_local["kappa_cv"] = max(
        (float(kappa_cv["lower"]), float(kappa_cv["upper"])),
        key=lambda value: abs(value - float(values["kappa_cv"])),
    )
    local_fields = generate(changed_local)

    assert not np.array_equal(baseline.columns["eps_bed"], kappa_fields.columns["eps_bed"])
    assert not np.array_equal(baseline.columns["eps_bed"], factor_fields.columns["eps_bed"])
    assert any(not np.array_equal(baseline.columns[name], kappa_fields.columns[name]) for name in ("Kxx", "Kxy", "Kyy"))
    assert all(np.array_equal(baseline.columns[name], factor_fields.columns[name]) for name in ("Kxx", "Kxy", "Kyy"))
    assert np.array_equal(baseline.columns["eps_bed"], local_fields.columns["eps_bed"])
    assert any(not np.array_equal(baseline.columns[name], local_fields.columns[name]) for name in ("Kxx", "Kxy", "Kyy"))
    assert baseline.metadata["porosity"]["texture_source"] == "z_background"
    assert baseline.metadata["porosity"]["background_field_sha256"] == local_fields.metadata["porosity"]["background_field_sha256"]
