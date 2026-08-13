# ruff: noqa: S101, SLF001
"""Fixed Kozeny-Carman coupling and bounded packing-scatter invariants."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src import generation
from src.datasets.packages import dataset_packages_planning as package_planning
from src.generation.cases import generation_cases_case as case_service
from src.generation.cases import generation_cases_fields as fields
from src.generation.cases import generation_cases_sampling as sampling
from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_porosity as porosity
from src.generation.contracts import generation_contracts_profiles as profiles


def _campaign_path(profile: str, purpose: str = "family_generalization") -> Path:
    """Return one maintained campaign path."""
    return Path(f"configs/generation/campaigns/{profile}/{purpose}.yaml")


def _campaign(profile: str, purpose: str = "family_generalization") -> Any:
    """Load one maintained campaign without requiring COMSOL."""
    return generation.cases.config.load_campaign_config(
        _campaign_path(profile, purpose),
        require_executable=False,
    )


def _synthetic_coupling() -> dict[str, Any]:
    """Return one analytically simple coupling with two valid OOD tails."""
    coefficient = porosity.derive_reference_coefficient(1.0e-8, 0.4)

    def permeability(epsilon: float) -> float:
        return coefficient * porosity.kozeny_carman_response(epsilon)

    return porosity.resolve_porosity_coupling(
        material_family="synthetic",
        material_kappa_nominal=1.0e-8,
        eps_bed_cal_ref=0.4,
        authored_permeability_support={
            "lower": permeability(0.33),
            "upper": permeability(0.47),
        },
        packing_porosity_mean_support={"lower": 0.35, "upper": 0.45},
        eps_min_global=0.2,
        eps_max_global=0.8,
        authored_kappa_ood=[
            {"lower": permeability(0.25), "upper": permeability(0.30)},
            {"lower": permeability(0.50), "upper": permeability(0.55)},
        ],
    )


def _small_grid() -> dict[str, Any]:
    """Return one inexpensive Cartesian grid satisfying generation geometry."""
    return {
        "Lx": 0.6,
        "Ly": 0.36,
        "Lz": 0.8,
        "dx": 0.03,
        "dy": 0.03,
        "nx": 21,
        "ny": 13,
    }


def test_all_materials_resolve_fixed_calibration_and_joint_supports() -> None:
    """Protect fixed calibration, monotonic inversion, and every authored tail."""
    for profile in generation.contracts.available_profile_ids():
        campaign = _campaign(profile)
        assert campaign.material_inventory
        for family in campaign.material_inventory:
            batch = campaign.require_batch(
                material_family=family,
                sampling_regime="natural",
            )
            material = batch.scientific_values["material"]
            coupling = material["porosity_coupling"]
            coefficient = float(coupling["A_KC_reference"])
            nominal = float(coupling["material_kappa_nominal"])
            calibration = float(coupling["material_eps_bed_cal_ref"])
            registry = material["parameter_registry"]
            effective = coupling["effective_joint_permeability_support"]
            authored = coupling["authored_permeability_support"]
            kc_support = coupling["kc_compatible_permeability_support"]

            assert math.isfinite(coefficient)
            assert coefficient > 0.0
            assert coefficient == porosity.derive_reference_coefficient(nominal, calibration)
            assert porosity.solve_reference_porosity(
                nominal,
                coefficient,
                eps_min_global=float(registry["eps_min_global"]["value"]),
                eps_max_global=float(registry["eps_max_global"]["value"]),
            ) == pytest.approx(calibration, abs=2.0e-15)
            assert float(effective["lower"]) < float(effective["upper"])
            assert float(effective["lower"]) == max(float(authored["lower"]), float(kc_support["lower"]))
            assert float(effective["upper"]) == min(float(authored["upper"]), float(kc_support["upper"]))
            assert float(registry["kappa_mean"]["lower"]) == float(effective["lower"])
            assert float(registry["kappa_mean"]["upper"]) == float(effective["upper"])
            assert float(coupling["eps_kc_trend_interval"]["lower"]) < float(coupling["eps_kc_trend_interval"]["upper"])

            natural = coupling["natural_porosity_support"]
            for direction, tail in coupling["kappa_ood_porosity_supports"].items():
                mapped = [
                    porosity.solve_reference_porosity(
                        float(tail[name]),
                        coefficient,
                        eps_min_global=float(registry["eps_min_global"]["value"]),
                        eps_max_global=float(registry["eps_max_global"]["value"]),
                    )
                    for name in ("kappa_lower", "kappa_upper")
                ]
                assert mapped == pytest.approx([float(tail["porosity_lower"]), float(tail["porosity_upper"])], abs=2.0e-15)
                if direction == "lower":
                    assert mapped[1] < float(natural["lower"])
                else:
                    assert mapped[0] > float(natural["upper"])

            assert "porosity.kc_anchor_factor" not in registry
            assert "packing_scatter_z" not in registry
            assert "porosity.kc_anchor_factor" not in material["active_coordinate_names"]
            assert "packing_scatter_z" not in material["active_coordinate_names"]
            assert batch.scientific_values["schema_version"] == 1
            assert batch.scientific_values["generator_version"] == 1


def test_empty_or_invalid_supports_fail_closed() -> None:
    """Reject empty ID intersections and OOD mappings with the wrong physical side."""
    coefficient = porosity.derive_reference_coefficient(1.0e-8, 0.42)

    def permeability(epsilon: float) -> float:
        return coefficient * porosity.kozeny_carman_response(epsilon)

    with pytest.raises(ValueError, match="empty KC-compatible intersection"):
        porosity.resolve_porosity_coupling(
            material_family="empty",
            material_kappa_nominal=permeability(0.52),
            eps_bed_cal_ref=0.52,
            authored_permeability_support={
                "lower": permeability(0.50),
                "upper": permeability(0.55),
            },
            packing_porosity_mean_support={"lower": 0.35, "upper": 0.45},
            eps_min_global=0.2,
            eps_max_global=0.8,
        )

    with pytest.raises(ValueError, match="not below natural support"):
        porosity.resolve_porosity_coupling(
            material_family="bad_tail",
            material_kappa_nominal=1.0e-8,
            eps_bed_cal_ref=0.42,
            authored_permeability_support={
                "lower": permeability(0.40),
                "upper": permeability(0.46),
            },
            packing_porosity_mean_support={"lower": 0.35, "upper": 0.45},
            eps_min_global=0.2,
            eps_max_global=0.8,
            authored_kappa_ood=[
                {
                    "lower": permeability(0.36),
                    "upper": permeability(0.38),
                }
            ],
        )

    with pytest.raises(
        ValueError,
        match=r"Material outside_guard lower permeability OOD interval .* maps to porosity .* outside global guards .* natural porosity support",
    ):
        porosity.resolve_porosity_coupling(
            material_family="outside_guard",
            material_kappa_nominal=1.0e-8,
            eps_bed_cal_ref=0.42,
            authored_permeability_support={
                "lower": permeability(0.40),
                "upper": permeability(0.46),
            },
            packing_porosity_mean_support={"lower": 0.35, "upper": 0.45},
            eps_min_global=0.2,
            eps_max_global=0.8,
            authored_kappa_ood=[
                {
                    "lower": permeability(0.10),
                    "upper": permeability(0.15),
                }
            ],
        )


def test_truncated_scatter_uses_one_uniform_draw_and_mirrored_quantiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect open truncation, symmetry, determinism, and one-draw semantics."""
    calls: list[Any] = []

    class FakeRandom:
        def __init__(self, seed: int) -> None:
            calls.append(("seed", seed))

        def random(self) -> float:
            calls.append("draw")
            return 0.25

    monkeypatch.setattr(porosity.random, "Random", FakeRandom)
    assert porosity.draw_truncated_standard_normal(17) == porosity.truncated_standard_normal_quantile(0.25)
    assert calls == [("seed", 17), "draw"]

    for unit_coordinate in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        left = porosity.truncated_standard_normal_quantile(unit_coordinate)
        right = porosity.truncated_standard_normal_quantile(1.0 - unit_coordinate)
        assert porosity.PACKING_SCATTER_TRUNCATION_LOWER < left < porosity.PACKING_SCATTER_TRUNCATION_UPPER
        assert left + right == pytest.approx(0.0, abs=2.0e-14)


def test_packing_scatter_support_boundaries_and_ood_directions() -> None:
    """Protect zero boundary sigma, negative-margin failure, and OOD retention."""
    coupling = _synthetic_coupling()
    base_values = {
        "eps_min_global": 0.2,
        "eps_max_global": 0.8,
    }
    effective = coupling["effective_joint_permeability_support"]
    boundary = fields._packing_reference(
        {**base_values, "kappa_mean": float(effective["lower"])},
        coupling=coupling,
        active_ood_unit=None,
        packing_scatter_seed=123,
    )
    assert boundary["active_kappa_mean_support"] == effective
    assert boundary["packing_scatter_margin"] == 0.0
    assert boundary["packing_scatter_sigma"] == 0.0
    assert boundary["eps_reference"] == pytest.approx(boundary["eps_kc_trend"], abs=2.0e-15)

    invalid = copy.deepcopy(coupling)
    invalid["natural_porosity_support"] = {"lower": 0.36, "upper": 0.45}
    with pytest.raises(fields.PorositySupportError, match="outside active packing support") as error:
        fields._packing_reference(
            {**base_values, "kappa_mean": float(effective["lower"])},
            coupling=invalid,
            active_ood_unit=None,
            packing_scatter_seed=123,
        )
    assert error.value.retryable is False

    natural = coupling["natural_porosity_support"]
    for direction, tail in coupling["kappa_ood_porosity_supports"].items():
        reference = fields._packing_reference(
            {
                **base_values,
                "kappa_mean": math.sqrt(float(tail["kappa_lower"]) * float(tail["kappa_upper"])),
            },
            coupling=coupling,
            active_ood_unit="kappa_mean",
            packing_scatter_seed=456,
        )
        assert reference["packing_scatter_support_kind"] == f"kappa_mean_ood_{direction}"
        assert reference["active_kappa_mean_support"] == {
            "lower": float(tail["kappa_lower"]),
            "upper": float(tail["kappa_upper"]),
        }
        assert reference["packing_scatter_support_lower"] <= reference["eps_reference"] <= reference["packing_scatter_support_upper"]
        if direction == "lower":
            assert reference["eps_reference"] < float(natural["lower"])
        else:
            assert reference["eps_reference"] > float(natural["upper"])


def test_scatter_seed_is_semantic_and_independent_from_spatial_streams() -> None:
    """Protect case identity, paired profiles, retries, and permeability independence."""
    steady = _campaign("steady_flow")
    batch = steady.require_batch(material_family="lentil", sampling_regime="natural")
    sample = sampling.sample_case(batch, batch.case_indices[0])
    material = batch.scientific_values["material"]
    seeds = {"bed": 101, "pressure_bc": 202, "packing_scatter": 303}

    def generated(candidate_seeds: dict[str, int]) -> fields.SpatialFields:
        return fields.generate_spatial_fields(
            "steady_flow",
            _small_grid(),
            sample.values,
            seeds=candidate_seeds,
            family_bounds=None,
            porosity_coupling=material["porosity_coupling"],
            active_ood_unit=None,
        )

    baseline = generated(seeds)
    replay = generated(seeds)
    different_scatter = generated({**seeds, "packing_scatter": 304})
    different_bed = generated({**seeds, "bed": 102})

    for name in baseline.columns:
        assert np.array_equal(baseline.columns[name], replay.columns[name])
    assert baseline.metadata["porosity"]["packing_scatter_z"] == replay.metadata["porosity"]["packing_scatter_z"]
    assert baseline.metadata["porosity"]["packing_scatter_z"] != different_scatter.metadata["porosity"]["packing_scatter_z"]
    assert all(np.array_equal(baseline.columns[name], different_scatter.columns[name]) for name in ("Kxx", "Kxy", "Kyy"))
    assert not np.array_equal(baseline.columns["eps_bed"], different_scatter.columns["eps_bed"])
    assert baseline.metadata["porosity"]["packing_scatter_z"] == different_bed.metadata["porosity"]["packing_scatter_z"]
    assert baseline.metadata["porosity"]["eps_reference"] == different_bed.metadata["porosity"]["eps_reference"]

    retry_seeds = fields._complete_case_retry_seeds(seeds, 2)
    assert retry_seeds["packing_scatter"] == seeds["packing_scatter"]
    assert retry_seeds["bed"] != seeds["bed"]
    assert case_service._subseeds(batch, 1)["packing_scatter"] != case_service._subseeds(batch, 2)["packing_scatter"]

    paired = {
        profile: _campaign(profile, "technical_smoke").require_batch(
            material_family="lentil",
            sampling_regime="natural",
        )
        for profile in generation.contracts.available_profile_ids()
    }
    assert (
        case_service._subseeds(paired["steady_flow"], 1)["packing_scatter"]
        == case_service._subseeds(
            paired["transient_drying"],
            1,
        )["packing_scatter"]
    )


def test_density_ood_cannot_recalibrate_fixed_kc_coefficient() -> None:
    """Keep sampled density calibration out of the canonical KC coefficient."""
    campaign = _campaign("transient_drying")
    batch = campaign.require_batch(material_family="lentil", sampling_regime="parameter_ood")
    case_index = next(index for index in batch.case_indices if batch.case_assignment(index)["ood_unit_id"] == "density_calibration")
    sample = sampling.sample_case(batch, case_index)
    coupling = batch.scientific_values["material"]["porosity_coupling"]
    baseline = fields._packing_reference(
        sample.values,
        coupling=coupling,
        active_ood_unit="density_calibration",
        packing_scatter_seed=9001,
    )
    changed_values = dict(sample.values)
    changed_values["eps_bed_cal_ref"] = float(sample.values["eps_bed_cal_ref"]) + 0.05
    changed = fields._packing_reference(
        changed_values,
        coupling=coupling,
        active_ood_unit="density_calibration",
        packing_scatter_seed=9001,
    )
    assert baseline == changed
    assert baseline["A_KC_reference"] == coupling["A_KC_reference"]
    assert baseline["material_eps_bed_cal_ref"] != sample.values["eps_bed_cal_ref"]


def test_kappa_ood_package_evidence_is_an_authored_interval() -> None:
    """Keep permeability OOD on its ordinary one-unit interval evidence path."""
    for profile in generation.contracts.available_profile_ids():
        campaign = _campaign(profile)
        batch = campaign.require_batch(material_family="lentil", sampling_regime="parameter_ood")
        case_index = next(index for index in batch.case_indices if batch.case_assignment(index)["ood_unit_id"] == "kappa_mean")
        sample = sampling.sample_case(batch, case_index)
        evidence = package_planning._parameter_evidence(
            {
                "package_case_id": f"{profile}:lentil:{case_index}",
                "batch": batch,
                "case_payload": {
                    "ood": sample.ood_provenance,
                    "sampled_values": sample.values,
                    "coupled_selections": sample.coupled_selections,
                },
            }
        )
        assert evidence["selected_units"] == ["kappa_mean"]
        assert evidence["units_per_case"] == 1
        assert evidence["parameters"][0]["kind"] == "interval"
        assert evidence["parameters"][0]["transform"] == "log"
        assert evidence["parameters"][0]["coupled_selection"] is None


def test_new_case_schema_rejects_old_anchor_metadata_at_version_one(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject stale anchor artifacts by exact schema without a version bump."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "steady_flow",
            "lentil",
            "natural",
        ),
    )
    bundle = case_service.generate_case_input_bundle(config, 1, tmp_path / "current case")
    assert bundle.case_payload["schema_version"] == 1
    assert bundle.case_payload["generator_version"] == 1
    case_service.validate_case_payload_schema(bundle.case_payload)

    stale_top_level = copy.deepcopy(bundle.case_payload)
    stale_top_level["conditional_supports"] = {"porosity.kc_anchor_factor": {"support_kind": "natural"}}
    with pytest.raises(ValueError, match=r"unknown=.*conditional_supports"):
        case_service.validate_case_payload_schema(stale_top_level)

    stale_diagnostics = copy.deepcopy(bundle.case_payload)
    diagnostics = stale_diagnostics["spatial_diagnostics"]["porosity"]
    diagnostics["A_KC_case"] = diagnostics.pop("A_KC_reference")
    with pytest.raises(ValueError, match="Porosity diagnostics schema is invalid"):
        case_service.validate_case_payload_schema(stale_diagnostics)


def test_scatter_is_not_an_active_or_learning_input() -> None:
    """Keep latent scatter out of DOE, OOD, persisted sampled values, and input channels."""
    forbidden = {"packing_scatter", "packing_scatter_z", "porosity.kc_anchor_factor"}
    for profile in generation.contracts.available_profile_ids():
        campaign = _campaign(profile)
        for batch in campaign.batches:
            material = batch.scientific_values["material"]
            registry = material["parameter_registry"]
            assert forbidden.isdisjoint(registry)
            assert forbidden.isdisjoint(material["active_coordinate_names"])
            groups = materials.active_ood_groups(registry, profile)
            units = sampling.eligible_ood_units(material, groups=groups)
            assert forbidden.isdisjoint(unit["unit_id"] for unit in units)
            sample = sampling.sample_case(batch, batch.case_indices[0])
            assert forbidden.isdisjoint(sample.values)

    learning_inputs = (
        *profiles.STEADY_SPATIAL_INPUT_FIELDS,
        *profiles.TRANSIENT_SPATIAL_INPUT_FIELDS,
        *profiles.TRANSIENT_SCALAR_INPUT_FIELDS,
        *profiles.SCHEDULE_FIELDS,
    )
    assert forbidden.isdisjoint(learning_inputs)
