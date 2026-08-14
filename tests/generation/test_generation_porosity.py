# ruff: noqa: S101
"""Fixed Kozeny-Carman coupling and bounded packing-scatter invariants."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from src import generation
from src.generation.contracts import generation_contracts_porosity as porosity


def _campaign_path(profile: str, purpose: str = "family_generalization") -> Path:
    """Return one maintained campaign path."""
    return Path(f"configs/generation/campaigns/{profile}/{purpose}.yaml")


def _campaign(profile: str, purpose: str = "family_generalization") -> Any:
    """Load one maintained campaign without requiring COMSOL."""
    return generation.cases.config.load_campaign_config(
        _campaign_path(profile, purpose),
        require_executable=False,
    )


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
