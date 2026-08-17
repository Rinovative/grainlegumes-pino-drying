# ruff: noqa: S101
"""Fixed Kozeny-Carman coupling and bounded packing-scatter invariants."""

from __future__ import annotations

from typing import Any

import pytest

from src.generation.contracts import generation_contracts_porosity as porosity


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
