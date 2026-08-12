# ruff: noqa: S101, PLR2004, SLF001
"""Protect the canonical grid-resolved transient schedule contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import yaml

from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_fields as field_service
from src.generation.cases import generation_cases_sampling as sampling_service
from src.generation.cases import generation_cases_schedule as schedule_service
from src.generation.contracts import generation_contracts_materials as materials

_TIME = {
    "regular_times": [float(value) for value in range(169)],
    "interval": 1.0,
}
_FIXED = {
    "p_ref": 101325.0,
    "T_in_min": 298.15,
    "T_in_max": 313.15,
    "omega_min": 0.0025,
    "omega_max": 0.0145,
    "phi_operational_min": 0.05,
    "phi_operational_max": 0.85,
    "phi_clip_min": 1.0e-6,
    "phi_clip_max": 0.999,
}
_SEEDS = {"schedule_shared": 918273, "schedule_independent": 564738}


def _values(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "T_in_base": 308.15,
        "T_in_amp": 2.5,
        "omega_in_base": 0.0075,
        "omega_in_amp": 0.001,
        "T_amb": 293.15,
        "schedule.corr": -0.35,
        "schedule.timescale_rel": 0.08,
        "schedule.component_weights": {"smooth": 0.55, "event": 0.3, "trend": 0.15},
        "schedule.event_count": 2,
        "schedule.event_duration_rel": 0.1,
        "schedule.event_width_rel": 0.03,
    }
    values.update(overrides)
    return values


def _schedule(**overrides: Any) -> schedule_service.Schedule:
    return schedule_service.generate_schedule(
        _values(**overrides),
        _TIME,
        _FIXED,
        seeds=_SEEDS,
    )


def test_schedule_replays_byte_identically_and_uses_weights_once() -> None:
    """Protect deterministic replay and single-use simplex composition."""
    first = _schedule()
    second = _schedule()

    np.testing.assert_array_equal(first.values, second.values)
    assert first.metadata == second.metadata
    assert first.values.tobytes() == second.values.tobytes()
    assert first.metadata["component_weight_semantics"] == ("relative_contribution_used_once_before_complete_shape_normalization")
    assert first.metadata["component_availability"] == {
        "smooth": True,
        "event": True,
        "trend": True,
    }

    components = {
        "smooth": np.asarray([-1.0, 0.0, 1.0]),
        "event": np.asarray([1.0, -2.0, 1.0]),
        "trend": np.asarray([-0.5, 0.0, 0.5]),
    }
    weights = {"smooth": 0.2, "event": 0.3, "trend": 0.5}
    expected = sum(weights[name] * components[name] for name in weights)
    np.testing.assert_array_equal(schedule_service._compose(components, weights), expected)


def test_event_count_is_the_only_event_presence_switch() -> None:
    """Protect deterministic event semantics without hidden activation draws."""
    time = np.asarray(_TIME["regular_times"], dtype=np.float64)
    absent, absent_details = schedule_service._event_component(
        time,
        count=0,
        duration=16.8,
        width=5.04,
        random=np.random.default_rng(12),
    )
    first, first_details = schedule_service._event_component(
        time,
        count=3,
        duration=16.8,
        width=5.04,
        random=np.random.default_rng(12),
    )
    second, second_details = schedule_service._event_component(
        time,
        count=3,
        duration=16.8,
        width=5.04,
        random=np.random.default_rng(12),
    )

    np.testing.assert_array_equal(absent, np.zeros_like(time))
    assert absent_details == []
    np.testing.assert_array_equal(first, second)
    assert first_details == second_details
    assert len(first_details) == 3
    assert np.any(first)


@pytest.mark.parametrize("correlation", [-0.7, 0.0, 0.65])
def test_positive_amplitudes_and_correlation_are_exact(correlation: float) -> None:
    """Protect exact means, amplitudes, and discrete-node Pearson correlation."""
    result = _schedule(**{"schedule.corr": correlation})
    temperature = result.values[:, 1]
    humidity_ratio = result.values[:, 2]

    assert np.mean(temperature) == pytest.approx(308.15, abs=2.0e-13)
    assert np.mean(humidity_ratio) == pytest.approx(0.0075, abs=2.0e-15)
    assert np.max(np.abs(temperature - 308.15)) == pytest.approx(2.5, abs=2.0e-13)
    assert np.max(np.abs(humidity_ratio - 0.0075)) == pytest.approx(0.001, abs=2.0e-15)
    assert result.diagnostics["T_in_amp_realization_ratio"] == pytest.approx(1.0, abs=2.0e-13)
    assert result.diagnostics["omega_in_amp_realization_ratio"] == pytest.approx(1.0, abs=2.0e-13)
    assert result.diagnostics["realized_T_omega_correlation"] == pytest.approx(correlation, abs=2.0e-12)
    correlation_error = result.diagnostics["absolute_T_omega_correlation_error"]
    assert correlation_error is not None
    assert correlation_error <= schedule_service.CORRELATION_TOLERANCE


def test_zero_amplitude_is_exactly_constant_and_correlation_is_not_applicable() -> None:
    """Protect intentional zero variance without fabricated correlation."""
    result = _schedule(T_in_amp=0.0)

    np.testing.assert_array_equal(result.values[:, 1], np.full(169, 308.15))
    assert result.diagnostics["constant_T_in_bc"] is True
    assert result.diagnostics["realized_T_omega_correlation"] is None
    assert result.diagnostics["absolute_T_omega_correlation_error"] is None
    assert result.diagnostics["T_in_amp_realization_ratio"] is None


def test_grid_resolution_guards_smooth_and_event_scales() -> None:
    """Reject under-resolved smooth, event-edge, and duration features."""
    result = _schedule()
    assert result.diagnostics["smooth_scale_intervals"] == pytest.approx(13.44)
    assert result.diagnostics["minimum_event_width_intervals"] == pytest.approx(5.04)
    assert result.diagnostics["event_duration_intervals"] == pytest.approx(16.8)

    for name, value, match in (
        ("schedule.timescale_rel", 0.02, "timescale_rel"),
        ("schedule.event_width_rel", 0.005, "event_width_rel"),
        ("schedule.event_duration_rel", 0.02, "event_duration_rel"),
    ):
        with pytest.raises(ValueError, match=match):
            _schedule(**{name: value})


def test_authored_temporal_supports_must_resolve_on_the_grid() -> None:
    """Reject an authored fast tail below the canonical interval requirements."""
    supports = {
        "schedule.timescale_rel": {
            "lower": 0.05,
            "upper": 0.18,
            "ood": [{"lower": 0.024, "upper": 0.036}],
        },
        "schedule.event_width_rel": {
            "lower": 0.02,
            "upper": 0.04,
            "ood": [{"lower": 0.012, "upper": 0.016}],
        },
        "schedule.event_duration_rel": {
            "lower": 0.08,
            "upper": 0.16,
            "ood": [{"lower": 0.04, "upper": 0.06}],
        },
    }
    schedule_service.validate_temporal_support_resolution(supports, _TIME)
    supports["schedule.event_width_rel"]["ood"][0]["lower"] = 0.005
    with pytest.raises(ValueError, match="event_width_rel ood"):
        schedule_service.validate_temporal_support_resolution(supports, _TIME)


def test_temporal_generator_never_calls_spatial_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the temporal process independent from every spatial-field owner."""

    def fail_spatial_call(*_args: Any, **_kwargs: Any) -> None:
        message = "temporal generation invoked spatial generation"
        raise AssertionError(message)

    monkeypatch.setattr(field_service, "generate_spatial_fields", fail_spatial_call)
    _schedule()


def test_prechange_spatial_fields_remain_byte_identical(
    generation_config_factory: Any,
) -> None:
    """Keep schedule-only support changes outside spatial sampling and fields."""
    config_path, _template = generation_config_factory(natural_count=2)
    before_campaign = config_service.load_campaign_config(config_path, require_executable=False)
    before_batch = before_campaign.batch("transient_drying__lentil__natural")
    before = sampling_service.sample_case(before_batch, 1)

    operations_path = config_path.parent / "operations.yaml"
    operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
    schedule_supports = {
        "schedule.timescale_rel": (0.06, 0.16),
        "schedule.event_duration_rel": (0.09, 0.15),
        "schedule.event_width_rel": (0.025, 0.035),
    }
    for name, (lower, upper) in schedule_supports.items():
        operations["parameter_values"][name]["lower"] = lower
        operations["parameter_values"][name]["upper"] = upper
    operations_path.write_text(yaml.safe_dump(operations, sort_keys=False), encoding="utf-8")

    after_campaign = config_service.load_campaign_config(config_path, require_executable=False)
    after_batch = after_campaign.batch("transient_drying__lentil__natural")
    after = sampling_service.sample_case(after_batch, 1)
    schedule_names = frozenset(schedule_supports)
    spatial_and_other_names = set(before.values).difference(schedule_names)
    assert {name: before.values[name] for name in spatial_and_other_names} == {name: after.values[name] for name in spatial_and_other_names}
    assert any(before.values[name] != after.values[name] for name in schedule_names)
    assert before.units == after.units
    assert before.coupled_selections == after.coupled_selections
    assert before.block_provenance == after.block_provenance
    assert before.conditional_supports == after.conditional_supports
    assert before.ood_provenance == after.ood_provenance
    before_sampling = before_batch.scientific_values["sampling"]
    after_sampling = after_batch.scientific_values["sampling"]
    assert before_sampling["blocks"] == after_sampling["blocks"]
    before_position = before_batch.case_indices.index(1)
    after_position = after_batch.case_indices.index(1)
    for block, before_plan in before_sampling["blocks"].items():
        after_plan = after_sampling["blocks"][block]
        before_design = sampling_service.unit_design(
            before_sampling["method"],
            count=len(before_batch.case_indices),
            dimensions=int(before_plan["effective_dimension"]),
            seed=int(before_plan["design_seed"]),
        )
        after_design = sampling_service.unit_design(
            after_sampling["method"],
            count=len(after_batch.case_indices),
            dimensions=int(after_plan["effective_dimension"]),
            seed=int(after_plan["design_seed"]),
        )
        before_row = int(before_plan["permutation"][before_position])
        after_row = int(after_plan["permutation"][after_position])
        np.testing.assert_array_equal(before_design[before_row], after_design[after_row])

    seeds = {
        "bed": 1936762462,
        "pressure_bc": 990883689,
        "initial_moisture": 2503402048,
    }
    grid = {
        "Lx": 1.2,
        "Ly": 0.75,
        "Lz": 0.8,
        "nx": 25,
        "ny": 16,
        "dx": 1.2 / 24.0,
        "dy": 0.75 / 15.0,
        "boundaries_included": True,
    }

    def generate(batch: Any, sample: Any) -> field_service.SpatialFields:
        family = batch.scientific_values["material"]
        moisture_bounds = materials.initial_moisture_generation_bounds(
            family,
            sample.values,
            active_ood_unit=sample.ood_provenance["active_unit_id"],
        )
        return field_service.generate_spatial_fields(
            "transient_drying",
            grid,
            sample.values,
            seeds=seeds,
            family_bounds=moisture_bounds,
            packing_porosity_mean_support=family["packing_porosity_mean_support"],
            material_kappa_nominal=float(family["parameter_registry"]["kappa_mean"]["nominal"]),
            active_ood_unit=sample.ood_provenance["active_unit_id"],
        )

    before_fields = generate(before_batch, before)
    after_fields = generate(after_batch, after)
    for name in ("Kxx", "Kxy", "Kyy", "eps_bed", "X_0_db_field", "p_in_bc"):
        np.testing.assert_array_equal(before_fields.columns[name], after_fields.columns[name])
