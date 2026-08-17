# ruff: noqa: S101, PLR2004
"""Protect the canonical grid-resolved transient schedule contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.generation.cases import generation_cases_schedule as schedule_service

_TIME = {
    "regular_times": [float(value) for value in range(169)],
    "interval": 1.0,
}
_FIXED = {
    "p_ref": 101325.0,
    "T_in_min": 290.15,
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


def test_schedule_replays_deterministically() -> None:
    """Reproduce the same scientific schedule from the same explicit seeds."""
    first = _schedule()
    second = _schedule()

    np.testing.assert_array_equal(first.values, second.values)


def test_comsol_startup_handoff_preserves_canonical_schedule_and_rejoins_exactly() -> None:
    """Add one physical startup node without changing the canonical realization."""
    canonical = _schedule()
    canonical_values = canonical.values.copy()
    duration_h = 0.5

    handoff = schedule_service.build_comsol_boundary_schedule(
        canonical,
        {"enabled": True, "duration_h": duration_h},
        initial_temperature=293.15,
        pressure=float(_FIXED["p_ref"]),
    )

    np.testing.assert_array_equal(canonical.values, canonical_values)
    np.testing.assert_array_equal(handoff.values[:5, 0], [0.0, duration_h, 1.0, 2.0, 3.0])
    assert handoff.values[-1, 0] == 168.0
    assert handoff.values[0, 1] == 293.15
    assert handoff.values[0, 2] == canonical.values[0, 2]
    expected_start_phi = schedule_service.humidity_ratio_to_relative_humidity(
        handoff.values[:1, 2],
        handoff.values[:1, 1],
        pressure=float(_FIXED["p_ref"]),
    )[0]
    assert handoff.values[0, 3] == expected_start_phi

    fraction = duration_h / float(_TIME["interval"])
    expected_rejoin = canonical.values[0] + fraction * (canonical.values[1] - canonical.values[0])
    expected_rejoin[0] = duration_h
    expected_rejoin[3] = schedule_service.humidity_ratio_to_relative_humidity(
        expected_rejoin[2:3],
        expected_rejoin[1:2],
        pressure=float(_FIXED["p_ref"]),
    )[0]
    np.testing.assert_array_equal(handoff.values[1], expected_rejoin)
    np.testing.assert_array_equal(handoff.values[2:], canonical.values[1:])

    representative_times = np.asarray([duration_h, 1.0, 1.5, 17.25, 167.5])
    for column in (1, 2):
        original = np.interp(representative_times, canonical.values[:, 0], canonical.values[:, column])
        transformed = np.interp(representative_times, handoff.values[:, 0], handoff.values[:, column])
        np.testing.assert_allclose(transformed, original, rtol=0.0, atol=2.0e-15)


def test_comsol_startup_ramp_interpolates_temperature_and_humidity_ratio_linearly() -> None:
    """Derive the thermodynamic midpoint of one analytically simple ramp."""
    canonical_values = np.asarray(
        (
            (0.0, 302.0, 0.008, 0.0),
            (1.0, 310.0, 0.010, 0.0),
            (2.0, 308.0, 0.009, 0.0),
        ),
        dtype=np.float64,
    )
    canonical_values[:, 3] = schedule_service.humidity_ratio_to_relative_humidity(
        canonical_values[:, 2],
        canonical_values[:, 1],
        pressure=float(_FIXED["p_ref"]),
    )
    canonical = schedule_service.Schedule(
        values=canonical_values,
        metadata={
            "temperature_operational_bounds": [_FIXED["T_in_min"], _FIXED["T_in_max"]],
            "humidity_ratio_operational_bounds": [_FIXED["omega_min"], _FIXED["omega_max"]],
            "relative_humidity_operational_bounds": [
                _FIXED["phi_operational_min"],
                _FIXED["phi_operational_max"],
            ],
        },
    )
    handoff = schedule_service.build_comsol_boundary_schedule(
        canonical,
        {"enabled": True, "duration_h": 0.5},
        initial_temperature=298.0,
        pressure=float(_FIXED["p_ref"]),
    )

    np.testing.assert_array_equal(handoff.values[:, 0], (0.0, 0.5, 1.0, 2.0))
    ramp_start = handoff.values[0]
    ramp_end = handoff.values[1]
    midpoint_temperature = np.interp(0.25, handoff.values[:, 0], handoff.values[:, 1])
    midpoint_humidity_ratio = np.interp(0.25, handoff.values[:, 0], handoff.values[:, 2])
    assert midpoint_temperature == ramp_start[1] + 0.5 * (ramp_end[1] - ramp_start[1])
    assert midpoint_humidity_ratio == ramp_start[2] + 0.5 * (ramp_end[2] - ramp_start[2])
    assert midpoint_temperature == 302.0
    assert midpoint_humidity_ratio == 0.0085
    assert ramp_end[1] == 306.0
    assert ramp_end[2] == canonical_values[0, 2] + 0.5 * (canonical_values[1, 2] - canonical_values[0, 2])

    midpoint_phi = schedule_service.humidity_ratio_to_relative_humidity(
        np.asarray([midpoint_humidity_ratio]),
        np.asarray([midpoint_temperature]),
        pressure=float(_FIXED["p_ref"]),
    )[0]
    independently_interpolated_phi = ramp_start[3] + 0.5 * (ramp_end[3] - ramp_start[3])
    assert midpoint_phi != independently_interpolated_phi
    np.testing.assert_allclose(
        handoff.values[:, 3],
        schedule_service.humidity_ratio_to_relative_humidity(
            handoff.values[:, 2],
            handoff.values[:, 1],
            pressure=float(_FIXED["p_ref"]),
        ),
        rtol=64.0 * np.finfo(np.float64).eps,
        atol=64.0 * np.finfo(np.float64).eps,
    )


def test_disabled_comsol_startup_handoff_is_semantically_canonical() -> None:
    """Leave canonical schedule values unchanged when startup is disabled."""
    canonical = _schedule()
    handoff = schedule_service.build_comsol_boundary_schedule(
        canonical,
        {"enabled": False, "duration_h": 0.5},
        initial_temperature=293.15,
        pressure=float(_FIXED["p_ref"]),
    )

    np.testing.assert_array_equal(handoff.values, canonical.values)


def test_comsol_startup_handoff_rejects_invalid_physical_state() -> None:
    """Keep startup validity separate from the canonical operating envelope."""
    canonical = _schedule()
    with pytest.raises(ValueError, match="physically positive"):
        schedule_service.build_comsol_boundary_schedule(
            canonical,
            {"enabled": True, "duration_h": 0.5},
            initial_temperature=-1.0,
            pressure=float(_FIXED["p_ref"]),
        )


def test_comsol_startup_handoff_uses_cold_initial_state_without_preheating() -> None:
    """Begin exactly at the physical initial state outside regular T and RH bounds."""
    initial_temperature = 288.7
    canonical = _schedule(
        T_in_base=308.15,
        T_in_amp=0.0,
        omega_in_base=0.010625,
        omega_in_amp=0.0,
        T_amb=initial_temperature,
    )
    handoff = schedule_service.build_comsol_boundary_schedule(
        canonical,
        {"enabled": True, "duration_h": 0.5},
        initial_temperature=initial_temperature,
        pressure=float(_FIXED["p_ref"]),
    )

    expected_phi = schedule_service.humidity_ratio_to_relative_humidity(
        canonical.values[:1, 2],
        np.asarray([initial_temperature], dtype=np.float64),
        pressure=float(_FIXED["p_ref"]),
    )[0]
    startup = handoff.metadata["boundary_handoff"]["startup_ramp"]
    assert initial_temperature < _FIXED["T_in_min"]
    assert _FIXED["phi_operational_max"] < expected_phi < 1.0
    assert handoff.values[0, 1] == initial_temperature
    assert handoff.values[0, 2] == canonical.values[0, 2]
    assert handoff.values[0, 3] == expected_phi
    assert startup == {
        "enabled": True,
        "duration_h": 0.5,
        "temperature_start_policy": "use_initial_temperature_exactly",
        "initial_temperature_K": initial_temperature,
        "canonical_start_humidity_ratio_kg_per_kg": canonical.values[0, 2],
        "startup_relative_humidity": expected_phi,
        "humidity_start_policy": "preserve_canonical_omega_in_bc_and_recompute_phi_in_bc",
        "rejoin_policy": "interpolate_canonical_temperature_and_humidity_ratio_then_recompute_phi_in_bc",
    }
    assert handoff.metadata["boundary_handoff"]["handoff_version"] == 2


def test_comsol_startup_handoff_rejects_supersaturated_initial_state() -> None:
    """Fail closed instead of repairing a supersaturated physical startup state."""
    canonical = _schedule(
        T_in_base=308.15,
        T_in_amp=0.0,
        omega_in_base=0.010625,
        omega_in_amp=0.0,
        T_amb=288.7,
    )

    with pytest.raises(ValueError, match="physically invalid"):
        schedule_service.build_comsol_boundary_schedule(
            canonical,
            {"enabled": True, "duration_h": 0.5},
            initial_temperature=280.0,
            pressure=float(_FIXED["p_ref"]),
        )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        (1, 289.15),
        (2, 0.015),
        (3, 0.9),
    ],
)
def test_comsol_handoff_rejects_operational_rejoin_violation(
    column: int,
    invalid_value: float,
) -> None:
    """Keep authored operational envelopes on the rejoin and regular nodes."""
    canonical = _schedule()
    handoff = schedule_service.build_comsol_boundary_schedule(
        canonical,
        {"enabled": True, "duration_h": 0.5},
        initial_temperature=288.15,
        pressure=float(_FIXED["p_ref"]),
    )
    invalid = handoff.values.copy()
    invalid[1, column] = invalid_value

    with pytest.raises(ValueError, match="rejoin or canonical regular schedule nodes"):
        schedule_service.validate_comsol_boundary_schedule(
            invalid,
            regular_times=np.asarray(_TIME["regular_times"], dtype=np.float64),
            startup_ramp={"enabled": True, "duration_h": 0.5},
            initial_temperature=288.15,
            pressure=float(_FIXED["p_ref"]),
            metadata=handoff.metadata,
        )


def test_widened_symmetric_amplitude_produces_unclipped_low_temperature_tail() -> None:
    """Retain exact amplitude semantics while permitting a feasible deep excursion."""
    result = _schedule(T_in_base=303.15, T_in_amp=8.0)

    assert np.min(result.values[:, 1]) == pytest.approx(295.15)
    assert np.mean(result.values[:, 1]) == pytest.approx(303.15, abs=2.0e-13)
    assert np.max(np.abs(result.values[:, 1] - 303.15)) == pytest.approx(8.0)
    assert result.diagnostics["schedule_rejection_count"] == 0


def test_whole_schedule_is_rejected_when_exact_amplitude_cannot_be_safe() -> None:
    """Reject every unsafe candidate instead of clipping its temperature nodes."""
    with pytest.raises(ValueError, match="No feasible complete heater-only schedule"):
        _schedule(T_in_base=303.15, T_in_amp=13.5)


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
