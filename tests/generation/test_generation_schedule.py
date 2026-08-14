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


def test_schedule_replays_deterministically() -> None:
    """Reproduce the same scientific schedule from the same explicit seeds."""
    first = _schedule()
    second = _schedule()

    np.testing.assert_array_equal(first.values, second.values)


def test_comsol_startup_handoff_preserves_canonical_schedule_and_rejoins_exactly() -> None:
    """Add one physical startup node without changing the canonical realization."""
    canonical = _schedule()
    canonical_values = canonical.values.copy()
    duration_h = 1.0 / 6.0

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
    np.testing.assert_array_equal(handoff.values[1], expected_rejoin)
    np.testing.assert_array_equal(handoff.values[2:], canonical.values[1:])

    representative_times = np.asarray([duration_h, 0.25, 0.5, 1.0, 1.5, 17.25, 167.5])
    for column in range(1, len(schedule_service.profiles.SCHEDULE_FIELDS)):
        original = np.interp(representative_times, canonical.values[:, 0], canonical.values[:, column])
        transformed = np.interp(representative_times, handoff.values[:, 0], handoff.values[:, column])
        np.testing.assert_allclose(transformed, original, rtol=0.0, atol=2.0e-15)


def test_disabled_comsol_startup_handoff_is_semantically_canonical() -> None:
    """Leave canonical schedule values unchanged when startup is disabled."""
    canonical = _schedule()
    handoff = schedule_service.build_comsol_boundary_schedule(
        canonical,
        {"enabled": False, "duration_h": 1.0 / 6.0},
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
            {"enabled": True, "duration_h": 1.0 / 6.0},
            initial_temperature=-1.0,
            pressure=float(_FIXED["p_ref"]),
        )


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
