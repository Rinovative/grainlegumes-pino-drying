# ruff: noqa: S101, PLR2004
"""Protect the canonical grid-resolved transient schedule contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src import domain
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
_OSWIN = {"A_osw": 12.06202053, "B_osw": -0.0573838, "C_osw": 0.34338283}
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


def _policy(
    *,
    enabled: bool = True,
    duration_h: float = 0.5,
    max_relative_humidity: float = 0.90,
) -> dict[str, float | bool]:
    return {
        "enabled": enabled,
        "duration_h": duration_h,
        "initial_equilibrium_rh_dry_margin": 0.05,
        "max_relative_humidity": max_relative_humidity,
    }


def _initial_moisture(
    relative_humidity: np.ndarray,
    *,
    initial_temperature: float,
) -> np.ndarray:
    return domain.moisture.oswin_equilibrium_dry_basis_moisture(
        relative_humidity,
        initial_temperature,
        a_osw=_OSWIN["A_osw"],
        b_osw=_OSWIN["B_osw"],
        c_osw=_OSWIN["C_osw"],
    )


def _build_handoff(
    canonical: schedule_service.Schedule,
    *,
    initial_temperature: float = 293.15,
    source_air_temperature: float | None = None,
    equilibrium_relative_humidity: np.ndarray | None = None,
    enabled: bool = True,
    max_relative_humidity: float = 0.90,
) -> schedule_service.ComsolBoundarySchedule:
    if equilibrium_relative_humidity is None:
        equilibrium_relative_humidity = np.asarray(
            ((0.825, 0.852), (0.841, 0.872)),
            dtype=np.float64,
        )
    return schedule_service.build_comsol_boundary_schedule(
        canonical,
        _policy(enabled=enabled, max_relative_humidity=max_relative_humidity),
        initial_temperature=initial_temperature,
        source_air_temperature=(initial_temperature if source_air_temperature is None else source_air_temperature),
        initial_dry_basis_moisture=_initial_moisture(
            equilibrium_relative_humidity,
            initial_temperature=initial_temperature,
        ),
        oswin_parameters=_OSWIN,
        pressure=float(_FIXED["p_ref"]),
    )


def _simple_canonical(values: np.ndarray) -> schedule_service.Schedule:
    return schedule_service.Schedule(
        values=values,
        metadata={
            "temperature_operational_bounds": [_FIXED["T_in_min"], _FIXED["T_in_max"]],
            "humidity_ratio_operational_bounds": [_FIXED["omega_min"], _FIXED["omega_max"]],
            "relative_humidity_operational_bounds": [
                _FIXED["phi_operational_min"],
                _FIXED["phi_operational_max"],
            ],
        },
    )


def test_schedule_replays_deterministically() -> None:
    """Reproduce the same scientific schedule from the same explicit seeds."""
    first = _schedule()
    second = _schedule()

    np.testing.assert_array_equal(first.values, second.values)


def test_canonical_schedule_rejects_relative_humidity_above_maintained_maximum() -> None:
    """Keep the startup-only ceiling out of canonical candidate acceptance."""
    temperature = 290.15
    humidity_ratio = float(
        schedule_service.relative_humidity_to_humidity_ratio(
            np.asarray([0.86], dtype=np.float64),
            np.asarray([temperature], dtype=np.float64),
            pressure=float(_FIXED["p_ref"]),
        )[0]
    )

    with pytest.raises(ValueError, match=r"continuous operating envelope \[0\.05, 0\.85\]"):
        _schedule(
            T_in_base=temperature,
            T_in_amp=0.0,
            omega_in_base=humidity_ratio,
            omega_in_amp=0.0,
            T_amb=temperature,
        )


def test_canonical_schedule_rejects_humidity_ratio_above_maintained_maximum() -> None:
    """Keep the canonical stochastic humidity-ratio engineering envelope."""
    accepted = _schedule(omega_in_base=0.0145, omega_in_amp=0.0)
    assert float(np.max(accepted.values[:, 2])) == 0.0145

    with pytest.raises(ValueError, match="source-air engineering envelope"):
        _schedule(omega_in_base=0.0146, omega_in_amp=0.0)


def test_startup_ceiling_does_not_change_canonical_realization() -> None:
    """Keep startup policy out of canonical acceptance and deterministic resampling."""
    canonical = _schedule()
    canonical_values = canonical.values.copy()
    canonical_diagnostics = canonical.diagnostics.copy()

    first = _build_handoff(canonical, max_relative_humidity=0.90)
    second = _build_handoff(canonical, max_relative_humidity=0.95)

    np.testing.assert_array_equal(canonical.values, canonical_values)
    assert canonical.diagnostics == canonical_diagnostics
    np.testing.assert_array_equal(first.values, second.values)
    assert canonical.metadata["relative_humidity_operational_bounds"] == [0.05, 0.85]
    assert first.metadata["boundary_handoff"]["startup_ramp"]["startup_relative_humidity_max"] == 0.90
    assert second.metadata["boundary_handoff"]["startup_ramp"]["startup_relative_humidity_max"] == 0.95


@pytest.mark.parametrize(
    ("initial_temperature", "equilibrium_minimum", "expected_target"),
    [
        (294.02023891487676, 0.9150037786093982, 0.8650037786093981),
        (288.72816040244123, 0.9075541364185941, 0.857554136418594),
    ],
)
def test_enabled_startup_allows_target_between_canonical_and_startup_maximum(
    initial_temperature: float,
    equilibrium_minimum: float,
    expected_target: float,
) -> None:
    """Admit smoke-equivalent startup RH above 0.85 without widening canonical RH."""
    equilibrium = np.asarray((equilibrium_minimum, 0.96, 0.97), dtype=np.float64)
    handoff = _build_handoff(
        _schedule(T_amb=initial_temperature),
        initial_temperature=initial_temperature,
        equilibrium_relative_humidity=equilibrium,
    )
    startup = handoff.metadata["boundary_handoff"]["startup_ramp"]

    assert startup["startup_target_relative_humidity"] == expected_target
    assert 0.85 < startup["startup_target_relative_humidity"] < startup["startup_relative_humidity_max"]
    assert startup["realized_minimum_initial_cell_to_inlet_rh_margin"] >= (0.05 - schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE)
    assert startup["continuous_startup_relative_humidity_maximum"] <= (
        startup["startup_target_relative_humidity"] + schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE
    )
    assert startup["startup_relative_humidity_max_basis"] == "synthetic_startup_design_bound"
    assert handoff.metadata["relative_humidity_operational_bounds"] == [0.05, 0.85]


def test_deterministic_startup_may_exceed_canonical_humidity_ratio_without_clipping() -> None:
    """Apply canonical omega bounds only after the deterministic startup row."""
    initial_temperature = 303.15
    equilibrium = np.asarray((0.85, 0.87, 0.89), dtype=np.float64)
    canonical = _schedule(T_amb=initial_temperature)
    canonical_values = canonical.values.copy()
    handoff = _build_handoff(
        canonical,
        initial_temperature=initial_temperature,
        equilibrium_relative_humidity=equilibrium,
    )
    startup = handoff.metadata["boundary_handoff"]["startup_ramp"]
    expected_target = float(np.min(equilibrium)) - 0.05
    expected_omega = float(
        schedule_service.relative_humidity_to_humidity_ratio(
            np.asarray([expected_target], dtype=np.float64),
            np.asarray([initial_temperature], dtype=np.float64),
            pressure=float(_FIXED["p_ref"]),
        )[0]
    )

    assert expected_omega > _FIXED["omega_max"]
    assert handoff.values[0, 2] == expected_omega
    assert startup["startup_humidity_ratio_kg_per_kg"] == expected_omega
    assert startup["startup_target_relative_humidity"] == pytest.approx(expected_target)
    assert startup["realized_minimum_initial_cell_to_inlet_rh_margin"] >= (0.05 - schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE)
    assert startup["continuous_startup_relative_humidity_maximum"] <= 0.90
    np.testing.assert_array_equal(canonical.values, canonical_values)
    np.testing.assert_array_equal(handoff.values[2:], canonical.values[1:])
    assert np.all(handoff.values[1:, 2] <= _FIXED["omega_max"])


def test_startup_maximum_is_selected_when_stricter_than_margin_upper_bound() -> None:
    """Treat the startup ceiling as a compatible stricter drying requirement."""
    equilibrium = np.asarray((0.9517694178004942, 0.97, 0.98), dtype=np.float64)

    handoff = _build_handoff(
        _schedule(),
        equilibrium_relative_humidity=equilibrium,
    )
    startup = handoff.metadata["boundary_handoff"]["startup_ramp"]

    assert startup["startup_target_relative_humidity"] == pytest.approx(0.90)
    assert startup["startup_target_relative_humidity"] <= float(np.min(equilibrium)) - 0.05
    assert startup["realized_minimum_initial_cell_to_inlet_rh_margin"] >= 0.05


@pytest.mark.parametrize(
    ("equilibrium_minimum", "startup_maximum", "expected_target"),
    [
        (0.80, 0.90, 0.75),
        (0.95, 0.90, 0.90),
    ],
    ids=("margin_upper_bound_is_stricter", "upper_bounds_are_equal"),
)
def test_initial_equilibrium_startup_selects_the_strictest_compatible_upper_bound(
    equilibrium_minimum: float,
    startup_maximum: float,
    expected_target: float,
) -> None:
    """Select the deterministic minimum of the margin and startup upper bounds."""
    state = schedule_service.derive_initial_equilibrium_startup(
        _initial_moisture(
            np.asarray((equilibrium_minimum, 0.97, 0.98), dtype=np.float64),
            initial_temperature=293.15,
        ),
        initial_temperature=293.15,
        source_air_temperature=293.15,
        oswin_parameters=_OSWIN,
        dry_margin=0.05,
        pressure=float(_FIXED["p_ref"]),
        startup_relative_humidity_maximum=startup_maximum,
    )

    assert state.target_relative_humidity == pytest.approx(expected_target)
    assert state.target_relative_humidity <= state.minimum - 0.05 + schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE
    assert state.realized_minimum_margin >= 0.05


def test_initial_equilibrium_startup_rejects_an_empty_physical_interval() -> None:
    """Fail only when the margin-derived feasible RH interval is genuinely empty."""
    with pytest.raises(ValueError, match="empty feasible RH interval"):
        schedule_service.derive_initial_equilibrium_startup(
            _initial_moisture(
                np.asarray((0.04, 0.50, 0.60), dtype=np.float64),
                initial_temperature=293.15,
            ),
            initial_temperature=293.15,
            source_air_temperature=293.15,
            oswin_parameters=_OSWIN,
            dry_margin=0.05,
            pressure=float(_FIXED["p_ref"]),
            startup_relative_humidity_maximum=0.90,
        )


def test_initial_equilibrium_startup_uses_inverse_oswin_and_global_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use every exact packed-bed cell and let the global minimum own the target."""
    equilibrium = np.asarray(((0.70, 0.90), (0.82, 0.78)), dtype=np.float64)
    moisture = _initial_moisture(equilibrium, initial_temperature=293.15)
    original = domain.moisture.oswin_equilibrium_relative_humidity
    calls = 0

    def tracked_inverse(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        domain.moisture,
        "oswin_equilibrium_relative_humidity",
        tracked_inverse,
    )
    state = schedule_service.derive_initial_equilibrium_startup(
        moisture,
        initial_temperature=293.15,
        source_air_temperature=293.15,
        oswin_parameters=_OSWIN,
        dry_margin=0.05,
        pressure=float(_FIXED["p_ref"]),
        startup_relative_humidity_maximum=0.90,
    )

    assert calls == 1
    assert state.minimum == pytest.approx(float(np.min(equilibrium)))
    assert state.mean == pytest.approx(float(np.mean(equilibrium)))
    assert state.maximum == pytest.approx(float(np.max(equilibrium)))
    assert state.target_relative_humidity == pytest.approx(0.65)
    assert state.realized_minimum_margin >= (0.05 - schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE)
    mean_based_target = float(np.mean(equilibrium)) - 0.05
    assert float(np.min(equilibrium)) - mean_based_target < 0.05


def test_homogeneous_initial_equilibrium_field_preserves_absolute_margin() -> None:
    """Preserve five absolute RH percentage points for every homogeneous cell."""
    equilibrium = np.full((2, 3), 0.81, dtype=np.float64)
    state = schedule_service.derive_initial_equilibrium_startup(
        _initial_moisture(equilibrium, initial_temperature=293.15),
        initial_temperature=293.15,
        source_air_temperature=293.15,
        oswin_parameters=_OSWIN,
        dry_margin=0.05,
        pressure=float(_FIXED["p_ref"]),
        startup_relative_humidity_maximum=0.90,
    )

    assert state.target_relative_humidity == pytest.approx(0.76)
    assert state.realized_minimum_margin == pytest.approx(0.05)


@pytest.mark.parametrize(
    "moisture",
    [
        np.asarray([], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
        np.asarray([-0.1], dtype=np.float64),
    ],
)
def test_initial_equilibrium_startup_rejects_invalid_fields(
    moisture: np.ndarray,
) -> None:
    """Fail closed for empty, non-finite, and physically invalid moisture fields."""
    with pytest.raises(ValueError, match=r"moisture|Modified-Oswin"):
        schedule_service.derive_initial_equilibrium_startup(
            moisture,
            initial_temperature=293.15,
            source_air_temperature=293.15,
            oswin_parameters=_OSWIN,
            dry_margin=0.05,
            pressure=float(_FIXED["p_ref"]),
            startup_relative_humidity_maximum=0.90,
        )


@pytest.mark.parametrize(
    (
        "initial_temperature",
        "equilibrium_values",
        "expected_target",
        "expected_omega",
    ),
    [
        (293.15, (0.825, 0.859, 0.872), 0.775, 0.011301991584801987),
        (288.37, (0.776, 0.827, 0.848), 0.726, 0.007788855627705122),
    ],
)
def test_diagnostic_startup_cases_match_scientific_references(
    initial_temperature: float,
    equilibrium_values: tuple[float, float, float],
    expected_target: float,
    expected_omega: float,
) -> None:
    """Match the two test-owned full-precision diagnostic startup references."""
    equilibrium = np.asarray(equilibrium_values, dtype=np.float64)
    state = schedule_service.derive_initial_equilibrium_startup(
        _initial_moisture(
            equilibrium,
            initial_temperature=initial_temperature,
        ),
        initial_temperature=initial_temperature,
        source_air_temperature=initial_temperature,
        oswin_parameters=_OSWIN,
        dry_margin=0.05,
        pressure=float(_FIXED["p_ref"]),
        startup_relative_humidity_maximum=0.90,
    )

    assert state.minimum == pytest.approx(float(np.min(equilibrium)))
    assert state.mean == pytest.approx(float(np.mean(equilibrium)))
    assert state.maximum == pytest.approx(float(np.max(equilibrium)))
    assert state.target_relative_humidity == pytest.approx(expected_target)
    assert state.startup_humidity_ratio == pytest.approx(expected_omega)
    derived = schedule_service.humidity_ratio_to_relative_humidity(
        np.asarray([state.startup_humidity_ratio]),
        np.asarray([initial_temperature]),
        pressure=float(_FIXED["p_ref"]),
    )
    assert derived[0] == pytest.approx(
        expected_target,
        abs=schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE,
    )


def test_comsol_startup_handoff_preserves_canonical_schedule_and_rejoins_exactly() -> None:
    """Ramp both primitives to an unchanged canonical rejoin and regular schedule."""
    canonical = _schedule()
    canonical_values = canonical.values.copy()
    duration_h = 0.5

    handoff = _build_handoff(canonical)

    np.testing.assert_array_equal(canonical.values, canonical_values)
    np.testing.assert_array_equal(handoff.values[:5, 0], [0.0, duration_h, 1.0, 2.0, 3.0])
    assert handoff.values[-1, 0] == 168.0
    assert handoff.values[0, 1] == 293.15
    assert handoff.values[0, 2] != canonical.values[0, 2]
    regular_interval = _TIME["interval"]
    assert isinstance(regular_interval, float)
    fraction = duration_h / regular_interval
    expected_rejoin = canonical.values[0] + fraction * (canonical.values[1] - canonical.values[0])
    expected_rejoin[0] = duration_h
    np.testing.assert_array_equal(handoff.values[1], expected_rejoin)
    np.testing.assert_array_equal(handoff.values[2:], canonical.values[1:])

    representative_times = np.asarray([duration_h, 1.0, 1.5, 17.25, 167.5])
    for column in (1, 2):
        original = np.interp(representative_times, canonical.values[:, 0], canonical.values[:, column])
        transformed = np.interp(representative_times, handoff.values[:, 0], handoff.values[:, column])
        np.testing.assert_allclose(transformed, original, rtol=0.0, atol=2.0e-15)


def test_comsol_startup_ramp_interpolates_temperature_and_humidity_ratio_linearly() -> None:
    """Derive nonlinear RH only after linearly interpolating both primitives."""
    canonical_values = np.asarray(
        (
            (0.0, 302.0, 0.008),
            (1.0, 310.0, 0.010),
            (2.0, 308.0, 0.009),
        ),
        dtype=np.float64,
    )
    canonical = _simple_canonical(canonical_values)
    handoff = _build_handoff(
        canonical,
        initial_temperature=298.0,
        equilibrium_relative_humidity=np.asarray(((0.70, 0.73), (0.75, 0.72))),
    )

    np.testing.assert_array_equal(handoff.values[:, 0], (0.0, 0.5, 1.0, 2.0))
    ramp_start = handoff.values[0]
    ramp_end = handoff.values[1]
    midpoint_temperature = np.interp(0.25, handoff.values[:, 0], handoff.values[:, 1])
    midpoint_humidity_ratio = np.interp(0.25, handoff.values[:, 0], handoff.values[:, 2])
    assert midpoint_temperature == ramp_start[1] + 0.5 * (ramp_end[1] - ramp_start[1])
    assert midpoint_humidity_ratio == ramp_start[2] + 0.5 * (ramp_end[2] - ramp_start[2])
    assert ramp_end[1] == 306.0
    assert ramp_end[2] == canonical_values[0, 2] + 0.5 * (canonical_values[1, 2] - canonical_values[0, 2])

    midpoint_phi = schedule_service.humidity_ratio_to_relative_humidity(
        np.asarray([midpoint_humidity_ratio]),
        np.asarray([midpoint_temperature]),
        pressure=float(_FIXED["p_ref"]),
    )[0]
    endpoint_phi = schedule_service.humidity_ratio_to_relative_humidity(
        np.asarray((ramp_start[2], ramp_end[2])),
        np.asarray((ramp_start[1], ramp_end[1])),
        pressure=float(_FIXED["p_ref"]),
    )
    independently_interpolated_phi = 0.5 * (endpoint_phi[0] + endpoint_phi[1])
    assert midpoint_phi != independently_interpolated_phi
    assert handoff.values.shape[1] == 3


def test_continuous_derived_relative_humidity_detects_interval_interior_extremum() -> None:
    """Evaluate the nonlinear RH extremum after linear primitive interpolation."""
    temperature = np.asarray((306.0, 296.0), dtype=np.float64)
    humidity_ratio = np.asarray((0.011, 0.006), dtype=np.float64)

    minimum, maximum = schedule_service.derived_relative_humidity_extrema(
        temperature,
        humidity_ratio,
        pressure=float(_FIXED["p_ref"]),
    )
    endpoint_phi = schedule_service.humidity_ratio_to_relative_humidity(
        humidity_ratio,
        temperature,
        pressure=float(_FIXED["p_ref"]),
    )

    assert minimum == pytest.approx(float(np.min(endpoint_phi)))
    assert maximum > float(np.max(endpoint_phi))
    assert maximum == pytest.approx(0.3652169067)


def test_disabled_comsol_startup_handoff_is_semantically_canonical() -> None:
    """Leave canonical schedule values unchanged when startup is disabled."""
    canonical = _schedule()
    handoff = _build_handoff(canonical, enabled=False)

    np.testing.assert_array_equal(handoff.values, canonical.values)


def test_comsol_startup_handoff_rejects_invalid_physical_state() -> None:
    """Reject invalid primitive startup temperature before field derivation."""
    canonical = _schedule()
    with pytest.raises(ValueError, match="physically positive"):
        schedule_service.build_comsol_boundary_schedule(
            canonical,
            _policy(),
            initial_temperature=-1.0,
            source_air_temperature=293.15,
            initial_dry_basis_moisture=np.asarray([0.2]),
            oswin_parameters=_OSWIN,
            pressure=float(_FIXED["p_ref"]),
        )


def test_comsol_startup_handoff_uses_cold_initial_equilibrium_state() -> None:
    """Begin at exact bed temperature and five RH points below the driest cell."""
    initial_temperature = 288.7
    canonical = _schedule(
        T_in_base=308.15,
        T_in_amp=0.0,
        omega_in_base=0.010625,
        omega_in_amp=0.0,
        T_amb=initial_temperature,
    )
    equilibrium = np.asarray(((0.825, 0.852), (0.841, 0.872)))
    handoff = _build_handoff(
        canonical,
        initial_temperature=initial_temperature,
        equilibrium_relative_humidity=equilibrium,
    )

    startup = handoff.metadata["boundary_handoff"]["startup_ramp"]
    expected_target = float(np.min(equilibrium)) - 0.05
    expected_omega = schedule_service.relative_humidity_to_humidity_ratio(
        np.asarray([expected_target]),
        np.asarray([initial_temperature]),
        pressure=float(_FIXED["p_ref"]),
    )[0]
    assert initial_temperature < _FIXED["T_in_min"]
    assert handoff.values[0, 1] == initial_temperature
    assert handoff.values[0, 2] == expected_omega
    assert startup["policy_id"] == schedule_service.STARTUP_POLICY_ID
    assert startup["initial_equilibrium_rh_minimum"] == pytest.approx(float(np.min(equilibrium)))
    assert startup["initial_equilibrium_rh_mean"] == pytest.approx(float(np.mean(equilibrium)))
    assert startup["initial_equilibrium_rh_maximum"] == pytest.approx(float(np.max(equilibrium)))
    assert startup["initial_equilibrium_rh_dry_margin"] == 0.05
    assert startup["startup_relative_humidity_max"] == 0.90
    assert startup["startup_target_relative_humidity"] == pytest.approx(expected_target)
    assert startup["startup_humidity_ratio_kg_per_kg"] == expected_omega
    assert startup["realized_minimum_initial_cell_to_inlet_rh_margin"] >= (0.05 - schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE)
    assert startup["continuous_startup_relative_humidity_maximum"] <= (expected_target + schedule_service.STARTUP_RELATIVE_HUMIDITY_TOLERANCE)
    assert handoff.metadata["boundary_handoff"]["handoff_version"] == 1


def test_comsol_startup_handoff_rejects_supersaturated_source_air() -> None:
    """Retain source-air thermodynamic feasibility above the canonical omega ceiling."""
    canonical = _simple_canonical(
        np.asarray(
            (
                (0.0, 305.0, 0.008),
                (1.0, 306.0, 0.008),
                (2.0, 307.0, 0.008),
            ),
            dtype=np.float64,
        )
    )

    with pytest.raises(ValueError, match="source-air temperature"):
        _build_handoff(
            canonical,
            initial_temperature=303.15,
            source_air_temperature=293.15,
            equilibrium_relative_humidity=np.asarray((0.85, 0.87, 0.89), dtype=np.float64),
        )


def test_comsol_startup_handoff_rejects_source_air_hotter_than_inlet() -> None:
    """Fail closed instead of violating the heater-only source-air contract."""
    with pytest.raises(ValueError, match="heater-only"):
        _build_handoff(
            _schedule(),
            initial_temperature=293.15,
            source_air_temperature=294.15,
        )


def test_comsol_startup_handoff_rejects_invalid_canonical_rejoin_rh() -> None:
    """Reject an unchanged canonical rejoin that exceeds the fixed startup target."""
    canonical = _simple_canonical(
        np.asarray(
            (
                (0.0, 295.0, 0.008),
                (1.0, 295.0, 0.008),
                (2.0, 296.0, 0.007),
            ),
            dtype=np.float64,
        )
    )
    equilibrium = np.asarray(((0.45, 0.55), (0.60, 0.52)), dtype=np.float64)

    with pytest.raises(ValueError, match="Canonical startup rejoin"):
        _build_handoff(
            canonical,
            initial_temperature=293.15,
            equilibrium_relative_humidity=equilibrium,
        )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        (1, 289.15),
        (2, 0.015),
    ],
)
def test_comsol_handoff_rejects_operational_rejoin_violation(
    column: int,
    invalid_value: float,
) -> None:
    """Keep authored operational envelopes on the rejoin and regular nodes."""
    canonical = _schedule()
    handoff = _build_handoff(
        canonical,
        initial_temperature=288.15,
        equilibrium_relative_humidity=np.asarray(((0.825, 0.84), (0.85, 0.83))),
    )
    invalid = handoff.values.copy()
    invalid[1, column] = invalid_value

    with pytest.raises(ValueError, match=r"invalid|constraints"):
        schedule_service.validate_comsol_boundary_schedule(
            invalid,
            regular_times=np.asarray(_TIME["regular_times"], dtype=np.float64),
            startup_ramp=_policy(),
            initial_temperature=288.15,
            source_air_temperature=288.15,
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
    supports: dict[str, dict[str, Any]] = {
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
