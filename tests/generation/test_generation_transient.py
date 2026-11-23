"""Current transient time and post-horizon physical admission contracts.

Responsibilities:
  - Check regular/exact-stop time admission and one complete post-horizon exclusion
  - Check that a negative stored water state fails the conversion with its recorded undershoot numbers

Design principles:
  - Raw globals and final status must agree before canonical terminal projection

This module does NOT:
  - Interpolate time axes or change solver stopping controls
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from src import generation
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("times", "fragment"),
    [
        (np.asarray([0.0, 1.0, 3.0]), "missing_expected_times_h"),
        (np.asarray([0.0, 1.0, 1.0]), "duplicate_times_h"),
        (np.asarray([0.0, np.nan, 1.0]), "nonfinite_time_count"),
        (np.asarray([0.0, 2.0, 1.0]), "nonmonotonic_pairs_h"),
        (np.asarray([0.0, 0.5, 1.0]), "unexpected_inside_horizon_times_h"),
    ],
)
def test_transient_time_classification_rejects_malformed_axes_with_bounded_evidence(
    times: np.ndarray,
    fragment: str,
) -> None:
    """Keep malformed raw transient axes closed with actionable bounded evidence."""
    contract = {"start": 0.0, "stop": 2.0, "interval": 1.0, "regular_times": [0.0, 1.0, 2.0]}

    with pytest.raises(ValueError, match=fragment) as error:
        generation.publication.storage._classify_transient_times(times, contract)

    message = str(error.value)
    assert "expected_start_h" in message
    assert "expected_stop_h" in message
    assert "regular_state_count" in message
    assert "minimum_exported_time_h" in message
    assert "maximum_exported_time_h" in message


def test_transient_time_classification_preserves_regular_and_exact_stop_contracts() -> None:
    """Keep complete regular schedules and final in-horizon exact stops canonical."""
    contract = {"start": 0.0, "stop": 2.0, "interval": 1.0, "regular_times": [0.0, 1.0, 2.0]}

    regular, positions, exact_position, post_position = generation.publication.storage._classify_transient_times(
        np.asarray([0.0, 1.0, 2.0]),
        contract,
    )
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(positions, [0, 1, 2])
    assert exact_position is None
    assert post_position is None

    regular, positions, exact_position, post_position = generation.publication.storage._classify_transient_times(
        np.asarray([0.0, 1.0, 1.5]),
        contract,
    )
    np.testing.assert_array_equal(regular, [0.0, 1.0])
    np.testing.assert_array_equal(positions, [0, 1])
    assert exact_position == 2
    assert post_position is None


def test_transient_time_classification_excludes_only_one_final_complete_schedule_overshoot() -> None:
    """Admit one final post-horizon solver state only after the entire schedule exists."""
    contract = {"start": 0.0, "stop": 2.0, "interval": 1.0, "regular_times": [0.0, 1.0, 2.0]}
    regular, positions, exact_position, post_position = generation.publication.storage._classify_transient_times(
        np.asarray([0.0, 1.0, 2.0, 2.25]),
        contract,
    )
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(positions, [0, 1, 2])
    assert exact_position is None
    assert post_position == 3

    with pytest.raises(ValueError, match="post_horizon_times_h"):
        generation.publication.storage._classify_transient_times(
            np.asarray([0.0, 1.0, 2.25]),
            contract,
        )


def test_post_horizon_state_is_excluded_from_canonical_status_and_final_evidence() -> None:
    """Keep a post-horizon target crossing out of the admitted t-max outcome."""
    contract = {"start": 0.0, "stop": 2.0, "interval": 1.0, "regular_times": [0.0, 1.0, 2.0]}
    config = cast("Any", SimpleNamespace(scientific_values={"time": contract, "scientific_fixed_values": {"f_wet_dm_max": 0.05}}))
    regular_time = np.asarray([0.0, 1.0, 2.0])
    regular_fields = np.asarray(
        [
            [[[300.0]], [[0.4]], [[10.0]], [[10.0]]],
            [[[301.0]], [[0.4]], [[9.0]], [[9.0]]],
            [[[302.0]], [[0.4]], [[8.0]], [[8.0]]],
        ]
    )
    static_fields = np.zeros((len(profiles.TRANSIENT_STATIC_FIELD_NAMES), 1, 1))
    static_fields[profiles.TRANSIENT_STATIC_FIELD_NAMES.index("rho_bu_dry"), 0, 0] = 100.0
    global_values = np.zeros((4, len(profiles.GLOBAL_FIELD_NAMES)))
    global_values[:, 0] = [0.0, 1.0, 2.0, 2.25]
    global_values[:, profiles.GLOBAL_FIELD_NAMES.index("X_wb_bulk")] = [0.2, 0.18, 0.16, 0.02]
    global_values[:, profiles.GLOBAL_FIELD_NAMES.index("f_wet_dm")] = [1.0, 0.8, 0.08, 0.01]
    post_horizon_fields = np.asarray([[[299.0]], [[0.3]], [[5.0]], [[5.0]]])
    raw_final_values = {
        "t_final": 2.25,
        "f_wet_dm_final": 0.01,
        "X_wb_bulk_final": 0.02,
        "X_wb_max_final": 5.0 / 105.0,
        "T_min_final": 299.0,
        "T_max_final": 299.0,
        "phi_min_final": 0.3,
        "phi_max_final": 0.3,
    }
    final_status = np.asarray([[raw_final_values[name] for name in profiles.FINAL_STATUS_FIELDS]])
    canonical_global, canonical_final = generation.publication.storage._admit_transient_output_horizon(
        config,
        regular_time,
        static_fields,
        regular_fields,
        None,
        None,
        2.25,
        post_horizon_fields,
        global_values,
        final_status,
        f_surf=0.4,
    )
    status = generation.publication.storage._status(
        config,
        transient_time=regular_time,
        exact_stop_time=None,
        post_horizon_time=2.25,
        final_status=canonical_final,
        global_values=canonical_global,
        runtime_seconds=1.0,
        diagnostics=[],
    )

    np.testing.assert_array_equal(canonical_global[:, 0], [0.0, 1.0, 2.0])
    assert canonical_final[0, profiles.FINAL_STATUS_FIELDS.index("t_final")] == 2.0
    assert status["hit_t_max"] is True
    assert status["target_reached"] is False
    assert status["t_stop_exact"] == 2.0
    assert status["has_exact_stop_state"] is False
    assert status["post_horizon_export_state_ignored"] is True
    assert status["ignored_post_horizon_state_time"] == 2.25


def test_post_horizon_state_requires_raw_global_field_and_final_status_consistency() -> None:
    """Reject a discarded state whose raw Final Status conflicts with raw evidence."""
    contract = {"start": 0.0, "stop": 2.0, "interval": 1.0, "regular_times": [0.0, 1.0, 2.0]}
    config = cast("Any", SimpleNamespace(scientific_values={"time": contract}))
    regular_time = np.asarray([0.0, 1.0, 2.0])
    regular_fields = np.ones((3, len(profiles.TRANSIENT_FIELD_NAMES), 1, 1))
    post_horizon_fields = np.ones((len(profiles.TRANSIENT_FIELD_NAMES), 1, 1))
    static_fields = np.ones((len(profiles.TRANSIENT_STATIC_FIELD_NAMES), 1, 1))
    global_values = np.zeros((4, len(profiles.GLOBAL_FIELD_NAMES)))
    global_values[:, 0] = [0.0, 1.0, 2.0, 2.25]
    raw_final_values = {
        "t_final": 2.25,
        "f_wet_dm_final": 999.0,
        "X_wb_bulk_final": 0.0,
        "X_wb_max_final": 0.5,
        "T_min_final": 1.0,
        "T_max_final": 1.0,
        "phi_min_final": 1.0,
        "phi_max_final": 1.0,
    }
    final_status = np.asarray([[raw_final_values[name] for name in profiles.FINAL_STATUS_FIELDS]])

    with pytest.raises(ValueError, match="raw_final_status_conflicts_with_raw_post_horizon_state:f_wet_dm_final"):
        generation.publication.storage._admit_transient_output_horizon(
            config,
            regular_time,
            static_fields,
            regular_fields,
            None,
            None,
            2.25,
            post_horizon_fields,
            global_values,
            final_status,
            f_surf=0.4,
        )


def _water_states(w_surf: list[list[float]], w_int: list[list[float]]) -> np.ndarray:
    """Return stored transient states on a 1 x 2 grid with the given water values per state and constant T and phi."""
    states = np.empty((len(w_surf), len(profiles.TRANSIENT_FIELD_NAMES), 1, 2))
    states[:, profiles.TRANSIENT_FIELD_NAMES.index("T")] = 300.0
    states[:, profiles.TRANSIENT_FIELD_NAMES.index("phi")] = 0.5
    states[:, profiles.TRANSIENT_FIELD_NAMES.index("w_surf"), 0] = w_surf
    states[:, profiles.TRANSIENT_FIELD_NAMES.index("w_int"), 0] = w_int
    return states


_INTERMEDIATE_NEGATIVE = (
    np.asarray([0.0, 1.0, 2.0, 3.0]),
    _water_states([[10.0, 10.0], [-0.5, 2.0], [-0.75, -0.1], [1.0, 1.0]], [[10.0, 10.0], [3.0, -0.25], [1.0, 1.0], [0.5, 1.0]]),
    None,
    None,
    None,
    None,
)
_FINAL_NEGATIVE = (
    np.asarray([0.0, 1.0]),
    _water_states([[10.0, 10.0], [5.0, 5.0]], [[10.0, 10.0], [5.0, 5.0]]),
    1.5,
    _water_states([[0.2, 0.3]], [[-0.01, 0.4]])[0],
    None,
    None,
)
_POST_HORIZON_NEGATIVE = (
    np.asarray([0.0, 1.0, 2.0]),
    _water_states([[10.0, 10.0], [5.0, 5.0], [1.0, 1.0]], [[10.0, 10.0], [5.0, 5.0], [1.0, 1.0]]),
    None,
    None,
    2.25,
    _water_states([[0.1, 0.3]], [[-0.3, 0.4]])[0],
)
_NON_NEGATIVE = (
    np.asarray([0.0, 1.0]),
    _water_states([[10.0, 10.0], [0.0, 5.0]], [[10.0, 10.0], [5.0, 0.0]]),
    None,
    None,
    2.25,
    _water_states([[0.0, 1.0]], [[1.0, 0.0]])[0],
)


@pytest.mark.parametrize(
    ("transient", "expected"),
    [
        (
            _INTERMEDIATE_NEGATIVE,
            {
                "stored_state_count": 4,
                "first_negative_time_h": 1.0,
                "last_negative_time_h": 2.0,
                "minimum_time_h": 2.0,
                "post_horizon_state_checked": False,
                "w_surf": {"unit": "kg/m^3", "minimum": -0.75, "negative_value_count": 3, "negative_state_count": 2},
                "w_int": {"unit": "kg/m^3", "minimum": -0.25, "negative_value_count": 1, "negative_state_count": 1},
            },
        ),
        (
            _FINAL_NEGATIVE,
            {
                "stored_state_count": 3,
                "first_negative_time_h": 1.5,
                "last_negative_time_h": 1.5,
                "minimum_time_h": 1.5,
                "post_horizon_state_checked": False,
                "w_surf": {"unit": "kg/m^3", "minimum": 0.2, "negative_value_count": 0, "negative_state_count": 0},
                "w_int": {"unit": "kg/m^3", "minimum": -0.01, "negative_value_count": 1, "negative_state_count": 1},
            },
        ),
        (
            _POST_HORIZON_NEGATIVE,
            {
                "stored_state_count": 3,
                "first_negative_time_h": 2.25,
                "last_negative_time_h": 2.25,
                "minimum_time_h": 2.25,
                "post_horizon_state_checked": True,
                "w_surf": {"unit": "kg/m^3", "minimum": 0.1, "negative_value_count": 0, "negative_state_count": 0},
                "w_int": {"unit": "kg/m^3", "minimum": -0.3, "negative_value_count": 1, "negative_state_count": 1},
            },
        ),
        (_NON_NEGATIVE, None),
    ],
    ids=["intermediate_state", "exact_stop_state", "post_horizon_state", "non_negative"],
)
def test_conversion_rejects_any_negative_stored_water_state_with_its_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, transient: tuple[Any, ...], expected: dict[str, Any] | None
) -> None:
    """A negative w_surf or w_int value in any stored or post-horizon state fails the conversion unclipped; non-negative states pass on."""
    storage = generation.publication.storage
    profile = SimpleNamespace(id=profiles.TRANSIENT_DRYING_PROFILE)
    config = cast("Any", SimpleNamespace(profile=profile, scientific_values={"scientific_fixed_values": {}}))
    scalar_values = np.full(len(profiles.scalar_input_fields(profiles.TRANSIENT_DRYING_PROFILE)), 0.4)
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0]))
    fields = [array for array in transient if isinstance(array, np.ndarray) and array.ndim > 1]
    stored = [array.copy() for array in fields]

    def passed(*_args: Any, **_kwargs: Any) -> Any:
        message = "non-negative states reached the schedule adapter"
        raise RuntimeError(message)

    monkeypatch.setattr(storage, "_static_fields", lambda *_args: (*axes, np.ones((1, 1, 2))))
    monkeypatch.setattr(storage, "_stationary_fixed_values", lambda *_args: np.zeros(0))
    monkeypatch.setattr(storage.scalar_handoff_contract, "admit_case_scalar_handoff", lambda *_args: SimpleNamespace(values=scalar_values))
    monkeypatch.setattr(storage, "_transient_fields", lambda *_args, **_kwargs: transient)
    monkeypatch.setattr(storage, "_schedule_values", passed)

    def convert() -> None:
        storage.convert_exports_to_hdf5(config, {}, (), work_directory=tmp_path, runtime_directory=tmp_path, runtime_seconds=1.0)

    if expected is None:
        with pytest.raises(RuntimeError, match="reached the schedule adapter"):
            convert()
        return
    with pytest.raises(storage.NegativeWaterStateError) as error:
        convert()
    assert (error.value.reason_code, error.value.evidence) == ("negative_water_state", expected)
    assert str(expected["w_int"]["minimum"]) in str(error.value)
    assert (f"{expected['stored_state_count']} stored states and the post-horizon state at t=2.25 h" in str(error.value)) is expected[
        "post_horizon_state_checked"
    ]
    for array, original in zip(fields, stored, strict=True):
        np.testing.assert_array_equal(array, original)  # never clipped
