# ruff: noqa: S101, SLF001
"""Bounded mapping-probe and real-smoke diagnostic contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src import generation

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_observed_difference_and_mass_balance_metrics_have_no_invented_tolerance() -> None:
    """Report exact differences and a closed synthetic water balance."""
    left = np.zeros((2, 2), dtype=np.float64)
    right = np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    difference = generation.smoke._difference_metrics(
        left,
        right,
        x_axis=np.asarray([0.0, 0.5]),
        y_axis=np.asarray([0.0, 0.25]),
    )
    assert difference == {
        "maximum_absolute_difference": 2.0,
        "maximum_relative_difference": 1.0,
        "rmse": np.sqrt(5.0 / 4.0),
        "l2_difference": np.sqrt(5.0),
        "maximum_difference_location": {
            "x_index": 0,
            "y_index": 1,
            "x_m": 0.0,
            "y_m": 0.25,
        },
        "acceptance_tolerance": None,
    }

    global_values = np.zeros(
        (3, len(generation.profiles.GLOBAL_FIELD_NAMES)),
        dtype=np.float64,
    )
    columns = {name: index for index, name in enumerate(generation.profiles.GLOBAL_FIELD_NAMES)}
    global_values[:, columns["t"]] = [0.0, 1.0, 2.0]
    global_values[:, columns["m_w_gr"]] = 10.0
    global_values[:, columns["m_v_gas"]] = 1.0
    global_values[:, columns["m_dot_evap"]] = 0.1
    global_values[:, columns["m_dot_v_in"]] = 0.2
    global_values[:, columns["m_dot_v_out"]] = 0.2
    density = np.full((2, 2), 500.0)
    moisture_db = np.full((2, 2), 0.2)
    water = density * moisture_db
    case = generation.smoke._CaseEvidence(
        record={"case_id": "case_0001", "simulation_case_id": "a" * 64},
        static={"rho_bu_dry": density, "X_0_db_field": moisture_db},
        stationary_fixed={},
        scalars={"f_surf": 0.25},
        schedule=None,
        global_values=global_values,
        initial_state={"w_surf": water, "w_int": water},
    )
    balance = generation.smoke._mass_balance_case(case)
    assert balance["differential_total_water_residual"]["maximum_absolute"] == 0.0
    assert balance["integral_total_water_closure"]["maximum_absolute"] == 0.0
    assert balance["initial_reconstruction"]["maximum_absolute_X_db_error"] == 0.0
    assert balance["initial_reconstruction"]["maximum_absolute_X_wb_error"] == 0.0
    assert balance["acceptance_tolerance"] is None


def test_mapping_probe_reports_exact_unconfirmed_keys_and_actual_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep mapping observation explicit and prohibit silent mapping inference."""
    monkeypatch.setattr(
        generation.mapping_probe.common.paths,
        "get_project_root",
        lambda: tmp_path,
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        """simulation_profile: steady_flow
steady_flow_conditioning: null
exports:
- role: steady_flow_fields
  temporal_kind: stationary
  source:
    state: runtime_confirmed
    pattern: observed.csv
  delimiter: ','
  columns:
    x:
      state: mapping_probe_required
    y:
      state: declared_unverified
      source_header: y
""",
        encoding="utf-8",
    )
    raw = generation.mapping_probe._profile_mapping(profile)

    table = tmp_path / "observed.csv"
    table.write_text("x,y,p\n0,1,2\n3,4,5\n", encoding="utf-8")
    observation = generation.mapping_probe._table_observation(table)
    assert observation == {
        "delimiter": ",",
        "header": ["x", "y", "p"],
        "shape": [2, 3],
        "rectangular": True,
        "time_header_candidates": [],
    }
    comparison = generation.mapping_probe._mapping_comparison(
        raw,
        [{"relative_path": "observed.csv", "table": observation}],
        profile_path=profile,
    )
    assert comparison["required_corrections"] == ["profile.yaml:exports[0].columns.x"]
    assert comparison["optional_corrections"] == []
    assert comparison["aliases_used"] is False
