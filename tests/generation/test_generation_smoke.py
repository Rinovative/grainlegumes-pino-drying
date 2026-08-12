# ruff: noqa: S101, SLF001
"""Bounded mapping-probe and real-smoke diagnostic contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src import generation
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.runtime import generation_runtime_mapping_probe as mapping_probe


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
        (3, len(profiles.GLOBAL_FIELD_NAMES)),
        dtype=np.float64,
    )
    columns = {name: index for index, name in enumerate(profiles.GLOBAL_FIELD_NAMES)}
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
        mapping_probe.common.paths,
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
    raw = mapping_probe._profile_mapping(profile)

    table = tmp_path / "observed.csv"
    table.write_text("x,y,p\n0,1,2\n3,4,5\n", encoding="utf-8")
    observation = mapping_probe._table_observation(table)
    assert observation == {
        "delimiter": ",",
        "header": ["x", "y", "p"],
        "shape": [2, 3],
        "rectangular": True,
        "time_header_candidates": [],
    }
    comparison = mapping_probe._mapping_comparison(
        raw,
        [{"relative_path": "observed.csv", "table": observation}],
        profile_path=profile,
    )
    assert comparison["required_corrections"] == ["profile.yaml:exports[0].columns.x"]
    assert comparison["optional_corrections"] == []
    assert comparison["required_missing_exports"] == []
    assert comparison["optional_missing_exports"] == []
    assert comparison["aliases_used"] is False


def test_mapping_probe_distinguishes_missing_mismatched_and_confirmed_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report export execution failure before making any per-column claim."""
    monkeypatch.setattr(mapping_probe.common.paths, "get_project_root", lambda: tmp_path)
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
      state: runtime_confirmed
      source_header: x
""",
        encoding="utf-8",
    )
    raw = mapping_probe._profile_mapping(profile)

    missing = mapping_probe._mapping_comparison(raw, [], profile_path=profile)
    assert missing["required_missing_exports"] == [{"role": "steady_flow_fields", "declared_pattern": "observed.csv"}]
    assert missing["required_corrections"] == []
    assert (
        mapping_probe._mapping_probe_status(
            missing,
            exit_code=0,
            timed_out=False,
            start_error=None,
        )
        == "required_export_missing"
    )

    mismatched = mapping_probe._mapping_comparison(
        raw,
        [
            {
                "relative_path": "exports/observed.csv",
                "table": {"delimiter": ",", "header": ["wrong"]},
            }
        ],
        profile_path=profile,
    )
    assert mismatched["required_missing_exports"] == []
    assert mismatched["required_corrections"] == ["profile.yaml:exports[0].columns.x"]
    assert (
        mapping_probe._mapping_probe_status(
            mismatched,
            exit_code=0,
            timed_out=False,
            start_error=None,
        )
        == "mapping_update_required"
    )

    confirmed = mapping_probe._mapping_comparison(
        raw,
        [
            {
                "relative_path": "exports/observed.csv",
                "table": {"delimiter": ",", "header": ["x"]},
            }
        ],
        profile_path=profile,
    )
    assert confirmed["required_missing_exports"] == []
    assert confirmed["required_corrections"] == []
    assert (
        mapping_probe._mapping_probe_status(
            confirmed,
            exit_code=0,
            timed_out=False,
            start_error=None,
        )
        == "mapping_observation_complete"
    )


def test_fake_mapping_probe_uses_canonical_retained_command(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove mapping probes consume the shared retained-diagnostic builder path."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        retain_solved_model=True,
    )
    authored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_root = mapping_probe.common.paths.get_project_root().resolve()
    authored["profile_config"] = (config_path.parent / "profile.yaml").relative_to(project_root).as_posix()
    config_path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    report_path = mapping_probe.run_mapping_probe(
        config_path,
        storage_root=tmp_path / "mapping storage",
        work_root=tmp_path / "mapping work",
        cores_per_case=16,
    )
    report = mapping_probe.load_mapping_probe(report_path)
    command = report["command"]

    assert command[command.index("-inputfile") + 1] == "model.mph"
    assert command[command.index("-job") + 1] == "s1"
    assert command[command.index("-outputfile") + 1] == "solved.mph"
    assert command[command.index("-np") + 1] == "16"
    assert "-nosave" not in command


def test_real_smoke_comsol_contract_comes_from_paired_execution_config() -> None:
    """Bind receipt version evidence to configured paired execution contracts."""
    root = Path("configs/generation/campaigns")
    steady = generation.cases.config.load_campaign_config(
        root / "steady_flow/technical_smoke.yaml",
        require_executable=False,
    )
    transient = generation.cases.config.load_campaign_config(
        root / "transient_drying/technical_smoke.yaml",
        require_executable=False,
    )
    contract = generation.smoke._paired_comsol_contract((steady, transient))
    assert contract["module"] == steady.execution_values["site"]["comsol_module"]
    assert contract["executable"] == steady.execution_values["site"]["comsol_executable"]
    assert contract["required_version"] in contract["module"]

    changed_execution = copy.deepcopy(transient.execution_values)
    changed_execution["site"]["comsol_module"] = f"{contract['module']}.different"
    changed = replace(transient, execution_values=changed_execution)
    with pytest.raises(RuntimeError, match="must agree"):
        generation.smoke._paired_comsol_contract((steady, changed))
