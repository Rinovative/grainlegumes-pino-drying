# ruff: noqa: S101, PLR2004, ARG005
"""Six-material transient pilot planning, diagnostics, evidence, and cleanup."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_input as input_service
from src.generation.cases import generation_cases_sampling as sampling_service
from src.generation.publication import generation_publication_attempt as attempt_service
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_workspace as workspace_service
from src.generation.validation import generation_validation_pilot as pilot_service
from src.generation.validation import generation_validation_pilot_analysis as analysis_service

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def pilot_campaign_path(generation_config_factory: Any) -> Path:
    """Return one compact test-owned transient pilot campaign."""
    path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=3,
        campaign_purpose="family_generalization",
    )
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    campaign["campaign_purpose"] = "pilot_check"
    campaign.pop("membership")
    execution_path = path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["retention"]["pilot_check"] = "full"
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    campaign["sampling"] = {
        "method": "lhs",
        "seed_base": 9940,
        "cases_per_material": 3,
        "case_semantics": {
            "first": "nominal_reference",
            "remaining": "natural_pilot",
        },
    }
    campaign["dataset_packages"] = []
    path.write_text(
        yaml.safe_dump(campaign, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _pilot(campaign_path: Path) -> config_service.CampaignConfig:
    """Load the YAML-owned three-case-per-material pilot contract."""
    return config_service.load_campaign_config(campaign_path)


def _canonical_case_result(
    *,
    result_class: str = "PASS",
    failed_stage: str | None = None,
    solver_status: str = "success",
) -> dict[str, Any]:
    target_reached = None if failed_stage is not None else True
    drying_time_h = None if target_reached is not True else 72.0
    return {
        "material": "lentil",
        "material_role": "seen",
        "case_kind": "nominal_reference",
        "case_index": 1,
        "case_id": "case_00001",
        "solver_status": solver_status,
        "result_class": result_class,
        "target_reached": target_reached,
        "drying_time_h": drying_time_h,
        "drying_time_days": None if drying_time_h is None else drying_time_h / 24.0,
        "last_valid_time_h": None if failed_stage is not None else 72.0,
        "stop_reason": "runtime_failure" if failed_stage is not None else "target_stop",
        "failed_stage": failed_stage,
        "warning_count": 0,
        "final_X_wb_bulk": None if failed_stage is not None else 0.08,
        "final_f_wet_dm": None if failed_stage is not None else 0.04,
        "warnings": [],
    }


def _physical_inputs() -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], dict[str, list[float]], dict[str, float]]:
    shape = (2, 2)
    static = {
        "Kxx": np.full(shape, 2.0e-10),
        "Kxy": np.zeros(shape),
        "Kyy": np.full(shape, 2.0e-10),
        "eps_bed": np.full(shape, 0.4),
        "rho_bu_dry": np.full(shape, 700.0),
        "p": np.asarray([[10.0, 9.0], [2.0, 1.0]]),
        "u": np.full(shape, 0.1),
        "v": np.zeros(shape),
    }
    states = [
        {
            "T": np.full(shape, 300.0),
            "phi": np.full(shape, 0.4),
            "w_surf": np.full(shape, 100.0),
            "w_int": np.full(shape, 100.0),
        },
        {
            "T": np.full(shape, 305.0),
            "phi": np.full(shape, 0.3),
            "w_surf": np.full(shape, 90.0),
            "w_int": np.full(shape, 95.0),
        },
    ]
    globals_by_name = {
        "f_wet_dm": [1.0, 0.04],
        "m_dot_evap": [0.01, 0.005],
        "X_wb_bulk": [0.125, 0.115],
        "T_out_mean": [299.0, 303.0],
        "phi_out_mean": [0.4, 0.3],
        "m_dot_v_in": [0.001, 0.001],
        "m_dot_v_out": [0.01, 0.006],
    }
    scalars = {
        "eps_min_global": 0.25,
        "eps_max_global": 0.65,
        "f_surf": 0.3,
        "r_surf_0": 1.0e-5,
        "A_osw": 10.0,
        "B_osw": 0.1,
        "C_osw": 0.5,
        "X_target_wb": 0.06,
    }
    return static, states, globals_by_name, scalars


def test_pilot_planning_and_nominal_sampling_are_technical_only(
    pilot_campaign_path: Path,
) -> None:
    """Protect configured-family counts, technical-only membership, and explicit nominals."""
    campaign = _pilot(pilot_campaign_path)
    assert campaign.total_case_count == 3 * len(campaign.material_inventory)
    assert campaign.dataset_packages == ()
    assert campaign.campaign_purpose == "pilot_check"
    assert campaign.evaluation_regimes == ()
    assert campaign.membership == {}
    assert all(not members for members in campaign.material_memberships.values())
    assert tuple(batch.material_family for batch in campaign.batches) == campaign.material_inventory
    for batch in campaign.batches:
        assert [batch.case_assignment(index)["pilot_case_kind"] for index in batch.case_indices] == [
            "nominal_reference",
            "natural_pilot",
            "natural_pilot",
        ]
        assert all(batch.case_assignment(index)["ood_group"] is None for index in batch.case_indices)
    batch = campaign.batches[0]
    nominal = sampling_service.sample_case(batch, 1)
    natural = sampling_service.sample_case(batch, 2)
    registry = batch.scientific_values["material"]["parameter_registry"]
    assert nominal.values["T_in_base"] == registry["T_in_base"]["nominal"]
    assert nominal.values["schedule.component_weights"] == registry["schedule.component_weights"]["nominal"]
    assert nominal.ood_provenance["seed_search"] is False
    assert all(record["sampling_kind"] == "explicit_configured_nominal" for record in nominal.block_provenance.values())
    assert "sampling_kind" not in next(iter(natural.block_provenance.values()))


def test_pilot_nominal_fails_closed_without_an_explicit_value(
    pilot_campaign_path: Path,
) -> None:
    """Protect fail-closed nominal construction without hidden midpoints."""
    batch = _pilot(pilot_campaign_path).batches[0]
    scientific = json.loads(json.dumps(batch.scientific_values))
    scientific["material"]["parameter_registry"]["T_in_base"].pop("nominal")
    malformed = replace(batch, scientific_values=scientific)
    with pytest.raises(ValueError, match="T_in_base"):
        sampling_service.sample_case(malformed, 1)


def test_balance_quadrature_formulas_and_native_statistics_have_no_tolerance() -> None:
    """Protect all three water balances and tolerance-free native statistics."""
    time_h = np.asarray([0.0, 1.0, 2.0])
    evaporation = np.full(3, 1.0 / 3600.0)
    vapor_in = np.full(3, 0.5 / 3600.0)
    vapor_out = np.full(3, 0.2 / 3600.0)
    result = analysis_service.water_balance_diagnostics(
        time_h,
        {
            "m_w_gr": [10.0, 9.0, 8.0],
            "m_v_gas": [2.0, 3.3, 4.6],
            "m_dot_evap": evaporation,
            "m_dot_v_in": vapor_in,
            "m_dot_v_out": vapor_out,
        },
    )
    assert analysis_service.trapezoidal_interval_integrals(time_h, evaporation) == pytest.approx([1.0, 1.0])
    for name in ("total_water", "granular_water", "gas_water"):
        assert result[name]["interval_residual_kg"] == pytest.approx([0.0, 0.0], abs=2.0e-15)
        assert result[name]["acceptance_tolerance"] is None
    native = analysis_service.native_mass_balance_statistics([-2.0, 1.0, 0.5])
    assert native == {
        "min": -2.0,
        "max": 1.0,
        "max_abs": 2.0,
        "mean_abs": pytest.approx(7.0 / 6.0),
        "rms": pytest.approx(np.sqrt(5.25 / 3.0)),
        "final": 0.5,
        "unit": "kg/s",
        "acceptance_tolerance": None,
    }


@pytest.mark.parametrize(
    ("target_reached", "final_time_h", "expected", "expected_time"),
    [
        (True, 72.0, "PASS", 72.0),
        (True, 12.0, "TOO_FAST", 12.0),
        (False, 96.0, "NOT_DRY_WITHIN_HORIZON", None),
    ],
)
def test_nominal_duration_states_do_not_fabricate_censored_times(
    target_reached: bool,
    final_time_h: float,
    expected: str,
    expected_time: float | None,
) -> None:
    """Protect nominal duration states and uncensored-time semantics."""
    result = analysis_service.duration_diagnostic(
        case_kind="nominal_reference",
        target_reached=target_reached,
        final_time_h=final_time_h,
        last_regular_time_h=min(final_time_h, 96.0),
        final_x_wb_bulk=0.08,
        final_f_wet_dm=0.04 if target_reached else 0.2,
        configured_threshold=0.05,
        configured_horizon_h=96.0,
        previous_regular_f_wet_dm=0.06,
    )
    assert result["result"] == expected
    assert result["drying_time_h"] == expected_time
    assert result["right_censored"] is (not target_reached)
    assert result["adequacy_window_h"] == {
        "minimum": 24.0,
        "maximum": 96.0,
        "minimum_basis": "pilot_protocol_lower_diagnostic",
        "maximum_basis": "resolved_case_time_horizon",
    }


@pytest.mark.parametrize("case_kind", ["nominal_reference", "natural_pilot"])
@pytest.mark.parametrize("target_reached", [True, False])
def test_duration_rejects_post_horizon_final_times(case_kind: str, target_reached: bool) -> None:
    """Protect the configured horizon from fabricated observed or censoring times."""
    with pytest.raises(ValueError, match="horizon"):
        analysis_service.duration_diagnostic(
            case_kind=case_kind,
            target_reached=target_reached,
            final_time_h=49.0,
            last_regular_time_h=48.0,
            final_x_wb_bulk=0.08,
            final_f_wet_dm=0.04 if target_reached else 0.2,
            configured_threshold=0.05,
            configured_horizon_h=48.0,
            previous_regular_f_wet_dm=0.06,
        )


def test_physical_bounds_extrema_monotonicity_and_schedule_are_generic() -> None:
    """Protect generic physical bounds, extrema, trends, and schedule checks."""
    static, states, globals_by_name, scalars = _physical_inputs()
    physical, diagnostics = analysis_service.field_and_physical_diagnostics(
        static=static,
        transient_states=states,
        globals_by_name=globals_by_name,
        scalars=scalars,
    )
    assert physical["status"] == "pass"
    assert physical["permeability"]["positive_definite"] is True
    assert physical["local_m_evap"]["status"] == "derived_from_binding_formula_and_canonical_states"
    assert physical["local_m_evap"]["nonnegative"] is True
    assert diagnostics["run_extrema"]["T_min_run"] == 300.0
    assert diagnostics["run_extrema"]["T_max_run"] == 305.0
    assert diagnostics["run_extrema"]["X_target_wb"] == 0.06
    assert diagnostics["airflow"]["velocity_magnitude_max"] == 0.1
    trends = analysis_service.monotonicity_diagnostics(
        {
            "m_w_gr": [3.0, 2.0, 2.5, 2.4],
            "X_wb_bulk": [0.3, 0.2, 0.21, 0.1],
            "f_wet_dm": [1.0, 0.5, 0.6, 0.2],
        }
    )
    assert trends["m_w_gr"] == {
        "positive_step_count": 1,
        "largest_positive_step": 0.5,
        "total_positive_excursion": 0.5,
        "automatic_failure_tolerance": None,
    }
    schedule = analysis_service.schedule_diagnostics(
        [[0.0, 305.0, 0.01], [1.0, 306.0, 0.011]],
        schedule_metadata={
            "min_phi_source_air": 0.35,
            "max_phi_source_air": 0.55,
            "min_heater_temperature_rise": 4.0,
            "conversion_pressure": {"name": "p_ref", "value": 101325.0, "unit": "Pa", "owner": "package_fixed"},
            "boundary_handoff": {
                "startup_ramp": {"enabled": False, "duration_h": 1.0 / 6.0},
                "rejoin_row": None,
            },
        },
        ambient_temperature=300.0,
        phi_operational_min=0.05,
        phi_operational_max=0.85,
    )
    assert schedule["status"] == "pass"

    startup_schedule = analysis_service.schedule_diagnostics(
        [
            [0.0, 300.0, 0.01],
            [1.0 / 6.0, 305.0, 0.01],
            [1.0, 306.0, 0.011],
        ],
        schedule_metadata={
            "min_phi_source_air": 0.35,
            "max_phi_source_air": 0.55,
            "min_heater_temperature_rise": 4.0,
            "conversion_pressure": {"name": "p_ref", "value": 101325.0, "unit": "Pa", "owner": "package_fixed"},
            "boundary_handoff": {
                "startup_ramp": {"enabled": True, "duration_h": 1.0 / 6.0},
                "canonical_start_row": [0.0, 305.0, 0.01],
                "rejoin_row": [1.0 / 6.0, 305.0, 0.01],
            },
        },
        ambient_temperature=300.0,
        phi_operational_min=0.05,
        phi_operational_max=0.85,
    )
    assert startup_schedule["status"] == "pass"
    assert startup_schedule["checks"]["phi_in_bc_operational"] is True
    assert startup_schedule["checks"]["startup_phi_in_bc_physical"] is True

    invalid_static = {name: values.copy() for name, values in static.items()}
    invalid_states = [{name: values.copy() for name, values in state.items()} for state in states]
    invalid_globals = {name: list(values) for name, values in globals_by_name.items()}
    invalid_static["eps_bed"][0, 0] = 0.9
    invalid_static["Kxy"][0, 0] = 1.0
    invalid_static["rho_bu_dry"][0, 1] = -1.0
    invalid_states[0]["T"][0, 0] = -2.0
    invalid_states[0]["phi"][0, 1] = 1.2
    invalid_states[0]["w_surf"][1, 0] = -5.0
    invalid_globals["f_wet_dm"][0] = 1.2
    invalid_globals["m_dot_evap"][0] = -0.1
    invalid, _diagnostics = analysis_service.field_and_physical_diagnostics(
        static=invalid_static,
        transient_states=invalid_states,
        globals_by_name=invalid_globals,
        scalars=scalars,
    )
    quantities = {record["quantity"] for record in invalid["violations"]}
    assert invalid["status"] == "violation"
    assert {
        "eps_bed",
        "permeability_tensor",
        "rho_bu_dry",
        "T",
        "phi",
        "w_surf",
        "X_db",
        "X_wb",
        "f_wet_dm",
        "m_dot_evap",
    }.issubset(quantities)


def test_storage_projection_uses_only_successful_measured_hdf5() -> None:
    """Protect measured-artifact projections against hidden target constants."""
    records = [
        {
            "storage": {
                "canonical_hdf5_bytes": 1000,
                "regular_state_count": 10,
                "transient_dataset_storage_bytes": 400,
                "global_dataset_storage_bytes": 100,
            }
        },
        {
            "storage": {
                "canonical_hdf5_bytes": 2000,
                "regular_state_count": 20,
                "transient_dataset_storage_bytes": 800,
                "global_dataset_storage_bytes": 200,
            }
        },
    ]
    result = analysis_service.production_storage_projection(
        records,
        target_case_count=7,
        regular_state_count=11,
    )
    assert result["basis"] == "observed_real_pilot_based_estimate"
    assert result["target_case_count"] == 7
    assert result["regular_state_count"] == 11
    assert result["mean_based_bytes"] == 10_500
    assert result["median_based_bytes"] == 10_500
    assert result["min_hdf5_bytes_per_case"] == 1000
    assert result["max_hdf5_bytes_per_case"] == 2000
    assert result["full_horizon_projection"]["projected_bytes"] == 9_100
    assert result["full_horizon_projection"]["exact"] is False
    assert result["storage_budget_guard"] is None
    unavailable = analysis_service.production_storage_projection(
        [],
        target_case_count=7,
        regular_state_count=11,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["mean_based_bytes"] is None
    with pytest.raises(ValueError, match="target_case_count"):
        analysis_service.production_storage_projection(
            records,
            target_case_count=0,
            regular_state_count=11,
        )


def test_pre_cleanup_storage_inventories_record_exact_files_and_case_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect exact fixed-point source accounting before pilot cleanup."""
    storage = tmp_path / "storage"
    storage.mkdir()
    run_id = "pilot_inventory_test"
    campaign = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    meta = storage / "01_generation/meta/batches/pilot_batch"
    raw = storage / "01_generation/raw/pilot_batch"
    processed = storage / "01_generation/processed/pilot_batch"
    attempts = storage / f"01_generation/attempts/pilot_batch/case_00002/{run_id}"
    failure = meta / "pilot_failure_evidence/case_00002/failure.json"
    for directory in (
        campaign,
        meta,
        raw / "exports",
        processed,
        attempts,
        failure.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (campaign / "campaign_terminal.json").write_bytes(b"terminal\n")
    (meta / "solver.log").write_bytes(b"log\n")
    failure.write_bytes(b"failure\n")
    export = raw / "exports/fields.csv"
    export.write_bytes(b"x,y\n1,2\n")
    hdf5 = processed / "case.h5"
    hdf5.write_bytes(b"canonical-hdf5")
    (attempts / "attempt.json").write_text("{}\n", encoding="utf-8")

    def relative(path: Path) -> str:
        return path.relative_to(storage).as_posix()

    terminal = {
        "cases": [
            {
                "material": "lentil",
                "case_id": "case_00001",
                "terminal_state": "success",
                "raw_directory": relative(raw),
                "processed_directory": relative(processed),
            },
            {
                "material": "chickpea",
                "case_id": "case_00002",
                "terminal_state": "failure",
                "failure_evidence": {"relative_path": relative(failure)},
            },
        ]
    }
    plan = {
        "campaign_directory": relative(campaign),
        "batches": [
            {
                "meta_directory": relative(meta),
                "raw_directory": relative(raw),
                "processed_directory": relative(processed),
                "attempt_directories": [relative(attempts)],
            }
        ],
    }
    monkeypatch.setattr(pilot_service, "validate_pilot_terminal", lambda *args, **kwargs: terminal)
    monkeypatch.setattr(pilot_service, "pilot_transfer_plan", lambda *args, **kwargs: plan)

    result = pilot_service.record_cpu_source_inventory(run_id, storage_root=storage)
    receipt = campaign / pilot_service.PILOT_SOURCE_INVENTORY_FILENAME
    assert result["cpu_case_bytes_before_cleanup"] == {
        "lentil:case_00001": export.stat().st_size + hdf5.stat().st_size,
        "chickpea:case_00002": failure.stat().st_size,
    }
    assert result["cpu_case_file_counts"] == {
        "lentil:case_00001": 2,
        "chickpea:case_00002": 1,
    }
    assert result["cpu_logs_bytes"] == (meta / "solver.log").stat().st_size
    assert result["cpu_exports_bytes"] == export.stat().st_size
    assert relative(attempts) not in result["cleanup_eligible_publication_directories"]
    all_files = [path for root in (campaign, meta, raw, processed, attempts) for path in root.rglob("*") if path.is_file()]
    unique_files = {path.resolve() for path in all_files}
    assert result["cpu_source_bytes_before_cleanup"] == sum(path.stat().st_size for path in unique_files)
    assert result["cpu_source_file_count_before_cleanup"] == len(unique_files)
    assert receipt.resolve() in unique_files
    assert pilot_service.validate_cpu_source_inventory(run_id, storage_root=storage) == result

    staging = workspace_service.create_transfer_staging(storage_root=storage, run_id=run_id)
    staging_campaign = campaign_evidence.campaign_run_directory(run_id, storage_root=staging)
    staging_campaign.mkdir(parents=True)
    (staging_campaign / "campaign_terminal.json").write_bytes(b"terminal\n")
    staging_result = pilot_service.record_transfer_staging_inventory(
        run_id,
        staging_root=staging,
    )
    staging_files = [path for path in staging.rglob("*") if path.is_file()]
    assert staging_result["transfer_staging_bytes_before_cleanup"] == sum(path.stat().st_size for path in staging_files)
    assert staging_result["transfer_staging_file_count"] == len(staging_files)
    assert (
        pilot_service.validate_transfer_staging_inventory(
            run_id,
            storage_root=staging,
            require_staging_present=True,
        )
        == staging_result
    )


def test_failed_pilot_retains_bounded_attempt_before_scratch_disappears(
    pilot_campaign_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain bounded hash-admitted pilot evidence before scratch disappears."""
    batch = _pilot(pilot_campaign_path).batches[0]
    storage = tmp_path / "storage"
    input_service.generate_input_cases(batch, 1, storage_root=storage)
    prepared = runtime_service.prepare_case_work_directory(
        batch,
        1,
        storage_root=storage,
        work_root=tmp_path / "work",
    )
    (prepared.runtime_directory / "solver.log").write_text(
        "solver failed\n",
        encoding="utf-8",
    )
    (prepared.runtime_directory / "status.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (prepared.exports_directory / "partial.csv").write_text(
        "t,value\n0,1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", "pilot-full-retention")
    receipt = runtime_service.record_case_failure(
        batch,
        1,
        RuntimeError("synthetic pilot failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=prepared.work_directory,
        storage_root=storage,
        failure_stage="solver",
    )
    attempt = attempt_service.load_attempt(receipt.parent)
    attempt_service.record_attempt_cleanup(
        attempt,
        status="complete",
        reclaimed_bytes=0,
        error=None,
    )
    assert runtime_service.case_failure_is_recorded(
        batch,
        1,
        storage_root=storage,
    )
    assert attempt.payload["retention_policy"] == "compact"
    retained = attempt.payload["retained_inventory"]
    assert {
        "payload/runtime/solver.log",
        "payload/runtime/status.json",
    }.issubset(retained)
    assert "payload/case.json" not in retained
    assert "payload/model.mph" not in retained
    assert "payload/exports/partial.csv" not in retained
    shutil.rmtree(prepared.work_directory)
    assert runtime_service.case_failure_is_recorded(
        batch,
        1,
        storage_root=storage,
    )
    retained_log = attempt.directory / "payload/runtime/solver.log"
    retained_log.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity"):
        runtime_service.case_failure_is_recorded(
            batch,
            1,
            storage_root=storage,
        )


def test_pilot_terminal_references_canonical_attempt_without_copying_it(
    pilot_campaign_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one in-place attempt authority for a failed pilot case."""
    campaign = _pilot(pilot_campaign_path)
    full_batch = campaign.batches[0]
    batch = replace(
        full_batch,
        case_indices=(1,),
        assignments={1: full_batch.assignments[1]},
    )
    campaign = replace(
        campaign,
        material_roles={role: (batch.material_family,) if role == batch.material_role else () for role in config_service.MATERIAL_ROLES},
        material_memberships={batch.material_family: ()},
        total_case_count=1,
        batches=(batch,),
    )
    run_id = "pilot-attempt-terminal"
    storage = tmp_path / "storage"
    input_service.generate_input_cases(batch, 1, storage_root=storage)
    prepared = runtime_service.prepare_case_work_directory(
        batch,
        1,
        storage_root=storage,
        work_root=tmp_path / "work",
    )
    attempt = attempt_service.publish_case_attempt(
        batch,
        1,
        campaign_run_id=run_id,
        case_state="failed",
        failure_stage="solver",
        reason="synthetic pilot solver failure",
        solver_git_commit="a" * 40,
        processing_git_commit="a" * 40,
        work_directory=prepared.work_directory,
        storage_root=storage,
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        exit_code=7,
        timed_out=False,
        quality_flags=(
            {
                "code": "solver_native_failure_count",
                "severity": "warning",
                "stage": "solver",
                "message": "Synthetic native solver failure count.",
                "metrics": {"Tfail": 1},
                "thresholds": {},
                "source_artifacts": [],
                "recorded_at": "2026-08-18T00:00:00+00:00",
                "quality_flag": True,
            },
        ),
    )
    attempt_service.record_attempt_cleanup(
        attempt,
        status="complete",
        reclaimed_bytes=0,
        error=None,
    )
    campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    ).mkdir(parents=True)
    manifest = {
        "campaign_config": str(pilot_campaign_path),
        "git_commit": "a" * 40,
        "slurm_job_ids": ["123"],
        "scheduler_job_name": "pilot-attempt-terminal",
        "scheduler_log_directory": str(tmp_path / "logs"),
    }
    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        campaign_evidence,
        "campaign_for_run",
        lambda *_args, **_kwargs: campaign,
    )

    pilot_service.finalize_pilot_campaign(run_id, storage_root=storage)
    terminal = pilot_service.validate_pilot_terminal(run_id, storage_root=storage)

    record = terminal["cases"][0]
    evidence = record["failure_evidence"]
    assert record["simulation_case_id"] == attempt.payload["simulation_case_id"]
    assert evidence["evidence_kind"] == "attempt"
    assert storage / evidence["relative_path"] == attempt.receipt_path
    assert storage / evidence["evidence_directory"] == attempt.directory
    assert evidence["scratch_cleanup"]["status"] == "complete"
    assert not (runtime_service.batch_meta_directory(batch, storage_root=storage) / "pilot_failure_evidence").exists()
    result = pilot_service._failure_case_result(  # noqa: SLF001 -- focused result mapping
        record,
        storage=storage,
    )
    assert result["simulation_case_id"] == attempt.payload["simulation_case_id"]
    assert result["warning_count"] == 1
    assert result["warnings"] == ["solver_native_failure_count"]


def test_staging_cleanup_writes_pending_transaction_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect durable staging-cleanup intent before exact deletion."""
    storage = tmp_path / "storage"
    storage.mkdir()
    run_id = "pilot_cleanup_test"
    staging = workspace_service.create_transfer_staging(storage_root=storage, run_id=run_id)
    (staging / "payload.bin").write_bytes(b"pilot")
    files = [path for path in staging.rglob("*") if path.is_file()]
    expected_bytes = sum(path.stat().st_size for path in files)
    expected_files = len(files)
    directory = pilot_service.pilot_check_directory(run_id, storage_root=storage)
    directory.mkdir(parents=True)
    common.serialization.atomic_write_json(directory / pilot_service.PILOT_PRE_CLEANUP_FILENAME, {"bound": True})
    inventory = {
        "transfer_staging_path": str(staging),
        "transfer_staging_bytes_before_cleanup": expected_bytes,
        "transfer_staging_file_count": expected_files,
    }
    monkeypatch.setattr(pilot_service, "validate_pilot_pre_cleanup", lambda *args, **kwargs: {"bound": True})
    monkeypatch.setattr(pilot_service, "validate_transfer_staging_inventory", lambda *args, **kwargs: inventory)
    original_cleanup = workspace_service.cleanup_transfer_staging

    def observed_cleanup(*args: Any, **kwargs: Any) -> int:
        pending = json.loads((directory / pilot_service.PILOT_STAGING_CLEANUP_FILENAME).read_text(encoding="utf-8"))
        assert pending["status"] == "pending"
        return original_cleanup(*args, **kwargs)

    monkeypatch.setattr(workspace_service, "cleanup_transfer_staging", observed_cleanup)
    receipt = pilot_service.cleanup_recorded_transfer_staging(
        run_id,
        storage_root=storage,
        confirm=True,
    )
    assert receipt["status"] == "complete"
    assert receipt["removed"] is True
    assert receipt["reclaimed_bytes"] == expected_bytes
    assert not staging.exists()


def test_staging_cleanup_accounts_for_residual_after_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep received-byte provenance while reclaiming only live staging bytes."""
    storage = tmp_path / "storage"
    storage.mkdir()
    run_id = "pilot_atomic_publication"
    staging = workspace_service.create_transfer_staging(storage_root=storage, run_id=run_id)
    staging_campaign = campaign_evidence.campaign_run_directory(run_id, storage_root=staging)
    staging_campaign.mkdir(parents=True)
    (staging_campaign / "campaign_terminal.json").write_bytes(b"terminal evidence" + bytes([10]))
    inventory = pilot_service.record_transfer_staging_inventory(
        run_id,
        staging_root=staging,
    )

    destination_campaign = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    destination_campaign.parent.mkdir(parents=True, exist_ok=True)
    staging_campaign.replace(destination_campaign)
    residual_files = [path for path in staging.rglob("*") if path.is_file()]
    residual_bytes = sum(path.stat().st_size for path in residual_files)
    assert residual_bytes < inventory["transfer_staging_bytes_before_cleanup"]
    assert (
        pilot_service.validate_transfer_staging_inventory(
            run_id,
            storage_root=storage,
            require_staging_present=True,
        )
        == inventory
    )

    check_directory = pilot_service.pilot_check_directory(run_id, storage_root=storage)
    check_directory.mkdir(parents=True)
    common.serialization.atomic_write_json(
        check_directory / pilot_service.PILOT_PRE_CLEANUP_FILENAME,
        {"bound": True},
    )
    monkeypatch.setattr(pilot_service, "validate_pilot_pre_cleanup", lambda *args, **kwargs: {"bound": True})

    receipt = pilot_service.cleanup_recorded_transfer_staging(
        run_id,
        storage_root=storage,
        confirm=True,
    )

    assert receipt["status"] == "complete"
    assert receipt["expected_bytes"] == residual_bytes
    assert receipt["expected_file_count"] == len(residual_files)
    assert receipt["reclaimed_bytes"] == residual_bytes
    assert not staging.exists()
    assert (destination_campaign / pilot_service.PILOT_STAGING_INVENTORY_FILENAME).is_file()


def test_cleanup_finalization_uses_live_staging_cleanup_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind final cleanup bytes to residual intent, not received provenance."""
    storage = tmp_path / "storage"
    storage.mkdir()
    run_id = "pilot_residual_finalization"
    check_directory = pilot_service.pilot_check_directory(run_id, storage_root=storage)
    check_directory.mkdir(parents=True)
    common.serialization.atomic_write_json(
        pilot_service.pilot_receipt_path(run_id, storage_root=storage),
        {
            "transfer_staging_inventory": {
                "transfer_staging_bytes_before_cleanup": 10_000,
            },
            "cleanup": {"cleanup_requested": False},
        },
    )
    cleanup_path = check_directory / pilot_service.PILOT_STAGING_CLEANUP_FILENAME
    common.serialization.atomic_write_json(
        cleanup_path,
        {
            "expected_bytes": 17,
            "status": "complete",
            "removed": True,
            "reclaimed_bytes": 17,
        },
    )
    cleanup_sha256 = common.serialization.file_sha256(cleanup_path)
    monkeypatch.setattr(pilot_service, "validate_pilot_pre_cleanup", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pilot_service,
        "_write_receipt_and_views",
        lambda _run_id, receipt, **_kwargs: receipt,
    )

    receipt = pilot_service.finalize_cleanup_receipt(
        run_id,
        storage_root=storage,
        workflow_evidence=pilot_service.CleanupWorkflowEvidence(
            campaign_run_id=run_id,
            status="skipped_by_request",
            receipt_sha256=None,
            reclaimed_bytes=0,
        ),
        cpu_source_removed=False,
        cpu_bytes_reclaimed=0,
        cpu_cleanup_receipt_sha256=None,
        transfer_staging_removed=True,
        staging_bytes_reclaimed=17,
        staging_cleanup_receipt_sha256=cleanup_sha256,
    )

    assert receipt["cleanup"]["transfer_staging"]["bytes_reclaimed"] == 17
    assert receipt["transfer_staging_inventory"]["transfer_staging_bytes_before_cleanup"] == 10_000


def test_missing_retained_evidence_blocks_pre_cleanup(tmp_path: Path) -> None:
    """Protect fail-closed cleanup authorization when GPU evidence is missing."""
    storage = tmp_path / "storage"
    directory = pilot_service.pilot_check_directory("missing_evidence", storage_root=storage)
    directory.mkdir(parents=True)
    common.serialization.atomic_write_json(
        directory / pilot_service.PILOT_PRE_CLEANUP_FILENAME,
        {
            "schema_kind": pilot_service.PILOT_RECEIPT_SCHEMA_KIND,
            "schema_version": pilot_service.PILOT_SCHEMA_VERSION,
            "pilot_check_id": "missing_evidence",
            "cleanup": {"authorized": True},
            "cases": [_canonical_case_result()],
            "production_storage_projection": {
                "target_campaign_id": "transient_target",
                "target_campaign_digest": "a" * 64,
                "simulation_profile": "transient_drying",
                "target_case_count": 7,
                "regular_state_count": 11,
                "regular_time_start_h": 0.0,
                "time_horizon_h": 10.0,
                "status": "unavailable",
                "mean_based_bytes": None,
                "median_based_bytes": None,
            },
            "retained_evidence_paths": [str(storage / "does-not-exist")],
        },
    )
    with pytest.raises(ValueError, match="pre-cleanup evidence"):
        pilot_service.validate_pilot_pre_cleanup("missing_evidence", storage_root=storage)
