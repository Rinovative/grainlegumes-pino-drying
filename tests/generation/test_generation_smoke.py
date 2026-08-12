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
from src.generation.contracts import generation_contracts_mapping as mapping_contract
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


def test_mapping_probe_reports_exact_incomplete_keys_and_actual_table(
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
        """schema_kind: generation_profile
schema_version: 2
simulation_profile: steady_flow
steady_flow_conditioning: null
exports:
- role: steady_flow_fields
  temporal_kind: stationary
  source: observed.csv
  delimiter: ','
  columns:
    x: null
    y: y
""",
        encoding="utf-8",
    )
    raw = mapping_probe._profile_mapping(profile)

    table = tmp_path / "observed.csv"
    table.write_text("x,y,p\n0,1,2\n3,4,5\n", encoding="utf-8")
    observation = mapping_probe._table_observation(table)
    assert observation == {
        "delimiter": ",",
        "raw_header": ["x", "y", "p"],
        "shape": [2, 3],
        "rectangular": True,
        "comsol_metadata": {},
        "time_header_candidates": [],
    }
    comparison = mapping_probe._mapping_comparison(
        raw,
        [{"relative_path": "observed.csv", "table": observation}],
        profile_path=profile,
    )
    assert comparison["required_corrections"] == ["profile.yaml:exports[0].columns.x"]
    assert comparison["required_missing_exports"] == []
    assert comparison["aliases_used"] is False


def test_mapping_probe_distinguishes_missing_mismatched_and_confirmed_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report export execution failure before making any per-column claim."""
    monkeypatch.setattr(mapping_probe.common.paths, "get_project_root", lambda: tmp_path)
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        """schema_kind: generation_profile
schema_version: 2
simulation_profile: steady_flow
steady_flow_conditioning: null
exports:
- role: steady_flow_fields
  temporal_kind: stationary
  source: observed.csv
  delimiter: ','
  columns:
    x: x
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
                "table": {"delimiter": ",", "raw_header": ["wrong"]},
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
                "table": {"delimiter": ",", "raw_header": ["x"]},
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
    steady_observation = report["mapping_comparison"]["observations"][0]

    assert report["status"] == "mapping_observation_complete"
    assert steady_observation["raw_header"] == steady_observation["canonical_header"]
    assert steady_observation["parsed_shape"][0] > 0
    assert command[command.index("-inputfile") + 1] == "model.mph"
    assert command[command.index("-job") + 1] == "b1"
    assert command[command.index("-outputfile") + 1] == "solved.mph"
    assert command[command.index("-np") + 1] == "16"
    assert "-nosave" not in command


@pytest.mark.parametrize(
    ("export_mode", "expected_status"),
    [
        ("mismatch", "mapping_update_required"),
        ("missing", "required_export_missing"),
    ],
)
def test_fake_mapping_probe_preserves_negative_status_semantics(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_mode: str,
    expected_status: str,
) -> None:
    """Keep genuine header mismatches distinct from absent required exports."""
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
    monkeypatch.setenv("FAKE_COMSOL_MAPPING_EXPORT_MODE", export_mode)

    report_path = mapping_probe.run_mapping_probe(
        config_path,
        storage_root=tmp_path / f"{export_mode} storage",
        work_root=tmp_path / f"{export_mode} work",
        cores_per_case=16,
    )
    report = mapping_probe.load_mapping_probe(report_path)

    assert report["status"] == expected_status
    if export_mode == "mismatch":
        assert report["required_exports_missing"] == []
        assert any(key.endswith(".columns.x") for key in report["fields_requiring_correction"])
    else:
        assert report["required_exports_missing"] == [{"role": "steady_flow_fields", "declared_pattern": "airflow.csv"}]
        assert report["fields_requiring_correction"] == []


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


def _successful_mapping_report(expected: dict[str, Any]) -> dict[str, Any]:
    """Return one compact successful report for pure evidence-validity tests."""
    return {
        "schema_kind": mapping_probe.MAPPING_PROBE_SCHEMA_KIND,
        "schema_version": mapping_probe.MAPPING_PROBE_SCHEMA_VERSION,
        "status": "mapping_observation_complete",
        "simulation_profile": expected["simulation_profile"],
        "mapping_contract_sha256": expected["mapping_contract_sha256"],
        "git_commit": "a" * 40,
        "template": {"sha256": expected["template_sha256"]},
        "comsol": {
            "exact_version": expected["comsol_version"],
            "version_output": f"COMSOL Multiphysics {expected['comsol_version']}",
        },
        "fields_requiring_correction": [],
        "required_exports_missing": [],
        "exit_code": 0,
        "timed_out": False,
        "start_error": None,
        "mapping_auto_detection_used": False,
        "production_solve_started": False,
        "mapping_comparison": {
            "required_corrections": [],
            "required_missing_exports": [],
            "observations": [
                {
                    "role": "steady_flow_fields",
                    "matched_relative_paths": ["observed.csv"],
                    "delimiter_matches": True,
                    "temporal_structure_error": None,
                    "columns": {
                        "x": {
                            "declared_source_header": "x",
                            "observed_exact_header_match": True,
                        }
                    },
                }
            ],
        },
        "actual_file_inventory": [
            {
                "relative_path": "observed.csv",
                "sha256": "e" * 64,
                "size_bytes": 1,
            }
        ],
    }


def test_readiness_accepts_complete_profiles_with_matching_probe_evidence(
    tmp_path: Path,
) -> None:
    """Protect the real-run regression where successful evidence must unblock mappings."""
    storage = tmp_path / "readiness storage"
    storage.mkdir()
    version_output = "COMSOL Multiphysics 6.4.0.293"
    campaigns = {
        "steady_flow": Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        "transient_drying": Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
    }
    for profile_id, campaign_path in campaigns.items():
        expected = mapping_probe.build_mapping_evidence_context(
            campaign_path,
            comsol_version_output=version_output,
        )
        root = storage / "01_generation/meta/mapping_probes" / f"{profile_id}-probe"
        produced = root / "produced_files"
        produced.mkdir(parents=True)
        observed = produced / "observed.csv"
        observed.write_bytes(profile_id.encode("utf-8"))
        report = _successful_mapping_report(expected)
        report["actual_file_inventory"][0].update(
            {
                "sha256": mapping_probe.common.serialization.file_sha256(observed),
                "size_bytes": observed.stat().st_size,
            }
        )
        (root / "mapping_probe.json").write_text(
            __import__("json").dumps(report),
            encoding="utf-8",
        )

    report = generation.readiness.build_readiness_report(
        campaigns["steady_flow"],
        campaigns["transient_drying"],
        storage_root=storage,
        comsol_version_output=version_output,
    )
    assert report["profile_mapping_configuration_complete"] is True
    assert report["mapping_evidence_complete"] is True
    assert report["profile_mapping_complete"] is True
    assert all(evidence["status"] == "mapping_evidence_valid" for evidence in report["mapping_evidence"].values())


def test_mapping_contract_fingerprint_tracks_only_mapping_semantics(
    generation_config_factory: Any,
) -> None:
    """Invalidate every mapping-relevant change while ignoring unrelated metadata."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    campaign = generation.cases.config.load_campaign_config(config_path)
    output = campaign.batches[0].scientific_values["output_contract"]
    original = mapping_contract.mapping_contract_sha256("transient_drying", output)

    mutations = []
    for key, value in (
        ("pattern", "changed.csv"),
        ("delimiter", ","),
        ("temporal_kind", "changed_time_semantics"),
    ):
        changed = copy.deepcopy(output)
        changed["exports"][0][key] = value
        mutations.append(changed)
    changed = copy.deepcopy(output)
    first_logical = next(iter(changed["exports"][0]["columns"]))
    changed["exports"][0]["columns"][first_logical] = "changed_header"
    mutations.append(changed)
    changed = copy.deepcopy(output)
    changed["exports"][0]["units"][first_logical] = "changed_unit"
    mutations.append(changed)
    changed = copy.deepcopy(output)
    changed["exports"][0]["columns"].pop(first_logical)
    changed["exports"][0]["units"].pop(first_logical)
    mutations.append(changed)

    assert all(mapping_contract.mapping_contract_sha256("transient_drying", changed) != original for changed in mutations)
    unrelated = copy.deepcopy(output)
    unrelated["display_metadata"] = {"title": "ignored"}
    assert mapping_contract.mapping_contract_sha256("transient_drying", unrelated) == original


def test_mapping_evidence_validity_tuple_excludes_git_commit(
    generation_config_factory: Any,
) -> None:
    """Bind evidence to semantics, template, COMSOL, and verifier but not commit."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    expected = mapping_probe.build_mapping_evidence_context(
        config_path,
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
    )
    report = _successful_mapping_report(expected)
    assert mapping_probe.evaluate_mapping_probe_report(report, expected)["valid"] is True

    report["git_commit"] = "b" * 40
    assert mapping_probe.evaluate_mapping_probe_report(report, expected)["valid"] is True

    variants = (
        ("template", {"sha256": "c" * 64}, "template SHA-256 changed"),
        ("mapping_contract_sha256", "d" * 64, "output mapping contract changed"),
        ("comsol", {"exact_version": "6.5.0.1", "version_output": "COMSOL 6.5.0.1"}, "COMSOL version changed"),
        ("schema_version", mapping_probe.MAPPING_PROBE_SCHEMA_VERSION - 1, "verifier version differs"),
    )
    for key, value, reason in variants:
        changed = copy.deepcopy(report)
        changed[key] = value
        result = mapping_probe.evaluate_mapping_probe_report(changed, expected)
        assert result["valid"] is False
        assert reason in result["reasons"]


def test_mapping_evidence_requires_matching_successful_probe(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reproduce complete declarations becoming ready only from matching evidence."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    storage = tmp_path / "evidence storage"
    storage.mkdir()
    expected = mapping_probe.build_mapping_evidence_context(
        config_path,
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
    )
    missing = mapping_probe.discover_mapping_evidence(storage_root=storage, expected=expected)
    assert missing["status"] == "mapping_evidence_missing"

    root = storage / "01_generation/meta/mapping_probes/probe-1"
    produced = root / "produced_files"
    produced.mkdir(parents=True)
    observed = produced / "observed.csv"
    observed.write_bytes(b"x")
    report = _successful_mapping_report(expected)
    report["actual_file_inventory"][0]["sha256"] = mapping_probe.common.serialization.file_sha256(observed)
    (root / "mapping_probe.json").write_text(
        __import__("json").dumps(report),
        encoding="utf-8",
    )
    valid = mapping_probe.discover_mapping_evidence(storage_root=storage, expected=expected)
    assert valid["status"] == "mapping_evidence_valid"

    report["status"] = "mapping_update_required"
    report["fields_requiring_correction"] = ["profile.yaml:exports[0].columns.x"]
    report["mapping_comparison"]["required_corrections"] = report["fields_requiring_correction"]
    (root / "mapping_probe.json").write_text(
        __import__("json").dumps(report),
        encoding="utf-8",
    )
    failed = mapping_probe.discover_mapping_evidence(storage_root=storage, expected=expected)
    assert failed["status"] == "mapping_evidence_invalid"
    assert "probe status is 'mapping_update_required'" in failed["reason"]
