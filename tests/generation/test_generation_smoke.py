# ruff: noqa: ARG001, EM101, S101, SLF001, TRY003
"""Profile smoke-evidence and paired runtime diagnostic contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest
import yaml

from src import generation
from src.generation.contracts import generation_contracts_mapping as mapping_contract
from src.generation.contracts import generation_contracts_profiles as profiles


def test_transient_smoke_reads_fixed_airflow_values_from_scientific_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep package-fixed airflow values out of transient sampled scalars."""
    path = tmp_path / "transient-case.h5"
    fixed = {
        "T_flow_ref": 300.65,
        "p_ref": 101325.0,
        "p_out": 0.0,
    }
    scalar_names = profiles.TRANSIENT_SCALAR_INPUT_FIELDS
    scalar_values = np.arange(1, len(scalar_names) + 1, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        static = handle.create_group("static").create_dataset(
            "fields",
            data=np.ones((1, 2, 2), dtype=np.float64),
        )
        static.attrs["field_names"] = json.dumps(["epsilon"])
        scalar = handle.create_group("scalar").create_dataset(
            "values",
            data=scalar_values,
        )
        scalar.attrs["field_names"] = json.dumps(list(scalar_names))
        handle.create_group("schedule").create_dataset(
            "values",
            data=np.asarray([[0.0, 300.0, 0.01]], dtype=np.float64),
        )
        handle.create_group("global").create_dataset(
            "values",
            data=np.asarray([[0.0, 1.0]], dtype=np.float64),
        )
        transient = handle.create_group("transient").create_dataset(
            "fields",
            data=np.ones((1, 1, 2, 2), dtype=np.float64),
        )
        transient.attrs["field_names"] = json.dumps(["T"])
        provenance = handle.create_group("provenance")
        provenance.create_dataset(
            "scientific_config_json",
            data=json.dumps({"scientific_fixed_values": fixed}),
        )
        provenance.create_dataset(
            "stationary_fixed_ownership_json",
            data=json.dumps(
                {
                    name: {
                        "owner": "package_fixed",
                        "fixed_value": value,
                        "unit": "K" if name == "T_flow_ref" else "Pa",
                    }
                    for name, value in fixed.items()
                }
            ),
        )
        provenance.create_dataset(
            "source_exports_json",
            data=json.dumps(
                {
                    "global.csv": {
                        "logical_role": "global_timeseries",
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                    }
                }
            ),
        )
    monkeypatch.setattr(
        generation.smoke.storage_service,
        "validate_case_hdf5",
        lambda *_args, **_kwargs: {
            "case_input_id": "b" * 64,
            "simulation_case_id": "c" * 64,
        },
    )

    _static, observed_fixed, scalars, *_remaining = generation.smoke._load_hdf5(
        path,
        profile_id=profiles.TRANSIENT_DRYING_PROFILE,
        campaign_run_id="transient-smoke__0123456789abcdef",
        batch_id="transient-smoke-batch",
        case_id="case_0001",
    )

    assert observed_fixed == fixed
    assert set(scalars) == set(profiles.TRANSIENT_SCALAR_INPUT_FIELDS)
    assert not set(profiles.STATIONARY_FIXED_FIELDS).intersection(scalars)

    incomplete_fixed = dict(fixed)
    del incomplete_fixed["T_flow_ref"]
    with h5py.File(path, "r+") as handle:
        scientific_dataset = handle.get("provenance/scientific_config_json")
        assert isinstance(scientific_dataset, h5py.Dataset)
        scientific_dataset[()] = json.dumps({"scientific_fixed_values": incomplete_fixed})
    with pytest.raises(
        ValueError,
        match=r"fixed-value ownership is invalid: .*field=T_flow_ref",
    ) as caught:
        generation.smoke._load_hdf5(
            path,
            profile_id=profiles.TRANSIENT_DRYING_PROFILE,
            campaign_run_id="transient-smoke__0123456789abcdef",
            batch_id="transient-smoke-batch",
            case_id="case_0001",
        )
    message = str(caught.value)
    assert "field=T_flow_ref" in message
    assert "profile=transient_drying" in message
    assert "case=case_0001" in message
    assert str(path) in message


def _successful_smoke_report(
    expected: dict[str, Any],
    *,
    run_id: str,
    git_commit: str = "a" * 40,
) -> dict[str, Any]:
    """Return compact complete profile evidence for pure validity tests."""
    cases = [
        {
            "case_id": f"case_{index:04d}",
            "case_input_id": f"input-{index}",
            "simulation_case_id": f"{index + 1:064x}",
            "hdf5_sha256": f"{index + 2:064x}",
            "publication_provenance_sha256": f"{index + 3:064x}",
        }
        for index in range(expected["required_case_count"])
    ]
    report: dict[str, Any] = {
        "schema_kind": generation.smoke.TECHNICAL_SMOKE_EVIDENCE_SCHEMA_KIND,
        "schema_version": generation.smoke.TECHNICAL_SMOKE_EVIDENCE_SCHEMA_VERSION,
        "status": "technical_smoke_complete",
        "recorded_at": "2026-08-12T00:00:00+00:00",
        "simulation_profile": expected["simulation_profile"],
        "mapping_contract_sha256": expected["mapping_contract_sha256"],
        "template": {"relative_path": "simulation/template.mph", "sha256": expected["template_sha256"]},
        "comsol": {
            "exact_version": expected["comsol_version"],
            "version_output": f"COMSOL Multiphysics {expected['comsol_version']}",
        },
        "technical_smoke_contract_sha256": expected["technical_smoke_contract_sha256"],
        "technical_smoke_campaign_id": expected["technical_smoke_campaign_id"],
        "campaign_run_id": run_id,
        "git_commit": git_commit,
        "required_case_count": expected["required_case_count"],
        "cases": cases,
        "workflow_gate_sha256": "f" * 64,
    }
    report["evidence_digest"] = generation.smoke.common.serialization.canonical_json_sha256(report)
    return report


def _write_smoke_evidence(storage: Path, report: dict[str, Any]) -> Path:
    """Write one compact test evidence record under its canonical campaign path."""
    path = storage / "01_generation/meta/campaigns" / report["campaign_run_id"] / "technical_smoke_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _validated_case_summary(batch: Any, case_index: int) -> generation.smoke._CaseEvidence:
    """Return one compact summary at the validated publication boundary."""
    identity = case_index + 1
    return generation.smoke._CaseEvidence(
        record={
            "case_id": batch.case_id(case_index),
            "case_input_id": f"case-input-{identity}",
            "simulation_case_id": f"{identity:064x}",
            "hdf5": {"sha256": f"{identity + 10:064x}"},
            "publication": {"provenance_sha256": f"{identity + 20:064x}"},
        },
        static={},
        stationary_fixed={},
        scalars={},
        schedule=None,
        global_values=None,
        initial_state={},
    )


def _patch_finalizer_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    campaign: Any,
    cases: dict[int, generation.smoke._CaseEvidence],
) -> None:
    """Provide compact successful lifecycle evidence to the real finalizer."""

    def campaign_for_run(
        _run_id: str,
        *,
        storage_root: Path | str | None = None,
    ) -> Any:
        assert storage_root is not None
        return campaign

    def validate_workflow(
        _run_id: str,
        *,
        storage_root: Path | str | None = None,
    ) -> dict[str, Any]:
        assert storage_root is not None
        return {
            "cleanup_requested": False,
            "cpu_cleanup_complete": {
                "status": "skipped_by_request",
                "evidence": None,
            },
            "workflow_gate_sha256": "f" * 64,
        }

    def validate_terminal(
        _run_id: str,
        *,
        storage_root: Path | str | None = None,
    ) -> dict[str, Any]:
        assert storage_root is not None
        return {"git_commit": "a" * 40}

    def case_evidence(
        _batch: Any,
        case_index: int,
        *,
        campaign_run_id: str,
        storage: Path,
    ) -> generation.smoke._CaseEvidence:
        assert campaign_run_id
        assert storage.is_dir()
        try:
            return cases[case_index]
        except KeyError as error:
            message = f"Required technical-smoke case {case_index} is not admissible."
            raise FileNotFoundError(message) from error

    monkeypatch.setattr(
        generation.smoke.campaign_evidence,
        "campaign_for_run",
        campaign_for_run,
    )
    monkeypatch.setattr(
        generation.smoke.workflow_service,
        "validate_all_workflow_receipt",
        validate_workflow,
    )
    monkeypatch.setattr(
        generation.smoke.campaign_runtime,
        "validate_terminal_campaign",
        validate_terminal,
    )
    monkeypatch.setattr(generation.smoke, "_case_evidence", case_evidence)


def test_finalizer_atomically_publishes_and_reuses_complete_smoke_evidence(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real finalization, immutable reuse, and stale-identity rejection."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        natural_count=2,
        retain_raw_csv=True,
        retain_solved_model=True,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    cases = {case_index: _validated_case_summary(batch, case_index) for case_index in batch.case_indices}
    storage = tmp_path / "complete smoke storage"
    storage.mkdir()
    run_id = "synthetic-complete-smoke"
    version_output = "COMSOL Multiphysics 6.4.0.293"
    _patch_finalizer_dependencies(monkeypatch, campaign, cases)

    atomic_paths: list[Path] = []
    atomic_write_json = generation.smoke.common.serialization.atomic_write_json

    def record_atomic_write(
        destination: Path | str,
        payload: Any,
        *,
        indent: int = 2,
    ) -> Path:
        atomic_paths.append(Path(destination).resolve())
        return atomic_write_json(destination, payload, indent=indent)

    monkeypatch.setattr(
        generation.smoke.common.serialization,
        "atomic_write_json",
        record_atomic_write,
    )
    expected = generation.smoke.build_technical_smoke_evidence_context(
        campaign.source_path,
        comsol_version_output=version_output,
    )

    path = generation.smoke.finalize_technical_smoke_evidence(
        run_id,
        comsol_version_output=version_output,
        storage_root=storage,
    )
    report = generation.smoke.load_technical_smoke_evidence(
        path,
        storage_root=storage,
    )

    assert atomic_paths == [path]
    assert report["simulation_profile"] == campaign.profile.id
    assert report["mapping_contract_sha256"] == expected["mapping_contract_sha256"]
    assert report["template"]["sha256"] == expected["template_sha256"]
    assert report["comsol"]["exact_version"] == expected["comsol_version"]
    assert report["schema_kind"] == expected["verifier_schema_kind"]
    assert report["schema_version"] == expected["verifier_schema_version"]
    assert report["technical_smoke_campaign_id"] == campaign.campaign_id
    assert report["technical_smoke_contract_sha256"] == campaign.campaign_digest
    assert report["required_case_count"] == len(batch.case_indices)
    assert [case["case_id"] for case in report["cases"]] == [cases[case_index].record["case_id"] for case_index in batch.case_indices]
    assert [case["hdf5_sha256"] for case in report["cases"]] == [cases[case_index].record["hdf5"]["sha256"] for case_index in batch.case_indices]
    assert [case["publication_provenance_sha256"] for case in report["cases"]] == [
        cases[case_index].record["publication"]["provenance_sha256"] for case_index in batch.case_indices
    ]
    assert (
        generation.smoke.evaluate_technical_smoke_evidence(
            report,
            expected,
        )["valid"]
        is True
    )

    original_bytes = path.read_bytes()
    reused = generation.smoke.finalize_technical_smoke_evidence(
        run_id,
        comsol_version_output=version_output,
        storage_root=storage,
    )
    assert reused == path
    assert atomic_paths == [path]
    assert path.read_bytes() == original_bytes

    with pytest.raises(ValueError, match="COMSOL version changed"):
        generation.smoke.finalize_technical_smoke_evidence(
            run_id,
            comsol_version_output="COMSOL Multiphysics 6.5.0.1",
            storage_root=storage,
        )
    assert atomic_paths == [path]
    assert path.read_bytes() == original_bytes


def test_finalizer_publishes_nothing_for_an_incomplete_required_smoke(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed before evidence publication when one required case is absent."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        natural_count=2,
        retain_raw_csv=True,
        retain_solved_model=True,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    first_case = batch.case_indices[0]
    cases = {first_case: _validated_case_summary(batch, first_case)}
    storage = tmp_path / "partial smoke storage"
    storage.mkdir()
    run_id = "synthetic-partial-smoke"
    version_output = "COMSOL Multiphysics 6.4.0.293"
    _patch_finalizer_dependencies(monkeypatch, campaign, cases)

    def reject_publication(*_args: Any, **_kwargs: Any) -> Path:
        pytest.fail("Incomplete technical smoke reached evidence publication.")

    monkeypatch.setattr(
        generation.smoke.common.serialization,
        "atomic_write_json",
        reject_publication,
    )
    evidence_path = generation.smoke.technical_smoke_evidence_path(
        run_id,
        storage_root=storage,
    )

    with pytest.raises(FileNotFoundError, match="is not admissible"):
        generation.smoke.finalize_technical_smoke_evidence(
            run_id,
            comsol_version_output=version_output,
            storage_root=storage,
        )

    assert not evidence_path.exists()
    status = generation.smoke.technical_smoke_evidence_status(
        campaign.source_path,
        storage_root=storage,
        comsol_version_output=version_output,
    )
    assert status["status"] == "technical_smoke_evidence_missing"


def test_readiness_accepts_complete_profiles_with_matching_smoke_evidence(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Require completed technical-smoke evidence without static verification state."""
    storage = tmp_path / "readiness storage"
    storage.mkdir()
    version_output = "COMSOL Multiphysics 6.4.0.293"
    steady_path, _steady_template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=3,
        campaign_purpose="family_generalization",
    )
    transient_path, _transient_template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
        natural_count=3,
        campaign_purpose="family_generalization",
    )
    steady_smoke_path, _steady_smoke_template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        campaign_purpose="technical_runtime_smoke",
    )
    transient_smoke_path, _transient_smoke_template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
        campaign_purpose="technical_runtime_smoke",
    )
    for family_path, smoke_path in (
        (steady_path, steady_smoke_path),
        (transient_path, transient_smoke_path),
    ):
        shared_profile = (family_path.parent / "profile.yaml").resolve()
        for campaign_path in (family_path, smoke_path):
            campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
            campaign["profile_config"] = str(shared_profile)
            campaign_path.write_text(
                yaml.safe_dump(campaign, sort_keys=False),
                encoding="utf-8",
            )
    campaigns = {
        "steady_flow": steady_path,
        "transient_drying": transient_path,
    }
    for index, campaign_path in enumerate(campaigns.values()):
        expected = generation.smoke.build_technical_smoke_evidence_context(
            campaign_path,
            comsol_version_output=version_output,
        )
        _write_smoke_evidence(
            storage,
            _successful_smoke_report(expected, run_id=f"technical-smoke-{index}"),
        )

    report = generation.readiness.build_readiness_report(
        campaigns["steady_flow"],
        campaigns["transient_drying"],
        storage_root=storage,
        comsol_version_output=version_output,
    )
    assert report["profile_mapping_configuration_complete"] is True
    assert report["technical_smoke_evidence_complete"] is True
    assert report["profile_mapping_complete"] is True
    assert all(evidence["status"] == "technical_smoke_evidence_valid" for evidence in report["technical_smoke_evidence"].values())


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


def test_smoke_evidence_validity_tuple_excludes_git_commit(
    generation_config_factory: Any,
) -> None:
    """Bind evidence to semantics, template, COMSOL, verifier, and smoke contract."""
    campaign, _template = generation_config_factory(
        simulation_profile="steady_flow",
        campaign_purpose="technical_runtime_smoke",
    )
    expected = generation.smoke.build_technical_smoke_evidence_context(
        campaign,
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
    )
    report = _successful_smoke_report(expected, run_id="steady-smoke")
    assert generation.smoke.evaluate_technical_smoke_evidence(report, expected)["valid"] is True

    report["git_commit"] = "b" * 40
    assert generation.smoke.evaluate_technical_smoke_evidence(report, expected)["valid"] is True

    variants = (
        ("template", {"relative_path": "template.mph", "sha256": "c" * 64}, "template SHA-256 changed"),
        ("mapping_contract_sha256", "d" * 64, "output mapping contract changed"),
        ("comsol", {"exact_version": "6.5.0.1", "version_output": "COMSOL 6.5.0.1"}, "COMSOL version changed"),
        (
            "schema_version",
            generation.smoke.TECHNICAL_SMOKE_EVIDENCE_SCHEMA_VERSION + 1,
            "technical-smoke verifier version differs",
        ),
    )
    for key, value, reason in variants:
        changed = copy.deepcopy(report)
        changed[key] = value
        result = generation.smoke.evaluate_technical_smoke_evidence(changed, expected)
        assert result["valid"] is False
        assert reason in result["reasons"]


def test_smoke_evidence_requires_complete_success_and_is_profile_scoped(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject missing, failed, and partial evidence without coupling profiles."""
    storage = tmp_path / "evidence storage"
    storage.mkdir()
    version_output = "COMSOL Multiphysics 6.4.0.293"
    steady_campaign, _steady_template = generation_config_factory(
        simulation_profile="steady_flow",
        campaign_purpose="technical_runtime_smoke",
    )
    transient_campaign, _transient_template = generation_config_factory(
        simulation_profile="transient_drying",
        campaign_purpose="technical_runtime_smoke",
    )
    expected = generation.smoke.build_technical_smoke_evidence_context(
        steady_campaign,
        comsol_version_output=version_output,
    )

    missing = generation.smoke.discover_technical_smoke_evidence(
        storage_root=storage,
        expected=expected,
    )
    assert missing["status"] == "technical_smoke_evidence_missing"

    report = _successful_smoke_report(expected, run_id="steady-smoke")
    path = _write_smoke_evidence(storage, report)
    valid = generation.smoke.technical_smoke_evidence_status(
        steady_campaign,
        storage_root=storage,
        comsol_version_output=version_output,
    )
    assert valid["status"] == "technical_smoke_evidence_valid"
    transient = generation.smoke.technical_smoke_evidence_status(
        transient_campaign,
        storage_root=storage,
        comsol_version_output=version_output,
    )
    assert transient["status"] == "technical_smoke_evidence_missing"

    report["status"] = "technical_smoke_failed"
    report["evidence_digest"] = generation.smoke.common.serialization.canonical_json_sha256(
        {key: value for key, value in report.items() if key != "evidence_digest"}
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    failed = generation.smoke.discover_technical_smoke_evidence(
        storage_root=storage,
        expected=expected,
    )
    assert failed["status"] == "technical_smoke_evidence_invalid"

    report = _successful_smoke_report(expected, run_id="steady-smoke")
    report["cases"].pop()
    report["evidence_digest"] = generation.smoke.common.serialization.canonical_json_sha256(
        {key: value for key, value in report.items() if key != "evidence_digest"}
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    partial = generation.smoke.discover_technical_smoke_evidence(
        storage_root=storage,
        expected=expected,
    )
    assert partial["status"] == "technical_smoke_evidence_invalid"
    assert "every required case" in partial["reason"]


def _paired_receipt_payload(recorded_at: str) -> dict[str, Any]:
    """Return a compact authoritative paired receipt payload."""
    return {
        "schema_kind": generation.smoke.REAL_SMOKE_SCHEMA_KIND,
        "schema_version": generation.smoke.REAL_SMOKE_SCHEMA_VERSION,
        "status": "observations_complete_no_scientific_acceptance_threshold",
        "recorded_at": recorded_at,
        "scientific_git_commits": {"steady_flow": "a" * 40, "transient_drying": "b" * 40},
        "material_family_inventory": ["synthetic"],
        "source_binding": {"resolved_campaigns": [], "bundle_digest": "c" * 64},
        "templates": {},
        "profile_mappings": {},
        "campaigns": [
            {"campaign_run_id": "steady-smoke", "workflow_gate_sha256": "d" * 64},
            {"campaign_run_id": "transient-smoke", "workflow_gate_sha256": "e" * 64},
        ],
        "cases": [{"hdf5": {"identity": {"available_learning_views": ("field",)}}}],
        "comsol": {"version_output": "COMSOL Multiphysics 6.4.0.293"},
        "slurm": {"case_job_ids": ["1"]},
        "dataset_packages": [{"dataset_id": "required", "payload_sha256": "f" * 64}],
        "cpu_source_retention": {
            "steady": {"cleanup_requested": True},
            "transient": {"cleanup_requested": True},
            "parent_success_required_before_cleanup": True,
        },
        "template_equivalence": {},
        "mass_balance": {},
        "acceptance_tolerances": {},
    }


def test_real_smoke_receipt_reuses_normalized_current_and_repairs_stale_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize JSON types and repair stale current evidence without rerunning science."""
    storage = tmp_path / "storage"
    storage.mkdir()

    def build_payload(
        _steady: str,
        _transient: str,
        *,
        storage: Path,
        recorded_at: str,
        comsol_version_output: str,
    ) -> dict[str, Any]:
        assert storage.is_dir()
        assert comsol_version_output.startswith("COMSOL")
        return _paired_receipt_payload(recorded_at)

    monkeypatch.setattr(generation.smoke, "_build_payload", build_payload)
    atomic_write = generation.smoke.common.serialization.atomic_write_json
    writes: list[Path] = []

    def record_write(destination: Path | str, payload: Any, *, indent: int = 2) -> Path:
        writes.append(Path(destination))
        return atomic_write(destination, payload, indent=indent)

    monkeypatch.setattr(generation.smoke.common.serialization, "atomic_write_json", record_write)
    path = generation.smoke.finalize_real_smoke(
        "steady-smoke",
        "transient-smoke",
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
        storage_root=storage,
    )
    original = path.read_bytes()
    assert generation.smoke.validate_real_smoke_receipt(path, storage_root=storage)["cases"][0]["hdf5"]["identity"]["available_learning_views"] == [
        "field"
    ]
    assert (
        generation.smoke.finalize_real_smoke(
            "steady-smoke", "transient-smoke", comsol_version_output="COMSOL Multiphysics 6.4.0.293", storage_root=storage
        )
        == path
    )
    assert path.read_bytes() == original
    assert writes == [path]

    stale = json.loads(original)
    stale["cases"][0]["hdf5"]["identity"]["available_learning_views"] = ["stale"]
    path.write_text(json.dumps(stale), encoding="utf-8")
    stale_bytes = path.read_bytes()
    generation.smoke.finalize_real_smoke(
        "steady-smoke", "transient-smoke", comsol_version_output="COMSOL Multiphysics 6.4.0.293", storage_root=storage
    )
    history, replacement, link = generation.smoke._repair_paths(
        path, stale_bytes, generation.smoke.validate_real_smoke_receipt(path, storage_root=storage)["receipt_digest"]
    )
    assert history.read_bytes() == stale_bytes
    assert replacement.is_file()
    linkage = json.loads(link.read_text(encoding="utf-8"))
    assert linkage["stale_receipt_file_sha256"] == generation.smoke.hashlib.sha256(stale_bytes).hexdigest()
    assert "case_artifact_binding" in linkage["mismatch_categories"]
    status = generation.smoke.validate_current_real_smoke_receipts(storage_root=storage)
    assert len(status["valid_current_receipts"]) == 1
    assert status["historical_stale_receipts"][0]["history_path"] == str(history)
    assert writes[-3:] == [replacement, link, path]
    repair_write_count = len(writes)
    assert (
        generation.smoke.finalize_real_smoke(
            "steady-smoke",
            "transient-smoke",
            comsol_version_output="COMSOL Multiphysics 6.4.0.293",
            storage_root=storage,
        )
        == path
    )
    assert len(writes) == repair_write_count

    linkage["canonical_receipt_relative_path"] = "01_generation/meta/real_smoke/missing.json"
    link.write_text(json.dumps(linkage), encoding="utf-8")
    invalid_status = generation.smoke.validate_current_real_smoke_receipts(storage_root=storage)
    assert invalid_status["historical_stale_receipts"] == []
    assert invalid_status["unrecoverable_invalid_receipts"][0]["path"] == str(link)


@pytest.mark.parametrize(
    "artifact_failure",
    [
        "required case.h5 hash is invalid",
        "required raw export hash is invalid",
        "required Dataset package hash is invalid",
    ],
)
def test_real_smoke_repair_fails_closed_before_history_on_authoritative_artifact_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_failure: str,
) -> None:
    """Do not mutate stale receipt history when required current evidence is invalid."""
    storage = tmp_path / "storage"
    storage.mkdir()
    healthy = True

    def build_payload(
        _steady: str,
        _transient: str,
        *,
        storage: Path,
        recorded_at: str,
        comsol_version_output: str,
    ) -> dict[str, Any]:
        if not healthy:
            raise ValueError(artifact_failure)
        return _paired_receipt_payload(recorded_at)

    monkeypatch.setattr(generation.smoke, "_build_payload", build_payload)
    path = generation.smoke.finalize_real_smoke(
        "steady-smoke", "transient-smoke", comsol_version_output="COMSOL Multiphysics 6.4.0.293", storage_root=storage
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["dataset_packages"][0]["payload_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    stale_bytes = path.read_bytes()
    healthy = False
    with pytest.raises(ValueError, match="required"):
        generation.smoke.finalize_real_smoke(
            "steady-smoke", "transient-smoke", comsol_version_output="COMSOL Multiphysics 6.4.0.293", storage_root=storage
        )
    assert path.read_bytes() == stale_bytes
    assert not (path.parent / "receipt_repairs").exists()


def test_real_smoke_boundary_ignores_cleanup_completion_and_additive_packages() -> None:
    """Bind requested policy and launch packages without mutable cleanup state."""
    campaign: Any = SimpleNamespace(
        campaign_id="campaign-id",
        campaign_purpose="technical_runtime_smoke",
        campaign_digest="a" * 64,
        profile=SimpleNamespace(id="steady_flow"),
        batches=[SimpleNamespace(batch_id="batch-id")],
        source_files=(Path("mutable.yaml"),),
        dataset_packages=({"dataset_name": "required"},),
    )
    terminal = {
        "batches": [{"batch_id": "batch-id"}],
        "git_commit": "b" * 40,
        "slurm_job_ids": ["11"],
    }
    pending = {
        "dataset_ids": ["required"],
        "workflow_gate_sha256": "c" * 64,
        "cleanup_requested": True,
        "cpu_cleanup_complete": {"status": "pending", "evidence": None},
    }
    complete = {
        **pending,
        "cpu_cleanup_complete": {
            "status": "complete",
            "evidence": {"receipt_sha256": "d" * 64},
        },
        "cpu_bytes_reclaimed": 10_000,
    }

    assert generation.smoke._campaign_record(
        "run-id",
        campaign,
        terminal,
        pending,
    ) == generation.smoke._campaign_record(
        "run-id",
        campaign,
        terminal,
        complete,
    )

    additive: Any = SimpleNamespace(
        **{
            **campaign.__dict__,
            "source_files": (Path("mutated.yaml"),),
            "dataset_packages": (
                {"dataset_name": "required"},
                {"dataset_name": "unrelated-extension"},
            ),
        }
    )
    assert generation.smoke._source_binding((campaign,)) == generation.smoke._source_binding((additive,))
    workflow_with_extension_metadata = {
        **pending,
        "dataset_package_hashes": [
            {
                "dataset_id": "required",
                "manifest_sha256": "e" * 64,
                "payload_sha256": "f" * 64,
            }
        ],
        "package_inspection_results": [{"status": "valid"}],
        "loader_smoke_results": [{"workers_0": "pass", "workers_2": "pass"}],
        "current_additive_packages": ["unrelated-extension"],
    }
    required = generation.smoke._dataset_records((workflow_with_extension_metadata,))
    assert [record["dataset_id"] for record in required] == ["required"]


def test_real_smoke_diagnostics_classify_bounded_json_paths(tmp_path: Path) -> None:
    """Report each receipt ownership category without unbounded payload output."""
    expected = generation.smoke._json_normalize(_paired_receipt_payload("2026-08-19T00:00:00+00:00"))
    expected["receipt_digest"] = generation.smoke.common.serialization.canonical_json_sha256(expected)
    receipt = copy.deepcopy(expected)
    receipt["recorded_at"] = "2026-08-20T00:00:00+00:00"
    receipt["source_binding"]["bundle_digest"] = "0" * 64
    receipt["templates"]["steady_flow"] = {"sha256": "1" * 64}
    receipt["cases"][0]["hdf5"]["identity"]["available_learning_views"] = ["changed"]
    receipt["dataset_packages"][0]["payload_sha256"] = "2" * 64
    receipt["campaigns"][0]["workflow_gate_sha256"] = "3" * 64
    receipt["cpu_source_retention"]["steady"]["cleanup_requested"] = False
    receipt["slurm"]["case_job_ids"] = ["2"]
    receipt["mass_balance"] = {"changed": True}
    receipt["cases"].extend(copy.deepcopy(receipt["cases"][0]) for _ in range(generation.smoke._MAX_MISMATCH_PATHS))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    diagnostics = generation.smoke._receipt_diagnostics(
        receipt_path,
        receipt,
        expected,
    )

    expected_categories = {
        "scientific_source_binding",
        "template_binding",
        "case_artifact_binding",
        "dataset_package_binding",
        "workflow_gate_binding",
        "cpu_retention_state",
        "timestamps",
        "job_ids",
        "other_lifecycle_metadata",
    }
    assert set(diagnostics["mismatch_categories"]) == expected_categories
    assert diagnostics["difference_count"] > generation.smoke._MAX_MISMATCH_PATHS
    assert diagnostics["differences_truncated"] is True
    assert diagnostics["repair_safe"] is True
    assert len(diagnostics["differing_json_paths"]) == generation.smoke._MAX_MISMATCH_PATHS


def test_interrupted_real_smoke_replacement_remains_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave stale canonical bytes and a validated replacement after interruption."""
    storage = tmp_path / "storage"
    storage.mkdir()

    def build_payload(
        _steady: str,
        _transient: str,
        *,
        storage: Path,
        recorded_at: str,
        comsol_version_output: str,
    ) -> dict[str, Any]:
        return _paired_receipt_payload(recorded_at)

    monkeypatch.setattr(generation.smoke, "_build_payload", build_payload)
    path = generation.smoke.finalize_real_smoke(
        "steady-smoke",
        "transient-smoke",
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
        storage_root=storage,
    )
    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["cases"][0]["hdf5"]["identity"]["available_learning_views"] = ["stale"]
    path.write_text(json.dumps(stale), encoding="utf-8")
    stale_bytes = path.read_bytes()

    atomic_write = generation.smoke.common.serialization.atomic_write_json
    interrupt_canonical = True

    def interrupt_after_link(
        destination: Path | str,
        payload: Any,
        *,
        indent: int = 2,
    ) -> Path:
        target = Path(destination)
        if interrupt_canonical and target == path and (path.parent / "receipt_repairs").exists():
            raise OSError("synthetic interruption before canonical replacement")
        return atomic_write(target, payload, indent=indent)

    monkeypatch.setattr(
        generation.smoke.common.serialization,
        "atomic_write_json",
        interrupt_after_link,
    )
    with pytest.raises(OSError, match="synthetic interruption"):
        generation.smoke.finalize_real_smoke(
            "steady-smoke",
            "transient-smoke",
            comsol_version_output="COMSOL Multiphysics 6.4.0.293",
            storage_root=storage,
        )
    assert path.read_bytes() == stale_bytes
    repairs = path.parent / "receipt_repairs" / path.stem
    replacements = list((repairs / "replacements").glob("*.json"))
    links = list((repairs / "links").glob("*.json"))
    assert len(replacements) == 1
    assert len(links) == 1

    interrupt_canonical = False
    generation.smoke.finalize_real_smoke(
        "steady-smoke",
        "transient-smoke",
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
        storage_root=storage,
    )
    assert generation.smoke.validate_real_smoke_receipt(
        path,
        storage_root=storage,
    )["cases"][0]["hdf5"]["identity"]["available_learning_views"] == ["field"]


def test_real_smoke_repair_history_survives_a_later_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every historical link valid when current evidence changes again."""
    storage = tmp_path / "storage"
    storage.mkdir()
    current_view = ["field"]

    def build_payload(
        _steady: str,
        _transient: str,
        *,
        storage: Path,
        recorded_at: str,
        comsol_version_output: str,
    ) -> dict[str, Any]:
        payload = _paired_receipt_payload(recorded_at)
        payload["cases"][0]["hdf5"]["identity"]["available_learning_views"] = tuple(current_view)
        return payload

    monkeypatch.setattr(generation.smoke, "_build_payload", build_payload)
    path = generation.smoke.finalize_real_smoke(
        "steady-smoke",
        "transient-smoke",
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
        storage_root=storage,
    )
    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["cases"][0]["hdf5"]["identity"]["available_learning_views"] = ["stale"]
    path.write_text(json.dumps(stale), encoding="utf-8")
    generation.smoke.finalize_real_smoke(
        "steady-smoke",
        "transient-smoke",
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
        storage_root=storage,
    )

    current_view[:] = ["field-next"]
    generation.smoke.finalize_real_smoke(
        "steady-smoke",
        "transient-smoke",
        comsol_version_output="COMSOL Multiphysics 6.4.0.293",
        storage_root=storage,
    )

    status = generation.smoke.validate_current_real_smoke_receipts(storage_root=storage)
    expected_history_count = 2
    assert len(status["valid_current_receipts"]) == 1
    assert len(status["historical_stale_receipts"]) == expected_history_count
    assert status["unrecoverable_invalid_receipts"] == []


def test_compatible_completed_smoke_run_reuses_validated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select a scientifically identical completed child without new solves."""
    storage = tmp_path / "storage"
    run_id = "steady_flow_technical_smoke_v1__0123456789abcdef"
    run_directory = storage / "01_generation/meta/campaigns" / run_id
    run_directory.mkdir(parents=True)
    (run_directory / generation.workflow.ALL_WORKFLOW_RECEIPT_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    config_path = generation.smoke.common.paths.get_project_root() / "configs/generation/campaigns/steady_flow/technical_smoke.yaml"
    campaign = generation.cases.config.load_campaign_config(
        config_path,
        require_executable=True,
    )
    manifest = {
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "state": "complete",
    }
    batch = campaign.batches[0]
    cases = tuple(_validated_case_summary(batch, case_index) for case_index in batch.case_indices)
    monkeypatch.setattr(
        generation.smoke.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        generation.smoke,
        "_validate_campaign",
        lambda *_args, **_kwargs: (
            campaign,
            {"git_commit": "a" * 40},
            {"cpu_cleanup_complete": {"status": "skipped_by_request"}},
            cases,
        ),
    )

    result = generation.smoke.find_compatible_completed_technical_smoke_run(
        config_path,
        storage_root=storage,
    )

    assert result["status"] == "compatible_complete"
    assert result["campaign_run_id"] == run_id
    assert result["cpu_source_state"] == "skipped_by_request"


def test_compatible_smoke_discovery_repairs_then_revalidates_completed_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revalidate all artifacts after retained-export HDF5 reconstruction."""
    storage = tmp_path / "storage"
    run_id = "steady_flow_technical_smoke_v1__abcdef0123456789"
    run_directory = storage / "01_generation/meta/campaigns" / run_id
    run_directory.mkdir(parents=True)
    (run_directory / generation.workflow.ALL_WORKFLOW_RECEIPT_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    config_path = generation.smoke.common.paths.get_project_root() / "configs/generation/campaigns/steady_flow/technical_smoke.yaml"
    campaign = generation.cases.config.load_campaign_config(
        config_path,
        require_executable=True,
    )
    manifest = {
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "state": "complete",
    }
    batch = campaign.batches[0]
    cases = tuple(_validated_case_summary(batch, case_index) for case_index in batch.case_indices)
    validation_phases: list[str] = []
    repair_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        generation.smoke.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )

    def validate_after_repair(*_args: Any, **_kwargs: Any) -> Any:
        validation_phases.append("initial" if not validation_phases else "post-repair")
        if len(validation_phases) == 1:
            message = "synthetic corrupt case.h5"
            raise ValueError(message)
        return (
            campaign,
            {"git_commit": "a" * 40},
            {"cpu_cleanup_complete": {"status": "skipped_by_request"}},
            cases,
        )

    def repair_case(candidate: Any, case_index: int, **_kwargs: Any) -> dict[str, Any]:
        repair_calls.append((candidate.batch_id, case_index))
        case_id = candidate.case_id(case_index)
        return {
            "status": "complete",
            "batch_id": candidate.batch_id,
            "case_id": case_id,
            "receipt": f"reconstructions/{candidate.batch_id}/{case_id}.json",
        }

    monkeypatch.setattr(generation.smoke, "_validate_campaign", validate_after_repair)
    monkeypatch.setattr(
        generation.smoke.campaign_runtime,
        "validate_transferred_campaign",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        generation.smoke.runtime_service,
        "repair_completed_case_hdf5_from_retained_exports",
        repair_case,
    )

    result = generation.smoke.find_compatible_completed_technical_smoke_run(
        config_path,
        storage_root=storage,
    )

    expected_calls = [(candidate.batch_id, case_index) for candidate in campaign.batches for case_index in candidate.case_indices]
    assert validation_phases == ["initial", "post-repair"]
    assert repair_calls == expected_calls
    assert result["status"] == "compatible_complete"
    assert len(result["hdf5_reconstructions"]) == len(expected_calls)


def test_compatible_smoke_discovery_marks_invalid_transfer_as_repairable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve completed science while requiring canonical transfer recovery."""
    storage = tmp_path / "storage"
    run_id = "steady_flow_technical_smoke_v1__1234567890abcdef"
    run_directory = storage / "01_generation/meta/campaigns" / run_id
    run_directory.mkdir(parents=True)
    (run_directory / generation.workflow.ALL_WORKFLOW_RECEIPT_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    config_path = generation.smoke.common.paths.get_project_root() / "configs/generation/campaigns/steady_flow/technical_smoke.yaml"
    campaign = generation.cases.config.load_campaign_config(
        config_path,
        require_executable=True,
    )
    manifest = {
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "state": "complete",
    }
    monkeypatch.setattr(
        generation.smoke.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )

    def invalid_workflow(*_args: Any, **_kwargs: Any) -> Any:
        message = "Transfer completion receipt or GPU publication is invalid"
        raise ValueError(message)

    monkeypatch.setattr(generation.smoke, "_validate_campaign", invalid_workflow)
    monkeypatch.setattr(
        generation.smoke.campaign_runtime,
        "validate_transferred_campaign",
        invalid_workflow,
    )

    result = generation.smoke.find_compatible_completed_technical_smoke_run(
        config_path,
        storage_root=storage,
    )

    assert result["status"] == "compatible_repairable"
    assert result["campaign_run_id"] == run_id
    assert "transfer" in result["error"]


def test_compatible_smoke_discovery_fails_closed_for_matching_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse to hide corrupt host evidence for matching Smoke science."""
    storage = tmp_path / "storage"
    run_id = "steady_flow_technical_smoke_v1__fedcba9876543210"
    run_directory = storage / "01_generation/meta/campaigns" / run_id
    run_directory.mkdir(parents=True)
    (run_directory / generation.workflow.ALL_WORKFLOW_RECEIPT_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    config_path = generation.smoke.common.paths.get_project_root() / "configs/generation/campaigns/steady_flow/technical_smoke.yaml"
    campaign = generation.cases.config.load_campaign_config(
        config_path,
        require_executable=True,
    )
    manifest = {
        "campaign_id": campaign.campaign_id,
        "campaign_digest": campaign.campaign_digest,
        "selected_batch_names": [batch.batch_name for batch in campaign.batches],
        "state": "complete",
    }
    monkeypatch.setattr(
        generation.smoke.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )

    def reject_corrupt_candidate(*_args: Any, **_kwargs: Any) -> Any:
        message = "synthetic corrupt case.h5"
        raise ValueError(message)

    monkeypatch.setattr(
        generation.smoke,
        "_validate_campaign",
        reject_corrupt_candidate,
    )
    monkeypatch.setattr(
        generation.smoke.campaign_runtime,
        "validate_transferred_campaign",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        generation.smoke.runtime_service,
        "repair_completed_case_hdf5_from_retained_exports",
        reject_corrupt_candidate,
    )

    with pytest.raises(RuntimeError, match="cannot be safely reused") as caught:
        generation.smoke.find_compatible_completed_technical_smoke_run(
            config_path,
            storage_root=storage,
        )
    assert run_id in str(caught.value)
    assert "synthetic corrupt case.h5" in str(caught.value)


@pytest.mark.parametrize(
    ("version_output", "expected"),
    [
        ("COMSOL Multiphysics 6.4.0.293", "6.4.0.293"),
        ("COMSOL 6.4.0", "6.4.0"),
    ],
)
def test_exact_comsol_version_parser(version_output: str, expected: str) -> None:
    """Keep exact runtime version identity independent of module naming."""
    assert generation.smoke.parse_comsol_exact_version(version_output) == expected
