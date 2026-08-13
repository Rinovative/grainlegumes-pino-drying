# ruff: noqa: S101
"""Failed-case retention, diagnostic policy, and durable integrity contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import common, generation
from src.generation.publication import generation_publication_storage as storage_service
from src.generation.runtime import generation_runtime_batch as runtime_service
from src.generation.runtime import generation_runtime_diagnostics as diagnostics_service


def _transient_config(
    generation_config_factory: Any,
    fake_comsol: Path,
    *,
    retain_raw_csv: bool = True,
    retain_solved_model: bool = True,
) -> Any:
    """Return one compact resolved transient Technical-Smoke batch."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        executable=fake_comsol,
        retain_raw_csv=retain_raw_csv,
        retain_solved_model=retain_solved_model,
    )
    return generation.cases.config.load_generation_config(
        config_path,
        only_batch="transient_drying__lentil__natural",
    )


def _purpose_config(
    config: Any,
    *,
    campaign_purpose: str,
    retain_raw_csv: bool,
    retain_solved_model: bool,
) -> Any:
    """Return a test-only policy projection without mutating its source config."""
    return replace(
        config,
        scientific_values={
            **config.scientific_values,
            "campaign_purpose": campaign_purpose,
        },
        execution_values={
            **config.execution_values,
            "retention": {
                "retain_raw_csv": retain_raw_csv,
                "retain_solved_model": retain_solved_model,
            },
        },
    )


def _configured_export_relatives(config: Any) -> set[str]:
    """Return retained paths selected by the resolved output contract."""
    return {f"exports/{contract['pattern']}" for contract in config.scientific_values["output_contract"]["exports"]}


def _force_conversion_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject conversion after the synthetic solver has produced valid exports."""

    def reject_conversion(*_args: Any, **_kwargs: Any) -> None:
        message = "synthetic post-export conversion failure"
        raise ValueError(message)

    monkeypatch.setattr(
        storage_service,
        "convert_exports_to_hdf5",
        reject_conversion,
    )


def test_technical_smoke_conversion_failure_retains_and_diagnoses_before_cleanup(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain configured bytes and raw-output diagnostics before scratch cleanup."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    _force_conversion_failure(monkeypatch)

    with pytest.raises(
        generation.runtime.CaseExecutionError,
        match="synthetic post-export conversion failure",
    ) as raised:
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=16,
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    assert not raised.value.work_directory.exists()
    receipt_path = generation.runtime.case_failure_path(
        config,
        1,
        storage_root=storage,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_kind"] == "simulation_case_failure"
    assert receipt["schema_version"] == 1
    retained = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    expected = {
        "case.json",
        "fields.csv",
        "scalars.csv",
        "schedule.csv",
        "solved.mph",
        "diagnostics/initial_state_diagnostic.json",
        "diagnostics/initial_state_diagnostic.csv",
        *_configured_export_relatives(config),
    }
    assert expected.issubset(receipt["retained_artifacts"])
    assert all((retained / relative).is_file() for relative in expected)
    assert receipt["failure_stage"] == "conversion"
    assert receipt["error"]["message"] == "synthetic post-export conversion failure"
    assert receipt["failure_diagnostics"]["transient_initial_state"]["status"] == "complete"
    assert receipt["scratch_cleanup"]["status"] == "complete"
    diagnostic = json.loads((retained / "diagnostics/initial_state_diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["validator"]["rtol"] == config.scientific_values["storage"]["float32_rtol"]
    assert diagnostic["validator"]["atol"] == config.scientific_values["storage"]["float32_atol"]
    assert diagnostic["diagnostic_classification"] == "approximately_canonical"
    assert generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=storage,
    )


def test_technical_smoke_failure_before_exports_records_no_fabricated_diagnostic(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the original solver failure and mark unavailable diagnostic inputs."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")

    with pytest.raises(generation.runtime.CaseExecutionError) as raised:
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=16,
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    receipt = json.loads(
        generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        ).read_text(encoding="utf-8")
    )
    assert receipt["error"]["message"] == "COMSOL case exited with status 7."
    assert receipt["failure_diagnostics"]["transient_initial_state"]["status"] == "inputs_unavailable"
    assert not any(relative.startswith("diagnostics/") for relative in receipt["retained_artifacts"])
    assert not raised.value.work_directory.exists()


@pytest.mark.parametrize(
    ("purpose", "retain_raw_csv", "retain_solved_model"),
    [
        ("family_generalization", False, False),
        ("pilot_check", True, True),
    ],
)
def test_non_smoke_failure_never_runs_quantitative_diagnostic(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purpose: str,
    retain_raw_csv: bool,
    retain_solved_model: bool,
) -> None:
    """Keep Production compact and Pilot free of transient quantitative diagnosis."""
    base = _transient_config(generation_config_factory, fake_comsol)
    config = _purpose_config(
        base,
        campaign_purpose=purpose,
        retain_raw_csv=retain_raw_csv,
        retain_solved_model=retain_solved_model,
    )
    work = tmp_path / purpose
    (work / "exports").mkdir(parents=True)
    (work / "case.json").write_text("{}\n", encoding="utf-8")
    (work / "fields.csv").write_text("x\n0\n", encoding="utf-8")
    (work / "scalars.csv").write_text("name,value\na,1\n", encoding="utf-8")
    (work / "schedule.csv").write_text("t\n0\n", encoding="utf-8")
    (work / "solved.mph").write_bytes(b"synthetic solved model\n")
    for contract in config.scientific_values["output_contract"]["exports"]:
        (work / "exports" / contract["pattern"]).write_text(
            "malformed but retained\n",
            encoding="utf-8",
        )
    called = {"diagnostic": False}

    def reject_diagnostic(*_args: Any, **_kwargs: Any) -> None:
        called["diagnostic"] = True
        message = "diagnostic must not run"
        raise AssertionError(message)

    monkeypatch.setattr(
        diagnostics_service,
        "write_initial_state_diagnostic",
        reject_diagnostic,
    )
    receipt_path = generation.runtime.record_case_failure(
        config,
        1,
        ValueError("authoritative failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=work,
        storage_root=tmp_path / f"{purpose}-storage",
        failure_stage="conversion",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert called["diagnostic"] is False
    assert receipt["failure_diagnostics"] == {}
    if purpose == "family_generalization":
        assert receipt["retained_artifacts"] == {}
        assert not generation.runtime.case_failure_artifacts_directory(
            config,
            1,
            storage_root=tmp_path / f"{purpose}-storage",
        ).exists()
    else:
        assert "solved.mph" in receipt["retained_artifacts"]
        assert _configured_export_relatives(config).issubset(receipt["retained_artifacts"])


def test_diagnostic_failure_cannot_mask_original_case_failure(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist secondary diagnostic failure while re-raising conversion failure."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    _force_conversion_failure(monkeypatch)

    def fail_diagnostic(
        *_args: Any,
        output_directory: Path,
        **_kwargs: Any,
    ) -> None:
        common.serialization.atomic_write_json(
            output_directory / "initial_state_diagnostic.json",
            {"partial": True},
        )
        message = "synthetic diagnostic failure"
        raise RuntimeError(message)

    monkeypatch.setattr(
        diagnostics_service,
        "write_initial_state_diagnostic",
        fail_diagnostic,
    )
    with pytest.raises(
        generation.runtime.CaseExecutionError,
        match="synthetic post-export conversion failure",
    ):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=16,
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    receipt = json.loads(
        generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        ).read_text(encoding="utf-8")
    )
    assert receipt["error"]["message"] == "synthetic post-export conversion failure"
    assert receipt["failure_diagnostics"]["transient_initial_state"] == {
        "status": "failed",
        "error": "RuntimeError: synthetic diagnostic failure",
        "json_relative_path": None,
        "csv_relative_path": None,
    }


def test_retention_failure_is_secondary_and_scratch_still_cleans(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the original case error when durable raw staging itself fails."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    _force_conversion_failure(monkeypatch)

    def fail_copy(*_args: Any, **_kwargs: Any) -> None:
        message = "synthetic retention copy failure"
        raise OSError(message)

    monkeypatch.setattr(runtime_service, "_copy_failure_file", fail_copy)
    with pytest.raises(
        generation.runtime.CaseExecutionError,
        match="synthetic post-export conversion failure",
    ) as raised:
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=16,
            storage_root=storage,
            work_root=tmp_path / "work",
        )

    receipt = json.loads(
        generation.runtime.case_failure_path(
            config,
            1,
            storage_root=storage,
        ).read_text(encoding="utf-8")
    )
    assert receipt["error"]["message"] == "synthetic post-export conversion failure"
    assert receipt["retained_artifacts"] == {}
    assert receipt["failure_diagnostics"] == {}
    assert receipt["retention_error"] == {
        "type": "OSError",
        "message": "synthetic retention copy failure",
        "prior_artifacts_preserved": False,
    }
    assert receipt["scratch_cleanup"]["status"] == "complete"
    assert not raised.value.work_directory.exists()


def test_solved_only_retention_does_not_inspect_unsafe_raw_export_root(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Keep independently enabled solved retention decoupled from raw exports."""
    base = _transient_config(generation_config_factory, fake_comsol)
    config = _purpose_config(
        base,
        campaign_purpose="family_generalization",
        retain_raw_csv=False,
        retain_solved_model=True,
    )
    work = tmp_path / "work"
    work.mkdir()
    foreign = tmp_path / "foreign-exports"
    foreign.mkdir()
    (work / "exports").symlink_to(foreign, target_is_directory=True)
    (work / "solved.mph").write_bytes(b"solved\n")

    receipt_path = generation.runtime.record_case_failure(
        config,
        1,
        ValueError("synthetic solved-only failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=work,
        storage_root=tmp_path / "storage",
        failure_stage="solver",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt["retained_artifacts"]) == {"solved.mph"}
    assert receipt["retention_error"] is None
    assert receipt["failure_diagnostics"] == {}


def _record_mock_diagnostic_failure(
    config: Any,
    storage: Path,
    work: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Record one complete small artifact set for integrity mutation tests."""
    (work / "exports").mkdir(parents=True)
    for name in ("case.json", "fields.csv", "scalars.csv", "schedule.csv"):
        (work / name).write_text("{}\n", encoding="utf-8")
    (work / "solved.mph").write_bytes(b"model\n")
    for contract in config.scientific_values["output_contract"]["exports"]:
        (work / "exports" / contract["pattern"]).write_text(
            "export\n",
            encoding="utf-8",
        )

    def write_diagnostic(
        *_args: Any,
        output_directory: Path,
        **_kwargs: Any,
    ) -> Any:
        json_path = common.serialization.atomic_write_json(
            output_directory / "initial_state_diagnostic.json",
            {"diagnostic": True},
        )
        csv_path = common.serialization.atomic_write_text(
            output_directory / "initial_state_diagnostic.csv",
            "x,y\n0,0\n",
        )
        return SimpleNamespace(json_path=json_path, csv_path=csv_path)

    monkeypatch.setattr(
        diagnostics_service,
        "write_initial_state_diagnostic",
        write_diagnostic,
    )
    return generation.runtime.record_case_failure(
        config,
        1,
        ValueError("synthetic failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=work,
        storage_root=storage,
        failure_stage="conversion",
    )


def test_retry_staging_failure_preserves_prior_immutable_artifact_set(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the previous complete set until replacement staging succeeds."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    _record_mock_diagnostic_failure(config, storage, work, monkeypatch)
    retained = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    prior = {path.relative_to(retained).as_posix(): common.serialization.file_sha256(path) for path in retained.rglob("*") if path.is_file()}

    def fail_copy(*_args: Any, **_kwargs: Any) -> None:
        message = "retry staging failed"
        raise OSError(message)

    monkeypatch.setattr(runtime_service, "_copy_failure_file", fail_copy)
    receipt_path = generation.runtime.record_case_failure(
        config,
        1,
        ValueError("retry case failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=work,
        storage_root=storage,
        failure_stage="conversion",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    observed = {path.relative_to(retained).as_posix(): common.serialization.file_sha256(path) for path in retained.rglob("*") if path.is_file()}
    assert observed == prior
    assert set(receipt["retained_artifacts"]) == set(prior)
    assert receipt["retention_error"] == {
        "type": "OSError",
        "message": "retry staging failed",
        "prior_artifacts_preserved": True,
    }
    assert generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=storage,
    )


def test_final_directory_replace_failure_restores_prior_artifacts(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore the prior immutable set when final directory publication fails."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    work = tmp_path / "work"
    _record_mock_diagnostic_failure(config, storage, work, monkeypatch)
    target = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    prior = {path.relative_to(target).as_posix(): common.serialization.file_sha256(path) for path in target.rglob("*") if path.is_file()}
    original_replace = Path.replace

    def fail_final_replace(source: Path, destination: Path) -> Path:
        if Path(destination) == target and source != target and ".previous." not in source.name:
            message = "synthetic final directory replace failure"
            raise OSError(message)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_final_replace)
    receipt_path = generation.runtime.record_case_failure(
        config,
        1,
        ValueError("retry case failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=work,
        storage_root=storage,
        failure_stage="conversion",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    observed = {path.relative_to(target).as_posix(): common.serialization.file_sha256(path) for path in target.rglob("*") if path.is_file()}
    assert observed == prior
    assert receipt["retention_error"] == {
        "type": "OSError",
        "message": "synthetic final directory replace failure",
        "prior_artifacts_preserved": True,
    }
    assert generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=storage,
    )


def test_receipt_tampering_cannot_bypass_retention_or_diagnostic_policy(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Re-derive Production policy instead of trusting receipt-declared hashes."""
    base = _transient_config(generation_config_factory, fake_comsol)
    config = _purpose_config(
        base,
        campaign_purpose="family_generalization",
        retain_raw_csv=False,
        retain_solved_model=False,
    )
    storage = tmp_path / "storage"
    receipt_path = generation.runtime.record_case_failure(
        config,
        1,
        ValueError("production failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )
    artifacts = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    (artifacts / "diagnostics").mkdir(parents=True)
    diagnostic_json = common.serialization.atomic_write_json(
        artifacts / "diagnostics/initial_state_diagnostic.json",
        {"tampered": True},
    )
    diagnostic_csv = common.serialization.atomic_write_text(
        artifacts / "diagnostics/initial_state_diagnostic.csv",
        "x,y\n0,0\n",
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["retained_artifacts"] = {
        path.relative_to(artifacts).as_posix(): {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (diagnostic_json, diagnostic_csv)
    }
    payload["failure_diagnostics"] = {
        "transient_initial_state": {
            "status": "complete",
            "error": None,
            "json_relative_path": "diagnostics/initial_state_diagnostic.json",
            "csv_relative_path": "diagnostics/initial_state_diagnostic.csv",
        }
    }
    common.serialization.atomic_write_json(receipt_path, payload)

    with pytest.raises(ValueError, match="diagnostic evidence violates execution policy"):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=storage,
        )

    diagnostic_json.unlink()
    diagnostic_csv.unlink()
    payload["failure_diagnostics"] = {}
    payload["retained_artifacts"] = {}
    solved = artifacts / "solved.mph"
    solved.write_bytes(b"forbidden\n")
    payload["retained_artifacts"]["solved.mph"] = {
        "sha256": common.serialization.file_sha256(solved),
        "size_bytes": solved.stat().st_size,
    }
    common.serialization.atomic_write_json(receipt_path, payload)
    with pytest.raises(ValueError, match="retained artifacts violate resolved retention policy"):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=storage,
        )


def test_forged_retention_error_cannot_admit_production_diagnostics(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Keep a forged retention error from weakening Production policy."""
    base = _transient_config(generation_config_factory, fake_comsol)
    config = _purpose_config(
        base,
        campaign_purpose="family_generalization",
        retain_raw_csv=False,
        retain_solved_model=False,
    )
    storage = tmp_path / "storage"
    receipt_path = generation.runtime.record_case_failure(
        config,
        1,
        ValueError("production failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )
    artifacts = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    (artifacts / "diagnostics").mkdir(parents=True)
    diagnostic_json = common.serialization.atomic_write_json(
        artifacts / "diagnostics/initial_state_diagnostic.json",
        {"forged": True},
    )
    diagnostic_csv = common.serialization.atomic_write_text(
        artifacts / "diagnostics/initial_state_diagnostic.csv",
        "x,y\n0,0\n",
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["retained_artifacts"] = {
        path.relative_to(artifacts).as_posix(): {
            "sha256": common.serialization.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in (diagnostic_json, diagnostic_csv)
    }
    payload["retention_error"] = {
        "type": "OSError",
        "message": "forged",
        "prior_artifacts_preserved": True,
    }
    common.serialization.atomic_write_json(receipt_path, payload)

    with pytest.raises(
        ValueError,
        match="retained artifacts violate resolved retention policy",
    ):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=storage,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "exports/airflow.csv",
        "diagnostics/initial_state_diagnostic.json",
        "diagnostics/initial_state_diagnostic.csv",
    ],
)
def test_retained_failure_mutation_fails_hash_validation(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    """Reject mutations to retained raw exports and both diagnostic formats."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    _record_mock_diagnostic_failure(
        config,
        storage,
        tmp_path / "work",
        monkeypatch,
    )
    retained = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    assert (retained / relative).is_file()
    (retained / relative).write_bytes(b"mutated\n")

    with pytest.raises(ValueError, match="retained-artifact identity is invalid"):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=storage,
        )


def test_retained_failure_symlink_and_unexpected_file_fail_closed(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject symlink replacement and exact-membership expansion."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    _record_mock_diagnostic_failure(
        config,
        storage,
        tmp_path / "work",
        monkeypatch,
    )
    retained = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    diagnostic = retained / "diagnostics/initial_state_diagnostic.json"
    diagnostic.unlink()
    diagnostic.symlink_to(tmp_path / "foreign.json")
    with pytest.raises(ValueError, match="symbolic link"):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=storage,
        )

    diagnostic.unlink()
    common.serialization.atomic_write_json(diagnostic, {"diagnostic": True})
    (retained / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="membership changed"):
        generation.runtime.case_failure_is_recorded(
            config,
            1,
            storage_root=storage,
        )


def test_partial_staging_is_never_accepted_and_success_clears_failure_artifacts(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore partial siblings and clear canonical failure state after recovery."""
    config = _transient_config(generation_config_factory, fake_comsol)
    storage = tmp_path / "storage"
    artifact_target = generation.runtime.case_failure_artifacts_directory(
        config,
        1,
        storage_root=storage,
    )
    partial = artifact_target.parent / "case_0001.partial"
    partial.mkdir(parents=True)
    (partial / "diagnostic.json").write_text("{}\n", encoding="utf-8")
    assert not generation.runtime.case_failure_is_recorded(
        config,
        1,
        storage_root=storage,
    )

    monkeypatch.setenv("FAKE_COMSOL_MODE", "failure")
    with pytest.raises(generation.runtime.CaseExecutionError):
        generation.runtime.run_case(
            config,
            1,
            cores_per_case=16,
            storage_root=storage,
            work_root=tmp_path / "failed-work",
        )
    assert artifact_target.is_dir()

    monkeypatch.setenv("FAKE_COMSOL_MODE", "success")
    outcome = generation.runtime.run_case(
        config,
        1,
        cores_per_case=16,
        storage_root=storage,
        work_root=tmp_path / "successful-work",
    )
    assert outcome.status == "completed"
    assert not generation.runtime.case_failure_path(
        config,
        1,
        storage_root=storage,
    ).exists()
    assert not artifact_target.exists()
