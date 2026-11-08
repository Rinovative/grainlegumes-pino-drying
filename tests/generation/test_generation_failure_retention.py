"""Attempt retention, stage separation, and hash-admission contracts.

Responsibilities:
  - Check stage-separated failure evidence, one retention policy for every failure stage, license compaction, and hashes

Design principles:
  - Cleanup cannot remove unique scientific output or unadmitted attempt evidence

This module does NOT:
  - Solve failed cases or choose production retention defaults
"""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.publication import generation_publication_attempt as attempt_service
from src.generation.runtime import generation_runtime_batch as runtime_service

if TYPE_CHECKING:
    from pathlib import Path


def _run_id(config: Any) -> str:
    """Bind attempt evidence to the current full source campaign."""
    campaign = generation.cases.config.load_campaign_config(config.source_path)
    return generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)


def _natural_batch_name() -> str:
    """Return the synthetic transient natural-batch selector."""
    return generation.cases.config.build_batch_name(
        "transient_drying",
        "lentil",
        "natural",
    )


def _configured_case(
    generation_config_factory: Any,
    tmp_path: Path,
    *,
    campaign_purpose: str,
    full_retention: bool = False,
) -> tuple[Any, Path, Any]:
    """Prepare canonical raw inputs and one isolated test-owned workspace."""
    config_path, _template = generation_config_factory(
        campaign_purpose=campaign_purpose,
        natural_count=3,
        retain_raw_csv=full_retention,
        retain_solved_model=full_retention,
    )
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=_natural_batch_name(),
    )
    storage = tmp_path / f"{campaign_purpose} storage"
    campaign = generation.cases.config.load_campaign_config(config_path)
    lifecycle.create_campaign(campaign, storage_root=storage)
    generation.cases.input_generation.generate_input_cases(
        config,
        len(config.case_indices),
        storage_root=storage,
    )
    prepared = runtime_service.prepare_case_work_directory(
        config,
        1,
        storage_root=storage,
        work_root=tmp_path / f"{campaign_purpose} work",
    )
    return config, storage, prepared


def _populate_attempt_artifacts(config: Any, prepared: Any) -> tuple[str, ...]:
    """Write tiny exports and runtime evidence."""
    runtime = prepared.runtime_directory
    for name in ("solver.log", "stdout.log", "stderr.log"):
        (runtime / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    common.serialization.atomic_write_json(
        runtime / "timing.json",
        {
            "started_at": "2026-08-18T00:00:00+00:00",
            "ended_at": "2026-08-18T00:00:01+00:00",
            "runtime_s": 1.0,
        },
    )
    common.serialization.atomic_write_json(
        runtime / "execution_provenance.json",
        {"schema_kind": "synthetic_execution", "git_commit": "a" * 40},
    )
    common.serialization.atomic_write_json(
        runtime / "processing_provenance.json",
        {
            "schema_kind": "synthetic_processing",
            "solver_git_commit": "a" * 40,
            "processing_git_commit": "a" * 40,
        },
    )
    common.serialization.atomic_write_json(
        runtime / "status.json",
        {
            "schema_kind": "simulation_case_status",
            "schema_version": generation.publication.storage.STATUS_SCHEMA_VERSION,
            "case_state": "successful",
            "quality_flags": [],
        },
    )
    (runtime / "case.h5").write_bytes(b"synthetic converted payload\n")
    (prepared.work_directory / "solved.mph").write_bytes(b"synthetic solved model\n")
    exported: list[str] = []
    export_root = prepared.exports_directory
    for contract in config.scientific_values["output_contract"]["exports"]:
        pattern = str(contract["pattern"])
        if any(character in pattern for character in "*?[]"):
            pytest.fail("Synthetic attempt tests require exact export patterns.")
        path = export_root / pattern
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {contract['role']} export\n", encoding="utf-8")
        exported.append(path.relative_to(prepared.work_directory).as_posix())
    return tuple(exported)


def _publish(
    config: Any,
    storage: Path,
    prepared: Any,
    *,
    case_state: attempt_service.AttemptCaseState,
    failure_stage: str,
) -> attempt_service.AttemptEvidence:
    """Publish one deterministic synthetic attempt."""
    return attempt_service.publish_case_attempt(
        config,
        1,
        campaign_run_id=_run_id(config),
        case_state=case_state,
        failure_stage=failure_stage,
        reason=f"synthetic {failure_stage} failure",
        solver_git_commit="a" * 40,
        processing_git_commit="a" * 40,
        work_directory=prepared.work_directory,
        storage_root=storage,
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node01",
        exit_code=7,
        timed_out=False,
    )


def test_stage_states_preserve_postsolver_success() -> None:
    """Keep solver, export, conversion, diagnostics, and publication distinct."""
    assert attempt_service.derive_stage_states(
        "exports_failed",
        "exports",
    ) == {
        "solver_state": "succeeded",
        "exports_state": "failed",
        "conversion_state": "not_started",
        "diagnostics_state": "not_started",
        "publication_state": "not_started",
    }
    assert (
        attempt_service.derive_stage_states(
            "conversion_failed",
            "conversion",
        )["solver_state"]
        == "succeeded"
    )
    publication = attempt_service.derive_stage_states(
        "publication_failed",
        "publication",
    )
    assert publication["conversion_state"] == "succeeded"
    assert publication["diagnostics_state"] == "complete"
    assert publication["publication_state"] == "failed"


def test_full_policy_retains_failure_evidence(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Retain the failed source model and exports under the declared full policy."""
    config, storage, prepared = _configured_case(
        generation_config_factory,
        tmp_path,
        campaign_purpose="source_generation",
        full_retention=True,
    )
    exported = _populate_attempt_artifacts(config, prepared)

    attempt = _publish(
        config,
        storage,
        prepared,
        case_state="conversion_failed",
        failure_stage="conversion",
    )

    assert config.execution_values["retention_policy"] == "full"
    assert attempt.payload["retention_policy"] == "full"
    retained = attempt.payload["retained_inventory"]
    assert "payload/model.mph" in retained
    assert "payload/solved.mph" in retained
    assert "payload/runtime/stdout.log" in retained
    assert "payload/runtime/solver.log" not in retained  # the combined copy would repeat the retained stdout and stderr
    assert all(f"payload/{relative}" in retained for relative in exported)
    assert "payload/runtime/case.h5" in retained
    assert not any(path.name == "_SUCCESS" for path in attempt.directory.rglob("*"))


def _publish_oversized_license_attempt(
    config: Any,
    storage: Path,
    prepared: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> attempt_service.AttemptEvidence:
    """Synthesize one pre-cleanup full license payload for compaction tests."""

    def retain_all(
        _config: Any,
        *,
        work_directory: Path | None,
    ) -> tuple[str, tuple[Path, ...], list[str]]:
        assert work_directory is not None
        paths = tuple(
            candidate.relative_to(work_directory)
            for candidate in sorted(work_directory.rglob("*"))
            if candidate.is_file() and not candidate.is_symlink()
        )
        return "full", paths, []

    with monkeypatch.context() as scoped:
        scoped.setattr(attempt_service, "_retention_paths", retain_all)
        return _publish(
            config,
            storage,
            prepared,
            case_state="license_blocked",
            failure_stage="solver",
        )


def _license_wait(attempt: attempt_service.AttemptEvidence) -> dict[str, Any]:
    """Return compact strong license evidence bound to one synthetic license-only attempt."""
    return {
        "campaign_run_id": attempt.payload["campaign_run_id"],
        "batch_id": attempt.payload["batch_id"],
        "case_id": attempt.payload["case_id"],
        "scientific_config_digest": attempt.payload["scientific_config_digest"],
        "classification": generation.runtime.license.TEMPORARY_LICENSE_CAPACITY,
        "feature": "Brinkman Equations (br)",
        "error_code": "-4",
        "matched_signatures": ["License error: -4"],
        "solver_progress_started": False,
        "expected_exports_exist": False,
        "raw_excerpt": ("Could not obtain license for 'Brinkman Equations (br)'.\nLicensed number of users already reached.\nLicense error: -4."),
        "retry_budget_remaining": True,
    }


def test_license_only_full_payload_is_safely_compacted(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reclaim copied full-retention inputs while preserving immutable audit evidence."""
    config, storage, prepared = _configured_case(
        generation_config_factory,
        tmp_path,
        campaign_purpose="source_generation",
    )
    license_text = "Could not obtain license for 'Brinkman Equations (br)'.\nLicensed number of users already reached.\nLicense error: -4."
    prepared.runtime_directory.joinpath("stdout.log").write_text(license_text, encoding="utf-8")
    attempt = _publish_oversized_license_attempt(
        config,
        storage,
        prepared,
        monkeypatch,
    )
    original_receipt = attempt.receipt_path.read_bytes()
    assert any(path.name == "case.json" for path in (attempt.directory / "payload").rglob("*"))

    audit = attempt_service.compact_license_only_attempt_payload(
        config,
        1,
        _run_id(config),
        _license_wait(attempt),
        storage_root=storage,
    )

    assert audit is not None
    assert audit["status"] == "complete"
    assert audit["reclaimed_bytes"] > 0
    assert attempt.receipt_path.read_bytes() == original_receipt
    assert not (attempt.directory / "payload").exists()
    assert attempt_service.load_attempt(attempt.directory).payload == attempt.payload


def test_license_cleanup_refuses_unique_scientific_output(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a license-only attempt intact whenever canonical HDF5 output is present."""
    config, storage, prepared = _configured_case(
        generation_config_factory,
        tmp_path,
        campaign_purpose="source_generation",
    )
    prepared.runtime_directory.joinpath("stdout.log").write_text(
        "Could not obtain license for 'Brinkman Equations (br)'. Licensed number of users already reached. License error: -4.",
        encoding="utf-8",
    )
    prepared.runtime_directory.joinpath("case.h5").write_bytes(b"unique scientific output")
    attempt = _publish_oversized_license_attempt(
        config,
        storage,
        prepared,
        monkeypatch,
    )

    assert (
        attempt_service.compact_license_only_attempt_payload(
            config,
            1,
            _run_id(config),
            _license_wait(attempt),
            storage_root=storage,
        )
        is None
    )
    assert (attempt.directory / "payload/runtime/case.h5").read_bytes() == b"unique scientific output"


def test_attempt_admission_requires_exact_version_one_schema(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject unsupported versions, unknown fields, and false stage claims."""
    config, storage, prepared = _configured_case(
        generation_config_factory,
        tmp_path,
        campaign_purpose="source_generation",
    )
    _populate_attempt_artifacts(config, prepared)
    attempt = _publish(
        config,
        storage,
        prepared,
        case_state="failed",
        failure_stage="solver",
    )
    original = json.loads(attempt.receipt_path.read_text(encoding="utf-8"))

    assert {
        attempt_service.ATTEMPT_SCHEMA_VERSION,
        attempt_service.CLEANUP_SCHEMA_VERSION,
        generation.publication.storage.STATUS_SCHEMA_VERSION,
    } == {1}

    malformed_receipts = (
        {**original, "schema_version": 0},
        {**original, "unexpected_field": "not admitted"},
        {**original, "solver_state": "succeeded"},
    )
    for malformed in malformed_receipts:
        common.serialization.atomic_write_json(attempt.receipt_path, malformed)
        with pytest.raises(ValueError, match=r"schema|stage states"):
            attempt_service.load_attempt(attempt.directory)
    common.serialization.atomic_write_json(attempt.receipt_path, original)
    assert attempt_service.load_attempt(attempt.directory).payload == original


def test_production_solver_failure_is_compact_and_hash_admitted(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omit large payloads and bound logs for a Production solver failure."""
    config, storage, prepared = _configured_case(
        generation_config_factory,
        tmp_path,
        campaign_purpose="source_generation",
    )
    exported = _populate_attempt_artifacts(config, prepared)
    maximum_log_bytes = 128
    monkeypatch.setattr(
        attempt_service,
        "_MAX_RETAINED_RUNTIME_LOG_BYTES",
        maximum_log_bytes,
    )
    for name in ("solver.log", "stdout.log", "stderr.log"):
        (prepared.runtime_directory / name).write_bytes(
            b"HEAD-" + b"x" * maximum_log_bytes + b"-TAIL\n",
        )
    (prepared.runtime_directory / "solver_steps.csv").write_text("step,time_s\n1,1.6\n", encoding="utf-8")

    attempt = _publish(
        config,
        storage,
        prepared,
        case_state="failed",
        failure_stage="solver",
    )

    assert attempt.payload["retention_policy"] == "compact"
    retained = attempt.payload["retained_inventory"]
    assert "payload/runtime/stdout.log" in retained
    assert "payload/model.mph" not in retained
    assert "payload/solved.mph" not in retained
    assert all(f"payload/{relative}" not in retained for relative in exported)
    for name in ("stdout.log", "stderr.log"):
        retained_log = (attempt.directory / "payload/runtime" / name).read_bytes()
        assert len(retained_log) <= maximum_log_bytes
        assert retained_log.startswith(b"HEAD-")
        assert retained_log.endswith(b"-TAIL\n")
        assert b"retained log middle omitted" in retained_log
        complete = attempt.directory / "payload/runtime" / f"{name}.gz"
        assert gzip.decompress(complete.read_bytes()) == b"HEAD-" + b"x" * maximum_log_bytes + b"-TAIL\n"
        assert f"payload/runtime/{name}.gz" in retained
    assert "payload/runtime/solver_steps.csv" in retained
    stdout_log = attempt.directory / "payload/runtime/stdout.log"
    stdout_log.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact identity"):
        attempt_service.load_attempt(attempt.directory)


def test_solver_step_history_is_written_from_the_complete_stdout(generation_config_factory: Any, tmp_path: Path) -> None:
    """The case job keeps every transient time step of its stdout as a CSV in its runtime directory."""
    _config, _storage, prepared = _configured_case(generation_config_factory, tmp_path, campaign_purpose="source_generation")
    (prepared.runtime_directory / "stdout.log").write_text(
        "Step        Time    Stepsize      Res  Jac  Sol Order Tfail NLfail   LinErr   LinRes\n"
        "   1      1.6355     0.42589       18    6   18     1     0      0  4.8e-16  4.6e-16\n"
        "   2      2.0614     0.42589       20    7   20     1     0      1  4.8e-16  4.6e-16\n",
        encoding="utf-8",
    )

    path = runtime_service._write_solver_step_history(prepared)

    assert path == prepared.runtime_directory / "solver_steps.csv"
    assert path is not None
    assert path.read_text(encoding="utf-8").splitlines() == [
        "step,time_s,step_size_s,residual_evaluations,jacobian_evaluations,linear_solves,order,time_failures,nonlinear_failures,linear_error,linear_residual",
        "1,1.6355,0.42589,18,6,18,1,0,0,4.8e-16,4.6e-16",
        "2,2.0614,0.42589,20,7,20,1,0,1,4.8e-16,4.6e-16",
    ]


@pytest.mark.parametrize(
    ("case_state", "failure_stage"),
    [("conversion_failed", "conversion"), ("publication_failed", "publication")],
)
def test_conversion_and_publication_failures_follow_the_solver_failure_compaction(
    generation_config_factory: Any,
    tmp_path: Path,
    case_state: attempt_service.AttemptCaseState,
    failure_stage: str,
) -> None:
    """A post-solver failure keeps the compact logs and provenance and records its exports, converted HDF5 and models as omitted."""
    config, storage, prepared = _configured_case(
        generation_config_factory,
        tmp_path,
        campaign_purpose="source_generation",
    )
    exported = _populate_attempt_artifacts(config, prepared)

    attempt = _publish(config, storage, prepared, case_state=case_state, failure_stage=failure_stage)

    assert attempt.payload["retention_policy"] == "compact"
    retained = attempt.payload["retained_inventory"]
    assert {"payload/runtime/stdout.log", "payload/runtime/status.json", "payload/runtime/processing_provenance.json"} <= set(retained)
    omitted_payload = {"payload/model.mph", "payload/solved.mph", "payload/runtime/case.h5", *(f"payload/{relative}" for relative in exported)}
    assert omitted_payload.isdisjoint(retained)
    assert set(attempt.payload["intentionally_omitted_artifacts"]) == {"model.mph", "solved.mph", "runtime/case.h5", *exported}
    assert attempt.payload["unexpectedly_missing_artifacts"] == []
    assert attempt_service.load_attempt(attempt.directory).payload == attempt.payload
