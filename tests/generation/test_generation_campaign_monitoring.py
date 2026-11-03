"""Per-case campaign reconciliation and human monitoring contracts.

Responsibilities:
  - Check reconciliation, capacity categories, terminal evidence, and bounded human summaries
  - Check the driver's live board: its cadence, its estimate and its marks for failed, replaced, and stuck cases
  - Check the per-case time window the status takes from the persisted resolved configuration

Design principles:
  - Unknown scheduler state and license waits stay distinct from scientific failure
  - Summary tests check selection, bounds, and omitted evidence rather than exact layout
  - An estimate without evidence is asserted to read unknown, never a guessed value

This module does NOT:
  - Submit cases or manufacture solver progress
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any

import pytest
from support.fake_slurm import NODE_ID, dispatch_record, open_node_job, submit

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation import generation_campaign_status as status_service
from src.generation.cli import cli_generation
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_stop as stop_service

if TYPE_CHECKING:
    from collections.abc import Mapping


def _synthetic_retry_policy(
    *,
    initial_delay_seconds: float = 20.0,
    maximum_delay_seconds: float = 30.0,
) -> dict[str, Any]:
    """Return one test-owned bounded node licence-wait policy."""
    return {
        "enabled": True,
        "initial_delay_seconds": initial_delay_seconds,
        "maximum_delay_seconds": maximum_delay_seconds,
        "maximum_wait_seconds": 3600.0,
    }


def _patch_synthetic_input_references(
    monkeypatch: pytest.MonkeyPatch,
    campaign: Any,
) -> None:
    """Supply inert current source ownership when a test replaces admission."""
    sources = {
        batch.batch_name: {
            "input_generation_id": f"input-{index:024x}",
            "source_git_commit": "a" * 40,
            "case_indices": list(batch.case_indices),
        }
        for index, batch in enumerate(campaign.batches, start=1)
    }
    references = {batch.batch_name: {case_index: object() for case_index in batch.case_indices} for batch in campaign.batches}
    monkeypatch.setattr(generation.campaign, "_persisted_input_sources", lambda _manifest: sources)
    monkeypatch.setattr(
        generation.campaign,
        "_campaign_input_references",
        lambda _campaign, **_kwargs: references,
    )


def _submission(campaign: Any, task: cluster.CampaignTask, job_id: str, index: int) -> dict[str, Any]:
    """Return one exact synthetic persisted dispatch record of ``task`` in the node job ``job_id``."""
    return {
        "submission_index": index,
        "mode": "initial",
        "recorded_at": f"2026-01-01T00:0{index}:00+00:00",
        "case": {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
        },
        "node_job_token": NODE_ID,
        "job_id": job_id,
        "ticket_seq": index,
        "step_name": generation.campaign._step_name(campaign, task, submission_index=index),
        "attempt_index": 1,
        "status": "dispatched",
        "error": None,
    }


def _active_case(case_id: str, job_id: str, node: str, simulated_time: float) -> dict[str, Any]:
    """Return one active case with available transient raw evidence."""
    return {
        "batch_name": "transient_drying__lentil__natural",
        "batch_id": "batch-id",
        "case_index": int(case_id.rsplit("_", maxsplit=1)[1]),
        "case_id": case_id,
        "state": "active",
        "reason": "RUNNING",
        "submission_count": 1,
        "latest_job_id": job_id,
        "latest_job_name": f"campaign-{job_id}",
        "scheduler_state": "RUNNING",
        "node": node,
        "submit_time": "2026-01-01T00:00:00",
        "start_time": "2026-01-01T00:01:00",
        "elapsed": "00:12:03",
        "temporary_license_retry": None,
        "runtime_progress": {
            "availability": "available",
            "reason": None,
            "phase": "transient_drying",
            "terminal": False,
            "parser_state": "available",
            "comsol_progress_percent": 4.0,
            "simulated_time_seconds": simulated_time,
            "step_index": 220,
            "step_size_seconds": 0.075,
            "order": 2,
            "time_failures": 1,
            "nonlinear_failures": 25,
            "age_seconds": 4.0,
            "stale": False,
        },
    }


def test_resume_reconciles_during_atomic_status_recovery_publication(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one active campaign observable while a worker publishes recovery."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=1,
        max_admission_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    _patch_synthetic_input_references(monkeypatch, campaign)
    monkeypatch.setattr(lifecycle, "campaign_from_frozen_run", lambda *_args, **_kwargs: campaign)
    monkeypatch.setattr(lifecycle, "candidate_records", lambda *_args, **_kwargs: [])
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_directory = tmp_path / "campaign"
    run_directory.mkdir()
    manifest: dict[str, Any] = {
        "campaign_run_id": run_id,
        "campaign_config": str(config_path),
        "execution_config_digest": generation.campaign._execution_config_digest(campaign),
        "git_commit": "a" * 40,
        "slurm_job_ids": [],
        "node_jobs": [],
        "node_job_intent": None,
        "scheduler_log_directory": str(run_directory / "scheduler"),
        "submission_config": {
            "cores_per_case": 16,
            "max_admission_cases": 1,
            "node_jobs": 2,
            "concurrent_cases_per_node": 4,
            "maximum_failed_cases": 5,
            "license_wait": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "admission_reservations": [],
        "submissions": [],
        "state": "active",
        "remote_storage_root": str(tmp_path),
        "campaign_meta_directory": str(run_directory),
    }
    node_job = open_node_job(manifest, tmp_path, job_id="638288")
    step_name = dispatch_record(manifest, campaign, task, node_job)["step_name"]

    def scheduler_output(command: list[str]) -> tuple[str, str | None]:
        """Answer the live queue with the running node job and its running case step; accounting has nothing terminal yet."""
        if command[0] == "squeue":
            return "638288|RUNNING|None|node-a|2026-08-20T09:00:00|2026-08-20T09:01:00|00:01:00", None
        if command[-1] == f"--format={cluster.STEP_ACCOUNTING_FORMAT}":
            return f"638288.0|{step_name}|RUNNING|0:0|2026-08-20T09:00:10|2026-08-20T09:01:00|Unknown|00:01:00|node-a|16|test|", None
        return "", None

    monkeypatch.setattr(generation.campaign, "_scheduler_output", scheduler_output)
    monkeypatch.setattr(
        generation.campaign.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        generation.campaign.campaign_evidence,
        "campaign_from_manifest",
        lambda _manifest: campaign,
    )
    monkeypatch.setattr(
        generation.campaign.campaign_evidence,
        "campaign_run_directory",
        lambda *_args, **_kwargs: run_directory,
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "case_failure_is_recorded",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        generation.campaign.progress_service,
        "load_runtime_progress",
        lambda *_args, **_kwargs: {
            "availability": "unavailable",
            "reason": "not_reported",
            "age_seconds": None,
            "stale": None,
        },
    )

    work_directory = tmp_path / "status-work"
    work_directory.mkdir()
    prelaunch = stop_service.prepare_capacity_checkout_status(
        work_directory,
        checkout_index=3,
    )
    (work_directory / stop_service.STOP_STATUS_FILENAME).write_text(
        "1787228251108\nError",
        encoding="utf-8",
    )
    artifact = stop_service.inspect_capacity_checkout_status(
        prelaunch,
        process_id=7734,
        process_exit_code=0,
        temporary_capacity_classified=True,
        solver_progress_started=False,
        required_exports_exist=False,
        scientific_result_exists=False,
    )
    classification = license_service.classify_temporary_license_capacity(
        """Could not obtain license for 'CFD Module'.
Licensed number of users already reached.
"""
    )
    assert artifact is not None
    assert classification is not None
    started_at = datetime(2026, 8, 20, 9, 0, tzinfo=UTC)
    staged: Queue[Path] = Queue()
    releases: Queue[None] = Queue()
    original_fsync = common.serialization._fsync_file

    def gated_fsync(temporary_path: Path) -> None:
        original_fsync(temporary_path)
        staged.put(temporary_path)
        releases.get(timeout=10.0)

    monkeypatch.setattr(common.serialization, "_fsync_file", gated_fsync)
    statuses: list[Mapping[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            license_service.record_in_allocation_status_artifact_recovery,
            batch,
            task.case_index,
            campaign_run_id=run_id,
            dispatch_key="638288-1",
            checkout_started_at=started_at,
            checkout_ended_at=started_at + timedelta(seconds=1),
            hostname="synthetic-node",
            artifact=artifact,
            classification=classification,
            cleanup_state="pending",
            storage_root=tmp_path,
        )
        try:
            temporary_path = staged.get(timeout=10.0)
            assert temporary_path.is_file()
            resumed = generation.campaign.resume_campaign(
                run_id,
                storage_root=tmp_path,
                status_callback=statuses.append,
            )
            assert resumed["state"] == "active"
            assert len(statuses) == 1
            assert statuses[0]["cases"][0]["state"] == "running"
            assert statuses[0]["cases"][0]["status_artifact_recovery_count"] == 0
        finally:
            releases.put(None)
        receipt_path = future.result(timeout=10.0)

    recoveries = license_service.load_in_allocation_status_artifact_recoveries(
        batch,
        task.case_index,
        campaign_run_id=run_id,
        dispatch_key="638288-1",
        storage_root=tmp_path,
    )
    assert receipt_path.name == "checkout_0003.json"
    assert [record["cleanup_state"] for record in recoveries] == ["pending"]


def test_monitor_signatures_separate_phase_changes_from_rate_limited_advancement() -> None:
    """Make phase changes urgent while solver-row advancement stays rate-limited."""
    status: dict[str, Any] = {
        "campaign_run_id": "transient_smoke__0123456789abcdef",
        "campaign_state": "running",
        "cases": [_active_case("case_0001", "610083", "hpc113", 100.0)],
    }
    state_signature, progress_signature = status_service.campaign_monitor_signatures(status)
    advanced = deepcopy(status)
    advanced["cases"][0]["runtime_progress"]["simulated_time_seconds"] = 200.0
    advanced_state, advanced_progress = status_service.campaign_monitor_signatures(advanced)
    phased = deepcopy(advanced)
    phased["cases"][0]["runtime_progress"]["phase"] = "collecting_exports"
    phased_state, _phased_progress = status_service.campaign_monitor_signatures(phased)

    assert advanced_state == state_signature
    assert advanced_progress != progress_signature
    assert phased_state != advanced_state


def test_campaign_status_cli_reuses_summary_and_monitor_formatter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep human and automatic output in the same reusable Python owner."""
    status: dict[str, Any] = {
        "campaign_run_id": "transient_smoke__0123456789abcdef",
        "campaign_state": "running",
        "cases": [_active_case("case_0001", "610083", "hpc113", 100.0)],
    }
    monkeypatch.setattr(cli_generation.campaign_runtime, "campaign_status", lambda *_args, **_kwargs: status)

    assert cli_generation.main(["campaign-status", status["campaign_run_id"], "--format", "summary", "--storage-root", str(tmp_path)]) == 0
    summary = capsys.readouterr().out
    assert "case_0001" in summary

    assert (
        cli_generation.main(
            [
                "campaign-status",
                status["campaign_run_id"],
                "--format",
                "monitor",
                "--max-active-cases",
                "1",
                "--storage-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    monitor = capsys.readouterr().out
    header, rendered = monitor.split("\n", maxsplit=1)
    assert len(header.split("\t")) == 4
    assert header.startswith("campaign-monitor\trunning\t")
    assert rendered == summary


def test_terminal_timestamp_projection_preserves_ambiguous_slurm_end() -> None:
    """Normalize only proven timezones and never guess a naïve Slurm End value."""
    raw = "2026-08-20T09:49:19"

    ambiguous = generation.campaign._terminal_timestamp_projection(
        raw,
        source_timezone=None,
    )
    normalized = generation.campaign._terminal_timestamp_projection(
        raw,
        source_timezone=timezone(timedelta(hours=2), name="authoritative-site"),
    )

    assert ambiguous.original_value == raw
    assert ambiguous.source_timezone is None
    assert ambiguous.normalized_utc_value is None
    assert ambiguous.normalization_reason == "terminal_timestamp_ambiguous_for_presentation"
    assert normalized.original_value == raw
    assert normalized.source_timezone == "authoritative-site"
    assert normalized.normalized_utc_value == "2026-08-20T07:49:19+00:00"
    assert normalized.normalization_reason == "terminal_timestamp_normalized"


def test_terminal_timestamp_writer_rejects_naive_canonical_value_before_write(
    tmp_path: Path,
) -> None:
    """Require every new canonical projection and its record time to be aware."""
    task = cluster.CampaignTask(
        batch_name="test-batch",
        batch_id="test-batch-id",
        execution_key="lentil-natural",
        case_index=1,
        case_id="case_0001",
    )
    manifest = {"campaign_run_id": "source~steady_flow~timestamp-writer__0123456789abcdef"}
    invalid = generation.campaign.TerminalTimestampProjection(
        original_value="2026-08-20T09:49:19+00:00",
        source_timezone="embedded_offset",
        normalized_utc_value="2026-08-20T09:49:19",
        normalization_owner=(generation.campaign._TERMINAL_TIMESTAMP_OWNER),
        normalization_reason="terminal_timestamp_timezone_aware",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        generation.campaign._record_terminal_timestamp_evidence(
            manifest,
            task,
            "101-1",
            invalid,
            storage_root=tmp_path,
        )

    path = generation.campaign._terminal_timestamp_evidence_path(
        manifest,
        task,
        "101-1",
        storage_root=tmp_path,
    )
    assert not path.exists()

    valid = generation.campaign._terminal_timestamp_projection(
        "2026-08-20T09:49:19Z",
        source_timezone=None,
    )
    recorded = generation.campaign._record_terminal_timestamp_evidence(
        manifest,
        task,
        "101-1",
        valid,
        storage_root=tmp_path,
    )
    assert recorded is not None
    payload = json.loads(recorded.read_text(encoding="utf-8"))
    recorded_at = datetime.fromisoformat(payload["recorded_at"])
    assert recorded_at.tzinfo is not None
    assert recorded_at.utcoffset() is not None
    assert payload["normalized_utc_value"] == "2026-08-20T09:49:19+00:00"
    assert payload["dispatch_key"] == "101-1"
    assert payload["schema_version"] == 2


def test_global_case_integrity_error_still_escapes_reconciliation(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep scientific publication and identity corruption campaign-global."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    _patch_synthetic_input_references(monkeypatch, campaign)
    manifest: dict[str, Any] = {
        "campaign_run_id": "source~steady_flow~global-integrity__0123456789abcdef",
        "git_commit": "c" * 40,
        "slurm_job_ids": [],
        "submission_intent": None,
        "admission_reservations": [],
        "submissions": [],
    }
    message = "contradictory immutable publication hash"
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
    )

    with pytest.raises(RuntimeError, match="immutable publication hash"):
        generation.campaign._reconciled(
            manifest,
            campaign,
            {"active": {}, "accounted": {}},
            storage_root=tmp_path,
            compact_license_attempt_payloads=True,
        )


def test_duplicate_active_case_ownership_remains_campaign_global(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reject two active Slurm jobs claiming one case before local isolation."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    submissions = [
        _submission(campaign, task, "701", 1),
        _submission(campaign, task, "702", 2),
    ]
    manifest = {
        "campaign_run_id": "duplicate-active__0123456789abcdef",
        "git_commit": "d" * 40,
        "slurm_job_ids": ["701", "702"],
        "submission_intent": None,
        "admission_reservations": [],
        "submissions": submissions,
    }
    scheduler = {
        "active": {
            "701-1": ["701-1", "RUNNING"],
            "702-2": ["702-2", "PENDING"],
        },
        "accounted": {},
    }

    with pytest.raises(RuntimeError, match="Conflicting active Slurm step ownership"):
        generation.campaign._reconciled(
            manifest,
            campaign,
            scheduler,
            storage_root=tmp_path,
            compact_license_attempt_payloads=True,
        )


@pytest.mark.parametrize(
    ("target_reached", "hit_t_max", "final_bulk_moisture"),
    [(True, False, 0.118), (False, True, 0.134)],
)
def test_successful_transient_status_uses_validated_final_bulk_moisture(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_reached: bool,
    hit_t_max: bool,
    final_bulk_moisture: float,
) -> None:
    """Project canonical moisture for target-stop and maximum-horizon success."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    processed = tmp_path / "processed" / task.case_id
    evidence = common.paths.resolve_generation_result_evidence_directory(processed)
    evidence.mkdir(parents=True)
    status_path = evidence / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "quality_flag_count": 0,
                "t_stop_exact": 60.3223,
                "target_reached": target_reached,
                "hit_t_max": hit_t_max,
                "f_wet_dm_final": 0.0129802,
                "units": {
                    "t_stop_exact": "h",
                    "f_wet_dm_final": "1",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "processed_case_directory",
        lambda *_args, **_kwargs: processed,
    )
    canonical_reads: list[Path] = []

    def read_final_bulk(path: Path) -> float:
        canonical_reads.append(path)
        return final_bulk_moisture

    monkeypatch.setattr(
        generation.campaign.storage_service,
        "read_transient_final_bulk_moisture",
        read_final_bulk,
    )

    summary = generation.campaign._successful_status_summary(
        batch,
        task,
        storage_root=tmp_path,
    )

    target = batch.scientific_values["material"]["parameter_registry"]["X_target_wb"]["nominal"]
    assert summary["simulated_end_time"] == pytest.approx(60.3223)
    assert summary["final_bulk_moisture_wb"] == pytest.approx(final_bulk_moisture)
    assert summary["target_moisture_wb"] == pytest.approx(target)
    assert canonical_reads == [processed / "case.h5"]
    assert "final_moisture_value" not in summary
    persisted = json.loads(status_path.read_text(encoding="utf-8"))
    assert persisted["f_wet_dm_final"] == pytest.approx(0.0129802)
    assert persisted["target_reached"] is target_reached
    assert persisted["hit_t_max"] is hit_t_max


def test_successful_transient_status_fails_without_final_bulk_moisture(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail projection when validated success lacks required canonical evidence."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    processed = tmp_path / "processed" / task.case_id
    common.paths.resolve_generation_result_evidence_directory(processed).mkdir(parents=True)
    (common.paths.resolve_generation_result_evidence_directory(processed) / "status.json").write_text(
        json.dumps(
            {
                "quality_flag_count": 0,
                "t_stop_exact": 60.3223,
                "units": {"t_stop_exact": "h"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "processed_case_directory",
        lambda *_args, **_kwargs: processed,
    )

    with pytest.raises(RuntimeError, match="lacks required canonical final bulk-moisture evidence"):
        generation.campaign._successful_status_summary(
            batch,
            task,
            storage_root=tmp_path,
        )


def test_successful_completion_uses_admitted_processing_provenance(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project the existing timezone-aware successful-processing timestamp."""
    config_path, _template = generation_config_factory()
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    processed = tmp_path / "processed" / task.case_id
    common.paths.resolve_generation_result_evidence_directory(processed).mkdir(parents=True)
    processing_path = common.paths.resolve_generation_result_evidence_directory(processed) / "processing_provenance.json"
    processing_path.write_text(
        json.dumps({"recorded_at": "2026-08-20T10:15:30+02:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "processed_case_directory",
        lambda *_args, **_kwargs: processed,
    )

    projected = generation.campaign._successful_completion_at(
        "successful",
        batch,
        task,
        storage_root=tmp_path,
    )

    assert projected.completed_at == "2026-08-20T08:15:30+00:00"
    assert projected.original_value == "2026-08-20T10:15:30+02:00"


def _successful_case(
    index: int,
    *,
    completed_at: str | None,
    material: str = "lentil",
    latest_job_id: str | None = None,
    simulated_end_time: float | None = None,
    final_bulk_moisture_wb: float | None = None,
    target_moisture_wb: float | None = None,
) -> dict[str, Any]:
    """Return one synthetic successful campaign-status row."""
    return {
        "case_id": f"case_{index:04d}",
        "batch_name": f"transient_drying__{material}__natural",
        "material": material,
        "state": "successful",
        "reason": "validated_case_evidence",
        "completed_at": completed_at,
        "latest_job_id": latest_job_id,
        "elapsed": "99:59:59",
        "simulated_end_time": simulated_end_time,
        "simulated_end_time_unit": "h" if simulated_end_time is not None else None,
        "final_bulk_moisture_wb": final_bulk_moisture_wb,
        "target_moisture_wb": target_moisture_wb,
    }


def _failed_case(
    index: int,
    *,
    failed_at: str | None,
    classification: str = "failed",
) -> dict[str, Any]:
    """Return one synthetic genuine terminal failure row."""
    return {
        "case_id": f"case_{index:04d}",
        "batch_name": "transient_drying__lentil__natural",
        "material": "lentil",
        "state": "failed",
        "classified_state": classification,
        "failure_stage": "solver",
        "solver_state": "failed",
        "reason": "bounded synthetic failure",
        "failed_at": failed_at,
    }


def _license_blocked_case(index: int) -> dict[str, Any]:
    """Return one license-blocked row carrying bulky raw FlexNet evidence."""
    return {
        "batch_name": "transient_drying__kidney_bean__natural",
        "case_id": f"case_{index:04d}",
        "state": "license_blocked",
        "reason": "temporary_license_capacity",
        "latest_job_id": str(629_820 + index),
        "temporary_license_retry": {
            "retry_count": 1,
            "next_retry_at": "2026-08-20T00:07:09Z",
            "cumulative_wait_seconds": 60,
            "matched_signatures": ["licensed number of users already reached"],
            "raw_excerpt": "FlexNet error " * 500,
            "license_path": "/complete/internal/license/path",
        },
        "in_allocation_license_window": {
            "realised_window_seconds": 120.3,
            "checkout_attempt_count": 8,
            "raw_excerpt": "Licensed number of users already reached",
        },
    }


def _case_in_state(state: str, index: int) -> dict[str, Any]:
    """Return one synthetic row whose terminal time increases with its index."""
    timestamp = (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)).isoformat()
    if state == "successful":
        return _successful_case(index, completed_at=timestamp)
    if state == "failed":
        return _failed_case(index, failed_at=timestamp)
    if state == "license_blocked":
        return _license_blocked_case(index)
    return {"case_id": f"case_{index:04d}", "batch_name": "transient_drying__lentil__natural", "material": "lentil", "state": state}


_RAW_MACHINE_EVIDENCE = (
    "FlexNet error",
    "Licensed number of users",
    "matched_signatures",
    "/complete/internal/license/path",
    "content_excerpt",
    "content_sha256",
    "/private/worker",
    "automatic_continuation_allowed",
    "/storage/",
    "{",
)


def test_summary_renders_every_case_state_without_raw_machine_evidence() -> None:
    """Name each actionable case and keep bulky diagnostics in canonical evidence."""
    failed = _failed_case(3, failed_at="2026-08-20T09:49:19+00:00")
    failed.update(
        {
            "reason": (
                'Unsafe COMSOL capacity-checkout status artifact: {"content_excerpt":"1787228251108","content_sha256":"'
                + "a" * 64
                + '","status_path":"/private/worker/solved.mph.status"}'
            ),
            "evidence_path": "/storage/01_generation/attempts/case_0003/attempt.json",
            "automatic_continuation_allowed": False,
        }
    )
    status = {
        "campaign_run_id": "mixed__0123456789abcdef",
        "campaign_state": "running",
        "admission_blocked": True,
        "admission_block_reason": "solver_failure_threshold_exceeded",
        "cases": [
            _active_case("case_0001", "610083", "hpc113", 120.0),
            _license_blocked_case(2),
            failed,
            {"case_id": "case_0004", "batch_name": "transient_drying__lentil__natural", "state": "scheduler_pending", "latest_job_id": "4102"},
            _successful_case(5, completed_at="2026-08-20T10:00:00+00:00"),
            {"case_id": "case_0006", "batch_name": "transient_drying__chickpea__natural", "material": "chickpea", "state": "never_started"},
        ],
    }

    rendered = status_service.format_campaign_status_summary(status)

    for visible in (
        "case_0001",
        "610083",
        "hpc113",
        "case_0002",
        "case_0003",
        "case_0004",
        "case_0005",
        "chickpea",
        "solver_failure_threshold_exceeded",
    ):
        assert visible in rendered
    assert "case_0006" not in rendered
    assert [raw for raw in _RAW_MACHINE_EVIDENCE if raw in rendered] == []


@pytest.mark.parametrize(
    ("state", "bucket"),
    [
        ("successful", "successful"),
        ("active", "running"),
        ("admission_waiting", "running"),
        ("license_blocked", "license_blocked"),
        ("scheduler_unknown", "scheduler_pending"),
        ("never_started", "not_admitted"),
        ("timed_out", "failed"),
    ],
)
def test_case_states_map_to_one_summary_bucket(state: str, bucket: str) -> None:
    """Never count license waits, scheduler gaps, or reservations as failures."""
    assert status_service._case_bucket({"state": state}) == bucket


def test_scheduler_unknown_projects_as_nonterminal_pending_work() -> None:
    """Keep missing scheduler visibility explicit without inventing failure."""
    public = generation.campaign._public_task_view(
        {
            "batch_name": "transient_drying__lentil__natural",
            "batch_id": "batch-id",
            "case_index": 1,
            "case_id": "case_0001",
            "state": "scheduler_unknown",
            "reason": "4102",
            "latest_job_id": "4102",
            "runtime_progress": {"availability": "unavailable", "reason": "not_reported"},
        },
        None,
    )

    assert public["state"] == "scheduler_pending"
    assert public["classified_state"] == "scheduler_unknown"
    assert public["terminal_decision"] is None


def test_bounded_active_view_prefers_executing_cases() -> None:
    """Render every active case by default and executing work first when bounded."""
    waiting = _active_case("case_0001", "610083", "hpc113", 0.0)
    waiting["runtime_progress"]["phase"] = "acquiring_comsol_license"
    executing = _active_case("case_0002", "610084", "hpc114", 662.4)
    status = {"campaign_run_id": "active__0123456789abcdef", "campaign_state": "running", "cases": [waiting, executing]}

    explicit = status_service.format_campaign_status_summary(status)
    bounded = status_service.format_campaign_status_summary(status, max_active_cases=1)

    assert "case_0001" in explicit
    assert "case_0002" in explicit
    assert "case_0002" in bounded
    assert "case_0001" not in bounded


@pytest.mark.parametrize("state", ["successful", "failed", "license_blocked", "never_started"])
def test_summary_detail_stays_bounded_at_campaign_scale(state: str) -> None:
    """Keep the total count while the rendered detail does not grow with campaign size."""

    def render(count: int) -> str:
        cases = [_case_in_state(state, index) for index in range(1, count + 1)]
        return status_service.format_campaign_status_summary({"campaign_run_id": "scale", "campaign_state": "running", "cases": cases})

    small, large = render(25), render(1_050)

    assert "1050" in large
    assert len(large.splitlines()) == len(small.splitlines())
    assert len(set(re.findall(r"case_\d{4}", large))) <= status_service._MAX_ACTIONABLE_DETAILS


@pytest.mark.parametrize("state", ["successful", "failed"])
def test_summary_shows_latest_terminal_cases_newest_first(state: str) -> None:
    """Select the latest three by terminal time, then unavailable times by identity."""
    field = "completed_at" if state == "successful" else "failed_at"

    def render(timestamps: Mapping[int, str | None]) -> str:
        cases = []
        for index, timestamp in timestamps.items():
            case = _case_in_state(state, index)
            case[field] = timestamp
            cases.append(case)
        return status_service.format_campaign_status_summary({"campaign_run_id": "terminal", "campaign_state": "running", "cases": cases})

    newest = render(
        {
            1: "2026-01-01T00:02:00+00:00",
            2: "2026-01-01T00:05:00+00:00",
            3: "2026-01-01T00:03:00+00:00",
            4: "2026-01-01T00:04:00+00:00",
            5: "2026-01-01T00:06:00+00:00",
        }
    )
    fallback = render({4: "2026-08-20T09:49:19", 2: "2026-08-20T08:00:00+00:00", 1: "2026-08-20T08:00:00+00:00", 3: None})

    for rendered, shown, omitted in (
        (newest, ("case_0005", "case_0002", "case_0004"), ("case_0001", "case_0003")),
        (fallback, ("case_0001", "case_0002", "case_0003"), ("case_0004",)),
    ):
        positions = [rendered.index(case_id) for case_id in shown]
        assert positions == sorted(positions)
        assert not any(case_id in rendered for case_id in omitted)


def test_summary_formatting_reads_no_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render only supplied status fields without filesystem or hashing work."""

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        message = "Formatting attempted external or filesystem work"
        raise AssertionError(message)

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(status_service.common.serialization, "file_sha256", forbidden)
    cases = [
        _successful_case(1, completed_at="2026-01-01T00:01:00+00:00", simulated_end_time=60.3, final_bulk_moisture_wb=0.116, target_moisture_wb=0.12),
        _successful_case(2, completed_at=None),
    ]

    rendered = status_service.format_campaign_status_summary({"campaign_run_id": "pure", "campaign_state": "running", "cases": cases})

    assert "case_0001" in rendered
    assert "case_0002" in rendered


def test_completed_benchmark_status_prints_validated_final_markdown() -> None:
    """Expose the persisted final benchmark result without recomputing it."""
    rendered = status_service.format_benchmark_status_summary(
        {
            "suite_name": "core scaling",
            "benchmark_run_id": "benchmark",
            "state": "complete",
            "wave_count": 4,
            "work_units": [],
            "final_summary": {
                "path": "/evidence/summary.md",
                "markdown": "# Validated benchmark result\n\nRecommended: 4 cores.\n",
            },
        }
    )

    assert "/evidence/summary.md" in rendered
    assert "# Validated benchmark result" in rendered


def _board_case(material: str, case_id: str, state: str, **extra: Any) -> dict[str, Any]:
    """Return one case of the canonical status with only the fields the live board reads."""
    return {
        "material": material,
        "batch_name": f"transient_drying__{material}__natural",
        "case_id": case_id,
        "state": state,
        "classified_state": state,
        "terminal_decision": None,
        "runtime_progress": {"availability": "unavailable", "reason": "no_job"},
        **extra,
    }


def _board_running(case_id: str, simulated_hours: float, elapsed_seconds: float, *, horizon_hours: float | None = 168.0) -> dict[str, Any]:
    runtime = {
        "availability": "available",
        "phase": "transient_drying",
        "parser_state": "available",
        "simulated_time_seconds": simulated_hours * 3600,
        "step_size_seconds": 40.0,
        "time_failures": 3,
        "nonlinear_failures": 0,
        "elapsed_seconds": elapsed_seconds,
        "age_seconds": 12.0,
        "stale": False,
    }
    return _board_case("lentil", case_id, "running", runtime_progress=runtime, physical_start_hours=0.0, physical_horizon_hours=horizon_hours)


def test_board_estimates_the_remaining_time_from_progress_and_the_median_finished_case() -> None:
    """Running cases extrapolate their simulated time; each waiting case takes the median finished duration on the seat that frees first."""
    finished = [
        _board_case("chickpea", f"case_{index:04d}", "successful", elapsed=elapsed) for index, elapsed in enumerate(("00:20:00", "00:40:00"), 1)
    ]
    cases = [_board_running("case_0001", 56.0, 1000.0), *finished, _board_case("lentil", "case_0002", "never_started")]
    board = status_service.CampaignBoard(started=0.0, seats=2, verify_seconds=120.0)
    text = board.observe({"campaign_run_id": "run", "campaign_state": "active", "cases": cases}, now=125.0)
    assert text is not None
    header, *lines = text.splitlines()
    # The running case needs 1000 s * (168 - 56) / 56 = 2000 s; the waiting case takes the 1800 s median on the free seat: max = 2000 s.
    assert header == "Board  done 2  running 1  waiting 1  failed 0  of 4  elapsed 2 min  remaining ~33 min"
    assert any(line.startswith("lentil    case_0001  running  33 %  56 h / 168 h  step 40 s") and line.endswith("ETA ~33 min") for line in lines)
    assert any("case_0002  waiting  not started yet" in line for line in lines)


def test_board_gives_a_whole_case_to_one_seat() -> None:
    """One waiting case of 3600 s on eight free seats still takes 3600 s; it is never spread over the seats."""
    cases = [_board_case("lentil", "case_0001", "successful", elapsed="01:00:00"), _board_case("lentil", "case_0002", "never_started")]
    text = status_service.CampaignBoard(started=0.0, seats=8, verify_seconds=120.0).observe(
        {"campaign_run_id": "run", "campaign_state": "active", "cases": cases}, now=0.0
    )
    assert text is not None
    assert text.splitlines()[0].endswith(f"remaining ~{status_service.format_duration(3600.0)}")


def test_campaign_status_carries_each_case_window_from_the_persisted_resolved_config_and_the_board_counts_past_its_start(
    generation_config_factory: Any, world: Any, tmp_path: Path
) -> None:
    """The status takes each case's window from its batch's persisted resolved config; a window from 24 h shows 25 % at 66 h simulated.

    Input generation admits only windows that start at zero, so the non-zero start is read from a test-written resolved config.
    """
    config_path, _template = generation_config_factory(simulation_profile="transient_drying", scheduler_kind="slurm", natural_count=1, node_jobs=1)
    campaign = generation.cases.config.load_campaign_config(config_path)
    run_id = submit(campaign, world)["campaign_run_id"]
    (case,) = generation.campaign.campaign_status(run_id, storage_root=world.storage)["cases"]
    assert (case["physical_start_hours"], case["physical_horizon_hours"]) == (0.0, 168.0)
    resolved = generation.campaign.batch_runtime.batch_meta_directory(campaign.batches[0], storage_root=tmp_path) / "resolved_generation_config.json"
    resolved.parent.mkdir(parents=True)
    resolved.write_text(json.dumps({"time": {"start": 24.0, "stop": 192.0}}), encoding="utf-8")
    window = generation.campaign._persisted_time_window(campaign.batches[0], storage_root=tmp_path)
    assert window == {"physical_start_hours": 24.0, "physical_horizon_hours": 168.0}
    running = {**case, **window, "state": "running", "runtime_progress": _board_running("case_0001", 66.0, 600.0)["runtime_progress"]}
    text = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0).observe(
        {"campaign_run_id": run_id, "campaign_state": "active", "cases": [running]}, now=0.0
    )
    assert text is not None
    assert "25 %  42 h / 168 h" in text


def test_board_says_unknown_when_the_evidence_for_an_estimate_is_missing() -> None:
    """Without a horizon, a running case, or a finished case to take the median from, the estimate is not guessed."""
    running = {"campaign_run_id": "run", "campaign_state": "active", "cases": [_board_running("case_0001", 56.0, 1000.0, horizon_hours=None)]}
    unknown_horizon = status_service.CampaignBoard(started=0.0, seats=2, verify_seconds=120.0).observe(running, now=0.0)
    assert unknown_horizon is not None
    assert "remaining unknown" in unknown_horizon
    assert "case_0001  running  transient drying  ETA unknown" in unknown_horizon
    waiting = {"campaign_run_id": "run", "campaign_state": "active", "cases": [_board_case("lentil", "case_0001", "scheduler_pending")]}
    no_median = status_service.CampaignBoard(started=0.0, seats=2, verify_seconds=120.0).observe(waiting, now=0.0)
    assert no_median is not None
    assert "remaining unknown" in no_median


def test_board_names_the_typed_reason_and_the_exact_replacement_of_each_failed_slot() -> None:
    """A failed line shows the typed reason and the case that took its slot by ``parent_failed_case_id``; undecided says so."""
    stalled = "stalled: 0.98 h of 168 h simulated after 900 s solving"
    decided = _board_case("lentil", "case_0001", "failed", failure_reason="stalled", failure_detail=stalled, terminal_decision="failed")
    undecided = _board_case("chickpea", "case_0001", "failed", failure_reason="solver_error", failure_detail="COMSOL exited with status 7")
    unrelated = _board_case("lentil", "case_0003", "scheduler_pending")
    replacement = _board_case("lentil", "case_0004", "scheduler_pending", parent_failed_case_id="case_0001")
    text = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0).observe(
        {"campaign_run_id": "run", "campaign_state": "active", "cases": [decided, undecided, unrelated, replacement]}, now=0.0
    )
    assert text is not None
    assert f"lentil    case_0001  failed   {stalled}  replaced by case_0004" in text
    assert "chickpea  case_0001  failed   solver_error: COMSOL exited with status 7  no decision recorded" in text
    # A failed replacement passes the same slot on to the next replacement that names its failed primary.
    failed_replacement = {
        **replacement,
        "state": "failed",
        "classified_state": "failed",
        "terminal_decision": "failed",
        "failure_reason": "timed_out",
    }
    successor = _board_case("lentil", "case_0005", "scheduler_pending", parent_failed_case_id="case_0001")
    after = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0).observe(
        {"campaign_run_id": "run", "campaign_state": "active", "cases": [decided, failed_replacement, successor]}, now=0.0
    )
    assert after is not None
    assert "case_0001  failed   " + stalled + "  replaced by case_0004" in after
    assert "case_0004  failed   timed_out  replaced by case_0005" in after


@pytest.mark.parametrize(
    "recorded",
    [
        {"terminal_decision": "failed"},
        {"failure_reason": "stalled"},
        {"failure_detail": "stalled: 0.98 h of 168 h simulated after 900 s solving"},
        {"parent_failed_case_id": "case_0001"},
    ],
)
def test_board_reprints_at_once_when_a_decision_or_its_lineage_is_recorded_while_another_case_runs(recorded: dict[str, str]) -> None:
    """A recorded decision, typed reason, detail, or parent link is a state change, not progress waiting for the repeat interval."""
    running = _board_running("case_0001", 1.0, 100.0)
    failed = _board_case("lentil", "case_0002", "failed")
    board = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0)
    assert board.observe({"campaign_run_id": "run", "campaign_state": "active", "cases": [running, failed]}, now=0.0) is not None
    assert board.observe({"campaign_run_id": "run", "campaign_state": "active", "cases": [running, {**failed, **recorded}]}, now=10.0) is not None


def test_a_case_without_a_state_and_a_status_without_node_jobs_read_as_unknown_not_failed() -> None:
    """Missing evidence never becomes a failure: no state is pending scheduler evidence, and absent node jobs are no node jobs."""
    assert status_service._case_bucket({"case_id": "case_0001"}) == "scheduler_pending"
    status: dict[str, Any] = {
        "campaign_run_id": "run",
        "campaign_state": "active",
        "node_jobs": None,
        "cases": [_board_case("lentil", "case_0001", "running")],
    }
    assert status_service.campaign_monitor_signatures(status) == status_service.campaign_monitor_signatures({**status, "node_jobs": []})


def test_board_marks_a_case_whose_solver_evidence_stays_unchanged_and_repeats_only_when_due() -> None:
    """The board is silent between a state change and the repeat interval; unchanged evidence older than 600 s is marked."""
    status = {"campaign_run_id": "run", "campaign_state": "active", "cases": [_board_running("case_0001", 1.0, 100.0)]}
    board = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0)
    assert board.observe(status, now=0.0) is not None
    assert board.observe(status, now=299.0) is None
    repeated = board.observe(status, now=300.0)
    assert repeated is not None
    assert "no progress" not in repeated
    marked = board.observe(status, now=601.0)
    assert marked is not None
    assert marked.splitlines()[-1].endswith("no progress for 10 min")
    advanced = {**status, "cases": [_board_running("case_0001", 2.0, 200.0)]}
    moved = board.observe(advanced, now=901.0)
    assert moved is not None
    assert "no progress" not in moved


def test_an_unverified_failure_reads_verifying_then_failed_and_the_board_reprints_when_the_window_closes() -> None:
    """A COMPLETED step without visible result evidence, even with its success decision seen, is verifying for the settle window, then failed.

    The window's end reprints the board.
    """
    unverified = _board_case("lentil", "case_0001", "failed", reason="completed_without_valid_case_evidence", terminal_decision="successful")
    status = {"campaign_run_id": "run", "campaign_state": "active", "cases": [unverified]}
    board = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0)
    first = board.observe(status, now=0.0)
    assert first is not None
    assert "failed 0  verifying 1  of 1" in first.splitlines()[0]
    assert "lentil  case_0001  verifying  step ended, result not yet visible" in first
    assert board.observe(status, now=119.0) is None
    expired = board.observe(status, now=120.0)
    assert expired is not None
    assert "failed 1  of 1" in expired.splitlines()[0]
    assert "verifying" not in expired
    assert "lentil  case_0001  failed   completed_without_valid_case_evidence  decision recorded" in expired


def test_another_failure_reason_or_a_recorded_failure_reads_failed_at_once() -> None:
    """Only the COMPLETED-without-evidence reason waits for its window; a stalled case or a recorded failed decision reads failed at once."""
    stalled = _board_case("lentil", "case_0001", "failed", failure_reason="stalled", terminal_decision="failed")
    decided = _board_case("lentil", "case_0002", "failed", reason="completed_without_valid_case_evidence", terminal_decision="failed")
    text = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0).observe(
        {"campaign_run_id": "run", "campaign_state": "active", "cases": [stalled, decided]}, now=0.0
    )
    assert text is not None
    assert "failed 2  of 2" in text.splitlines()[0]
    assert "verifying" not in text
    assert "lentil  case_0001  failed   stalled  decision recorded" in text
    assert "lentil  case_0002  failed   completed_without_valid_case_evidence  decision recorded" in text


def test_a_verifying_case_that_gets_its_result_reads_done_and_a_new_failure_starts_a_new_window() -> None:
    """A verifying case whose evidence becomes visible reads done at once; a later failure with the same reason verifies again."""
    unverified = _board_case("lentil", "case_0001", "failed", reason="completed_without_valid_case_evidence")
    board = status_service.CampaignBoard(started=0.0, seats=1, verify_seconds=120.0)
    assert board.observe({"campaign_run_id": "run", "campaign_state": "active", "cases": [unverified]}, now=0.0) is not None
    successful = _board_case("lentil", "case_0001", "successful", elapsed="00:01:00")
    done = board.observe({"campaign_run_id": "run", "campaign_state": "active", "cases": [successful]}, now=10.0)
    assert done is not None
    assert "done 1  running 0  waiting 0  failed 0  of 1" in done.splitlines()[0]
    assert "lentil  case_0001  done" in done
    assert "verifying" not in done
    again = board.observe({"campaign_run_id": "run", "campaign_state": "active", "cases": [unverified]}, now=20.0)
    assert again is not None
    assert "lentil  case_0001  verifying  step ended, result not yet visible" in again
