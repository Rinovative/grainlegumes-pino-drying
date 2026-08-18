# ruff: noqa: PLR2004, S101
"""Per-case campaign reconciliation and human monitoring contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from src import generation
from src.generation import generation_campaign_status as status_service
from src.generation.cli import cli_generation
from src.generation.runtime import generation_runtime_cluster as cluster

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _submission(task: cluster.CampaignTask, job_id: str, index: int) -> dict[str, Any]:
    """Return one exact synthetic persisted submission."""
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
        "job_name": f"campaign-{index:04d}",
        "command": ["sbatch"],
        "job_id": job_id,
        "status": "submitted",
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
            "simulated_time_seconds": simulated_time,
            "step_size_seconds": 0.075,
            "order": 2,
            "time_failures": 1,
            "nonlinear_failures": 25,
            "age_seconds": 4.0,
            "stale": False,
        },
    }


def test_campaign_status_exposes_deterministic_cases_from_one_scheduler_query(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map successful, active, failed, retry, and unsent cases without new queries."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm", natural_count=6)
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    assert len(tasks) == 6
    run_id = "synthetic_campaign__0123456789abcdef"
    run_directory = tmp_path / "campaign"
    run_directory.mkdir()
    submissions = [_submission(task, str(100 + index), index) for index, task in enumerate(tasks[:5], start=1)]
    manifest = {
        "campaign_run_id": run_id,
        "git_commit": "a" * 40,
        "slurm_job_ids": [record["job_id"] for record in submissions],
        "scheduler_job_name": "campaign",
        "scheduler_log_directory": str(run_directory / "scheduler"),
        "submission_config": {"maximum_failed_cases": 10},
        "submission_intent": None,
        "submissions": submissions,
        "state": "active",
        "remote_storage_root": str(tmp_path),
        "campaign_meta_directory": str(run_directory),
    }
    scheduler = {
        "squeue": {"command": ["squeue"], "output": "102|RUNNING", "error": None},
        "sacct": {"command": ["sacct"], "output": "101|COMPLETED", "error": None},
        "active": {
            "102": [
                "102",
                "RUNNING",
                "None",
                "node-b",
                "2026-01-01T00:01:00",
                "2026-01-01T00:02:00",
                "00:03:04",
            ],
            "105": [
                "105",
                "PENDING",
                "Resources",
                "(Priority)",
                "2026-01-01T00:05:00",
                "N/A",
                "00:00:00",
            ],
        },
        "accounted": {
            "101": ["101", "COMPLETED", "0:0", "submit-a", "start-a", "end-a", "00:01:00", "node-a", "16", "standard"],
            "103": ["103", "FAILED", "1:0", "submit-c", "start-c", "end-c", "00:02:00", "node-c", "16", "standard"],
            "104": ["104", "FAILED", "1:0", "submit-d", "start-d", "end-d", "00:01:00", "node-d", "16", "standard"],
        },
    }
    query_count = {"value": 0}

    def scheduler_evidence(_job_ids: list[str]) -> dict[str, Any]:
        query_count["value"] += 1
        return scheduler

    retry = {"retry_budget_remaining": True}
    monkeypatch.setattr(generation.campaign.campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(generation.campaign.campaign_evidence, "campaign_from_manifest", lambda _manifest: campaign)
    monkeypatch.setattr(generation.campaign.campaign_evidence, "campaign_run_directory", lambda *_args, **_kwargs: run_directory)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", scheduler_evidence)
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda _batch, case_index, **_kwargs: case_index == tasks[0].case_index,
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "case_failure_is_recorded",
        lambda _batch, case_index, **_kwargs: case_index == tasks[2].case_index,
    )
    monkeypatch.setattr(
        generation.campaign.license_service,
        "latest_attempt_for_job",
        lambda *_args, job_id, **_kwargs: retry if job_id == "104" else None,
    )
    monkeypatch.setattr(generation.campaign.license_service, "retry_attempt_is_eligible", lambda _attempt: True)
    monkeypatch.setattr(
        generation.campaign.progress_service,
        "load_runtime_progress",
        lambda _run_id, job_id, _identity, **_kwargs: (
            {"availability": "available", "phase": "transient_drying", "age_seconds": 5.0, "stale": False}
            if job_id == "102"
            else {"availability": "unavailable", "reason": "not_reported", "age_seconds": None, "stale": None}
        ),
    )

    status = generation.campaign.campaign_status(run_id, storage_root=tmp_path)

    assert query_count["value"] == 1
    assert [case["case_id"] for case in status["cases"]] == [task.case_id for task in tasks]
    assert [case["state"] for case in status["cases"]] == [
        "successful",
        "active",
        "failed",
        "retry_eligible",
        "pending",
        "never_started",
    ]
    active = status["cases"][1]
    assert active["latest_job_id"] == "102"
    assert active["latest_job_name"] == "campaign-0002"
    assert active["scheduler_state"] == "RUNNING"
    assert active["node"] == "node-b"
    assert active["submit_time"] == "2026-01-01T00:01:00"
    assert active["start_time"] == "2026-01-01T00:02:00"
    assert active["elapsed"] == "00:03:04"
    assert active["runtime_progress"]["phase"] == "transient_drying"
    pending = status["cases"][4]
    assert pending["latest_job_id"] == "105"
    assert pending["scheduler_state"] == "PENDING"
    assert pending["node"] is None
    assert pending["start_time"] is None
    assert pending["elapsed"] is None
    assert status["cases"][5]["latest_job_id"] is None
    assert status["cases"][5]["runtime_progress"]["reason"] == "no_job"
    state_with_progress = status["campaign_state"]

    monkeypatch.setattr(
        generation.campaign.progress_service,
        "load_runtime_progress",
        lambda *_args, **_kwargs: {"availability": "unavailable", "reason": "malformed", "age_seconds": None, "stale": None},
    )
    without_progress = generation.campaign.campaign_status(run_id, storage_root=tmp_path)
    assert query_count["value"] == 2
    assert without_progress["campaign_state"] == state_with_progress


def test_conversion_failure_cannot_terminalize_an_active_campaign(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive terminal partial failure only after every active case finishes."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    run_id = "synthetic_campaign__fedcba9876543210"
    run_directory = tmp_path / "campaign"
    run_directory.mkdir()
    manifest = {
        "campaign_run_id": run_id,
        "git_commit": "a" * 40,
        "slurm_job_ids": ["101", "102"],
        "scheduler_job_name": "campaign",
        "scheduler_log_directory": str(run_directory / "scheduler"),
        "submission_config": {"maximum_failed_cases": 0},
        "submission_intent": None,
        "submissions": [],
        "remote_storage_root": str(tmp_path),
        "campaign_meta_directory": str(run_directory),
        "state": "active",
    }

    def view(task: cluster.CampaignTask, state: str) -> dict[str, Any]:
        return {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
            "state": state,
            "quality_flag_count": 0,
            "postprocessing_replay_available": state == "conversion_failed",
        }

    reconciled = {"value": ([view(tasks[0], "conversion_failed"), view(tasks[1], "active")], 0, 1)}
    scheduler = {
        "squeue": {"command": ["squeue"], "output": "", "error": None},
        "sacct": {"command": ["sacct"], "output": "", "error": None},
        "active": {},
        "accounted": {},
    }
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
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: scheduler,
    )
    monkeypatch.setattr(
        generation.campaign,
        "_reconciled",
        lambda *_args, **_kwargs: reconciled["value"],
    )

    running = generation.campaign.campaign_status(run_id, storage_root=tmp_path)

    assert running["campaign_state"] == "running"
    assert running["counts"]["conversion_failed"] == 1
    assert running["counts"]["active"] == 1
    assert running["failure_circuit_breaker_tripped"] is False

    reconciled["value"] = (
        [view(tasks[0], "conversion_failed"), view(tasks[1], "successful")],
        0,
        0,
    )
    terminal = generation.campaign.campaign_status(run_id, storage_root=tmp_path)

    assert terminal["campaign_state"] == "feeding"
    assert terminal["postprocessing_replay_available_cases"] == 1

    reconciled["value"] = (
        [view(tasks[0], "conversion_failed"), view(tasks[1], "successful")],
        0,
        0,
    )
    reconciled["value"][0][0]["postprocessing_replay_available"] = False
    unresolved = generation.campaign.campaign_status(run_id, storage_root=tmp_path)
    assert unresolved["campaign_state"] == "completed_with_failures"


def test_human_summary_shows_two_cases_and_bounds_only_automatic_inventory() -> None:
    """Render every explicit active case while bounding workflow snapshots."""
    status = {
        "campaign_run_id": "transient_smoke__0123456789abcdef",
        "campaign_state": "running",
        "cases": [
            _active_case("case_0001", "610083", "hpc113", 1_328.4),
            _active_case("case_0002", "610084", "hpc114", 662.4),
        ],
    }

    explicit = status_service.format_campaign_status_summary(status)
    bounded = status_service.format_campaign_status_summary(status, max_active_cases=1)

    for value in ("case_0001", "case_0002", "610083", "610084", "hpc113", "hpc114", "transient_drying", "0.075 s", "Tfail=1", "NLfail=25"):
        assert value in explicit
    assert "case_0001" in bounded
    assert "case_0002" not in bounded
    assert "1 additional active case(s) omitted" in bounded


def test_monitor_signatures_separate_phase_changes_from_rate_limited_advancement() -> None:
    """Make phase changes urgent while solver-row advancement stays rate-limited."""
    status = {
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
    status = {
        "campaign_run_id": "transient_smoke__0123456789abcdef",
        "campaign_state": "running",
        "cases": [_active_case("case_0001", "610083", "hpc113", 100.0)],
    }
    monkeypatch.setattr(cli_generation.campaign_runtime, "campaign_status", lambda *_args, **_kwargs: status)

    assert cli_generation.main(["campaign-status", status["campaign_run_id"], "--format", "summary", "--storage-root", str(tmp_path)]) == 0
    summary = capsys.readouterr().out
    assert "Campaign:" in summary
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
