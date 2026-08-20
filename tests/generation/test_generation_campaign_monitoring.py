# ruff: noqa: PLR2004, S101, SLF001
"""Per-case campaign reconciliation and human monitoring contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src import generation
from src.generation import generation_campaign_status as status_service
from src.generation.cli import cli_generation
from src.generation.runtime import generation_runtime_cluster as cluster


def _synthetic_retry_policy(
    *,
    initial_delay_seconds: float = 20.0,
    maximum_delay_seconds: float = 30.0,
) -> dict[str, Any]:
    """Return one test-owned unlimited temporary-license retry policy."""
    return {
        "enabled": True,
        "initial_delay_seconds": initial_delay_seconds,
        "maximum_delay_seconds": maximum_delay_seconds,
        "maximum_wait_seconds": None,
    }


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
        "submission_config": {
            "cores_per_case": 16,
            "max_admission_cases": 3,
            "max_running_cases": None,
            "maximum_failed_cases": 10,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
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
            "101": ["101", "COMPLETED", "0:0", "submit-a", "start-a", "2026-01-01T00:01:00+00:00", "00:01:00", "node-a", "16", "standard"],
            "103": ["103", "FAILED", "1:0", "submit-c", "start-c", "end-c", "00:02:00", "node-c", "16", "standard"],
            "104": ["104", "FAILED", "1:0", "submit-d", "start-d", "end-d", "00:01:00", "node-d", "16", "standard"],
        },
    }
    query_count = {"value": 0}

    def scheduler_evidence(_job_ids: list[str]) -> dict[str, Any]:
        query_count["value"] += 1
        return scheduler

    wait = {
        "retry_budget_remaining": True,
        "first_blocked_at": "2026-01-01T00:00:00+00:00",
        "next_retry_at": "2026-01-01T00:00:20+00:00",
    }
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
        "latest_wait_for_job",
        lambda *_args, job_id, **_kwargs: wait if job_id == "104" else None,
    )
    monkeypatch.setattr(generation.campaign.license_service, "wait_record_is_eligible", lambda _attempt: True)
    monkeypatch.setattr(
        generation.campaign.progress_service,
        "load_runtime_progress",
        lambda _run_id, job_id, _identity, **_kwargs: (
            {
                "availability": "available",
                "phase": "transient_drying",
                "parser_state": "available",
                "age_seconds": 5.0,
                "stale": False,
            }
            if job_id == "102"
            else {"availability": "unavailable", "reason": "not_reported", "age_seconds": None, "stale": None}
        ),
    )

    status = generation.campaign.campaign_status(run_id, storage_root=tmp_path)

    assert query_count["value"] == 1
    assert [case["case_id"] for case in status["cases"]] == [task.case_id for task in tasks]
    assert [case["state"] for case in status["cases"]] == [
        "successful",
        "running",
        "failed",
        "license_blocked",
        "scheduler_pending",
        "never_started",
    ]
    assert status["cases"][0]["completed_at"] == "2026-01-01T00:01:00+00:00"
    assert (
        sum(
            status["work_unit_counts"][state]
            for state in ("successful", "running", "scheduler_pending", "license_blocked", "never_started", "failed")
        )
        == status["work_unit_counts"]["total"]
        == 6
    )
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
    assert status["admission"] == {
        "count": 2,
        "maximum": 3,
        "components": {
            "pending": 1,
            "starting": 0,
            "license_waiting": 1,
            "acquiring_license": 0,
        },
    }
    state_with_progress = status["campaign_state"]

    monkeypatch.setattr(
        generation.campaign.progress_service,
        "load_runtime_progress",
        lambda *_args, **_kwargs: {"availability": "unavailable", "reason": "malformed", "age_seconds": None, "stale": None},
    )
    without_progress = generation.campaign.campaign_status(run_id, storage_root=tmp_path)
    assert query_count["value"] == 2
    assert without_progress["campaign_state"] == state_with_progress
    assert without_progress["admission"]["count"] == 3
    assert without_progress["admission"]["components"]["starting"] == 1


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
        "submission_config": {
            "cores_per_case": 16,
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 0,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
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


def test_replay_blocked_failures_leave_free_normal_admission_unblocked(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report replay failures without overlapping categories or a false admission block."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=12,
        max_admission_cases=2,
        maximum_failed_cases=5,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    run_id = "replay-blocked-status__0123456789abcdef"
    run_directory = tmp_path / "replay-blocked-status"
    run_directory.mkdir()
    manifest = {
        "campaign_run_id": run_id,
        "git_commit": "a" * 40,
        "slurm_job_ids": [],
        "scheduler_job_name": "campaign",
        "scheduler_log_directory": str(run_directory / "scheduler"),
        "submission_config": {
            "cores_per_case": 16,
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 5,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "submissions": [],
        "state": "completed_with_failures",
        "remote_storage_root": str(tmp_path),
        "campaign_meta_directory": str(run_directory),
    }
    views = [
        {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
            "state": "conversion_failed" if index < 2 else "never_started",
            "failure_stage": "conversion" if index < 2 else None,
            "quality_flag_count": 0,
            "temporary_license_retry": None,
            "license_retry_eligible": False,
            "postprocessing_replay_available": index < 2,
            "replay_eligible": False,
            "replay_blocked": index < 2,
        }
        for index, task in enumerate(tasks)
    ]
    monkeypatch.setattr(generation.campaign.campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(generation.campaign.campaign_evidence, "campaign_from_manifest", lambda _manifest: campaign)
    monkeypatch.setattr(generation.campaign.campaign_evidence, "campaign_run_directory", lambda *_args, **_kwargs: run_directory)
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: {
            "squeue": {"command": [], "output": "", "error": None},
            "sacct": {"command": [], "output": "", "error": None},
            "active": {},
            "accounted": {},
        },
    )
    monkeypatch.setattr(generation.campaign, "_reconciled", lambda *_args, **_kwargs: (views, 0, 0))

    status = generation.campaign.campaign_status(run_id, storage_root=tmp_path)

    assert status["campaign_state"] == "feeding"
    assert status["admission_blocked"] is False
    assert status["admission_block_reason"] is None
    assert status["failure_counts"]["conversion_failed"] == 2
    assert status["failure_counts"]["replay_blocked"] == 2
    assert status["work_unit_counts"]["failed"] == 2
    assert status["work_unit_counts"]["never_started"] == 10
    assert status["work_unit_counts"]["total"] == 12


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
    assert "displayed=1  additional=1" in bounded


def test_human_summary_exposes_pinned_execution_resources() -> None:
    """Render the persisted source and allocation identity for operators."""
    status = {
        "campaign_run_id": "transient_campaign__0123456789abcdef",
        "campaign_state": "running",
        "git_commit": "a" * 40,
        "execution_config_digest": "b" * 64,
        "submission_config": {
            "cores_per_case": 8,
            "max_admission_cases": 2,
            "max_running_cases": 3,
        },
        "admission": {
            "count": 2,
            "maximum": 2,
            "components": {
                "pending": 0,
                "starting": 0,
                "license_waiting": 2,
                "acquiring_license": 0,
            },
        },
        "cases": [],
    }

    rendered = status_service.format_campaign_status_summary(status)

    assert f"commit={'a' * 40}" in rendered
    assert f"config_digest={'b' * 64}" in rendered
    assert "cores_per_case=8" in rendered
    assert "max_admission_cases=2" in rendered
    assert "max_running_cases=3" in rendered
    assert "Admission: 2/2" in rendered
    assert "pending=0  starting=0  acquiring_license=0  license_waiting=2" in rendered
    assert "stagger=" not in rendered

    status["submission_config"]["max_running_cases"] = None
    assert "max_running_cases=unlimited" in status_service.format_campaign_status_summary(status)


def test_human_summary_compacts_active_and_exhausted_license_windows() -> None:
    """Show bounded operational counts without retained FlexNet excerpts."""
    status = {
        "campaign_run_id": "license-windows__0123456789abcdef",
        "campaign_state": "running",
        "admission": {
            "count": 2,
            "maximum": 2,
            "components": {
                "pending": 0,
                "starting": 0,
                "acquiring_license": 1,
                "license_waiting": 1,
            },
        },
        "cases": [
            {
                "case_id": "case_0001",
                "batch_name": "transient_drying__lentil__natural",
                "state": "running",
                "runtime_progress": {
                    "availability": "available",
                    "phase": "acquiring_comsol_license",
                    "license_window_seconds": 47.0,
                    "license_window_limit_seconds": 120.0,
                    "license_checkout_attempt_count": 4,
                    "last_license_result": "temporary_license_capacity",
                },
            },
            {
                "case_id": "case_0002",
                "batch_name": "transient_drying__lentil__natural",
                "state": "license_blocked",
                "reason": "in_allocation_license_window_exhausted",
                "in_allocation_license_window": {
                    "realised_window_seconds": 120.3,
                    "checkout_attempt_count": 8,
                    "raw_excerpt": "Licensed number of users already reached",
                },
                "temporary_license_retry": {
                    "retry_count": 2,
                    "next_retry_at": "2026-08-20T00:00:15+00:00",
                    "cumulative_wait_seconds": 45.0,
                },
            },
        ],
    }

    rendered = status_service.format_campaign_status_summary(status)

    assert "phase=acquiring_comsol_license" in rendered
    assert "window=47 s / 120 s" in rendered
    assert "checkouts=4" in rendered
    assert "reason=in_allocation_license_window_exhausted" in rendered
    assert "window=120 s  checkouts=8" in rendered
    assert "Licensed number of users" not in rendered
    assert "{" not in rendered


def test_human_summary_keeps_license_capacity_out_of_failed_cases() -> None:
    """Render temporary licence capacity as an operational blocked section."""
    status = {
        "campaign_run_id": "transient_campaign__0123456789abcdef",
        "campaign_state": "license_blocked",
        "cases": [
            {
                "batch_name": "transient_drying__lentil__natural",
                "case_id": "case_0001",
                "state": "license_blocked",
                "reason": "temporary_license_capacity",
            }
        ],
    }

    rendered = status_service.format_campaign_status_summary(status)

    assert "License-blocked cases:" in rendered
    assert "state=license_blocked" in rendered
    assert "Failed cases:" not in rendered
    assert "failed=0" in rendered


def test_human_summary_compacts_failed_replay_evidence() -> None:
    """Keep actionable replay state while omitting internal replay metadata."""
    status = {
        "campaign_run_id": "replay-blocked__0123456789abcdef",
        "campaign_state": "feeding",
        "cases": [
            {
                "batch_name": "transient_drying__chickpea__natural",
                "case_id": "case_0001",
                "state": "failed",
                "classified_state": "conversion_failed",
                "reason": "postprocessing replay failed " + "with internal detail " * 30,
                "failure_stage": "conversion",
                "solver_state": "succeeded",
                "postprocessing_state": "replay_blocked",
                "replay_eligible": False,
                "replay_running": False,
                "replay_blocked": True,
                "replay_block_reason": "unchanged_replay_identity",
                "replay_attempt_count": 1,
                "replay_evidence_path": "/storage/01_generation/attempts/replay_failure.json",
                "automatic_continuation_allowed": False,
            },
            {
                "batch_name": "transient_drying__chickpea__natural",
                "case_id": "case_0002",
                "state": "never_started",
            },
        ],
    }

    rendered = status_service.format_campaign_status_summary(status)

    for value in (
        "state=conversion_failed",
        "stage=conversion",
        "solver=succeeded",
        "replay=blocked",
        "evidence=.../01_generation/attempts/replay_failure.json",
    ):
        assert value in rendered
    for internal in (
        "postprocessing_state",
        "replay_eligible",
        "replay_running",
        "replay_blocked",
        "replay_block_reason",
        "replay_attempt_count",
        "replay_evidence_path",
        "automatic_continuation_allowed",
    ):
        assert internal not in rendered
    reason_line = next(line for line in rendered.splitlines() if "reason=" in line)
    assert len(reason_line) <= 171
    assert max(map(len, rendered.splitlines())) <= 180


def test_human_summary_bounds_license_rows_and_omits_machine_evidence() -> None:
    """Show retry actions while retaining raw license evidence only in JSON."""
    cases = [
        {
            "batch_name": "transient_drying__kidney_bean__natural",
            "case_id": f"case_{index:04d}",
            "state": "license_blocked",
            "reason": "temporary_license_capacity",
            "latest_job_id": str(629_820 + index),
            "temporary_license_retry": {
                "feature": "Equilibrium Moisture Transport in Porous Media",
                "error_code": "-4,132",
                "retry_count": index + 1,
                "next_retry_at": "2026-08-20T00:07:09Z",
                "cumulative_wait_seconds": 60,
                "matched_signatures": ["licensed number of users already reached"],
                "raw_excerpt": "FlexNet error " * 500,
                "license_path": "/complete/internal/license/path",
            },
        }
        for index in range(22)
    ]

    rendered = status_service.format_campaign_status_summary({"campaign_run_id": "license-run", "campaign_state": "running", "cases": cases})

    for value in (
        "state=license_blocked",
        "retry=1",
        "next_retry=2026-08-20T00:07:09Z",
        "cumulative_wait=60 s",
        "displayed=20  additional=2",
    ):
        assert value in rendered
    for internal in ("raw_excerpt", "matched_signatures", "FlexNet error", "license_path", "[", "{"):
        assert internal not in rendered
    assert max(map(len, rendered.splitlines())) <= 180


def test_running_summary_keeps_progress_without_replay_metadata() -> None:
    """Keep useful solver progress and omit unrelated replay evidence."""
    case = _active_case("case_0002", "629565", "hpc119", 125.496)
    case.update(
        {
            "replay_eligible": False,
            "replay_running": False,
            "replay_blocked": False,
            "replay_attempt_count": 0,
            "postprocessing_state": "not_applicable",
        }
    )
    case["runtime_progress"]["last_solver_log_update_at"] = "2026-08-19T19:11:41Z"

    rendered = status_service.format_campaign_status_summary({"campaign_run_id": "running", "campaign_state": "running", "cases": [case]})

    for value in (
        "job=629565",
        "node=hpc119",
        "phase=transient_drying",
        "progress=4%",
        "simulated_time=0.03486 h",
        "step=220",
        "step_size=0.075 s",
        "Tfail=1",
        "NLfail=25",
        "last_solver_update=2026-08-19T19:11:41Z",
        "age=4 s ago",
    ):
        assert value in rendered
    for internal in ("replay_eligible", "replay_running", "replay_blocked", "replay_attempt_count", "postprocessing_state"):
        assert internal not in rendered


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


def test_common_status_categories_are_disjoint_and_solver_progress_is_phase_bound() -> None:
    """Count every work unit once and hide stale solver values after solving."""
    cases = [
        {"case_id": "success", "batch_name": "batch", "state": "successful"},
        _active_case("case_0001", "101", "node-a", 120.0),
        {"case_id": "queued", "batch_name": "batch", "state": "pending", "latest_job_id": "102", "scheduler_state": "PENDING"},
        {"case_id": "blocked", "batch_name": "batch", "state": "license_blocked"},
        {"case_id": "unsent", "batch_name": "batch", "state": "never_started", "latest_job_id": None},
        {"case_id": "failed", "batch_name": "batch", "state": "failed"},
    ]
    status = {"campaign_run_id": "run", "campaign_state": "running", "cases": cases}
    rendered = status_service.format_campaign_status_summary(status)
    for category in ("successful", "running", "scheduler_pending", "license_blocked", "never_started", "failed"):
        assert f"{category}=1" in rendered
    assert "total=6" in rendered

    exporting = deepcopy(cases[1])
    exporting["runtime_progress"]["phase"] = "collecting_exports"
    exporting_text = status_service.format_campaign_status_summary({**status, "cases": [exporting]})
    assert "phase=collecting_exports" in exporting_text
    assert "progress=" not in exporting_text
    assert "simulated_time=" not in exporting_text
    assert "step=" not in exporting_text
    assert "Tfail=" not in exporting_text


def test_admission_waiting_is_distinct_from_never_started_in_status() -> None:
    """Present a durable logical admission without implying scheduler work."""
    waiting = {
        "case_id": "case_0001",
        "batch_name": "transient_drying__lentil__natural",
        "state": "admission_waiting",
    }
    unsent = {
        "case_id": "case_0002",
        "batch_name": "transient_drying__lentil__natural",
        "state": "never_started",
    }

    rendered = status_service.format_campaign_status_summary(
        {
            "campaign_run_id": "reserved-admission",
            "campaign_state": "license_blocked",
            "cases": [waiting, unsent],
        }
    )

    assert "admission_waiting=1" in rendered
    assert "never_started=1" in rendered
    never_started_section = rendered.split("Never started:", maxsplit=1)[1]
    assert "total: 1" in never_started_section


def _successful_case(
    index: int,
    *,
    completed_at: str,
    simulated_end_time: float | None = None,
    final_moisture_value: float | None = None,
) -> dict[str, Any]:
    """Return one synthetic successful campaign-status row."""
    return {
        "case_id": f"case_{index:04d}",
        "batch_name": "transient_drying__lentil__natural",
        "material": "lentil",
        "state": "successful",
        "reason": "validated_case_evidence",
        "completed_at": completed_at,
        "elapsed": "99:59:59",
        "simulated_end_time": simulated_end_time,
        "simulated_end_time_unit": "h" if simulated_end_time is not None else None,
        "final_moisture_name": "f_wet_dm_final" if final_moisture_value is not None else None,
        "final_moisture_value": final_moisture_value,
        "final_moisture_unit": "1" if final_moisture_value is not None else None,
    }


@pytest.mark.parametrize(
    ("completed_count", "displayed_count"),
    [(0, 0), (1, 1), (3, 3), (4, 3), (600, 3), (1_050, 3)],
)
def test_recent_completions_are_bounded_at_campaign_scale(
    completed_count: int,
    displayed_count: int,
) -> None:
    """Keep the full success count while rendering at most three details."""
    cases = [
        _successful_case(index, completed_at=f"2026-01-{1 + index // 86:02d}T{index % 24:02d}:00:00+00:00") for index in range(1, completed_count + 1)
    ]

    rendered = status_service.format_campaign_status_summary({"campaign_run_id": "bounded-completions", "campaign_state": "running", "cases": cases})

    assert f"successful={completed_count}" in rendered
    assert rendered.count("  state=successful") == displayed_count
    if completed_count == 0:
        assert "Recently completed cases" not in rendered
    else:
        assert f"Recently completed cases (latest {displayed_count} of {completed_count}):" in rendered
    assert "additional completed cases omitted" not in rendered


def test_recent_completions_use_timestamp_then_stable_plan_order() -> None:
    """Select newest terminal timestamps and preserve plan order for exact ties."""
    cases = [
        _successful_case(4, completed_at="2026-01-01T00:04:00+00:00"),
        _successful_case(2, completed_at="2026-01-01T00:05:00+00:00"),
        _successful_case(1, completed_at="2026-01-01T00:05:00+00:00"),
        _successful_case(3, completed_at="2026-01-01T00:03:00+00:00"),
    ]

    rendered = status_service.format_campaign_status_summary({"campaign_run_id": "ordered-completions", "campaign_state": "running", "cases": cases})
    section = rendered.split("Recently completed cases", maxsplit=1)[1]

    assert [section.index(case_id) for case_id in ("case_0002", "case_0001", "case_0004")] == sorted(
        section.index(case_id) for case_id in ("case_0002", "case_0001", "case_0004")
    )
    assert "case_0003" not in section


def test_new_completion_replaces_the_oldest_displayed_case() -> None:
    """Drop the previous oldest row when a newer terminal completion arrives."""
    cases = [_successful_case(index, completed_at=f"2026-01-01T00:0{index}:00+00:00") for index in range(1, 4)]
    initial = status_service.format_campaign_status_summary({"campaign_run_id": "rolling-completions", "campaign_state": "running", "cases": cases})
    updated = status_service.format_campaign_status_summary(
        {
            "campaign_run_id": "rolling-completions",
            "campaign_state": "running",
            "cases": [*cases, _successful_case(4, completed_at="2026-01-01T00:04:00+00:00")],
        }
    )

    assert "case_0001" in initial
    assert "case_0001" not in updated
    assert "case_0004" in updated
    assert updated.count("  state=successful") == 3


def test_completed_formatting_is_pure_and_optional_terminal_fields_are_conditional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use only supplied status fields and omit unavailable terminal science."""

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        message = "Formatting attempted external or filesystem work"
        raise AssertionError(message)

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(status_service.common.serialization, "file_sha256", forbidden)
    available = _successful_case(
        1,
        completed_at="2026-01-01T00:01:00+00:00",
        simulated_end_time=168.0,
        final_moisture_value=0.125,
    )
    unavailable = _successful_case(2, completed_at="2026-01-01T00:02:00+00:00")

    rendered = status_service.format_campaign_status_summary(
        {"campaign_run_id": "terminal-fields", "campaign_state": "running", "cases": [available, unavailable]}
    )

    assert "simulated_end=168 h" in rendered
    assert "f_wet_dm_final=0.125 1" in rendered
    assert "elapsed=99:59:59" not in rendered
    assert "simulated_end=unavailable" not in rendered
    assert "final_moisture=unavailable" not in rendered
    assert max(map(len, rendered.splitlines())) < 220


def test_human_summary_groups_never_started_campaigns_at_material_scale() -> None:
    """Scale never-started output with ordered materials rather than case count."""
    campaign_shapes = (
        (18, (("field_pea", 3), ("rapeseed", 3), ("sunflower_seed", 3))),
        (600, (("lentil", 120), ("chickpea", 120), ("field_pea", 120), ("rapeseed", 120), ("sunflower_seed", 120))),
        (1_050, (("lentil", 210), ("chickpea", 210), ("field_pea", 210), ("rapeseed", 210), ("sunflower_seed", 210))),
    )
    for total, groups in campaign_shapes:
        never_started_count = sum(count for _material, count in groups)
        successful_count = total - never_started_count
        cases = [{"case_id": f"successful_{index:04d}", "batch_name": "completed", "state": "successful"} for index in range(successful_count)]
        for material, count in groups:
            cases.extend(
                {
                    "case_id": f"{material}_{index:04d}",
                    "batch_name": f"transient_drying__{material}__natural",
                    "material": material,
                    "state": "never_started",
                }
                for index in range(count)
            )

        rendered = status_service.format_campaign_status_summary(
            {"campaign_run_id": f"campaign-{total}", "campaign_state": "running", "cases": cases}
        )
        never_started = rendered.split("Never started:\n", maxsplit=1)[1]

        assert f"never_started={never_started_count}" in rendered
        assert f"total={total}" in rendered
        assert [never_started.index(f"{material}: {count}") for material, count in groups] == sorted(
            never_started.index(f"{material}: {count}") for material, count in groups
        )
        assert f"total: {never_started_count}" in never_started
        assert not any(f"{material}_0000" in never_started for material, _count in groups)
        assert len(never_started.splitlines()) == len(groups) + 1


def test_benchmark_summary_uses_bounded_common_work_unit_vocabulary() -> None:
    """Render benchmark work units in supplied resolved order without inventory expansion."""
    status = {
        "suite_name": "core scaling",
        "benchmark_run_id": "benchmark",
        "state": "running",
        "wave_count": 2,
        "current_wave": {"wave_position": 1, "variant_id": "cores_4", "cores_per_case": 4},
        "work_units": [
            {"variant_id": "cores_4", "case_role": "canary", "work_unit_id": "four-canary", "state": "running"},
            {"variant_id": "cores_4", "case_role": "measurement", "work_unit_id": "four-measurement", "state": "never_started"},
            {"variant_id": "cores_8", "case_role": "canary", "work_unit_id": "eight-canary", "state": "never_started"},
        ],
        "partial_evaluation": {"variants": [], "recommended_cores_per_case": 4},
    }

    rendered = status_service.format_benchmark_status_summary(status, max_active_cases=1)

    assert "successful=0" in rendered
    assert "running=1" in rendered
    assert "never_started=2" in rendered
    assert "four-canary" in rendered
    assert "Partial wave evaluation (provisional):" in rendered


def test_pilot_license_observability_uses_existing_solver_and_wait_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report successful overlap and license delays without submitting work."""
    batch = SimpleNamespace(
        batch_id="batch-id",
        batch_storage_name="batch-storage",
        case_indices=(1, 2),
        case_id=lambda index: f"case_{index:04d}",
    )
    campaign = SimpleNamespace(batches=(batch,))
    views = [{"batch_id": "batch-id", "case_index": index, "state": "successful"} for index in (1, 2)]
    processed_roots: dict[int, Path] = {}
    intervals = {
        1: ("2026-01-01T00:00:00+00:00", "2026-01-01T00:10:00+00:00"),
        2: ("2026-01-01T00:05:00+00:00", "2026-01-01T00:15:00+00:00"),
    }
    for case_index, (started_at, ended_at) in intervals.items():
        directory = tmp_path / f"processed-{case_index}"
        directory.mkdir()
        (directory / "execution_provenance.json").write_text(
            json.dumps({"result": {"started_at": started_at, "ended_at": ended_at}}),
            encoding="utf-8",
        )
        processed_roots[case_index] = directory
    attempt_root = tmp_path / "attempts"
    retry_count = 17
    for attempt_index in range(1, retry_count + 1):
        receipt = attempt_root / f"attempt_{attempt_index:04d}" / "attempt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "campaign_run_id": "pilot-run",
                    "batch_id": "batch-id",
                    "case_id": "case_0001",
                    "case_state": "license_blocked",
                    "attempt_index": attempt_index,
                    "job_id": str(500 + attempt_index),
                    "elapsed_seconds": 3.5,
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "processed_case_directory",
        lambda _batch, case_index, **_kwargs: processed_roots[case_index],
    )
    monkeypatch.setattr(
        generation.campaign.license_service,
        "load_temporary_license_wait",
        lambda _batch, case_index, **_kwargs: (
            {
                "retry_count": retry_count,
                "cumulative_wait_seconds": 1_020.0,
                "feature": "COMSOL Multiphysics",
                "error_code": "-4,132",
                "recent_job_ids": [str(job_id) for job_id in range(502, 518)],
            }
            if case_index == 1
            else None
        ),
    )
    monkeypatch.setattr(
        generation.campaign.common.paths,
        "resolve_generation_attempt_case_directory",
        lambda *_args, **_kwargs: attempt_root,
    )

    observed = generation.campaign._pilot_license_observability(
        cast("generation.cases.config.CampaignConfig", campaign),
        views,
        run_id="pilot-run",
        storage_root=tmp_path,
    )

    assert observed == {
        "observed_peak_solver_concurrency": 2,
        "successful_solver_start_count": 2,
        "license_blocked_submission_count": 17,
        "accumulated_license_wait_seconds": 1_020.0,
        "accumulated_license_probe_seconds": 59.5,
        "detected_license_features": ["COMSOL Multiphysics"],
        "detected_license_error_codes": ["-4,132"],
        "observed_license_concurrency_lower_bound": 2,
    }
    rendered = status_service.format_campaign_status_summary(
        {
            "campaign_run_id": "pilot-run",
            "campaign_state": "successful",
            "cases": [],
            "license_observability": observed,
        }
    )
    assert "Material-pilot license observability" not in rendered
    assert "observed_license_concurrency_lower_bound" not in rendered
    assert "detected_license_features" not in rendered


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

    assert "Final validated benchmark summary" in rendered
    assert "/evidence/summary.md" in rendered
    assert "# Validated benchmark result" in rendered
