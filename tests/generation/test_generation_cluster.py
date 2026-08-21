# ruff: noqa: S101, PLR2004, SLF001
"""Dynamic per-case feeder, Slurm identity, and transfer contracts."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from src import common, generation
from src.generation.cli import cli_generation
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_workspace as workspace


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


def _scheduler(
    *,
    active: dict[str, list[str]] | None = None,
    accounted: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return one successful synthetic scheduler reconciliation view."""
    active_rows = {} if active is None else active
    accounted_rows = {} if accounted is None else accounted
    return {
        "squeue": {"command": ["squeue"], "output": "", "error": None},
        "sacct": {"command": ["sacct"], "output": "", "error": None},
        "active": active_rows,
        "accounted": accounted_rows,
    }


def _synthetic_input_references(campaign: Any) -> dict[str, dict[int, Any]]:
    """Return opaque references for tests that replace content validation."""
    references: dict[str, dict[int, Any]] = {}
    for task in cluster.campaign_tasks(campaign):
        references.setdefault(task.batch_name, {})[task.case_index] = object()
    return references


def test_one_case_submission_and_local_only_concurrency(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect ordinary one-case jobs and isolate local development concurrency."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    assert len(tasks) == campaign.total_case_count
    assert len({(task.batch_id, task.case_id) for task in tasks}) == len(tasks)

    task = tasks[0]
    command = cluster.build_campaign_case_slurm_submission_command(
        campaign,
        task,
        run_id="synthetic__0123456789abcdef",
        scheduler_log_directory=tmp_path.resolve(),
        scheduler_job_name="vp2-synthetic-0001",
        attempt_index=1,
    )
    assert command[0] == "sbatch"
    assert "--nodes=1" in command
    assert "--ntasks=1" in command
    assert f"--cpus-per-task={campaign.execution_values['cluster']['cores_per_case']}" in command
    assert not any(argument.startswith("--array") for argument in command)
    assert "--exclusive" not in command
    assert f"--output={tmp_path.resolve()}/slurm-%j.out" in command
    wrapped = command[-1]
    assert wrapped.startswith("--wrap=")
    worker_arguments = shlex.split(wrapped.removeprefix("--wrap="))
    launcher_index = next(index for index, argument in enumerate(worker_arguments) if argument.endswith("generation_campaign_node.sh"))
    launcher = Path(worker_arguments[launcher_index])
    assert worker_arguments[launcher_index + 1] == str(launcher.parents[1])
    assert task.batch_name in worker_arguments
    assert str(task.case_index) in worker_arguments
    assert "GENERATION_ATTEMPT_INDEX=1" in worker_arguments

    plan = cluster.build_local_resource_plan(
        cores_per_case=1,
        max_parallel_cases=2,
        remaining_cases=5,
    )
    assert plan.effective_parallel_cases == 2
    assert not hasattr(plan, "max_nodes")
    assert not hasattr(plan, "cases_per_node")


def test_scheduler_argv_and_duplicate_safe_campaign_reconciliation(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise first submission and exact persisted-job reconciliation."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=3,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    commit = "8" * 40
    storage = tmp_path / "storage"
    calls: list[list[str]] = []
    submitted_ids = iter(("12345", "12346"))
    scheduler_mode = {"value": "active"}

    def fake_scheduler(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        calls.append(arguments)
        if arguments[0] == "sbatch":
            discovery = generation.cases.admission.discover_input_batches(storage)
            assert not discovery.issues
            assert sum(len(source.cases) for source in discovery.sources) == campaign.total_case_count
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=f"{next(submitted_ids)}\n",
                stderr="",
            )
        if arguments[0] == "squeue":
            assert arguments == [
                "squeue",
                "--noheader",
                "--jobs=12345",
                "--format=%i|%T|%R|%N|%V|%S|%M",
            ]
            output = "12345|PENDING|Resources|\n" if scheduler_mode["value"] == "active" else ""
            return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")
        if arguments[0] == "sacct":
            assert arguments == [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                "12345",
                "--format=JobIDRaw,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocCPUS,Partition",
            ]
            output = "" if scheduler_mode["value"] == "active" else "12345|COMPLETED|0:0\n"
            return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")
        raise AssertionError(arguments)

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign.subprocess, "run", fake_scheduler)

    first_progress: list[dict[str, Any]] = []
    first = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
        progress=lambda event: first_progress.append(dict(event)),
    )
    assert first["slurm_job_ids"] == ["12345"]
    assert [command[0] for command in calls] == ["sbatch"]
    persisted = campaign_evidence.load_campaign_run(
        first["campaign_run_id"],
        storage_root=storage,
    )
    assert persisted["submissions"][0]["job_id"] == "12345"
    assert [event["scheduler_submissions"] for event in first_progress if event.get("operation") == "scheduler_submission"] == [1]

    blocked = AssertionError("Current canonical inputs were regenerated before resubmission.")
    active_progress: list[dict[str, Any]] = []
    with monkeypatch.context() as scoped:
        scoped.setattr(
            generation.cases.input_generation.case_service,
            "generate_case_input_bundle",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(blocked),
        )
        active = generation.campaign.submit_campaign(
            campaign,
            git_commit=commit,
            storage_root=storage,
            progress=lambda event: active_progress.append(dict(event)),
        )
    assert active["slurm_job_ids"] == ["12345"]
    assert [command[0] for command in calls].count("sbatch") == 1
    assert not any(event.get("operation") == "scheduler_submission" for event in active_progress)

    unchanged_progress: list[dict[str, Any]] = []
    unchanged = generation.campaign.feed_campaign(
        first["campaign_run_id"],
        storage_root=storage,
        progress=lambda event: unchanged_progress.append(dict(event)),
    )
    assert unchanged["slurm_job_ids"] == ["12345"]
    assert [command[0] for command in calls].count("sbatch") == 1
    assert not any(event.get("operation") == "scheduler_submission" for event in unchanged_progress)

    scheduler_mode["value"] = "completed"
    first_case = tasks[0]
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda batch, case_index, **_kwargs: (batch.batch_name, case_index) == (first_case.batch_name, first_case.case_index),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_successful_status_summary",
        lambda *_args, **_kwargs: generation.campaign._empty_successful_status_summary(),
    )
    advanced_progress: list[dict[str, Any]] = []
    advanced = generation.campaign.feed_campaign(
        first["campaign_run_id"],
        storage_root=storage,
        progress=lambda event: advanced_progress.append(dict(event)),
    )
    assert advanced["slurm_job_ids"] == ["12345", "12346"]
    assert advanced["submissions"][1]["case"]["case_id"] == tasks[1].case_id
    assert [event["scheduler_submissions"] for event in advanced_progress if event.get("operation") == "scheduler_submission"] == [2]


def test_invalid_current_inputs_abort_before_campaign_persistence_or_sbatch(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid current evidence before any scientific job is submitted."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    commit = "7" * 40
    storage = tmp_path / "storage"
    monkeypatch.setenv("GENERATION_GIT_COMMIT", commit)
    generated = generation.cases.input_generation.generate_input_cases(
        batch,
        1,
        storage_root=storage,
    )
    manifest_path = generated.metadata_directory / "input_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "publishing"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scheduler_calls: list[list[str]] = []

    def reject_scheduler(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        scheduler_calls.append(command)
        message = "Invalid canonical inputs reached scheduler submission."
        raise AssertionError(message)

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign.subprocess, "run", reject_scheduler)
    run_id = generation.campaign.campaign_run_id(campaign, git_commit=commit)

    with pytest.raises(FileExistsError, match="incomplete or invalid"):
        generation.campaign.submit_campaign(
            campaign,
            git_commit=commit,
            storage_root=storage,
        )

    assert scheduler_calls == []
    assert not campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    ).exists()


def test_license_retry_waits_then_resubmits_the_same_case_once(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hold same-key work until one oldest probe proves solver startup."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=2,
        max_admission_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    commit = "9" * 40
    storage = tmp_path / "storage"
    submitted_ids = iter(("4101", "4102", "4103"))
    submit_commands: list[list[str]] = []
    retry_eligible = {"value": False}
    retry_active = {"value": False}
    probe_progress = {"value": False}
    completed_cases: set[int] = set()
    wait_record = {
        "classification": "temporary_license_capacity",
        "retry_budget_remaining": True,
        "first_blocked_at": "2026-01-01T00:00:00+00:00",
        "next_retry_at": "2026-01-01T00:00:20+00:00",
    }

    def fake_submit(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        assert arguments[0] == "sbatch"
        submit_commands.append(arguments)
        job_id = next(submitted_ids)
        if job_id == "4102":
            retry_active["value"] = True
        return subprocess.CompletedProcess(arguments, 0, stdout=f"{job_id}\n", stderr="")

    def scheduler_evidence(job_ids: list[str]) -> dict[str, Any]:
        active = {"4102": ["4102", "RUNNING"]} if retry_active["value"] and "4102" in job_ids else {}
        return _scheduler(
            active=active,
            accounted={job_id: [job_id, "COMPLETED"] for job_id in job_ids if job_id not in active},
        )

    def latest_wait(
        *_args: Any,
        job_id: str,
        **_kwargs: Any,
    ) -> dict[str, Any] | None:
        return wait_record if job_id == "4101" else None

    def runtime_progress(_run_id: str, job_id: str, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if job_id == "4102" and probe_progress["value"]:
            return {
                "availability": "available",
                "phase": "transient_drying",
                "parser_state": "available",
            }
        return {
            "availability": "unavailable",
            "reason": "not_reported",
            "age_seconds": None,
            "stale": None,
        }

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", scheduler_evidence)
    monkeypatch.setattr(generation.campaign.subprocess, "run", fake_submit)
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda _batch, case_index, **_kwargs: case_index in completed_cases,
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "case_failure_is_recorded",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        generation.campaign.license_service,
        "latest_wait_for_job",
        latest_wait,
    )
    monkeypatch.setattr(
        generation.campaign.license_service,
        "wait_record_is_eligible",
        lambda _attempt: retry_eligible["value"],
    )
    monkeypatch.setattr(
        generation.campaign.license_service,
        "load_in_allocation_license_window",
        lambda *_args, job_id, **_kwargs: (
            {
                "outcome": "window_exhausted",
                "reason": "in_allocation_license_window_exhausted",
            }
            if job_id == "4101"
            else None
        ),
    )
    monkeypatch.setattr(
        generation.campaign.progress_service,
        "load_runtime_progress",
        runtime_progress,
    )
    monkeypatch.setattr(
        generation.campaign,
        "_finalize_completed_batches",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation.campaign,
        "_successful_status_summary",
        lambda *_args, **_kwargs: generation.campaign._empty_successful_status_summary(),
    )

    initial = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    run_id = str(initial["campaign_run_id"])
    assert initial["slurm_job_ids"] == ["4101"]

    waiting = generation.campaign.feed_campaign(run_id, storage_root=storage)
    assert waiting["state"] == "license_blocked"
    assert waiting["slurm_job_ids"] == ["4101"]

    retry_eligible["value"] = True
    retry_snapshots: list[dict[str, Any]] = []
    retried = generation.campaign.resume_campaign(
        run_id,
        storage_root=storage,
        status_callback=lambda status: retry_snapshots.append(dict(status)),
    )
    assert retried["slurm_job_ids"] == ["4101", "4102"]
    assert [record["mode"] for record in retried["submissions"]] == [
        "initial",
        "license_retry",
    ]
    assert retried["submissions"][1]["case"] == retried["submissions"][0]["case"]
    assert len(retry_snapshots) == 1
    retry_snapshot = retry_snapshots[0]
    retry_case = next(case for case in retry_snapshot["cases"] if case["case_id"] == tasks[0].case_id)
    assert retry_case["state"] == "scheduler_pending"
    assert retry_case["classified_state"] == "pending"
    assert retry_case["reason"] == "scheduler_snapshot_predates_submission"
    assert retry_case["automatic_continuation_allowed"] is True
    assert retry_snapshot["failed_cases"] == 0
    assert retry_snapshot["work_unit_counts"]["failed"] == 0
    assert retry_snapshot["failure_circuit_breaker_tripped"] is False
    assert retry_snapshot["admission_blocked"] is True
    assert retry_snapshot["admission_block_reason"] == "max_admission_cases_reached"

    visible_snapshots: list[dict[str, Any]] = []
    unresolved = generation.campaign.resume_campaign(
        run_id,
        storage_root=storage,
        status_callback=lambda status: visible_snapshots.append(dict(status)),
    )
    assert unresolved["slurm_job_ids"] == ["4101", "4102"]
    assert len(submit_commands) == 2
    assert len(visible_snapshots) == 1
    visible_case = next(case for case in visible_snapshots[0]["cases"] if case["case_id"] == tasks[0].case_id)
    assert visible_case["state"] == "running"
    assert visible_case["classified_state"] == "active"
    assert visible_snapshots[0]["failed_cases"] == 0
    assert visible_snapshots[0]["failure_circuit_breaker_tripped"] is False

    probe_progress["value"] = True
    released = generation.campaign.feed_campaign(run_id, storage_root=storage)
    assert released["slurm_job_ids"] == ["4101", "4102", "4103"]
    assert released["submissions"][-1]["case"]["case_id"] == tasks[1].case_id
    assert released["submissions"][-1]["mode"] == "initial"

    retry_active["value"] = False
    completed_cases.update(task.case_index for task in tasks)
    terminal = generation.campaign.feed_campaign(run_id, storage_root=storage)
    assert terminal["state"] == "complete"


def test_stale_failure_allows_fresh_submission_without_active_job_duplication(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignore a prior execution failure while preserving active-job priority."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=2,
        max_running_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    storage = tmp_path / "stale campaign storage"
    source_commit = "7" * 40
    monkeypatch.setenv("GENERATION_GIT_COMMIT", source_commit)
    monkeypatch.setenv(
        "GENERATION_CAMPAIGN_RUN_ID",
        "source-campaign__0123456789abcdef",
    )
    generation.cases.input_generation.generate_input_cases(
        batch,
        len(batch.case_indices),
        storage_root=storage,
    )
    generation.runtime.record_case_failure(
        batch,
        task.case_index,
        RuntimeError("prior synthetic failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node-source",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )

    current_commit = "8" * 40
    current_run_id = generation.campaign.campaign_run_id(
        campaign,
        git_commit=current_commit,
    )
    monkeypatch.setenv("GENERATION_GIT_COMMIT", current_commit)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", current_run_id)
    scheduler = _scheduler()
    submissions: list[list[str]] = []

    def submit(command: list[str], **_kwargs: Any) -> str:
        submissions.append(command)
        return "321"

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: current_commit)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: scheduler)
    monkeypatch.setattr(generation.campaign, "_submit_case", submit)

    manifest = generation.campaign.submit_campaign(
        campaign,
        git_commit=current_commit,
        storage_root=storage,
    )
    assert manifest["slurm_job_ids"] == ["321"]
    assert len(submissions) == 1

    scheduler["active"] = {"321": ["321", "RUNNING", "node-new", "node-new"]}
    active = generation.campaign.feed_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert active["slurm_job_ids"] == ["321"]
    assert len(submissions) == 1


def test_current_failure_is_not_automatically_retried(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never retry a failed scientific case during same-config continuation."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=1,
        maximum_failed_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    storage = tmp_path / "current failure storage"
    commit = "9" * 40
    run_id = generation.campaign.campaign_run_id(campaign, git_commit=commit)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", commit)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
    generation.cases.input_generation.generate_input_cases(
        batch,
        len(batch.case_indices),
        storage_root=storage,
    )
    generation.runtime.record_case_failure(
        batch,
        task.case_index,
        RuntimeError("current synthetic failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node-current",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )
    submitted: list[list[str]] = []

    def submit(command: list[str], **_kwargs: Any) -> str:
        submitted.append(command)
        return "654"

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: _scheduler(),
    )
    monkeypatch.setattr(generation.campaign, "_submit_case", submit)

    unchanged = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    assert unchanged["state"] == "completed_with_failures"
    assert submitted == []


@pytest.mark.parametrize(
    ("case_state", "replay_available", "expected_action", "expected_state"),
    [
        ("successful", False, "reuse", "complete"),
        ("never_started", False, "submit_initial", "active"),
        ("cancelled", False, "submit_resume", "active"),
        ("interrupted", False, "submit_resume", "active"),
        ("conversion_failed", True, "replay", "complete"),
        ("publication_failed", True, "replay", "complete"),
        ("exports_failed", False, "stop", "completed_with_failures"),
        ("failed", False, "stop", "completed_with_failures"),
        ("timed_out", False, "stop", "completed_with_failures"),
    ],
)
def test_campaign_resume_matrix_never_silently_retries_solver_failures(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_state: str,
    replay_available: bool,
    expected_action: str,
    expected_state: str,
) -> None:
    """Protect reuse, restart, replay, and explicit-retry boundaries."""
    config_path, _template = generation_config_factory(natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    target = tasks[0]
    run_id = "resume-matrix__0123456789abcdef"
    commit = "c" * 40
    storage = tmp_path / case_state
    campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    ).mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_run_id": run_id,
        "git_commit": commit,
        "slurm_job_ids": [],
        "submissions": [],
        "submission_intent": None,
        "submission_config": {
            "max_admission_cases": 1,
            "max_running_cases": None,
            "maximum_failed_cases": 0,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "state": "active",
    }
    views = [
        {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
            "state": (case_state if task == target else "successful"),
            "postprocessing_replay_available": (replay_available if task == target else False),
            "attempt_campaign_run_id": (run_id if task == target and replay_available else None),
        }
        for task in tasks
    ]
    submissions: list[tuple[str, str]] = []
    replays: list[tuple[str, int]] = []
    finalized: list[bool] = []

    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        campaign_evidence,
        "campaign_from_manifest",
        lambda _manifest: campaign,
    )
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: _scheduler(),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_reconciled",
        lambda *_args, **_kwargs: (views, 0, 0),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_campaign_input_references",
        lambda *_args, **_kwargs: _synthetic_input_references(campaign),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_repository_commit",
        lambda: commit,
    )
    monkeypatch.setattr(
        generation.campaign,
        "_write_campaign_manifest",
        lambda payload, **_kwargs: dict(payload),
    )

    def submit(
        payload: dict[str, Any],
        _campaign: Any,
        task: Any,
        *,
        mode: str,
        storage_root: Path | str | None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del storage_root
        submissions.append((task.case_id, mode))
        payload["state"] = "active"
        return dict(payload)

    monkeypatch.setattr(generation.campaign, "_submit_one", submit)

    def replay(batch: Any, case_index: int, **_kwargs: Any) -> Any:
        replays.append((batch.batch_name, case_index))
        return SimpleNamespace(status="replayed")

    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "replay_case_postprocessing",
        replay,
    )

    def case_is_valid(_batch: Any, case_index: int, **_kwargs: Any) -> bool:
        if case_index != target.case_index:
            return True
        return case_state == "successful" or (expected_action == "replay" and bool(replays))

    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        case_is_valid,
    )
    monkeypatch.setattr(
        generation.campaign,
        "_successful_status_summary",
        lambda *_args, **_kwargs: generation.campaign._empty_successful_status_summary(),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_finalize_completed_batches",
        lambda *_args, **_kwargs: finalized.append(True),
    )

    resumed = generation.campaign.resume_campaign(
        run_id,
        storage_root=storage,
    )

    assert resumed["state"] == expected_state
    if expected_action == "submit_initial":
        assert submissions == [(target.case_id, "initial")]
    elif expected_action == "submit_resume":
        assert submissions == [(target.case_id, "resume")]
    else:
        assert submissions == []
    if expected_action == "replay":
        assert replays == [(target.batch_name, target.case_index)]
    else:
        assert replays == []
    assert bool(finalized) is (expected_action in {"reuse", "replay"})


def test_malformed_persisted_job_id_fails_before_scheduler_query(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject non-numeric durable job identity without querying Slurm."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "7" * 40
    storage = tmp_path / "storage"
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_submit_case", lambda *_args, **_kwargs: "123")
    manifest = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    path = campaign_evidence.campaign_run_manifest_path(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["slurm_job_ids"] = ["123 --all"]
    payload["submissions"][0]["job_id"] = "123 --all"
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_output",
        lambda _command: pytest.fail("malformed persisted identity reached scheduler query"),
    )

    with pytest.raises(ValueError, match="submission state is malformed"):
        generation.campaign.feed_campaign(
            manifest["campaign_run_id"],
            storage_root=storage,
        )


def test_feeder_restores_one_pending_job_without_limiting_running_jobs(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit one at a time, keep max_admission_cases=1, and allow running growth."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=6,
        max_admission_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "a" * 40
    submitted = iter(("101", "102", "103", "104"))
    scheduler_state: dict[str, list[str]] = {}

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(
        generation.campaign,
        "_submit_case",
        lambda *_args, **_kwargs: next(submitted),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: _scheduler(active=scheduler_state),
    )
    monkeypatch.setattr(
        generation.campaign.progress_service,
        "load_runtime_progress",
        lambda _run_id, job_id, _identity, **_kwargs: (
            {
                "availability": "available",
                "phase": "transient_drying",
                "parser_state": "available",
            }
            if scheduler_state.get(job_id, [None, None])[1] == "RUNNING"
            else {"availability": "unavailable", "reason": "not_reported"}
        ),
    )
    storage = tmp_path / "storage"
    manifest = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    assert manifest["slurm_job_ids"] == ["101"]
    assert manifest["submission_config"]["max_admission_cases"] == 1
    first_case = manifest["submissions"][0]["case"]
    assert first_case["case_id"] == cluster.campaign_tasks(campaign)[0].case_id

    scheduler_state["101"] = ["101", "PENDING", "Resources", ""]
    unchanged = generation.campaign.feed_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert unchanged["slurm_job_ids"] == ["101"]

    scheduler_state["101"] = ["101", "RUNNING", "node-a", "node-a"]
    second = generation.campaign.feed_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert second["slurm_job_ids"] == ["101", "102"]
    assert second["submissions"][1]["case"] != first_case

    scheduler_state["102"] = ["102", "PENDING", "Resources", ""]
    still_two = generation.campaign.feed_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert still_two["slurm_job_ids"] == ["101", "102"]

    scheduler_state["102"] = ["102", "RUNNING", "node-b", "node-b"]
    third = generation.campaign.feed_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert third["slurm_job_ids"] == ["101", "102", "103"]

    scheduler_state["103"] = ["103", "PENDING", "Resources", ""]
    saturated = generation.campaign.feed_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert saturated["slurm_job_ids"] == ["101", "102", "103"]

    scheduler_state["103"] = ["103", "RUNNING", "node-c", "node-c"]
    scheduler_state["999"] = ["999", "PENDING", "Resources", ""]
    unrelated_pending_is_ignored = generation.campaign.feed_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert unrelated_pending_is_ignored["slurm_job_ids"] == ["101", "102", "103", "104"]


def test_feeder_refills_max_admission_cases_after_one_pending_case(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit the next unsent case when one scheduler-pending case leaves buffer capacity."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=3,
        max_admission_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "e" * 40
    submitted = iter(("301", "302"))
    scheduler_state: dict[str, list[str]] = {}
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_submit_case", lambda *_args, **_kwargs: next(submitted))
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: _scheduler(active=scheduler_state))
    storage = tmp_path / "storage"
    initial = generation.campaign.submit_campaign(campaign, git_commit=commit, storage_root=storage)
    manifest_path = campaign_evidence.campaign_run_manifest_path(initial["campaign_run_id"], storage_root=storage)
    persisted = campaign_evidence.load_campaign_run(initial["campaign_run_id"], storage_root=storage)
    persisted["submission_config"]["max_admission_cases"] = 2
    generation.campaign.common.serialization.atomic_write_json(manifest_path, persisted)

    scheduler_state["301"] = ["301", "PENDING", "Resources", ""]
    advanced = generation.campaign.feed_campaign(initial["campaign_run_id"], storage_root=storage)

    assert advanced["slurm_job_ids"] == ["301", "302"]
    assert advanced["submissions"][1]["case"]["case_id"] == cluster.campaign_tasks(campaign)[1].case_id


def test_feeder_pending_jobs_do_not_consume_running_capacity(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill four pending slots independently of a two-running-case cap."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=5,
        max_admission_cases=1,
        max_running_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "f" * 40
    submitted = iter(("401", "402", "403", "404"))
    scheduler_state: dict[str, list[str]] = {}
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_submit_case", lambda *_args, **_kwargs: next(submitted))
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: _scheduler(active=scheduler_state))
    storage = tmp_path / "storage"
    initial = generation.campaign.submit_campaign(campaign, git_commit=commit, storage_root=storage)
    manifest_path = campaign_evidence.campaign_run_manifest_path(initial["campaign_run_id"], storage_root=storage)
    persisted = campaign_evidence.load_campaign_run(initial["campaign_run_id"], storage_root=storage)
    persisted["submission_config"]["max_admission_cases"] = 4
    generation.campaign.common.serialization.atomic_write_json(manifest_path, persisted)

    scheduler_state["401"] = ["401", "PENDING", "Resources", ""]
    filled = generation.campaign.feed_campaign(initial["campaign_run_id"], storage_root=storage)

    assert filled["slurm_job_ids"] == ["401", "402", "403", "404"]
    assert filled["state"] == "active"


def test_feeder_skips_valid_success_before_submitting_next_unsent_case(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep immutable success authoritative when a prior job leaves squeue."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=4,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    commit = "f" * 40
    job_ids = iter(("401", "402"))
    scheduler = _scheduler()
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_submit_case", lambda *_args, **_kwargs: next(job_ids))
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: scheduler)
    completed: set[tuple[str, int]] = set()
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda batch, case_index, **_kwargs: (batch.batch_name, case_index) in completed,
    )
    monkeypatch.setattr(
        generation.campaign,
        "_successful_status_summary",
        lambda *_args, **_kwargs: generation.campaign._empty_successful_status_summary(),
    )
    storage = tmp_path / "storage"
    manifest = generation.campaign.submit_campaign(campaign, git_commit=commit, storage_root=storage)
    completed.add((tasks[0].batch_name, tasks[0].case_index))
    scheduler["accounted"] = {"401": ["401", "COMPLETED", "0:0"]}
    advanced = generation.campaign.feed_campaign(manifest["campaign_run_id"], storage_root=storage)
    assert advanced["slurm_job_ids"] == ["401", "402"]
    assert advanced["submissions"][-1]["case"]["case_id"] == tasks[1].case_id


def test_optional_running_cap_blocks_only_while_capacity_is_occupied(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hold new work at the explicit cap and advance after success frees capacity."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=2,
        maximum_failed_cases=2,
        max_running_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    commit = "b" * 40
    job_ids = iter(("201", "202"))
    scheduler = _scheduler()
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_submit_case", lambda *_args, **_kwargs: next(job_ids))
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: scheduler)
    storage = tmp_path / "storage"
    manifest = generation.campaign.submit_campaign(campaign, git_commit=commit, storage_root=storage)

    scheduler["active"] = {"201": ["201", "RUNNING", "node-a", "node-a"]}
    capped = generation.campaign.feed_campaign(manifest["campaign_run_id"], storage_root=storage)
    assert capped["slurm_job_ids"] == ["201"]

    scheduler["active"] = {}
    scheduler["accounted"] = {"201": ["201", "COMPLETED", "0:0"]}
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda batch, case_index, **_kwargs: (batch.batch_name, case_index) == (tasks[0].batch_name, tasks[0].case_index),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_successful_status_summary",
        lambda *_args, **_kwargs: generation.campaign._empty_successful_status_summary(),
    )
    advanced = generation.campaign.feed_campaign(manifest["campaign_run_id"], storage_root=storage)
    assert advanced["slurm_job_ids"] == ["201", "202"]
    assert advanced["submissions"][-1]["mode"] == "initial"
    assert advanced["submissions"][-1]["case"]["case_id"] == tasks[1].case_id


@pytest.mark.parametrize("maximum_failed_cases", [0, 2, 5])
def test_failure_circuit_breaker_uses_configured_n_plus_one_and_monitors_active_jobs(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    maximum_failed_cases: int,
) -> None:
    """Feed through N failures, stop at N+1, and retain active monitoring."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=maximum_failed_cases + 3,
        maximum_failed_cases=maximum_failed_cases,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    commit = "d" * 40
    run_id = f"failure-threshold-{maximum_failed_cases}__0123456789abcdef"
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=tmp_path)
    run_directory.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_run_id": run_id,
        "git_commit": commit,
        "slurm_job_ids": [],
        "submission_config": {
            "max_admission_cases": 1,
            "max_running_cases": None,
            "maximum_failed_cases": maximum_failed_cases,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "submissions": [],
        "state": "active",
    }

    def view(task: Any, state: str) -> dict[str, Any]:
        return {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
            "state": state,
            "failure_stage": "solver" if state in {"failed", "timed_out"} else None,
            "temporary_license_retry": None,
            "postprocessing_replay_available": False,
        }

    reconciled: dict[str, tuple[list[dict[str, Any]], int, int]] = {"value": ([], 0, 0)}
    submitted: list[tuple[str, str]] = []
    monkeypatch.setattr(campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(campaign_evidence, "campaign_from_manifest", lambda _manifest: campaign)
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: _scheduler())
    monkeypatch.setattr(generation.campaign, "_reconciled", lambda *_args, **_kwargs: reconciled["value"])
    monkeypatch.setattr(
        generation.campaign.common.serialization,
        "atomic_write_json",
        lambda *_args, **_kwargs: None,
    )

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: Any,
        *,
        mode: str,
        storage_root: Path | str | None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del storage_root
        submitted.append((task.case_id, mode))
        return {**payload, "state": "active"}

    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)

    reconciled["value"] = (
        [view(task, "failed" if index < maximum_failed_cases else "never_started") for index, task in enumerate(tasks)],
        0,
        0,
    )
    feedable = generation.campaign.feed_campaign(run_id, storage_root=tmp_path)
    assert feedable["state"] == "active"
    assert submitted == [(tasks[maximum_failed_cases].case_id, "initial")]

    submitted.clear()
    manifest["state"] = "active"
    reconciled["value"] = (
        [view(task, "failed" if index <= maximum_failed_cases else "never_started") for index, task in enumerate(tasks)],
        0,
        0,
    )
    tripped = generation.campaign.feed_campaign(run_id, storage_root=tmp_path)
    assert tripped["state"] == "failure_threshold_reached"
    assert submitted == []

    manifest["state"] = "active"
    reconciled["value"] = (
        [
            view(
                task,
                "failed" if index <= maximum_failed_cases else "running" if index == maximum_failed_cases + 1 else "never_started",
            )
            for index, task in enumerate(tasks)
        ],
        0,
        1,
    )
    monitored = generation.campaign.feed_campaign(run_id, storage_root=tmp_path)
    assert monitored["state"] == "active"
    assert submitted == []


@pytest.mark.parametrize(
    ("failure_count", "maximum_failed_cases", "expected_tripped"),
    [
        (0, 5, False),
        (1, 5, False),
        (3, 5, False),
        (5, 5, False),
        (6, 5, True),
    ],
)
def test_solver_failure_threshold_has_an_exact_exceeds_boundary(
    failure_count: int,
    maximum_failed_cases: int,
    expected_tripped: bool,
) -> None:
    """Count only solver failures and trip strictly above the configured limit."""
    views: list[dict[str, Any]] = [
        {
            "state": "failed",
            "failure_stage": "solver",
            "temporary_license_retry": None,
        }
        for _index in range(failure_count)
    ]
    views.extend(
        (
            {"state": "conversion_failed"},
            {"state": "publication_failed"},
            {
                "state": "failed",
                "failure_stage": "solver",
                "temporary_license_retry": {"classification": "temporary_license_capacity"},
            },
            {"state": "case_reconciliation_failed", "failure_stage": "reconciliation"},
        )
    )

    assert (
        generation.campaign._solver_failure_threshold_exceeded(
            views,
            maximum_failed_cases=maximum_failed_cases,
        )
        is expected_tripped
    )


def test_replay_failures_do_not_precede_or_starve_normal_admission(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill pending capacity before independently attempting every eligible replay."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=12,
        max_admission_cases=2,
        maximum_failed_cases=5,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    run_id = "replay-admission__0123456789abcdef"
    commit = "e" * 40
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=tmp_path)
    run_directory.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_run_id": run_id,
        "git_commit": commit,
        "slurm_job_ids": [],
        "submission_intent": None,
        "submission_config": {
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 5,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submissions": [],
        "state": "active",
    }
    views = [
        {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
            "state": "conversion_failed" if index < 2 else "never_started",
            "failure_stage": "conversion" if index < 2 else None,
            "temporary_license_retry": None,
            "postprocessing_replay_available": index < 2,
            "replay_eligible": index < 2,
            "replay_blocked": False,
            "attempt_campaign_run_id": run_id if index < 2 else None,
        }
        for index, task in enumerate(tasks)
    ]
    events: list[str] = []
    submitted: list[str] = []
    replayed: list[str] = []
    monkeypatch.setattr(campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(campaign_evidence, "campaign_from_manifest", lambda _manifest: campaign)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: _scheduler())
    monkeypatch.setattr(generation.campaign, "_reconciled", lambda *_args, **_kwargs: (views, 0, 0))
    monkeypatch.setattr(
        generation.campaign,
        "_campaign_input_references",
        lambda *_args, **_kwargs: _synthetic_input_references(campaign),
    )
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_write_campaign_manifest", lambda payload, **_kwargs: dict(payload))
    monkeypatch.setattr(
        generation.campaign,
        "_refresh_resume_task_views",
        lambda *_args, **_kwargs: [dict(view) for view in views],
    )
    monkeypatch.setattr(generation.campaign.batch_runtime, "completed_case_is_valid", lambda *_args, **_kwargs: False)

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: Any,
        *,
        mode: str,
        storage_root: Path | str | None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del storage_root
        assert mode == "initial"
        events.append("submit")
        submitted.append(task.case_id)
        payload["submissions"].append(
            {
                "case": {
                    "batch_name": task.batch_name,
                    "batch_id": task.batch_id,
                    "case_index": task.case_index,
                    "case_id": task.case_id,
                }
            }
        )
        return payload

    def fail_replay(batch: Any, case_index: int, **_kwargs: Any) -> None:
        events.append("replay")
        replayed.append(batch.case_id(case_index))
        message = "deterministic replay failure"
        raise generation.runtime.batch.CaseLocalReplayError(message)

    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)
    monkeypatch.setattr(generation.campaign.batch_runtime, "replay_case_postprocessing", fail_replay)

    resumed = generation.campaign.resume_campaign(run_id, storage_root=tmp_path)

    assert resumed["state"] == "active"
    assert submitted == [tasks[2].case_id, tasks[3].case_id]
    assert replayed == [tasks[0].case_id, tasks[1].case_id]
    assert events == ["submit", "submit", "replay", "replay"]

    def fail_integrity(*_args: Any, **_kwargs: Any) -> None:
        message = "conflicting immutable publication hash"
        raise generation.runtime.batch.ReplayIntegrityError(message)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            generation.campaign,
            "_fill_submission_capacity",
            lambda payload, *_args, **_kwargs: payload,
        )
        scoped.setattr(
            generation.campaign.batch_runtime,
            "replay_case_postprocessing",
            fail_integrity,
        )
        with pytest.raises(
            generation.runtime.batch.ReplayIntegrityError,
            match="immutable publication hash",
        ):
            generation.campaign.resume_campaign(run_id, storage_root=tmp_path)


def _admission_view(
    task: Any,
    state: str,
    *,
    retry_eligible: bool = False,
    solver_started: bool = False,
) -> dict[str, Any]:
    """Return one synthetic logical admission view."""
    return {
        "batch_name": task.batch_name,
        "batch_id": task.batch_id,
        "case_index": task.case_index,
        "case_id": task.case_id,
        "state": state,
        "runtime_progress": (
            {
                "availability": "available",
                "phase": "transient_drying",
                "parser_state": "available",
            }
            if solver_started
            else {"availability": "unavailable", "reason": "not_reported"}
        ),
        "license_retry_active": False,
        "license_retry_eligible": retry_eligible,
        "license_first_blocked_at": (f"2026-01-01T00:00:{task.case_index:02d}+00:00" if state == "license_blocked" else None),
        "license_next_retry_at": (f"2026-01-01T00:01:{task.case_index:02d}+00:00" if state == "license_blocked" else None),
        "temporary_license_retry": (
            {
                "classification": "temporary_license_capacity",
                "next_retry_at": f"2026-01-01T00:01:{task.case_index:02d}+00:00",
                "retry_count": 1,
            }
            if state == "license_blocked"
            else None
        ),
        "failure_stage": "solver" if state == "license_blocked" else None,
    }


@pytest.mark.parametrize(
    ("state", "solver_started", "expected"),
    [
        ("never_started", False, False),
        ("pending", False, True),
        ("active", False, True),
        ("active", True, False),
        ("scheduler_unknown", False, True),
        ("license_blocked", False, True),
        ("interrupted", False, True),
        ("conversion_failed", False, False),
        ("publication_failed", False, False),
        ("successful", False, False),
        ("failed", False, False),
        ("timed_out", False, False),
    ],
)
def test_logical_admission_membership_is_lifecycle_owned(
    state: str,
    solver_started: bool,
    expected: bool,
) -> None:
    """Count pre-solver lifecycle states and release confirmed solvers."""
    view = {
        "state": state,
        "runtime_progress": (
            {
                "availability": "available",
                "phase": "transient_drying",
                "parser_state": "available",
            }
            if solver_started
            else {"availability": "unavailable"}
        ),
    }
    assert generation.campaign._view_consumes_admission(view) is expected


@pytest.mark.parametrize("failure_count", [1, 3])
def test_terminal_failures_below_budget_do_not_stop_unrelated_work(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_count: int,
) -> None:
    """Keep monitoring one active case and admit fresh cases below the budget."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=failure_count + 3,
        max_admission_cases=2,
        maximum_failed_cases=5,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    failed = [
        {
            **_admission_view(task, "failed"),
            "failure_stage": "solver",
            "temporary_license_retry": None,
        }
        for task in tasks[:failure_count]
    ]
    active = _admission_view(
        tasks[failure_count],
        "active",
        solver_started=True,
    )
    fresh = [_admission_view(task, "never_started") for task in tasks[failure_count + 1 :]]
    views = [*failed, active, *fresh]
    manifest = {
        "campaign_run_id": f"failure-continuation-{failure_count}__0123456789abcdef",
        "git_commit": "e" * 40,
        "slurm_job_ids": ["811"],
        "submission_config": {
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 5,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "admission_reservations": [],
        "submissions": [],
        "state": "active",
    }
    scheduler = _scheduler(active={"811": ["811", "RUNNING"]})
    submitted: list[str] = []

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: cluster.CampaignTask,
        *,
        mode: str,
        storage_root: Path | str | None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del storage_root
        assert mode == "initial"
        submitted.append(task.case_id)
        return payload

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: "e" * 40)
    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)
    monkeypatch.setattr(
        generation.campaign.common.serialization,
        "atomic_write_json",
        lambda *_args, **_kwargs: None,
    )

    advanced = generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=1,
        scheduler=scheduler,
        storage_root=tmp_path,
    )

    assert advanced["state"] == "active"
    assert submitted == [view["case_id"] for view in fresh]
    assert scheduler["active"] == {"811": ["811", "RUNNING"]}
    assert (
        generation.campaign._solver_failure_threshold_exceeded(
            views,
            maximum_failed_cases=5,
        )
        is False
    )


@pytest.mark.parametrize(
    (
        "failure_count",
        "capacity_occupied",
        "maximum_failed_cases",
        "expected_blocked",
        "expected_reason",
        "expected_circuit_breaker",
    ),
    [
        (0, True, 5, True, "max_admission_cases_reached", False),
        (1, False, 5, False, None, False),
        (1, True, 5, True, "max_admission_cases_reached", False),
        (2, False, 1, True, "solver_failure_threshold_exceeded", True),
    ],
)
def test_admission_block_reason_is_independent_of_displayed_failure_count(
    generation_config_factory: Any,
    failure_count: int,
    capacity_occupied: bool,
    maximum_failed_cases: int,
    expected_blocked: bool,
    expected_reason: str | None,
    expected_circuit_breaker: bool,
) -> None:
    """Attribute capacity and failure-budget blocks to their actual owner."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=4,
        max_admission_cases=1,
        maximum_failed_cases=maximum_failed_cases,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    views = [
        {
            **_admission_view(task, "failed"),
            "failure_stage": "solver",
            "temporary_license_retry": None,
        }
        for task in tasks[:failure_count]
    ]
    if capacity_occupied:
        views.append(_admission_view(tasks[failure_count], "pending"))
    fresh_index = failure_count + int(capacity_occupied)
    views.extend(_admission_view(task, "never_started") for task in tasks[fresh_index:])
    manifest = {
        "submission_config": {
            "max_admission_cases": 1,
            "max_running_cases": None,
            "maximum_failed_cases": maximum_failed_cases,
        },
        "submission_intent": None,
        "submissions": [],
        "admission_reservations": [],
    }

    blocked, reason = generation.campaign._normal_admission_status(
        manifest,
        views,
        running_jobs=0,
    )

    assert blocked is expected_blocked
    assert reason == expected_reason
    assert (
        generation.campaign._solver_failure_threshold_exceeded(
            views,
            maximum_failed_cases=maximum_failed_cases,
        )
        is expected_circuit_breaker
    )


def test_durable_submission_intent_counts_one_logical_case() -> None:
    """Count one unresolved intent without counting its job history."""
    view = {
        "batch_id": "batch",
        "case_index": 1,
        "state": "never_started",
        "runtime_progress": {"availability": "unavailable"},
    }
    manifest = {
        "submission_intent": 3,
        "submissions": [
            {"submission_index": 1, "mode": "initial", "case": {"batch_id": "batch", "case_index": 1}},
            {"submission_index": 2, "mode": "license_retry", "case": {"batch_id": "batch", "case_index": 1}},
            {"submission_index": 3, "mode": "license_retry", "case": {"batch_id": "batch", "case_index": 1}},
        ],
    }

    assert generation.campaign._logical_admission_case_keys(manifest, [view]) == frozenset({("batch", 1)})


def test_two_due_license_retries_submit_independently(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit both due logical slot owners without a campaign launch gate."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=3,
        max_admission_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    views = [
        _admission_view(tasks[0], "license_blocked", retry_eligible=True),
        _admission_view(tasks[1], "license_blocked", retry_eligible=True),
        _admission_view(tasks[2], "never_started"),
    ]
    due = "2026-01-01T00:00:00+00:00"
    for view in views[:2]:
        view["license_next_retry_at"] = due
        view["license_first_blocked_at"] = due
        view["temporary_license_retry"]["next_retry_at"] = due
    manifest: dict[str, Any] = {
        "campaign_run_id": "two-retries__0123456789abcdef",
        "git_commit": "f" * 40,
        "submission_config": {
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 5,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "submissions": [],
        "state": "license_blocked",
    }
    submitted: list[tuple[str, str]] = []
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: "f" * 40)
    monkeypatch.setattr(
        generation.campaign.common.serialization,
        "atomic_write_json",
        lambda *_args, **_kwargs: None,
    )

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: Any,
        *,
        mode: str,
        storage_root: Path | str | None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del storage_root
        submitted.append((task.case_id, mode))
        payload["submissions"].append(
            {
                "status": "submitted",
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "mode": mode,
                "case": {
                    "batch_id": task.batch_id,
                    "case_index": task.case_index,
                    "case_id": task.case_id,
                },
            }
        )
        return payload

    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)
    generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=0,
        scheduler={"active": {}},
        storage_root=tmp_path,
    )

    assert submitted == [
        (tasks[0].case_id, "license_retry"),
        (tasks[1].case_id, "license_retry"),
    ]
    assert generation.campaign._admission_summary(manifest, views)["count"] == 2
    assert views[2]["state"] == "never_started"


@pytest.mark.parametrize(
    "states",
    [("pending", "pending"), ("license_blocked", "license_blocked"), ("pending", "license_blocked")],
)
def test_shared_admission_limit_blocks_a_third_case(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    states: tuple[str, str],
) -> None:
    """Keep a never-started case out for every full two-case mixture."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=3,
        max_admission_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    views = [
        _admission_view(tasks[0], states[0]),
        _admission_view(tasks[1], states[1]),
        _admission_view(tasks[2], "never_started"),
    ]
    manifest: dict[str, Any] = {
        "campaign_run_id": "full-admission__0123456789abcdef",
        "git_commit": "e" * 40,
        "submission_config": {
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 5,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "submissions": [],
        "state": "active",
    }
    submitted: list[str] = []

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: cluster.CampaignTask,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        submitted.append(task.case_id)
        return payload

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: "e" * 40)
    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)
    monkeypatch.setattr(
        generation.campaign.common.serialization,
        "atomic_write_json",
        lambda *_args, **_kwargs: None,
    )

    generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=sum(state == "pending" for state in states),
        running_jobs=0,
        scheduler={"active": {}},
        storage_root=tmp_path,
    )

    assert submitted == []
    assert len(generation.campaign._logical_admission_case_keys(manifest, views)) == 2


def test_repeated_independent_retries_do_not_accumulate_blocked_cases(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry both oldest due owners while retaining exactly two logical slots."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=8,
        max_admission_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    views = [
        _admission_view(task, "license_blocked", retry_eligible=index < 2) if index < 2 else _admission_view(task, "never_started")
        for index, task in enumerate(tasks)
    ]
    common_due = "2026-01-01T00:00:00+00:00"
    for view in views[:2]:
        view["license_next_retry_at"] = common_due
        view["license_first_blocked_at"] = common_due
        view["temporary_license_retry"]["next_retry_at"] = common_due
    manifest: dict[str, Any] = {
        "campaign_run_id": "no-accumulation__0123456789abcdef",
        "git_commit": "d" * 40,
        "submission_config": {
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 5,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "submissions": [],
        "state": "license_blocked",
    }
    clock = {"value": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: "d" * 40)
    monkeypatch.setattr(
        generation.campaign.common.serialization,
        "atomic_write_json",
        lambda *_args, **_kwargs: None,
    )

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: Any,
        *,
        mode: str,
        storage_root: Path | str | None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del mode, storage_root
        payload["submissions"].append(
            {
                "submission_index": len(payload["submissions"]) + 1,
                "mode": "license_retry",
                "status": "submitted",
                "job_id": str(len(payload["submissions"]) + 1),
                "recorded_at": clock["value"].isoformat(),
                "case": {
                    "batch_id": task.batch_id,
                    "case_index": task.case_index,
                    "case_id": task.case_id,
                },
            }
        )
        view = views[tasks.index(task)]
        next_due = clock["value"] + timedelta(seconds=20)
        view["license_next_retry_at"] = next_due.isoformat()
        view["temporary_license_retry"]["next_retry_at"] = next_due.isoformat()
        view["temporary_license_retry"]["retry_count"] += 1
        return payload

    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)
    for _advance in range(5):
        generation.campaign._fill_submission_capacity(
            manifest,
            campaign,
            views,
            pending_jobs=0,
            running_jobs=0,
            scheduler={"active": {}},
            storage_root=tmp_path,
        )
        assert len(generation.campaign._logical_admission_case_keys(manifest, views)) == 2
        clock["value"] += timedelta(seconds=20)

    case_ids = [record["case"]["case_id"] for record in manifest["submissions"]]
    assert case_ids == [tasks[index % 2].case_id for index in range(10)]
    assert all(view["state"] == "never_started" for view in views[2:])
    assert len(generation.campaign._logical_admission_case_keys(manifest, views)) == 2


def test_restart_reconstructs_blocked_admission_without_admitting_more(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstruct two blocked slot owners from durable state after restart."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=3,
        max_admission_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    views = [
        _admission_view(tasks[0], "license_blocked"),
        _admission_view(tasks[1], "license_blocked"),
        _admission_view(tasks[2], "never_started"),
    ]
    manifest = json.loads(
        json.dumps(
            {
                "campaign_run_id": "restart-admission__0123456789abcdef",
                "git_commit": "b" * 40,
                "submission_config": {
                    "max_admission_cases": 2,
                    "max_running_cases": None,
                    "maximum_failed_cases": 5,
                    "temporary_license_retry": _synthetic_retry_policy(),
                },
                "submission_intent": None,
                "submissions": [
                    {
                        "submission_index": index,
                        "mode": "license_retry",
                        "status": "submitted",
                        "job_id": str(index),
                        "case": {
                            "batch_id": task.batch_id,
                            "case_index": task.case_index,
                            "case_id": task.case_id,
                        },
                    }
                    for index, task in enumerate(tasks[:2], start=1)
                ],
                "state": "license_blocked",
            }
        )
    )
    submitted: list[str] = []

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: cluster.CampaignTask,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        submitted.append(task.case_id)
        return payload

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: "b" * 40)
    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)
    monkeypatch.setattr(
        generation.campaign.common.serialization,
        "atomic_write_json",
        lambda *_args, **_kwargs: None,
    )

    assert generation.campaign._admission_summary(manifest, views)["count"] == 2
    generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=0,
        scheduler={"active": {}},
        storage_root=tmp_path,
    )

    assert submitted == []
    assert generation.campaign._admission_summary(manifest, views) == {
        "count": 2,
        "maximum": 2,
        "components": {
            "pending": 0,
            "starting": 0,
            "license_waiting": 2,
            "acquiring_license": 0,
        },
    }


def test_solver_start_releases_slots_for_progressive_discovery(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release admission at solver start and admit the next case immediately."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=5,
        max_admission_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    views = [
        _admission_view(tasks[0], "active", solver_started=True),
        _admission_view(tasks[1], "license_blocked"),
        _admission_view(tasks[2], "never_started"),
        _admission_view(tasks[3], "never_started"),
        _admission_view(tasks[4], "never_started"),
    ]
    manifest: dict[str, Any] = {
        "campaign_run_id": "progressive__0123456789abcdef",
        "git_commit": "c" * 40,
        "submission_config": {
            "max_admission_cases": 2,
            "max_running_cases": None,
            "maximum_failed_cases": 5,
            "temporary_license_retry": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "submissions": [],
        "admission_reservations": [],
        "state": "active",
    }
    submitted: list[str] = []
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: "c" * 40)
    monkeypatch.setattr(
        generation.campaign.common.serialization,
        "atomic_write_json",
        lambda *_args, **_kwargs: None,
    )

    def submit_one(
        payload: dict[str, Any],
        _campaign: Any,
        task: Any,
        *,
        mode: str,
        storage_root: Path | str | None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del storage_root
        submitted.append(task.case_id)
        payload["submissions"].append(
            {
                "submission_index": len(payload["submissions"]) + 1,
                "mode": mode,
                "status": "submitted",
                "job_id": str(len(payload["submissions"]) + 1),
                "recorded_at": "2026-01-01T00:00:00+00:00",
                "case": {
                    "batch_id": task.batch_id,
                    "case_index": task.case_index,
                    "case_id": task.case_id,
                },
            }
        )
        return payload

    monkeypatch.setattr(generation.campaign, "_submit_one", submit_one)

    generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=1,
        scheduler={"active": {}},
        storage_root=tmp_path,
    )
    assert submitted == [tasks[2].case_id]
    assert manifest["admission_reservations"] == []

    views[2] = _admission_view(tasks[2], "active", solver_started=True)
    generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=2,
        scheduler={"active": {}},
        storage_root=tmp_path,
    )
    assert submitted == [tasks[2].case_id, tasks[3].case_id]

    views[3] = _admission_view(tasks[3], "license_blocked")
    generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=2,
        scheduler={"active": {}},
        storage_root=tmp_path,
    )
    assert submitted == [tasks[2].case_id, tasks[3].case_id]
    assert views[4]["state"] == "never_started"
    assert generation.campaign._admission_summary(manifest, views)["components"] == {
        "pending": 0,
        "starting": 0,
        "license_waiting": 2,
        "acquiring_license": 0,
    }


def test_admission_reservations_are_exact_unsent_plan_members(
    generation_config_factory: Any,
) -> None:
    """Reject foreign or already-submitted operational reservations."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    reservation = {
        "batch_name": task.batch_name,
        "batch_id": task.batch_id,
        "case_index": task.case_index,
        "case_id": task.case_id,
    }
    manifest = {"admission_reservations": [reservation], "submissions": []}

    campaign_evidence._validate_admission_reservations_for_campaign(manifest, campaign)

    foreign = deepcopy(manifest)
    foreign["admission_reservations"][0]["case_id"] = "foreign_case"
    with pytest.raises(ValueError, match="resolved plan member"):
        campaign_evidence._validate_admission_reservations_for_campaign(foreign, campaign)

    submitted = deepcopy(manifest)
    submitted["submissions"] = [{"case": reservation}]
    with pytest.raises(ValueError, match="submission history"):
        campaign_evidence._validate_admission_reservations_for_campaign(submitted, campaign)


def test_license_retry_activity_ends_at_common_solver_progress() -> None:
    """Keep pending and starting retries in admission until solver work."""
    retry = {"mode": "license_retry"}
    unavailable = {"availability": "unavailable", "reason": "not_reported"}
    starting = {
        "availability": "available",
        "phase": "starting_solver",
        "parser_state": "unavailable",
    }
    solving = {
        "availability": "available",
        "phase": "transient_drying",
        "parser_state": "available",
    }

    assert generation.campaign._license_retry_is_active("pending", retry, unavailable)
    assert generation.campaign._license_retry_is_active("active", retry, starting)
    assert not generation.campaign._license_retry_is_active("active", retry, solving)

    retrying_view = {
        "batch_id": "batch",
        "case_index": 1,
        "state": "active",
        "runtime_progress": starting,
        "license_retry_active": True,
    }
    waiting_view = {
        "batch_id": "batch",
        "case_index": 2,
        "state": "license_blocked",
        "runtime_progress": unavailable,
        "license_retry_active": False,
    }
    manifest = {
        "submission_config": {"max_admission_cases": 2},
        "submission_intent": None,
    }
    assert generation.campaign._admission_summary(manifest, [retrying_view, waiting_view])["components"] == {
        "pending": 0,
        "starting": 0,
        "license_waiting": 1,
        "acquiring_license": 1,
    }


def test_graceful_then_force_cancel_share_campaign_owner_and_stay_nonterminal(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signal running workers gracefully first and force only on request."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "c" * 40
    scheduler = _scheduler()
    commands: list[list[str]] = []
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(
        generation.campaign,
        "_submit_case",
        lambda *_args, **_kwargs: "301",
    )
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: scheduler,
    )

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        persisted = campaign_evidence.load_campaign_run(
            manifest["campaign_run_id"],
            storage_root=storage,
        )
        expected_state = "force_cancel_requested" if "--signal=KILL" in command else "cancel_requested"
        assert persisted["state"] == expected_state
        assert (
            campaign_evidence.campaign_run_directory(
                manifest["campaign_run_id"],
                storage_root=storage,
            )
            / "cancellations.json"
        ).is_file()
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(generation.campaign.subprocess, "run", run)
    storage = tmp_path / "storage"
    manifest = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    scheduler["active"] = {
        "301": ["301", "RUNNING", "node-a", "node-a"],
    }

    graceful = generation.campaign.cancel_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert commands == [["scancel", "--signal=TERM", "--batch", "301"]]
    assert graceful["schema_version"] == 1
    assert graceful["attempts"][-1]["mode"] == "graceful"
    running = generation.campaign.campaign_status(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert running["campaign_state"] == "running"
    assert running["cancellation_requested"] is True

    forced = generation.campaign.cancel_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
        force=True,
    )
    assert commands[-1] == ["scancel", "--signal=KILL", "--full", "301"]
    assert [attempt["mode"] for attempt in forced["attempts"]] == [
        "graceful",
        "force",
    ]

    scheduler["active"] = {}
    scheduler["accounted"] = {"301": ["301", "CANCELLED", "0:15"]}
    cancelled = generation.campaign.campaign_status(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert cancelled["campaign_state"] == "cancelled"
    assert cancelled["counts"]["cancelled"] == 1


def test_interrupted_submission_intent_recovers_exact_job_without_duplicate(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a lost sbatch response from its unique persisted job name."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "c" * 40
    storage = tmp_path / "storage"
    run_id = generation.campaign.campaign_run_id(campaign, git_commit=commit)
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)

    def lost_response(*_args: Any, **_kwargs: Any) -> str:
        message = "synthetic lost sbatch response"
        raise OSError(message)

    monkeypatch.setattr(generation.campaign, "_submit_case", lost_response)
    with pytest.raises(OSError, match="lost sbatch response"):
        generation.campaign.submit_campaign(campaign, git_commit=commit, storage_root=storage)
    intent = campaign_evidence.load_campaign_run(run_id, storage_root=storage)
    assert intent["submission_intent"] == 1
    job_name = intent["submissions"][0]["job_name"]

    def scheduler_output(command: list[str]) -> tuple[str, str | None]:
        if any(argument == f"--name={job_name}" for argument in command):
            return "98765|PENDING|Resources", None
        if command[0] == "squeue":
            assert "--jobs=98765" in command
            return "98765|PENDING|Resources|", None
        if command[0] == "sacct":
            assert command[3:5] == ["--jobs", "98765"]
            return "", None
        raise AssertionError(command)

    monkeypatch.setattr(generation.campaign, "_scheduler_output", scheduler_output)
    recovered = generation.campaign.feed_campaign(run_id, storage_root=storage)
    assert recovered["submission_intent"] is None
    assert recovered["slurm_job_ids"] == ["98765"]
    assert len(recovered["submissions"]) == 1


def test_resume_recovers_next_case_intent_while_first_case_is_pending(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover a lost second response before admitting another logical case."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=3,
        max_admission_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    commit = "d" * 40
    storage = tmp_path / "storage"
    submitted = iter(("101",))
    real_scheduler_evidence = generation.campaign._scheduler_evidence
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_evidence",
        lambda _job_ids: _scheduler(),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_submit_case",
        lambda *_args, **_kwargs: next(submitted),
    )
    initial = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    manifest_path = campaign_evidence.campaign_run_manifest_path(
        initial["campaign_run_id"],
        storage_root=storage,
    )
    manifest = campaign_evidence.load_campaign_run(
        initial["campaign_run_id"],
        storage_root=storage,
    )
    manifest["submission_config"]["max_admission_cases"] = 2
    generation.campaign.common.serialization.atomic_write_json(manifest_path, manifest)

    def lost_response(*_args: Any, **_kwargs: Any) -> str:
        message = "synthetic lost second sbatch response"
        raise OSError(message)

    monkeypatch.setattr(generation.campaign, "_submit_case", lost_response)
    with pytest.raises(OSError, match="lost second sbatch response"):
        generation.campaign._submit_one(
            manifest,
            campaign,
            tasks[1],
            mode="initial",
            storage_root=storage,
        )
    interrupted = campaign_evidence.load_campaign_run(
        initial["campaign_run_id"],
        storage_root=storage,
    )
    assert interrupted["submission_intent"] is not None
    intent_job_name = interrupted["submissions"][-1]["job_name"]

    def scheduler_output(command: list[str]) -> tuple[str, str | None]:
        if any(argument == f"--name={intent_job_name}" for argument in command):
            return "202|PENDING|Resources", None
        if command[0] == "squeue":
            return "101|PENDING|Resources|\n202|PENDING|Resources|", None
        if command[0] == "sacct":
            return "", None
        raise AssertionError(command)

    monkeypatch.setattr(generation.campaign, "_scheduler_output", scheduler_output)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", real_scheduler_evidence)
    monkeypatch.setattr(
        generation.campaign,
        "_submit_case",
        lambda *_args, **_kwargs: pytest.fail("resume duplicated the recovered case"),
    )

    resumed = generation.campaign.resume_campaign(
        initial["campaign_run_id"],
        storage_root=storage,
    )

    assert resumed["submission_intent"] is None
    assert resumed["slurm_job_ids"] == ["101", "202"]
    assert len(resumed["submissions"]) == 2


def test_config_owned_plan_is_machine_parseable_and_read_only(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose configured resources as JSON without allocating campaign state."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "d" * 40
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)

    status = cli_generation.main(
        [
            "plan-campaign",
            str(config_path),
            "--git-commit",
            commit,
            "--storage-root",
            str(storage),
        ]
    )

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "planned"
    assert output["submission_config"]["cores_per_case"] == campaign.execution_values["cluster"]["cores_per_case"]
    assert output["submission_config"]["max_admission_cases"] == campaign.execution_values["submission"]["max_admission_cases"]
    assert not Path(output["paths"]["run_root"]).exists()


def test_submission_policy_changes_execution_but_not_scientific_case_identity(
    generation_config_factory: Any,
) -> None:
    """Keep feeder cadence out of case inputs while binding run provenance."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    original = generation.cases.config.load_campaign_config(config_path)
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["submission"]["max_admission_cases"] = 2
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    changed = generation.cases.config.load_campaign_config(config_path)

    assert [task.case_id for task in cluster.campaign_tasks(changed)] == [task.case_id for task in cluster.campaign_tasks(original)]
    assert [batch.scientific_config_digest for batch in changed.batches] == [batch.scientific_config_digest for batch in original.batches]
    assert [batch.case_input_config_digest for batch in changed.batches] == [batch.case_input_config_digest for batch in original.batches]
    commit = "9" * 40
    assert generation.campaign.campaign_run_id(changed, git_commit=commit) != generation.campaign.campaign_run_id(
        original,
        git_commit=commit,
    )


def test_campaign_discovery_uses_schema_kind_and_deterministic_paths(
    generation_config_factory: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Protect filename-independent discovery while ignoring unrelated YAML."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    (tmp_path / "unrelated.yaml").write_text("schema_kind: unrelated\n", encoding="utf-8")

    discovered = generation.cases.config.discover_campaign_configs(tmp_path)

    assert tuple(campaign.source_path for campaign in discovered) == (config_path.resolve(),)
    status = cli_generation.main(["list-campaigns"])
    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert [record["source_path"] for record in output["campaigns"]] == [str(config_path.resolve())]
    assert output["campaigns"][0]["campaign_purpose"] == discovered[0].campaign_purpose


def test_validate_config_exposes_resolved_campaign_ownership(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose resolved counts, seeds, packages, and purpose scope without mutation."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path, require_executable=False)
    monkeypatch.setattr(
        cli_generation.readiness_service,
        "campaign_unresolved_gates",
        lambda _path: {},
    )

    status = cli_generation.main(["validate-config", str(config_path), "--allow-incomplete"])

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["campaign_purpose"] == campaign.campaign_purpose
    assert output["case_counts"]["derived_total"] == campaign.total_case_count
    assert output["seed_plan"]["campaign_seed"] == campaign.batches[0].scientific_values["campaign_seed"]
    assert output["dataset_package_requests"] == [{"evaluation_regime": "id", "source_role": "seen"}]
    assert len(output["dataset_package_inventory"]) == len(campaign.dataset_packages)


def test_transfer_publication_keeps_validated_source_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect marked staging, destination hashes, source retention, and retries."""
    run_id = "synthetic_transfer__0123456789abcdef"
    destination = tmp_path / "destination storage"
    staging = workspace.create_transfer_staging(
        storage_root=destination,
        run_id=run_id,
    )
    assert staging.parent == (destination / ".incoming").resolve()
    campaign_directory = f"01_generation/meta/campaigns/{run_id}"
    relative_directories = (
        "01_generation/meta/batches/synthetic_batch",
        "01_generation/raw/synthetic_batch",
        "01_generation/processed/synthetic_batch",
        f"01_generation/attempts/synthetic_batch/case_00001/{run_id}",
        campaign_directory,
    )
    for index, relative in enumerate(relative_directories):
        directory = staging / relative
        directory.mkdir(parents=True)
        (directory / f"payload-{index}.txt").write_text(
            f"payload-{index}\n",
            encoding="utf-8",
        )
    terminal_path = staging / campaign_directory / "campaign_terminal.json"
    terminal_path.write_text('{"status":"terminal"}\n', encoding="utf-8")
    terminal = {
        "campaign_id": "synthetic_campaign_id",
        "git_commit": "d" * 40,
        "batches": [{}],
    }
    plan = {
        "campaign_directory": campaign_directory,
        "batches": [
            {
                "meta_directory": relative_directories[0],
                "raw_directory": relative_directories[1],
                "processed_directory": relative_directories[2],
                "attempt_directories": [relative_directories[3]],
            }
        ],
    }
    source_bytes = {
        relative: {path.relative_to(staging / relative).as_posix(): path.read_bytes() for path in (staging / relative).rglob("*") if path.is_file()}
        for relative in relative_directories
    }
    destination_checks = {"count": 0}

    def fake_terminal(
        _run_id: str,
        *,
        storage_root: Path | str | None = None,
    ) -> dict[str, Any]:
        assert storage_root is not None
        root = Path(storage_root).resolve()
        if root == destination.resolve():
            destination_checks["count"] += 1
            for relative, expected in source_bytes.items():
                published = destination / relative
                assert {
                    path.relative_to(published).as_posix(): path.read_bytes()
                    for path in published.rglob("*")
                    if path.is_file() and path.relative_to(published).as_posix() not in campaign_evidence.POST_TRANSFER_OPERATIONAL_PATHS
                } == expected
        return terminal

    monkeypatch.setattr(
        generation.campaign,
        "validate_terminal_campaign",
        fake_terminal,
    )
    monkeypatch.setattr(
        generation.campaign,
        "campaign_transfer_plan",
        lambda *_args, **_kwargs: plan,
    )
    receipt = generation.campaign.publish_transferred_campaign(
        run_id,
        staging_root=staging,
        destination_root=destination,
        source_host="cpu.example",
        source_storage_root="/remote/storage",
    )
    assert receipt["status"] == "transfer_complete"
    assert receipt["source_removed"] is False
    assert receipt["transferred_file_count"] == 6
    assert receipt["transferred_bytes"] == sum(len(payload) for directory in source_bytes.values() for payload in directory.values())
    assert len(receipt["files"]) == receipt["transferred_file_count"]
    assert destination_checks["count"] == 2
    assert all(not (staging / relative).exists() for relative in relative_directories)
    assert all((destination / relative).is_dir() for relative in relative_directories)

    repeated = generation.campaign.publish_transferred_campaign(
        run_id,
        staging_root=staging,
        destination_root=destination,
        source_host="cpu.example",
        source_storage_root="/remote/storage",
    )
    assert repeated == receipt
    assert destination_checks["count"] == 4

    receipt_path = destination / campaign_directory / "transfer_complete.json"
    immutable_receipt = receipt_path.read_bytes()
    (destination / campaign_directory / "dataset_packages_complete.lock").touch()
    (destination / campaign_directory / "dataset_packages_complete.json").write_text("{}\n", encoding="utf-8")
    (destination / campaign_directory / campaign_evidence.TECHNICAL_SMOKE_EVIDENCE_FILENAME).write_text(
        "{}\n",
        encoding="utf-8",
    )
    validated = campaign_evidence.validate_transfer_receipt(
        run_id,
        terminal=terminal,
        plan=plan,
        storage_root=destination,
    )
    current_inventory = campaign_evidence.transfer_inventory_from_plan(plan, storage_root=destination)
    assert validated == receipt
    assert receipt_path.read_bytes() == immutable_receipt
    assert current_inventory == {
        "file_count": receipt["transferred_file_count"],
        "size_bytes": receipt["transferred_bytes"],
        "files": receipt["files"],
        "inventory_sha256": receipt["transfer_inventory_sha256"],
    }

    authority = generation.campaign.campaign_transfer_authority(
        run_id,
        storage_root=destination,
    )
    receipt_path.write_text("{}\n", encoding="utf-8")
    repaired = generation.campaign.repair_transferred_campaign(
        run_id,
        source_host="cpu.example",
        source_storage_root="/remote/storage",
        authority=authority,
        storage_root=destination,
    )
    assert repaired["status"] == "transfer_complete"
    assert repaired["files"] == receipt["files"]
    assert repaired["directories"] != receipt["directories"]

    recovery_staging = workspace.create_transfer_staging(
        storage_root=destination,
        run_id=run_id,
    )
    for relative in relative_directories:
        shutil.copytree(destination / relative, recovery_staging / relative)
    missing_path = destination / relative_directories[0] / "payload-0.txt"
    missing_path.unlink()
    recovered = generation.campaign.publish_transferred_campaign(
        run_id,
        staging_root=recovery_staging,
        destination_root=destination,
        source_host="cpu.example",
        source_storage_root="/remote/storage",
    )
    assert recovered["status"] == "transfer_complete"
    assert missing_path.read_bytes() == b"payload-0\n"

    nested = destination / campaign_directory / "nested"
    nested.mkdir()
    nested_lock = nested / "dataset_packages_complete.lock"
    nested_lock.touch()
    with pytest.raises(ValueError, match="Transfer completion receipt or GPU publication is invalid"):
        campaign_evidence.validate_transfer_receipt(
            run_id,
            terminal=terminal,
            plan=plan,
            storage_root=destination,
        )
    nested_lock.unlink()
    nested.rmdir()

    transferred_path = destination / relative_directories[0] / "payload-0.txt"
    transferred_path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Transfer completion receipt or GPU publication is invalid"):
        campaign_evidence.validate_transfer_receipt(
            run_id,
            terminal=terminal,
            plan=plan,
            storage_root=destination,
        )

    source_bytes[relative_directories[0]]["payload-0.txt"] = b"mutated\n"
    invalid_receipt = receipt_path.read_bytes()
    with pytest.raises(RuntimeError, match="differs from the canonical CPU"):
        generation.campaign.repair_transferred_campaign(
            run_id,
            source_host="cpu.example",
            source_storage_root="/remote/storage",
            authority=authority,
            storage_root=destination,
        )
    assert receipt_path.read_bytes() == invalid_receipt

    unmarked = tmp_path / "unmarked staging"
    unmarked.mkdir()
    with pytest.raises(ValueError, match="marker"):
        generation.campaign.publish_transferred_campaign(
            run_id,
            staging_root=unmarked,
            destination_root=destination,
            source_host="cpu.example",
            source_storage_root="/remote/storage",
        )


def test_case_attempt_index_counts_only_real_slurm_submissions(
    generation_config_factory: Any,
) -> None:
    """Do not consume a case attempt number when sbatch never returns a job ID."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    case = generation.campaign._task_payload(task)
    manifest = {
        "submissions": [
            {"case": case, "status": "submission_failed", "job_id": None},
        ]
    }
    assert generation.campaign._next_case_attempt_index(manifest, task) == 1
    manifest["submissions"].extend(
        (
            {"case": case, "status": "submitted", "job_id": "4101"},
            {"case": case, "status": "submission_failed", "job_id": None},
        )
    )
    assert generation.campaign._next_case_attempt_index(manifest, task) == 2


def test_partial_transfer_publication_is_distinct_and_hash_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish partial evidence without creating complete campaign authority."""
    run_id = "synthetic_partial__0123456789abcdef"
    destination = tmp_path / "destination"
    staging = workspace.create_transfer_staging(
        storage_root=destination,
        run_id=run_id,
    )
    campaign_directory = f"01_generation/meta/campaigns/{run_id}"
    directories = (
        "01_generation/meta/batches/partial_batch",
        "01_generation/raw/partial_batch",
        "01_generation/processed/partial_batch",
        f"01_generation/attempts/partial_batch/case_00002/{run_id}",
    )
    for index, relative in enumerate((*directories, campaign_directory)):
        directory = staging / relative
        directory.mkdir(parents=True)
        (directory / f"payload-{index}.txt").write_text(
            f"payload-{index}\n",
            encoding="utf-8",
        )
    staged_case = staging / directories[2] / "case_0001" / "case.h5"
    staged_case.parent.mkdir()
    staged_case.write_bytes(b"canonical-case")
    successful = [
        {
            "batch_name": "partial_batch",
            "batch_id": "partial-batch-id",
            "case_id": "case_0001",
            "case_index": 1,
            "state": "successful",
            "classified_state": "successful",
        }
    ]
    failed = [
        {
            "batch_name": "partial_batch",
            "batch_id": "partial-batch-id",
            "case_id": "case_0002",
            "case_index": 2,
            "state": "failed",
            "classified_state": "failed",
        }
    ]
    plan = {
        "campaign_run_id": run_id,
        "campaign_name": "partial",
        "git_commit": "d" * 40,
        "campaign_config": "partial.yaml",
        "campaign_directory": campaign_directory,
        "batches": [
            {
                "batch_name": "partial_batch",
                "batch_id": "partial-batch-id",
                "case_count": 2,
                "meta_directory": directories[0],
                "raw_directory": directories[1],
                "processed_directory": directories[2],
                "attempt_directories": [directories[3]],
            }
        ],
    }
    partial_path = staging / campaign_directory / "campaign_partial.json"
    partial_path.write_text(
        json.dumps(
            {
                "schema_kind": "generation_campaign_partial",
                "schema_version": 1,
                "campaign_run_id": run_id,
                "campaign_id": "campaign-id",
                "git_commit": "d" * 40,
                "campaign_state": "completed_with_failures",
                "successful_cases": successful,
                "failed_cases": failed,
                "resume_command": f"resume {run_id}",
                "recorded_at": "2026-08-20T00:00:00+00:00",
                "transfer_plan": plan,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generation.campaign,
        "partial_campaign_transfer_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {
            "campaign_run_id": run_id,
            "campaign_config": "partial.yaml",
            "git_commit": "d" * 40,
        },
    )
    partial_batch = SimpleNamespace(
        batch_name="partial_batch",
        batch_id="partial-batch-id",
        case_indices=(1, 2),
        case_id=lambda index: f"case_{index:04d}",
    )
    monkeypatch.setattr(
        campaign_evidence,
        "campaign_from_manifest",
        lambda _manifest: SimpleNamespace(
            campaign_id="campaign-id",
            campaign_name="partial",
            batches=(partial_batch,),
        ),
    )
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda *_args, **_kwargs: True,
    )

    receipt = generation.campaign.publish_transferred_campaign(
        run_id,
        staging_root=staging,
        destination_root=destination,
        source_host="cpu.example",
        source_storage_root="/remote/storage",
        partial=True,
    )

    assert receipt["schema_version"] == 1
    assert receipt["status"] == "partial"
    assert receipt["successful_cases"] == successful
    assert receipt["failed_cases"] == failed
    assert receipt["source_removed"] is False
    assert not (destination / campaign_directory / "transfer_complete.json").exists()
    assert (destination / campaign_directory / "transfer_partial.json").is_file()

    published_case = destination / directories[2] / "case_0001" / "case.h5"
    case_identity = (
        published_case.stat().st_ino,
        common.serialization.file_sha256(published_case),
    )
    repeated_staging = workspace.create_transfer_staging(
        storage_root=destination,
        run_id=run_id,
    )
    for relative in (*directories, campaign_directory):
        shutil.copytree(
            destination / relative,
            repeated_staging / relative,
        )
    repeated = generation.campaign.publish_transferred_campaign(
        run_id,
        staging_root=repeated_staging,
        destination_root=destination,
        source_host="cpu.example",
        source_storage_root="/remote/storage",
        partial=True,
    )
    assert all(record["status"] == "reused" for record in repeated["directories"])
    assert (
        published_case.stat().st_ino,
        common.serialization.file_sha256(published_case),
    ) == case_identity

    conflicting_staging = workspace.create_transfer_staging(
        storage_root=destination,
        run_id=run_id,
    )
    for relative in (*directories, campaign_directory):
        shutil.copytree(
            destination / relative,
            conflicting_staging / relative,
        )
    with pytest.raises(FileExistsError, match="partial transfer identity conflicts"):
        generation.campaign.publish_transferred_campaign(
            run_id,
            staging_root=conflicting_staging,
            destination_root=destination,
            source_host="other-cpu.example",
            source_storage_root="/remote/storage",
            partial=True,
        )

    published_partial = destination / campaign_directory / "campaign_partial.json"
    payload = json.loads(published_partial.read_text(encoding="utf-8"))
    payload["resume_command"] = "resume conflicting-run"
    published_partial.write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Partial transfer receipt"):
        generation.campaign.validate_partially_transferred_campaign(
            run_id,
            storage_root=destination,
        )
