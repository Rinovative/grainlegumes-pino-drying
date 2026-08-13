# ruff: noqa: S101, PLR2004, SLF001
"""Dynamic per-case feeder, Slurm identity, and transfer contracts."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import generation
from src.generation.cli import cli_generation
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_workspace as workspace


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
                "--format=%i|%T|%R|%N",
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

    first = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    assert first["slurm_job_ids"] == ["12345"]
    assert [command[0] for command in calls] == ["sbatch"]
    persisted = campaign_evidence.load_campaign_run(
        first["campaign_run_id"],
        storage_root=storage,
    )
    assert persisted["submissions"][0]["job_id"] == "12345"

    active = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    assert active["slurm_job_ids"] == ["12345"]
    assert [command[0] for command in calls].count("sbatch") == 1

    scheduler_mode["value"] = "completed"
    first_case = tasks[0]
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "completed_case_is_valid",
        lambda batch, case_index, **_kwargs: (batch.batch_name, case_index) == (first_case.batch_name, first_case.case_index),
    )
    advanced = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    assert advanced["slurm_job_ids"] == ["12345", "12346"]
    assert advanced["submissions"][1]["case"]["case_id"] == tasks[1].case_id


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
    old_commit = "7" * 40
    monkeypatch.setenv("GENERATION_GIT_COMMIT", old_commit)
    monkeypatch.setenv(
        "GENERATION_CAMPAIGN_RUN_ID",
        "old-campaign__0123456789abcdef",
    )
    generation.runtime.record_case_failure(
        batch,
        task.case_index,
        RuntimeError("old synthetic failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node-old",
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


def test_current_failure_still_requires_explicit_retry(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the configured failure threshold for the exact current execution."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path)
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    storage = tmp_path / "current failure storage"
    commit = "9" * 40
    run_id = generation.campaign.campaign_run_id(campaign, git_commit=commit)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", commit)
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", run_id)
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

    stopped = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    assert stopped["state"] == "failure_threshold_reached"
    assert submitted == []

    resumed = generation.campaign.resume_campaign(run_id, storage_root=storage)
    assert resumed["slurm_job_ids"] == ["654"]
    assert len(submitted) == 1
    assert resumed["submissions"][0]["mode"] == "resume"


def test_scheduler_queries_support_multiple_ids_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach optional squeue selections and preserve genuine query failures."""
    commands: list[list[str]] = []

    def capture(command: list[str]) -> tuple[str, str | None]:
        commands.append(command)
        return "", None

    monkeypatch.setattr(generation.campaign, "_scheduler_output", capture)
    empty = generation.campaign._scheduler_evidence([])
    assert empty["squeue"]["command"] == []
    assert commands == []

    evidence = generation.campaign._scheduler_evidence(["123", "124"])
    assert commands == [
        ["squeue", "--noheader", "--jobs=123,124", "--format=%i|%T|%R|%N"],
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            "123,124",
            "--format=JobIDRaw,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocCPUS,Partition",
        ],
    ]
    generation.campaign._require_scheduler_evidence(evidence)

    for failed_owner in ("squeue", "sacct"):

        def fail_query(
            command: list[str],
            *,
            owner: str = failed_owner,
        ) -> tuple[str, str | None]:
            return ("", "synthetic scheduler failure") if command[0] == owner else ("", None)

        monkeypatch.setattr(generation.campaign, "_scheduler_output", fail_query)
        failed = generation.campaign._scheduler_evidence(["123"])
        with pytest.raises(RuntimeError, match=rf"{failed_owner} failed: synthetic scheduler failure"):
            generation.campaign._require_scheduler_evidence(failed)


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
    """Submit one at a time, keep pending_buffer=1, and allow running growth."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=6,
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
    storage = tmp_path / "storage"
    manifest = generation.campaign.submit_campaign(
        campaign,
        git_commit=commit,
        storage_root=storage,
    )
    assert manifest["slurm_job_ids"] == ["101"]
    assert manifest["submission_config"]["pending_buffer"] == 1
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


def test_feeder_does_not_submit_when_recovered_state_exceeds_pending_buffer(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not cancel or add work when two exact jobs are already pending."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        natural_count=5,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "e" * 40
    submitted: list[str] = []
    next_job_ids = iter(("301", "302", "303"))

    def submit_one(*_args: Any, **_kwargs: Any) -> str:
        job_id = next(next_job_ids)
        submitted.append(job_id)
        return job_id

    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign, "_submit_case", submit_one)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: _scheduler())
    storage = tmp_path / "storage"
    manifest = generation.campaign.submit_campaign(campaign, git_commit=commit, storage_root=storage)
    tasks = cluster.campaign_tasks(campaign)
    manifest = generation.campaign._submit_one(
        manifest,
        campaign,
        tasks[1],
        mode="initial",
        storage_root=storage,
    )
    assert submitted == ["301", "302"]

    scheduler = _scheduler(
        active={
            "301": ["301", "PENDING", "Resources", ""],
            "302": ["302", "PENDING", "Resources", ""],
        }
    )
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: scheduler)
    unchanged = generation.campaign.feed_campaign(manifest["campaign_run_id"], storage_root=storage)
    assert unchanged["slurm_job_ids"] == ["301", "302"]
    assert submitted == ["301", "302"]


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
    storage = tmp_path / "storage"
    manifest = generation.campaign.submit_campaign(campaign, git_commit=commit, storage_root=storage)
    completed.add((tasks[0].batch_name, tasks[0].case_index))
    scheduler["accounted"] = {"401": ["401", "COMPLETED", "0:0"]}
    advanced = generation.campaign.feed_campaign(manifest["campaign_run_id"], storage_root=storage)
    assert advanced["slurm_job_ids"] == ["401", "402"]
    assert advanced["submissions"][-1]["case"]["case_id"] == tasks[1].case_id


def test_optional_running_cap_and_failure_threshold_require_explicit_resume(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honor an optional running cap and retry failed cases only explicitly."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        max_running_cases=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
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
    scheduler["accounted"] = {"201": ["201", "FAILED", "1:0"]}
    monkeypatch.setattr(generation.campaign.batch_runtime, "completed_case_is_valid", lambda *_args, **_kwargs: False)
    first_task = cluster.campaign_tasks(campaign)[0]
    monkeypatch.setattr(
        generation.campaign.batch_runtime,
        "case_failure_is_recorded",
        lambda batch, case_index, **_kwargs: batch.batch_name == first_task.batch_name and case_index == first_task.case_index,
    )
    stopped = generation.campaign.feed_campaign(manifest["campaign_run_id"], storage_root=storage)
    assert stopped["state"] == "failure_threshold_reached"
    assert stopped["slurm_job_ids"] == ["201"]

    resumed = generation.campaign.resume_campaign(manifest["campaign_run_id"], storage_root=storage)
    assert resumed["slurm_job_ids"] == ["201", "202"]
    assert resumed["submissions"][-1]["mode"] == "resume"
    assert resumed["submissions"][-1]["case"]["case_id"] == first_task.case_id


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


def test_config_owned_plan_has_no_production_resource_overrides(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep production feeder and allocation choices exclusively in execution YAML."""
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
    assert output["submission_config"]["pending_buffer"] == campaign.execution_values["submission"]["pending_buffer"]
    assert not any(argument.startswith("--array") for argument in output["first_submission_command"])
    assert "--exclusive" not in output["first_submission_command"]
    assert not Path(output["paths"]["run_root"]).exists()
    with pytest.raises(SystemExit) as error:
        cli_generation.main(["plan-campaign", str(config_path), "--max-nodes", "2"])
    assert error.value.code == 2


def test_submission_policy_changes_execution_but_not_scientific_case_identity(
    generation_config_factory: Any,
) -> None:
    """Keep feeder cadence out of case inputs while binding run provenance."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    original = generation.cases.config.load_campaign_config(config_path)
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["submission"]["pending_buffer"] = 2
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
    assert "workflow" not in output


def test_workflow_catalog_rejects_duplicate_profile_purpose_matches(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require unambiguous semantic campaign selection for the no-argument workflow."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.cases.config.load_campaign_config(config_path, require_executable=False)
    stationary = profiles.resolve_profile(profiles.STEADY_FLOW_PROFILE)
    transient = profiles.resolve_profile(profiles.TRANSIENT_DRYING_PROFILE)

    def variant(name: str, purpose: str, profile: Any) -> Any:
        return replace(
            campaign,
            source_path=config_path.with_name(f"{name}.yaml"),
            campaign_purpose=purpose,
            profile=profile,
        )

    discovered = (
        variant("family-stationary", "family_generalization", stationary),
        variant("family-transient", "family_generalization", transient),
        variant("smoke-stationary", "technical_runtime_smoke", stationary),
        variant("smoke-transient-a", "technical_runtime_smoke", transient),
        variant("smoke-transient-b", "technical_runtime_smoke", transient),
    )
    monkeypatch.setattr(
        cli_generation.config_service,
        "discover_campaign_configs",
        lambda *_args, **_kwargs: discovered,
    )

    with pytest.raises(ValueError, match=r"exactly one 'technical_runtime_smoke' transient campaign; discovered 2"):
        cli_generation._campaign_catalog(require_workflow=True)


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
    assert output["material_inventory"] == ["lentil"]
    assert output["case_counts"]["derived_total"] == campaign.total_case_count
    assert sum(output["counts"].values()) == campaign.total_case_count
    assert output["seed_plan"]["campaign_seed"] == campaign.batches[0].scientific_values["campaign_seed"]
    assert output["seed_plan"]["paired_equivalence_seed"] == campaign.paired_equivalence_seed
    assert output["seed_plan"]["membership_seed"] is None
    assert output["dataset_package_requests"] == [{"evaluation_regime": "id", "source_role": "seen"}]
    assert len(output["dataset_package_inventory"]) == len(campaign.dataset_packages)
    assert output["parameter_ood"]["batches"] == {}
    assert output["technical_smoke_plan"]["learning_membership"] == "none"
    assert output["pilot_plan"] is None
    assert output["static_sentinel_workload"] is None


def test_transfer_publication_keeps_validated_source_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect marked staging, destination hashes, source retention, and retries."""
    run_id = "synthetic_transfer__0123456789abcdef"
    source_storage = tmp_path / "source storage"
    destination = tmp_path / "destination storage"
    staging = workspace.create_transfer_staging(
        storage_root=source_storage,
        run_id=run_id,
    )
    campaign_directory = f"01_generation/meta/campaigns/{run_id}"
    relative_directories = (
        "01_generation/meta/batches/synthetic_batch",
        "01_generation/raw/synthetic_batch",
        "01_generation/processed/synthetic_batch",
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
                assert {
                    path.relative_to(staging / relative).as_posix(): path.read_bytes() for path in (staging / relative).rglob("*") if path.is_file()
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
    assert receipt["transferred_file_count"] == 5
    assert receipt["transferred_bytes"] == sum(len(payload) for directory in source_bytes.values() for payload in directory.values())
    assert len(receipt["files"]) == receipt["transferred_file_count"]
    assert destination_checks["count"] == 2
    assert all((staging / relative).is_dir() for relative in relative_directories)
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
