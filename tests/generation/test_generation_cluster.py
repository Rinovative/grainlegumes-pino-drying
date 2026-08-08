# ruff: noqa: S101, PLR2004, SLF001
"""Campaign-wide CPU resource, Slurm, status, and execution-override contracts."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from src import generation
from src.generation.cli import cli_generation


def test_resource_equations_and_one_shared_slurm_pool(generation_config_factory: Any) -> None:
    """Protect hard caps and one campaign submission instead of per-batch pools."""
    config_path, _template = generation_config_factory(
        scheduler_kind="slurm",
        material_families=generation.materials.MATERIAL_FAMILIES,
    )
    campaign = generation.config.load_campaign_config(config_path)
    remaining = sum(len(batch.case_indices) for batch in campaign.batches)
    assert remaining == 17
    plan = generation.cluster.build_resource_plan(
        max_nodes=2,
        cases_per_node=2,
        cores_per_case=8,
        max_parallel_cases=3,
        cores_per_node=32,
        remaining_cases=remaining,
    )
    assert plan.effective_parallel_cases == 3
    assert plan.effective_nodes == 2
    command = generation.cluster.build_campaign_slurm_submission_command(campaign, plan=plan)
    assert command[0] == "sbatch"
    assert "--nodes=1" in command
    assert "--array=0-1%2" in command
    assert "--cpus-per-task=16" in command
    assert len([argument for argument in command if argument.startswith("--wrap=")]) == 1
    wrapped = command[-1]
    assert "run-campaign-worker" not in wrapped
    assert "generation_campaign_node.sh" in wrapped
    assert "--only-batch" not in wrapped
    assert len(generation.cluster.campaign_tasks(campaign)) == remaining

    with pytest.raises(ValueError, match="cores_per_node"):
        generation.cluster.build_resource_plan(
            max_nodes=2,
            cases_per_node=5,
            cores_per_case=8,
            max_parallel_cases=4,
            cores_per_node=32,
            remaining_cases=10,
        )
    with pytest.raises(ValueError, match=r"max_nodes \* cases_per_node"):
        generation.cluster.build_resource_plan(
            max_nodes=2,
            cases_per_node=2,
            cores_per_case=8,
            max_parallel_cases=5,
            cores_per_node=32,
            remaining_cases=10,
        )


def test_campaign_worker_enforces_one_cap_across_subbatches(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect the campaign-global slot pool when many subbatches share a worker."""
    config_path, _template = generation_config_factory()
    campaign = generation.config.load_campaign_config(config_path)
    plan = generation.cluster.build_resource_plan(
        max_nodes=1,
        cases_per_node=4,
        cores_per_case=1,
        max_parallel_cases=2,
        cores_per_node=32,
        remaining_cases=5,
    )
    tracker = {"active": 0, "maximum": 0, "calls": 0}
    lock = threading.Lock()

    def never_completed(*_args: Any, **_kwargs: Any) -> bool:
        return False

    def fake_run_case(*_args: Any, **_kwargs: Any) -> object:
        with lock:
            tracker["active"] += 1
            tracker["calls"] += 1
            tracker["maximum"] = max(tracker["maximum"], tracker["active"])
        time.sleep(0.03)
        with lock:
            tracker["active"] -= 1
        return object()

    monkeypatch.setattr(generation.cluster.runtime_service, "completed_case_is_valid", never_completed)
    monkeypatch.setattr(generation.cluster.runtime_service, "run_case", fake_run_case)
    result = generation.cluster.run_campaign_worker(
        campaign,
        plan=plan,
        worker_index=0,
        worker_count=1,
        scheduler_kind="slurm",
        storage_root=tmp_path / "storage",
        work_root=tmp_path / "work",
    )
    assert tracker == {"active": 0, "maximum": 2, "calls": 5}
    assert len(result.completed_tasks) == 5
    assert {task.batch_name for task in result.completed_tasks} == {
        "transient_drying__lentil__natural",
        "transient_drying__lentil__parameter_ood",
    }


def test_submit_manifest_and_fake_scheduler_status(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect exact-commit submission evidence and resumable scheduler queries."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.config.load_campaign_config(config_path)
    plan = generation.cluster.build_resource_plan(
        max_nodes=1,
        cases_per_node=2,
        cores_per_case=4,
        max_parallel_cases=2,
        cores_per_node=32,
        remaining_cases=5,
    )
    commit = "a" * 40
    run_id = generation.campaign_runtime.campaign_run_id(
        campaign,
        git_commit=commit,
        resource_plan=plan,
    )
    intent_path = generation.campaign_runtime._run_manifest_path(
        run_id,
        storage_root=tmp_path / "storage",
    )
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        if command[0] == "sbatch":
            assert kwargs["env"]["GENERATION_GIT_COMMIT"] == commit
            assert kwargs["env"]["GENERATION_CAMPAIGN_RUN_ID"] == run_id
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            assert intent["state"] == "submitting"
            assert intent["slurm_job_ids"] == []
            assert any(argument.startswith("--output=") for argument in command)
            assert any(argument.startswith("--error=") for argument in command)
            return subprocess.CompletedProcess(command, 0, stdout="12345;synthetic\n", stderr="")
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="12345_0|RUNNING|node-a\n", stderr="")
        if command[0] == "sacct":
            return subprocess.CompletedProcess(command, 0, stdout="12345|RUNNING|0:0\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(generation.campaign_runtime, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign_runtime.subprocess, "run", fake_run)
    manifest = generation.campaign_runtime.submit_campaign(
        campaign,
        resource_plan=plan,
        git_commit=commit,
        storage_root=tmp_path / "storage",
    )
    assert manifest["git_commit"] == commit
    assert manifest["campaign_config"].startswith("configs/generation/campaigns/")
    assert not Path(manifest["campaign_config"]).is_absolute()
    assert manifest["slurm_job_ids"] == ["12345"]
    assert manifest["state"] == "submitted"
    assert manifest["scheduler_job_name"].startswith("vp2-")
    assert Path(manifest["scheduler_log_directory"]).is_dir()
    assert len(manifest["submission_command"]) > 1
    reused = generation.campaign_runtime.submit_campaign(
        campaign,
        resource_plan=plan,
        git_commit=commit,
        storage_root=tmp_path / "storage",
    )
    assert reused == manifest
    loaded = generation.campaign_runtime.load_campaign_run(
        manifest["campaign_run_id"],
        storage_root=tmp_path / "storage",
    )
    assert loaded == manifest
    failed_batch = campaign.batches[0]
    failed_case_index = failed_batch.case_indices[0]
    generation.runtime.record_case_failure(
        failed_batch,
        failed_case_index,
        RuntimeError("synthetic solver failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node-a",
        work_directory=tmp_path / "failed-work",
        storage_root=tmp_path / "storage",
    )
    status = generation.campaign_runtime.campaign_status(
        manifest["campaign_run_id"],
        storage_root=tmp_path / "storage",
    )
    assert status["campaign_state"] == "active"
    assert status["squeue"]["output"].startswith("12345_0|RUNNING")
    assert status["sacct"]["output"].startswith("12345|RUNNING")
    failed_status = next(item for item in status["batches"] if item["batch_id"] == failed_batch.batch_id)
    assert failed_status["failed"] == 1
    assert failed_status["pending"] == failed_status["planned"] - 1
    assert [command[0] for command in observed] == ["sbatch", "squeue", "sacct"]


def test_interrupted_submission_receipt_is_recovered_by_status(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect durable intent and scheduler-name recovery after a lost response."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.config.load_campaign_config(config_path)
    plan = generation.cluster.build_resource_plan(
        max_nodes=1,
        cases_per_node=1,
        cores_per_case=4,
        max_parallel_cases=1,
        cores_per_node=32,
        remaining_cases=5,
    )
    commit = "b" * 40
    run_id = generation.campaign_runtime.campaign_run_id(
        campaign,
        git_commit=commit,
        resource_plan=plan,
    )
    storage = tmp_path / "storage"

    def lose_submission_response(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        message = "synthetic lost sbatch response"
        raise OSError(message)

    monkeypatch.setattr(generation.campaign_runtime, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign_runtime.subprocess, "run", lose_submission_response)
    with pytest.raises(OSError, match="synthetic lost sbatch response"):
        generation.campaign_runtime.submit_campaign(
            campaign,
            resource_plan=plan,
            git_commit=commit,
            storage_root=storage,
        )
    intent = generation.campaign_runtime.load_campaign_run(run_id, storage_root=storage)
    assert intent["state"] == "submitting"
    assert intent["slurm_job_ids"] == []

    def fake_scheduler(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert any(argument == f"--name={intent['scheduler_job_name']}" for argument in command)
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout="98765_0|RUNNING|node-b\n", stderr="")
        if command[0] == "sacct":
            return subprocess.CompletedProcess(command, 0, stdout="98765|RUNNING|0:0\n", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(generation.campaign_runtime.subprocess, "run", fake_scheduler)
    status = generation.campaign_runtime.campaign_status(run_id, storage_root=storage)
    assert status["campaign_state"] == "active"
    assert status["slurm_job_ids"] == ["98765"]
    recovered = generation.campaign_runtime.load_campaign_run(run_id, storage_root=storage)
    assert recovered["state"] == "submitted"
    assert recovered["slurm_job_ids"] == ["98765"]


def test_cli_allows_only_execution_overrides(
    generation_config_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Protect predeclared batch selection and prohibit ad hoc scientific selectors."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.config.load_campaign_config(config_path)
    modified = campaign.with_wall_time("01:30:00")
    assert modified.campaign_digest == campaign.campaign_digest
    assert [batch.scientific_config_digest for batch in modified.batches] == [batch.scientific_config_digest for batch in campaign.batches]
    status = cli_generation.main(
        [
            "print-campaign-submit",
            str(config_path),
            "--only-batch",
            "transient_drying__lentil__natural",
            "--wall-time",
            "01:30:00",
            "--max-nodes",
            "1",
            "--cases-per-node",
            "2",
            "--cores-per-case",
            "4",
            "--max-parallel-cases",
            "2",
            "--cores-per-node",
            "32",
        ]
    )
    assert status == 0
    output = capsys.readouterr().out
    assert "--only-batch transient_drying__lentil__natural" in output
    assert "--time=01:30:00" in output
    with pytest.raises(SystemExit) as error:
        cli_generation.main(["validate-config", str(config_path), "--material", "lentil"])
    assert error.value.code == 2
