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
    assert manifest["submission_history"][-1]["kind"] == "initial"
    assert manifest["submission_history"][-1]["job_id"] == "12345"
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
    assert status["campaign_state"] == "running"
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
    assert status["campaign_state"] == "running"
    assert status["slurm_job_ids"] == ["98765"]
    recovered = generation.campaign_runtime.load_campaign_run(run_id, storage_root=storage)
    assert recovered["state"] == "submitted"
    assert recovered["slurm_job_ids"] == ["98765"]


def test_cli_allows_only_execution_overrides(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Protect nonmutating planning and prohibit ad hoc scientific selectors."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.config.load_campaign_config(config_path)
    modified = campaign.with_wall_time("01:30:00")
    assert modified.campaign_digest == campaign.campaign_digest
    assert [batch.scientific_config_digest for batch in modified.batches] == [batch.scientific_config_digest for batch in campaign.batches]
    commit = "a" * 40
    storage = tmp_path / "storage"
    storage.mkdir()
    monkeypatch.setattr(generation.campaign_runtime, "_repository_commit", lambda: commit)
    status = cli_generation.main(
        [
            "plan-campaign",
            str(config_path),
            "--only-batch",
            "transient_drying__lentil__natural",
            "--wall-time",
            "01:30:00",
            "--git-commit",
            commit,
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
            "--storage-root",
            str(storage),
        ]
    )
    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "planned"
    assert output["filesystem_mutated"] is False
    assert "--only-batch transient_drying__lentil__natural" in output["submission_command"][-1]
    assert "--time=01:30:00" in output["submission_command"]
    assert not Path(output["paths"]["run_root"]).exists()
    with pytest.raises(SystemExit) as error:
        cli_generation.main(["validate-config", str(config_path), "--material", "lentil"])
    assert error.value.code == 2


def test_cancel_then_resume_submits_only_incomplete_validated_membership(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect active-attempt refusal, cancellation evidence, and exact resume size."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    campaign = generation.config.load_campaign_config(config_path)
    total_cases = sum(len(batch.case_indices) for batch in campaign.batches)
    plan = generation.cluster.build_resource_plan(
        max_nodes=2,
        cases_per_node=2,
        cores_per_case=4,
        max_parallel_cases=3,
        cores_per_node=32,
        remaining_cases=total_cases,
    )
    commit = "c" * 40
    observed: list[list[str]] = []
    active = {"value": True}
    submitted_ids = iter(("11111", "22222"))

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        if command[0] == "sbatch":
            job_id = next(submitted_ids)
            return subprocess.CompletedProcess(command, 0, stdout=f"{job_id};synthetic\n", stderr="")
        if command[0] == "squeue":
            output = "11111_0|RUNNING|node-a\n" if active["value"] else ""
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        if command[0] == "scancel":
            active["value"] = False
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(generation.campaign_runtime, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign_runtime.subprocess, "run", fake_run)
    storage = tmp_path / "storage"
    manifest = generation.campaign_runtime.submit_campaign(
        campaign,
        resource_plan=plan,
        git_commit=commit,
        storage_root=storage,
    )
    with pytest.raises(RuntimeError, match="previous campaign Slurm attempt is active"):
        generation.campaign_runtime.resume_campaign(
            manifest["campaign_run_id"],
            storage_root=storage,
        )
    receipt = generation.campaign_runtime.cancel_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert receipt["attempts"][-1]["command"] == ["scancel", "11111"]
    cancelled = generation.campaign_runtime.load_campaign_run(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert cancelled["state"] == "cancel_requested"

    completed_batch = campaign.batches[0]
    completed_index = completed_batch.case_indices[0]
    monkeypatch.setattr(
        generation.campaign_runtime.batch_runtime,
        "completed_case_is_valid",
        lambda batch, case_index, **_kwargs: batch.batch_id == completed_batch.batch_id and case_index == completed_index,
    )
    resumed = generation.campaign_runtime.resume_campaign(
        manifest["campaign_run_id"],
        storage_root=storage,
    )
    assert resumed["slurm_job_ids"] == ["11111", "22222"]
    assert [attempt["kind"] for attempt in resumed["submission_history"]] == [
        "initial",
        "resume",
    ]
    assert resumed["submission_history"][-1]["job_id"] == "22222"
    assert f"--remaining-cases {total_cases - 1}" in resumed["submission_command"][-1]
    assert [command[0] for command in observed] == [
        "sbatch",
        "squeue",
        "scancel",
        "squeue",
        "sbatch",
    ]


def test_transfer_publication_keeps_validated_source_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect marked staging, destination hashes, source retention, and retries."""
    run_id = "synthetic_transfer__0123456789abcdef"
    source_storage = tmp_path / "source storage"
    destination = tmp_path / "destination storage"
    staging = generation.workspace.create_transfer_staging(
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
        generation.campaign_runtime,
        "validate_terminal_campaign",
        fake_terminal,
    )
    monkeypatch.setattr(
        generation.campaign_runtime,
        "campaign_transfer_plan",
        lambda *_args, **_kwargs: plan,
    )
    receipt = generation.campaign_runtime.publish_transferred_campaign(
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

    repeated = generation.campaign_runtime.publish_transferred_campaign(
        run_id,
        staging_root=staging,
        destination_root=destination,
        source_host="cpu.example",
        source_storage_root="/remote/storage",
    )
    assert repeated == receipt
    assert destination_checks["count"] == 4
    unmarked = tmp_path / "unmarked staging"
    unmarked.mkdir()
    with pytest.raises(ValueError, match="marker"):
        generation.campaign_runtime.publish_transferred_campaign(
            run_id,
            staging_root=unmarked,
            destination_root=destination,
            source_host="cpu.example",
            source_storage_root="/remote/storage",
        )
