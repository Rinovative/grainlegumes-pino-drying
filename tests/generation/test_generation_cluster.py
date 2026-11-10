"""Node-job feeder, dispatch identity, resume and admission, and shared dispatcher primitives.

Responsibilities:
  - Check node-job feeding, exact Slurm node-job and step ownership, interrupted-dispatch recovery, resume, and cancellation
  - Check the resume matrix, the settled admission cap, the failure budget, and the admission block reasons
  - Check the shared dispatcher primitives: node-job slot queue rows and job-step accounting rows

Design principles:
  - Persisted intents prevent duplicate submission; operational changes preserve science
  - Hand-built run records carry one leased node job so dispatch, drain, and ticket files stay real

This module does NOT:
  - Contact a live scheduler or silently retry solver failures
"""

from __future__ import annotations

import json
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from support.fake_slurm import (
    COMMIT,
    Crash,
    FakeSlurm,
    bind_worker,
    crash_at,
    dispatch_record,
    feed,
    load_manifest,
    make_campaign,
    node_directory,
    open_node_job,
    submit,
)
from support.generation_fixtures import make_generation_config_factory

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cli import cli_generation
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_license as license_service
from src.generation.runtime import generation_runtime_node as node


def _synthetic_retry_policy() -> dict[str, Any]:
    """Return one test-owned bounded licence-wait policy."""
    return {"enabled": True, "maximum_wait_seconds": 3600.0}


def _scheduler(*, allocations: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Return one successful synthetic scheduler view: the live node-job allocations and no step accounting."""
    return {
        "squeue": {"command": ["squeue"], "output": "", "error": None},
        "sacct": {"command": ["sacct"], "output": "", "error": None},
        "sacct_steps": {"command": ["sacct"], "output": "", "error": None},
        "allocations": {"active": {} if allocations is None else allocations, "accounted": {}},
        "steps": {},
        "active": {},
        "accounted": {},
        "not_started": [],
    }


def _record_dispatches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Replace the ticket publication by a recorder of ``(case id, mode)`` that leaves a dispatched record as the real dispatch does."""
    recorded: list[tuple[str, str]] = []

    def dispatch(manifest: dict[str, Any], campaign: Any, task: Any, node_job: dict[str, Any], *, mode: str, **_kwargs: Any) -> dict[str, Any]:
        recorded.append((task.case_id, mode))
        dispatch_record(manifest, campaign, task, node_job, mode=mode)
        return manifest

    monkeypatch.setattr(generation.campaign, "_dispatch_case", dispatch)
    return recorded


def _set_native_launch_evidence(monkeypatch: pytest.MonkeyPatch, commit: str) -> None:
    """Supply explicit test-owned source and runtime evidence for Slurm jobs."""
    monkeypatch.setattr(source_service, "validate_admitted_source_before_publication", lambda: None)
    monkeypatch.setenv("GENERATION_GIT_COMMIT", commit)
    monkeypatch.setenv("GENERATION_SOURCE_SHA256", "b" * 64)
    monkeypatch.setenv(
        "GENERATION_NATIVE_VENV",
        str(common.paths.get_runtime_root().resolve() / "venvs" / "native"),
    )


def _assert_latest_epoch_worker_binding(
    manifest: dict[str, Any],
    storage: Path,
    work_root: Path,
    campaign_name: str,
    source_commit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the maintained worker entrypoint to read the dispatch that a superseding idle epoch owns."""
    bind_worker(monkeypatch, manifest, index=2, step="1", attempt="2", cpus="2")
    monkeypatch.setattr(source_service, "required_git_commit", lambda: source_commit)
    worker_seen: list[tuple[str, int, str]] = []
    worker_outcome = SimpleNamespace(status="interrupted")

    def run_worker(worker_campaign: Any, task: Any, **_kwargs: Any) -> Any:
        worker_seen.append((worker_campaign.source_name, worker_campaign.execution_values["cluster"]["cores_per_case"], task.case_id))
        return worker_outcome

    monkeypatch.setattr(generation.campaign.cluster_service, "run_campaign_case", run_worker)
    submission = manifest["submissions"][1]["case"]
    assert (
        generation.campaign.run_campaign_case_job(
            manifest["campaign_run_id"],
            submission["batch_name"],
            submission["case_index"],
            storage_root=storage,
            work_root=work_root,
        )
        is worker_outcome
    )
    assert worker_seen == [(campaign_name, 2, "case_0001")]


def _assert_epoch_append_cannot_rebind_existing_dispatches(
    manifest: dict[str, Any], campaign: Any, storage: Path, campaign_name: str, original_commit: str, source_commit: str
) -> None:
    """Reject caller-owned boundaries before publishing any new epoch file."""
    epoch_dir = common.paths.resolve_generation_epochs_directory(
        common.paths.resolve_generation_execution_directory(
            common.paths.resolve_generation_campaign_directory("steady_flow", campaign_name, storage_root=storage)
        )
    )
    next_epoch = epoch_dir / "epoch_0004.json"
    binding = generation.campaign._requested_execution_binding(campaign, source_commit)
    for protected in ({"first_submission_index": 2}, {"campaign_run_id": manifest["campaign_run_id"]}):
        with pytest.raises(ValueError, match="cannot override generated identity or submission boundaries"):
            lifecycle.append_execution_epoch(manifest, {**binding, **protected})
        assert not next_epoch.exists()
        assert generation.campaign._execution_binding(manifest, submission_index=1)["git_commit"] == original_commit
        assert generation.campaign._execution_binding(manifest, submission_index=2)["git_commit"] == source_commit


def test_canonical_active_resume_rebinds_source_and_resources_without_changing_science(
    generation_config_factory: Any,
    world: FakeSlurm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drained source run binds later node jobs and case dispatches to a new operational epoch."""
    campaign = make_campaign(generation_config_factory, natural_count=2, max_admission_cases=1)
    canonical_path = campaign.source_path
    name = campaign.source_name
    original_commit, next_commit = COMMIT, "c" * 40
    storage = world.storage

    def snapshot_binding(commit: str, fingerprint: str) -> tuple[str, str]:
        snapshot_dir = storage / "source_snapshots" / fingerprint
        snapshot_dir.mkdir(parents=True)
        tracked_diff = snapshot_dir / "tracked.diff"
        tracked_diff.write_bytes(b"")
        snapshot = snapshot_dir / "snapshot.json"
        snapshot.write_text(
            json.dumps(
                {
                    "schema_kind": "generation_source_snapshot",
                    "git_commit": commit,
                    "source_fingerprint_sha256": fingerprint,
                    "tracked_diff_sha256": common.serialization.file_sha256(tracked_diff),
                    "untracked": [],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(snapshot), common.serialization.file_sha256(snapshot)

    origin_snapshot, origin_snapshot_sha = snapshot_binding(original_commit, "b" * 64)
    monkeypatch.setenv("GENERATION_SOURCE_SNAPSHOT", origin_snapshot)
    monkeypatch.setenv("GENERATION_SOURCE_SNAPSHOT_SHA256", origin_snapshot_sha)
    commit = {"value": original_commit}
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit["value"])
    submitted: list[str] = []
    submit_node_job = cluster.submit_node_job

    def recording_submit(command: list[str], *, git_commit: str, environment: dict[str, str]) -> str:
        assert environment["GENERATION_CAMPAIGN_RUN_ID"] == f"source~steady_flow~{name}"
        submitted.append(git_commit)
        return submit_node_job(command, git_commit=git_commit, environment=environment)

    monkeypatch.setattr(cluster, "submit_node_job", recording_submit)
    first = submit(campaign, world)
    assert submitted == [original_commit]
    origin = {
        "path": origin_snapshot,
        "sha256": origin_snapshot_sha,
        "source_fingerprint_sha256": "b" * 64,
    }
    assert first["source_snapshot"] == origin
    assert lifecycle.execution_epochs(first)[0]["source_snapshot"] == origin_snapshot
    world.add_step("7001", 0, first["submissions"][0]["step_name"], "CANCELLED", "0:15")
    world.set_state("7001", "COMPLETED")  # a new operational epoch is admitted only once every previous node job has ended
    execution_path = canonical_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["run"]["cores_per_case"] = 2
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    updated = generation.cases.config.load_campaign_config(canonical_path)
    assert updated.campaign_digest == campaign.campaign_digest
    commit["value"] = next_commit
    monkeypatch.setenv("GENERATION_GIT_COMMIT", next_commit)
    monkeypatch.setenv("GENERATION_SOURCE_SHA256", "d" * 64)
    next_snapshot, next_snapshot_sha = snapshot_binding(next_commit, "d" * 64)
    monkeypatch.setenv("GENERATION_SOURCE_SNAPSHOT", next_snapshot)
    monkeypatch.setenv("GENERATION_SOURCE_SNAPSHOT_SHA256", next_snapshot_sha)
    monkeypatch.delenv(source_service.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(RuntimeError, match="--continue-with-current-code"):
        generation.campaign.submit_campaign(updated, git_commit=next_commit, storage_root=storage, max_admission_cases=2)
    assert len(lifecycle.execution_epochs(first)) == 1
    # From here on the launcher was asked to continue the bound campaign with the current code.
    monkeypatch.setenv(source_service.SOURCE_TRANSITION_ENVIRONMENT_VARIABLE, source_service.EXPLICIT_SOURCE_TRANSITION)
    fill = generation.campaign._fill_submission_capacity

    def stop_after_epoch(*_args: Any, **_kwargs: Any) -> None:
        message = "test-owned stop after durable epoch admission"
        raise RuntimeError(message)

    monkeypatch.setattr(generation.campaign, "_fill_submission_capacity", stop_after_epoch)
    with pytest.raises(RuntimeError, match="test-owned stop after durable epoch admission"):
        generation.campaign.submit_campaign(updated, git_commit=next_commit, storage_root=storage, max_admission_cases=2)
    monkeypatch.setattr(generation.campaign, "_fill_submission_capacity", fill)
    assert [epoch["first_submission_index"] for epoch in lifecycle.execution_epochs(first)] == [1, 2]
    assert submitted == [original_commit]

    # The cancelled case restarts in a new node job that the newest epoch binds.
    final_commit = "e" * 40
    commit["value"] = final_commit
    monkeypatch.setenv("GENERATION_GIT_COMMIT", final_commit)
    monkeypatch.setenv("GENERATION_SOURCE_SHA256", "f" * 64)
    final_snapshot, final_snapshot_sha = snapshot_binding(final_commit, "f" * 64)
    monkeypatch.setenv("GENERATION_SOURCE_SNAPSHOT", final_snapshot)
    monkeypatch.setenv("GENERATION_SOURCE_SNAPSHOT_SHA256", final_snapshot_sha)
    resumed = generation.campaign.submit_campaign(updated, git_commit=final_commit, storage_root=storage, max_admission_cases=2)
    assert resumed["git_commit"] == original_commit
    assert resumed["source_snapshot"] == origin
    assert resumed["execution_config_digest"] == first["execution_config_digest"]
    assert submitted == [original_commit, final_commit]
    assert resumed["slurm_job_ids"] == ["7001", "7002"]
    retry = resumed["submissions"][1]
    assert (retry["mode"], retry["attempt_index"], retry["job_id"], retry["case"]) == ("resume", 2, "7002", first["submissions"][0]["case"])
    tickets = [json.loads(node.ticket_path(node_directory(campaign, world, job), 1).read_text(encoding="utf-8")) for job in resumed["node_jobs"]]
    assert [(ticket["cores"], ticket["argv"][-1]) for ticket in tickets] == [(1, "1"), (2, "2")]
    epochs = lifecycle.execution_epochs(resumed)
    assert [epoch["git_commit"] for epoch in epochs] == [original_commit, next_commit, final_commit]
    assert [(epoch["source_snapshot"], epoch["source_snapshot_sha256"]) for epoch in epochs] == [
        (origin_snapshot, origin_snapshot_sha),
        (next_snapshot, next_snapshot_sha),
        (final_snapshot, final_snapshot_sha),
    ]
    assert [epoch["source_sha256"] for epoch in epochs] == ["b" * 64, "d" * 64, "f" * 64]
    assert [epoch["first_submission_index"] for epoch in epochs] == [1, 2, 2]
    assert generation.campaign._execution_binding(resumed, submission_index=1)["git_commit"] == original_commit
    assert generation.campaign._execution_binding(resumed, submission_index=2)["git_commit"] == final_commit

    _assert_latest_epoch_worker_binding(resumed, storage, tmp_path / "worker", name, final_commit, monkeypatch)
    _assert_epoch_append_cannot_rebind_existing_dispatches(resumed, updated, storage, name, original_commit, final_commit)

    invalid = {**generation.campaign._requested_execution_binding(updated, final_commit), "execution_config_digest": "0" * 64}
    with pytest.raises(ValueError, match="changed execution contract"):
        lifecycle.append_execution_epoch(resumed, invalid)
    assert len(lifecycle.execution_epochs(resumed)) == len(epochs)
    common_path = canonical_path.parent / "common.yaml"
    scientific = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    scientific["grid"]["nx"] += 2
    scientific["grid"]["ny"] += 2
    common_path.write_text(yaml.safe_dump(scientific, sort_keys=False), encoding="utf-8")
    changed_science = generation.cases.config.load_campaign_config(canonical_path)
    assert changed_science.campaign_digest != campaign.campaign_digest
    assert changed_science.source_name == campaign.source_name
    with pytest.raises(ValueError, match="conflicting frozen scientific semantics"):
        lifecycle.create_campaign(changed_science, storage_root=storage)


def test_unexpected_operational_failure_stops_feeding_until_explicit_same_campaign_run(
    generation_config_factory: Any,
    world: FakeSlurm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained worker exception cannot silently launch another case or retry."""
    campaign = make_campaign(generation_config_factory, natural_count=2, max_admission_cases=1)
    name = campaign.source_name
    storage = world.storage
    initial = submit(campaign, world)
    first = initial["submissions"][0]
    first_case = first["case"]
    assert (initial["slurm_job_ids"], len(initial["submissions"])) == (["7001"], 1)
    lifecycle.record_candidate_attempt(
        "steady_flow",
        name,
        "lentil",
        str(first_case["case_id"]),
        attempt_index=1,
        stage="publication",
        failure_class="publication",
        reason_code="ValueError",
        retryable=True,
        detail="synthetic unexpected publication error",
        evidence={"dispatch_key": "7001-1", "exception_type": "ValueError"},
        storage_root=storage,
    )
    world.add_step("7001", 0, first["step_name"], "RUNNING")
    with pytest.raises(RuntimeError, match=r"stopped automatic feeding.*7001-1"):
        feed(campaign, world)
    with pytest.raises(RuntimeError, match="requires terminal step accounting"):
        submit(campaign, world)
    assert load_manifest(campaign, world)["submissions"] == initial["submissions"]
    world.clear_steps()
    world.add_step("7001", 0, first["step_name"], "FAILED", "1:0")
    for _ in range(2):
        with pytest.raises(RuntimeError, match=r"stopped automatic feeding.*7001-1"):
            feed(campaign, world)
    stopped = load_manifest(campaign, world)
    assert (stopped["slurm_job_ids"], len(stopped["submissions"])) == (["7001"], 1)
    assert lifecycle.candidate_records("steady_flow", name, storage_root=storage)[0]["attempts"][0]["evidence"]["dispatch_key"] == "7001-1"

    wait_record = {
        "classification": "temporary_license_capacity",
        "retry_budget_remaining": True,
        "first_blocked_at": "2026-01-01T00:00:00+00:00",
        "next_retry_at": "2026-01-01T00:00:20+00:00",
    }
    monkeypatch.setattr(
        generation.campaign.license_service,
        "latest_wait_for_dispatch",
        lambda *_args, dispatch_key, **_kwargs: wait_record if dispatch_key == "7001-1" else None,
    )
    monkeypatch.setattr(generation.campaign.license_service, "wait_record_is_eligible", lambda _attempt: True)
    resumed = submit(campaign, world)
    assert resumed["slurm_job_ids"] == ["7001"]  # the retry is a new ticket in the node job that is still alive
    retry = resumed["submissions"][1]
    assert (retry["case"], retry["mode"], retry["attempt_index"], retry["job_id"], retry["ticket_seq"]) == (first_case, "resume", 2, "7001", 2)
    pending = feed(campaign, world)
    assert (pending["slurm_job_ids"], len(pending["submissions"])) == (["7001"], 2)


def _synthetic_input_sources(campaign: Any) -> dict[str, dict[str, Any]]:
    """Return current persisted source ownership for isolated resume tests."""
    return {
        batch.batch_name: {
            "input_generation_id": f"input-{index:024x}",
            "source_git_commit": "a" * 40,
            "case_indices": list(batch.case_indices),
        }
        for index, batch in enumerate(campaign.batches, start=1)
    }


def _synthetic_input_references(campaign: Any) -> dict[str, dict[int, Any]]:
    """Return opaque references for tests that replace content validation."""
    references: dict[str, dict[int, Any]] = {}
    for task in cluster.campaign_tasks(campaign):
        references.setdefault(task.batch_name, {})[task.case_index] = object()
    return references


def test_campaign_tasks_are_unique(generation_config_factory: Any) -> None:
    """Every planned case has one exact identity."""
    config_path, _template = generation_config_factory()
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    assert len(tasks) == campaign.total_case_count
    assert len({(task.batch_id, task.case_id) for task in tasks}) == len(tasks)


def test_tree_mode_node_job_submission_runs_from_the_tree_with_explicit_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A controller running from a frozen tree submits a node job that executes from it and names every root, tree and snapshot."""
    tree_id = "a" * 12 + "-" + "b" * 12
    runtime = tmp_path / "runtime"
    tree = runtime / "sources" / tree_id
    storage = tmp_path / "storage"
    storage.mkdir()
    node_directory_path = tmp_path / "node"
    node_directory_path.mkdir()
    snapshot = storage / "source_snapshots" / tree_id / "snapshot.json"
    for name, value in (
        ("SOURCE_TREE_ID", tree_id),
        ("PROJECT_ROOT", str(tree)),
        ("STORAGE_ROOT", str(storage)),
        ("RUNTIME_ROOT", str(runtime)),
        ("GENERATION_SOURCE_SHA256", "b" * 64),
        ("GENERATION_SOURCE_SNAPSHOT", str(snapshot)),
        ("GENERATION_SOURCE_SNAPSHOT_SHA256", "c" * 64),
        ("GENERATION_NATIVE_VENV", str(runtime / "venvs/native")),
    ):
        monkeypatch.setenv(name, value)
    make_generation_config_factory(tree)  # places the tree's node launcher

    arguments: dict[str, Any] = {
        "seat": 1,
        "submission_token": "abcdef0123456789",
        "node_directory": node_directory_path,
        "storage_root": storage,
        "cores_per_node": 2,
        "node_job_memory": "4G",
        "node_job_wall_time": "01:00:00",
        "partition": "standard",
        "exclusions": [],
        "comsol_module": "Comsol/test",
        "python_executable": "python3",
        "comsol_executable": "comsol",
    }

    command = cluster.build_node_job_submission_command(**arguments)
    assert f"--chdir={tree}" in command
    worker_arguments = shlex.split(command[-1].removeprefix("--wrap="))
    assert worker_arguments[worker_arguments.index(str(tree / "scripts/generation_node.sh")) + 1] == str(tree)
    for binding in (
        f"PROJECT_ROOT={tree}",
        f"SOURCE_TREE_ID={tree_id}",
        f"RUNTIME_ROOT={runtime}",
        f"STORAGE_ROOT={storage}",
        f"GENERATION_SOURCE_SNAPSHOT={snapshot}",
        "GENERATION_SOURCE_SNAPSHOT_SHA256=" + "c" * 64,
    ):
        assert worker_arguments.count(binding) == 1
    monkeypatch.delenv("GENERATION_SOURCE_SNAPSHOT")
    monkeypatch.delenv("GENERATION_SOURCE_SNAPSHOT_SHA256")
    with pytest.raises(ValueError, match="Source-tree workers require"):
        cluster.build_node_job_submission_command(**arguments)


def test_scheduler_argv_and_duplicate_safe_campaign_reconciliation(
    generation_config_factory: Any,
    world: FakeSlurm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise first node-job submission and exact persisted-allocation reconciliation: a live node job takes later cases as tickets."""
    campaign = make_campaign(generation_config_factory, natural_count=3, max_admission_cases=1)
    tasks = cluster.campaign_tasks(campaign)
    storage = world.storage
    submit_node_job = cluster.submit_node_job

    def checked_submit(command: list[str], **kwargs: Any) -> str:
        discovery = generation.cases.admission.discover_input_batches(storage)
        assert not discovery.issues
        assert sum(len(source.cases) for source in discovery.sources) == campaign.total_case_count
        return submit_node_job(command, **kwargs)

    monkeypatch.setattr(cluster, "submit_node_job", checked_submit)

    first_progress: list[dict[str, Any]] = []
    first = submit(campaign, world, progress=lambda event: first_progress.append(dict(event)))
    assert first["slurm_job_ids"] == ["7001"]
    assert len(world.sbatch_arguments()) == 1
    assert world.leases == ["7001"]
    persisted = load_manifest(campaign, world)
    dispatch = persisted["submissions"][0]
    assert (dispatch["job_id"], dispatch["node_job_token"], dispatch["ticket_seq"]) == ("7001", persisted["node_jobs"][0]["token"], 1)
    assert node.ticket_path(node_directory(campaign, world, persisted["node_jobs"][0]), 1).is_file()
    assert persisted["scheduler_log_directory"] == str(common.paths.get_runtime_root().resolve() / "logs" / "generation")
    assert [event["scheduler_submissions"] for event in first_progress if event.get("operation") == "scheduler_submission"] == [1]

    blocked = AssertionError("Current canonical inputs were regenerated before resubmission.")
    active_progress: list[dict[str, Any]] = []
    with monkeypatch.context() as scoped:
        scoped.setattr(
            generation.cases.input_generation.case_service,
            "generate_case_input_bundle",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(blocked),
        )
        active = submit(campaign, world, progress=lambda event: active_progress.append(dict(event)))
    assert active["slurm_job_ids"] == ["7001"]
    assert len(world.sbatch_arguments()) == 1
    assert not any(event.get("operation") == "scheduler_submission" for event in active_progress)

    unchanged_progress: list[dict[str, Any]] = []
    unchanged = generation.campaign.feed_campaign(
        first["campaign_run_id"],
        storage_root=storage,
        progress=lambda event: unchanged_progress.append(dict(event)),
    )
    assert unchanged["slurm_job_ids"] == ["7001"]
    assert len(world.sbatch_arguments()) == 1
    assert not any(event.get("operation") == "scheduler_submission" for event in unchanged_progress)

    world.add_step("7001", 0, dispatch["step_name"], "COMPLETED")
    world.set_state("7001", "COMPLETED")  # a larger admission cap is a new execution epoch: it needs every previous node job ended
    first_case = tasks[0]
    # The first case has no real publication here, so recording its success decision is out of this test's scope.
    monkeypatch.setattr(
        generation.campaign, "_reconcile_success_decisions", lambda _manifest, _campaign, _views, records, _scheduler, **_kwargs: records
    )
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
    advanced = submit(campaign, world, max_admission_cases=2, progress=lambda event: advanced_progress.append(dict(event)))
    assert advanced["slurm_job_ids"] == ["7001", "7002"]
    assert len(world.sbatch_arguments()) == 2
    assert advanced["submissions"][1]["case"]["case_id"] == tasks[1].case_id
    assert (advanced["submissions"][1]["job_id"], advanced["submissions"][1]["ticket_seq"]) == ("7002", 1)
    assert [event["scheduler_submissions"] for event in advanced_progress if event.get("operation") == "scheduler_submission"] == [2]

    # Liveness is read once per cycle for the current user only, never by job selection; accounting names exactly the persisted allocations.
    queue_formats = {"%i|%j|%T|%k", "%i|%T|%R|%N|%V|%S|%M"}
    for line in world.log("squeue.log"):
        arguments = line.split()
        assert arguments[0] == "--noheader"
        assert sum(argument.startswith("--user=") for argument in arguments) == 1
        assert not any(argument.startswith("--jobs") for argument in arguments)
        assert arguments[-1].removeprefix("--format=") in queue_formats
    for line in world.log("sacct.log"):
        arguments = line.split()
        assert arguments[:3] == ["--noheader", "--parsable2", "--jobs"]
        assert arguments[3] in {"7001", "7001,7002"}
        assert arguments[4] in {
            "--format=JobIDRaw,State,ExitCode,Submit,Start,End,Elapsed,NodeList,AllocCPUS,Partition",
            f"--format={cluster.STEP_ACCOUNTING_FORMAT}",
        }


def test_archived_owned_job_uses_exact_accounting_without_claiming_unrelated_live_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired live-queue ID still has terminal sacct evidence for its allocation and steps, and no active ownership."""
    commands: list[list[str]] = []

    def scheduler_output(command: list[str]) -> tuple[str, str | None]:
        commands.append(command)
        if command[0] == "squeue":
            assert any(argument.startswith("--user=") for argument in command)
            assert not any(argument.startswith("--jobs") for argument in command)
            return "99999|RUNNING|unrelated|node-other|2026-10-02T00:00:00|2026-10-02T00:00:01|00:00:01", None
        assert command[0] == "sacct"
        assert command[3:5] == ["--jobs", "845436"]
        if command[5] == f"--format={cluster.STEP_ACCOUNTING_FORMAT}":
            return (
                "845436.0|sf-lentil-c0001-s1|FAILED|2:0|2026-10-02T00:00:00|2026-10-02T00:00:05|2026-10-02T00:30:00|00:29:55|node-a|1|test|1024K",
                None,
            )
        return "845436|FAILED|2:0", None

    monkeypatch.setattr(generation.campaign, "_scheduler_output", scheduler_output)
    evidence = generation.campaign._scheduler_evidence(["845436"])
    generation.campaign._require_scheduler_evidence(evidence)
    assert evidence["allocations"]["active"] == {}
    assert evidence["allocations"]["accounted"]["845436"][1:3] == ["FAILED", "2:0"]
    assert evidence["steps"][("845436", "sf-lentil-c0001-s1")]["state"] == "FAILED"
    assert evidence["active"] == evidence["accounted"] == {}  # per-dispatch rows come from the dispatch records, not from the allocation
    assert [command[0] for command in commands] == ["squeue", "sacct", "sacct"]


@pytest.mark.parametrize(
    "live_row",
    [
        "845436|PENDING",
        "845436|PENDING|Resources|N/A|2026-10-02T00:00:00|N/A|00:00:00|unexpected",
    ],
)
def test_owned_live_queue_row_requires_exact_seven_field_protocol(live_row: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject truncated and extended owned rows before terminal accounting can hide them."""

    def scheduler_output(command: list[str]) -> tuple[str, str | None]:
        if command[0] == "squeue":
            assert command[-1] == "--format=%i|%T|%R|%N|%V|%S|%M"
            return live_row, None
        assert command[0] == "sacct"
        return "845436|FAILED|2:0", None

    monkeypatch.setattr(generation.campaign, "_scheduler_output", scheduler_output)
    with pytest.raises(ValueError, match="ambiguous owned scheduler liveness evidence"):
        generation.campaign._scheduler_evidence(["845436"])


def test_invalid_current_inputs_abort_before_campaign_persistence_or_sbatch(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid current evidence before any scientific job is submitted."""
    config_path, _template = generation_config_factory(
        natural_count=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    commit = "7" * 40
    storage = tmp_path / "storage"
    monkeypatch.setenv("GENERATION_GIT_COMMIT", commit)
    lifecycle.create_campaign(campaign, storage_root=storage)
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
    assert not campaign_evidence.campaign_run_manifest_path(
        run_id,
        storage_root=storage,
    ).is_file()


@pytest.mark.parametrize(
    ("case_state", "expected_action", "expected_state"),
    [
        ("successful", "reuse", "complete"),
        ("never_started", "submit_initial", "active"),
        ("cancelled", "submit_resume", "active"),
        ("interrupted", "submit_resume", "active"),
        ("conversion_failed", "stop", "completed_with_failures"),
        ("publication_failed", "stop", "completed_with_failures"),
        ("exports_failed", "stop", "completed_with_failures"),
        ("failed", "stop", "completed_with_failures"),
        ("timed_out", "stop", "completed_with_failures"),
    ],
)
def test_campaign_resume_matrix_never_silently_retries_solver_failures(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_state: str,
    expected_action: str,
    expected_state: str,
) -> None:
    """Protect canonical reuse and restart boundaries and the stop at every recorded failure; a settled campaign drains its node job."""
    config_path, _template = generation_config_factory(natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    target = tasks[0]
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="c" * 40)
    commit = "c" * 40
    storage = tmp_path / case_state
    campaign_evidence.campaign_run_directory(
        run_id,
        storage_root=storage,
    ).mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_run_id": run_id,
        "remote_storage_root": str(storage),
        "execution_config_digest": generation.campaign._execution_config_digest(campaign),
        "git_commit": commit,
        "slurm_job_ids": [],
        "node_jobs": [],
        "node_job_intent": None,
        "submissions": [],
        "submission_intent": None,
        "submission_config": {
            "max_admission_cases": 1,
            "node_jobs": 2,
            "concurrent_cases_per_node": 4,
            "license_wait": _synthetic_retry_policy(),
        },
        "state": "active",
    }
    node_job = open_node_job(manifest, storage)
    views = [
        {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
            "state": (case_state if task == target else "successful"),
        }
        for task in tasks
    ]
    monkeypatch.setattr(lifecycle, "campaign_from_frozen_run", lambda *_args, **_kwargs: campaign)
    records = [
        {
            "reservation": {"batch_id": task.batch_id, "material_family": "lentil", "case_id": task.case_id, "origin": "primary"},
            "decision": {"outcome": "successful"} if task != target or case_state == "successful" else None,
            "attempts": [],
        }
        for task in tasks
    ]
    monkeypatch.setattr(lifecycle, "candidate_records", lambda *_args, **_kwargs: records)
    monkeypatch.setattr(lifecycle, "replacement_policy", lambda *_args, **_kwargs: [])  # no batch is stopped by its success rate
    # The stubbed records have no campaign tree behind them, so the derived cases view of a completion is stubbed too.
    monkeypatch.setattr(generation.publication.catalog, "campaign_cases_csv", lambda *_args, **_kwargs: "material,case_id\n")

    submissions = _record_dispatches(monkeypatch)
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
        lambda _job_ids: _scheduler(allocations={"7001": ["7001", "RUNNING"]}),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_reconciled",
        lambda *_args, **_kwargs: (views, 0, 0),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_persisted_input_sources",
        lambda _manifest: _synthetic_input_sources(campaign),
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

    def case_is_valid(_batch: Any, case_index: int, **_kwargs: Any) -> bool:
        return case_index != target.case_index or case_state == "successful"

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
        lifecycle,
        "seal_selected_pool",
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
    assert bool(finalized) is (expected_action == "reuse")
    drained = expected_action not in {"submit_initial", "submit_resume"}
    assert (node_job["draining"], node_job["stop_ticket_seq"]) == (drained, 1 if drained else None)
    assert node.ticket_path(campaign_evidence.campaign_node_job_directory(run_id, node_job["token"], storage_root=storage), 1).is_file() is drained


def test_malformed_persisted_job_id_fails_before_scheduler_query(
    generation_config_factory: Any,
    world: FakeSlurm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject non-numeric durable job identity without querying Slurm."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    path = campaign_evidence.campaign_run_manifest_path(manifest["campaign_run_id"], storage_root=world.storage)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["slurm_job_ids"] = ["7001 --all"]
    payload["node_jobs"][0]["job_id"] = "7001 --all"
    payload["submissions"][0]["job_id"] = "7001 --all"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        generation.campaign,
        "_scheduler_output",
        lambda _command: pytest.fail("malformed persisted identity reached scheduler query"),
    )

    with pytest.raises(ValueError, match="submission state is malformed"):
        feed(campaign, world)


def test_conversion_failures_do_not_starve_normal_admission(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep canonical admission independent of recorded conversion failures, which hold no admission."""
    config_path, _template = generation_config_factory(
        natural_count=12,
        max_admission_cases=2,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="e" * 40)
    commit = "e" * 40
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=tmp_path)
    run_directory.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_run_id": run_id,
        "remote_storage_root": str(tmp_path),
        "execution_config_digest": generation.campaign._execution_config_digest(campaign),
        "git_commit": commit,
        "slurm_job_ids": [],
        "node_jobs": [],
        "node_job_intent": None,
        "submission_intent": None,
        "submission_config": {
            "max_admission_cases": 2,
            "node_jobs": 2,
            "concurrent_cases_per_node": 4,
            "license_wait": _synthetic_retry_policy(),
        },
        "submissions": [],
        "state": "active",
    }
    open_node_job(manifest, tmp_path)
    views = [
        {
            "batch_name": task.batch_name,
            "batch_id": task.batch_id,
            "case_index": task.case_index,
            "case_id": task.case_id,
            "state": "conversion_failed" if index < 2 else "never_started",
            "failure_stage": "conversion" if index < 2 else None,
            "temporary_license_retry": None,
        }
        for index, task in enumerate(tasks)
    ]
    monkeypatch.setattr(lifecycle, "campaign_from_frozen_run", lambda *_args, **_kwargs: campaign)
    monkeypatch.setattr(lifecycle, "candidate_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(lifecycle, "replacement_policy", lambda *_args, **_kwargs: [])

    monkeypatch.setattr(campaign_evidence, "load_campaign_run", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(campaign_evidence, "campaign_from_manifest", lambda _manifest: campaign)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", lambda _job_ids: _scheduler(allocations={"7001": ["7001", "RUNNING"]}))
    monkeypatch.setattr(generation.campaign, "_reconciled", lambda *_args, **_kwargs: (views, 0, 0))
    monkeypatch.setattr(
        generation.campaign,
        "_persisted_input_sources",
        lambda _manifest: _synthetic_input_sources(campaign),
    )
    monkeypatch.setattr(
        generation.campaign,
        "_campaign_input_references",
        lambda *_args, **_kwargs: _synthetic_input_references(campaign),
    )
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: commit)
    monkeypatch.setattr(generation.campaign.batch_runtime, "completed_case_is_valid", lambda *_args, **_kwargs: False)
    dispatched = _record_dispatches(monkeypatch)

    resumed = generation.campaign.resume_campaign(run_id, storage_root=tmp_path)

    assert resumed["state"] == "active"
    assert dispatched == [(tasks[2].case_id, "initial"), (tasks[3].case_id, "initial")]


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
        "in_allocation_license_window": (
            {
                "solver_progress_started": True,
                "outcome": "solver_progress_started",
                "reason": "solver_progress_started",
            }
            if solver_started
            else None
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
        "in_allocation_license_window": (
            {
                "solver_progress_started": True,
                "outcome": "solver_progress_started",
                "reason": "solver_progress_started",
            }
            if solver_started
            else None
        ),
    }
    assert generation.campaign._view_consumes_admission(view) is expected


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (("successful", "never_started", "never_started", "never_started"), "staged"),
        (("failed", "never_started", "never_started", "never_started"), "completed_with_failures"),
        (("successful", "publication_failed", "never_started", "never_started"), "completed_with_failures"),
        (("successful", "license_blocked", "never_started", "never_started"), "license_blocked"),
        (("successful", "scheduler_unknown", "never_started", "never_started"), "active"),
    ],
)
def test_settled_admission_cap_persists_staged_only_for_a_failure_free_subset(
    generation_config_factory: Any,
    tmp_path: Path,
    states: tuple[str, ...],
    expected: str,
) -> None:
    """Persist an honest capped success state without relabeling failure or uncertainty; a settled state drains the node job."""
    admitted = sum(state != "never_started" for state in states)
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=len(states), max_admission_cases=admitted)
    campaign = generation.cases.config.load_campaign_config(config_path)
    tasks = cluster.campaign_tasks(campaign)
    storage = tmp_path / "storage"
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_directory = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_directory.mkdir(parents=True)
    manifest = generation.campaign._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_directory,
        scheduler_log_directory=tmp_path / "scheduler",
        storage_root=storage,
        input_sources=_synthetic_input_sources(campaign),
    )
    node_job = open_node_job(manifest, storage)
    views = [_admission_view(task, state) for task, state in zip(tasks, states, strict=True)]
    for task, state in zip(tasks, states, strict=True):
        if state != "never_started":
            dispatch_record(manifest, campaign, task, node_job)
    result = generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=0,
        scheduler=_scheduler(),
        storage_root=storage,
        tasks=tasks,
        max_admission_cases=admitted,
    )
    persisted = json.loads(campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage).read_text())
    assert result["state"] == persisted["state"] == expected
    assert len(result["submissions"]) == admitted
    assert campaign_evidence._validate_campaign_run_header(persisted, run_id=run_id) == ["7001"]
    settled = expected in {"staged", "completed_with_failures"}
    assert persisted["node_jobs"][0]["draining"] is settled
    assert (persisted["node_jobs"][0]["stop_ticket_seq"] is not None) is settled


@pytest.mark.parametrize("stopped", ["own", "other"])
def test_only_a_stopped_batch_loses_fresh_admission_while_its_running_case_continues(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stopped: str,
) -> None:
    """A stopped batch admits no fresh case while its running case keeps the node job; another batch's stop leaves it admitting."""
    failure_count = 2
    config_path, _template = generation_config_factory(
        natural_count=failure_count + 3,
        max_admission_cases=failure_count + 3,
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
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="e" * 40)
    campaign_evidence.campaign_run_directory(run_id, storage_root=tmp_path).mkdir(parents=True)
    manifest: dict[str, Any] = {
        "campaign_run_id": run_id,
        "execution_config_digest": generation.campaign._execution_config_digest(campaign),
        "git_commit": "e" * 40,
        "remote_storage_root": str(tmp_path),
        "slurm_job_ids": [],
        "node_jobs": [],
        "node_job_intent": None,
        "submission_config": {
            "max_admission_cases": failure_count + 3,
            "node_jobs": 2,
            "concurrent_cases_per_node": 4,
            "license_wait": _synthetic_retry_policy(),
        },
        "submission_intent": None,
        "admission_reservations": [],
        "submissions": [],
        "state": "active",
    }
    node_job = open_node_job(manifest, tmp_path, job_id="811")
    scheduler = _scheduler(allocations={"811": ["811", "RUNNING"]})
    for task in tasks[:failure_count]:  # every earlier case was admitted fresh and its step ended in failure
        key = f"811-{dispatch_record(manifest, campaign, task, node_job)['submission_index']}"
        scheduler["accounted"][key] = [key, "FAILED", "1:0"]
    dispatch_record(manifest, campaign, tasks[failure_count], node_job)  # the case that is still running
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: "e" * 40)
    dispatched = _record_dispatches(monkeypatch)

    advanced = generation.campaign._fill_submission_capacity(
        manifest,
        campaign,
        views,
        pending_jobs=0,
        running_jobs=1,
        scheduler=scheduler,
        storage_root=tmp_path,
        stopped={tasks[0].batch_id if stopped == "own" else "another-original-batch"},
        max_admission_cases=failure_count + 3,
    )

    assert advanced["state"] == "active"
    assert dispatched == ([] if stopped == "own" else [(view["case_id"], "initial") for view in fresh])
    assert [job["draining"] for job in advanced["node_jobs"]] == [False]
    assert scheduler["allocations"]["active"] == {"811": ["811", "RUNNING"]}


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


def test_admission_reservations_are_exact_unsent_plan_members(
    generation_config_factory: Any,
) -> None:
    """Reject foreign or already-submitted operational reservations."""
    config_path, _template = generation_config_factory(
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
    solver_receipt = {
        "solver_progress_started": True,
        "outcome": "solver_progress_started",
        "reason": "solver_progress_started",
    }
    assert generation.campaign._license_retry_is_active("pending", retry, None)
    assert generation.campaign._license_retry_is_active("active", retry, None)
    assert not generation.campaign._license_retry_is_active("active", retry, solver_receipt)


def test_graceful_then_force_cancel_share_campaign_owner_and_stay_nonterminal(
    generation_config_factory: Any,
    world: FakeSlurm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signal running node jobs gracefully first and force only on request; the request is durable before any command."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    run_id = manifest["campaign_run_id"]
    world.set_state("7001", "RUNNING", node="node-a")
    commands: list[list[str]] = []
    real_run = subprocess.run

    def run(command: list[str], **kwargs: Any) -> Any:
        if command[0] == "scancel":
            expected_state = "force_cancel_requested" if "--signal=KILL" in command else "cancel_requested"
            assert load_manifest(campaign, world)["state"] == expected_state
            assert common.paths.resolve_generation_cancellations_path(
                campaign_evidence.campaign_execution_directory(run_id, storage_root=world.storage)
            ).is_file()
            commands.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(generation.campaign.subprocess, "run", run)

    graceful = generation.campaign.cancel_campaign(run_id, storage_root=world.storage)
    assert commands == [["scancel", "--signal=TERM", "--batch", "7001"]]
    assert graceful["attempts"][-1]["mode"] == "graceful"
    running = generation.campaign.campaign_status(run_id, storage_root=world.storage)
    assert running["campaign_state"] == "cancel_requested"

    forced = generation.campaign.cancel_campaign(run_id, storage_root=world.storage, force=True)
    assert commands[-1] == ["scancel", "--signal=KILL", "--full", "7001"]
    assert [attempt["mode"] for attempt in forced["attempts"]] == [
        "graceful",
        "force",
    ]

    world.set_state("7001", "CANCELLED")
    cancelled = generation.campaign.campaign_status(run_id, storage_root=world.storage)
    assert cancelled["campaign_state"] == "force_cancel_requested"


def test_cancelled_canonical_campaign_requires_explicit_terminal_continuation(generation_config_factory: Any, world: FakeSlurm) -> None:
    """Keep cancellation sticky until exact node jobs drain and the user reruns CONFIG."""
    campaign = make_campaign(generation_config_factory)
    first = submit(campaign, world)
    run_id = first["campaign_run_id"]
    world.set_state("7001", "RUNNING", node="node-a")
    generation.campaign.cancel_campaign(run_id, storage_root=world.storage)
    assert feed(campaign, world)["state"] == "cancel_requested"
    assert len(world.sbatch_arguments()) == 1
    with pytest.raises(RuntimeError, match="terminal scheduler evidence"):
        submit(campaign, world)
    for nonterminal in ("RUNNING", "PENDING", "COMPLETING", "UNKNOWN"):
        world.set_state("7001", nonterminal)
        with pytest.raises(RuntimeError, match="terminal scheduler evidence"):
            submit(campaign, world)
        assert load_manifest(campaign, world)["state"] == "cancel_requested"
        assert len(world.sbatch_arguments()) == 1
    world.set_state("7001", "CANCELLED")
    world.add_step("7001", 0, first["submissions"][0]["step_name"], "CANCELLED", "0:15")  # the step the cancellation interrupted
    assert feed(campaign, world)["state"] == "cancel_requested"
    assert generation.campaign.campaign_status(run_id, storage_root=world.storage)["suggested_next_command"] == (
        f"./scripts/generation run {first['campaign_config']}"
    )
    resumed = submit(campaign, world)
    assert resumed["campaign_run_id"] == run_id
    assert resumed["slurm_job_ids"] == ["7001", "7002"]
    assert (resumed["submissions"][-1]["mode"], resumed["submissions"][-1]["attempt_index"], resumed["submissions"][-1]["job_id"]) == (
        "resume",
        2,
        "7002",
    )
    assert len(world.sbatch_arguments()) == 2
    receipt = common.paths.resolve_generation_cancellations_path(campaign_evidence.campaign_execution_directory(run_id, storage_root=world.storage))
    assert receipt.is_file()


def test_canonical_attempt_uses_persisted_dispatch_identity_for_exact_ownership(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Join the producer's allocation and submission-index receipt to one owned dispatch and epoch."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    storage = world.storage
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    bind_worker(monkeypatch, manifest)
    receipt_path = generation.runtime.record_case_failure(
        batch,
        task.case_index,
        RuntimeError("synthetic input failure"),
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node-a",
        work_directory=None,
        storage_root=storage,
        scratch_cleanup_status="not_created",
        failure_stage="input",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (receipt["job_id"], receipt["slurm_step_id"], receipt["submission_index"]) == ("7001", "0", 1)
    assert "scheduler_job_id" not in receipt
    reference = generation.cases.input_generation.admit_configured_input_references(
        batch, storage_root=storage, validation_depth="evidence", git_commit=COMMIT
    )[task.case_index]
    admitted = generation.campaign._admitted_case_attempt(manifest, batch, task, input_reference=reference, storage_root=storage)
    assert admitted is not None
    assert admitted.attempt.receipt_path == receipt_path
    assert admitted.current is True
    for field, wrong in (("job_id", "7002"), ("submission_index", 2), ("solver_git_commit", "b" * 40)):
        tampered = {**receipt, field: wrong}
        receipt_path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Attempt evidence disagrees"):
            generation.campaign._admitted_case_attempt(manifest, batch, task, input_reference=reference, storage_root=storage)


@pytest.mark.parametrize("entry_path", ["authored_rule", "no_retry_flag"])
def test_capacity_failure_with_disabled_license_protection_is_a_recorded_case_failure(
    entry_path: str,
    generation_config_factory: Any,
    world: FakeSlurm,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the licence protection a capacity error is an ordinary solver failure, not an unreadable licence block."""
    campaign = make_campaign(generation_config_factory, license_retry_enabled=entry_path != "authored_rule")
    if entry_path == "no_retry_flag":
        campaign = generation.cases.config.load_campaign_config(campaign.source_path, execution_overrides={"no_retry": True})
    assert campaign.execution_values["runtime"]["license_wait"]["enabled"] is False
    manifest = submit(campaign, world)
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    bind_worker(monkeypatch, manifest)
    evidence = license_service.TemporaryLicenseCapacityClassification(
        classification=license_service.TEMPORARY_LICENSE_CAPACITY,
        feature="COMSOL license acquisition",
        license_code=None,
        matched_signatures=("synthetic",),
        raw_excerpt="synthetic",
    )
    error = license_service.TemporaryLicenseCapacityError(
        "COMSOL could not obtain temporary floating-license capacity.",
        work_directory=tmp_path,
        command=("comsol",),
        exit_code=1,
        evidence=evidence,
    )
    generation.runtime.record_case_failure(
        batch,
        task.case_index,
        error,
        worker_slot=0,
        scheduler_kind="slurm",
        allocated_node="node-a",
        work_directory=None,
        storage_root=world.storage,
        scratch_cleanup_status="not_created",
        failure_stage="solver",
    )

    assert generation.runtime.case_failure_is_recorded(
        batch, task.case_index, storage_root=world.storage, execution_run_id=manifest["campaign_run_id"], git_commit=COMMIT
    )
    world.add_step("7001", 0, manifest["submissions"][0]["step_name"], "FAILED", "1:0")
    world.set_state("7001", "COMPLETED")
    status = generation.campaign.campaign_status(manifest["campaign_run_id"], storage_root=world.storage)
    (case,) = status["cases"]
    assert (case["state"], case["failure_stage"]) == ("failed", "solver")


def test_resume_recovers_next_case_intent_while_first_case_is_pending(
    generation_config_factory: Any,
    world: FakeSlurm,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover an interrupted second dispatch before admitting another logical case, without a second node job."""
    campaign = make_campaign(generation_config_factory, natural_count=3, max_admission_cases=2)
    crash_at(
        monkeypatch,
        generation.campaign.node_service,
        "ensure_node_ticket",
        after=False,
        when=lambda _directory, ticket: ticket.seq == 2,
    )
    with pytest.raises(Crash):
        submit(campaign, world)
    interrupted = load_manifest(campaign, world)
    assert interrupted["submission_intent"] == 2
    assert [item["status"] for item in interrupted["submissions"]] == ["dispatched", "dispatching"]
    directory = node_directory(campaign, world, interrupted["node_jobs"][0])
    assert sorted(path.name for path in (directory / "tickets").iterdir()) == ["000001.json"]

    resumed = generation.campaign.resume_campaign(interrupted["campaign_run_id"], storage_root=world.storage)

    assert resumed["submission_intent"] is None
    assert resumed["slurm_job_ids"] == ["7001"]
    assert [item["status"] for item in resumed["submissions"]] == ["dispatched", "dispatched"]
    assert len({item["case"]["case_id"] for item in resumed["submissions"]}) == 2
    assert sorted(path.name for path in (directory / "tickets").iterdir()) == ["000001.json", "000002.json"]
    assert len(world.sbatch_arguments()) == 1


def test_config_owned_plan_is_machine_parseable_and_read_only(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose configured resources as JSON without allocating campaign state."""
    config_path, _template = generation_config_factory()
    campaign = generation.cases.config.load_campaign_config(config_path)
    commit = "d" * 40
    _set_native_launch_evidence(monkeypatch, commit)
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
    config_path, _template = generation_config_factory()
    original = generation.cases.config.load_campaign_config(config_path)
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["run"]["max_admission_cases"] = 2
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")
    changed = generation.cases.config.load_campaign_config(config_path)

    assert [task.case_id for task in cluster.campaign_tasks(changed)] == [task.case_id for task in cluster.campaign_tasks(original)]
    assert [batch.scientific_config_digest for batch in changed.batches] == [batch.scientific_config_digest for batch in original.batches]
    assert [batch.case_input_config_digest for batch in changed.batches] == [batch.case_input_config_digest for batch in original.batches]
    commit = "9" * 40
    assert generation.campaign.campaign_run_id(changed, git_commit=commit) == generation.campaign.campaign_run_id(
        original,
        git_commit=commit,
    )


def test_validate_config_exposes_resolved_campaign_ownership(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose resolved counts, seeds, packages, and purpose scope without mutation."""
    config_path, _template = generation_config_factory()
    campaign = generation.cases.config.load_campaign_config(config_path, require_executable=False)
    monkeypatch.setattr(
        cli_generation.readiness_service,
        "campaign_unresolved_gates",
        lambda _path, **_kwargs: {},
    )

    status = cli_generation.main(["validate-config", str(config_path), "--allow-incomplete"])

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["campaign_purpose"] == campaign.campaign_purpose
    assert output["case_counts"]["derived_total"] == campaign.total_case_count
    assert output["seed_plan"]["campaign_seed"] == campaign.batches[0].scientific_values["campaign_seed"]
    assert "dataset_package_requests" not in output
    assert "dataset_package_inventory" not in output


def test_resume_rejects_unsupported_handle_before_runtime_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an invalid execution handle before storage, locks, or scheduler work."""

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        message = "invalid handle reached runtime effects"
        raise AssertionError(message)

    monkeypatch.setattr(generation.campaign.campaign_evidence, "campaign_run_directory", forbidden)
    monkeypatch.setattr(generation.campaign, "_scheduler_evidence", forbidden)
    with pytest.raises(ValueError, match="canonical source campaign execution handle"):
        generation.campaign.resume_campaign("unsafe_execution_handle")


# -- shared dispatcher primitives: seat queue and step accounting --


def _fake_squeue(monkeypatch: pytest.MonkeyPatch, *, stdout: str = "", returncode: int = 0, stderr: str = "") -> list[list[str]]:
    """Replace the squeue runner and the user name; return the argument vectors it receives."""
    commands: list[list[str]] = []

    def run(arguments: Any) -> subprocess.CompletedProcess[str]:
        commands.append(list(arguments))
        return subprocess.CompletedProcess(arguments, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(cluster, "_run_squeue", run)
    monkeypatch.setattr(cluster.getpass, "getuser", lambda: "tester")
    return commands


def test_query_seat_jobs_returns_every_own_nonterminal_seat_job_with_state_and_comment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows of any dispatcher come back as job ID to (seat, state, comment); a seat without a row is free."""
    commands = _fake_squeue(monkeypatch, stdout="101|gen-seat-1|RUNNING|tokenaaaa\n\n102|gen-seat-2|PENDING|\n103|gen-seat-1|PENDING|a|b\n")
    assert cluster.query_seat_jobs([2, 1, 2]) == {
        "101": (1, "RUNNING", "tokenaaaa"),
        "102": (2, "PENDING", ""),
        "103": (1, "PENDING", "a|b"),
    }
    assert commands == [["squeue", "--noheader", "--user=tester", "--name=gen-seat-1,gen-seat-2", "--format=%i|%j|%T|%k"]]
    commands = _fake_squeue(monkeypatch)
    assert cluster.query_seat_jobs([1]) == {}
    assert commands[0][3] == "--name=gen-seat-1"
    commands = _fake_squeue(monkeypatch)
    assert cluster.query_seat_jobs([4, 3]) == {}
    assert commands[0][3] == "--name=gen-seat-3,gen-seat-4"


@pytest.mark.parametrize("seats", [[], [0], [5], [True], [1, 5]])
def test_query_seat_jobs_rejects_missing_or_unknown_seats_before_any_query(monkeypatch: pytest.MonkeyPatch, seats: list[int]) -> None:
    """Only the node-job slots can be queried."""
    commands = _fake_squeue(monkeypatch)
    with pytest.raises(ValueError, match="slot"):
        cluster.query_seat_jobs(seats)
    assert commands == []


@pytest.mark.parametrize(
    ("stdout", "returncode", "match"),
    [
        ("", 1, "squeue: down"),
        ("101|gen-seat-1|RUNNING\n", 0, "malformed"),
        ("abc|gen-seat-1|RUNNING|token\n", 0, "malformed"),
        ("101_4|gen-seat-1|RUNNING|token\n", 0, "malformed"),
        ("101|other-job|RUNNING|token\n", 0, "malformed"),
        ("101|gen-seat-2|RUNNING|token\n", 0, "malformed"),
        ("101|gen-seat-1||token\n", 0, "malformed"),
        ("101|gen-seat-1|RUNNING|a\n101|gen-seat-1|PENDING|b\n", 0, "twice"),
    ],
)
def test_query_seat_jobs_fails_closed_on_a_failed_query_or_a_malformed_row(
    monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int, match: str
) -> None:
    """An unreadable queue is an error, never a free seat."""
    _fake_squeue(monkeypatch, stdout=stdout, returncode=returncode, stderr="squeue: down")
    with pytest.raises(RuntimeError, match=match):
        cluster.query_seat_jobs([1])


@pytest.mark.parametrize("failure", [FileNotFoundError("squeue"), subprocess.TimeoutExpired("squeue", 60)])
def test_query_seat_jobs_reports_an_unrunnable_squeue(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    """A missing executable or a timeout becomes a RuntimeError that keeps its cause."""

    def run(_arguments: Any) -> subprocess.CompletedProcess[str]:
        raise failure

    monkeypatch.setattr(cluster, "_run_squeue", run)
    with pytest.raises(RuntimeError, match="Could not query") as caught:
        cluster.query_seat_jobs([1])
    assert caught.value.__cause__ is failure


def _sacct_row(job_id: str, name: str, state: str = "COMPLETED", *, exit_code: str = "0:0", max_rss: str = "") -> str:
    """Return one parsable2 row in STEP_ACCOUNTING_FORMAT order."""
    times = ["2026-10-27T10:00:00", "2026-10-27T10:00:05", "2026-10-27T10:10:00", "00:09:55"]
    return "|".join([job_id, name, state, exit_code, *times, "hpc117", "4", "standard", max_rss])


def test_step_accounting_format_lists_the_fields_the_step_rows_are_parsed_from() -> None:
    """The single sacct format starts with the row identity, then carries every parsed field."""
    assert cluster.STEP_ACCOUNTING_FORMAT.split(",") == [
        "JobIDRaw",
        "JobName",
        "State",
        "ExitCode",
        "Submit",
        "Start",
        "End",
        "Elapsed",
        "NodeList",
        "AllocCPUS",
        "Partition",
        "MaxRSS",
    ]


def test_parse_step_accounting_keeps_only_numeric_steps_and_normalizes_states() -> None:
    """Allocation, batch and extern rows are skipped; the step state loses its annotation and suffix."""
    output = "\n".join(
        [
            _sacct_row("100", "gen-seat-1", "RUNNING"),
            _sacct_row("100.batch", "batch", "RUNNING"),
            _sacct_row("100.extern", "extern", "RUNNING"),
            _sacct_row("100.0", "case-1", "COMPLETED", max_rss="2048K"),
            _sacct_row("100.1", "case-2", "CANCELLED by 123", exit_code="0:15"),
            _sacct_row("100.2", "case-3", "CANCELLED+", exit_code="0:15"),
            _sacct_row("100.3", "case-4", "OUT_OF_MEMORY", exit_code="0:125", max_rss="1G"),
            "",
        ]
    )
    rows = cluster.parse_step_accounting(output)
    assert sorted(rows) == [("100", "case-1"), ("100", "case-2"), ("100", "case-3"), ("100", "case-4")]
    assert rows[("100", "case-1")] == {
        "step_id": "0",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "submit": "2026-10-27T10:00:00",
        "start": "2026-10-27T10:00:05",
        "end": "2026-10-27T10:10:00",
        "elapsed": "00:09:55",
        "node": "hpc117",
        "cpus": "4",
        "partition": "standard",
        "max_rss": "2048K",
    }
    assert (rows[("100", "case-2")]["state"], rows[("100", "case-2")]["exit_code"]) == ("CANCELLED", "0:15")
    assert rows[("100", "case-3")]["state"] == "CANCELLED"
    assert (rows[("100", "case-4")]["state"], rows[("100", "case-4")]["max_rss"]) == ("OUT_OF_MEMORY", "1G")
    assert rows[("100", "case-2")]["max_rss"] == ""
    assert cluster.parse_step_accounting("") == {}


@pytest.mark.parametrize("order", [(9, 10), (10, 9)])
def test_parse_step_accounting_keeps_the_highest_step_id_of_a_duplicate_name(order: tuple[int, int]) -> None:
    """A repeated step name inside one allocation resolves to its highest step ID by number, whatever the row order."""
    output = "\n".join(_sacct_row(f"100.{step}", "case-1", "FAILED" if step == 9 else "COMPLETED") for step in order)
    output += "\n" + _sacct_row("200.0", "case-1", "TIMEOUT")
    rows = cluster.parse_step_accounting(output)
    assert (rows[("100", "case-1")]["step_id"], rows[("100", "case-1")]["state"]) == ("10", "COMPLETED")
    assert rows[("200", "case-1")]["state"] == "TIMEOUT"
    assert len(rows) == 2


def test_parse_step_accounting_rejects_a_truncated_step_row() -> None:
    """A step row without every field is an error, while a truncated allocation row is not a step."""
    assert cluster.parse_step_accounting("100|gen-seat-1|RUNNING\n") == {}
    with pytest.raises(ValueError, match="malformed step row"):
        cluster.parse_step_accounting("100.0|case-1|COMPLETED|0:0\n")
