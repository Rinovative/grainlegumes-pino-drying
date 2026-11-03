"""Campaign node-job submission, ticket dispatch, and persisted-contract lifecycle.

Responsibilities:
  - Check the node-job seat choice, submission order, lease result, and recovery of interrupted submissions
  - Check dispatch capacity, ticket contents, dispatch order, recovery at each crash point, and draining
  - Check the run.json schema version 2 validators and the refusal of schema version 1
  - Check the per-dispatch scheduler view, not-started persistence, worker binding, cancellation replay, licence retry, and node-job status
  - Check that every validly published case gets exactly one success decision, from its worker or else from the feeder
  - Check that the feeder's automatic failure decision skips every ineligible attempt, dispatch, and pending intent

Design principles:
  - Fake sbatch, squeue, sacct, sinfo, and scancel executables precede every real Slurm command on PATH
  - A crash is a BaseException raised at one exact point; the next feed cycle is the recovery under test
  - A valid publication is a stand-in at the batch-runtime boundary; decisions and the feeder are the real ones

This module does NOT:
  - Run COMSOL or the node worker
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from support.fake_slurm import (
    COMMIT,
    NODE_ID,
    Crash,
    FakeSlurm,
    bind_worker,
    crash_at,
    feed,
    load_manifest,
    make_campaign,
    node_directory,
    run_id_of,
    submit,
)

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation import generation_campaign_status as status_formatting
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_dispatch_identity as dispatch_identity
from src.generation.runtime import generation_runtime_node as node
from src.generation.runtime import generation_runtime_progress as progress

if TYPE_CHECKING:
    from collections.abc import Callable


def test_first_feed_submits_a_node_job_and_dispatches_ticketed_case_steps(generation_config_factory: Any, world: FakeSlurm) -> None:
    """The intent, node directory, sbatch, lease, and ticket publish in order and leave the exact persisted contract."""
    campaign = make_campaign(generation_config_factory, natural_count=2)
    manifest = submit(campaign, world)

    assert manifest["schema_version"] == campaign_evidence.RUN_MANIFEST_SCHEMA_VERSION == 2
    assert "scheduler_job_name" not in manifest
    assert manifest["state"] == "active"
    assert manifest["node_job_intent"] is None
    assert manifest["submission_intent"] is None
    (node_job,) = manifest["node_jobs"]
    assert manifest["slurm_job_ids"] == [node_job["job_id"]] == ["7001"]
    assert node_job["status"] == "submitted"
    assert node_job["lease"] == "ok"
    assert node_job["seat"] == 1
    assert node_job["node_directory"] == f"_record/execution/node_jobs/{node_job['token']}"
    assert (node_job["draining"], node_job["stop_ticket_seq"], node_job["cancel_published"], node_job["error"]) == (False, None, False, None)
    assert world.leases == ["7001"]
    directory = node_directory(campaign, world, node_job)
    assert sorted(path.name for path in directory.iterdir()) == ["control", "receipts", "tickets"]

    (sbatch,) = world.sbatch_arguments()
    assert "--job-name=gen-seat-1" in sbatch
    assert f"--comment={node_job['token']}" in sbatch
    assert "--dependency=singleton" in sbatch
    assert "--cpus-per-task=2" in sbatch
    assert "--exclude=tiny1" in sbatch  # at-least rule: only the smaller node of the fake sinfo is excluded, the 64-CPU node stays admitted
    assert any(argument.startswith("--wrap=") and "node-worker" in argument and str(directory) in argument for argument in sbatch)

    first, second = manifest["submissions"]
    assert [record["submission_index"] for record in manifest["submissions"]] == [1, 2]
    assert [(record["status"], record["mode"], record["attempt_index"], record["ticket_seq"]) for record in manifest["submissions"]] == [
        ("dispatched", "initial", 1, 1),
        ("dispatched", "initial", 1, 2),
    ]
    assert (first["node_job_token"], first["job_id"]) == (node_job["token"], "7001")
    assert first["step_name"] == "sf-lentil-c0001-s1"
    assert second["step_name"] == "sf-lentil-c0002-s2"
    assert node_job["next_ticket_seq"] == 3

    ticket = json.loads(node.ticket_path(directory, 1).read_text(encoding="utf-8"))
    tree = common.paths.get_project_root().resolve()
    assert ticket["kind"] == "step"
    assert (ticket["group"], ticket["cores"], ticket["memory"], ticket["label"]) == (0, 1, "1G", first["step_name"])
    timeout = int(campaign.execution_values["runtime"]["timeout_seconds"])
    assert ticket["required_seconds"] == timeout + 600 + 120
    assert ticket["time_limit"] == f"{(timeout + 600) // 3600:02d}:{(timeout + 600) % 3600 // 60:02d}:{(timeout + 600) % 60:02d}"
    assert ticket["argv"] == [
        str(tree / "scripts" / "generation_node.sh"),
        str(tree),
        "campaign-case",
        manifest["campaign_run_id"],
        first["case"]["batch_name"],
        "1",
        "1",
    ]
    assert ticket["env"] == {
        "GENERATION_ATTEMPT_INDEX": "1",
        "GENERATION_CAMPAIGN_RUN_ID": manifest["campaign_run_id"],
        "GENERATION_DISPATCH_KEY": "7001-1",
        "GENERATION_SUBMISSION_INDEX": "1",
    }

    # Continuing the same campaign keeps the node jobs and dispatches and adds nothing while nothing is dispatchable.
    again = submit(campaign, world)
    assert again["node_jobs"] == manifest["node_jobs"]
    assert again["submissions"] == manifest["submissions"]
    assert len(world.sbatch_arguments()) == 1
    status = generation.campaign.campaign_status(run_id_of(campaign), storage_root=world.storage)
    assert status["slurm_job_ids"] == ["7001"]


def test_unresolved_dispatches_fill_a_node_job_before_a_second_seat_is_used(generation_config_factory: Any, world: FakeSlurm) -> None:
    """A node job holds twice its concurrent case slots in unresolved dispatches; the next dispatch needs the next seat."""
    campaign = make_campaign(generation_config_factory, natural_count=6)
    manifest = submit(campaign, world)

    first, second = manifest["node_jobs"]
    assert (first["seat"], second["seat"]) == (1, 2)
    owners = [record["node_job_token"] for record in manifest["submissions"]]
    assert owners == [first["token"]] * 4 + [second["token"]] * 2
    assert [record["ticket_seq"] for record in manifest["submissions"]] == [1, 2, 3, 4, 1, 2]
    assert [job["next_ticket_seq"] for job in manifest["node_jobs"]] == [5, 3]
    assert [arguments for arguments in world.sbatch_arguments() if "--job-name=gen-seat-2" in arguments]


def test_interrupted_node_job_intent_is_proven_absent_only_after_the_grace_and_a_working_query(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before sbatch leaves an intent that waits out its grace, then fails and lets a new node job go."""
    campaign = make_campaign(generation_config_factory)
    crash_at(monkeypatch, cluster, "submit_node_job", after=False)
    with pytest.raises(Crash):
        submit(campaign, world)
    intent = load_manifest(campaign, world)
    token = intent["node_job_intent"]
    assert token is not None
    assert [(job["token"], job["status"]) for job in intent["node_jobs"]] == [(token, "submitting")]
    assert intent["slurm_job_ids"] == []
    assert world.sbatch_arguments() == []
    assert sorted(path.name for path in node_directory(campaign, world, intent["node_jobs"][0]).iterdir()) == ["control", "receipts", "tickets"]

    # Within the grace nothing is decided and nothing is submitted.
    assert feed(campaign, world)["node_job_intent"] == token
    assert world.sbatch_arguments() == []

    # A failing queue query keeps the intent even when the grace is over: unknown is not absent.
    monkeypatch.setattr(generation.campaign, "_NODE_INTENT_GRACE_SECONDS", 0.0)
    world.fail("squeue")
    assert feed(campaign, world)["node_job_intent"] == token
    world.fail("squeue", failing=False)

    recovered = feed(campaign, world)
    failed, replacement = recovered["node_jobs"]
    assert (failed["token"], failed["status"], failed["job_id"]) == (token, "submission_failed", None)
    assert "no queued job" in failed["error"]
    assert (replacement["status"], replacement["job_id"]) == ("submitted", "7001")
    assert recovered["node_job_intent"] is None
    assert recovered["slurm_job_ids"] == ["7001"]
    assert len(world.sbatch_arguments()) == 1


@pytest.mark.parametrize("evidence", ["queue_comment", "worker_record"])
def test_node_job_submitted_before_the_crash_is_recovered_without_a_second_sbatch(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch, evidence: str
) -> None:
    """A job that sbatch accepted is found by its submission token in the queue, else by the worker's node record."""
    campaign = make_campaign(generation_config_factory)
    crash_at(monkeypatch, cluster, "submit_node_job", after=True)
    with pytest.raises(Crash):
        submit(campaign, world)
    intent = load_manifest(campaign, world)
    record = intent["node_jobs"][0]
    assert (intent["node_job_intent"], record["status"], intent["slurm_job_ids"]) == (record["token"], "submitting", [])
    assert world.jobs() == [["7001", "gen-seat-1", "PENDING", record["token"], ""]]
    if evidence == "worker_record":
        world.set_state("7001", "COMPLETED")
        node_record = {"schema": node.NODE_RECORD_SCHEMA, "schema_version": node.SCHEMA_VERSION, "job_id": "7001"}
        (node_directory(campaign, world, record) / node.NODE_RECORD_FILE).write_text(json.dumps(node_record), encoding="utf-8")

    recovered = feed(campaign, world)
    first = recovered["node_jobs"][0]
    assert (first["token"], first["status"], first["job_id"], first["lease"]) == (record["token"], "submitted", "7001", "ok")
    assert recovered["node_job_intent"] is None
    assert recovered["slurm_job_ids"][0] == "7001"
    if evidence == "queue_comment":
        assert len(world.sbatch_arguments()) == 1
        assert [item["node_job_token"] for item in recovered["submissions"]] == [record["token"]]
    else:
        # The recovered job already ended, so it takes no dispatch; a new node job runs the case.
        assert first["terminal_observed_at"] is not None
        assert len(world.sbatch_arguments()) == 2
        assert [item["job_id"] for item in recovered["submissions"]] == ["7002"]
        assert recovered["slurm_job_ids"] == ["7001", "7002"]


def test_dispatch_intent_before_the_ticket_is_completed_by_recovery(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the dispatch intent leaves no ticket; recovery publishes it once and finalizes the record."""
    campaign = make_campaign(generation_config_factory)
    crash_at(monkeypatch, generation.campaign.node_service, "ensure_node_ticket", after=False)
    with pytest.raises(Crash):
        submit(campaign, world)
    intent = load_manifest(campaign, world)
    (record,) = intent["submissions"]
    node_job = intent["node_jobs"][0]
    assert (intent["submission_intent"], record["status"], node_job["next_ticket_seq"]) == (1, "dispatching", 1)
    directory = node_directory(campaign, world, node_job)
    assert not node.ticket_path(directory, 1).exists()

    recovered = feed(campaign, world)
    (final,) = recovered["submissions"]
    assert (recovered["submission_intent"], final["status"], final["ticket_seq"]) == (None, "dispatched", 1)
    assert recovered["node_jobs"][0]["next_ticket_seq"] == 2
    assert sorted(path.name for path in (directory / "tickets").iterdir()) == ["000001.json"]
    assert len(world.sbatch_arguments()) == 1


def test_published_ticket_with_an_open_intent_is_finalized_without_a_duplicate(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the ticket was published replays the identical ticket and finalizes the same single record."""
    campaign = make_campaign(generation_config_factory)
    crash_at(monkeypatch, generation.campaign.node_service, "ensure_node_ticket", after=True)
    with pytest.raises(Crash):
        submit(campaign, world)
    intent = load_manifest(campaign, world)
    node_job = intent["node_jobs"][0]
    directory = node_directory(campaign, world, node_job)
    published = node.ticket_path(directory, 1).read_text(encoding="utf-8")
    assert (intent["submission_intent"], intent["submissions"][0]["status"]) == (1, "dispatching")

    recovered = feed(campaign, world)
    assert [(item["submission_index"], item["status"]) for item in recovered["submissions"]] == [(1, "dispatched")]
    assert recovered["submission_intent"] is None
    assert node.ticket_path(directory, 1).read_text(encoding="utf-8") == published
    assert sorted(path.name for path in (directory / "tickets").iterdir()) == ["000001.json"]


def test_dispatch_to_an_ended_node_job_without_a_ticket_is_not_started_and_restarts_without_an_attempt(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ticket never reached a node job that ended: the dispatch is not started and the case goes to a new node job."""
    campaign = make_campaign(generation_config_factory)
    crash_at(monkeypatch, generation.campaign.node_service, "ensure_node_ticket", after=False)
    with pytest.raises(Crash):
        submit(campaign, world)
    world.set_state("7001", "COMPLETED")

    recovered = feed(campaign, world)
    stale, retry = recovered["submissions"]
    assert (stale["status"], stale["job_id"], stale["ticket_seq"]) == ("not_started", "7001", 1)
    assert "ended before the ticket" in stale["error"]
    assert not node.ticket_path(node_directory(campaign, world, recovered["node_jobs"][0]), 1).exists()
    assert recovered["node_jobs"][0]["terminal_observed_at"] is not None
    assert recovered["node_jobs"][0]["next_ticket_seq"] == 1
    assert (retry["status"], retry["mode"], retry["attempt_index"], retry["job_id"], retry["ticket_seq"]) == ("dispatched", "restart", 1, "7002", 1)
    assert retry["step_name"].endswith("-s2")
    assert recovered["submission_intent"] is None
    assert recovered["slurm_job_ids"] == ["7001", "7002"]


def test_seats_held_by_other_jobs_block_submission_until_one_is_free(generation_config_factory: Any, world: FakeSlurm) -> None:
    """Any own job on a seat, whichever campaign submitted it, occupies that seat; unknown is not free."""
    campaign = make_campaign(generation_config_factory)
    world.add_foreign_job("gen-seat-1", "6001")
    world.add_foreign_job("gen-seat-2", "6002")
    blocked = submit(campaign, world)
    assert (blocked["node_jobs"], blocked["submissions"], blocked["state"]) == ([], [], "active")
    assert world.sbatch_arguments() == []

    world.set_state("6002", "COMPLETED")
    freed = feed(campaign, world)
    (node_job,) = freed["node_jobs"]
    assert node_job["seat"] == 2
    assert "--job-name=gen-seat-2" in world.sbatch_arguments()[0]
    assert [item["status"] for item in freed["submissions"]] == ["dispatched"]

    world.fail("squeue")
    with pytest.raises(RuntimeError, match=r"squeue|current scheduler user|Cannot reconcile"):
        feed(campaign, world)


def test_failed_tree_lease_cancels_the_node_job_and_blocks_dispatch(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lease that fails three times leaves a recorded, cancelled node job that takes no dispatch."""
    campaign = make_campaign(generation_config_factory)
    attempts: list[str] = []

    def failing_lease(job_id: str) -> bool:
        attempts.append(job_id)
        return False

    monkeypatch.setattr(cluster, "lease_executing_tree", failing_lease)
    with pytest.raises(RuntimeError, match="could not be leased"):
        submit(campaign, world)
    assert attempts == ["7001"] * 3
    failed = load_manifest(campaign, world)
    (node_job,) = failed["node_jobs"]
    assert (node_job["status"], node_job["job_id"], node_job["lease"], node_job["cancel_published"]) == ("submitted", "7001", "failed", True)
    assert node_job["error"] is not None
    assert failed["slurm_job_ids"] == ["7001"]
    assert failed["submissions"] == []
    assert (node_directory(campaign, world, node_job) / "control" / node.CANCEL_MARKER).is_file()
    assert world.log("scancel.log") == ["7001"]


def test_pending_lease_after_a_crash_is_finished_on_the_next_feed(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The submitted record exists with a pending lease when the feeder dies; the next feed leases it before dispatching."""
    campaign = make_campaign(generation_config_factory)

    def dying_lease(_job_id: str) -> bool:
        raise Crash

    monkeypatch.setattr(cluster, "lease_executing_tree", dying_lease)
    with pytest.raises(Crash):
        submit(campaign, world)
    pending = load_manifest(campaign, world)
    assert [(job["status"], job["lease"]) for job in pending["node_jobs"]] == [("submitted", "pending")]
    assert pending["submissions"] == []

    monkeypatch.setattr(cluster, "lease_executing_tree", lambda job_id: world.leases.append(job_id) is None)
    leased = feed(campaign, world)
    assert [(job["status"], job["lease"]) for job in leased["node_jobs"]] == [("submitted", "ok")]
    assert world.leases == ["7001"]
    assert [item["status"] for item in leased["submissions"]] == ["dispatched"]
    assert len(world.sbatch_arguments()) == 1


def test_idle_node_job_drains_with_a_stop_ticket_that_a_crash_cannot_lose(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing left to dispatch and every step finished, the node job gets exactly one stop ticket after its last ticket."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    node_job = manifest["node_jobs"][0]
    directory = node_directory(campaign, world, node_job)
    held = feed(campaign, world)["node_jobs"][0]
    assert (held["draining"], held["stop_ticket_seq"]) == (False, None)  # the dispatched step has no step accounting yet

    world.add_step("7001", 0, "sf-lentil-c0001-s1")
    crash_at(
        monkeypatch,
        generation.campaign.node_service,
        "ensure_node_ticket",
        after=False,
        when=lambda _directory, ticket: ticket.kind == node.STOP_KIND,
    )
    with pytest.raises(Crash):
        feed(campaign, world)
    interrupted = load_manifest(campaign, world)["node_jobs"][0]
    assert (interrupted["draining"], interrupted["stop_ticket_seq"], interrupted["next_ticket_seq"]) == (True, None, 2)

    drained = feed(campaign, world)["node_jobs"][0]
    assert (drained["draining"], drained["stop_ticket_seq"], drained["next_ticket_seq"]) == (True, 2, 3)
    stop = json.loads(node.ticket_path(directory, 2).read_text(encoding="utf-8"))
    assert stop["kind"] == "stop"
    assert sorted(path.name for path in (directory / "tickets").iterdir()) == ["000001.json", "000002.json"]

    # A draining node job takes no further dispatch and is not drained twice.
    assert feed(campaign, world)["node_jobs"][0] == drained


def test_settled_campaign_drains_every_open_node_job_even_with_running_steps(generation_config_factory: Any, world: FakeSlurm) -> None:
    """A settled campaign ends its node jobs after the running steps: the stop ticket follows the last dispatched ticket."""
    campaign = make_campaign(generation_config_factory, natural_count=2)
    manifest = submit(campaign, world)
    scheduler = {"allocations": {"active": {"7001": ["7001", "RUNNING"]}, "accounted": {}}}
    settled = generation.campaign._settle_campaign(manifest, "complete", scheduler, storage_root=world.storage)
    node_job = settled["node_jobs"][0]
    assert (settled["state"], node_job["draining"], node_job["stop_ticket_seq"], node_job["next_ticket_seq"]) == ("complete", True, 3, 4)
    assert campaign_evidence.load_campaign_run(settled["campaign_run_id"], storage_root=world.storage)["node_jobs"][0]["stop_ticket_seq"] == 3


def test_plan_reports_node_job_resources_without_mutation(generation_config_factory: Any, world: FakeSlurm) -> None:
    """The plan names the node-job and case-step resources instead of a per-case submission command."""
    campaign = make_campaign(generation_config_factory, cores_per_node=4)
    world.storage.mkdir()
    plan = generation.campaign.plan_campaign(campaign, git_commit=COMMIT, storage_root=world.storage)
    cluster_values = campaign.execution_values["cluster"]
    assert plan["filesystem_mutated"] is False
    assert "first_submission_command" not in plan
    assert plan["node_job"]["seats"] == [1, 2]
    assert plan["node_job"]["cores_per_node"] == 4
    assert plan["node_job"]["node_job_memory"] == cluster_values["node_job_memory"]
    assert plan["node_job"]["concurrent_cases_per_node"] == 4
    assert plan["node_job"]["unresolved_dispatches_per_node_job"] == 8
    assert plan["case_step"]["cores"] == 1
    assert plan["planned_cases"] == 1
    assert not Path(plan["paths"]["run_root"]).exists()
    assert world.sbatch_arguments() == []


def _valid_run(run_id: str) -> dict[str, Any]:
    """Return a minimal consistent schema-2 run record with one submitted node job and one dispatched case step."""
    return {
        "schema_kind": "generation_campaign_run",
        "schema_version": 2,
        "campaign_run_id": run_id,
        "slurm_job_ids": ["7001"],
        "node_job_intent": None,
        "submission_intent": None,
        "node_jobs": [
            {
                "token": NODE_ID,
                "seat": 1,
                "node_directory": f"_record/execution/node_jobs/{NODE_ID}",
                "job_id": "7001",
                "status": "submitted",
                "lease": "ok",
                "draining": False,
                "stop_ticket_seq": None,
                "next_ticket_seq": 2,
                "cancel_published": False,
                "recorded_at": "2026-10-27T10:00:00+00:00",
                "terminal_observed_at": None,
                "error": None,
            }
        ],
        "submissions": [
            {
                "submission_index": 1,
                "mode": "initial",
                "recorded_at": "2026-10-27T10:00:01+00:00",
                "case": {"batch_name": "batch", "batch_id": "batch", "case_index": 1, "case_id": "case_0001"},
                "node_job_token": NODE_ID,
                "job_id": "7001",
                "ticket_seq": 1,
                "step_name": "sf-lentil-c0001-s1",
                "attempt_index": 1,
                "status": "dispatched",
                "error": None,
            }
        ],
    }


def _second_dispatch(run: dict[str, Any], **values: Any) -> None:
    """Append a second dispatch record to a valid run and move the cursor past it."""
    record = {**run["submissions"][0], "submission_index": 2, "ticket_seq": 2, "step_name": "sf-lentil-c0001-s2", **values}
    run["submissions"].append(record)
    run["node_jobs"][0]["next_ticket_seq"] = 3


_MUTATIONS: dict[str, tuple[Callable[[dict[str, Any]], object], str]] = {
    "unknown_node_job_token": (lambda run: run["submissions"][0].update(node_job_token="f" * 16), "references no submitted node job"),
    "allocation_differs_from_its_node_job": (lambda run: run["submissions"][0].update(job_id="7002"), "references no submitted node job"),
    "duplicate_ticket": (lambda run: _second_dispatch(run, ticket_seq=1), "ticket is duplicated"),
    "dispatched_beyond_the_cursor": (lambda run: run["node_jobs"][0].update(next_ticket_seq=1), "ticket cursor"),
    "step_name_of_another_case": (lambda run: run["submissions"][0].update(step_name="sf-lentil-c0002-s1"), "dispatch record 1 is malformed"),
    "step_name_too_long": (lambda run: run["submissions"][0].update(step_name="sf-" + "x" * 50 + "-c0001-s1"), "dispatch record 1 is malformed"),
    "not_started_without_error": (lambda run: run["submissions"][0].update(status="not_started"), "dispatch record 1 is malformed"),
    "dispatching_without_intent": (
        lambda run: (run["submissions"][0].update(status="dispatching", ticket_seq=2), run["node_jobs"][0].update(next_ticket_seq=2)),
        "submission intent is inconsistent",
    ),
    "intent_without_dispatching": (lambda run: run.update(submission_intent=1), "submission intent is inconsistent"),
    "job_ids_disagree_with_node_jobs": (lambda run: run.update(slurm_job_ids=["7001", "7002"]), "disagree with node-job records"),
    "node_job_token_malformed": (lambda run: run["node_jobs"][0].__setitem__("token", "NOT-A-TOKEN"), "node-job record 1 is malformed"),
    "node_job_seat_unknown": (lambda run: run["node_jobs"][0].update(seat=5), "node-job record 1 is malformed"),
    "node_directory_of_another_token": (
        lambda run: run["node_jobs"][0].update(node_directory="_record/execution/node_jobs/" + "f" * 16),
        "node-job record 1 is malformed",
    ),
    "stop_ticket_without_draining": (lambda run: run["node_jobs"][0].update(stop_ticket_seq=1), "node-job record 1 is inconsistent"),
    "node_job_intent_names_nothing": (lambda run: run.update(node_job_intent=NODE_ID), "node-job intent is inconsistent"),
    "predates_node_mode": (lambda run: run.update(schema_version=1), "predates node mode"),
}


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_scheduler_ownership_reader_fails_closed_on_every_broken_node_job_or_dispatch_rule(world: FakeSlurm, name: str) -> None:
    """The light reader accepts the valid shape and rejects each broken equality, reference, uniqueness, or version rule."""
    run_id = "source~steady_flow~lentil-1__ref-range__lhs-seed-41001"
    path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=world.storage)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_valid_run(run_id)), encoding="utf-8")
    assert campaign_evidence.load_campaign_scheduler_job_ids(run_id, storage_root=world.storage) == ("7001",)

    run = _valid_run(run_id)
    mutate, message = _MUTATIONS[name]
    mutate(run)
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        campaign_evidence.load_campaign_scheduler_job_ids(run_id, storage_root=world.storage)


def test_scheduler_ownership_reader_refuses_while_a_node_job_submission_is_unresolved(world: FakeSlurm) -> None:
    """An unresolved node-job intent may hide an allocation, so the light reader refuses instead of answering partially."""
    run_id = "source~steady_flow~lentil-1__ref-range__lhs-seed-41001"
    run = _valid_run(run_id)
    submitting = {**run["node_jobs"][0], "token": "f" * 16, "node_directory": "_record/execution/node_jobs/" + "f" * 16}
    submitting.update(seat=2, job_id=None, status="submitting", lease="pending", next_ticket_seq=1)
    run["node_jobs"].append(submitting)
    run["node_job_intent"] = "f" * 16
    path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=world.storage)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(RuntimeError, match="node-job submission is unresolved"):
        campaign_evidence.load_campaign_scheduler_job_ids(run_id, storage_root=world.storage)


def test_run_manifest_of_schema_one_is_refused_for_execution_resume_admission_and_tree_restore(
    generation_config_factory: Any, world: FakeSlurm
) -> None:
    """A run.json written before node mode is neither continued, resumed, read for ownership, nor used to restore a source tree."""
    campaign = make_campaign(generation_config_factory)
    world.add_foreign_job("gen-seat-1", "6001")
    world.add_foreign_job("gen-seat-2", "6002")
    created = submit(campaign, world)
    run_id = created["campaign_run_id"]
    path = campaign_evidence.campaign_run_manifest_path(run_id, storage_root=world.storage)
    legacy = {key: value for key, value in created.items() if key not in {"node_jobs", "node_job_intent"}}
    legacy.update(schema_version=1, scheduler_job_name="sf-campaign-abcd")
    path.write_text(json.dumps(legacy), encoding="utf-8")

    refusals: list[Callable[[], Any]] = [
        lambda: campaign_evidence.load_campaign_run(run_id, storage_root=world.storage),
        lambda: campaign_evidence.load_campaign_scheduler_job_ids(run_id, storage_root=world.storage),
        lambda: generation.campaign.feed_campaign(run_id, storage_root=world.storage),
        lambda: generation.campaign.resume_campaign(run_id, storage_root=world.storage),
        lambda: submit(campaign, world),
        lambda: generation.campaign.campaign_status(run_id, storage_root=world.storage),
        lambda: generation.campaign.bound_source_snapshot(run_id, storage_root=world.storage),
    ]
    for refuse in refusals:
        with pytest.raises(ValueError, match="predates node mode"):
            refuse()
    assert world.sbatch_arguments() == []


def _age_node_job(campaign: Any, world: FakeSlurm, index: int = 0, *, seconds: int = 300) -> None:
    """Move the first observation of one ended node job into the past, as the passing of the not-started grace would."""
    path = campaign_evidence.campaign_run_manifest_path(run_id_of(campaign), storage_root=world.storage)
    run = json.loads(path.read_text(encoding="utf-8"))
    assert run["node_jobs"][index]["terminal_observed_at"] is not None
    run["node_jobs"][index]["terminal_observed_at"] = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
    path.write_text(json.dumps(run), encoding="utf-8")


def _finished_receipt(directory: Path, seq: int, *, seconds_ago: int, outcome: str = "exited", exit_code: int = 0) -> None:
    """Write the finished receipt a node worker publishes for one ticket, ended ``seconds_ago`` seconds ago."""
    ended = datetime.now(UTC) - timedelta(seconds=seconds_ago)
    receipt = {
        "schema": node.FINISHED_RECEIPT_SCHEMA,
        "schema_version": node.SCHEMA_VERSION,
        "seq": seq,
        "started_at": (ended - timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at": ended.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exit_code": exit_code,
        "host": "hpc1",
        "outcome": outcome,
    }
    node.receipt_path(directory, seq, "finished").write_text(json.dumps(receipt), encoding="utf-8")


def _status_case(campaign: Any, world: FakeSlurm, position: int = 0) -> dict[str, Any]:
    """Return one public case view of the read-only campaign status."""
    return generation.campaign.campaign_status(run_id_of(campaign), storage_root=world.storage)["cases"][position]


@pytest.mark.parametrize(
    ("step_state", "exit_code", "classified", "reason"),
    [
        ("COMPLETED", "0:0", "failed", "completed_without_valid_case_evidence"),
        ("OUT_OF_MEMORY", "0:125", "failed", "scheduler_out_of_memory"),
        ("FAILED", "1:0", "failed", "scheduler_failed"),
        ("TIMEOUT", "0:0", "timed_out", "scheduler_timeout"),
        ("CANCELLED", "0:15", "cancelled", "scheduler_cancelled"),
        ("NODE_FAIL", "0:0", "interrupted", "scheduler_node_fail"),
        ("RUNNING", "0:0", "active", "RUNNING"),
    ],
)
def test_step_accounting_not_the_allocation_decides_the_state_of_each_dispatch(
    generation_config_factory: Any, world: FakeSlurm, step_state: str, exit_code: str, classified: str, reason: str
) -> None:
    """The step's own state and times classify its case while the node job keeps running; an interrupted step restarts as a new dispatch."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    world.set_state("7001", "RUNNING")
    pending = _status_case(campaign, world)
    assert (pending["classified_state"], pending["scheduler_state"], pending["latest_job_id"]) == ("pending", "PENDING", "7001-1")
    world.add_step("7001", 0, manifest["submissions"][0]["step_name"], step_state, exit_code, node="hpc7")

    case = _status_case(campaign, world)
    assert (case["classified_state"], case["reason"], case["latest_job_id"]) == (classified, reason, "7001-1")
    assert case["scheduler_state"] == step_state
    assert case["node"] == "hpc7"
    after = feed(campaign, world)
    if classified in {"cancelled", "interrupted"}:
        restart = after["submissions"][1]
        assert (restart["mode"], restart["attempt_index"], restart["ticket_seq"], restart["status"]) == ("resume", 2, 2, "dispatched")
    else:
        assert len(after["submissions"]) == 1
    if reason == "scheduler_out_of_memory":
        assert generation.campaign._failure_population_counts([_status_case(campaign, world)])["solver_failed"] == 1


def test_finished_receipt_without_step_accounting_proves_a_launch_failure_only_after_the_grace(
    generation_config_factory: Any, world: FakeSlurm
) -> None:
    """A step that never got a Slurm step restarts without consuming an attempt once its receipt is older than the grace."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    directory = node_directory(campaign, world, manifest["node_jobs"][0])
    _finished_receipt(directory, 1, seconds_ago=30, outcome="launch_failed", exit_code=-1)

    recent = feed(campaign, world)
    assert [item["status"] for item in recent["submissions"]] == ["dispatched"]
    assert _status_case(campaign, world)["classified_state"] == "pending"

    _finished_receipt(directory, 1, seconds_ago=300, outcome="launch_failed", exit_code=-1)
    later = feed(campaign, world)
    first, retry = later["submissions"]
    assert first["status"] == "not_started"
    assert "no step accounting" in first["error"]
    assert (retry["status"], retry["mode"], retry["attempt_index"], retry["job_id"], retry["ticket_seq"]) == ("dispatched", "restart", 1, "7001", 2)
    assert later["node_jobs"][0]["next_ticket_seq"] == 3


def test_node_job_that_dies_with_started_and_unstarted_tickets_restarts_both_cases_with_the_right_attempts(
    generation_config_factory: Any, world: FakeSlurm
) -> None:
    """The started step is interrupted and counts an attempt; the step that never started restarts without one."""
    campaign = make_campaign(generation_config_factory, natural_count=2)
    manifest = submit(campaign, world)
    started, unstarted = manifest["submissions"]
    world.add_step("7001", 0, started["step_name"], "NODE_FAIL")
    world.set_state("7001", "NODE_FAIL")

    waiting = feed(campaign, world)
    assert waiting["state"] == "scheduler_unknown"  # the ended node job is observed, but the missing step row is not yet proof
    assert [item["status"] for item in waiting["submissions"]] == ["dispatched", "dispatched"]
    assert waiting["node_jobs"][0]["terminal_observed_at"] is not None

    _age_node_job(campaign, world)
    recovered = feed(campaign, world)
    assert [item["status"] for item in recovered["submissions"][:2]] == ["dispatched", "not_started"]
    assert recovered["submissions"][1]["error"] == "the node job ended before the step started"
    interrupted_case, restarted_case = recovered["submissions"][2:]
    assert (interrupted_case["case"], interrupted_case["mode"], interrupted_case["attempt_index"]) == (started["case"], "resume", 2)
    assert (restarted_case["case"], restarted_case["mode"], restarted_case["attempt_index"]) == (unstarted["case"], "restart", 1)
    assert {item["job_id"] for item in recovered["submissions"][2:]} == {"7002"}
    assert recovered["slurm_job_ids"] == ["7001", "7002"]


def test_attempt_of_an_older_dispatch_never_overrides_the_state_of_the_latest_started_dispatch(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only attempt evidence written by the case's latest started dispatch is admitted."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    case = manifest["submissions"][0]["case"]
    task = next(item for item in cluster.campaign_tasks(campaign) if item.case_index == case["case_index"])
    batch = campaign.batch(task.batch_name)
    storage = world.storage
    reference = generation.cases.input_generation.admit_configured_input_references(
        batch, storage_root=storage, validation_depth="evidence", git_commit=COMMIT
    )[task.case_index]
    older = SimpleNamespace(
        payload={
            "job_id": "7001",
            "submission_index": 1,
            "campaign_run_id": manifest["campaign_run_id"],
            "solver_git_commit": COMMIT,
            "input_generation_id": reference.source_id,
            "template": {"relative_path": batch.template_relative_path, "sha256": batch.template_sha256},
        },
        receipt_path=Path("attempt_0001/attempt.json"),
    )
    monkeypatch.setattr(generation.campaign.attempt_service, "latest_case_attempt", lambda *_a, **_k: older)
    monkeypatch.setattr(generation.campaign.input_service, "admit_persisted_input_case", lambda *_a, **_k: reference)
    expected = {
        "campaign_purpose": batch.scientific_values["campaign_purpose"],
        "batch_storage_name": batch.batch_storage_name,
        "batch_id": batch.batch_id,
        "batch_identity": batch.batch_identity,
        "input_generation_id": reference.source_id,
        "case_id": task.case_id,
        "case_index": task.case_index,
        "case_input_id": reference.case_input_id,
        "simulation_case_id": reference.simulation_case_id,
        "canonical_raw_case": reference.case_directory.resolve().relative_to(storage.resolve()).as_posix(),
        "scientific_config_digest": batch.scientific_config_digest,
        "export_contract_sha256": common.serialization.canonical_json_sha256(batch.scientific_values["output_contract"]),
    }
    older.payload.update(expected)

    admitted = generation.campaign._admitted_case_attempt(manifest, batch, task, input_reference=reference, storage_root=storage)
    assert admitted is not None
    assert admitted.current is True  # written by the only dispatch

    newer = {**manifest["submissions"][0], "submission_index": 2, "ticket_seq": 2, "step_name": "sf-lentil-c0001-s2", "attempt_index": 2}
    stale = generation.campaign._admitted_case_attempt(
        {**manifest, "submissions": [*manifest["submissions"], newer]}, batch, task, input_reference=reference, storage_root=storage
    )
    assert stale is not None
    assert stale.current is False

    older.payload["submission_index"] = 9
    with pytest.raises(RuntimeError, match="Attempt evidence disagrees"):
        generation.campaign._admitted_case_attempt(manifest, batch, task, input_reference=reference, storage_root=storage)


def _licence_views(monkeypatch: pytest.MonkeyPatch, *, exhausted: bool) -> None:
    """Make every reconciled case a licence-blocked case whose backoff is due (or whose wait budget is used up)."""
    real = generation.campaign._reconciled
    due = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()

    def reconciled(*args: Any, **kwargs: Any) -> Any:
        views, pending, running = real(*args, **kwargs)
        for view in views:
            view.update(
                state="license_blocked",
                license_retry_eligible=not exhausted,
                license_wait_exhausted=exhausted,
                license_first_blocked_at=due,
                license_next_retry_at=due,
            )
        return views, pending, running

    monkeypatch.setattr(generation.campaign, "_reconciled", reconciled)


def test_licence_blocked_case_is_redispatched_as_a_new_ticket_after_its_backoff(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A due licence retry becomes a license_retry dispatch with the next attempt in the node job that is still alive."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    world.add_step("7001", 0, manifest["submissions"][0]["step_name"], "COMPLETED")
    _licence_views(monkeypatch, exhausted=False)

    after = feed(campaign, world)
    first, retry = after["submissions"]
    assert first["status"] == "dispatched"
    assert (retry["mode"], retry["attempt_index"], retry["job_id"], retry["ticket_seq"]) == ("license_retry", 2, "7001", 2)
    assert retry["case"] == first["case"]
    assert retry["step_name"] != first["step_name"]
    assert after["state"] == "active"
    ticket = json.loads(node.ticket_path(node_directory(campaign, world, after["node_jobs"][0]), 2).read_text(encoding="utf-8"))
    assert ticket["env"]["GENERATION_DISPATCH_KEY"] == "7001-2"
    assert ticket["env"]["GENERATION_ATTEMPT_INDEX"] == "2"


def test_exhausted_licence_wait_budget_ends_the_campaign_in_its_own_state_and_drains_the_node_jobs(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a case has used its whole licence wait budget the campaign stops feeding and its node jobs end after the running steps."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    world.add_step("7001", 0, manifest["submissions"][0]["step_name"], "COMPLETED")
    _licence_views(monkeypatch, exhausted=True)

    exhausted = feed(campaign, world)
    assert exhausted["state"] == "license_wait_exhausted"
    assert len(exhausted["submissions"]) == 1
    assert (exhausted["node_jobs"][0]["draining"], exhausted["node_jobs"][0]["stop_ticket_seq"]) == (True, 2)
    assert campaign_evidence.load_campaign_run(exhausted["campaign_run_id"], storage_root=world.storage)["state"] == "license_wait_exhausted"


def test_cancel_marks_every_open_node_job_and_signals_only_the_scheduler_states_it_needs(generation_config_factory: Any, world: FakeSlurm) -> None:
    """A queued allocation is cancelled outright, a running one gets TERM for its batch shell, and force kills the whole step tree."""
    campaign = make_campaign(generation_config_factory, natural_count=6)
    manifest = submit(campaign, world)
    first, second = manifest["node_jobs"]
    world.set_state(second["job_id"], "RUNNING")
    receipt = generation.campaign.cancel_campaign(run_id_of(campaign), storage_root=world.storage)

    assert world.log("scancel.log") == [first["job_id"], f"--signal=TERM --batch {second['job_id']}"]
    (attempt,) = receipt["attempts"]
    assert (attempt["mode"], attempt["pending_job_ids"], attempt["running_job_ids"]) == ("graceful", [first["job_id"]], [second["job_id"]])
    assert sorted(attempt["succeeded_job_ids"]) == sorted([first["job_id"], second["job_id"]])
    cancelled = load_manifest(campaign, world)
    assert cancelled["state"] == "cancel_requested"
    assert [job["cancel_published"] for job in cancelled["node_jobs"]] == [True, True]
    for job in cancelled["node_jobs"]:
        assert (node_directory(campaign, world, job) / "control" / node.CANCEL_MARKER).is_file()

    forced = generation.campaign.cancel_campaign(run_id_of(campaign), storage_root=world.storage, force=True)
    assert world.log("scancel.log")[-1] == f"--signal=KILL --full {first['job_id']} {second['job_id']}"
    assert forced["attempts"][-1]["mode"] == "force"
    assert load_manifest(campaign, world)["state"] == "force_cancel_requested"
    with pytest.raises(RuntimeError, match="all previous node jobs have terminal"):
        submit(campaign, world)


def test_cancellation_is_a_desired_state_that_every_feed_replays_after_a_crash(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt and the state come before any command; a crash during the commands is finished by the next feed, once."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    job = manifest["node_jobs"][0]
    crash_at(monkeypatch, generation.campaign.node_service, "publish_node_cancel", after=True)
    with pytest.raises(Crash):
        generation.campaign.cancel_campaign(run_id_of(campaign), storage_root=world.storage)
    interrupted = load_manifest(campaign, world)
    assert interrupted["state"] == "cancel_requested"
    assert interrupted["node_jobs"][0]["cancel_published"] is False
    assert (node_directory(campaign, world, job) / "control" / node.CANCEL_MARKER).is_file()
    assert world.log("scancel.log") == []
    receipt_path = common.paths.resolve_generation_cancellations_path(
        campaign_evidence.campaign_execution_directory(run_id_of(campaign), storage_root=world.storage)
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["attempts"][0]["commands"] == []

    replayed = feed(campaign, world)  # a feed of a cancelled campaign neither admits nor dispatches; it finishes the cancellation
    assert replayed["state"] == "cancel_requested"
    assert replayed["node_jobs"][0]["cancel_published"] is True
    assert len(replayed["submissions"]) == 1
    assert world.log("scancel.log") == [job["job_id"]]
    attempts = json.loads(receipt_path.read_text(encoding="utf-8"))["attempts"]
    assert [item["succeeded_job_ids"] for item in attempts] == [[], [job["job_id"]]]

    feed(campaign, world)
    generation.campaign.resume_campaign(run_id_of(campaign), storage_root=world.storage)
    assert world.log("scancel.log") == [job["job_id"]]  # nothing left to command, so nothing is repeated
    assert len(json.loads(receipt_path.read_text(encoding="utf-8"))["attempts"]) == 2


def test_case_worker_binds_to_its_dispatch_and_fails_closed_otherwise(
    generation_config_factory: Any, world: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step binds by submission index, allocation, case, and attempt; a record that is not yet visible is awaited, then refused."""
    campaign = make_campaign(generation_config_factory, natural_count=2)
    manifest = submit(campaign, world)
    case = manifest["submissions"][1]["case"]
    seen: list[Any] = []
    outcome = SimpleNamespace(status="interrupted")
    monkeypatch.setattr(source_service, "required_git_commit", lambda: COMMIT)
    monkeypatch.setattr(
        generation.campaign.cluster_service, "run_campaign_case", lambda _campaign, task, **_k: seen.append(task.case_index) or outcome
    )
    run = {"storage_root": world.storage, "work_root": tmp_path / "work"}

    bind_worker(monkeypatch, manifest, index=2)
    assert generation.campaign.run_campaign_case_job(manifest["campaign_run_id"], case["batch_name"], case["case_index"], **run) is outcome
    assert seen == [case["case_index"]]
    with pytest.raises(RuntimeError, match="not bound to this exact campaign case"):
        generation.campaign.run_campaign_case_job(
            manifest["campaign_run_id"], case["batch_name"], manifest["submissions"][0]["case"]["case_index"], **run
        )
    bind_worker(monkeypatch, manifest, index=2, attempt="2")
    with pytest.raises(RuntimeError, match="not bound to this exact campaign case"):
        generation.campaign.run_campaign_case_job(manifest["campaign_run_id"], case["batch_name"], case["case_index"], **run)
    monkeypatch.delenv("GENERATION_CAMPAIGN_RUN_ID")
    with pytest.raises(RuntimeError, match="dispatch environment"):
        generation.campaign.run_campaign_case_job(manifest["campaign_run_id"], case["batch_name"], case["case_index"], **run)

    # The record can reach the node late: it is read again until it shows, then the step runs; without it the step fails closed.
    bind_worker(monkeypatch, manifest, index=2)
    reads = {"count": 0}
    real_load = campaign_evidence.load_campaign_run

    def late(*args: Any, **kwargs: Any) -> dict[str, Any]:
        loaded = real_load(*args, **kwargs)
        reads["count"] += 1
        return {**loaded, "submissions": []} if reads["count"] <= 2 else loaded

    sleeps: list[float] = []
    monkeypatch.setattr(campaign_evidence, "load_campaign_run", late)
    monkeypatch.setattr(generation.campaign, "sleep", sleeps.append)
    seen.clear()
    generation.campaign.run_campaign_case_job(manifest["campaign_run_id"], case["batch_name"], case["case_index"], **run)
    assert (len(sleeps), seen) == (2, [case["case_index"]])

    clock = iter([0.0, 1.0, 500.0])
    monkeypatch.setattr(generation.campaign, "monotonic", lambda: next(clock))
    reads["count"] = -100
    with pytest.raises(RuntimeError, match="no live record"):
        generation.campaign.run_campaign_case_job(manifest["campaign_run_id"], case["batch_name"], case["case_index"], **run)


def _campaign_with_dispatched_replacement(generation_config_factory: Any, world: FakeSlurm) -> tuple[Any, dict[str, Any]]:
    """Fail the only original of a capacity-one campaign and let the feeder dispatch its replacement as submission 2."""
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow", scheduler_kind="slurm", natural_count=1, max_admission_cases=4, node_jobs=1, cores_per_node=2
    )
    authored = {**yaml.safe_load(config_path.read_text(encoding="utf-8")), "replacement_capacity": {"lentil": 1}}
    campaign = generation.cases.config.load_campaign_config(config_path, authored_value=authored)
    first = submit(campaign, world)
    run_id = first["campaign_run_id"]
    lifecycle.record_candidate_attempt(
        "steady_flow",
        campaign.source_name,
        "lentil",
        "case_0001",
        attempt_index=1,
        stage="solver",
        failure_class="solver",
        reason_code=lifecycle.UNCLASSIFIED_SOLVER_REASON,
        retryable=False,
        detail="Synthetic non-converging solve",
        evidence={},
        storage_root=world.storage,
    )
    world.finish_dispatches(first, "FAILED", "1:0")
    generation.campaign.declare_candidate_failure(
        run_id, "lentil", "case_0001", reason_code="synthetic_nonconvergence", detail="Synthetic non-converging solve", storage_root=world.storage
    )
    fed = feed(campaign, world)
    case = fed["submissions"][-1]["case"]
    assert (case["case_id"], fed["submissions"][-1]["submission_index"]) == ("case_0002", 2)
    assert case["batch_name"] not in {batch["batch_name"] for batch in fed["batches"]}
    return campaign, fed


def test_dispatched_replacement_binds_its_runtime_progress_to_the_frozen_execution_batches(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement runs in its own frozen one-case batch outside the run record's batch list; only frozen batches bind."""
    _campaign, fed = _campaign_with_dispatched_replacement(generation_config_factory, world)
    run_id = fed["campaign_run_id"]
    case = fed["submissions"][-1]["case"]

    bind_worker(monkeypatch, fed, index=2, step="1")
    dispatch = dispatch_identity.current_case_dispatch()
    assert dispatch is not None
    reporter = progress.RuntimeProgressReporter.create(run_id, **case, dispatch=dispatch, storage_root=world.storage, hostname="node-a")
    assert reporter.update(phase="preparing", force=True)
    view = next(item for item in generation.campaign.campaign_status(run_id, storage_root=world.storage)["cases"] if item["case_id"] == "case_0002")
    assert view["runtime_progress"]["availability"] == "available"

    original = fed["batches"][0]
    for foreign in (
        {**case, "batch_id": original["batch_id"]},
        {**case, "batch_name": original["batch_name"]},
        {**case, "batch_id": f"{case['batch_name']}__0000000000000000"},
    ):
        with pytest.raises(progress.RuntimeProgressIdentityError, match="batch identity is not bound"):
            progress.RuntimeProgressReporter.create(run_id, **foreign, dispatch=dispatch, storage_root=world.storage, hostname="node-a")


def test_explicit_rerun_requeues_a_replacement_whose_dispatch_ended_with_a_retryable_error(generation_config_factory: Any, world: FakeSlurm) -> None:
    """The explicit path hands the feeder references of the authored batches only; the replacement's batch is admitted from the frozen run."""
    campaign, fed = _campaign_with_dispatched_replacement(generation_config_factory, world)
    lifecycle.record_candidate_attempt(
        "steady_flow",
        campaign.source_name,
        "lentil",
        "case_0002",
        attempt_index=1,
        stage="publication",
        failure_class="publication",
        reason_code="synthetic_publication_error",
        retryable=True,
        detail="Synthetic worker-side publication error",
        evidence={"dispatch_key": "7001-2"},
        storage_root=world.storage,
    )
    world.finish_dispatches(fed, "FAILED", "1:0")

    rerun = submit(campaign, world)
    assert [(item["case"]["case_id"], item["attempt_index"]) for item in rerun["submissions"][1:]] == [("case_0002", 1), ("case_0002", 2)]
    assert rerun["submissions"][-1]["case"] == fed["submissions"][-1]["case"]


def test_worker_interruption_receipts_are_keyed_by_dispatch_so_steps_of_one_allocation_do_not_collide(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each step of one node job records its own interruption evidence."""
    campaign = make_campaign(generation_config_factory, natural_count=2)
    manifest = submit(campaign, world)
    paths = []
    for index, step in ((1, "0"), (2, "1")):
        bind_worker(monkeypatch, manifest, index=index, step=step)
        paths.append(
            generation.campaign.record_worker_interruption(manifest["campaign_run_id"], storage_root=world.storage, signal_name="TERM", exit_code=143)
        )
    assert [path.name for path in paths] == ["7001-1.json", "7001-2.json"]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert [(item["dispatch_key"], item["slurm_job_id"], item["slurm_step_id"], item["submission_index"]) for item in records] == [
        ("7001-1", "7001", "0", 1),
        ("7001-2", "7001", "1", 2),
    ]
    monkeypatch.delenv("GENERATION_CAMPAIGN_RUN_ID")
    with pytest.raises(ValueError, match="dispatch environment"):
        generation.campaign.record_worker_interruption(manifest["campaign_run_id"], storage_root=world.storage, signal_name="TERM", exit_code=143)


def test_terminal_timestamp_evidence_is_keyed_by_the_dispatch(generation_config_factory: Any, world: FakeSlurm) -> None:
    """The Slurm End projection of a step is stored and read back under its dispatch key, never under an allocation id."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    task = cluster.campaign_tasks(campaign)[0]
    projection = generation.campaign._terminal_timestamp_projection("2026-10-27T10:30:00+00:00", source_timezone=None)
    path = generation.campaign._record_terminal_timestamp_evidence(manifest, task, "7001-1", projection, storage_root=world.storage)
    assert path is not None
    assert path.name == "7001-1.terminal_timestamp.json"
    stored = generation.campaign._load_terminal_timestamp_evidence(manifest, task, "7001-1", storage_root=world.storage)
    assert stored is not None
    assert stored["dispatch_key"] == "7001-1"
    assert "slurm_job_id" not in stored
    with pytest.raises(ValueError, match="Dispatch key"):
        generation.campaign._terminal_timestamp_evidence_path(manifest, task, "7001", storage_root=world.storage)


def test_operational_failure_is_bound_to_the_dispatch_that_raised_it(generation_config_factory: Any) -> None:
    """A retryable controller error blocks automatic feeding only while its own dispatch is the case's latest and has scheduler evidence."""
    campaign = make_campaign(generation_config_factory)
    batch = campaign.batches[0]

    def records(evidence: dict[str, Any]) -> list[dict[str, Any]]:
        attempt = {"class": "controller", "retryable": True, "evidence": evidence}
        return [{"attempts": [attempt], "decision": None, "reservation": {"batch_id": batch.batch_id, "origin": "primary", "case_id": "case_0001"}}]

    view = {"batch_id": batch.batch_id, "case_id": "case_0001", "latest_job_id": "7001-2", "state": "failed"}
    scheduler = {"active": {}, "accounted": {"7001-2": ["7001-2", "COMPLETED"]}}
    failures = generation.campaign._unexpected_operational_failure_views(campaign, records({"dispatch_key": "7001-2"}), [view], scheduler)
    assert failures == (view,)
    assert generation.campaign._unexpected_operational_failure_views(campaign, records({"dispatch_key": "7001-1"}), [view], scheduler) == ()
    with pytest.raises(RuntimeError, match="dispatch identity"):
        generation.campaign._unexpected_operational_failure_views(campaign, records({"slurm_job_id": "7001"}), [view], scheduler)


def test_status_lists_every_node_job_and_counts_dispatches(generation_config_factory: Any, world: FakeSlurm) -> None:
    """The public status names each node job with its seat, queue state, node, draining flag, and unresolved dispatches."""
    campaign = make_campaign(generation_config_factory, natural_count=6)
    submit(campaign, world)
    world.set_state("7002", "RUNNING", node="hpc9")
    status = generation.campaign.campaign_status(run_id_of(campaign), storage_root=world.storage)

    first, second = status["node_jobs"]
    assert (first["seat"], first["job_id"], first["state"], first["unresolved_dispatches"], first["open"]) == (1, "7001", "PENDING", 4, True)
    assert (second["seat"], second["job_id"], second["state"], second["node"], second["unresolved_dispatches"]) == (2, "7002", "RUNNING", "hpc9", 2)
    assert (status["open_node_jobs"], status["pending_jobs"], status["running_jobs"], status["node_job_intent"]) == (2, 6, 0, None)
    text = status_formatting.format_campaign_status_summary(status)
    assert "Node job: seat=1  job=7001  state=PENDING  unresolved=4" in text
    assert "Node job: seat=2  job=7002  state=RUNNING  node=hpc9  unresolved=2" in text


def test_ended_node_job_is_listed_as_not_open(generation_config_factory: Any, world: FakeSlurm) -> None:
    """Open node jobs gate the controller; an ended job is listed with its terminal state."""
    campaign = make_campaign(generation_config_factory)
    submit(campaign, world)
    world.set_state("7001", "COMPLETED")
    status = generation.campaign.campaign_status(run_id_of(campaign), storage_root=world.storage)
    assert (status["node_jobs"][0]["state"], status["node_jobs"][0]["open"], status["open_node_jobs"]) == ("COMPLETED", False, 0)


# -- review regressions --


def test_an_executed_dispatch_without_step_accounting_is_interrupted_not_unstarted(generation_config_factory: Any, world: FakeSlurm) -> None:
    """A case that left progress evidence never becomes not_started when its step row is missing; it consumed its attempt."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    record = manifest["submissions"][0]
    execution = campaign_evidence.campaign_execution_directory(run_id_of(campaign), storage_root=world.storage)
    progress = common.paths.resolve_generation_slurm_directory(execution) / f"{record['job_id']}-{record['submission_index']}.json"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("{}\n", encoding="utf-8")
    _finished_receipt(node_directory(campaign, world, manifest["node_jobs"][0]), 1, seconds_ago=300, exit_code=2)

    later = feed(campaign, world)
    first, retry = later["submissions"]
    assert first["status"] == "dispatched", "accounting lag must never turn an executed step into a step that never started"
    assert (retry["case"], retry["attempt_index"]) == (first["case"], 2)


def test_recovering_an_intent_keeps_a_force_cancellation(generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch) -> None:
    """A node job found by its submission token after a force cancellation is killed, and the state stays force-cancelled."""
    campaign = make_campaign(generation_config_factory)
    original = cluster.submit_node_job
    crash_at(monkeypatch, cluster, "submit_node_job", after=True)
    with pytest.raises(Crash):
        submit(campaign, world)
    monkeypatch.setattr(cluster, "submit_node_job", original)
    world_state = load_manifest(campaign, world)
    assert world_state["node_job_intent"] is not None

    generation.campaign.cancel_campaign(run_id_of(campaign), storage_root=world.storage, force=True)
    cancelled = load_manifest(campaign, world)
    assert (cancelled["state"], cancelled["node_job_intent"], cancelled["node_jobs"][0]["job_id"]) == ("force_cancel_requested", None, "7001")
    assert world.log("scancel.log")[-1] == "--signal=KILL --full 7001"


def test_a_failed_lease_stop_is_replayed_until_the_node_job_ended(
    generation_config_factory: Any, world: FakeSlurm, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after persisting the failed lease leaves no unleased allocation behind: every feed repeats marker and scancel."""
    campaign = make_campaign(generation_config_factory)
    original = generation.campaign._stop_unleased_node_job
    monkeypatch.setattr(cluster, "lease_executing_tree", lambda _job_id: False)
    crash_at(monkeypatch, generation.campaign, "_stop_unleased_node_job", after=False)
    with pytest.raises(Crash):
        submit(campaign, world)
    stranded = load_manifest(campaign, world)
    assert (stranded["node_jobs"][0]["lease"], stranded["node_jobs"][0]["cancel_published"]) == ("failed", False)
    assert world.log("scancel.log") == []

    monkeypatch.setattr(generation.campaign, "_stop_unleased_node_job", original)
    monkeypatch.setattr(cluster, "lease_executing_tree", lambda _job_id: True)
    feed(campaign, world)
    replayed = load_manifest(campaign, world)
    assert replayed["node_jobs"][0]["cancel_published"] is True
    assert world.log("scancel.log") == ["7001"]
    world.set_state("7001", "CANCELLED")
    feed(campaign, world)
    feed(campaign, world)
    assert world.log("scancel.log") == ["7001"], "an ended node job is not signalled again"


def test_an_observed_node_job_end_survives_accounting_retention(generation_config_factory: Any, world: FakeSlurm) -> None:
    """Once the feeder recorded a node job's end, a vanished accounting row never reopens it."""
    campaign = make_campaign(generation_config_factory)
    submit(campaign, world)
    world.set_state("7001", "COMPLETED")
    feed(campaign, world)
    assert load_manifest(campaign, world)["node_jobs"][0]["terminal_observed_at"] is not None
    (world.directory / "jobs").write_text("", encoding="utf-8")  # retention dropped the allocation from squeue and sacct
    status = generation.campaign.campaign_status(run_id_of(campaign), storage_root=world.storage)
    assert (status["node_jobs"][0]["open"], status["open_node_jobs"]) == (False, 0)


def _publish_valid_case(monkeypatch: pytest.MonkeyPatch, campaign: Any, manifest: dict[str, Any], storage: Path) -> None:
    """Make the only case a valid canonical publication: its HDF5 exists and the batch runtime admits and validates it."""
    task = cluster.campaign_tasks(campaign)[0]
    batch = campaign.batch(task.batch_name)
    source = generation.campaign._persisted_input_sources(manifest)[task.batch_name]
    reference = generation.cases.input_generation.admit_persisted_input_case(
        batch, task.case_index, source["input_generation_id"], storage_root=storage, validation_depth="evidence"
    )
    processed = generation.runtime.processed_case_directory(batch, task.case_index, storage_root=storage)
    artifact = processed / common.paths.GENERATION_RESULT_HDF5_FILENAME
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"synthetic canonical case")
    provenance = {
        "input_generation_id": reference.source_id,
        "case_input_id": reference.case_input_id,
        "simulation_case_id": reference.simulation_case_id,
        "artifacts": {artifact.name: {"sha256": common.serialization.file_sha256(artifact)}},
        "validation": None,
    }
    runtime = generation.campaign.batch_runtime
    monkeypatch.setattr(runtime, "completed_case_is_valid", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runtime, "validate_completed_case", lambda *_args, **_kwargs: provenance)


def _run_case_step(monkeypatch: pytest.MonkeyPatch, manifest: dict[str, Any], world: FakeSlurm, tmp_path: Path, runner: Callable[..., Any]) -> Any:
    """Run the case step of the first dispatch with ``runner`` as its case runtime, then leave the step environment."""
    case = manifest["submissions"][0]["case"]
    monkeypatch.setattr(source_service, "required_git_commit", lambda: COMMIT)
    monkeypatch.setattr(generation.campaign.cluster_service, "run_campaign_case", runner)
    bind_worker(monkeypatch, manifest)
    try:
        return generation.campaign.run_campaign_case_job(
            manifest["campaign_run_id"], case["batch_name"], case["case_index"], storage_root=world.storage, work_root=tmp_path / "work"
        )
    finally:
        monkeypatch.delenv("GENERATION_CAMPAIGN_RUN_ID")


def _only_candidate(campaign: Any, world: FakeSlurm) -> dict[str, Any]:
    """Return the record of the campaign's only candidate."""
    (record,) = lifecycle.candidate_records(campaign.profile.id, campaign.source_name, storage_root=world.storage)
    return record


def _case_with_cleanup_error(
    generation_config_factory: Any, world: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict[str, Any]]:
    """Submit the only case and let its step raise a scratch-cleanup error after a valid publication; return the campaign and its run record."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    _publish_valid_case(monkeypatch, campaign, manifest, world.storage)

    def cleanup_fails(*_args: Any, **_kwargs: Any) -> None:
        message = "Case publication is valid, but marked scratch cleanup failed: disk full"
        raise generation.runtime.CaseCleanupError(message)

    with pytest.raises(generation.runtime.CaseCleanupError):
        _run_case_step(monkeypatch, manifest, world, tmp_path, cleanup_fails)
    assert _only_candidate(campaign, world)["decision"] is None
    return campaign, manifest


def _spy_on_decisions(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every call of the candidate-decision writer, which also hashes the case HDF5, and still run it."""
    calls: list[Any] = []
    real_decide = lifecycle.decide_candidate
    monkeypatch.setattr(lifecycle, "decide_candidate", lambda *args, **kwargs: calls.append(args) or real_decide(*args, **kwargs))
    return calls


def test_cleanup_error_after_a_valid_publication_still_ends_in_the_success_decision(
    generation_config_factory: Any, world: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step that raises after its valid publication leaves no decision; the feeder records it once the step has ended."""
    campaign, manifest = _case_with_cleanup_error(generation_config_factory, world, tmp_path, monkeypatch)

    feed(campaign, world)
    assert _only_candidate(campaign, world)["decision"] is None, "the step is still active and may decide the case itself"

    world.finish_dispatches(manifest)
    world.set_state("7001", "COMPLETED")
    settled = feed(campaign, world)
    decision = _only_candidate(campaign, world)["decision"]
    assert decision is not None
    assert (decision["outcome"], decision["stage"], decision["detail"]) == ("successful", "publication", "Validated canonical case publication")
    assert decision["evidence"]["resources"] == {
        "dispatch_key": "7001-1",
        "slurm_job_id": "7001",
        "slurm_step_id": None,
        "submission_index": 1,
        "allocated_cores": 1,
    }
    assert settled["state"] == "complete"


def test_success_decision_of_the_case_step_is_not_rewritten_by_the_feeder(
    generation_config_factory: Any, world: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step that decided its case keeps its decision untouched when the feeder later sees the case decided."""
    campaign = make_campaign(generation_config_factory)
    manifest = submit(campaign, world)
    _publish_valid_case(monkeypatch, campaign, manifest, world.storage)
    _run_case_step(monkeypatch, manifest, world, tmp_path, lambda *_args, **_kwargs: SimpleNamespace(status="completed"))
    record = _only_candidate(campaign, world)
    decision_path = common.paths.resolve_generation_case_decision_path(record["case_directory"])
    decided = decision_path.read_bytes()
    assert record["decision"]["evidence"]["resources"]["slurm_step_id"] == "0"

    writes = _spy_on_decisions(monkeypatch)
    world.finish_dispatches(manifest)
    world.set_state("7001", "COMPLETED")
    assert feed(campaign, world)["state"] == "complete"
    assert writes == []
    assert decision_path.read_bytes() == decided


def test_metadata_only_feed_defers_the_success_decision_to_the_scheduled_feed(
    generation_config_factory: Any, world: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The light feed never hashes a case HDF5 for a decision: it asks for the scheduled feed, which then records it."""
    campaign, manifest = _case_with_cleanup_error(generation_config_factory, world, tmp_path, monkeypatch)
    world.finish_dispatches(manifest)
    world.set_state("7001", "COMPLETED")
    writes = _spy_on_decisions(monkeypatch)

    light = generation.campaign.feed_campaign(run_id_of(campaign), storage_root=world.storage, metadata_only=True)

    assert (light["continuation_requires_compute"], light["continuation_reason"]) == (True, "success_decision_reconciliation")
    assert writes == []
    assert _only_candidate(campaign, world)["decision"] is None

    settled = feed(campaign, world)
    assert len(writes) == 1
    assert _only_candidate(campaign, world)["decision"]["outcome"] == "successful"
    assert settled["state"] == "complete"


def test_repeated_feed_leaves_the_success_decision_of_the_feeder_unchanged(
    generation_config_factory: Any, world: FakeSlurm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the feeder has decided the case, every later feed, light or scheduled, leaves the decision bytes alone."""
    campaign, manifest = _case_with_cleanup_error(generation_config_factory, world, tmp_path, monkeypatch)
    world.finish_dispatches(manifest)
    world.set_state("7001", "COMPLETED")
    feed(campaign, world)
    decision_path = common.paths.resolve_generation_case_decision_path(_only_candidate(campaign, world)["case_directory"])
    decided = decision_path.read_bytes()
    writes = _spy_on_decisions(monkeypatch)

    assert feed(campaign, world)["state"] == "complete"
    assert "continuation_requires_compute" not in generation.campaign.feed_campaign(
        run_id_of(campaign), storage_root=world.storage, metadata_only=True
    )

    assert writes == []
    assert decision_path.read_bytes() == decided


@pytest.mark.parametrize(
    ("change", "recorded"),
    [
        ({}, True),
        ({"retryable": True}, False),
        ({"dispatch_key": "7001-1"}, False),
        ({"submission_intent": {"submission_index": 3}}, False),
        ({"node_job_intent": {"seat": 1}}, False),
        ({"reason_code": lifecycle.UNCLASSIFIED_SOLVER_REASON}, False),
    ],
    ids=["eligible", "retryable", "older_dispatch", "submission_intent", "node_job_intent", "unclassified"],
)
def test_the_feeder_decides_only_the_latest_terminal_runtime_stop(monkeypatch: pytest.MonkeyPatch, change: dict[str, Any], recorded: bool) -> None:
    """Only a non-retryable typed stop of the latest accounted dispatch is decided, and never while an intent is pending."""
    decisions: list[str] = []
    monkeypatch.setattr(lifecycle, "execution_batch", lambda _campaign, _reservation: SimpleNamespace(batch_id="batch"))
    monkeypatch.setattr(generation.campaign, "_record_failure_decision", lambda *_a, **kwargs: decisions.append(kwargs["reason_code"]))
    dispatches = [
        {"status": "dispatched", "job_id": "7001", "submission_index": index, "case": {"case_id": "case_0001", "batch_id": "batch"}}
        for index in (1, 2)
    ]
    manifest = {"submissions": dispatches, "submission_intent": change.get("submission_intent"), "node_job_intent": change.get("node_job_intent")}
    attempt = {
        "class": "solver",
        "retryable": change.get("retryable", False),
        "reason_code": change.get("reason_code", "stalled"),
        "detail": "stalled: 0.00 h of 1 h simulated after 900 s solving",
        "evidence": {"dispatch_key": change.get("dispatch_key", "7001-2")},
    }
    records = ({"reservation": {"case_id": "case_0001"}, "decision": None, "attempts": [attempt]},)
    scheduler = {"active": {}, "accounted": {"7001-1": {}, "7001-2": {}}}
    campaign: Any = SimpleNamespace()  # only the stubbed execution batch reads it
    assert generation.campaign._reconcile_runtime_failure_decisions(manifest, campaign, records, scheduler, storage_root=None) is recorded
    assert decisions == (["stalled"] if recorded else [])
