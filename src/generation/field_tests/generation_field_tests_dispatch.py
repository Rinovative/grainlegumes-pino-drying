"""Submit, reconcile, report, and cancel the static node jobs of one Generation field test.

Responsibilities:
  - Reconcile every unsealed member's latest attempt with its node-job identity, the worker receipts, and the queue
  - Dispatch gate members first and a base's variants only after its gate sealed success; seal variants of a failed gate
  - Write static node tickets and submit node jobs through the unchanged node worker, ticket, and submission builders
  - Give each seat one node job over a contiguous, near-equal share of the eligible members, with the node-job and site
    resources of the current execution configuration and the case shape and COMSOL executable of the base Source
  - Bind each submission to the roster version, the roster digest, and the digest of its member entry
  - Count dispositions for status and publish the cancel marker of every node job that has not ended, then report every
    refused node entry

Design principles:
  - A submission identity (node token, ticket sequence) is persisted before sbatch, so no submitted attempt is ever lost
  - Resume never redispatches a member whose attempt may still run: queued, running, or an unresolved submission waits;
    a submission without a recorded job id is reconciled by its token, the worker record, and the started receipt
  - Node jobs never select CPU types: a step records its CPU model, and inputs are compared within a last-digit tolerance
  - Node jobs share the Generation node-job slots, so field tests and campaigns together never exceed the node-job cap
  - A node job may hold more steps than its wall time fits: the worker starts no step without enough time left, and resume
    redispatches what never started

This module does NOT:
  - Run cases, decide dispositions of finished steps, or replace members
  - Freeze source trees; submission runs from an admitted frozen tree like every node-job dispatcher

"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from src import common
from src.generation.cases import generation_cases_config as config_service
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_node as node_service

from . import generation_field_tests_record as record
from . import generation_field_tests_roster as roster_service

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

# A submission intent without a queue row or worker record is proven lost only after this long (sbatch to queue visibility).
SUBMISSION_GRACE_SECONDS: Final = 300
_STEP_MODE: Final = "field-test-step"
_LABEL_TOKEN_LENGTH: Final = 8
_RECOVERED_JOB: Final = "job.recovered.json"


@dataclass(frozen=True, slots=True)
class Scheduler:
    """The scheduler operations dispatch needs: submit one sbatch command, list active node jobs, exclusions, and the tree lease."""

    submit: Callable[[Sequence[str]], str]
    active_jobs: Callable[[], Mapping[str, tuple[int, str, str]]]
    exclusions: Callable[[str, int], list[str]]
    lease: Callable[[str], bool]


def slurm_scheduler() -> Scheduler:
    """Return the Slurm operations of the existing node-job launcher, bound to the executing frozen source commit."""
    return Scheduler(
        submit=lambda command: cluster.submit_node_job(command, git_commit=os.environ.get("GENERATION_GIT_COMMIT", ""), environment={}),
        active_jobs=lambda: cluster.query_seat_jobs(range(1, config_service.MAXIMUM_NODE_JOBS + 1)),
        exclusions=cluster.standard_node_exclusions,
        lease=cluster.lease_executing_tree,
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    """Return one JSON object, or None while it does not exist."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _recorded_job_id(node_dir: Path) -> str | None:
    """Return the node job id sbatch reported or a resume recovered; a ``job.json`` with a null job id counts as missing."""
    for name in ("job.json", _RECOVERED_JOB):
        job = _read_json(node_dir / name)
        if job is not None and job.get("job_id") is not None:
            return str(job["job_id"])
    return None


def _attempt_ticket_state(
    directory: Path, submission: Mapping[str, Any], *, active: Mapping[str, tuple[int, str, str]], write: bool
) -> tuple[str, str | None]:
    """
    Return ``live``, ``never_started``, or ``ended`` for one submitted attempt, with its node job id when known.

    Without a recorded job id, the token's queue row or the node worker record names the job, which is then recorded.
    Without either, the attempt waits during the submission grace period; afterwards it never started only when its
    ticket has no started receipt.
    """
    node_dir = record.node_directory(directory, str(submission["node_token"]))
    seq = int(submission["ticket_seq"])
    started = node_service.read_node_receipt(node_dir, seq, "started")
    job_id = _recorded_job_id(node_dir)
    if job_id is None:
        rows = [job for job, (_seat, _state, comment) in active.items() if comment == submission["node_token"]]
        worker = node_service.read_node_record(node_dir, "node")
        job_id = rows[0] if rows else None if worker is None else str(worker["job_id"])
        if job_id is None:
            waited = (datetime.now(UTC) - datetime.fromisoformat(str(submission["submitted_at"]))).total_seconds()
            if waited < SUBMISSION_GRACE_SECONDS:
                return "live", None
            return ("never_started", None) if started is None else ("ended", None)
        if write:
            roster_service.publish_once(node_dir / _RECOVERED_JOB, {"job_id": job_id, "recorded_at": record.utc_now()})
    if job_id in active:
        return ("live", job_id) if node_service.read_node_receipt(node_dir, seq, "finished") is None else ("ended", job_id)
    return ("never_started", job_id) if started is None else ("ended", job_id)


def _reconcile_member(
    directory: Path,
    roster: Mapping[str, Any],
    member: Mapping[str, Any],
    *,
    active: Mapping[str, tuple[int, str, str]],
    write: bool,
) -> str:
    """Return ``sealed``, ``live``, ``retry``, or ``fresh`` for one member, recording the outcome of an attempt that ended without one."""
    member_id = str(member["member_id"])
    if record.read_seal(directory, member_id) is not None:
        return "sealed"
    numbers = record.attempt_numbers(directory, member_id)
    if not numbers:
        return "fresh"
    latest = numbers[-1]
    if record.read_outcome(directory, member_id, latest) is None:
        submission = record.read_submission(directory, member_id, latest)
        state, job_id = ("never_started", None) if submission is None else _attempt_ticket_state(directory, submission, active=active, write=write)
        if state == "live":
            return "live"
        if not write:
            return "retry"
        classification = record.NOT_STARTED if state == "never_started" else "infrastructure_stop"
        reason = "ticket_not_started" if state == "never_started" else "step_ended_without_outcome"
        detail = "The ticket never started in its node job." if state == "never_started" else "The step or its node job ended without an outcome."
        outcome = record.new_outcome(
            member_id, latest, classification, recorded_by="resume", failure={"reason_code": reason, "message": detail}, slurm_job_id=job_id
        )
        try:
            with record.member_lock(directory, member_id):
                if record.read_outcome(directory, member_id, latest) is None:
                    roster_service.publish_once(record.attempt_directory(directory, member_id, latest) / "outcome.json", outcome)
        except (common.locking.FileLockUnavailableError, FileExistsError):
            return "live"
    if write:
        with record.member_lock(directory, member_id):
            if record.settle(directory, roster, member) is not None:
                return "sealed"
    return "retry"


def _resources(campaign: config_service.CampaignConfig) -> dict[str, Any]:
    """
    Return the resources of a field-test node job and its steps.

    The case shape (cores, memory, timeout) and the COMSOL executable stay the base Source's frozen ones, because the
    solve reads its executable from the case's frozen configuration; the node-job and site values, and the node-job
    memory derived from them, come from the current execution file the Source names, resolved with the Source's case
    values so the one resolver validates the combination. The resolver derives the node-job memory from the current
    per-case memory, which no override sets, so a current value other than the Source's is refused.
    """
    frozen = campaign.execution_values
    case_cores, case_timeout = frozen["cluster"]["cores_per_case"], frozen["runtime"]["timeout_seconds"]
    current = config_service.resolve_execution_values(
        campaign.execution_config_path,
        campaign_purpose=campaign.campaign_purpose,
        execution_overrides={"cores_per_case": case_cores, "case_timeout_seconds": case_timeout},
    )
    if current["cluster"]["memory_per_case"] != frozen["cluster"]["memory_per_case"]:
        message = (
            f"{campaign.execution_config_path} sets memory_per_case {current['cluster']['memory_per_case']}, but the base Source "
            f"ran its cases with {frozen['cluster']['memory_per_case']}; the node-job memory would not fit its cases. "
            "Restore the Source's value before submitting."
        )
        raise ValueError(message)
    return {
        "cores_per_case": case_cores,
        "memory_per_case": frozen["cluster"]["memory_per_case"],
        "timeout_seconds": float(case_timeout),
        "comsol_executable": frozen["site"]["comsol_executable"],
        **{key: current["cluster"][key] for key in ("node_jobs", "node_job_memory", "node_job_wall_time", "scheduler_options")},
        **{key: current["site"][key] for key in ("partition", "cores_per_node", "comsol_module", "python_executable")},
    }


def _seat_chunks(items: Sequence[tuple[Mapping[str, Any], int]], seats: int) -> list[Sequence[tuple[Mapping[str, Any], int]]]:
    """Split ``items`` in order into ``min(seats, len(items))`` contiguous chunks of near-equal size, the larger ones first."""
    count = min(seats, len(items))
    chunks: list[Sequence[tuple[Mapping[str, Any], int]]] = []
    start = 0
    for index in range(count):
        end = start + len(items) // count + (index < len(items) % count)
        chunks.append(items[start:end])
        start = end
    return chunks


def _submit_node_job(
    directory: Path,
    roster: Mapping[str, Any],
    chunk: Sequence[tuple[Mapping[str, Any], int]],
    *,
    version: int,
    seat: int,
    resources: Mapping[str, Any],
    exclusions: Sequence[str],
    scheduler: Scheduler,
    storage: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Write one node job's tickets and attempt submissions, then submit it; a dry run only builds the sbatch command."""
    token = secrets.token_hex(8)
    node_dir = directory if dry_run else record.node_directory(directory, token)
    if not dry_run:
        for name in (node_service.TICKETS_DIRECTORY, node_service.RECEIPTS_DIRECTORY, node_service.CONTROL_DIRECTORY):
            (node_dir / name).mkdir(parents=True, exist_ok=True)
    command = cluster.build_node_job_submission_command(
        seat=seat,
        submission_token=token,
        node_directory=node_dir,
        storage_root=storage,
        cores_per_node=int(resources["cores_per_node"]),
        node_job_memory=str(resources["node_job_memory"]),
        node_job_wall_time=str(resources["node_job_wall_time"]),
        partition=str(resources["partition"]),
        exclusions=exclusions,
        comsol_module=str(resources["comsol_module"]),
        python_executable=str(resources["python_executable"]),
        comsol_executable=str(resources["comsol_executable"]),
        scheduler_options=list(resources["scheduler_options"]),
    )
    summary: dict[str, Any] = {"seat": seat, "members": [[str(member["member_id"]), attempt] for member, attempt in chunk], "command": command}
    if dry_run:
        return summary
    time_limit, required_seconds = cluster.case_step_limits(float(resources["timeout_seconds"]))
    tree = common.paths.get_project_root().resolve()
    cores = int(resources["cores_per_case"])
    tickets = [
        node_service.NodeTicket(
            seq=seq,
            kind=node_service.STEP_KIND,
            cores=cores,
            memory=str(resources["memory_per_case"]),
            time_limit=time_limit,
            required_seconds=required_seconds,
            label=f"ft-{token[:_LABEL_TOKEN_LENGTH]}-{seq:04d}",
            argv=(
                str(tree / "scripts" / node_service.LAUNCHER_NAME),
                str(tree),
                _STEP_MODE,
                str(roster["task"]),
                str(roster["test_id"]),
                str(member["member_id"]),
                str(attempt),
                str(cores),
            ),
        )
        for seq, (member, attempt) in enumerate(chunk, start=1)
    ]
    now = record.utc_now()
    for ticket, (member, attempt) in zip(tickets, chunk, strict=True):
        binding = {"roster_version": version, "roster_sha256": str(roster["roster_sha256"]), "member_sha256": roster_service.member_sha256(member)}
        submission = {"node_token": token, "ticket_seq": ticket.seq, "step_label": ticket.label, "submitted_at": now, **binding}
        record.write_submission(directory, str(member["member_id"]), attempt, submission)
        node_service.write_node_ticket(node_dir, ticket)
    # The chunk is fixed, so the stop ticket ends the node job after its last step instead of the worker's idle timeout.
    node_service.write_node_ticket(node_dir, node_service.NodeTicket(seq=len(tickets) + 1, kind=node_service.STOP_KIND))
    try:
        job_id = scheduler.submit(command)
    except Exception as error:
        roster_service.publish_once(node_dir / "job.json", {"job_id": None, "error": str(error), "recorded_at": record.utc_now()})
        raise
    roster_service.publish_once(node_dir / "job.json", {"job_id": job_id, "error": None, "recorded_at": record.utc_now()})
    summary.update(token=token, job_id=job_id, leased=scheduler.lease(job_id))
    return summary


def dispatch(
    task: str,
    test_id: str,
    *,
    mode: str,
    dry_run: bool = False,
    storage_root: Path | str | None = None,
    scheduler: Scheduler | None = None,
) -> dict[str, Any]:
    """
    Reconcile one field test and dispatch its eligible members: ``submit`` takes never-attempted members, ``resume`` absent retries.

    Gates are always eligible; a variant only once its base gate sealed success with equal inputs, and a variant whose gate
    sealed anything else is sealed ``gate_blocked`` without a step. A dry run writes nothing and returns the sbatch commands
    with the field-test directory standing in for each new node directory.

    Raises
    ------
    ValueError
        If ``mode`` is not ``submit`` or ``resume``, or the current execution file sets another per-case memory than the
        base Source ran with.

    """
    if mode not in {"submit", "resume"}:
        message = f"Dispatch mode must be submit or resume, got {mode!r}."
        raise ValueError(message)
    scheduler = scheduler or slurm_scheduler()
    storage = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage)
    roster, version = roster_service.load_roster_version(directory)
    resources = _resources(roster_service.load_base_campaign(task, roster["base_source"], storage_root=storage))
    with contextlib.nullcontext() if dry_run else record.dispatch_lock(directory):
        active = scheduler.active_jobs()
        states = {
            str(member["member_id"]): _reconcile_member(directory, roster, member, active=active, write=not dry_run) for member in roster["members"]
        }
        wanted = "fresh" if mode == "submit" else "retry"
        blocked: list[str] = []
        items: list[tuple[Mapping[str, Any], int]] = []
        for member in roster["members"]:
            member_id = str(member["member_id"])
            if states[member_id] != wanted:
                continue
            if member["role"] != "gate":
                gate = record.read_seal(directory, str(member["gate_member_id"]))
                if gate is None:
                    continue
                if not record.gate_succeeded(gate):
                    if not dry_run:
                        with record.member_lock(directory, member_id):
                            reason = f"Gate {member['gate_member_id']} sealed {gate['disposition']}; its base cannot be paired."
                            record.seal_without_attempt(directory, member, "gate_blocked", reason_code="gate_failed", message=reason)
                    blocked.append(member_id)
                    continue
            attempts = record.attempt_numbers(directory, member_id)
            items.append((member, (attempts[-1] if attempts else 0) + 1))
        exclusions = scheduler.exclusions(str(resources["partition"]), int(resources["cores_per_node"]))
        jobs = [
            _submit_node_job(
                directory,
                roster,
                chunk,
                version=version,
                seat=seat,
                resources=resources,
                exclusions=exclusions,
                scheduler=scheduler,
                storage=storage,
                dry_run=dry_run,
            )
            for seat, chunk in enumerate(_seat_chunks(items, int(resources["node_jobs"])), start=1)
        ]
    return {"mode": mode, "dry_run": dry_run, "states": _counts(states.values()), "gate_blocked": blocked, "node_jobs": jobs}


def _counts(values: Any) -> dict[str, int]:
    """Return how often each value occurs, in sorted key order."""
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def status(task: str, test_id: str, *, storage_root: Path | str | None = None) -> dict[str, Any]:
    """Count sealed dispositions and unsealed members (never attempted or with recorded attempts) without querying the scheduler."""
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    roster = roster_service.load_roster(directory)
    sealed: list[str] = []
    unsealed: list[str] = []
    for member in roster["members"]:
        seal = record.read_seal(directory, str(member["member_id"]))
        if seal is not None:
            sealed.append(str(seal["disposition"]))
        else:
            unsealed.append("attempted" if record.attempt_numbers(directory, str(member["member_id"])) else "never_attempted")
    return {
        "test_id": test_id,
        "task": task,
        "stage": roster["stage"],
        "roster_sha256": roster["roster_sha256"],
        "members": len(roster["members"]),
        "sealed": _counts(sealed),
        "unsealed": _counts(unsealed),
    }


def cancel(task: str, test_id: str, *, storage_root: Path | str | None = None) -> list[str]:
    """
    Publish the cancel marker of every field-test node job without an exit record; return their tokens.

    Raises
    ------
    ValueError
        After every valid node directory got its marker, naming each refused ``nodes/`` entry (a symbolic link, a
        linked control folder or an invalid name) and each marker that could not be published.

    """
    directory = common.paths.resolve_field_test_dir(task, test_id, storage_root=storage_root)
    roster_service.load_roster(directory)
    cancelled: list[str] = []
    refused: list[str] = []
    with record.dispatch_lock(directory):
        nodes = record.record_path(directory, "nodes")
        for entry in sorted(nodes.iterdir()) if nodes.is_dir() else ():
            try:
                # A marker is published only where the node directory and its control folder pass the symbolic-link check.
                node_dir = record.record_path(record.node_directory(directory, entry.name), node_service.CONTROL_DIRECTORY).parent
            except ValueError as error:
                refused.append(str(error))
                continue
            if node_service.read_node_record(node_dir, "exit") is None and (node_dir / node_service.TICKETS_DIRECTORY).is_dir():
                try:
                    node_service.publish_node_cancel(node_dir)
                except OSError as error:
                    # One node's broken marker path must not keep the later node jobs running.
                    refused.append(f"{node_dir.name}: {error}")
                    continue
                cancelled.append(node_dir.name)
    if refused:
        message = f"Cancelled node jobs {cancelled}; refused node entries: {'; '.join(refused)}"
        raise ValueError(message)
    return cancelled
