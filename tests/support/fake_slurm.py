"""Provide fake Slurm executables and a campaign driver for node-mode campaign tests.

Responsibilities:
  - Put fake sbatch, squeue, sacct, sinfo, and scancel executables with a controllable job table ahead of the real ones
  - Expose the job table, the step accounting rows, and the logged commands of that fake scheduler to a test
  - Drive a tiny steady-flow campaign through submit, feed, and status, and give a process the environment of one case step
  - Add leased node jobs and dispatched case records to a hand-built run record

Design principles:
  - The fake queue holds a node job in PENDING until a test moves it; PENDING node jobs accept tickets
  - A crash is a BaseException raised at one exact point, so the next feed cycle is the recovery under test

This module does NOT:
  - Run case steps, COMSOL, or the node worker
  - Judge any campaign contract
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from src import common, generation
from src.generation.contracts import generation_contracts_source as source_service
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_node as node_service

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

COMMIT = "a" * 40
NODE_ID = "0123456789abcdef"
TREE_ID = "a" * 12 + "-" + "b" * 12
_FAKE_SBATCH = """#!/bin/bash
(IFS=$'\\037'; printf '%s\\n' "$*" >> "$FAKE_SLURM_DIR/sbatch.log")
if [[ -e "$FAKE_SLURM_DIR/sbatch-fail" ]]; then echo "sbatch: refused" >&2; exit 1; fi
job=$(cat "$FAKE_SLURM_DIR/next_job")
echo $((job + 1)) > "$FAKE_SLURM_DIR/next_job"
name=""; comment=""
for argument in "$@"; do
  case "$argument" in
    --job-name=*) name="${argument#--job-name=}" ;;
    --comment=*) comment="${argument#--comment=}" ;;
  esac
done
printf '%s|%s|PENDING|%s|\\n' "$job" "$name" "$comment" >> "$FAKE_SLURM_DIR/jobs"
printf '%s\\n' "$job"
"""
_FAKE_SQUEUE = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_SLURM_DIR/squeue.log"
if [[ -e "$FAKE_SLURM_DIR/squeue-fail" ]]; then echo "squeue: down" >&2; exit 1; fi
while IFS='|' read -r id name state comment node; do
  [[ -z "$id" ]] && continue
  case "$state" in PENDING|RUNNING|CONFIGURING|COMPLETING) ;; *) continue ;; esac
  case "$*" in
    *"%i|%j|%T|%k"*) printf '%s|%s|%s|%s\\n' "$id" "$name" "$state" "$comment" ;;
    *) printf '%s|%s|None|%s|2026-10-27T10:00:00|N/A|0:00\\n' "$id" "$state" "$node" ;;
  esac
done < "$FAKE_SLURM_DIR/jobs"
"""
_FAKE_SACCT = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_SLURM_DIR/sacct.log"
if [[ -e "$FAKE_SLURM_DIR/sacct-fail" ]]; then echo "sacct: down" >&2; exit 1; fi
case "$*" in
  *JobName*) cat "$FAKE_SLURM_DIR/steps" 2>/dev/null; exit 0 ;;
esac
while IFS='|' read -r id name state comment node; do
  [[ -z "$id" ]] && continue
  printf '%s|%s|0:0|2026-10-27T10:00:00|2026-10-27T10:00:01|2026-10-27T10:10:00|00:10:00|%s|2|test\\n' "$id" "$state" "$node"
done < "$FAKE_SLURM_DIR/jobs"
"""
_FAKE_SINFO = """#!/bin/bash
printf 'hpc1|2\\nhpc2|2\\nbig1|64\\ntiny1|1\\n'
"""
_FAKE_SCANCEL = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_SLURM_DIR/scancel.log"
"""


class Crash(BaseException):
    """Simulated process death at one exact point."""


@dataclass
class FakeSlurm:
    """Fake Slurm state: the job table and the logs of every fake executable."""

    directory: Path
    storage: Path
    leases: list[str]

    def log(self, name: str) -> list[str]:
        """Return the lines of one fake Slurm log, or an empty list."""
        path = self.directory / name
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def jobs(self) -> list[list[str]]:
        """Return the job table rows ``[id, name, state, comment, node]``."""
        return [line.split("|") for line in self.log("jobs") if line]

    def set_state(self, job_id: str, state: str, node: str = "hpc1") -> None:
        """Move one job of the fake queue to ``state`` (a terminal state removes it from squeue)."""
        rows = [row if row[0] != job_id else [job_id, row[1], state, row[3], node] for row in self.jobs()]
        (self.directory / "jobs").write_text("".join("|".join(row) + "\n" for row in rows), encoding="utf-8")

    def add_foreign_job(self, name: str, job_id: str, state: str = "RUNNING") -> None:
        """Add a job of another dispatcher that holds one node-job slot."""
        with (self.directory / "jobs").open("a", encoding="utf-8") as handle:
            handle.write(f"{job_id}|{name}|{state}|foreigntoken|hpc2\n")

    def add_step(self, job_id: str, step_id: int, name: str, state: str = "COMPLETED", exit_code: str = "0:0", node: str = "hpc1") -> None:
        """Add one job-step accounting row of the fake sacct."""
        row = (
            f"{job_id}.{step_id}|{name}|{state}|{exit_code}|2026-10-27T10:00:00|2026-10-27T10:00:05|2026-10-27T10:30:00|00:29:55|{node}|1|test|1024K"
        )
        with (self.directory / "steps").open("a", encoding="utf-8") as handle:
            handle.write(row + "\n")

    def clear_steps(self) -> None:
        """Forget every job-step accounting row, so a test can replace the state of a step."""
        (self.directory / "steps").unlink(missing_ok=True)

    def finish_dispatches(self, manifest: dict[str, Any], state: str = "COMPLETED", exit_code: str = "0:0") -> None:
        """Add a step row in ``state`` for every dispatched case step of ``manifest`` that has none yet."""
        known = {line.split("|")[1] for line in self.log("steps")}
        for record in manifest["submissions"]:
            if record["status"] == "dispatched" and record["step_name"] not in known:
                self.add_step(record["job_id"], int(record["ticket_seq"]) - 1, record["step_name"], state, exit_code)

    def fail(self, command: str, *, failing: bool = True) -> None:
        """Make one fake Slurm command fail, or work again."""
        marker = self.directory / f"{command}-fail"
        if failing:
            marker.write_text("", encoding="utf-8")
        else:
            marker.unlink(missing_ok=True)

    def sbatch_arguments(self) -> list[list[str]]:
        """Return the argument vector of every sbatch call (the fake logs them joined by the unit separator)."""
        return [line.split("\x1f") for line in self.log("sbatch.log")]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSlurm:
    """Provide fake Slurm executables, a frozen-tree stand-in, and the admitted-source environment."""
    binary = tmp_path / "bin"
    binary.mkdir()
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    for name, body in (
        ("sbatch", _FAKE_SBATCH),
        ("squeue", _FAKE_SQUEUE),
        ("sacct", _FAKE_SACCT),
        ("sinfo", _FAKE_SINFO),
        ("scancel", _FAKE_SCANCEL),
    ):
        executable = binary / name
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)
    (slurm / "next_job").write_text("7001\n", encoding="utf-8")
    (slurm / "jobs").write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(slurm))
    monkeypatch.setenv("GENERATION_GIT_COMMIT", COMMIT)
    monkeypatch.setenv("GENERATION_SOURCE_SHA256", "b" * 64)
    monkeypatch.setenv("GENERATION_NATIVE_VENV", str(common.paths.get_runtime_root().resolve() / "venvs" / "native"))
    monkeypatch.setattr(source_service, "validate_admitted_source_before_publication", lambda: None)
    monkeypatch.setattr(generation.campaign, "_repository_commit", lambda: COMMIT)
    monkeypatch.setattr(cluster, "executing_tree_id", lambda: TREE_ID)
    monkeypatch.setattr(generation.campaign, "_NODE_LEASE_RETRY_SECONDS", 0.0)
    leases: list[str] = []

    def lease(job_id: str) -> bool:
        leases.append(job_id)
        return True

    monkeypatch.setattr(cluster, "lease_executing_tree", lease)
    storage = tmp_path / "storage"
    return FakeSlurm(slurm, storage, leases)


def make_campaign(factory: Any, **overrides: Any) -> Any:
    """Return a tiny steady-flow campaign whose node job holds two concurrent one-core cases."""
    values: dict[str, Any] = {
        "simulation_profile": "steady_flow",
        "scheduler_kind": "slurm",
        "natural_count": 1,
        "max_admission_cases": 8,
        "node_jobs": 2,
        "cores_per_case": 1,
        "cores_per_node": 2,
    }
    values.update(overrides)
    config_path, _template = factory(**values)
    return generation.cases.config.load_campaign_config(config_path)


def run_id_of(campaign: Any) -> str:
    """Return the campaign run identity at the fake commit."""
    return generation.campaign.campaign_run_id(campaign, git_commit=COMMIT)


def submit(campaign: Any, slurm: FakeSlurm, **overrides: Any) -> dict[str, Any]:
    """Submit or continue the campaign through the public entry point."""
    return generation.campaign.submit_campaign(campaign, git_commit=COMMIT, storage_root=slurm.storage, **overrides)


def feed(campaign: Any, slurm: FakeSlurm) -> dict[str, Any]:
    """Run one feed cycle of the campaign."""
    return generation.campaign.feed_campaign(run_id_of(campaign), storage_root=slurm.storage)


def load_manifest(campaign: Any, slurm: FakeSlurm) -> dict[str, Any]:
    """Return the persisted run record of the campaign."""
    return campaign_evidence.load_campaign_run(run_id_of(campaign), storage_root=slurm.storage)


def node_directory(campaign: Any, slurm: FakeSlurm, record: dict[str, Any]) -> Path:
    """Return the node directory of one node-job record."""
    return campaign_evidence.campaign_node_job_directory(run_id_of(campaign), record["token"], storage_root=slurm.storage)


def crash_at(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str, *, after: bool, when: Callable[..., bool] = lambda *_a, **_k: True) -> None:
    """Raise a simulated crash once, before or after the real call of ``owner.name`` that satisfies ``when``."""
    real = getattr(owner, name)
    pending = {"armed": True}

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not pending["armed"] or not when(*args, **kwargs):
            return real(*args, **kwargs)
        pending["armed"] = False
        if after:
            real(*args, **kwargs)
        raise Crash

    monkeypatch.setattr(owner, name, wrapped)


def bind_worker(
    monkeypatch: pytest.MonkeyPatch, manifest: dict[str, Any], *, index: int = 1, step: str = "0", attempt: str = "1", cpus: str = "1"
) -> None:
    """Give the process the step environment a node worker hands to one case ticket."""
    allocation = manifest["submissions"][index - 1]["job_id"]
    monkeypatch.setenv("GENERATION_CAMPAIGN_RUN_ID", manifest["campaign_run_id"])
    monkeypatch.setenv("SLURM_JOB_ID", allocation)
    monkeypatch.setenv("SLURM_STEP_ID", step)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", cpus)
    monkeypatch.setenv("GENERATION_SUBMISSION_INDEX", str(index))
    monkeypatch.setenv("GENERATION_DISPATCH_KEY", f"{allocation}-{index}")
    monkeypatch.setenv("GENERATION_ATTEMPT_INDEX", attempt)


def open_node_job(manifest: dict[str, Any], storage: Path, *, job_id: str = "7001") -> dict[str, Any]:
    """Add one submitted, leased node job with its real node directory to a hand-built run record."""
    record = {
        "token": NODE_ID,
        "seat": 1,
        "node_directory": campaign_evidence.node_job_relative_directory(NODE_ID),
        "job_id": job_id,
        "status": "submitted",
        "lease": "ok",
        "draining": False,
        "stop_ticket_seq": None,
        "next_ticket_seq": 1,
        "cancel_published": False,
        "recorded_at": "2026-10-27T10:00:00+00:00",
        "terminal_observed_at": None,
        "error": None,
    }
    directory = campaign_evidence.campaign_node_job_directory(manifest["campaign_run_id"], NODE_ID, storage_root=storage)
    for name in (node_service.TICKETS_DIRECTORY, node_service.RECEIPTS_DIRECTORY, node_service.CONTROL_DIRECTORY):
        (directory / name).mkdir(parents=True)
    manifest["node_jobs"].append(record)
    manifest["slurm_job_ids"].append(job_id)
    return record


def dispatch_record(manifest: dict[str, Any], campaign: Any, task: Any, node_job: dict[str, Any], *, mode: str = "initial") -> dict[str, Any]:
    """Append one dispatched record of ``task`` to ``node_job`` the way a published ticket leaves it, and return it."""
    index = len(manifest["submissions"]) + 1
    attempts = sum(item["case"]["case_index"] == task.case_index and item["status"] == "dispatched" for item in manifest["submissions"])
    record = {
        "submission_index": index,
        "mode": mode,
        "recorded_at": "2026-10-27T10:00:01+00:00",
        "case": generation.campaign._task_payload(task),
        "node_job_token": node_job["token"],
        "job_id": node_job["job_id"],
        "ticket_seq": int(node_job["next_ticket_seq"]),
        "step_name": generation.campaign._step_name(campaign, task, submission_index=index),
        "attempt_index": attempts + 1,
        "status": "dispatched",
        "error": None,
    }
    manifest["submissions"].append(record)
    node_job["next_ticket_seq"] = record["ticket_seq"] + 1
    return record
