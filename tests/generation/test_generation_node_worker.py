"""Generic node worker, ticket contract, and node job submission contracts.

Responsibilities:
  - Check ticket and receipt validation, idempotent ticket and cancel-marker publication, and strict-order scheduling
  - Check group barriers, core refill, and the remaining-time gate
  - Check cancel, TERM, kill escalation, invalid tickets, error handling, and the exit record with virtual time and fake processes
  - Check one real process path through fake srun, squeue, and scancel executables and the CLI command
  - Check that a campaign node job links its own logs into its campaign without changing its exit status
  - Check the plain node-log lines of a case step, and that a stop the runtime classified has no JSON error event
  - Check the node job sbatch argument vector and the standard-node exclusion query

Design principles:
  - Time and steps are injected wherever a test needs deterministic ordering; real subprocesses prove only the wiring
  - No test contacts a real scheduler: fake Slurm executables precede every real one on PATH

This module does NOT:
  - Run COMSOL, the case launcher, or campaign dispatch logic
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from src import common
from src.generation.cli import cli_generation
from src.generation.runtime import generation_runtime_batch as batch_runtime
from src.generation.runtime import generation_runtime_cluster as cluster
from src.generation.runtime import generation_runtime_node as node

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_FAKE_SRUN = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_SLURM_DIR/srun.log"
exports=""
while [[ "${1:-}" == --* ]]; do
  if [[ "$1" == --export=* ]]; then exports="${1#--export=}"; fi
  shift
done
IFS=',' read -ra pairs <<< "$exports"
for pair in "${pairs[@]}"; do
  if [[ "$pair" != ALL ]]; then export "$pair"; fi
done
exec "$@"
"""
_FAKE_SQUEUE = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_SLURM_DIR/squeue.log"
if [[ -e "$FAKE_SLURM_DIR/squeue-fail" ]]; then exit 1; fi
case "$*" in
  *"--steps"*) cat "$FAKE_SLURM_DIR/steps" 2>/dev/null ;;
  *"-o %L"*) cat "$FAKE_SLURM_DIR/remaining" ;;
  *"-o %u|%i|%C"*) cat "$FAKE_SLURM_DIR/queue" ;;
esac
"""
_FAKE_SCANCEL = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_SLURM_DIR/scancel.log"
"""
_FAKE_LAUNCHER = """#!/bin/bash
mode="$1"
tag="$2"
printf '%s\\n' "$tag" >> "$FAKE_SLURM_DIR/steps.log"
printf '%s|%s\\n' "$GENERATION_ATTEMPT_INDEX" "$STORAGE_ROOT" >> "$FAKE_SLURM_DIR/env.log"
case "$mode" in
  quick) exit 0 ;;
  fail) exit 7 ;;
  hold) while [[ ! -e "$FAKE_SLURM_DIR/release-$tag" ]]; do sleep 0.02; done ;;
esac
"""
_NODE_CORES = 8


@dataclass
class Harness:
    """Test environment: a node directory, a launcher file, fake Slurm executables, and their logs."""

    root: Path
    node_directory: Path
    launcher: Path
    slurm_directory: Path

    def ticket(
        self, seq: int, /, *, group: int = 0, cores: int = 2, required_seconds: int = 0, mode: str = "quick", **overrides: Any
    ) -> node.NodeTicket:
        """Return one valid step ticket whose fake launcher behaves as ``mode``."""
        values: dict[str, Any] = {
            "seq": seq,
            "kind": node.STEP_KIND,
            "group": group,
            "cores": cores,
            "memory": "20G",
            "time_limit": "00:40:00",
            "required_seconds": required_seconds,
            "label": f"case-{seq}",
            "argv": (str(self.launcher), mode, str(seq)),
            "env": {"GENERATION_ATTEMPT_INDEX": "1", "STORAGE_ROOT": "/storage"},
        }
        values.update(overrides)
        return node.NodeTicket(**values)

    def write(self, *tickets: node.NodeTicket) -> None:
        """Publish tickets in the given order."""
        for ticket in tickets:
            node.write_node_ticket(self.node_directory, ticket)

    def stop(self, seq: int) -> None:
        """Publish the stop ticket."""
        self.write(node.NodeTicket(seq=seq, kind=node.STOP_KIND))

    def log(self, name: str) -> list[str]:
        """Return the lines of one fake Slurm log, or an empty list."""
        path = self.slurm_directory / name
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def started(self) -> list[int]:
        """Return the sequence numbers that have a started receipt."""
        return sorted(int(path.name[:6]) for path in (self.node_directory / "receipts").glob("*.started.json"))

    def finished(self, seq: int) -> dict[str, Any]:
        """Return one finished receipt."""
        receipt = node.read_node_receipt(self.node_directory, seq, "finished")
        assert receipt is not None
        return receipt

    def exit_record(self) -> dict[str, Any]:
        """Return node_exit.json."""
        record = node.read_node_record(self.node_directory, "exit")
        assert record is not None
        return record


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    """Provide a node directory and fake Slurm executables that shadow every real Slurm command."""
    binary = tmp_path / "bin"
    binary.mkdir()
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    for name, body in (("srun", _FAKE_SRUN), ("squeue", _FAKE_SQUEUE), ("scancel", _FAKE_SCANCEL)):
        executable = binary / name
        executable.write_text(body, encoding="utf-8")
        executable.chmod(0o755)
    (slurm / "remaining").write_text("10:00:00\n", encoding="utf-8")
    (slurm / "queue").write_text("rino|4242|8\nother|77|4\n", encoding="utf-8")
    scripts = tmp_path / "tree" / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "generation_node.sh"
    launcher.write_text(_FAKE_LAUNCHER, encoding="utf-8")
    launcher.chmod(0o755)
    directory = tmp_path / "node"
    (directory / "tickets").mkdir(parents=True)
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(slurm))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_NAME", "gen-seat-1")
    monkeypatch.setenv("SLURMD_NODENAME", "hpc999")
    return Harness(tmp_path, directory, launcher, slurm)


class FakeProcess:
    """A scripted srun client: it ends when told to, on TERM when configured, or on KILL."""

    def __init__(self, *, exit_on_term: int | None = None) -> None:
        """Create a client that ends on TERM with ``exit_on_term`` when given."""
        self.exit_on_term = exit_on_term
        self.signals: list[int] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        """Return the scripted exit status."""
        return self.returncode

    def send_signal(self, sig: int, /) -> None:
        """Record the signal and end the client when the script says so."""
        self.signals.append(sig)
        if sig == signal.SIGKILL:
            self.returncode = -int(signal.SIGKILL)
        elif sig == signal.SIGTERM and self.exit_on_term is not None:
            self.returncode = self.exit_on_term

    def finish(self, code: int = 0) -> None:
        """End the client with ``code``."""
        self.returncode = code


@dataclass
class VirtualTime:
    """Monotonic clock whose sleep advances time and runs a per-test hook."""

    on_sleep: Callable[[VirtualTime], None] | None = None
    now: float = 0.0
    sleeps: int = 0

    def clock(self) -> float:
        """Return the virtual time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance virtual time and run the hook."""
        self.now += seconds
        self.sleeps += 1
        if self.on_sleep is not None:
            self.on_sleep(self)


@dataclass
class Run:
    """Fake-process worker run: launched processes, launch-time observations, and the result."""

    harness: Harness
    exit_on_term: int | None = None
    processes: dict[int, FakeProcess] = field(default_factory=dict)
    launch_failures: dict[int, Exception] = field(default_factory=dict)
    launches: list[int] = field(default_factory=list)
    remaining: Callable[[], float | None] = lambda: 1.0e9
    result: node.NodeWorkerResult | None = None
    labels: dict[str, int] = field(default_factory=dict)
    term_requests: list[list[str]] = field(default_factory=list)

    def launch(self, ticket: node.NodeTicket) -> FakeProcess:
        """Fake launcher: record the order and what the node had already finished."""
        failure = self.launch_failures.get(ticket.seq)
        if failure is not None:
            raise failure
        self.launches.append(ticket.seq)
        process = FakeProcess(exit_on_term=self.exit_on_term)
        self.processes[ticket.seq] = process
        self.labels[ticket.label] = ticket.seq
        return process

    def signal_steps(self, job_id: str, labels: Sequence[str]) -> None:
        """Fake Slurm step signalling: TERM reaches the step task (not the srun client), which may end on it."""
        assert job_id == "4242"
        self.term_requests.append(list(labels))
        for label in labels:
            process = self.processes[self.labels[label]]
            if self.exit_on_term is not None:
                process.returncode = self.exit_on_term

    def execute(self, virtual: VirtualTime, **kwargs: Any) -> node.NodeWorkerResult:
        """Run the worker with virtual time and fake processes; one poll is one virtual second unless a test says otherwise."""
        kwargs.setdefault("poll_seconds", 1.0)
        self.result = node.run_node_worker(
            self.harness.node_directory,
            cores=kwargs.pop("cores", _NODE_CORES),
            clock=virtual.clock,
            sleep=virtual.sleep,
            remaining_seconds=self.remaining,
            launch_step=self.launch,
            signal_steps=self.signal_steps,
            **kwargs,
        )
        return self.result


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 20.0) -> None:
    """Poll a condition of a worker thread running real subprocesses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition was not reached in time")


class WorkerThread:
    """Run the real worker with real subprocesses in a thread, so the test can steer the fake steps."""

    def __init__(self, harness: Harness, *, cores: int = _NODE_CORES, **kwargs: Any) -> None:
        """Start the worker thread."""
        self.result: node.NodeWorkerResult | None = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(harness, cores, kwargs), daemon=True)
        self._thread.start()

    def _run(self, harness: Harness, cores: int, kwargs: dict[str, Any]) -> None:
        try:
            self.result = node.run_node_worker(harness.node_directory, cores=cores, poll_seconds=0.02, idle_timeout_seconds=30.0, **kwargs)
        except BaseException as error:  # noqa: BLE001 -- surface a worker failure to the test thread
            self.error = error

    def join(self) -> node.NodeWorkerResult:
        """Wait for the worker and return its result."""
        self._thread.join(timeout=30.0)
        assert not self._thread.is_alive()
        assert self.error is None, self.error
        assert self.result is not None
        return self.result


def _release(harness: Harness, seq: int) -> None:
    (harness.slurm_directory / f"release-{seq}").write_text("", encoding="utf-8")


# -- contracts --


@pytest.mark.parametrize(
    "override",
    [
        {"cores": 0},
        {"cores": True},
        {"group": -1},
        {"required_seconds": -1},
        {"memory": "20"},
        {"memory": "0G"},
        {"memory": "20 G"},
        {"time_limit": "1:00:00"},
        {"time_limit": "00:00:00"},
        {"time_limit": "00:61:00"},
        {"label": ""},
        {"label": "has space"},
        {"label": "x" * 49},
        {"argv": ()},
        {"argv": ("generation_node.sh", "a")},
        {"argv": ("/nonexistent/generation_node.sh", "a")},
        {"env": {"PATH": "/bin"}},
        {"env": {"GENERATION_X": "a\nb"}},
        {"env": {"GENERATION_X": "a,b"}},
        {"env": {"GENERATION_X": "a\0b"}},
        {"kind": "other"},
        {"seq": 0},
        {"seq": 1_000_000},
    ],
)
def test_ticket_rejects_contract_violations(harness: Harness, override: dict[str, Any]) -> None:
    """Every step field is validated fail-closed."""
    with pytest.raises(ValueError, match=r"Ticket|ticket"):
        harness.ticket(1, **override)


def test_ticket_launcher_must_be_a_regular_non_symlink_generation_node_file(harness: Harness) -> None:
    """A symlink, a differently named file, and a directory are not launchers."""
    link = harness.root / "tree" / "scripts" / "link" / "generation_node.sh"
    link.parent.mkdir()
    link.symlink_to(harness.launcher)
    other = harness.root / "tree" / "scripts" / "other.sh"
    other.write_text("", encoding="utf-8")
    for argv in ((str(link), "quick", "1"), (str(other), "quick", "1"), (str(harness.launcher.parent), "quick", "1")):
        with pytest.raises(ValueError, match=r"generation_node\.sh"):
            harness.ticket(1, argv=argv)


def test_stop_ticket_carries_no_step_fields() -> None:
    """A stop ticket is exactly a sequence number and its kind."""
    stop = node.NodeTicket(seq=3, kind=node.STOP_KIND)
    assert stop.to_payload() == {"schema": node.TICKET_SCHEMA, "schema_version": 1, "seq": 3, "kind": "stop"}
    with pytest.raises(ValueError, match="stop ticket"):
        node.NodeTicket(seq=3, kind=node.STOP_KIND, cores=1)


def test_ticket_payload_round_trips_and_a_published_ticket_is_never_replaced(harness: Harness) -> None:
    """Payloads parse back to the same ticket, and a second write of one sequence fails."""
    ticket = harness.ticket(1, env={"GENERATION_B": "2", "GENERATION_A": "1"})
    path = node.write_node_ticket(harness.node_directory, ticket)
    assert path == harness.node_directory / "tickets" / "000001.json"
    parsed = node.NodeTicket.from_payload(json.loads(path.read_text(encoding="utf-8")), expected_seq=1)
    assert parsed == ticket
    assert list(parsed.env) == ["GENERATION_A", "GENERATION_B"]
    with pytest.raises(FileExistsError):
        node.write_node_ticket(harness.node_directory, harness.ticket(1, cores=4))
    assert json.loads(path.read_text(encoding="utf-8"))["cores"] == ticket.cores
    assert not any(entry.name.startswith(".") for entry in (harness.node_directory / "tickets").iterdir())


def test_ensure_node_ticket_publishes_once_and_accepts_an_identical_replay(harness: Harness) -> None:
    """A dispatcher that repeats a published ticket after a crash gets the same file back, untouched."""
    ticket = harness.ticket(1)
    path = node.ensure_node_ticket(harness.node_directory, ticket)
    assert path == harness.node_directory / "tickets" / "000001.json"
    assert node.NodeTicket.from_payload(json.loads(path.read_text(encoding="utf-8")), expected_seq=1) == ticket
    published = path.stat().st_mtime_ns
    assert node.ensure_node_ticket(harness.node_directory, harness.ticket(1)) == path
    assert path.stat().st_mtime_ns == published
    stop = node.NodeTicket(seq=2, kind=node.STOP_KIND)
    assert node.ensure_node_ticket(harness.node_directory, stop) == node.ensure_node_ticket(harness.node_directory, stop)
    assert not any(entry.name.startswith(".") for entry in (harness.node_directory / "tickets").iterdir())


def test_ensure_node_ticket_refuses_a_conflicting_or_unreadable_ticket_and_keeps_the_file(harness: Harness) -> None:
    """A different ticket of the same sequence, a stop in place of a step, or unreadable content is a FileExistsError."""
    path = node.ensure_node_ticket(harness.node_directory, harness.ticket(1))
    before = path.read_text(encoding="utf-8")
    for conflicting in (harness.ticket(1, cores=4), harness.ticket(1, label="other"), node.NodeTicket(seq=1, kind=node.STOP_KIND)):
        with pytest.raises(FileExistsError, match="different content"):
            node.ensure_node_ticket(harness.node_directory, conflicting)
    assert path.read_text(encoding="utf-8") == before
    for seq, content in ((2, "not json"), (3, "[1, 2]")):
        broken = node.ticket_path(harness.node_directory, seq)
        broken.write_text(content, encoding="utf-8")
        with pytest.raises(FileExistsError, match="not a JSON object"):
            node.ensure_node_ticket(harness.node_directory, harness.ticket(seq))
        assert broken.read_text(encoding="utf-8") == content


def test_publish_node_cancel_is_idempotent_and_refuses_a_non_file_marker(harness: Harness) -> None:
    """The marker is created with its directory once, a replay leaves it untouched, and a non-file marker is refused."""
    marker = harness.node_directory / node.CONTROL_DIRECTORY / node.CANCEL_MARKER
    assert not marker.exists()
    node.publish_node_cancel(harness.node_directory)
    assert marker.is_file()
    published = marker.stat().st_mtime_ns
    node.publish_node_cancel(harness.node_directory)
    assert marker.is_file()
    assert marker.stat().st_mtime_ns == published
    other = harness.root / "other-node"
    (other / node.CONTROL_DIRECTORY / node.CANCEL_MARKER).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        node.publish_node_cancel(other)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(extra=1), "differ from the contract"),
        (lambda payload: payload.pop("label"), "differ from the contract"),
        (lambda payload: payload.update(schema="other"), "must declare schema"),
        (lambda payload: payload.update(schema_version=True), "must declare schema"),
        (lambda payload: payload.update(argv="not-a-list"), "argv must be a list"),
        (lambda payload: payload.update(seq=2), "file name sequence"),
    ],
)
def test_ticket_parser_rejects_malformed_documents(harness: Harness, mutate: Callable[[dict[str, Any]], object], message: str) -> None:
    """Unknown, missing, and mistyped fields are errors, not defaults."""
    payload = harness.ticket(1).to_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        node.NodeTicket.from_payload(payload, expected_seq=1)
    with pytest.raises(ValueError, match="JSON object"):
        node.NodeTicket.from_payload(["not", "an", "object"])


def test_srun_step_command_pins_exact_resources_and_environment(harness: Harness) -> None:
    """Each step is one exact task with its own resources, a bounded start wait, the ticket label as step name, and the ticket environment."""
    ticket = harness.ticket(7, cores=8, label="td-lentil.natural_c0007-a01", env={"STORAGE_ROOT": "/s", "GENERATION_ATTEMPT_INDEX": "3"})
    command = node.build_srun_step_command(ticket)
    assert command == [
        "srun",
        "--exact",
        "--immediate=120",
        "--ntasks=1",
        "--nodes=1",
        "--cpus-per-task=8",
        "--mem=20G",
        "--time=00:40:00",
        "--job-name=td-lentil.natural_c0007-a01",
        "--export=ALL,GENERATION_ATTEMPT_INDEX=3,STORAGE_ROOT=/s",
        str(harness.launcher),
        "quick",
        "7",
    ]
    with pytest.raises(ValueError, match="step ticket"):
        node.build_srun_step_command(node.NodeTicket(seq=1, kind=node.STOP_KIND))


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("00:45", 45.0),
        ("5:07", 307.0),
        ("01:00:00", 3600.0),
        ("2-03:04:05", 2 * 86400 + 3 * 3600 + 4 * 60 + 5.0),
        (" 10:00:00 ", 36000.0),
        ("UNLIMITED", float("inf")),
        ("NOT_SET", None),
        ("", None),
        ("1-03:04", None),
        ("abc", None),
    ],
)
def test_remaining_time_parser(text: str, seconds: float | None) -> None:
    """The time-left forms of squeue parse; everything else is unknown."""
    assert node.parse_remaining_seconds(text) == seconds


# -- scheduling with injected processes and time --


def test_idle_worker_ends_after_the_idle_timeout_and_records_it(harness: Harness) -> None:
    """Nothing to run and nothing queued for idle_timeout_seconds ends the worker with a clean exit."""
    virtual = VirtualTime()
    result = Run(harness).execute(virtual, idle_timeout_seconds=900.0, poll_seconds=5.0)
    assert (result.reason, result.exit_code, result.started, result.finished) == ("idle_timeout", 0, 0, 0)
    assert 900.0 <= virtual.now < 920.0
    record = harness.exit_record()
    assert (record["reason"], record["started"], record["finished"], record["job_id"]) == ("idle_timeout", 0, 0, "4242")


def test_stop_ticket_waits_for_running_steps_and_publishes_receipts(harness: Harness) -> None:
    """The stop ticket ends the worker only after every step has a finished receipt."""
    harness.write(harness.ticket(1), harness.ticket(2))
    harness.stop(3)
    run = Run(harness)

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 3:
            assert harness.started() == [1, 2]
            run.processes[1].finish(0)
        if virtual.sleeps == 6:
            run.processes[2].finish(5)

    result = run.execute(VirtualTime(hook))
    assert (result.reason, result.exit_code, result.started, result.finished) == ("stop_ticket", 0, 2, 2)
    assert (harness.finished(1)["exit_code"], harness.finished(2)["exit_code"]) == (0, 5)
    assert harness.finished(1)["outcome"] == "exited"
    started = node.read_node_receipt(harness.node_directory, 1, "started")
    assert started is not None
    assert (started["host"], started["seq"]) == ("hpc999", 1)
    assert isinstance(started["worker_pid"], int)


def test_free_cores_are_refilled_in_strict_sequence_order(harness: Harness) -> None:
    """A later ticket that would fit never overtakes an earlier ticket that waits for cores."""
    harness.write(harness.ticket(1, cores=6), harness.ticket(2, cores=4), harness.ticket(3, cores=1))
    harness.stop(4)
    run = Run(harness)

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 3:
            assert run.launches == [1]
            run.processes[1].finish()
        if virtual.sleeps == 6:
            assert run.launches == [1, 2, 3]
            run.processes[2].finish()
            run.processes[3].finish()

    result = run.execute(VirtualTime(hook))
    assert (result.reason, result.started, result.finished) == ("stop_ticket", 3, 3)


def test_group_barrier_separates_core_variants(harness: Harness) -> None:
    """A ticket of a higher group starts only when no step of the previous group runs, although cores are free."""
    harness.write(harness.ticket(1, group=0), harness.ticket(2, group=0), harness.ticket(3, group=1))
    harness.stop(4)
    run = Run(harness)

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 3:
            assert run.launches == [1, 2]
            run.processes[1].finish()
        if virtual.sleeps == 5:
            assert run.launches == [1, 2]
            run.processes[2].finish()
        if virtual.sleeps == 8:
            assert run.launches == [1, 2, 3]
            run.processes[3].finish()

    assert run.execute(VirtualTime(hook)).reason == "stop_ticket"


def test_time_gate_starts_at_equality_and_waits_for_a_free_node_before_ending(harness: Harness) -> None:
    """A step starts when remaining >= required; one that cannot start ends the worker once nothing runs."""
    harness.write(harness.ticket(1, required_seconds=1000), harness.ticket(2, required_seconds=2000))
    run = Run(harness, remaining=lambda: 1000.0)

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 3:
            assert run.launches == [1]
            run.processes[1].finish()

    result = run.execute(VirtualTime(hook))
    assert (result.reason, result.exit_code, result.started, result.finished) == ("time_exhausted", 0, 1, 1)
    assert "ticket 2" in result.detail
    assert harness.started() == [1]


def test_one_unknown_remaining_time_poll_is_tolerated(harness: Harness) -> None:
    """A transient failure of the time probe delays the start but does not end the node job."""
    harness.write(harness.ticket(1))
    answers = iter([None, None, 600.0])
    run = Run(harness, remaining=lambda: next(answers))

    def hook(_virtual: VirtualTime) -> None:
        if 1 in run.processes:
            run.processes[1].finish()

    result = run.execute(VirtualTime(hook))
    assert run.launches == [1]
    assert (result.reason, result.started, result.finished) == ("idle_timeout", 1, 1)


def test_persistently_unknown_remaining_time_never_starts_a_step(harness: Harness) -> None:
    """After three polls without a known remaining time the worker fails closed and ends."""
    harness.write(harness.ticket(1))
    run = Run(harness, remaining=lambda: None)
    result = run.execute(VirtualTime())
    assert (result.reason, result.exit_code, result.started) == ("time_exhausted", 0, 0)
    assert "unknown" in result.detail
    assert run.launches == []


def test_cancel_marker_stops_launching_terms_steps_and_kills_after_the_grace_period(harness: Harness) -> None:
    """control/cancel inhibits further starts, sends TERM to every step once through Slurm, and kills the clients after 600 s."""
    harness.write(harness.ticket(1), harness.ticket(2))
    run = Run(harness)

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 3:
            (harness.node_directory / "control").mkdir()
            (harness.node_directory / "control" / "cancel").write_text("", encoding="utf-8")
            harness.write(harness.ticket(3))

    virtual = VirtualTime(hook)
    result = run.execute(virtual, cancel_grace_seconds=600.0)
    assert (result.reason, result.exit_code, result.started, result.finished) == ("cancelled", 0, 2, 2)
    assert harness.started() == [1, 2]
    assert run.term_requests == [["case-1", "case-2"]]
    assert [process.signals for process in run.processes.values()] == [[signal.SIGKILL]] * 2
    assert harness.finished(1)["exit_code"] == -int(signal.SIGKILL)
    assert 600.0 <= virtual.now < 700.0
    assert harness.log("scancel.log") == []
    assert harness.exit_record()["reason"] == "cancelled"


def test_steps_that_exit_on_term_are_not_killed(harness: Harness) -> None:
    """A step that stops on the forwarded TERM ends with its own status and receives no KILL."""
    harness.write(harness.ticket(1))
    run = Run(harness, exit_on_term=143)

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 2:
            (harness.node_directory / "control").mkdir()
            (harness.node_directory / "control" / "cancel").write_text("", encoding="utf-8")

    virtual = VirtualTime(hook)
    result = run.execute(virtual)
    assert result.reason == "cancelled"
    assert run.term_requests == [["case-1"]]
    assert run.processes[1].signals == []
    assert harness.finished(1)["exit_code"] == 143
    assert virtual.now < 30.0


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT])
def test_termination_signal_acts_like_cancel_with_reason_terminated(harness: Harness, number: signal.Signals) -> None:
    """TERM and INT (the launcher's trap and Slurm's time-limit signal) stop the worker and restore the handlers."""
    harness.write(harness.ticket(1), harness.ticket(2))
    run = Run(harness, exit_on_term=143)
    before = (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT))

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 3:
            signal.raise_signal(number)

    result = run.execute(VirtualTime(hook))
    assert (result.reason, result.exit_code) == ("terminated", 143)
    assert number.name in result.detail
    assert run.term_requests == [["case-1", "case-2"]]
    assert [process.signals for process in run.processes.values()] == [[]] * 2
    assert harness.exit_record()["reason"] == "terminated"
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def _step_payload(harness: Harness, seq: int, **overrides: Any) -> dict[str, Any]:
    return harness.ticket(seq).to_payload() | overrides


@pytest.mark.parametrize(
    "case",
    ["unknown_field", "stray_file", "broken_json", "seq_mismatch", "too_many_cores", "lower_group"],
)
def test_invalid_ticket_stops_the_worker_after_running_steps_finish(harness: Harness, case: str) -> None:
    """A visible ticket that violates the contract is never skipped: no further start, drain, exit code 3."""
    tickets = harness.node_directory / "tickets"
    harness.write(harness.ticket(1, group=1 if case == "lower_group" else 0))
    invalid: dict[str, str] = {
        "unknown_field": json.dumps(_step_payload(harness, 2, extra=1)),
        "stray_file": "",
        "broken_json": "{broken",
        "seq_mismatch": json.dumps(_step_payload(harness, 3)),
        "too_many_cores": json.dumps(_step_payload(harness, 2, cores=_NODE_CORES + 1)),
        "lower_group": json.dumps(_step_payload(harness, 2, group=0)),
    }
    (tickets / ("notes.txt" if case == "stray_file" else "000002.json")).write_text(invalid[case], encoding="utf-8")
    expected_started = 1 if case == "lower_group" else 0
    run = Run(harness)

    def hook(virtual: VirtualTime) -> None:
        if virtual.sleeps == 4 and 1 in run.processes:
            run.processes[1].finish()

    result = run.execute(VirtualTime(hook))
    assert (result.reason, result.exit_code, result.started, result.finished) == ("invalid_ticket", 3, expected_started, expected_started)
    assert run.launches == ([1] if expected_started else [])
    assert harness.exit_record()["reason"] == "invalid_ticket"


def test_ticket_after_the_stop_ticket_is_invalid(harness: Harness) -> None:
    """The stop ticket is last; anything behind it is a dispatcher defect."""
    harness.stop(1)
    harness.write(harness.ticket(2))
    result = Run(harness).execute(VirtualTime())
    assert result.reason == "invalid_ticket"
    assert "follows the stop ticket" in result.detail


def test_a_started_receipt_is_never_replaced_and_its_ticket_never_relaunched(harness: Harness) -> None:
    """If a started receipt already exists, the ticket is not launched and the receipt stays untouched."""
    harness.write(harness.ticket(1))
    receipts = harness.node_directory / "receipts"
    receipts.mkdir()
    existing = receipts / "000001.started.json"
    existing.write_text("original", encoding="utf-8")
    run = Run(harness)
    result = run.execute(VirtualTime())
    assert (result.reason, result.exit_code) == ("error", 1)
    assert run.launches == []
    assert existing.read_text(encoding="utf-8") == "original"


def test_a_node_directory_serves_one_worker(harness: Harness) -> None:
    """A second worker in the same directory refuses to start and leaves the first worker's records alone."""
    harness.stop(1)
    first = Run(harness).execute(VirtualTime())
    assert first.reason == "stop_ticket"
    node_record = (harness.node_directory / "node.json").read_text(encoding="utf-8")
    exit_record = (harness.node_directory / "node_exit.json").read_text(encoding="utf-8")
    second = Run(harness).execute(VirtualTime())
    assert (second.reason, second.exit_code) == ("error", 1)
    assert "not reused" in second.detail
    assert (harness.node_directory / "node.json").read_text(encoding="utf-8") == node_record
    assert (harness.node_directory / "node_exit.json").read_text(encoding="utf-8") == exit_record


def test_internal_error_terminates_running_steps_and_still_writes_the_exit_record(harness: Harness) -> None:
    """An unexpected failure stops every step with TERM and reports reason error."""
    harness.write(harness.ticket(1), harness.ticket(2))
    run = Run(harness, exit_on_term=143, launch_failures={2: RuntimeError("boom")})
    result = run.execute(VirtualTime())
    assert (result.reason, result.exit_code) == ("error", 1)
    assert "boom" in result.detail
    assert run.term_requests == [["case-1"]]
    assert run.processes[1].signals == []
    assert harness.finished(1)["exit_code"] == 143
    assert "boom" in harness.exit_record()["detail"]


def test_launch_failure_becomes_a_finished_receipt(harness: Harness) -> None:
    """A Popen OSError is evidence (launch_failed, -1), not a worker failure."""
    harness.write(harness.ticket(1))
    harness.stop(2)
    run = Run(harness, launch_failures={1: OSError("no srun")})
    result = run.execute(VirtualTime())
    assert (result.reason, result.started, result.finished) == ("stop_ticket", 1, 1)
    receipt = harness.finished(1)
    assert (receipt["outcome"], receipt["exit_code"], receipt["host"]) == ("launch_failed", -1, "hpc999")


def test_worker_rejects_invalid_arguments_before_writing_anything(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allocation, directory, and numeric arguments are validated before the first record."""
    with pytest.raises(ValueError, match="cores"):
        node.run_node_worker(harness.node_directory, cores=0)
    with pytest.raises(ValueError, match="absolute"):
        node.run_node_worker(Path("relative"), cores=2)
    with pytest.raises(ValueError, match="absolute"):
        node.run_node_worker(harness.root / "missing", cores=2)
    with pytest.raises(ValueError, match="positive"):
        node.run_node_worker(harness.node_directory, cores=2, poll_seconds=0)
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        node.run_node_worker(harness.node_directory, cores=2)
    assert not (harness.node_directory / "node.json").exists()


# -- real subprocess wiring through fake Slurm executables --


def test_fake_srun_end_to_end_runs_steps_concurrently_and_refills_cores(harness: Harness) -> None:
    """Two steps fill the node, the third starts when one ends, and every step runs through the real srun argument vector."""
    harness.write(harness.ticket(1, cores=4, mode="hold"), harness.ticket(2, cores=4, mode="hold"), harness.ticket(3, cores=4, mode="fail"))
    harness.stop(4)
    worker = WorkerThread(harness)
    _wait_until(lambda: harness.started() == [1, 2])
    time.sleep(0.2)
    assert harness.started() == [1, 2]
    _release(harness, 1)
    _wait_until(lambda: harness.started() == [1, 2, 3])
    _wait_until(lambda: node.read_node_receipt(harness.node_directory, 3, "finished") is not None)
    _release(harness, 2)
    result = worker.join()
    assert (result.reason, result.exit_code, result.started, result.finished) == ("stop_ticket", 0, 3, 3)
    assert [harness.finished(seq)["exit_code"] for seq in (1, 2, 3)] == [0, 0, 7]
    assert sorted(harness.log("steps.log")) == ["1", "2", "3"]
    first_call = next(line for line in harness.log("srun.log") if "--job-name=case-1 " in line)
    assert first_call.startswith(
        "--exact --immediate=120 --ntasks=1 --nodes=1 --cpus-per-task=4 --mem=20G --time=00:40:00 --job-name=case-1 --export=ALL,"
    )
    assert "GENERATION_ATTEMPT_INDEX=1" in first_call
    # The step process sees exactly the environment of its ticket.
    assert harness.log("env.log") == ["1|/storage"] * 3
    record = node.read_node_record(harness.node_directory, "node")
    assert record is not None
    assert (record["job_id"], record["host"], record["cores"], record["seat_name"]) == ("4242", "hpc999", _NODE_CORES, "gen-seat-1")
    assert record["foreign_jobs_at_start"] == [{"user": "other", "job_id": "77", "cpus": 4}]
    assert record["cpu_model"] is None or isinstance(record["cpu_model"], str)
    assert harness.exit_record()["foreign_jobs_at_end"] == [{"user": "other", "job_id": "77", "cpus": 4}]
    assert harness.log("scancel.log") == []


def test_default_cancel_signals_steps_through_scancel_and_never_terms_srun(harness: Harness) -> None:
    """Without an injected signaller, cancel resolves this job's step by name and sends TERM with scancel, then kills the client."""
    harness.write(harness.ticket(1, cores=4, mode="hold"))
    (harness.slurm_directory / "steps").write_text("4242.0|case-1\n4242.9|other\n999.0|case-1\n", encoding="utf-8")
    worker = WorkerThread(harness, cancel_grace_seconds=0.5)
    _wait_until(lambda: harness.started() == [1])
    (harness.node_directory / "control").mkdir()
    (harness.node_directory / "control" / "cancel").write_text("", encoding="utf-8")
    result = worker.join()
    assert result.reason == "cancelled"
    assert harness.log("scancel.log") == ["--signal=TERM 4242.0"]
    assert harness.finished(1)["exit_code"] == -int(signal.SIGKILL)


def test_default_time_probe_reads_squeue_and_gates_the_step(harness: Harness) -> None:
    """Without an injected probe the worker reads this job's remaining time through squeue %L."""
    (harness.slurm_directory / "remaining").write_text("00:10:00\n", encoding="utf-8")
    harness.write(harness.ticket(1, required_seconds=700))
    result = WorkerThread(harness).join()
    assert result.reason == "time_exhausted"
    assert "600 s" in result.detail
    assert harness.started() == []
    assert harness.log("srun.log") == []
    assert "-h -j 4242 -o %L" in harness.log("squeue.log")


def test_unreadable_queue_marks_foreign_jobs_unknown_and_fails_the_time_gate(harness: Harness) -> None:
    """A failing squeue is recorded as unknown (null), never as an empty node, and never starts a step."""
    (harness.slurm_directory / "squeue-fail").write_text("", encoding="utf-8")
    harness.write(harness.ticket(1))
    result = WorkerThread(harness).join()
    assert result.reason == "time_exhausted"
    record = node.read_node_record(harness.node_directory, "node")
    assert record is not None
    assert record["foreign_jobs_at_start"] is None
    assert harness.exit_record()["foreign_jobs_at_end"] is None


def test_cli_runs_the_worker_and_returns_its_exit_code(harness: Harness, capsys: pytest.CaptureFixture[str]) -> None:
    """run-node-worker prints its result as one plain line, which only a reader of the node log consumes, and returns the worker's status."""
    harness.stop(1)
    assert cli_generation.main(["run-node-worker", str(harness.node_directory), "--cores", "4"]) == 0
    assert capsys.readouterr().out == "Node worker ended: stop ticket 000001 (stop_ticket); steps started 0, finished 0\n"


def test_cli_links_a_campaign_node_job_log_and_a_failed_link_keeps_the_exit_code(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A campaign node job links its own logs into the campaign at start; a link that fails only warns."""
    campaign = harness.root / "campaign"
    worker = replace(harness, node_directory=campaign / "_record/execution/node_jobs" / ("f" * 16))
    (worker.node_directory / "tickets").mkdir(parents=True)
    worker.stop(1)
    out = harness.root / "node-4242.out"
    out.write_text("preflight\n", encoding="utf-8")
    # The .err file does not exist, so its link fails after the .out link succeeded.
    monkeypatch.setattr(cli_generation.host_service, "job_log_files", lambda _templates: (out, harness.root / "node-4242.err"))
    assert cli_generation.main(["run-node-worker", str(worker.node_directory), "--cores", "4"]) == 0
    assert (campaign / "logs/node-4242.out").samefile(out)
    assert "Warning: could not link job logs" in capsys.readouterr().err


_DURATION = r"(?:\d+ s|\d+ min \d\d s|\d+ h \d\d min)"
_ISO_WITH_ZONE = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d"


def _run_case_step(monkeypatch: pytest.MonkeyPatch, case_job: Callable[..., Any]) -> int:
    """Run the run-campaign-case command of one lentil case with a test-owned case job."""
    monkeypatch.setattr(cli_generation.campaign_runtime, "run_campaign_case_job", case_job)
    return cli_generation.main(["run-campaign-case", "run", "transient_drying__lentil__natural", "3"])


def test_a_finished_case_step_ends_with_one_searchable_line(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The outcome of a case step is one CASE END line with material, case, status, end time with zone and duration, not a JSON record."""
    assert _run_case_step(monkeypatch, lambda *_args, **_kwargs: SimpleNamespace(status="completed")) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert re.fullmatch(rf"CASE END lentil case_0003 completed at {_ISO_WITH_ZONE} after {_DURATION}\n", captured.out)


def test_a_failed_case_step_names_its_reason_in_words_before_the_failure_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failure the runtime did not classify is stated as plain text on stderr ahead of its failure JSON; the exit status stays two."""
    failure = "case input is not readable"

    def fail(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError(failure)

    assert _run_case_step(monkeypatch, fail) == 2
    captured = capsys.readouterr()
    reason, record = captured.err.splitlines()
    assert reason == f"lentil case_0003 stopped: {failure}"
    assert json.loads(record)["message"] == failure
    assert re.fullmatch(rf"CASE END lentil case_0003 failed at {_ISO_WITH_ZONE} after {_DURATION}: {failure}\n", captured.out)


def test_a_classified_case_stop_is_stated_once_in_words_without_a_json_error_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A stall, timeout or solver error that the runtime classified is only the plain line; the exit status stays two."""
    stalled = "stalled: 0.01 h of 168 h simulated after 967 s solving"

    def fail(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise batch_runtime.CaseExecutionError(stalled, work_directory=tmp_path, runtime_reason="stalled")

    assert _run_case_step(monkeypatch, fail) == 2
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [f"lentil case_0003 stopped: {stalled}"]
    assert re.fullmatch(rf"CASE END lentil case_0003 failed at {_ISO_WITH_ZONE} after {_DURATION}: {re.escape(stalled)}\n", captured.out)


def test_an_unclassified_case_execution_error_keeps_its_json_error_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A case execution failure without a runtime classification still carries its failure JSON with stage and work directory."""
    incomplete = "exports are incomplete"

    def fail(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise batch_runtime.CaseExecutionError(incomplete, work_directory=tmp_path, failure_stage="exports")

    assert _run_case_step(monkeypatch, fail) == 2
    reason, record = capsys.readouterr().err.splitlines()
    assert reason == f"lentil case_0003 stopped: {incomplete}"
    assert json.loads(record)["stage"] == "exports"


# -- node job submission --


def _launch_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("GENERATION_SOURCE_SHA256", "b" * 64)
    monkeypatch.setenv("GENERATION_NATIVE_VENV", str(common.paths.get_runtime_root().resolve() / "venvs" / "native"))
    for name in ("SOURCE_TREE_ID", "GENERATION_SOURCE_SNAPSHOT", "GENERATION_SOURCE_SNAPSHOT_SHA256"):
        monkeypatch.delenv(name, raising=False)


def _submission(tmp_path: Path, **overrides: Any) -> list[str]:
    storage = tmp_path / "storage"
    storage.mkdir(exist_ok=True)
    directory = tmp_path / "node-dir"
    directory.mkdir(exist_ok=True)
    values: dict[str, Any] = {
        "seat": 2,
        "submission_token": "abcdef0123456789",
        "node_directory": directory,
        "storage_root": storage,
        "cores_per_node": 32,
        "node_job_memory": "168G",
        "node_job_wall_time": "08:00:00",
        "partition": "standard",
        "exclusions": ["hpc12", "hpc11", "hpc12"],
        "comsol_module": "Comsol/v6.4",
        "python_executable": "/usr/bin/python3.12",
        "comsol_executable": "comsol",
    }
    values.update(overrides)
    return cluster.build_node_job_submission_command(**values)


def test_node_job_submission_binds_seat_token_resources_and_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One node job per slot: singleton name, recoverable token, fixed node type, no requeue, and the generic worker command."""
    _launch_evidence(monkeypatch)
    command = _submission(tmp_path, scheduler_options=["--account=grain"])
    runtime_logs = common.paths.get_runtime_root().resolve() / "logs" / "generation"
    repository = common.paths.get_project_root().resolve()
    assert command[:-1] == [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=32",
        "--mem=168G",
        "--time=08:00:00",
        "--partition=standard",
        "--exclude=hpc11,hpc12",
        "--job-name=gen-seat-2",
        "--dependency=singleton",
        "--comment=abcdef0123456789",
        "--no-requeue",
        f"--chdir={repository}",
        "--export=ALL",
        f"--output={runtime_logs}/node-%j.out",
        f"--error={runtime_logs}/node-%j.err",
        "--open-mode=append",
        "--signal=B:TERM@120",
        "--account=grain",
    ]
    assert command[-1].startswith("--wrap=")
    worker = shlex.split(command[-1].removeprefix("--wrap="))
    assert worker[0] == "env"
    launcher_index = next(index for index, argument in enumerate(worker) if argument.endswith("generation_node.sh"))
    assert worker[launcher_index + 1 :] == [str(repository), "node-worker", str(tmp_path / "node-dir"), "32"]
    bindings = worker[1:launcher_index]
    assert not any(binding.startswith("GENERATION_ATTEMPT_INDEX") for binding in bindings)
    for binding in (
        "GENERATION_GIT_COMMIT=" + "a" * 40,
        "GENERATION_SOURCE_SHA256=" + "b" * 64,
        f"STORAGE_ROOT={(tmp_path / 'storage').resolve()}",
        "GENERATION_COMSOL_MODULE=Comsol/v6.4",
        "GENERATION_PYTHON_EXECUTABLE=/usr/bin/python3.12",
        "GENERATION_COMSOL_EXECUTABLE=comsol",
    ):
        assert binding in bindings


def test_node_job_submission_without_exclusions_omits_the_exclude_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No node of another type: no --exclude argument at all."""
    _launch_evidence(monkeypatch)
    command = _submission(tmp_path, exclusions=[], seat=1)
    assert not any(argument.startswith("--exclude") for argument in command)
    assert "--job-name=gen-seat-1" in command


@pytest.mark.parametrize(
    "override",
    [
        {"seat": 0},
        {"seat": 5},
        {"seat": True},
        {"submission_token": "short"},
        {"submission_token": "UPPERCASE1234"},
        {"submission_token": "a" * 33},
        {"cores_per_node": 0},
        {"node_job_memory": "168"},
        {"node_job_memory": "0G"},
        {"node_job_wall_time": "8:00:00"},
        {"node_job_wall_time": "00:00:00"},
        {"partition": "a b"},
        {"partition": ""},
        {"exclusions": ["bad name"]},
        {"exclusions": ["--nodelist=x"]},
        {"scheduler_options": ["--mem=1G"]},
        {"scheduler_options": ["--dependency=afterok:1"]},
        {"scheduler_options": ["-N 2"]},
        {"scheduler_options": ["--comment=x"]},
        {"comsol_module": ""},
        {"comsol_executable": "a\nb"},
    ],
)
def test_node_job_submission_rejects_invalid_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[str, Any]) -> None:
    """Every input is validated fail-closed, and owned options cannot be overridden."""
    _launch_evidence(monkeypatch)
    with pytest.raises(ValueError, match=r"\w"):
        _submission(tmp_path, **override)


def test_node_job_submission_requires_existing_directories_and_launch_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or relative node directory, a bad storage root, and absent source evidence are refused."""
    _launch_evidence(monkeypatch)
    with pytest.raises(ValueError, match="Node directory"):
        _submission(tmp_path, node_directory=tmp_path / "missing")
    with pytest.raises(ValueError, match="Node directory"):
        _submission(tmp_path, node_directory=Path("relative"))
    with pytest.raises(ValueError, match="storage root"):
        _submission(tmp_path, storage_root=common.paths.get_project_root())
    monkeypatch.delenv("GENERATION_GIT_COMMIT")
    with pytest.raises(ValueError, match="GENERATION_GIT_COMMIT"):
        _submission(tmp_path)


def _sinfo(stdout: str, returncode: int = 0) -> Callable[[Any], subprocess.CompletedProcess[str]]:
    def run(arguments: Any) -> subprocess.CompletedProcess[str]:
        assert list(arguments) == ["sinfo", "-h", "-p", "standard", "-N", "-o", "%N|%c"]
        return subprocess.CompletedProcess(arguments, returncode, stdout=stdout, stderr="sinfo: down")

    return run


_MIXED_NODES = "hpc02|32\nhpc01|32\nhpc10|64\nhpc12|36\nhpc09|16\nhpc10|64\n\nhpc11|128\n"


def test_standard_node_exclusions_keep_the_larger_nodes() -> None:
    """Only the nodes with fewer CPUs than the node job requests are excluded, sorted and unique."""
    assert cluster.standard_node_exclusions("standard", 32, sinfo=_sinfo(_MIXED_NODES)) == ["hpc09"]
    assert cluster.standard_node_exclusions("standard", 32, sinfo=_sinfo("hpc01|36\n")) == []


@pytest.mark.parametrize(
    ("stdout", "returncode", "match"),
    [
        ("hpc01|32\n", 1, "status 1"),
        ("hpc01|16\nhpc02|24\n", 0, "no node with at least 32"),
        ("", 0, "no node with at least 32"),
        ("hpc01 32\n", 0, "malformed"),
        ("hpc01|many\n", 0, "malformed"),
        ("hpc01|32|x\n", 0, "malformed"),
        ("bad name|32\n", 0, "malformed"),
        ("hpc01|32\nhpc01|64\n", 0, "malformed"),
    ],
)
def test_standard_node_exclusions_fail_closed(stdout: str, returncode: int, match: str) -> None:
    """A failing or ambiguous sinfo, or a partition without a large enough node, is an error."""
    with pytest.raises(RuntimeError, match=match):
        cluster.standard_node_exclusions("standard", 32, sinfo=_sinfo(stdout, returncode))


def test_standard_node_exclusions_report_a_missing_sinfo_and_reject_bad_arguments() -> None:
    """An unavailable sinfo executable is a RuntimeError; invalid arguments are ValueErrors."""

    def missing(_arguments: Any) -> subprocess.CompletedProcess[str]:
        message = "no sinfo"
        raise FileNotFoundError(message)

    with pytest.raises(RuntimeError, match="no sinfo"):
        cluster.standard_node_exclusions("standard", 32, sinfo=missing)
    with pytest.raises(ValueError, match="Partition"):
        cluster.standard_node_exclusions("a b", 32, sinfo=_sinfo(""))
    with pytest.raises(ValueError, match="cores_per_node"):
        cluster.standard_node_exclusions("standard", 0, sinfo=_sinfo(""))
